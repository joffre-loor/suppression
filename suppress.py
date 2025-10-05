#!/usr/bin/env python3
# realtime_filter.py
# Mac mic -> model -> XM5 headphones. Control via suppress_api.py.
# Uses AudioSep separation + EXTREME spectral masking + OLA. Polls API for {mode, classes}.

import os, sys, time, tempfile, io, contextlib, threading
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import get_window
from datetime import datetime
import requests

# =========================
# Devices (by name contains)
# =========================
INPUT_DEVICE_NAME  = "MacBook Pro Microphone"
OUTPUT_DEVICE_NAME = "WH-1000XM5"

# =========================
# AudioSep repo path / import
# =========================
AUDIOSEP_REPO = Path(__file__).resolve().parent / "AudioSep"
if str(AUDIOSEP_REPO) not in sys.path:
    sys.path.append(str(AUDIOSEP_REPO))

import torch
from pipeline import build_audiosep, separate_audio as inference

# =========================
# Control plane
# =========================
API_BASE = os.getenv("SUPPRESS_API_BASE", "http://127.0.0.1:8000")  # change to your laptop IP

DEFAULT_PROMPT = "music"
DEFAULT_MODE   = "drop"

class ControlState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = DEFAULT_MODE
        self.prompts = [DEFAULT_PROMPT]
        self.version = -1

    def update_from_api(self, d: dict):
        with self.lock:
            self.mode = d.get("mode", DEFAULT_MODE)
            clz = d.get("classes", [])
            self.prompts = clz if clz else [DEFAULT_PROMPT]
            self.version = d.get("version", self.version)

    def snapshot(self):
        with self.lock:
            return self.mode, list(self.prompts), self.version

CTRL = ControlState()

def poll_control_state(stop_event: threading.Event, every_sec=0.25):
    url = f"{API_BASE}/suppress/current"
    last_version = None
    while not stop_event.is_set():
        try:
            r = requests.get(url, timeout=0.5)
            r.raise_for_status()
            CTRL.update_from_api(r.json())
            _, _, v = CTRL.snapshot()
            if v != last_version:
                last_version = v
                print(f"[CTRL] mode={CTRL.mode} prompts={CTRL.prompts} v={v}", flush=True)
        except Exception:
            pass
        stop_event.wait(every_sec)

# =========================
# Defaults / I/O params
# =========================
SR        = 32000
CHUNK_S   = 1.0                   # analysis window (seconds)
HOP_S     = 0.5                   # 50% overlap
CHANNELS  = 1

# =========================
# EXTREME spectral mask params
# =========================
NFFT          = 4096
HOP           = NFFT // 2
WIN           = get_window("hann", NFFT).astype(np.float32)
WIN_SQ        = (WIN * WIN).astype(np.float32)
EPS           = np.float32(1e-7)

P_MASK        = 4.0
ALPHA_DROP    = 0.999
ALPHA_KEEP    = 1.00
MASK_GAMMA    = 1.8
DROP_POWER    = 2.0
DROP_FLOOR_DB = -70.0

TWO_PASS      = True

# Presence / gating
SEP_GATE_DB   = -45.0
ATTACK_COEFF  = 0.25
RELEASE_COEFF = 0.08
HOLD_SECONDS  = 0.20

# Output hygiene
OUTPUT_SOFT_LIMIT_DB = -1.0
DC_ALPHA             = 0.995
OUT_FLOOR_DB         = -55.0

# =========================
# Capture (before/after)
# =========================
CAPTURE_ROOT = Path("captures"); CAPTURE_ROOT.mkdir(exist_ok=True)
RUN_DIR = CAPTURE_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"); RUN_DIR.mkdir(parents=True, exist_ok=True)
INPUT_WAV_PATH  = RUN_DIR / "input_raw.wav"
OUTPUT_WAV_PATH = RUN_DIR / "output_after.wav"

# =========================
# Helpers (float32-safe)
# =========================
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float32), dtype=np.float32) + np.float32(1e-12)))

def lin_to_db(x: float) -> float:
    return float(20.0 * np.log10(max(x, 1e-12)))

def dc_block(x: np.ndarray, z=[np.float32(0.0)]) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    y = np.empty_like(x, dtype=np.float32)
    prev = float(z[0]); prev_y = 0.0
    a = float(DC_ALPHA)
    for i, s in enumerate(x):
        curr = s - prev + a * prev_y
        y[i] = curr
        prev_y = curr
        prev = s
    z[0] = np.float32(prev)
    return y

def to_mono(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1: return a.astype(np.float32, copy=False)
    return np.mean(a, axis=1, dtype=np.float32)

class OnePole:
    def __init__(self, alpha_up=ATTACK_COEFF, alpha_down=RELEASE_COEFF):
        self.y = np.float32(0.0); self.au = float(alpha_up); self.ad = float(alpha_down)
    def __call__(self, x: float) -> float:
        a = self.au if x > self.y else self.ad
        self.y = (1 - a) * self.y + a * x
        return float(self.y)

class HoldGate:
    def __init__(self, thresh_db=SEP_GATE_DB, hold_s=HOLD_SECONDS, hop_s=HOP_S):
        self.thr = 10**(thresh_db/20.0)
        self.hold = max(1, int(hold_s / hop_s))
        self.c = 0; self.open=False
    def update(self, presence_lin: float) -> float:
        if presence_lin >= self.thr:
            self.open=True; self.c=self.hold
        else:
            if self.c>0: self.c-=1; self.open=True
            else: self.open=False
        return 1.0 if self.open else 0.0

def soft_limiter(x: np.ndarray, target_db=OUTPUT_SOFT_LIMIT_DB) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    target = np.float32(10.0 ** (target_db / 20.0))
    peak = np.max(np.abs(x)).astype(np.float32) + np.float32(1e-9)
    if peak <= target:
        return x
    y = np.tanh(x * (np.float32(1.5) / target)) * (target / np.tanh(np.float32(1.5)))
    return y.astype(np.float32, copy=False)

def postprocess(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if lin_to_db(rms(x)) < OUT_FLOOR_DB:
        return np.zeros_like(x, dtype=np.float32)
    y = soft_limiter(x, OUTPUT_SOFT_LIMIT_DB)
    return y.astype(np.float32, copy=False)

# =========================
# STFT / ISTFT (Hann, 50% hop)
# =========================
def stft(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if len(x) < NFFT:
        x = np.pad(x, (0, NFFT - len(x)), mode='constant')
    frames = 1 + (len(x) - NFFT) // HOP if len(x) >= NFFT else 1
    out = []
    pos = 0
    for _ in range(frames):
        seg = x[pos:pos+NFFT]
        if len(seg) < NFFT:
            seg = np.pad(seg, (0, NFFT - len(seg)), mode='constant')
        out.append(np.fft.rfft(seg * WIN))
        pos += HOP
    return np.stack(out, axis=0)

def istft(X: np.ndarray) -> np.ndarray:
    frames, freqs = X.shape
    out_len = NFFT + (frames-1)*HOP
    y = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    pos = 0
    for i in range(frames):
        spec = X[i]
        seg = np.fft.irfft(spec, n=NFFT).astype(np.float32)
        y[pos:pos+NFFT] += seg * WIN
        wsum[pos:pos+NFFT] += WIN_SQ
        pos += HOP
    wsum = np.where(wsum < 1e-6, 1.0, wsum).astype(np.float32)
    y /= wsum
    return y.astype(np.float32, copy=False)

# =========================
# Spectral masking (single and multi-target union)
# =========================
def spectral_drop_or_keep(mix: np.ndarray, sep: np.ndarray, mode: str,
                          alpha_drop=ALPHA_DROP, alpha_keep=ALPHA_KEEP,
                          p=P_MASK, gamma=MASK_GAMMA, drop_power=DROP_POWER,
                          drop_floor_db=DROP_FLOOR_DB) -> np.ndarray:
    X = stft(mix)
    S = stft(sep)
    F = max(X.shape[0], S.shape[0])
    if X.shape[0] < F:
        X = np.vstack([X, np.zeros((F - X.shape[0], X.shape[1]), dtype=X.dtype)])
    if S.shape[0] < F:
        S = np.vstack([S, np.zeros((F - S.shape[0], S.shape[1]), dtype=S.dtype)])
    aX = np.abs(X).astype(np.float32, copy=False)
    aS = np.abs(S).astype(np.float32, copy=False)
    aR = np.clip(aX - aS, 0.0, None).astype(np.float32, copy=False)
    M = (aS**p) / (aS**p + aR**p + EPS)
    M = np.clip(M, 0.0, 1.0).astype(np.float32, copy=False)
    if gamma != 1.0:
        M = (M ** np.float32(gamma)).astype(np.float32, copy=False)
    if mode == "keep":
        A = np.float32(alpha_keep)
        Y = (M * A).astype(np.float32, copy=False) * X
    else:
        A = np.float32(alpha_drop)
        floor_lin = np.float32(10.0 ** (drop_floor_db / 20.0))
        G = np.clip(1.0 - (A * M), floor_lin, 1.0).astype(np.float32, copy=False)
        if drop_power != 1.0:
            G = (G ** np.float32(drop_power)).astype(np.float32, copy=False)
        Y = G * X
    y = istft(Y)
    if len(y) > len(mix): y = y[:len(mix)]
    elif len(y) < len(mix): y = np.pad(y, (0, len(mix)-len(y)), mode='constant')
    return y.astype(np.float32, copy=False)

def spectral_drop_or_keep_multi(mix: np.ndarray, sep_list, mode: str,
                                alpha_drop=ALPHA_DROP, alpha_keep=ALPHA_KEEP,
                                p=P_MASK, gamma=MASK_GAMMA, drop_power=DROP_POWER,
                                drop_floor_db=DROP_FLOOR_DB) -> np.ndarray:
    X = stft(mix)
    aX = np.abs(X).astype(np.float32, copy=False)
    M_union = np.zeros_like(aX, dtype=np.float32)

    for sep in sep_list:
        S = stft(sep)
        F = max(X.shape[0], S.shape[0])
        if S.shape[0] < F:
            S = np.vstack([S, np.zeros((F - S.shape[0], S.shape[1]), dtype=S.dtype)])
        if X.shape[0] < F:
            X = np.vstack([X, np.zeros((F - X.shape[0], X.shape[1]), dtype=X.dtype)])
            aX = np.abs(X).astype(np.float32, copy=False)

        aS = np.abs(S).astype(np.float32, copy=False)
        aR = np.clip(aX - aS, 0.0, None).astype(np.float32, copy=False)
        M  = (aS**p) / (aS**p + aR**p + EPS)
        if gamma != 1.0:
            M = (M ** np.float32(gamma)).astype(np.float32, copy=False)
        M = np.clip(M, 0.0, 1.0)
        M_union = np.maximum(M_union, M, out=M_union)

    if mode == "keep":
        A = np.float32(alpha_keep)
        Y = (M_union * A).astype(np.float32, copy=False) * X
    else:
        A = np.float32(alpha_drop)
        floor_lin = np.float32(10.0 ** (drop_floor_db / 20.0))
        G = np.clip(1.0 - (A * M_union), floor_lin, 1.0).astype(np.float32, copy=False)
        if drop_power != 1.0:
            G = (G ** np.float32(drop_power)).astype(np.float32, copy=False)
        Y = G * X

    y = istft(Y)
    if len(y) > len(mix): y = y[:len(mix)]
    elif len(y) < len(mix): y = np.pad(y, (0, len(mix)-len(y)), mode='constant')
    return y.astype(np.float32, copy=False)

# =========================
# Device & Model
# =========================
if torch.backends.mps.is_available(): device = torch.device("mps")
elif torch.cuda.is_available():       device = torch.device("cuda")
else:                                  device = torch.device("cpu")

print(f"[INFO] Loading AudioSep on {device} ...", flush=True)
model = build_audiosep(
    config_yaml=str(AUDIOSEP_REPO / "config" / "audiosep_base.yaml"),
    checkpoint_path=str(AUDIOSEP_REPO / "checkpoint" / "audiosep_base_4M_steps.ckpt"),
    device=device,
)

chunk_len = int(CHUNK_S * SR)          # 32000
hop_len   = int(HOP_S * SR)            # 16000
assert hop_len * 2 == chunk_len, "Use 50% overlap for clean OLA."

# Hann for outer OLA (frame domain)
syn_win = get_window("hann", chunk_len).astype(np.float32)

# Rolling input buffer
in_ring  = np.zeros(chunk_len, dtype=np.float32)
in_write = 0
filled   = 0

# OLA buffer spanning one chunk; read hop_len each hop
ola_buf = np.zeros(chunk_len, dtype=np.float32)

print(f"[INFO] Capturing to:\n  BEFORE: {INPUT_WAV_PATH}\n  AFTER : {OUTPUT_WAV_PATH}")
input_sink  = sf.SoundFile(str(INPUT_WAV_PATH),  mode="w", samplerate=SR, channels=1, subtype="PCM_16")
output_sink = sf.SoundFile(str(OUTPUT_WAV_PATH), mode="w", samplerate=SR, channels=1, subtype="PCM_16")

print("[INFO] Running (50% OLA + EXTREME spectral mask + multi-target). Ctrl+C to stop.", flush=True)

_dc_state = [np.float32(0.0)]
have_processed_once = False

# Temp WAVs for model I/O
tmpdir = tempfile.TemporaryDirectory()
mix_wav = Path(tmpdir.name) / "chunk_mix.wav"
sep_wav = Path(tmpdir.name) / "chunk_sep.wav"

# Presence controllers
smoother = OnePole()
gate     = HoldGate()

# Start control poller
stop_poll = threading.Event()
t = threading.Thread(target=poll_control_state, args=(stop_poll,), daemon=True)
t.start()

def find_device_by_name(target_name: str, want_input: bool):
    name_lower = target_name.lower()
    devs = sd.query_devices()
    if want_input:
        for idx, d in enumerate(devs):
            if name_lower in d.get("name","").lower() and d.get("max_input_channels",0) > 0:
                return idx
    else:
        for idx, d in enumerate(devs):
            if name_lower in d.get("name","").lower() and d.get("max_output_channels",0) > 0:
                return idx
    return None

try:
    _ = sd.query_devices()
except Exception as e:
    print(f"[ERR] Could not query devices: {e}", flush=True); sys.exit(1)

INPUT_DEVICE_INDEX  = find_device_by_name(INPUT_DEVICE_NAME,  want_input=True)
OUTPUT_DEVICE_INDEX = find_device_by_name(OUTPUT_DEVICE_NAME, want_input=False)
if INPUT_DEVICE_INDEX is None:
    print(f"[ERR] Could not find input device like: '{INPUT_DEVICE_NAME}'", flush=True); sys.exit(1)
if OUTPUT_DEVICE_INDEX is None:
    print(f"[ERR] Could not find output device like: '{OUTPUT_DEVICE_NAME}'", flush=True); sys.exit(1)

sd.check_input_settings(device=INPUT_DEVICE_INDEX, channels=CHANNELS, samplerate=SR, dtype="float32")
sd.check_output_settings(device=OUTPUT_DEVICE_INDEX, channels=CHANNELS, samplerate=SR, dtype="float32")

try:
    with sd.Stream(device=(INPUT_DEVICE_INDEX, OUTPUT_DEVICE_INDEX),
                   samplerate=SR, blocksize=hop_len, dtype="float32",
                   channels=CHANNELS, latency="low") as stream:
        while True:
            mode_now, prompts_now, _ = CTRL.snapshot()

            frames, _ = stream.read(hop_len)
            x = to_mono(frames)
            x = dc_block(x, _dc_state)
            input_sink.write(x)

            # play last half from OLA
            out_seg = ola_buf[:hop_len].astype(np.float32, copy=False)
            out_seg = postprocess(out_seg)
            output_sink.write(out_seg)
            write_buf = out_seg if have_processed_once else np.zeros(out_seg.shape, dtype=np.float32)
            stream.write(write_buf[:, None])

            # shift OLA
            ola_buf[:-hop_len] = ola_buf[hop_len:]
            ola_buf[-hop_len:] = np.float32(0.0)

            # rolling input buffer
            end = in_write + hop_len
            if end <= len(in_ring):
                in_ring[in_write:end] = x
            else:
                first = len(in_ring) - in_write
                in_ring[in_write:] = x[:first]
                in_ring[:end - len(in_ring)] = x[first:]
            in_write = end % len(in_ring)
            filled = min(len(in_ring), filled + hop_len)

            if filled < chunk_len:
                continue

            # extract analysis chunk
            if in_write - chunk_len >= 0:
                chunk = in_ring[in_write - chunk_len:in_write].copy()
            else:
                a = in_ring[in_write - chunk_len:].copy()
                b = in_ring[:in_write].copy()
                chunk = np.concatenate([a, b]).astype(np.float32, copy=False)

            # Separate per prompt and union masks
            try:
                sep_list = []
                for prompt in prompts_now:
                    sf.write(mix_wav, chunk, SR)
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                        inference(model, str(mix_wav), prompt, str(sep_wav), device)
                    sep, sr_out = sf.read(sep_wav, dtype="float32", always_2d=False)
                    if sr_out != SR:
                        idx = np.linspace(0, len(sep)-1, len(chunk)).astype(np.float32)
                        sep = np.interp(idx, np.arange(len(sep), dtype=np.float32), sep.astype(np.float32)).astype(np.float32)
                    if len(sep) < len(chunk):
                        sep = np.pad(sep.astype(np.float32, copy=False), (0, len(chunk)-len(sep)), mode='constant')
                    sep = sep[:len(chunk)].astype(np.float32, copy=False)
                    sep_list.append(sep)

                # Presence estimate
                pr_raw = 0.0 if not sep_list else float(np.clip(max(rms(s) for s in sep_list) / (rms(chunk) + 1e-9), 0.0, 1.0))
                pr_s   = OnePole()(pr_raw)  # fresh smoother for each frame
                pr_g   = gate.update(pr_s)
                presence = pr_s * pr_g

                alpha = ALPHA_KEEP * (0.8 + 0.2 * presence) if mode_now == "keep" else ALPHA_DROP

                proc = spectral_drop_or_keep_multi(
                    chunk, sep_list, mode_now,
                    alpha_drop=alpha, alpha_keep=alpha,
                    p=P_MASK, gamma=MASK_GAMMA,
                    drop_power=DROP_POWER, drop_floor_db=DROP_FLOOR_DB
                )

                if TWO_PASS and mode_now == "drop":
                    resid = (chunk - proc).astype(np.float32, copy=False)
                    proc  = spectral_drop_or_keep_multi(
                        proc, [resid], "drop",
                        alpha_drop=min(0.9995, ALPHA_DROP * 1.01),
                        alpha_keep=ALPHA_KEEP,
                        p=max(P_MASK, 4.0),
                        gamma=max(MASK_GAMMA, 1.8),
                        drop_power=max(DROP_POWER, 2.0),
                        drop_floor_db=DROP_FLOOR_DB
                    )

                proc_win = (proc * syn_win).astype(np.float32, copy=False)
                ola_buf += proc_win
                have_processed_once = True

            except Exception as e:
                print(f"[WARN] separation failed: {e}", flush=True)
                # keep streaming

except KeyboardInterrupt:
    print("\n[INFO] Stopping and closing files...")
finally:
    stop_poll.set()
    try: input_sink.close()
    except: pass
    try: output_sink.close()
    except: pass
    print(f"[INFO] Saved:\n  BEFORE: {INPUT_WAV_PATH}\n  AFTER : {OUTPUT_WAV_PATH}")
