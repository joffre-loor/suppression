#!/usr/bin/env python3
# Extreme suppression, spectral-mask + proper OLA, prompt hot-reload
# Multi-target (union) prompts, DTYPE-SAFE (float32 end-to-end)

import os, sys, time, tempfile, io, contextlib
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf  
from scipy.signal import get_window
from datetime import datetime

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
# Defaults / I/O params
# =========================
DEFAULT_PROMPT = "music"
DEFAULT_MODE   = "drop"           # or "keep"

SR        = 32000
CHUNK_S   = 1.0                   # analysis window (seconds)
HOP_S     = 0.5                   # 50% overlap
CHANNELS  = 1
PROMPT_FILE = Path("filter_list.txt")
RELOAD_EVERY_SEC = 0.25

# =========================
# EXTREME spectral mask params (more aggressive)
# =========================
NFFT          = 4096              # higher = better freq selectivity (more CPU)
HOP           = NFFT // 2
WIN           = get_window("hann", NFFT).astype(np.float32)
WIN_SQ        = (WIN * WIN).astype(np.float32)
EPS           = np.float32(1e-7)

P_MASK        = 4.0               # Wiener exponent (peakier mask)
ALPHA_DROP    = 0.999             # mask scale in DROP
ALPHA_KEEP    = 1.00              # mask scale in KEEP
MASK_GAMMA    = 1.8               # mask sharpening (>1)
DROP_POWER    = 2.0               # compound drop power
DROP_FLOOR_DB = -70.0             # min per-bin gain in DROP

TWO_PASS      = True              # run a second pass on residual (DROP only)

# Presence / gating (minor alpha modulation only)
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
# Multi-target prompt parsing
# =========================
def _split_items(s: str):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 1 and " " in parts[0]:
        parts = [p for p in parts[0].split() if p]
    return [p for p in parts if p]

def parse_prompt_file(p: Path):
    mode = DEFAULT_MODE
    items = []
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        print("[WARN] prompt file not readable; using defaults")
        return [DEFAULT_PROMPT], mode
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        print("[WARN] prompt file empty; using defaults")
        return [DEFAULT_PROMPT], mode

    for ln in lines:
        low = ln.lower()
        if low.startswith("drop:"):
            mode = "drop"
            payload = ln.split(":",1)[1].strip()
            items += _split_items(payload)
        elif low.startswith("keep:"):
            mode = "keep"
            payload = ln.split(":",1)[1].strip()
            items += _split_items(payload)
        elif low.startswith("[") and low.endswith("]"):
            mode = "drop"
            items += _split_items(ln)
        else:
            items.append(ln)
    items = [it for it in (it.strip() for it in items) if it]
    return (items or [DEFAULT_PROMPT]), mode

# =========================
# STFT / ISTFT (Hann, 50% hop) — float32-safe
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

# Rolling input buffer (last chunk_len samples)
in_ring  = np.zeros(chunk_len, dtype=np.float32)
in_write = 0
filled   = 0

# OLA buffer spanning one chunk; read hop_len each hop
ola_buf = np.zeros(chunk_len, dtype=np.float32)

# Prompt hot-reload (content-hash)
current_prompts, current_mode = parse_prompt_file(PROMPT_FILE)  # list[str], mode
_last_check = 0.0
_last_hash  = None
print(f"[INFO] Initial target -> mode='{current_mode}' prompts={current_prompts}", flush=True)
print(f"[INFO] Watching prompt file at: {PROMPT_FILE.resolve()}")

def file_text_or_none(p: Path):
    try: return p.read_text(encoding="utf-8", errors="ignore")
    except Exception: return None

def maybe_reload_prompt():
    global current_prompts, current_mode, _last_check, _last_hash
    now = time.time()
    if now - _last_check < RELOAD_EVERY_SEC: return
    _last_check = now
    txt = file_text_or_none(PROMPT_FILE)
    if txt is None: return
    h = hash(txt)
    if h == _last_hash: return
    _last_hash = h
    prompts, md = parse_prompt_file(PROMPT_FILE)
    md = (md or DEFAULT_MODE).strip().lower()
    if prompts != current_prompts or md != current_mode:
        current_prompts, current_mode = prompts, md
        print(f"[INFO] Updated target -> mode='{current_mode}' prompts={current_prompts}", flush=True)

# Device resolve
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

# Files
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

# Presence controllers (minor alpha modulation only)
smoother = OnePole()
gate     = HoldGate()

try:
    with sd.Stream(device=(INPUT_DEVICE_INDEX, OUTPUT_DEVICE_INDEX),
                   samplerate=SR, blocksize=hop_len, dtype="float32",
                   channels=CHANNELS, latency="low") as stream:
        while True:
            maybe_reload_prompt()

            frames, _ = stream.read(hop_len)
            x = to_mono(frames)
            x = dc_block(x, _dc_state)
            input_sink.write(x)

            # Playback front half of OLA (ensure float32)
            out_seg = ola_buf[:hop_len].astype(np.float32, copy=False)
            out_seg = postprocess(out_seg)
            output_sink.write(out_seg)
            write_buf = out_seg if have_processed_once else np.zeros(out_seg.shape, dtype=np.float32)
            write_buf = write_buf.astype(np.float32, copy=False)
            stream.write(write_buf[:, None])

            # Roll OLA by hop
            ola_buf[:-hop_len] = ola_buf[hop_len:]
            ola_buf[-hop_len:] = np.float32(0.0)

            # Update in_ring with new audio
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

            # Extract analysis chunk
            if in_write - chunk_len >= 0:
                chunk = in_ring[in_write - chunk_len:in_write].copy()
            else:
                a = in_ring[in_write - chunk_len:].copy()
                b = in_ring[:in_write].copy()
                chunk = np.concatenate([a, b]).astype(np.float32, copy=False)

            # === Multi-target separation and union mask ===
            try:
                sep_list = []
                for prompt in current_prompts:
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

                # Presence (use loudest separated stem)
                pr_raw = 0.0 if not sep_list else float(np.clip(max(rms(s) for s in sep_list) / (rms(chunk) + 1e-9), 0.0, 1.0))
                pr_s   = smoother(pr_raw)
                pr_g   = gate.update(pr_s)
                presence = pr_s * pr_g

                # Choose alpha
                alpha = ALPHA_KEEP * (0.8 + 0.2 * presence) if current_mode == "keep" else ALPHA_DROP

                # Multi-target spectral suppression
                proc = spectral_drop_or_keep_multi(chunk, sep_list, current_mode,
                                                   alpha_drop=alpha, alpha_keep=alpha,
                                                   p=P_MASK, gamma=MASK_GAMMA,
                                                   drop_power=DROP_POWER, drop_floor_db=DROP_FLOOR_DB)

                # Optional second pass (DROP only)
                if TWO_PASS and current_mode == "drop":
                    resid = (chunk - proc).astype(np.float32, copy=False)
                    proc  = spectral_drop_or_keep_multi(proc, [resid], "drop",
                                                        alpha_drop=min(0.9995, ALPHA_DROP * 1.01),
                                                        alpha_keep=ALPHA_KEEP,
                                                        p=max(P_MASK, 4.0), gamma=max(MASK_GAMMA, 1.8),
                                                        drop_power=max(DROP_POWER, 2.0),
                                                        drop_floor_db=DROP_FLOOR_DB)

                # Outer OLA (window then add)
                proc = proc.astype(np.float32, copy=False)
                proc_win = (proc * syn_win).astype(np.float32, copy=False)
                ola_buf += proc_win
                have_processed_once = True

                # Debug (light)
                if int(time.time() * 2) % 2 == 0:
                    print(f"[DBG] mode={current_mode} prompts={current_prompts} "
                          f"mix={lin_to_db(rms(chunk)):5.1f}dB "
                          f"presence={presence:0.3f} alpha={alpha:0.3f}", flush=True)

            except Exception as e:
                print(f"[WARN] separation failed: {e}", flush=True)
                # keep streaming; try again next hop

except KeyboardInterrupt:
    print("\n[INFO] Stopping and closing files...")
finally:
    try: input_sink.close()
    except: pass
    try: output_sink.close()
    except: pass
    print(f"[INFO] Saved:\n  BEFORE: {INPUT_WAV_PATH}\n  AFTER : {OUTPUT_WAV_PATH}")
