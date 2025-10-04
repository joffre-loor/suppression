#!/usr/bin/env python3
import os, sys, time, tempfile, io, contextlib
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import get_window
from datetime import datetime

# =========================
#  Fixed device NAMES (by name, not index)
# =========================
INPUT_DEVICE_NAME  = "MacBook Pro Microphone"  # capture the room
OUTPUT_DEVICE_NAME = "WH-1000XM5"              # your headphones (output-capable entry)

# ------- AudioSep repo path -------
AUDIOSEP_REPO = Path(__file__).resolve().parent / "AudioSep"
if str(AUDIOSEP_REPO) not in sys.path:
    sys.path.append(str(AUDIOSEP_REPO))

import torch
from pipeline import build_audiosep, separate_audio as inference  # correct import

# ------- Defaults (used if file missing/invalid) -------
DEFAULT_PROMPT = "music"   # target to remove or keep
DEFAULT_MODE   = "drop"    # "drop" (remove target) or "keep" (isolate target)

# ------- IO / Realtime params -------
SR       = 32000           # AudioSep expects 32 kHz
CHUNK_S  = 1.0             # seconds per separation chunk
HOP_S    = 0.5             # seconds between chunk starts (50% overlap)
CHANNELS = 1               # mono
ATTEN_DB = 60.0            # suppression depth when mode="drop"
PROMPT_FILE = Path("filter_list.txt")
RELOAD_EVERY_SEC = 1.0

# ------- Artifact controls -------
SEP_GATE_DB          = -35.0   # ignore separated stem if below this relative level
ATTACK_COEFF         = 0.2     # smoother attack
RELEASE_COEFF        = 0.08    # smoother release
HOLD_SECONDS         = 0.20    # hold time after gate opens
OUTPUT_SOFT_LIMIT_DB = -0.8    # soft limit target (dBFS)
DC_ALPHA             = 0.995   # DC blocker coefficient (per-sample)

# ------- Small-spike cleanup -------
OUT_FLOOR_DB   = -50.0   # hard gate floor on the playback hop
EXP_THRESH_DB  = -45.0   # expander threshold
EXP_RATIO      = 2.0     # expander ratio (2:1)
DELTA_LIMIT_DB = -10.0   # per-sample step clamp (dBFS), e.g. -10 dB ≈ 0.316 linear

# ------- Capture files (before/after) -------
CAPTURE_ROOT = Path("captures")
CAPTURE_ROOT.mkdir(exist_ok=True)
RUN_DIR = CAPTURE_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
INPUT_WAV_PATH  = RUN_DIR / "input_raw.wav"
OUTPUT_WAV_PATH = RUN_DIR / "output_after.wav"

# ------- Helpers -------
def db_to_lin(db): 
    return 10 ** (-abs(db) / 20.0)

def lin_to_db(x): 
    return 20.0 * np.log10(max(x, 1e-12))

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

def dc_block(x, z=[0.0]):
    """Simple one-pole DC blocker."""
    y = np.empty_like(x)
    prev = z[0]
    prev_y = 0.0
    for i, s in enumerate(x):
        curr = s - prev + DC_ALPHA * prev_y
        y[i] = curr
        prev_y = curr
        prev = s
    z[0] = prev
    return y

class OnePole:
    """Simple attack/release smoother for a control signal in [0,1]."""
    def __init__(self, alpha_up=ATTACK_COEFF, alpha_down=RELEASE_COEFF):
        self.y = 0.0
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
    def __call__(self, x: float) -> float:
        a = self.alpha_up if x > self.y else self.alpha_down
        self.y = (1 - a) * self.y + a * x
        return self.y

class HoldGate:
    """Gate with level threshold (dB) and hold time to prevent chatter."""
    def __init__(self, thresh_db=SEP_GATE_DB, hold_s=HOLD_SECONDS, hop_s=HOP_S):
        self.thresh_lin = 10**(thresh_db/20.0)
        self.hold_frames = max(1, int(hold_s / hop_s))
        self.counter = 0
        self.open = False
    def update(self, presence_lin: float) -> float:
        if presence_lin >= self.thresh_lin:
            self.open = True
            self.counter = self.hold_frames
        else:
            if self.counter > 0:
                self.counter -= 1
                self.open = True
            else:
                self.open = False
        return 1.0 if self.open else 0.0

def soft_limiter(x, target_db=OUTPUT_SOFT_LIMIT_DB):
    """Gentle limiter to prevent occasional overs."""
    target = 10**(target_db/20.0)
    peak = np.max(np.abs(x)) + 1e-9
    if peak <= target:
        return x
    return np.tanh(x * (1.5 / target)) * (target / np.tanh(1.5))

def overlap_add(out_buf, weight_buf, write_pos, chunk, window):
    """OLA with weight tracking for perfect normalization at read time."""
    n = len(chunk)
    end = write_pos + n
    if end <= len(out_buf):
        out_buf[write_pos:end] += chunk * window
        weight_buf[write_pos:end] += window
    else:
        first = len(out_buf) - write_pos
        out_buf[write_pos:] += (chunk[:first] * window[:first])
        out_buf[:end - len(out_buf)] += (chunk[first:] * window[first:])
        weight_buf[write_pos:] += window[:first]
        weight_buf[:end - len(out_buf)] += window[first:]
    return out_buf, weight_buf, end % len(out_buf)

def to_mono(x):
    return x if x.ndim == 1 else np.mean(x, axis=1)

def parse_prompt_file(p: Path):
    """
    Accepts one of:
      [music]
      drop: music
      keep: music
    First valid token wins. Whitespace/case ignored.
    """
    DEFAULT_PROMPT = "music"
    DEFAULT_MODE   = "drop"
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return DEFAULT_PROMPT, DEFAULT_MODE
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return DEFAULT_PROMPT, DEFAULT_MODE
    for ln in lines:
        low = ln.lower()
        if low.startswith("drop:"):
            val = low.split("drop:", 1)[1].strip(" []\t")
            if val: return val, "drop"
        if low.startswith("keep:"):
            val = low.split("keep:", 1)[1].strip(" []\t")
            if val: return val, "keep"
        if low.startswith("[") and low.endswith("]") and len(low) > 2:
            val = low[1:-1].strip()
            if val: return val, "drop"
    return DEFAULT_PROMPT, DEFAULT_MODE

def find_device_by_name(target_name: str, want_input: bool):
    """Find device index by (substring) name and capability."""
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

# --- Expander & spike clamp ---
def downward_expander(x: np.ndarray, thresh_db=EXP_THRESH_DB, ratio=EXP_RATIO):
    thr = 10**(thresh_db/20.0)
    level = rms(x) + 1e-12
    if level >= thr:
        return x
    g = (level / thr) ** max(0.0, (ratio - 1.0))
    g = float(np.clip(g, 0.0, 1.0))
    return x * g

def clamp_spikes(x: np.ndarray, limit_db=DELTA_LIMIT_DB):
    limit = 10**(limit_db/20.0)  # e.g., -10 dB -> ~0.316 linear step
    if limit >= 1.0:
        return x
    y = x.copy()
    for i in range(1, len(y)):
        step = y[i] - y[i-1]
        max_step = np.sign(step) * min(abs(step), limit)
        y[i] = y[i-1] + max_step
    return y

# ------- Device selection (prefer Apple Silicon GPU -> CUDA -> CPU) -------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[INFO] Loading AudioSep on {device} ...", flush=True)
model = build_audiosep(
    config_yaml=str(AUDIOSEP_REPO / "config" / "audiosep_base.yaml"),
    checkpoint_path=str(AUDIOSEP_REPO / "checkpoint" / "audiosep_base_4M_steps.ckpt"),
    device=device,
)

# ------- Buffers / constants -------
chunk_len = int(CHUNK_S * SR)
hop_len   = int(HOP_S * SR)

# Hann synthesis window; normalize via weight buffer for perfect OLA
window = get_window("hann", chunk_len, fftbins=True).astype(np.float32)

in_ring = np.zeros(chunk_len, dtype=np.float32)
in_write = 0

out_ring    = np.zeros(2 * chunk_len, dtype=np.float32)
weight_ring = np.zeros(2 * chunk_len, dtype=np.float32)
out_read = 0
out_write = 0

drop_gain = db_to_lin(ATTEN_DB)

tmpdir = tempfile.TemporaryDirectory()
mix_wav = Path(tmpdir.name) / "chunk_mix.wav"
sep_wav = Path(tmpdir.name) / "chunk_sep.wav"

# Controls
sep_level_smoother = OnePole()
gate               = HoldGate()

# ------- Prompt hot-reload state -------
current_prompt, current_mode = parse_prompt_file(PROMPT_FILE)
last_check = 0.0
last_mtime = PROMPT_FILE.stat().st_mtime if PROMPT_FILE.exists() else 0.0
print(f"[INFO] Initial target -> mode='{current_mode}' prompt='{current_prompt}'", flush=True)

def maybe_reload_prompt():
    global current_prompt, current_mode, last_check, last_mtime
    now = time.time()
    if now - last_check < RELOAD_EVERY_SEC:
        return
    last_check = now
    if PROMPT_FILE.exists():
        mtime = PROMPT_FILE.stat().st_mtime
        if mtime != last_mtime:
            last_mtime = mtime
            new_prompt, new_mode = parse_prompt_file(PROMPT_FILE)
            if (new_prompt != current_prompt) or (new_mode != current_mode):
                current_prompt, current_mode = new_prompt, new_mode
                print(f"[INFO] Updated target -> mode='{current_mode}' prompt='{current_prompt}'", flush=True)

# ------- Resolve devices by NAME (not index) -------
try:
    _ = sd.query_devices()
except Exception as e:
    print(f"[ERR] Could not query devices: {e}", flush=True)
    sys.exit(1)

INPUT_DEVICE_INDEX  = find_device_by_name(INPUT_DEVICE_NAME,  want_input=True)
OUTPUT_DEVICE_INDEX = find_device_by_name(OUTPUT_DEVICE_NAME, want_input=False)

if INPUT_DEVICE_INDEX is None:
    print(f"[ERR] Could not find input device named like: '{INPUT_DEVICE_NAME}'", flush=True)
    print("      Run: python -m sounddevice  and paste the list here if you need help.")
    sys.exit(1)
if OUTPUT_DEVICE_INDEX is None:
    print(f"[ERR] Could not find output device named like: '{OUTPUT_DEVICE_NAME}'", flush=True)
    sys.exit(1)

inp_desc  = sd.query_devices(INPUT_DEVICE_INDEX)
out_desc  = sd.query_devices(OUTPUT_DEVICE_INDEX)
print(f"[INFO] Using INPUT  [{INPUT_DEVICE_INDEX}]: {inp_desc['name']}  (in:{inp_desc['max_input_channels']}, out:{inp_desc['max_output_channels']})")
print(f"[INFO] Using OUTPUT [{OUTPUT_DEVICE_INDEX}]: {out_desc['name']}  (in:{out_desc['max_input_channels']}, out:{out_desc['max_output_channels']})")

# Extra sanity checks
try:
    sd.check_input_settings(device=INPUT_DEVICE_INDEX, channels=CHANNELS, samplerate=SR, dtype="float32")
    sd.check_output_settings(device=OUTPUT_DEVICE_INDEX, channels=CHANNELS, samplerate=SR, dtype="float32")
except Exception as e:
    print(f"[ERR] Device settings check failed: {e}", flush=True)
    sys.exit(1)

print(f"[INFO] Capturing to:\n  BEFORE (raw mic):  {INPUT_WAV_PATH}\n  AFTER  (processed): {OUTPUT_WAV_PATH}")

# Open capture files (append throughout the run)
input_sink  = sf.SoundFile(str(INPUT_WAV_PATH),  mode="w", samplerate=SR, channels=1, subtype="PCM_16")
output_sink = sf.SoundFile(str(OUTPUT_WAV_PATH), mode="w", samplerate=SR, channels=1, subtype="PCM_16")

# ------- Audio stream loop -------
print("[INFO] Running. Edit 'filter_list.txt' to change target (e.g., '[music]' or 'keep: speech'). Ctrl+C to stop.", flush=True)

# DC state
_dc_state = [0.0]

try:
    with sd.Stream(device=(INPUT_DEVICE_INDEX, OUTPUT_DEVICE_INDEX),
                   samplerate=SR, blocksize=hop_len, dtype="float32",
                   channels=CHANNELS, latency="low") as stream:
        filled = 0
        while True:
            frames, _ = stream.read(hop_len)
            x = to_mono(frames)

            # DC block the input a touch to avoid low-frequency bias
            x = dc_block(x, _dc_state)

            # Write raw mic to capture ("before")
            input_sink.write(x)

            # Write input to ring
            end = in_write + hop_len
            if end <= len(in_ring):
                in_ring[in_write:end] = x
            else:
                first = len(in_ring) - in_write
                in_ring[in_write:] = x[:first]
                in_ring[:end - len(in_ring)] = x[first:]
            in_write = end % len(in_ring)

            # Prepare one hop to play from output ring with weight normalization
            play_end = out_read + hop_len
            if play_end <= len(out_ring):
                play   = out_ring[out_read:play_end].copy()
                wslice = weight_ring[out_read:play_end].copy()
                out_ring[out_read:play_end]    = 0.0
                weight_ring[out_read:play_end] = 0.0
            else:
                first = len(out_ring) - out_read
                play   = np.concatenate([out_ring[out_read:], out_ring[:play_end - len(out_ring)]])
                wslice = np.concatenate([weight_ring[out_read:], weight_ring[:play_end - len(out_ring)]])
                out_ring[out_read:]    = 0.0
                weight_ring[out_read:] = 0.0
                out_ring[:play_end - len(out_ring)]    = 0.0
                weight_ring[:play_end - len(out_ring)] = 0.0
            out_read = play_end % len(out_ring)

            # Normalize overlapped window sums to unity to avoid combing/pumping
            wsafe = np.where(wslice > 1e-6, wslice, 1.0)
            play_norm = play / wsafe

            # --- Small-spike cleanup on the playback hop ---
            hop_rms = rms(play_norm)
            if lin_to_db(hop_rms) < OUT_FLOOR_DB:
                play_norm[:] = 0.0
            else:
                play_norm = downward_expander(play_norm, EXP_THRESH_DB, EXP_RATIO)
                play_norm = soft_limiter(play_norm, OUTPUT_SOFT_LIMIT_DB)
                play_norm = clamp_spikes(play_norm, DELTA_LIMIT_DB)

            # Write processed to capture ("after")
            output_sink.write(play_norm)

            # Before first processed chunk, just pass dry mic; otherwise processed
            if filled < chunk_len and np.allclose(play_norm, 0.0):
                stream.write(x[:, None])
            else:
                stream.write(play_norm[:, None])

            # Build up enough audio for first chunk
            filled += hop_len
            if filled < chunk_len:
                maybe_reload_prompt()
                continue

            # Extract analysis chunk (last CHUNK_S from ring)
            if in_write - chunk_len >= 0:
                chunk = in_ring[in_write - chunk_len:in_write].copy()
            else:
                a = in_ring[in_write - chunk_len:].copy()
                b = in_ring[:in_write].copy()
                chunk = np.concatenate([a, b])

            # Hot-reload prompt if file changed
            maybe_reload_prompt()

            # Run separation on the chunk
            try:
                sf.write(mix_wav, chunk, SR)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    inference(model, str(mix_wav), current_prompt, str(sep_wav), device)

                sep, sr_out = sf.read(sep_wav, dtype="float32", always_2d=False)
                if sr_out != SR:
                    idx = np.linspace(0, len(sep)-1, len(chunk))
                    sep = np.interp(idx, np.arange(len(sep)), sep)
                sep = sep[:len(chunk)]

                # Relative presence & gating
                mix_r = rms(chunk)
                sep_r = rms(sep)
                presence_raw = np.clip(sep_r / (mix_r + 1e-9), 0.0, 1.0)
                gate_open = gate.update(presence_raw)
                presence_s = OnePole()(presence_raw) * gate_open

                # If mode=keep, output stem only (smoothed); else subtract stem (scaled)
                if current_mode == "keep":
                    proc = sep * presence_s
                else:
                    g = (1.0 - drop_gain) * presence_s
                    proc = chunk - g * sep

                # OLA into output ring with tracked weights for perfect normalization
                out_ring, weight_ring, out_write = overlap_add(out_ring, weight_ring, out_write, proc, window)

            except Exception as e:
                print(f"[WARN] separation failed: {e}", flush=True)
                continue

except KeyboardInterrupt:
    print("\n[INFO] Stopping and closing files...")

finally:
    try:
        input_sink.close()
    except Exception:
        pass
    try:
        output_sink.close()
    except Exception:
        pass
    print(f"[INFO] Saved:\n  BEFORE: {INPUT_WAV_PATH}\n  AFTER : {OUTPUT_WAV_PATH}")
