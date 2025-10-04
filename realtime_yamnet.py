# realtime_yamnet.py  (keep this name forever)
import argparse, queue, sys, time, re, os
from collections import deque
from math import gcd
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly

# Cut TF log noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# -------- Core config --------
TARGET_SR = 16000
WIN_SECS  = 0.975
HOP_SECS  = 0.10
FRAME_LEN = int(WIN_SECS * TARGET_SR)

print("Loading YAMNet…")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

def load_labels():
    try:
        p = yamnet.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(p, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return [ln.split(",")[-1].strip('"') for ln in lines[1:]]
    except Exception:
        # fallback numeric names
        return [f"class_{i}" for i in range(521)]
LABELS = load_labels()

class EMA:
    def __init__(self, alpha=0.6):
        self.alpha = float(alpha)
        self.v = None
    def update(self, x: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v

ring = deque(maxlen=FRAME_LEN)
q = queue.Queue()
level_ema = 0.0

def meter_update(x: np.ndarray):
    global level_ema
    rms = np.sqrt(np.mean(np.square(x))) + 1e-12
    level_ema = 0.9 * level_ema + 0.1 * rms

def audio_cb(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata[:, 0].astype(np.float32, copy=True))

def clearline(use_ansi=True):
    if use_ansi:
        sys.stdout.write("\x1b[2K\r")  # ANSI clear line + CR
        sys.stdout.flush()

def main(args):
    # resolve input device index
    if args.device is None:
        d = sd.default.device
        input_idx = d[0] if isinstance(d, (list, tuple)) else d
    else:
        input_idx = args.device

    devinfo = sd.query_devices(input_idx, 'input')
    native_sr = int(devinfo.get("default_samplerate") or 48000)

    # resample ratios (native -> TARGET_SR)
    up, down = TARGET_SR, native_sr
    g = gcd(up, down); up //= g; down //= g

    # stream blocksize: ~hop seconds at native rate (>=256)
    hop_native = max(256, int(args.hop * native_sr))

    # smoothing + filters
    ema = EMA(alpha=args.smooth) if args.smooth > 0 else None
    include_re = re.compile(args.include, re.I) if args.include else None
    exclude_re = re.compile(args.exclude, re.I) if args.exclude else None
    use_ansi = not args.no_ansi

    print(f"Using input device #{input_idx}: {devinfo['name']}")
    print(f"Native SR {native_sr} Hz → resample → {TARGET_SR} Hz")
    print(f"topk={args.topk}  min_prob={args.min_prob:.2f}  smooth={args.smooth:.2f}  include='{args.include or ''}'  exclude='{args.exclude or ''}'")
    print("(Ctrl+C to stop)\n")

    # warm up model to avoid first-hop stall
    _ = yamnet(np.zeros(FRAME_LEN, dtype=np.float32))

    last = 0.0
    with sd.InputStream(device=input_idx,
                        samplerate=native_sr,
                        channels=1,
                        blocksize=hop_native,
                        dtype="float32",
                        callback=audio_cb,
                        latency="low"):
        while True:
            buf = q.get()
            meter_update(buf)

            # resample chunk to target sr for model
            resampled = resample_poly(buf, up, down).astype(np.float32)

            # fill rolling window
            ring.extend(resampled)
            if len(ring) < FRAME_LEN:
                if time.time() - last > 0.5:
                    lvl_db = 20*np.log10(level_ema + 1e-6)
                    if use_ansi:
                        clearline(use_ansi); print(f"[level ~ {lvl_db:5.1f} dB] filling buffer…", end="", flush=True)
                    else:
                        print(f"[level ~ {lvl_db:5.1f} dB] filling buffer…", flush=True)
                    last = time.time()
                continue

            # run model
            window = np.array(ring, dtype=np.float32)
            scores, _, _ = yamnet(window)            # [T, C]
            avg = scores.numpy().mean(axis=0)        # [C]
            if ema: avg = ema.update(avg)

            # filter & sort
            items = []
            for i, p in enumerate(avg):
                name = LABELS[i]
                if include_re and not include_re.search(name):  # whitelist
                    continue
                if exclude_re and exclude_re.search(name):      # blacklist
                    continue
                if p < args.min_prob:
                    continue
                items.append((name, float(p)))

            if items:
                items.sort(key=lambda x: x[1], reverse=True)
                items = items[:args.topk]
            else:
                # show best single class if all filtered
                i_top = int(np.argmax(avg))
                items = [(LABELS[i_top], float(avg[i_top]))]

            # HUD at hop cadence
            if time.time() - last >= args.hop - 1e-3:
                lvl_db = 20*np.log10(level_ema + 1e-6)
                line = " | ".join(f"{n}: {p:.2f}" for n, p in items)
                if use_ansi:
                    clearline(use_ansi); print(f"[level ~ {lvl_db:5.1f} dB] {line}", end="", flush=True)
                else:
                    print(f"[level ~ {lvl_db:5.1f} dB] {line}", flush=True)
                last = time.time()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",   type=int, default=None, help="Input device index (see sd.query_devices())")
    ap.add_argument("--hop",      type=float, default=HOP_SECS, help="Decision cadence in seconds")
    ap.add_argument("--topk",     type=int, default=5)
    ap.add_argument("--min_prob", type=float, default=0.12, help="Minimum probability to display")
    ap.add_argument("--smooth",   type=float, default=0.6,  help="EMA alpha 0..1 (0 disables smoothing)")
    # quick shaping: whitelist / blacklist regexes
    ap.add_argument("--include",  type=str, default=None,   help="Regex of labels to keep (case-insensitive)")
    ap.add_argument("--exclude",  type=str, default="room|hall|public space|rural|urban|animal|bird|horse|rail|train|vehicle|transport",
                    help="Regex of labels to drop (case-insensitive)")
    ap.add_argument("--no_ansi", action="store_true", help="Disable ANSI clearline; print a new line each hop")
    # notebook-safe parse:
    args, _ = ap.parse_known_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nStopped.")
