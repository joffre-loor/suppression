import argparse, queue, sys, time, re, os
from collections import deque
from math import gcd
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

import torch, numpy as _np
torch.serialization.add_safe_globals([_np.core.multiarray._reconstruct])
_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _patched_load

from panns_inference import AudioTagging

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = os.path.join(os.path.dirname(__file__), "panns_data")
os.environ["PANNs_DATA"] = ROOT

TARGET_SR = 32000
WIN_SECS  = 1.0
HOP_SECS  = 0.10
FRAME_LEN = int(WIN_SECS * TARGET_SR)

CKPT = os.path.join(ROOT, "Cnn14_mAP=0.431.pth")
print(f"Checkpoint path: {CKPT}")
model = AudioTagging(checkpoint_path=CKPT, device="cpu")
LABELS = getattr(model, "labels", [f"class_{i}" for i in range(527)])

class EMA:
    def __init__(self, a=0.3): self.a, self.v = float(a), None
    def update(self, x):
        self.v = x if self.v is None else (self.a * x + (1 - self.a) * self.v)
        return self.v

ring = deque(maxlen=FRAME_LEN)
q = queue.Queue()
level_ema = 0.0

def meter_update(x):
    global level_ema
    rms = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
    level_ema = 0.9 * level_ema + 0.1 * rms

def audio_cb(indata, frames, time_info, status):
    if status: print(status, file=sys.stderr)
    q.put(indata[:, 0].astype(np.float32, copy=True))

def clearline(use_ansi=True):
    if use_ansi:
        sys.stdout.write("\x1b[2K\r"); sys.stdout.flush()

def get_probs(window):
    # window: 1-D float32 audio at 32 kHz
    x = np.asarray(window, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]  # (batch=1, time)

    import torch
    with torch.no_grad():
        t = torch.from_numpy(x)  # (1, N)
        out = model.model(t, None)  # call the actual CNN14 module

    # out is a dict with 'clipwise_output' as a torch.Tensor
    clip = out.get("clipwise_output", None)
    if clip is None:
        raise RuntimeError("panns model did not return 'clipwise_output'")

    arr = clip.cpu().numpy().astype(np.float32)  # (1, 527)
    return arr[0]
    # Support both APIs:
    #  - new: inference(audio, sr)
    #  - old: inference(audio)  (expects 32 kHz already)
    try:
        out = model.inference(window, TARGET_SR)   # new API
    except TypeError:
        out = model.inference(window)              # old API

    if isinstance(out, dict) and "clipwise_output" in out:
        arr = out["clipwise_output"]
        if isinstance(arr, np.ndarray):
            return arr.astype(np.float32)

    if isinstance(out, dict) and "tags" in out:
        probs = np.zeros(len(LABELS), dtype=np.float32)
        name_to_idx = {n: i for i, n in enumerate(LABELS)}
        for name, p in out["tags"]:
            i = name_to_idx.get(name)
            if i is not None:
                probs[i] = float(p)
        return probs

    if isinstance(out, (list, tuple)):
        probs = np.zeros(len(LABELS), dtype=np.float32)
        name_to_idx = {n: i for i, n in enumerate(LABELS)}
        for name, p in out:
            i = name_to_idx.get(name)
            if i is not None:
                probs[i] = float(p)
        return probs

    raise RuntimeError("Unexpected panns-inference output format")


def main(args):
    if args.device is None:
        d = sd.default.device
        input_idx = d[0] if isinstance(d, (list, tuple)) else d
    else:
        input_idx = args.device

    dinfo = sd.query_devices(input_idx, "input")
    native_sr = int(dinfo.get("default_samplerate") or 48000)

    up, down = TARGET_SR, native_sr
    g = gcd(up, down); up //= g; down //= g
    hop_native = max(256, int(args.hop * native_sr))

    ema = EMA(args.smooth) if args.smooth > 0 else None
    include_re = re.compile(args.include, re.I) if args.include else None
    exclude_re = re.compile(args.exclude, re.I) if args.exclude else None
    use_ansi = not args.no_ansi

    print(f"Using input device #{input_idx}: {dinfo['name']}")
    print(f"Native SR {native_sr} Hz → resample → {TARGET_SR} Hz (PANNs CNN14)")
    print(f"topk={args.topk}  min_prob={args.min_prob:.2f}  smooth={args.smooth:.2f}  force_topk={args.force_topk}")
    print("(Ctrl+C to stop)\n")

    last = 0.0
    with sd.InputStream(device=input_idx, samplerate=native_sr, channels=1,
                        blocksize=hop_native, dtype="float32",
                        callback=audio_cb, latency="low"):
        while True:
            buf = q.get()
            meter_update(buf)
            resampled = resample_poly(buf, up, down).astype(np.float32)

            ring.extend(resampled)
            if len(ring) < FRAME_LEN:
                if time.time() - last > 0.5:
                    lvl = 20*np.log10(level_ema + 1e-6)
                    if use_ansi: clearline(use_ansi); print(f"[level ~ {lvl:5.1f} dB] filling buffer…", end="", flush=True)
                    else: print(f"[level ~ {lvl:5.1f} dB] filling buffer…", flush=True)
                    last = time.time()
                continue

            window = np.array(ring, dtype=np.float32)
            probs = get_probs(window)
            if ema: probs = ema.update(probs)

            items = []
            for i, p in enumerate(probs):
                name = LABELS[i] if i < len(LABELS) else f"class_{i}"
                if include_re and not include_re.search(name): continue
                if exclude_re and exclude_re.search(name): continue
                if p >= args.min_prob: items.append((name, float(p)))

            if items:
                items.sort(key=lambda x: x[1], reverse=True)
                items = items[:args.topk]
            elif args.force_topk:
                idx = np.argsort(probs)[-args.topk:][::-1]
                items = [(LABELS[i] if i < len(LABELS) else f"class_{i}", float(probs[i])) for i in idx]
            else:
                idx = int(np.argmax(probs))
                nm = LABELS[idx] if idx < len(LABELS) else f"class_{idx}"
                items = [(nm, float(probs[idx]))]

            if time.time() - last >= args.hop - 1e-3:
                lvl = 20*np.log10(level_ema + 1e-6)
                line = " | ".join(f"{n}: {p:.2f}" for n, p in items)
                if use_ansi: clearline(use_ansi); print(f"[level ~ {lvl:5.1f} dB] {line}", end="", flush=True)
                else: print(f"[level ~ {lvl:5.1f} dB] {line}", flush=True)
                last = time.time()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",   type=int, default=None)
    ap.add_argument("--hop",      type=float, default=HOP_SECS)
    ap.add_argument("--topk",     type=int, default=10)
    ap.add_argument("--min_prob", type=float, default=0.05)
    ap.add_argument("--smooth",   type=float, default=0.3)
    ap.add_argument("--include",  type=str, default=None)
    ap.add_argument("--exclude",  type=str, default="")
    ap.add_argument("--force_topk", action="store_true")
    ap.add_argument("--no_ansi", action="store_true")
    args, _ = ap.parse_known_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nStopped.")
