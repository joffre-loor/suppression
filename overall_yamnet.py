# overall_yamnet_min.py
import argparse, queue, sys, time, re, os
from collections import deque
from math import gcd
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
        return [f"class_{i}" for i in range(521)]
LABELS = load_labels()
NUM_CLASSES = len(LABELS)

class EMA:
    def __init__(self, alpha=0.6):
        self.alpha = float(alpha)
        self.v = None
    def update(self, x: np.ndarray) -> np.ndarray:
        if self.v is None: self.v = x
        else: self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v

q = queue.Queue()
def audio_cb(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata[:, 0].astype(np.float32, copy=True))

def main(args):
    # device
    if args.device is None:
        d = sd.default.device
        input_idx = d[0] if isinstance(d, (list, tuple)) else d
    else:
        input_idx = args.device
    devinfo = sd.query_devices(input_idx, 'input')
    native_sr = int(devinfo.get("default_samplerate") or 48000)

    # resample ratio native -> TARGET_SR
    up, down = TARGET_SR, native_sr
    g = gcd(up, down); up //= g; down //= g
    hop_native = max(256, int(args.hop * native_sr))

    ema = EMA(alpha=args.smooth) if args.smooth > 0 else None
    include_re = re.compile(args.include, re.I) if args.include else None
    exclude_re = re.compile(args.exclude, re.I) if args.exclude else None

    # recency-weighted accumulator (decaying “so-far”)
    cum_scores = np.zeros(NUM_CLASSES, dtype=np.float32)
    last_seen  = np.full(NUM_CLASSES, -np.inf, dtype=np.float32)
    last_accum_ts = time.time()

    def decay_cumulative(now_ts):
        nonlocal last_accum_ts, cum_scores
        dt = max(0.0, now_ts - last_accum_ts)
        if dt > 0 and args.half_life > 0:
            decay = 0.5 ** (dt / args.half_life)
            cum_scores *= decay
        last_accum_ts = now_ts

    print(f"Using input device #{input_idx}: {devinfo['name']}")
    print(f"Native SR {native_sr} Hz → resample → {TARGET_SR} Hz")
    print(f"hop={args.hop:.2f}s  topk={args.topk}  half_life={args.half_life:.1f}s")
    print("(Ctrl+C to stop)\n")

    _ = yamnet(np.zeros(FRAME_LEN, dtype=np.float32))  # warmup

    ring = deque(maxlen=FRAME_LEN)
    last_print_line = None
    last_print_time = 0.0
    started_at = time.time()

    try:
        with sd.InputStream(device=input_idx,
                            samplerate=native_sr,
                            channels=1,
                            blocksize=hop_native,
                            dtype="float32",
                            callback=audio_cb,
                            latency="low"):
            while True:
                buf = q.get()
                resampled = resample_poly(buf, up, down).astype(np.float32)
                ring.extend(resampled)
                now = time.time()
                if len(ring) < FRAME_LEN:
                    continue

                # model
                window = np.array(ring, dtype=np.float32)
                scores, _, _ = yamnet(window)
                avg = scores.numpy().mean(axis=0)
                if ema: avg = ema.update(avg)

                # include/exclude + thresholds for contribution
                mask = np.ones(NUM_CLASSES, dtype=bool)
                if include_re:
                    mask[:] = False
                    for i, n in enumerate(LABELS):
                        if include_re.search(n): mask[i] = True
                if exclude_re:
                    for i, n in enumerate(LABELS):
                        if exclude_re.search(n): mask[i] = False
                contrib = np.where(mask, avg, 0.0).astype(np.float32)
                if args.min_prob > 0:
                    contrib = np.where(contrib >= args.min_prob, contrib, 0.0).astype(np.float32)

                # optional per-hop Speech cap before accumulation
                try:
                    idx_speech = LABELS.index("Speech")
                except ValueError:
                    idx_speech = None
                if idx_speech is not None:
                    if args.cap_when_others:
                        if contrib[idx_speech] > args.speech_max_share and np.any(np.delete(contrib, idx_speech) > 0):
                            contrib[idx_speech] = args.speech_max_share
                    else:
                        contrib[idx_speech] = min(contrib[idx_speech], args.speech_max_share)

                # accumulate with decay
                decay_cumulative(now)
                if args.accum_min_prob > 0:
                    contrib = np.where(contrib >= args.accum_min_prob, contrib, 0.0).astype(np.float32)
                cum_scores += contrib
                nz = np.where(contrib > 0)[0]
                if nz.size: last_seen[nz] = now

                # build CURRENT “so-far (recency)” list
                fresh_mask = (now - last_seen) <= args.stale_seconds
                eligible = np.where(fresh_mask & (cum_scores > 0))[0]
                if eligible.size == 0:
                    # if nothing fresh, allow single best so it’s not blank
                    if np.max(cum_scores) > 0:
                        eligible = np.array([int(np.argmax(cum_scores))])
                    else:
                        eligible = np.array([], dtype=int)

                totalsum = float(np.sum(cum_scores[eligible])) or 1.0
                pairs = [(i, LABELS[i], float(cum_scores[i] / totalsum)) for i in eligible]

                if args.sofar_min_share > 0:
                    pairs = [p for p in pairs if p[2] >= args.sofar_min_share]
                    if not pairs and eligible.size:
                        i0 = int(eligible[np.argmax(cum_scores[eligible])])
                        pairs = [(i0, LABELS[i0], float(cum_scores[i0] / totalsum))]

                pairs.sort(key=lambda x: x[2], reverse=True)
                pairs = pairs[:args.topk]

                # simplified one-line text (rounded shares)
                line_struct = tuple((n, round(s, 2)) for _, n, s in pairs)
                # only print if changed vs last line (labels or any share change > eps)
                changed = False
                if last_print_line is None or len(line_struct) != len(last_print_line):
                    changed = True
                else:
                    for (n1, s1), (n2, s2) in zip(line_struct, last_print_line):
                        if n1 != n2 or abs(s1 - s2) > args.change_eps:
                            changed = True
                            break

                if changed:
                    text = " | ".join(f"{n}: {s:.2f}" for n, s in line_struct) if line_struct else "(silence)"
                    print(text, flush=True)
                    last_print_line = line_struct
                    last_print_time = now

    except KeyboardInterrupt:
        pass
    finally:
        # final summary (top-k on decayed cum_scores, no staleness)
        duration = time.time() - started_at
        top_idx = np.argsort(-cum_scores)[:args.topk]
        top = [(LABELS[i], float(cum_scores[i])) for i in top_idx if cum_scores[i] > 0]
        tot = sum(v for _, v in top) or 1.0
        print("\n===== summary =====")
        print(f"duration: {duration:.1f}s  half_life: {args.half_life:.1f}s  hop: {args.hop:.2f}s")
        for name, v in top:
            print(f"{name:24s} share={v/tot:.2f} score={v:.3f}")
        print("===================")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",   type=int, default=None)
    ap.add_argument("--hop",      type=float, default=HOP_SECS)
    ap.add_argument("--topk",     type=int, default=5)
    ap.add_argument("--smooth",   type=float, default=0.30)
    ap.add_argument("--min_prob", type=float, default=0.01)
    ap.add_argument("--include",  type=str, default=None)
    ap.add_argument("--exclude",  type=str, default=None)

    # recency accumulator
    ap.add_argument("--half_life", type=float, default=30.0)
    ap.add_argument("--accum_min_prob", type=float, default=0.05)

    # staleness & visibility
    ap.add_argument("--stale_seconds",  type=float, default=45.0)
    ap.add_argument("--sofar_min_share", type=float, default=0.02)

    # speech capping
    ap.add_argument("--speech_max_share", type=float, default=0.60)
    ap.add_argument("--cap_when_others", action="store_true")

    # change detection
    ap.add_argument("--change_eps", type=float, default=0.01,
                    help="Reprint only if any displayed share changes by > this amount or labels change")

    args, _ = ap.parse_known_args()
    main(args)
