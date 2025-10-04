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

FAMILIES = [
    ("speech",      r"\bspeech\b|conversation|monologue|speaker|narration|chatter"),
    ("music",       r"\bmusic\b|song|singing|instrument|piano|guitar|drum|violin|sax|bass|cello|orchestra|choir"),
    ("keyboard",    r"\bkeyboard\b|typing"),
    ("mouse",       r"\bmouse\b|click|double[- ]?click|scroll wheel"),
    ("clicks",      r"\bclick\b|tap|clack|clink|ping|snap|tick|tock|chop|typewriter"),
    ("hvac",        r"fan|hvac|air conditioning|appliance|vent|ac\b|hum|blower"),
    ("traffic",     r"traffic|vehicle|car\b|bus|truck|motor|road|highway|intersection"),
    ("train",       r"\btrain\b|rail|subway|metro|tram"),
    ("aircraft",    r"airplane|aircraft|jet|helicopter|airport|propeller"),
    ("bicycle",     r"bicycle|bike|pedal|chain|bell"),
    ("siren",       r"siren|ambulance|police|fire engine"),
    ("horn",        r"car horn|horn honk|honk"),
    ("alarm",       r"\balarm\b|buzzer|beep|timer|smoke alarm"),
    ("door",        r"door|knock|slam|doorbell"),
    ("dog",         r"dog|bark|woof|yap|growl"),
    ("cat",         r"cat|meow|purr|hiss"),
    ("baby",        r"baby|infant|cry|whimper|coo"),
    ("laughter",    r"laugh|laughter|giggle|chuckle"),
    ("cough_sneeze",r"cough|sneeze|sniffle|throat clear|hiccup"),
    ("footsteps",   r"footstep|walking|running|jog|stomp|heel"),
    ("crowd",       r"crowd|applause|clap|cheer|audience"),
    ("kitchen",     r"cooking|frying|sizzle|boil|kettle|microwave|cutlery|dishes|plate|glass"),
    ("water",       r"water|sink|tap|shower|rain|drip|stream|fountain|flush"),
    ("weather",     r"wind|storm|thunder|rainstorm|hail"),
    ("construction",r"construction|hammer|saw|drill|jackhammer|nail gun|sandpaper"),
    ("office",      r"printer|scanner|photocopier|fax|stapler|paper rustle"),
    ("camera",      r"camera|shutter|single-lens reflex|dslr|snapshot"),
    ("phone",       r"phone|ringtone|notification|text tone|vibrate"),
    ("sports",      r"basketball bounce|tennis|soccer|whistle|skateboard"),
]

class EMA:
    def __init__(self, alpha=0.3):
        self.alpha = float(alpha)
        self.v = None
    def update(self, x: np.ndarray) -> np.ndarray:
        if self.v is None: self.v = x
        else: self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v

ring = deque(maxlen=FRAME_LEN)
q = queue.Queue()
level_ema = 0.0

def meter_update(x: np.ndarray):
    global level_ema
    rms = np.sqrt(np.mean(np.square(x))) + 1e-12
    level_ema = 0.9 * level_ema + 0.1 * rms

def audio_cb(indata, frames, time_info, status):
    if status: print(status, file=sys.stderr)
    q.put(indata[:, 0].astype(np.float32, copy=True))

def clearline(use_ansi=True):
    if use_ansi:
        sys.stdout.write("\x1b[2K\r")
        sys.stdout.flush()

def build_family_map():
    comp = [(name, re.compile(rx, re.I)) for name, rx in FAMILIES]
    idx_to_fam = [[] for _ in range(len(LABELS))]
    for i, lab in enumerate(LABELS):
        for fam_idx, (_, cre) in enumerate(comp):
            if cre.search(lab):
                idx_to_fam[i].append(fam_idx)
    return comp, idx_to_fam

def main(args):
    if args.device is None:
        d = sd.default.device
        input_idx = d[0] if isinstance(d, (list, tuple)) else d
    else:
        input_idx = args.device

    devinfo = sd.query_devices(input_idx, 'input')
    native_sr = int(devinfo.get("default_samplerate") or 48000)

    up, down = TARGET_SR, native_sr
    g = gcd(up, down); up //= g; down //= g
    hop_native = max(256, int(args.hop * native_sr))

    ema = EMA(alpha=args.smooth) if args.smooth > 0 else None
    include_re = re.compile(args.include, re.I) if args.include else None
    exclude_re = re.compile(args.exclude, re.I) if args.exclude else None
    use_ansi = not args.no_ansi

    if args.half_life <= 0: args.half_life = 30.0
    decay = 1.0 - 2.0**(-args.hop / float(args.half_life))
    hist = np.zeros(len(LABELS), dtype=np.float32)

    fam_comp, idx_to_fam = build_family_map()
    fam_hist = np.zeros(len(fam_comp), dtype=np.float32)

    print(f"Using input device #{input_idx}: {devinfo['name']}")
    print(f"Native SR {native_sr} Hz → resample → {TARGET_SR} Hz")
    print(f"overall view = {'families' if args.family else 'labels'} | half_life={args.half_life:.1f}s | hop={args.hop:.2f}s | topk={args.topk}")
    print("(Ctrl+C to stop)\n")

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
            resampled = resample_poly(buf, up, down).astype(np.float32)
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

            window = np.array(ring, dtype=np.float32)
            scores, _, _ = yamnet(window)
            avg = scores.numpy().mean(axis=0)
            if ema: avg = ema.update(avg)

            hist = (1.0 - decay) * hist + decay * avg

            if args.family:
                fam_hist *= (1.0 - decay)
                for i, p in enumerate(avg):
                    if p <= 0: continue
                    fam_idxs = idx_to_fam[i]
                    if not fam_idxs: continue
                    inc = decay * p / max(1, len(fam_idxs))
                    for fi in fam_idxs:
                        fam_hist[fi] += inc
                items = []
                for fi, p in enumerate(fam_hist):
                    name = fam_comp[fi][0]
                    if include_re and not include_re.search(name): continue
                    if exclude_re and exclude_re.search(name): continue
                    items.append((name, float(p)))
            else:
                items = []
                for i, p in enumerate(hist):
                    name = LABELS[i]
                    if include_re and not include_re.search(name): continue
                    if exclude_re and exclude_re.search(name): continue
                    items.append((name, float(p)))

            items.sort(key=lambda x: x[1], reverse=True)
            items = items[:args.topk] if len(items) >= args.topk else items

            if time.time() - last >= args.hop - 1e-3:
                lvl_db = 20*np.log10(level_ema + 1e-6)
                line = " | ".join(f"{n}: {p:.2f}" for n, p in items)
                prefix = "overall-fam" if args.family else "overall"
                if use_ansi:
                    clearline(use_ansi); print(f"[level ~ {lvl_db:5.1f} dB] {prefix} → {line}", end="", flush=True)
                else:
                    print(f"[level ~ {lvl_db:5.1f} dB] {prefix} → {line}", flush=True)
                last = time.time()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",   type=int, default=None)
    ap.add_argument("--hop",      type=float, default=HOP_SECS)
    ap.add_argument("--topk",     type=int, default=10)
    ap.add_argument("--smooth",   type=float, default=0.3)
    ap.add_argument("--include",  type=str, default=None)
    ap.add_argument("--exclude",  type=str, default="")
    ap.add_argument("--no_ansi", action="store_true")
    ap.add_argument("--half_life", type=float, default=30.0)
    ap.add_argument("--family", action="store_true")
    args, _ = ap.parse_known_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nStopped.")
