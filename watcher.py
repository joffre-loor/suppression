# watcher.py
import time, requests, os

API = os.getenv("SUPPRESS_API_BASE", "http://127.0.0.1:8000")
last_version = -1
print(f"[watcher] polling {API}/suppress/current")

try:
    while True:
        try:
            r = requests.get(f"{API}/suppress/current", timeout=2)
            r.raise_for_status()
            d = r.json()
            v = d.get("version")
            if v != last_version:
                last_version = v
                print(f"[control] v={v} mode={d.get('mode')} classes={d.get('classes')} profile={d.get('profile')}")
        except Exception as e:
            print(f"[watcher] error: {e}")
        time.sleep(0.25)
except KeyboardInterrupt:
    print("\n[watcher] bye")
