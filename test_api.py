"""Quick end-to-end test for /predict endpoint."""
import requests, json, sys

wav = "_test_audio.wav"
url = "http://127.0.0.1:5000/predict"

print(f"[TEST] Posting {wav} to {url} ...")
with open(wav, "rb") as f:
    r = requests.post(url, files={"audio": ("audio.wav", f, "audio/wav")}, timeout=120)

print(f"[TEST] HTTP {r.status_code}")
d = r.json()
if "error" in d:
    print("[FAIL] Error:", d["error"])
    sys.exit(1)

print("[PASS] Emotion:", d["emotion"])
print("[PASS] Songs:")
for i, s in enumerate(d.get("songs", []), 1):
    print(f"  {i}. {s['title']} — {s['artist']}")
print("[PASS] Session id:", d.get("session", {}).get("id"))
