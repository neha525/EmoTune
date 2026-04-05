"""Quick smoke-test: POST _test_audio.wav to /predict and print results."""
import requests, json

with open("_test_audio.wav", "rb") as f:
    r = requests.post(
        "http://127.0.0.1:5000/predict",
        files={"audio": ("test.wav", f, "audio/wav")},
    )

print("Status :", r.status_code)
if r.status_code != 200:
    print("Error  :", r.text)
else:
    data = r.json()
    print("Emotion:", data.get("emotion"))
    print("Conf   :", data.get("confidence"))
    print("Probs  :", json.dumps(data.get("probs", {}), indent=2))
    songs = data.get("songs", [])
    print(f"Songs ({len(songs)}):")
    for s in songs[:5]:
        print(f"  - {s['title']} by {s['artist']}")
