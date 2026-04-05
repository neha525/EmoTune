# Speech Emotion Recognition & Music Recommendation System

A Python-based system that records your voice, detects your emotion using a **Clustering-Based Dual Network (CNN + Bi-LSTM)**, and recommends songs from Spotify or a local dataset.

## Quick Start -- Web UI

```powershell
cd speech_emotion_music

# Double-click start.bat  OR run:
$env:PYTHONUTF8=1; $env:TF_ENABLE_ONEDNN_OPTS=0; python app.py
```

Open **http://127.0.0.1:5000/** in your browser.

- **Upload Audio** -- drag-and-drop or click *Upload Audio* to pick any `.wav/.mp3/.ogg/.m4a/.flac` file, then click *Analyse Emotion*
- **Record** -- click *Record*, speak into your microphone, click *Stop* (or it auto-stops) -- analysis starts automatically
- History of all sessions appears in the left sidebar

> **Microphone note:** Click *Record* in your browser and allow the microphone permission prompt. Works on Chrome, Edge, and Firefox at `http://127.0.0.1:5000/` (localhost is treated as a secure context).

## Spotify Setup (Optional)

Without Spotify credentials the app uses the local `songs.csv` database (86 curated tracks). With credentials it also fetches **album art, 30-second previews, and popularity scores**.

**Step 1** — Create a free Spotify Developer app:
1. Go to https://developer.spotify.com/dashboard and log in
2. Click **Create App** → give it any name/description
3. Set the Redirect URI to `http://localhost:8888/callback` (required even though it is unused)
4. Copy the **Client ID** and **Client Secret**

**Step 2** — Set credentials before launching:

```powershell
# PowerShell (temporary — current session only)
$env:SPOTIPY_CLIENT_ID     = "paste_your_client_id_here"
$env:SPOTIPY_CLIENT_SECRET = "paste_your_client_secret_here"
$env:PYTHONUTF8=1; $env:TF_ENABLE_ONEDNN_OPTS=0; python app.py
```

Or add them to a **`.env` file** in the project root (auto-sourced on Windows with `python-dotenv`):
```
SPOTIPY_CLIENT_ID=paste_your_client_id_here
SPOTIPY_CLIENT_SECRET=paste_your_client_secret_here
```

When credentials are set, song cards in the web UI show album art and a ▶ Spotify link.

## Project Structure

```
speech_emotion_music/
+-- app.py                  -- Flask web server
+-- session_logger.py       -- Session history logger
+-- start.bat               -- One-click launcher
+-- templates/index.html    -- Web UI
+-- static/style.css        -- UI styles
+-- static/app.js           -- UI logic
+-- dataset/
|   +-- songs.csv           -- 86 curated songs (6 emotions)
+-- models/
|   +-- ser_model.h5        -- Trained model
+-- feature_extraction.py
+-- train_model.py
+-- emotion_predictor.py
+-- music_recommendation.py
+-- record_audio.py
+-- main.py
+-- requirements.txt
```

## Setup

```
Microphone Input
      ↓
Signal Standardisation (trim silence, normalize amplitude)
      ↓
Feature Extraction (MFCC, Chroma, Spectral Contrast, ZCR, Mel Spectrogram)
      ↓
┌─────────────────────┐    ┌────────────────────────┐
│  CNN Sub-Network    │    │  Bi-LSTM Sub-Network   │
│ (spatial patterns)  │    │ (temporal patterns)     │
└─────────┬───────────┘    └──────────┬─────────────┘
          └──────────── Fusion ───────┘
                           ↓
               Softmax Emotion Classifier
                    (6 emotions)
                           ↓
              Cosine Similarity Matching
                           ↓
              Top-5 Music Recommendations
```

## Project Structure

```
speech_emotion_music/
├── dataset/
│   ├── README.md        ← Dataset download instructions
│   └── songs.csv        ← Local song database (50 songs, 6 emotions)
├── models/
│   ├── README.md        ← What gets saved here
│   └── ser_model.h5     ← (generated after training)
├── recordings/          ← (auto-created on first run)
├── feature_extraction.py
├── train_model.py
├── emotion_predictor.py
├── music_recommendation.py
├── record_audio.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **Windows users**: If `sounddevice` fails, install PortAudio first:
> ```bash
> pip install pipwin && pipwin install pyaudio
> ```

### 2. (Optional) Install Pillow for spectrogram resizing
```bash
pip install Pillow
```

### 3. (Optional) Set Spotify credentials
If you want live Spotify recommendations, create a free app at
https://developer.spotify.com/dashboard and set:
```powershell
$env:SPOTIPY_CLIENT_ID     = "your_client_id"
$env:SPOTIPY_CLIENT_SECRET = "your_client_secret"
```

## Quick Start (No Training Required)

Run the full pipeline in **demo mode** using the local song dataset:
```bash
python main.py --local
```

Load an existing WAV file instead of using the microphone:
```bash
python main.py --no-mic --wav-path path/to/audio.wav --local
```

## Training the Model

### Step 1: Download a dataset
See [`dataset/README.md`](dataset/README.md) for download links.

### Step 2: Train
```powershell
# Standard (RAVDESS)
python train_model.py --dataset_dir dataset/ravdess --epochs 60

# Recommended — augmentation + attention + label smoothing (best accuracy)
python train_model.py --dataset_dir dataset/ravdess --epochs 60 --augment --attention --label-smoothing 0.1
```
Training saves:
- `models/ser_model.h5` — the trained model
- `models/emotion_classes.npy` — label order
- `models/lstm_shape.npy` — LSTM sequence dimensions
- `models/training_history.png` — accuracy/loss curves

### Step 3: Run with trained model
```bash
python main.py
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--no-mic` | Load WAV from disk instead of recording |
| `--wav-path PATH` | Path to WAV file (default: `recordings/recording.wav`) |
| `--spotify` | Use Spotify API for recommendations |
| `--local` | Use local `songs.csv` (no API key needed) |
| `--no-plot` | Skip MFCC visualisation |

## Module Overview

| File | Responsibility |
|------|---------------|
| `record_audio.py` | Mic recording, mono conversion, amplitude normalisation |
| `feature_extraction.py` | MFCC + delta + delta² (140 features/frame), Chroma, Spectral Contrast, ZCR, Mel Spectrogram |
| `train_model.py` | CNN + Bi-LSTM dual network with temporal attention + label smoothing + cosine LR |
| `emotion_predictor.py` | Load model, run inference, demo heuristic fallback |
| `music_recommendation.py` | Cosine similarity matching + Spotify/local recommendation |
| `main.py` | End-to-end pipeline orchestration |

## Emotions Supported

| Label | Training Code |
|-------|--------------|
| 😊 Happy | `happy` |
| 😢 Sad | `sad` |
| 😡 Angry | `angry` |
| 😐 Neutral | `neutral` |
| 😨 Fear | `fear` |
| 😲 Surprise | `surprise` |

## Example Output

```
╔══════════════════════════════════════════════════════════════╗
║   Speech Emotion Recognition  &  Music Recommendation       ║
╚══════════════════════════════════════════════════════════════╝

🎙️  Recording voice for 5 seconds …
   Starting in 3…  2…  1…
   ● Recording now — please speak!
   ■ Done recording.

🔬  Extracting features …
🧠  Predicting emotion …

────────────────────────────────────────────────────────────
   🎭  Emotion detected : HAPPY
────────────────────────────────────────────────────────────

🎵  Recommended Songs for emotion: HAPPY
────────────────────────────────────────────────────────────
  1. Happy  –  Pharrell Williams
  2. Can't Stop the Feeling  –  Justin Timberlake
  3. Good Time  –  Owl City
  4. Uptown Funk  –  Bruno Mars
  5. Best Day of My Life  –  American Authors
────────────────────────────────────────────────────────────
```

## License

MIT
