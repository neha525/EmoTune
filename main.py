"""
main.py
-------
End-to-end Speech Emotion Recognition & Music Recommendation pipeline.

Usage examples
--------------
  # Live microphone + local dataset (no API key needed)
  python main.py

  # Load an existing WAV instead of recording
  python main.py --no-mic --wav-path recordings/recording.wav

  # Use Spotify API (set env vars SPOTIPY_CLIENT_ID + SPOTIPY_CLIENT_SECRET)
  python main.py --spotify

  # Local mode only (skip Spotify)
  python main.py --local

  # Skip MFCC plot
  python main.py --no-plot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Force UTF-8 output so emoji/Unicode display correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────────────────────
BANNER = r"""
+============================================================+
|   Speech Emotion Recognition & Music Recommendation       |
|   Clustering-Based Dual Network  (CNN + Bi-LSTM)          |
+============================================================+
"""


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(args: argparse.Namespace) -> None:
    print(BANNER)

    # ── Step 1: Obtain audio ─────────────────────────────────────────────────
    if args.no_mic:
        wav_path = Path(args.wav_path)
        if not wav_path.exists():
            print(f"[ERROR] File not found: {wav_path}")
            sys.exit(1)
        print(f"[LOAD] Loading audio from: {wav_path}")
        from record_audio import load_audio
        audio, sr = load_audio(wav_path)
    else:
        from record_audio import record_audio
        save_path = Path("recordings/recording.wav")
        audio, sr = record_audio(save_path=save_path)

    print(f"   Audio shape: {audio.shape}, Sample rate: {sr} Hz")

    # ── Step 2: Feature extraction ────────────────────────────────────────────
    print("\n[EXTRACT] Extracting features...")
    from feature_extraction import extract_features, visualize_mfcc
    cnn_input, lstm_input = extract_features(audio, sr)
    print(f"   CNN input  shape : {cnn_input.shape}")
    print(f"   LSTM input shape : {lstm_input.shape}")

    # ── Step 3: Emotion prediction ────────────────────────────────────────────
    print("\n[PREDICT] Predicting emotion...")
    from emotion_predictor import predict_emotion, EmotionPredictor
    label, probs = predict_emotion(audio=audio, sr=sr)

    print(f"\n{'─'*52}")
    print(f"   🎭  Emotion detected : {label.upper()}")
    print(f"{'─'*52}")
    print("\n   Probability breakdown:")
    EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise"]
    for emo, p in zip(EMOTIONS, probs):
        bar = "█" * int(p * 30)
        print(f"   {emo:>10s}: {p:.3f}  {bar}")

    # ── Step 4: Visualisation ─────────────────────────────────────────────────
    if not args.no_plot:
        plot_path = Path("mfcc_plot.png")
        visualize_mfcc(audio, sr, save_path=plot_path)

    # ── Step 5: Music recommendation ─────────────────────────────────────────
    print("\n[RECOMMEND] Fetching music recommendations...")
    from music_recommendation import recommend_music, print_recommendations

    if args.spotify:
        mode = "spotify"
    elif args.local:
        mode = "local"
    else:
        mode = "auto"   # try Spotify, fall back to local

    songs = recommend_music(label, probs, mode=mode)
    print_recommendations(songs, label)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n✅  Pipeline complete!")
    if not args.no_plot:
        print(f"   MFCC plot saved to: mfcc_plot.png")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition & Music Recommendation"
    )
    # Audio source
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--no-mic",
        action="store_true",
        help="Load audio from --wav-path instead of recording from microphone",
    )
    parser.add_argument(
        "--wav-path",
        type=str,
        default="recordings/recording.wav",
        help="Path to a WAV file (used with --no-mic)",
    )

    # Music source
    msc = parser.add_mutually_exclusive_group()
    msc.add_argument("--spotify", action="store_true", help="Use Spotify API")
    msc.add_argument("--local",   action="store_true", help="Use local songs.csv")

    # Misc
    parser.add_argument("--no-plot", action="store_true", help="Skip MFCC visualisation")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)

