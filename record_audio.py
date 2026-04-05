"""
record_audio.py
---------------
Records audio from the default microphone and applies signal-level
standardisation (mono conversion, noise trimming, amplitude normalisation).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Force UTF-8 output so emoji display correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.util


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 22_050          # Hz
DURATION: int = 5                  # seconds
CHANNELS: int = 1                  # mono recording
OUTPUT_DIR: Path = Path("recordings")


# ──────────────────────────────────────────────────────────────────────────────
# Recording
# ──────────────────────────────────────────────────────────────────────────────
def record_audio(
    duration: int = DURATION,
    sample_rate: int = SAMPLE_RATE,
    save_path: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    """Record *duration* seconds of microphone audio.

    Parameters
    ----------
    duration:    Recording length in seconds.
    sample_rate: Target sample rate in Hz.
    save_path:   If given, saves the recording as a WAV file to this path.

    Returns
    -------
    (audio, sample_rate)
        ``audio`` is a 1-D float32 NumPy array of normalised samples.
    """
    print(f"\n[REC] Recording voice for {duration} seconds...")
    for t in range(3, 0, -1):
        print(f"   Starting in {t}...")
        time.sleep(1)
    print("   [*] Recording now - please speak!")

    raw: np.ndarray = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    print("   [DONE] Recording complete.\n")

    # Flatten to 1-D if stereo was somehow captured
    audio: np.ndarray = raw.flatten()

    audio = standardise(audio, sample_rate)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(save_path), audio, sample_rate)
        print(f"[SAVED] Recording saved to: {save_path}")

    return audio, sample_rate


def load_audio(
    file_path: str | Path,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Load a WAV/MP3 file from disk and apply standardisation.

    Returns
    -------
    (audio, sample_rate)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio, sr = librosa.load(str(file_path), sr=sample_rate, mono=True)
    audio = standardise(audio, sr)
    return audio, sr


# ──────────────────────────────────────────────────────────────────────────────
# Signal-Level Standardisation
# ──────────────────────────────────────────────────────────────────────────────
def standardise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply signal-level standardisation.

    Steps
    -----
    1. Convert to float32.
    2. Trim leading/trailing silences.
    3. Normalise amplitude to [-1, 1].

    Parameters
    ----------
    audio:       Raw 1-D audio samples.
    sample_rate: Sample rate (Hz).

    Returns
    -------
    Processed 1-D float32 array.
    """
    # 1. Ensure float32
    audio = audio.astype(np.float32)

    # 2. Trim leading / trailing silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # 3. Normalise amplitude to [-1, 1] (avoid div-by-zero on silent clips)
    audio = librosa.util.normalize(audio)

    return audio


# ──────────────────────────────────────────────────────────────────────────────
# CLI helper
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio, sr = record_audio(save_path=OUTPUT_DIR / "recording.wav")
    print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
