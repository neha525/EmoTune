"""
feature_extraction.py
---------------------
Extracts audio features (MFCC, delta-MFCC, delta-delta-MFCC, Chroma,
Spectral Contrast, ZCR, Mel Spectrogram) and prepares them for both
the CNN (2-D image) and Bi-LSTM (sequence) sub-networks.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Force UTF-8 output so emoji display correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from typing import Tuple

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 22_050
N_MFCC: int = 40
N_CHROMA: int = 12
N_MELS: int = 128
N_FFT: int = 1_024        # 1024 is faster than 2048 and sufficient for inference
HOP_LENGTH: int = 512
MAX_AUDIO_SECONDS: int = 8  # cap long recordings — silence beyond this rarely helps

# Desired spatial dimensions for the CNN input (height × width)
CNN_HEIGHT: int = 128
CNN_WIDTH: int = 128


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def extract_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract CNN input (2-D spectrogram) and LSTM input (sequence) from audio.

    Parameters
    ----------
    audio : 1-D float32 array of audio samples.
    sr    : Sample rate in Hz.

    Returns
    -------
    cnn_input  : shape (CNN_HEIGHT, CNN_WIDTH, 1) — normalised Mel spectrogram.
    lstm_input : shape (n_frames, n_features)     — per-frame feature vector.

    Feature stack (per frame)
    -------------------------
    - 40 MFCCs
    - 40 delta-MFCCs  (first-order dynamics)
    - 40 delta²-MFCCs (second-order dynamics / acceleration)
    - 12 Chroma
    -  7 Spectral Contrast
    -  1 Zero Crossing Rate
    Total: 140 features per frame
    """
    # ── 1. Pre-processing: trim silence + cap length ──────────────────────────
    audio, _ = librosa.effects.trim(audio, top_db=20)          # strip leading/trailing silence
    max_samples = MAX_AUDIO_SECONDS * sr
    if len(audio) > max_samples:
        audio = audio[:max_samples]                             # cap to avoid slow long files

    # ── 2. Compute STFT once — reuse across all features ─────────────────────
    # All librosa spectral features internally compute STFT. By passing the
    # pre-computed magnitude *D*, we avoid repeating the FFT ~5 extra times.
    D = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))

    # ── 3. Mel Spectrogram (CNN input) ───────────────────────────────────────
    mel_basis = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS)
    mel       = np.dot(mel_basis, D ** 2)                      # power spectrogram → mel
    mel_db    = librosa.power_to_db(mel, ref=np.max)
    cnn_input = _resize_spectrogram(mel_db, CNN_HEIGHT, CNN_WIDTH)
    cnn_input = cnn_input[..., np.newaxis]                      # add channel dim

    # ── 4. Per-frame features (LSTM input) — all reuse D ─────────────────────
    mfcc        = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
    delta_mfcc  = librosa.feature.delta(mfcc)                  # 1st-order dynamics
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)         # 2nd-order (acceleration)
    chroma      = librosa.feature.chroma_stft(S=D, sr=sr, n_chroma=N_CHROMA)
    contrast    = librosa.feature.spectral_contrast(S=D, sr=sr)
    zcr         = librosa.feature.zero_crossing_rate(y=audio, hop_length=HOP_LENGTH)

    # Stack along feature axis  → (n_features, n_frames) then transpose
    lstm_input = np.vstack([mfcc, delta_mfcc, delta2_mfcc, chroma, contrast, zcr]).T
    lstm_input = _normalise_2d(lstm_input)

    return cnn_input.astype(np.float32), lstm_input.astype(np.float32)


def extract_features_from_file(
    file_path: str | Path,
    sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: load a WAV file then call :func:`extract_features`."""
    audio, sr = librosa.load(str(file_path), sr=sr, mono=True)
    return extract_features(audio, sr)


def get_flat_feature_vector(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a single, flat feature vector suitable for classical ML or
    cosine-similarity comparisons.

    The vector is the column-wise mean of the per-frame feature matrix
    produced by :func:`extract_features`.
    """
    _, lstm_input = extract_features(audio, sr)
    return lstm_input.mean(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────
def visualize_mfcc(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    save_path: str | Path = "mfcc_plot.png",
) -> None:
    """Plot and save the MFCC spectrogram for *audio*.

    Parameters
    ----------
    audio     : 1-D float32 audio samples.
    sr        : Sample rate.
    save_path : File path for the output PNG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH),
        ref=np.max,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor="#1a1a2e")
    fig.suptitle("Feature Visualisation", color="white", fontsize=16, fontweight="bold")

    # — Mel Spectrogram —
    ax1 = axes[0]
    img1 = librosa.display.specshow(
        mel_db, sr=sr, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax1, cmap="magma",
    )
    ax1.set_title("Mel Spectrogram (dB)", color="white", fontsize=13)
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="white")
    ax1.xaxis.label.set_color("white")
    ax1.yaxis.label.set_color("white")
    fig.colorbar(img1, ax=ax1, format="%+2.0f dB").ax.yaxis.set_tick_params(color="white")

    # — MFCC —
    ax2 = axes[1]
    img2 = librosa.display.specshow(
        mfcc, sr=sr, hop_length=HOP_LENGTH,
        x_axis="time", ax=ax2, cmap="coolwarm",
    )
    ax2.set_title(f"MFCCs ({N_MFCC} coefficients)", color="white", fontsize=13)
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="white")
    ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white")
    fig.colorbar(img2, ax=ax2).ax.yaxis.set_tick_params(color="white")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PLOT] MFCC plot saved to: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────
def _resize_spectrogram(spec: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a 2-D spectrogram to (height, width) using bilinear interpolation."""
    from PIL import Image  # lazy import — optional dep
    img = Image.fromarray(spec).resize((width, height), Image.BILINEAR)
    return np.array(img)


def _normalise_2d(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise a 2-D array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# ──────────────────────────────────────────────────────────────────────────────
# CLI helper
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import librosa

    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <path/to/audio.wav>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading: {path}")
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    cnn_in, lstm_in = extract_features(audio, sr)
    print(f"CNN input shape : {cnn_in.shape}")
    print(f"LSTM input shape: {lstm_in.shape}")
    visualize_mfcc(audio, sr, save_path="mfcc_plot.png")
