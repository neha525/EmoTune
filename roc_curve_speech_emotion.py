"""
roc_curve_speech_emotion.py
============================
Generates and visualises multi-class ROC Curves (one-vs-rest) for the
Speech Emotion Recognition (SER) model — CNN + Bi-LSTM architecture.

Supports:
  - micro-average ROC
  - macro-average ROC
  - Per-emotion ROC curves with AUC values

Output
------
  Saves  ->  roc_curve_speech_emotion.png   (in the same folder as this script)

Usage
-----
  # Fresh run against RAVDESS dataset:
  python roc_curve_speech_emotion.py --dataset_dir dataset/ravdess

  # Use cached predictions (fast re-plot):
  python roc_curve_speech_emotion.py

  # Force fresh inference (ignore cache):
  python roc_curve_speech_emotion.py --dataset_dir dataset/ravdess --no-cache

  # Limit samples per emotion class:
  python roc_curve_speech_emotion.py --dataset_dir dataset/ravdess --max_per_class 60
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

# ── Environment setup ────────────────────────────────────────────────────────
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

# ── Project paths ─────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
MODELS_DIR  = BASE / "models"
MODEL_PATH  = MODELS_DIR / "ser_model.h5"
LSTM_SHAPE  = MODELS_DIR / "lstm_shape.npy"
CLASS_FILE  = MODELS_DIR / "emotion_classes.npy"
CACHE_FILE  = MODELS_DIR / "roc_speech_emotion_cache.npz"
OUTPUT_PNG  = BASE / "roc_curve_speech_emotion.png"

# ── Emotion classes ───────────────────────────────────────────────────────────
EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise"]

# ── Dataset label maps ────────────────────────────────────────────────────────
RAVDESS_MAP = {
    "01": "neutral", "02": "neutral",
    "03": "happy",   "04": "sad",
    "05": "angry",   "06": "fear",
    "07": "surprise",
}
CREMAD_MAP = {
    "NEU": "neutral", "HAP": "happy", "SAD": "sad",
    "ANG": "angry",   "FEA": "fear",  "DIS": "surprise",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Audio loading
# ══════════════════════════════════════════════════════════════════════════════
def load_audio(file_path: str | Path, sample_rate: int = 22_050):
    """Load a WAV file and return (audio_array, sample_rate).

    Parameters
    ----------
    file_path   : Path to the audio file.
    sample_rate : Target sample rate in Hz. Resamples if necessary.

    Returns
    -------
    audio : 1-D float32 numpy array of audio samples.
    sr    : Actual sample rate used.
    """
    import librosa
    audio, sr = librosa.load(str(file_path), sr=sample_rate, mono=True)
    return audio, sr


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Feature extraction  (mirrors the training pipeline exactly)
# ══════════════════════════════════════════════════════════════════════════════
def extract_mfcc_features(audio: np.ndarray, sr: int = 22_050):
    """Extract CNN and LSTM features from raw audio (matches training pipeline).

    CNN  input : Mel-spectrogram resized to (128, 128, 1)
    LSTM input : 140-dim per-frame feature vector
                 [40 MFCC | 40 Δ-MFCC | 40 ΔΔ-MFCC | 12 Chroma | 7 SpContrast | 1 ZCR]

    Returns
    -------
    cnn_input  : np.ndarray, shape (128, 128, 1)
    lstm_input : np.ndarray, shape (n_frames, 140)
    """
    import librosa
    from PIL import Image

    N_MFCC     = 40
    N_CHROMA   = 12
    N_MELS     = 128
    N_FFT      = 2_048
    HOP_LENGTH = 512
    CNN_H, CNN_W = 128, 128

    # — CNN: Mel spectrogram —
    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img    = Image.fromarray(mel_db).resize((CNN_W, CNN_H), Image.BILINEAR)
    cnn_input = np.array(img, dtype=np.float32)[..., np.newaxis]

    # — LSTM: Per-frame feature stack —
    mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    chroma      = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=N_CHROMA, hop_length=HOP_LENGTH)
    contrast    = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=HOP_LENGTH)
    zcr         = librosa.feature.zero_crossing_rate(y=audio, hop_length=HOP_LENGTH)

    lstm_raw = np.vstack([mfcc, delta_mfcc, delta2_mfcc, chroma, contrast, zcr]).T
    mn, mx   = lstm_raw.min(), lstm_raw.max()
    lstm_input = ((lstm_raw - mn) / (mx - mn + 1e-8)).astype(np.float32)

    return cnn_input, lstm_input


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Model loading
# ══════════════════════════════════════════════════════════════════════════════
def load_model(model_path: str | Path = MODEL_PATH):
    """Load the trained Keras/TensorFlow SER model from disk.

    Parameters
    ----------
    model_path : Path to the .h5 model file.

    Returns
    -------
    model      : Compiled Keras model ready for inference.
    lstm_shape : (T, F) tuple — expected LSTM input dimensions.
    classes    : List of emotion class names (in model output order).
    """
    import tensorflow as tf

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Model not found at: {model_path}\n"
            "Make sure 'models/ser_model.h5' exists."
        )

    print(f"[MODEL] Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    print(f"[MODEL] Loaded successfully.")

    lstm_shape = tuple(np.load(str(LSTM_SHAPE)).astype(int))
    classes    = np.load(str(CLASS_FILE), allow_pickle=True).tolist()
    print(f"[MODEL] LSTM input shape : {lstm_shape}")
    print(f"[MODEL] Emotion classes  : {classes}")

    return model, lstm_shape, classes


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Dataset discovery & label parsing
# ══════════════════════════════════════════════════════════════════════════════
def _label_from_path(path: Path) -> str | None:
    """Parse the emotion label from a WAV filename (RAVDESS / CREMA-D / TESS)."""
    name = path.stem.upper()

    # RAVDESS:  03-01-01-01-01-01-01.wav  → 3rd token is emotion code
    m = re.match(r"^\d{2}-\d{2}-(\d{2})-", name)
    if m:
        return RAVDESS_MAP.get(m.group(1))

    # CREMA-D:  1001_DFA_ANG_XX.wav  → 3rd token
    parts = name.split("_")
    if len(parts) >= 3 and parts[2] in CREMAD_MAP:
        return CREMAD_MAP[parts[2]]

    # TESS / generic:  OAF_angry.wav, OAF_Fear.wav …
    if len(parts) >= 2:
        emo = parts[-1].lower()
        if emo in EMOTIONS:
            return emo

    # Fallback: emotion word anywhere in filename
    for emo in EMOTIONS:
        if emo in name.lower():
            return emo

    return None


def find_dataset_dir() -> Path | None:
    """Auto-detect the dataset directory within the project folder."""
    candidates = [
        "dataset", "data", "ravdess", "tess", "cremad",
        "dataset/ravdess", "dataset/tess", "dataset/cremad",
    ]
    for c in candidates:
        p = BASE / c
        if p.is_dir() and list(p.rglob("*.wav")):
            return p
    return None


def load_test_data(dataset_dir: Path, max_per_class: int = 80):
    """Collect audio file paths and string labels from the dataset directory.

    Parameters
    ----------
    dataset_dir   : Root folder containing .wav files (searched recursively).
    max_per_class : Maximum number of samples to use per emotion class.

    Returns
    -------
    paths  : List[Path]  — audio file paths.
    labels : List[str]   — corresponding emotion labels.
    """
    from collections import defaultdict

    buckets: dict[str, list[Path]] = defaultdict(list)
    for fp in dataset_dir.rglob("*.wav"):
        lbl = _label_from_path(fp)
        if lbl and lbl in EMOTIONS:
            buckets[lbl].append(fp)

    paths, labels = [], []
    for emo in EMOTIONS:
        items = buckets[emo][:max_per_class]
        paths  += items
        labels += [emo] * len(items)

    if not labels:
        raise RuntimeError(
            f"No labelled .wav files found in '{dataset_dir}'.\n"
            "Check the path or filename format (RAVDESS / CREMA-D / TESS expected)."
        )

    dist = {e: labels.count(e) for e in EMOTIONS}
    print(f"[DATA] {len(labels)} samples selected  |  Distribution: {dist}")
    return paths, labels


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Batch prediction
# ══════════════════════════════════════════════════════════════════════════════
def predict_emotions(paths: list, model, lstm_shape: tuple, classes: list) -> np.ndarray:
    """Run the model on every audio file and collect per-class probabilities.

    Parameters
    ----------
    paths      : List of audio file paths.
    model      : Loaded Keras model.
    lstm_shape : (T, F) — LSTM time-steps and features.
    classes    : Emotion class names in model output order.

    Returns
    -------
    y_pred : np.ndarray, shape (n_samples, n_emotions) — predicted probabilities.
    """
    T, F = lstm_shape
    n    = len(paths)
    all_probs = []

    print(f"[PRED] Running inference on {n} audio files ...")
    for i, fp in enumerate(paths, 1):
        if i % 20 == 0 or i == n:
            print(f"  [{i:>4}/{n}] predicting ...", end="\r")
        try:
            audio, sr  = load_audio(fp)
            cnn_in, lstm_in = extract_mfcc_features(audio, sr)

            # Pad / trim LSTM sequence to match training shape
            lstm_in = lstm_in[:T, :F]
            if lstm_in.shape[0] < T:
                pad = np.zeros((T - lstm_in.shape[0], lstm_in.shape[1]), np.float32)
                lstm_in = np.vstack([lstm_in, pad])

            probs = model.predict(
                [cnn_in[np.newaxis], lstm_in[np.newaxis]], verbose=0
            )[0]

            # Re-order to EMOTIONS order (in case model order differs)
            reordered = np.array(
                [probs[classes.index(e)] if e in classes else 0.0 for e in EMOTIONS],
                dtype=np.float32,
            )
            all_probs.append(reordered)

        except Exception as exc:
            print(f"\n  [SKIP] {fp.name}: {exc}")
            # Assign uniform probs for failed files
            all_probs.append(np.full(len(EMOTIONS), 1.0 / len(EMOTIONS), np.float32))

    print()
    return np.array(all_probs)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ROC curve computation
# ══════════════════════════════════════════════════════════════════════════════
def compute_roc_curves(y_true_bin: np.ndarray, y_score: np.ndarray):
    """Compute per-class, micro-average, and macro-average ROC curves.

    Parameters
    ----------
    y_true_bin : One-hot encoded true labels, shape (n_samples, n_classes).
    y_score    : Predicted probabilities,      shape (n_samples, n_classes).

    Returns
    -------
    fpr       : dict  {class_idx → fpr array, 'micro' → micro fpr, 'macro' → macro fpr}
    tpr       : dict  {class_idx → tpr array, 'micro' → micro tpr, 'macro' → macro tpr}
    roc_auc   : dict  {class_idx → AUC,       'micro' → micro AUC, 'macro' → macro AUC}
    """
    from sklearn.metrics import roc_curve, auc

    n_classes = len(EMOTIONS)
    fpr, tpr, roc_auc = {}, {}, {}

    # — Per-class (one-vs-rest) —
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i]        = auc(fpr[i], tpr[i])

    # — Micro-average (treat all classes as one big binary problem) —
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), y_score.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # — Macro-average (average per-class TPR over shared FPR grid) —
    all_fpr  = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"]    = all_fpr
    tpr["macro"]    = mean_tpr
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    return fpr, tpr, roc_auc


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Plotting
# ══════════════════════════════════════════════════════════════════════════════
def plot_multiclass_roc(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    out_path: Path = OUTPUT_PNG,
) -> None:
    """Generate and save the multi-class ROC curve figure.

    Parameters
    ----------
    y_true_bin : One-hot true labels.
    y_score    : Predicted probabilities.
    out_path   : Where to save the PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fpr, tpr, roc_auc = compute_roc_curves(y_true_bin, y_score)

    # ── Colour palette (dark-mode GitHub style) ────────────────────────────
    BG      = "#0d1117"
    PANEL   = "#161b22"
    GRID    = "#21262d"
    TEXT    = "#c9d1d9"
    MUTED   = "#8b949e"
    DIAG    = "#3d444d"

    EMOTION_COLORS = {
        "happy":    "#58a6ff",   # blue
        "sad":      "#79c0ff",   # light blue
        "angry":    "#f78166",   # coral / red
        "neutral":  "#a5d6ff",   # pale blue
        "fear":     "#d2a8ff",   # purple
        "surprise": "#56d364",   # green
    }
    MICRO_C = "#ff7b72"   # warm red
    MACRO_C = "#e3b341"   # gold

    # ── Figure layout ─────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.facecolor":    PANEL,
        "figure.facecolor":  BG,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "grid.color":        GRID,
        "legend.facecolor":  PANEL,
        "legend.edgecolor":  GRID,
        "legend.labelcolor": TEXT,
        "text.color":        TEXT,
    })

    fig, ax = plt.subplots(figsize=(9, 8), facecolor=BG)
    ax.set_facecolor(PANEL)

    # ── Diagonal reference line ────────────────────────────────────────────
    ax.plot([0, 1], [0, 1], color=DIAG, lw=1.4, ls="--",
            label="Random Classifier  (AUC = 0.50)")

    # ── Per-emotion ROC curves ─────────────────────────────────────────────
    for i, emo in enumerate(EMOTIONS):
        color = EMOTION_COLORS.get(emo, "#ffffff")
        label = f"{emo.capitalize():<9}  AUC = {roc_auc[i]:.3f}"
        ax.plot(fpr[i], tpr[i], color=color, lw=2.0, alpha=0.90, label=label)

    # ── Micro-average ──────────────────────────────────────────────────────
    ax.plot(
        fpr["micro"], tpr["micro"],
        color=MICRO_C, lw=2.4, ls=":",
        label=f"Micro-average  AUC = {roc_auc['micro']:.3f}",
    )

    # ── Macro-average (shaded) ─────────────────────────────────────────────
    ax.plot(
        fpr["macro"], tpr["macro"],
        color=MACRO_C, lw=2.6, ls="-.",
        label=f"Macro-average  AUC = {roc_auc['macro']:.3f}",
    )
    ax.fill_between(fpr["macro"], tpr["macro"], alpha=0.07, color=MACRO_C)

    # ── Zoomed inset (upper-left corner) ──────────────────────────────────
    axins = ax.inset_axes([0.10, 0.50, 0.38, 0.38])
    axins.set_facecolor(PANEL)
    axins.plot([0, 1], [0, 1], color=DIAG, lw=1.0, ls="--")
    for i, emo in enumerate(EMOTIONS):
        axins.plot(fpr[i], tpr[i], color=EMOTION_COLORS.get(emo, "#fff"), lw=1.4, alpha=0.85)
    axins.plot(fpr["micro"], tpr["micro"], color=MICRO_C, lw=1.8, ls=":")
    axins.plot(fpr["macro"], tpr["macro"], color=MACRO_C, lw=2.0, ls="-.")
    axins.set_xlim(0.0, 0.25)
    axins.set_ylim(0.70, 1.00)
    axins.tick_params(colors=MUTED, labelsize=7)
    axins.set_title("Zoom: low FPR region", color=MUTED, fontsize=7.5, pad=3)
    for spine in axins.spines.values():
        spine.set_edgecolor(GRID)
    ax.indicate_inset_zoom(axins, edgecolor=GRID)

    # ── Axes labels & title ────────────────────────────────────────────────
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title(
        "ROC Curve — Speech Emotion Recognition\n"
        "CNN + Bi-LSTM  |  One-vs-Rest  |  RAVDESS Dataset",
        fontsize=14, pad=14,
    )
    ax.grid(True, lw=0.6, alpha=0.5)

    # ── Legend ────────────────────────────────────────────────────────────
    ax.legend(
        loc="lower right", fontsize=9.5,
        framealpha=0.75, prop={"family": "monospace"},
    )

    # ── Footer annotation ─────────────────────────────────────────────────
    footer = (
        f"micro AUC: {roc_auc['micro']:.4f}  |  "
        f"macro AUC: {roc_auc['macro']:.4f}  |  "
        f"Emotions: {', '.join(EMOTIONS)}"
    )
    fig.text(0.5, 0.005, footer, ha="center", fontsize=8.5, color=MUTED)

    plt.tight_layout(rect=[0, 0.025, 1, 1])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"[DONE] ROC curve saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Main entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-class ROC curves for the SER model."
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=None,
        help="Path to the dataset root folder (searched recursively for .wav files).",
    )
    parser.add_argument(
        "--max_per_class", type=int, default=80,
        help="Maximum number of audio samples to use per emotion class (default: 80).",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force re-run inference even if a cache file already exists.",
    )
    parser.add_argument(
        "--model_path", type=str, default=str(MODEL_PATH),
        help="Path to the .h5 model file (default: models/ser_model.h5).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ROC Curve — Speech Emotion Recognition")
    print("  CNN + Bi-LSTM Architecture  |  RAVDESS / CREMA-D / TESS")
    print("=" * 60)

    # ── Step 1: Try cache ──────────────────────────────────────────────────
    if not args.no_cache and CACHE_FILE.exists():
        print(f"\n[CACHE] Loading cached predictions from:\n        {CACHE_FILE}")
        cached     = np.load(str(CACHE_FILE))
        y_score    = cached["y_score"]
        y_true_bin = cached["y_true_bin"]
        print(f"[CACHE] {y_score.shape[0]} samples loaded from cache.")

    else:
        # ── Step 2: Locate dataset ─────────────────────────────────────────
        ds_dir = Path(args.dataset_dir) if args.dataset_dir else find_dataset_dir()
        if ds_dir is None or not ds_dir.exists():
            print(
                "\n[ERROR] Dataset directory not found.\n"
                "        Provide it with:  --dataset_dir <path/to/dataset>",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"\n[DATA] Dataset directory: {ds_dir}")

        # ── Step 3: Load test data ─────────────────────────────────────────
        paths, labels = load_test_data(ds_dir, max_per_class=args.max_per_class)

        # ── Step 4: Load model ─────────────────────────────────────────────
        model, lstm_shape, classes = load_model(args.model_path)

        # ── Step 5: Predict emotions ───────────────────────────────────────
        y_score = predict_emotions(paths, model, lstm_shape, classes)

        # ── Binarise labels (one-hot) ──────────────────────────────────────
        from sklearn.preprocessing import label_binarize
        emo_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
        y_int      = np.array([emo_to_idx[l] for l in labels])
        y_true_bin = label_binarize(y_int, classes=list(range(len(EMOTIONS))))

        # ── Save cache for fast re-runs ────────────────────────────────────
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(CACHE_FILE), y_score=y_score, y_true_bin=y_true_bin)
        print(f"[CACHE] Predictions cached → {CACHE_FILE}")

    # ── Step 6: Compute & plot ROC curves ─────────────────────────────────
    print(f"\n[PLOT] Generating ROC curve ...")
    plot_multiclass_roc(y_true_bin, y_score, out_path=OUTPUT_PNG)

    print("\n[SUMMARY] AUC scores:")
    _, _, roc_auc = compute_roc_curves(y_true_bin, y_score)
    for i, emo in enumerate(EMOTIONS):
        bar = "█" * int(roc_auc[i] * 20)
        print(f"  {emo:<9} AUC = {roc_auc[i]:.4f}  {bar}")
    print(f"  {'micro':<9} AUC = {roc_auc['micro']:.4f}")
    print(f"  {'macro':<9} AUC = {roc_auc['macro']:.4f}")
    print(f"\n[OUTPUT] Saved → {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
