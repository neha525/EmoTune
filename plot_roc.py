"""
plot_roc.py
-----------
Generates a multi-class ROC curve (one-vs-rest) for the trained SER model.
Caches predictions to models/roc_cache.npz so re-runs are instant.

Usage
-----
    python plot_roc.py --dataset_dir dataset/ravdess
    python plot_roc.py --dataset_dir dataset/ravdess --no-cache
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

BASE   = Path(__file__).parent
MODELS = BASE / "models"
CACHE  = MODELS / "roc_cache.npz"

EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise"]

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


def _label_from_path(path: Path):
    name = path.stem.upper()
    m = re.match(r"^\d{2}-\d{2}-(\d{2})-", name)
    if m:
        return RAVDESS_MAP.get(m.group(1))
    parts = name.split("_")
    if len(parts) >= 3 and parts[2] in CREMAD_MAP:
        return CREMAD_MAP[parts[2]]
    if len(parts) >= 3:
        emo = parts[-1].lower()
        if emo in EMOTIONS:
            return emo
    for emo in EMOTIONS:
        if emo in name.lower():
            return emo
    return None


def find_dataset_dir():
    candidates = ["dataset", "data", "ravdess", "tess", "cremad",
                  "dataset/ravdess", "dataset/tess", "dataset/cremad"]
    for c in candidates:
        p = BASE / c
        if p.is_dir() and list(p.rglob("*.wav")):
            return p
    return None


def collect_samples(dataset_dir: Path, max_per_class: int = 80):
    from collections import defaultdict
    buckets = defaultdict(list)
    for fp in dataset_dir.rglob("*.wav"):
        lbl = _label_from_path(fp)
        if lbl and lbl in EMOTIONS:
            buckets[lbl].append(fp)

    paths, labels = [], []
    for emo in EMOTIONS:
        items = buckets[emo][:max_per_class]
        paths  += items
        labels += [emo] * len(items)

    total = len(labels)
    if total == 0:
        raise RuntimeError(f"No labelled .wav files found in {dataset_dir}")
    print(f"[DATA] {total} samples selected  "
          f"({dict((e, labels.count(e)) for e in EMOTIONS)})")
    return paths, labels


def predict_all(paths, model, lstm_shape, classes):
    from feature_extraction import extract_features
    import librosa

    T, F = lstm_shape
    all_probs = []
    n = len(paths)
    for i, fp in enumerate(paths, 1):
        if i % 20 == 0 or i == n:
            print(f"  Predicting {i}/{n} ...", end="\r")
        try:
            audio, sr = librosa.load(str(fp), sr=22_050, mono=True)
            cnn_in, lstm_in = extract_features(audio, sr)
            lstm_in = lstm_in[:T, :F]
            if lstm_in.shape[0] < T:
                lstm_in = np.vstack([
                    lstm_in,
                    np.zeros((T - lstm_in.shape[0], lstm_in.shape[1]), np.float32)
                ])
            probs = model.predict(
                [cnn_in[np.newaxis], lstm_in[np.newaxis]], verbose=0
            )[0]
            reordered = np.array([
                probs[classes.index(e)] if e in classes else 0.0
                for e in EMOTIONS
            ], dtype=np.float32)
            all_probs.append(reordered)
        except Exception as exc:
            print(f"\n  [SKIP] {fp.name}: {exc}")
            all_probs.append(np.ones(len(EMOTIONS), np.float32) / len(EMOTIONS))

    print()
    return np.array(all_probs)


def plot_roc(y_true_bin, y_score, out: Path):
    from sklearn.metrics import roc_curve, auc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_classes = len(EMOTIONS)

    # ── Compute per-class ROC ─────────────────────────────────────────────────
    fpr, tpr, roc_auc = {}, {}, {}
    for i, emo in enumerate(EMOTIONS):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    # ── Palette ───────────────────────────────────────────────────────────────
    BG    = "#0d1117"
    PANEL = "#161b22"
    GRID  = "#21262d"
    TEXT  = "#c9d1d9"
    SHADE = "#8b949e"
    DIAG  = "#3d444d"

    COLORS = [
        "#58a6ff",   # happy   – blue
        "#79c0ff",   # sad     – light blue
        "#f78166",   # angry   – coral
        "#a5d6ff",   # neutral – pale blue
        "#d2a8ff",   # fear    – purple
        "#56d364",   # surprise – green
    ]
    MACRO_C = "#e3b341"

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

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
    ax.set_facecolor(PANEL)

    # Diagonal
    ax.plot([0, 1], [0, 1], color=DIAG, lw=1.4, ls="--",
            label="Random (AUC = 0.50)")

    # Per-class curves
    for i, emo in enumerate(EMOTIONS):
        label = f"{emo.capitalize():<9} AUC = {roc_auc[i]:.3f}"
        ax.plot(fpr[i], tpr[i], color=COLORS[i], lw=2.0, alpha=0.88, label=label)

    # Macro-average
    ax.plot(all_fpr, mean_tpr,
            color=MACRO_C, lw=2.6, ls="-.",
            label=f"Macro avg    AUC = {macro_auc:.3f}")

    # Shaded region under macro curve
    ax.fill_between(all_fpr, mean_tpr, alpha=0.06, color=MACRO_C)

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", color=TEXT, fontsize=12)
    ax.set_ylabel("True Positive Rate", color=TEXT, fontsize=12)
    ax.set_title("ROC Curve — Speech Emotion Recognition\n(One-vs-Rest, per emotion)",
                 color=TEXT, fontsize=13, pad=12)
    ax.grid(True, lw=0.6, alpha=0.5)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.7,
              prop={"family": "monospace"})

    fig.text(0.5, 0.01,
             f"Macro AUC: {macro_auc:.4f}   |   "
             f"Classes: {', '.join(EMOTIONS)}",
             ha="center", fontsize=8.5, color=SHADE)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(str(out), dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"DONE  ROC curve saved -> {out}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--max_per_class", type=int, default=80)
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-run inference even if cache exists")
    args = parser.parse_args()

    # Try cache first
    if not args.no_cache and CACHE.exists():
        print(f"[CACHE] Loading cached predictions from {CACHE}")
        cached = np.load(str(CACHE))
        y_score    = cached["y_score"]
        y_true_bin = cached["y_true_bin"]
        print(f"[CACHE] Loaded {y_score.shape[0]} samples.")
    else:
        ds_dir = Path(args.dataset_dir) if args.dataset_dir else find_dataset_dir()
        if ds_dir is None or not ds_dir.exists():
            print("[ERROR] Dataset not found. Pass --dataset_dir <path>", file=sys.stderr)
            sys.exit(1)
        print(f"[DATA] Dataset: {ds_dir}")

        paths, labels = collect_samples(ds_dir, max_per_class=args.max_per_class)

        import tensorflow as tf
        model_path = MODELS / "ser_model.h5"
        if not model_path.exists():
            print("[ERROR] models/ser_model.h5 not found.", file=sys.stderr)
            sys.exit(1)
        print(f"[MODEL] Loading {model_path} ...")
        model = tf.keras.models.load_model(str(model_path))

        lstm_shape = tuple(np.load(str(MODELS / "lstm_shape.npy")).astype(int))
        classes    = np.load(str(MODELS / "emotion_classes.npy"),
                             allow_pickle=True).tolist()
        print(f"[MODEL] LSTM shape: {lstm_shape}  |  Classes: {classes}")

        print("[PRED] Running inference ...")
        y_score = predict_all(paths, model, lstm_shape, classes)

        emo_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
        y_int      = np.array([emo_to_idx[l] for l in labels])
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_int, classes=list(range(len(EMOTIONS))))

        np.savez(str(CACHE), y_score=y_score, y_true_bin=y_true_bin)
        print(f"[CACHE] Predictions saved to {CACHE}")

    out = MODELS / "roc_curve.png"
    print("[PLOT] Generating ROC curve ...")
    plot_roc(y_true_bin, y_score, out)


if __name__ == "__main__":
    main()
