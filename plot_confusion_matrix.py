"""
plot_confusion_matrix.py
------------------------
Generates a styled confusion matrix for the trained SER model.
Caches predictions to models/cm_cache.npz so re-runs are instant.

Usage
-----
    python plot_confusion_matrix.py --dataset_dir dataset/ravdess
    python plot_confusion_matrix.py --dataset_dir dataset/ravdess --no-cache
    python plot_confusion_matrix.py --dataset_dir dataset/ravdess --normalize
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
CACHE  = MODELS / "cm_cache.npz"

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


# ── Label helpers ─────────────────────────────────────────────────────────────
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


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_all(paths, labels, model, lstm_shape, classes):
    from feature_extraction import extract_features
    import librosa

    T, F = lstm_shape
    y_pred, y_true = [], []
    n = len(paths)

    emo_to_idx = {e: i for i, e in enumerate(EMOTIONS)}

    for i, (fp, lbl) in enumerate(zip(paths, labels), 1):
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
            pred_class = classes[np.argmax(probs)]
            y_pred.append(emo_to_idx.get(pred_class, 0))
            y_true.append(emo_to_idx[lbl])
        except Exception as exc:
            print(f"\n  [SKIP] {fp.name}: {exc}")

    print()
    return np.array(y_true), np.array(y_pred)


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, normalize: bool, out: Path):
    from sklearn.metrics import (confusion_matrix, classification_report,
                                 accuracy_score, f1_score)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # ── Palette ───────────────────────────────────────────────────────────────
    BG    = "#0d1117"
    PANEL = "#161b22"
    GRID  = "#21262d"
    TEXT  = "#c9d1d9"
    SHADE = "#8b949e"

    # Compute raw cm first (needed for annotations in normalized mode)
    cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTIONS))))

    if normalize:
        cm_plot = cm_raw.astype(float)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1          # avoid div/0
        cm_plot = cm_plot / row_sums
        fmt_fn  = lambda v: f"{v:.2f}"
        title_sfx = " (Normalized)"
    else:
        cm_plot = cm_raw.astype(float)
        fmt_fn  = lambda v: str(int(v))
        title_sfx = ""

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.facecolor":    PANEL,
        "figure.facecolor":  BG,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "text.color":        TEXT,
    })

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
    ax.set_facecolor(PANEL)

    # Colour map: dark background → vivid blue tones
    cmap = plt.cm.get_cmap("YlOrRd")
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=(1.0 if normalize else cm_raw.max()))

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    cbar.outline.set_edgecolor(GRID)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)

    # Tick labels
    labels_cap = [e.capitalize() for e in EMOTIONS]
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_xticklabels(labels_cap, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(labels_cap, fontsize=10)

    # Cell annotations
    thresh = cm_plot.max() / 2.0
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            val = cm_plot[i, j]
            color = "#0d1117" if val > thresh else TEXT
            txt = fmt_fn(val)
            if normalize:
                txt += f"\n({cm_raw[i, j]})"          # show raw count too
            ax.text(j, i, txt,
                    ha="center", va="center",
                    color=color, fontsize=9,
                    fontweight="bold" if i == j else "normal")

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=8)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=8)
    ax.set_title(
        f"Confusion Matrix — Speech Emotion Recognition{title_sfx}",
        color=TEXT, fontsize=13, pad=14
    )

    # ── Per-class metrics footer ───────────────────────────────────────────────
    report   = classification_report(y_true, y_pred,
                                     target_names=labels_cap,
                                     output_dict=True, zero_division=0)
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    footer = (f"Overall Accuracy: {acc:.4f}   |   Macro F1: {macro_f1:.4f}   |   "
              f"Samples: {len(y_true)}")
    fig.text(0.5, 0.01, footer,
             ha="center", fontsize=8.5, color=SHADE)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(str(out), dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"DONE  Confusion matrix saved -> {out}")

    # ── Console report ────────────────────────────────────────────────────────
    print(f"\n[METRICS] Overall Accuracy : {acc:.4f}")
    print(f"[METRICS] Macro F1 Score   : {macro_f1:.4f}")
    print(f"\n[CLASSIFICATION REPORT]\n")
    print(classification_report(y_true, y_pred,
                                target_names=labels_cap, zero_division=0))


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",   type=str, default=None)
    parser.add_argument("--max_per_class", type=int, default=80)
    parser.add_argument("--normalize",     action="store_true",
                        help="Show row-normalised values (recall per class)")
    parser.add_argument("--no-cache",      action="store_true",
                        help="Force re-run inference even if cache exists")
    args = parser.parse_args()

    # ── Try cache ─────────────────────────────────────────────────────────────
    if not args.no_cache and CACHE.exists():
        print(f"[CACHE] Loading cached predictions from {CACHE}")
        cached = np.load(str(CACHE))
        y_true = cached["y_true"]
        y_pred = cached["y_pred"]
        print(f"[CACHE] Loaded {len(y_true)} samples.")
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
        y_true, y_pred = predict_all(paths, labels, model, lstm_shape, classes)

        np.savez(str(CACHE), y_true=y_true, y_pred=y_pred)
        print(f"[CACHE] Predictions saved to {CACHE}")

    suffix = "_normalized" if args.normalize else ""
    out = MODELS / f"confusion_matrix{suffix}.png"
    print("[PLOT] Generating confusion matrix ...")
    plot_confusion_matrix(y_true, y_pred, normalize=args.normalize, out=out)


if __name__ == "__main__":
    main()
