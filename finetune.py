"""
finetune.py
-----------
Fine-tunes the existing ser_model.h5 on RAVDESS with:
  - Lower learning rate  (1e-4)  to avoid over-shooting the learned weights
  - Larger batch size    (64)    for faster steps
  - CosineDecay LR schedule for smooth convergence
  - EarlyStopping patience = 15  (more patience since LR is low)
  - Label smoothing = 0.1        to regularise the already-trained model

Usage
-----
    python finetune.py --dataset_dir dataset/ravdess --epochs 40
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows: force UTF-8 + suppress spam
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")

EMOTIONS        = ["happy", "sad", "angry", "neutral", "fear", "surprise"]
BATCH_SIZE      = 64
MODEL_PATH      = Path("models/ser_model.h5")
CLASSES_PATH    = Path("models/emotion_classes.npy")
LSTM_SHAPE_PATH = Path("models/lstm_shape.npy")


# ─────────────────────────────────────────────────────────────────────────────
# Load dataset (reuse train_model loader)
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare(dataset_dir: str):
    from train_model import load_dataset, _pad_lstm

    cnn_raw, lstm_raw, labels = load_dataset(dataset_dir, augment=False)

    le = LabelEncoder()
    le.fit(EMOTIONS)
    y     = le.transform(labels)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=len(EMOTIONS))

    X_cnn  = np.stack(cnn_raw, axis=0)
    X_lstm = _pad_lstm(lstm_raw)

    # Pad / trim X_lstm to match the saved model's expected timestep length
    if LSTM_SHAPE_PATH.exists():
        saved_t, saved_f = np.load(str(LSTM_SHAPE_PATH)).astype(int)
        T_cur = X_lstm.shape[1]
        if T_cur < saved_t:
            pad = np.zeros((X_lstm.shape[0], saved_t - T_cur, X_lstm.shape[2]),
                           dtype=np.float32)
            X_lstm = np.concatenate([X_lstm, pad], axis=1)
            print(f"[SHAPE] Padded LSTM from T={T_cur} -> T={saved_t}")
        elif T_cur > saved_t:
            X_lstm = X_lstm[:, :saved_t, :]
            print(f"[SHAPE] Cropped LSTM from T={T_cur} -> T={saved_t}")

    idx = np.arange(len(y))
    tr, val = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)
    return X_cnn, X_lstm, y_cat, y, tr, val


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tune
# ─────────────────────────────────────────────────────────────────────────────
def finetune(dataset_dir: str, epochs: int = 40) -> None:
    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {MODEL_PATH}  |  batch={BATCH_SIZE}  |  epochs={epochs}")
    print(f"{'='*60}\n")

    # 1. Load data
    X_cnn, X_lstm, y_cat, y, tr, val = load_and_prepare(dataset_dir)
    print(f"[DATA] Train: {len(tr)}  Val: {len(val)}")
    print(f"[DATA] X_cnn: {X_cnn.shape}  X_lstm: {X_lstm.shape}\n")

    # 2. Load existing model
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found at {MODEL_PATH}. Run train_model.py first.")
        sys.exit(1)
    print(f"[LOAD] Loading model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    # 3. Re-compile with low LR + label smoothing
    steps_per_epoch = len(tr) // BATCH_SIZE
    total_steps     = steps_per_epoch * epochs
    lr_schedule     = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=total_steps,
        alpha=1e-6,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    # 4. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
    ]

    # 5. Fit
    history = model.fit(
        [X_cnn[tr], X_lstm[tr]], y_cat[tr],
        validation_data=([X_cnn[val], X_lstm[val]], y_cat[val]),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # 6. Summary
    best_val_acc = max(history.history["val_accuracy"])
    best_ep      = history.history["val_accuracy"].index(best_val_acc) + 1
    print(f"\n[RESULT] Best val accuracy: {best_val_acc:.4f}  (epoch {best_ep})")

    # 7. Save plot
    _plot(history, Path("models/finetune_history.png"))
    print(f"[DONE] Model updated -> {MODEL_PATH}")


def _plot(history, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#1a1a2e")
    for ax, (tk, vk), title in zip(
        axes,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Loss"],
    ):
        ax.plot(history.history[tk], label="Train", color="#e94560", linewidth=2)
        ax.plot(history.history[vk], label="Val",   color="#53d8fb", linewidth=2)
        ax.set_title(title, color="white", fontsize=13)
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#16213e", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT]  Fine-tune history -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune existing SER model")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--epochs",      type=int, default=40)
    args = parser.parse_args()
    finetune(args.dataset_dir, args.epochs)
