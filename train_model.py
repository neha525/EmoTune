"""
train_model.py
--------------
Trains a Clustering-Based Dual Network:
  - CNN sub-network   — captures spatial patterns from Mel spectrogram
  - Bi-LSTM sub-network — captures temporal patterns from feature sequences
  - Decision-fusion head — concatenates both outputs -> Softmax emotion label

Supported datasets (auto-labelled from file names):
  - RAVDESS  — Actor_*/03-01-<emotion>-...wav
  - TESS     — <actor>/<actor>_<emotion>_...wav
  - CREMA-D  — <id>_<word>_<emotion>_...wav

Usage
-----
    # Standard training
    python train_model.py --dataset_dir dataset/ravdess --epochs 60

    # With data augmentation (noise, time-stretch, pitch-shift) — ~4x data
    python train_model.py --dataset_dir dataset/ravdess --epochs 60 --augment

The trained model is saved to models/ser_model.h5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Windows: force UTF-8 + suppress oneDNN spam
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EMOTIONS        = ["happy", "sad", "angry", "neutral", "fear", "surprise"]
CNN_H, CNN_W    = 128, 128
BATCH_SIZE      = 32
MODEL_SAVE_PATH = Path("models/ser_model.h5")

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


# ─────────────────────────────────────────────────────────────────────────────
# Audio Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def _augment_audio(audio: np.ndarray, sr: int, rng: np.random.Generator) -> list:
    """Return 3 augmented copies of *audio*.

    Augmentations applied:
      1. Gaussian noise injection
      2. Time stretching  (0.85x – 1.15x)
      3. Pitch shifting   (-3 to +3 semitones)
    """
    import librosa

    variants: list = []

    # 1. Gaussian noise
    noise = rng.normal(0, 0.005, size=audio.shape).astype(np.float32)
    variants.append(np.clip(audio + noise, -1.0, 1.0))

    # 2. Time stretch
    rate = float(rng.uniform(0.85, 1.15))
    try:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        variants.append(librosa.util.normalize(stretched))
    except Exception:
        variants.append(audio.copy())

    # 3. Pitch shift
    n_steps = int(rng.integers(-3, 4))
    try:
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        variants.append(librosa.util.normalize(shifted))
    except Exception:
        variants.append(audio.copy())

    return variants


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────
def _label_from_path(path: Path) -> str | None:
    name = path.stem.upper()

    # RAVDESS: 03-01-03-01-01-01-01
    m = re.match(r"^\d{2}-\d{2}-(\d{2})-", name)
    if m:
        return RAVDESS_MAP.get(m.group(1))

    # CREMA-D: 1001_DFA_ANG_XX
    parts = name.split("_")
    if len(parts) >= 3 and parts[2] in CREMAD_MAP:
        return CREMAD_MAP[parts[2]]

    # TESS: OAF_WORD_EMOTION
    if len(parts) >= 3:
        emo = parts[-1].lower()
        if emo in EMOTIONS:
            return emo

    # Generic fallback
    for emo in EMOTIONS:
        if emo in name.lower():
            return emo

    return None


def load_dataset(
    dataset_dir: str | Path,
    augment: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Walk *dataset_dir* and return (cnn_inputs, lstm_inputs, labels).

    If *augment* is True, 3 synthetic variants are appended for every
    original sample (noise, time-stretch, pitch-shift), effectively
    quadrupling the training set to improve generalisation.
    """
    from feature_extraction import extract_features
    import librosa

    dataset_dir = Path(dataset_dir)
    wav_files   = list(dataset_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {dataset_dir}")

    cnn_inputs: List[np.ndarray] = []
    lstm_inputs: List[np.ndarray] = []
    labels: List[str] = []
    skipped = 0
    rng = np.random.default_rng(42)

    desc = "Extracting features" + (" + augmenting" if augment else "")
    for fp in tqdm(wav_files, desc=desc):
        label = _label_from_path(fp)
        if label is None:
            skipped += 1
            continue
        try:
            audio, sr = librosa.load(str(fp), sr=22_050, mono=True)

            # Original
            cnn_in, lstm_in = extract_features(audio, sr)
            cnn_inputs.append(cnn_in)
            lstm_inputs.append(lstm_in)
            labels.append(label)

            # Augmented variants
            if augment:
                for aug in _augment_audio(audio, sr, rng):
                    cnn_a, lstm_a = extract_features(aug, sr)
                    cnn_inputs.append(cnn_a)
                    lstm_inputs.append(lstm_a)
                    labels.append(label)

        except Exception as exc:
            print(f"  [SKIP] {fp.name}: {exc}")
            skipped += 1

    orig = len(wav_files) - skipped
    aug_n = len(labels) - orig
    print(
        f"\n[DATA] {orig} original + {aug_n} augmented = {len(labels)} total samples"
        f"  ({skipped} skipped)"
    )
    return cnn_inputs, lstm_inputs, labels


def _pad_lstm(sequences: List[np.ndarray]) -> np.ndarray:
    max_len = max(s.shape[0] for s in sequences)
    n_feat  = sequences[0].shape[1]
    out = np.zeros((len(sequences), max_len, n_feat), dtype=np.float32)
    for i, s in enumerate(sequences):
        out[i, : s.shape[0], :] = s
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────
def build_cnn(input_shape: Tuple[int, int, int] = (CNN_H, CNN_W, 1)) -> Model:
    """CNN sub-network — spatial feature extraction from Mel spectrogram."""
    inp = Input(shape=input_shape, name="cnn_input")
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    cnn_out = layers.Dense(128, activation="relu", name="cnn_features")(x)
    return Model(inp, cnn_out, name="CNN_SubNetwork")


def build_bilstm(n_timesteps: int, n_features: int, use_attention: bool = False) -> Model:
    """Bi-LSTM sub-network — temporal feature extraction.

    If *use_attention* is True, a scaled dot-product Attention layer is added
    after the first Bi-LSTM to focus on the most emotionally salient frames.
    """
    inp = Input(shape=(n_timesteps, n_features), name="lstm_input")
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.Dropout(0.3)(x)

    if use_attention:
        # Self-attention: query = mean-pooled, value/key = full sequence
        pool = layers.GlobalAveragePooling1D(keepdims=True)(x)  # (B,1,F)
        attn = layers.Attention(name="temporal_attention")([pool, x])  # (B,1,F)
        attn = layers.Flatten()(attn)                                    # (B,F)
        x = layers.Bidirectional(layers.LSTM(64))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Concatenate()([x, attn])
        lstm_out = layers.Dense(128, activation="relu", name="lstm_features")(x)
    else:
        x = layers.Bidirectional(layers.LSTM(64))(x)
        x = layers.Dropout(0.3)(x)
        lstm_out = layers.Dense(128, activation="relu", name="lstm_features")(x)

    return Model(inp, lstm_out, name="BiLSTM_SubNetwork")


def build_fusion_model(
    n_emotions: int,
    n_timesteps: int,
    n_features: int,
    use_attention: bool = False,
) -> Model:
    """Full dual-network with decision-fusion head."""
    cnn_model  = build_cnn()
    lstm_model = build_bilstm(n_timesteps, n_features, use_attention=use_attention)

    fused  = layers.Concatenate(name="fusion")([cnn_model.output, lstm_model.output])
    x      = layers.Dense(128, activation="relu")(fused)
    x      = layers.Dropout(0.3)(x)
    x      = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(n_emotions, activation="softmax", name="emotion_output")(x)

    return Model(inputs=[cnn_model.input, lstm_model.input],
                 outputs=output, name="SER_DualNetwork")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(
    dataset_dir: str,
    epochs: int            = 60,
    batch_size: int        = BATCH_SIZE,
    save_path: Path        = MODEL_SAVE_PATH,
    augment: bool          = False,
    label_smoothing: float = 0.0,
    use_attention: bool    = False,
) -> None:
    print(f"\n{'='*60}")
    print(f"  SER Training  |  augment={'ON' if augment else 'OFF'}  |  epochs={epochs}")
    print(f"  label_smoothing={label_smoothing}  |  attention={'ON' if use_attention else 'OFF'}")
    print(f"{'='*60}\n")

    # 1. Load + augment
    cnn_raw, lstm_raw, labels = load_dataset(dataset_dir, augment=augment)

    # 2. Encode labels
    le = LabelEncoder()
    le.fit(EMOTIONS)
    y     = le.transform(labels)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=len(EMOTIONS))

    # 3. Stack arrays
    X_cnn  = np.stack(cnn_raw, axis=0)   # (N, H, W, 1)
    X_lstm = _pad_lstm(lstm_raw)          # (N, T, F)
    n_timesteps, n_features = X_lstm.shape[1], X_lstm.shape[2]
    print(f"[DATA] X_cnn: {X_cnn.shape}  X_lstm: {X_lstm.shape}")

    # 4. Train / val split (stratified)
    idx = np.arange(len(y))
    tr, val = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)
    print(f"[DATA] Train: {len(tr)}  Val: {len(val)}\n")

    # 5. Build model
    model = build_fusion_model(len(EMOTIONS), n_timesteps, n_features,
                               use_attention=use_attention)
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    model.summary()

    # Save LSTM shape for inference-time sequence padding
    meta_path = save_path.parent / "lstm_shape.npy"
    np.save(str(meta_path), np.array([n_timesteps, n_features]))
    print(f"\n[SAVE] LSTM shape -> {meta_path}")

    # 6. Callbacks
    # Cosine Annealing: restart LR every ~15 epochs for smoother convergence
    def cosine_lr(epoch: int, lr: float) -> float:
        import math
        t_max = max(epochs // 4, 10)
        return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + math.cos(math.pi * (epoch % t_max) / t_max))

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                      verbose=1),
        keras.callbacks.LearningRateScheduler(cosine_lr, verbose=0),
        keras.callbacks.ModelCheckpoint(str(save_path), save_best_only=True,
                                        verbose=0),
    ]

    # 7. Fit
    history = model.fit(
        [X_cnn[tr], X_lstm[tr]], y_cat[tr],
        validation_data=([X_cnn[val], X_lstm[val]], y_cat[val]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    # 8. Summarise results
    best_val_acc = max(history.history["val_accuracy"])
    best_val_ep  = history.history["val_accuracy"].index(best_val_acc) + 1
    print(f"\n[RESULT] Best val accuracy: {best_val_acc:.4f}  (epoch {best_val_ep})")

    # 9. Save training curves
    _plot_history(history, save_path.parent / "training_history.png")
    print(f"[DONE]  Model saved -> {save_path}")

    # 10. Save label encoder classes
    classes_path = save_path.parent / "emotion_classes.npy"
    np.save(str(classes_path), le.classes_)
    print(f"[SAVE]  Emotion classes -> {classes_path}")


def _plot_history(history, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#1a1a2e")
    for ax, (train_key, val_key), title in zip(
        axes,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Loss"],
    ):
        ax.plot(history.history[train_key], label="Train", color="#e94560", linewidth=2)
        ax.plot(history.history[val_key],   label="Val",   color="#53d8fb", linewidth=2)
        ax.set_title(title, color="white", fontsize=13)
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#16213e", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT]  Training history -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SER dual-network model")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Dataset root (RAVDESS / TESS / CREMA-D)")
    parser.add_argument("--epochs",      type=int, default=60)
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--save_path",   type=str, default=str(MODEL_SAVE_PATH))
    parser.add_argument("--augment",          action="store_true",
                        help="Apply noise / time-stretch / pitch-shift augmentation")
    parser.add_argument("--label-smoothing",  type=float, default=0.0,
                        help="Label smoothing epsilon (0 = off, recommended: 0.1)")
    parser.add_argument("--attention",        action="store_true",
                        help="Add temporal attention layer after Bi-LSTM")
    args = parser.parse_args()

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    train(
        args.dataset_dir,
        args.epochs,
        args.batch_size,
        Path(args.save_path),
        args.augment,
        label_smoothing=args.label_smoothing,
        use_attention=args.attention,
    )
