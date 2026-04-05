"""
emotion_predictor.py
--------------------
Loads the trained SER dual-network and predicts the emotion label for a given
audio clip.  Falls back to a rule-based heuristic demo when no trained model
is available (useful for smoke-testing without a full dataset).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Force UTF-8 output so emoji display correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path("models/ser_model.h5")
CLASSES_PATH = Path("models/emotion_classes.npy")
SHAPE_PATH   = Path("models/lstm_shape.npy")

EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise"]


# ──────────────────────────────────────────────────────────────────────────────
# Predictor class
# ──────────────────────────────────────────────────────────────────────────────
class EmotionPredictor:
    """Load the trained model once and expose a ``predict`` method.

    Parameters
    ----------
    model_path : Path to the saved ``.h5`` model file.
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self._model = None
        self._classes: list[str] = EMOTIONS
        self._lstm_shape: Tuple[int, int] | None = None
        self._demo_mode: bool = False

        if model_path.exists():
            import tensorflow as tf
            print(f"[MODEL] Loading model from: {model_path}")
            self._model = tf.keras.models.load_model(str(model_path))
            # Load class order
            if CLASSES_PATH.exists():
                self._classes = np.load(str(CLASSES_PATH), allow_pickle=True).tolist()
            # Load LSTM shape
            if SHAPE_PATH.exists():
                shape = np.load(str(SHAPE_PATH))
                self._lstm_shape = (int(shape[0]), int(shape[1]))
        else:
            print(
                "[WARN] No trained model found at models/ser_model.h5.\n"
                "   Running in DEMO mode - predictions are simulated.\n"
                "   Train a model first:  python train_model.py --dataset_dir <dir>\n"
            )
            self._demo_mode = True

    # ── Public API ────────────────────────────────────────────────────────────
    def predict(
        self,
        audio: np.ndarray,
        sr: int = 22_050,
    ) -> Tuple[str, np.ndarray]:
        """Predict emotion from a 1-D audio array.

        Returns
        -------
        (emotion_label, probability_vector)
            ``emotion_label`` is one of the six emotion strings.
            ``probability_vector`` is a (6,) float32 array summing to 1.
        """
        if self._demo_mode or self._model is None:
            return self._heuristic_predict(audio, sr)

        from feature_extraction import extract_features

        cnn_in, lstm_in = extract_features(audio, sr)

        # Pad / truncate LSTM sequence to trained length
        if self._lstm_shape is not None:
            lstm_in = self._pad_or_crop(lstm_in, self._lstm_shape)

        cnn_in  = cnn_in[np.newaxis, ...]    # (1, H, W, 1)
        lstm_in = lstm_in[np.newaxis, ...]   # (1, T, F)

        probs: np.ndarray = self._model.predict([cnn_in, lstm_in], verbose=0)[0]
        label = self._classes[int(np.argmax(probs))]
        return label, probs.astype(np.float32)

    def predict_from_file(self, file_path: str | Path) -> Tuple[str, np.ndarray]:
        """Convenience wrapper: load audio then call :meth:`predict`."""
        import librosa
        # Cap at 8 s on load — avoids slow FFT over long silence / long recordings
        audio, sr = librosa.load(str(file_path), sr=22_050, mono=True, duration=8.0)
        return self.predict(audio, sr)

    def emotion_vector_dict(self, probs: np.ndarray) -> Dict[str, float]:
        """Return a ``{emotion: probability}`` mapping."""
        return {emo: float(probs[i]) for i, emo in enumerate(self._classes)}

    # ── Internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _pad_or_crop(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        T, F = shape
        cur_T, cur_F = arr.shape
        # Crop along time if too long
        arr = arr[:T, :F]
        # Pad with zeros if too short
        if arr.shape[0] < T:
            pad = np.zeros((T - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])
        return arr

    @staticmethod
    def _heuristic_predict(audio: np.ndarray, sr: int) -> Tuple[str, np.ndarray]:
        """Rule-based demo predictor using simple audio statistics.

        This is NOT meant to be accurate — it exists only so the full pipeline
        can be exercised without a trained model.
        """
        import librosa

        # Energy & pitch rough proxies
        rms    = float(np.sqrt(np.mean(audio ** 2)))
        zcr    = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        spec_c = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))

        # Very naive heuristic thresholds
        if rms > 0.15 and zcr > 0.12:
            idx = EMOTIONS.index("angry")
        elif rms > 0.12 and spec_c > 3000:
            idx = EMOTIONS.index("happy")
        elif rms < 0.04:
            idx = EMOTIONS.index("sad")
        elif zcr < 0.06:
            idx = EMOTIONS.index("neutral")
        elif spec_c > 4000:
            idx = EMOTIONS.index("surprise")
        else:
            idx = EMOTIONS.index("fear")

        probs = np.full(len(EMOTIONS), 0.05, dtype=np.float32)
        probs[idx] = 0.75
        probs /= probs.sum()
        return EMOTIONS[idx], probs


# ──────────────────────────────────────────────────────────────────────────────
# Convenience singleton
# ──────────────────────────────────────────────────────────────────────────────
_predictor: EmotionPredictor | None = None


def get_predictor() -> EmotionPredictor:
    global _predictor
    if _predictor is None:
        _predictor = EmotionPredictor()
    return _predictor


def predict_emotion(
    audio: np.ndarray | None = None,
    sr: int = 22_050,
    file_path: str | Path | None = None,
) -> Tuple[str, np.ndarray]:
    """Top-level convenience function.

    Provide either *audio* (NumPy array) or *file_path* (WAV path).
    """
    p = get_predictor()
    if file_path is not None:
        return p.predict_from_file(file_path)
    if audio is not None:
        return p.predict(audio, sr)
    raise ValueError("Must provide either 'audio' or 'file_path'.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI helper
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        label, probs = predict_emotion(file_path=path)
    else:
        # Demo on synthetic noise
        dummy = np.random.randn(22_050 * 4).astype(np.float32) * 0.1
        label, probs = predict_emotion(audio=dummy)

    print(f"\n[RESULT] Emotion detected: {label.upper()}")
    print("\n   Probabilities:")
    for emo, p in zip(EMOTIONS, probs):
        bar = "#" * int(p * 30)
        print(f"   {emo:>10s}: {p:.3f}  {bar}")
