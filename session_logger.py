"""
session_logger.py
-----------------
Appends each prediction + recommendation to history.json so the web UI
can display a running session history.

API
---
    log_session(emotion, probs, songs)  ->  None
    load_history(n=20)                  ->  List[dict]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

# Force UTF-8 I/O on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

HISTORY_FILE = Path("history.json")
EMOTIONS     = ["happy", "sad", "angry", "neutral", "fear", "surprise"]

EMOTION_EMOJI: Dict[str, str] = {
    "happy":    "😄",
    "sad":      "😢",
    "angry":    "😠",
    "neutral":  "😐",
    "fear":     "😨",
    "surprise": "😲",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def log_session(
    emotion: str,
    probs: List[float] | np.ndarray | Dict,
    songs: List[Dict],
    wav_path: str | None = None,
) -> Dict:
    """Append one prediction session to history.json and return the entry.

    *probs* may be:
      - a dict  {emotion: probability}  (preferred — already correctly keyed)
      - a list / np.ndarray             (legacy — zipped with EMOTIONS order)
    """
    if isinstance(probs, dict):
        probs_dict = {emo: round(float(p), 4) for emo, p in probs.items()}
    else:
        if isinstance(probs, np.ndarray):
            probs = probs.tolist()
        probs_dict = {emo: round(float(p), 4) for emo, p in zip(EMOTIONS, probs)}

    entry = {
        "id":        _next_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "emotion":   emotion,
        "emoji":     EMOTION_EMOJI.get(emotion, "🎵"),
        "probs":     probs_dict,
        "songs":     songs,
        "wav_path":  str(wav_path) if wav_path else None,
    }

    history = _load_raw()
    history.append(entry)
    _save_raw(history)
    return entry


def load_history(n: int = 20) -> List[Dict]:
    """Return the last *n* sessions, newest first."""
    history = _load_raw()
    return list(reversed(history[-n:]))


def clear_history() -> None:
    """Wipe all stored sessions."""
    _save_raw([])


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw() -> List[Dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_raw(history: List[Dict]) -> None:
    with HISTORY_FILE.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)


def _next_id() -> int:
    history = _load_raw()
    return (history[-1]["id"] + 1) if history else 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    dummy_probs = np.array([0.8, 0.05, 0.05, 0.04, 0.03, 0.03])
    dummy_songs = [{"title": "Test Song", "artist": "Test Artist", "emotion": "happy"}]
    entry = log_session("happy", dummy_probs, dummy_songs)
    print("[TEST] Logged entry:", json.dumps(entry, indent=2, ensure_ascii=False))

    history = load_history(5)
    print(f"[TEST] History has {len(history)} entries.")
    print("[PASS] session_logger OK")
