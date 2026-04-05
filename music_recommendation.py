"""
music_recommendation.py
-----------------------
Recommends songs based on the detected emotion using:

  Option A - Spotify API (requires SPOTIPY_CLIENT_ID + SPOTIPY_CLIENT_SECRET)
  Option B - Local songs.csv dataset (always available, no API key needed)

Cosine similarity is used to rank songs whose emotion embedding is closest
to the predicted emotion probability vector.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Force UTF-8 output so emoji display correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise"]
LOCAL_CSV = Path("dataset/songs.csv")
TOP_N = 5

# Spotify emotion → search query mapping
SPOTIFY_QUERIES: Dict[str, str] = {
    "happy":   "happy pop dance feel-good",
    "sad":     "sad acoustic heartbreak slow",
    "angry":   "angry rock metal intense",
    "neutral": "chill lofi ambient relaxing",
    "fear":    "suspense dark atmospheric",
    "surprise":"energetic upbeat unexpected",
}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def recommend_music(
    emotion_label: str,
    emotion_vector: np.ndarray | List[float],
    mode: str = "auto",
    top_n: int = TOP_N,
) -> List[Dict[str, str]]:
    """Return *top_n* song recommendations for the detected emotion.

    Parameters
    ----------
    emotion_label  : Predicted emotion string (e.g. ``"happy"``).
    emotion_vector : Probability vector of shape (6,).
    mode           : ``"spotify"`` | ``"local"`` | ``"auto"``
                     Auto tries Spotify first, falls back to local.
    top_n          : Number of songs to return.

    Returns
    -------
    List of dicts with keys: ``title``, ``artist``, ``emotion``.
    """
    emotion_vector = np.array(emotion_vector, dtype=np.float32).reshape(1, -1)

    if mode == "spotify" or mode == "auto":
        try:
            results = _spotify_recommend(emotion_label, top_n)
            if results:
                return results
        except Exception as exc:
            if mode == "spotify":
                raise
            print(f"[WARN] Spotify unavailable ({exc}). Falling back to local dataset.")

    return _local_recommend(emotion_label, emotion_vector, top_n)


# ──────────────────────────────────────────────────────────────────────────────
# Spotify recommendation
# ──────────────────────────────────────────────────────────────────────────────
def _spotify_recommend(emotion_label: str, top_n: int) -> List[Dict[str, str]]:
    """Fetch top *top_n* tracks from Spotify matching the emotion."""
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    client_id     = os.environ.get("SPOTIPY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET environment variables."
        )

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        )
    )

    query = SPOTIFY_QUERIES.get(emotion_label, emotion_label)
    results = sp.search(q=query, type="track", limit=top_n)
    tracks = results["tracks"]["items"]

    songs = []
    for t in tracks:
        artists   = ", ".join(a["name"] for a in t["artists"])
        album_art = ""
        images    = t.get("album", {}).get("images", [])
        if images:
            # prefer the smallest image (last in list) that's still ≥64px
            album_art = next(
                (img["url"] for img in reversed(images) if img.get("height", 0) >= 64),
                images[-1]["url"],
            )
        songs.append(
            {
                "title":       t["name"],
                "artist":      artists,
                "emotion":     emotion_label,
                "url":         t["external_urls"].get("spotify", ""),
                "preview_url": t.get("preview_url") or "",
                "album_art":   album_art,
                "popularity":  t.get("popularity", 0),
            }
        )
    return songs


# ──────────────────────────────────────────────────────────────────────────────
# Local dataset recommendation
# ──────────────────────────────────────────────────────────────────────────────
def _local_recommend(
    emotion_label: str,
    emotion_vector: np.ndarray,
    top_n: int,
) -> List[Dict[str, str]]:
    """Rank songs from songs.csv by cosine similarity to *emotion_vector*."""
    if not LOCAL_CSV.exists():
        raise FileNotFoundError(f"Local dataset not found: {LOCAL_CSV}")

    df = pd.read_csv(LOCAL_CSV)

    # Extract embedding columns (named e0 … e5)
    embed_cols = [f"e{i}" for i in range(len(EMOTIONS))]
    embeddings = df[embed_cols].values.astype(np.float32)

    # Cosine similarity against the predicted emotion vector
    similarities = cosine_similarity(emotion_vector, embeddings)[0]
    df = df.copy()
    df["similarity"] = similarities

    # Prefer songs whose primary emotion matches, but rank all by similarity
    top = df.sort_values("similarity", ascending=False).head(top_n)

    songs = []
    for _, row in top.iterrows():
        songs.append(
            {
                "title":      row["title"],
                "artist":     row["artist"],
                "emotion":    row["emotion"],
                "similarity": f"{row['similarity']:.3f}",
            }
        )
    return songs


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print helper
# ──────────────────────────────────────────────────────────────────────────────
def print_recommendations(songs: List[Dict[str, str]], emotion_label: str) -> None:
    """Print a formatted recommendation list to stdout."""
    print(f"\n[MUSIC] Recommended Songs for emotion: {emotion_label.upper()}")
    print("-" * 52)
    for i, song in enumerate(songs, start=1):
        url_part = f"  -> {song['url']}" if "url" in song and song["url"] else ""
        print(f"  {i}. {song['title']}  -  {song['artist']}{url_part}")
    print("-" * 52)


# ──────────────────────────────────────────────────────────────────────────────
# CLI helper
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    emo = sys.argv[1] if len(sys.argv) > 1 else "happy"
    # Dummy probability vector
    idx = EMOTIONS.index(emo) if emo in EMOTIONS else 0
    vec = np.full(len(EMOTIONS), 0.04)
    vec[idx] = 0.80
    vec /= vec.sum()

    songs = recommend_music(emo, vec, mode="local")
    print_recommendations(songs, emo)
