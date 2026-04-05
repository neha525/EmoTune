"""
download_dataset.py
-------------------
Downloads the RAVDESS audio-speech dataset from Zenodo and extracts it
into dataset/ravdess/.

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Source: https://zenodo.org/record/1188976
File  : Audio_Speech_Actors_01-24.zip  (~360 MB)
Actors: 24 professional actors (12 male, 12 female)
Labels: neutral, calm, happy, sad, angry, fearful, disgust, surprised

Usage
-----
    python download_dataset.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

# Force UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Missing: pip install requests tqdm")
    sys.exit(1)

# ─── Config ───────────────────────────────────────────────────────────────────
ZENODO_URL = (
    "https://zenodo.org/record/1188976/files/"
    "Audio_Speech_Actors_01-24.zip?download=1"
)
ZIP_PATH    = Path("dataset/ravdess.zip")
EXTRACT_DIR = Path("dataset/ravdess")


# ─── Download helper ──────────────────────────────────────────────────────────
def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    print(f"[DOWNLOAD] Downloading RAVDESS ({total // 1_048_576} MB)...")

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024,
        desc="  Progress", ncols=80,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65_536):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"[SAVED] Zip saved to: {dest}")


# ─── Extract helper ───────────────────────────────────────────────────────────
def extract(zip_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[EXTRACT] Extracting to: {extract_to} ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="  Extracting", ncols=80):
            zf.extract(member, extract_to)

    print(f"[DONE] Extraction complete. Files in: {extract_to}")


# ─── Verify ───────────────────────────────────────────────────────────────────
def verify(extract_to: Path) -> int:
    wav_files = list(extract_to.rglob("*.wav"))
    print(f"[CHECK] Found {len(wav_files)} .wav files in {extract_to}")
    return len(wav_files)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== RAVDESS Dataset Downloader ===\n")

    # Skip download if already extracted
    if EXTRACT_DIR.exists() and verify(EXTRACT_DIR) > 100:
        print("[SKIP] Dataset already exists. Ready to train!")
        print(f"       Run: python train_model.py --dataset_dir {EXTRACT_DIR} --epochs 50")
        sys.exit(0)

    # 1. Download
    if not ZIP_PATH.exists():
        download(ZENODO_URL, ZIP_PATH)
    else:
        print(f"[SKIP] Zip already present: {ZIP_PATH}")

    # 2. Extract
    extract(ZIP_PATH, EXTRACT_DIR)

    # 3. Verify
    count = verify(EXTRACT_DIR)
    if count < 100:
        print(f"[WARN] Expected ~1440 files, found only {count}. Check the zip.")
    else:
        print(f"\n[OK] Dataset ready with {count} audio files.")
        print(f"     Now run: python train_model.py --dataset_dir {EXTRACT_DIR} --epochs 50")

    # 4. Clean up zip to save space
    answer = input("\nDelete the zip file to save ~360 MB? [y/N]: ").strip().lower()
    if answer == "y":
        ZIP_PATH.unlink()
        print("[CLEAN] Zip deleted.")
