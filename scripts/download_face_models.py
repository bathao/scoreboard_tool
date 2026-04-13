"""Download InsightFace buffalo_l ONNX models for player identification.

Downloads the ArcFace recognition model (w600k_r50.onnx) from the official
InsightFace GitHub release. The full buffalo_l zip is fetched and the required
model file is extracted.

Usage:
    python scripts/download_face_models.py

Output:
    data/models/face/w600k_r50.onnx  (~166 MB)
"""
from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "data" / "models" / "face"

# Official InsightFace buffalo_l release (stable since v0.7)
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

# Only these files are needed from the zip (recognition model only)
REQUIRED_FILES = {"w600k_r50.onnx"}


def download_with_progress(url: str) -> bytes:
    """Download url, printing a simple progress indicator."""
    import urllib.request

    print(f"Downloading: {url}")
    data = bytearray()
    MB = 1024 * 1024

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunk_size = 256 * 1024
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            data.extend(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                mb = downloaded / MB
                print(f"\r  {pct:3d}%  {mb:.1f} MB", end="", flush=True)
    print()
    return bytes(data)


def extract_models(zip_bytes: bytes, dest_dir: Path) -> list[str]:
    """Extract only the required model files from the zip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for entry in zf.infolist():
            fname = Path(entry.filename).name
            if fname in REQUIRED_FILES:
                out_path = dest_dir / fname
                data = zf.read(entry.filename)
                out_path.write_bytes(data)
                size_mb = len(data) / (1024 * 1024)
                print(f"  Extracted {fname}  ({size_mb:.1f} MB)  -> {out_path}")
                extracted.append(fname)
    return extracted


def check_existing() -> list[str]:
    missing = []
    for fname in REQUIRED_FILES:
        if not (MODEL_DIR / fname).exists():
            missing.append(fname)
    return missing


def main() -> int:
    print(f"\nInsightFace model downloader")
    print(f"Target dir: {MODEL_DIR}\n")

    missing = check_existing()
    if not missing:
        print("All models already present:")
        for fname in REQUIRED_FILES:
            size_mb = (MODEL_DIR / fname).stat().st_size / (1024 * 1024)
            print(f"  OK  {fname}  ({size_mb:.1f} MB)")
        return 0

    print(f"Missing: {missing}")
    print()

    try:
        zip_bytes = download_with_progress(BUFFALO_L_URL)
        extracted = extract_models(zip_bytes, MODEL_DIR)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("\nManual fallback:")
        print(f"  1. Download {BUFFALO_L_URL}")
        print(f"  2. Extract w600k_r50.onnx into {MODEL_DIR}/")
        return 1

    if extracted:
        print(f"\nDownloaded: {extracted}")
    else:
        print("\nERROR: Required model files not found in the zip.")
        return 1

    # Verify
    still_missing = check_existing()
    if still_missing:
        print(f"ERROR: Still missing after download: {still_missing}")
        return 1

    size_mb = (MODEL_DIR / "w600k_r50.onnx").stat().st_size / (1024 * 1024)
    print(f"w600k_r50.onnx  {size_mb:.1f} MB  ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
