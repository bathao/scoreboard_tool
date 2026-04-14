"""Enroll a player into the face DB from manually-provided face images.

The user provides one or more image files (face crops or full frames).
The script embeds each image and averages the embeddings before enrolling.

Usage:
    python scripts/enroll_player.py --name "Trần Quang Vinh" --images img1.jpg img2.jpg img3.jpg
    python scripts/enroll_player.py --name "Trần Quang Vinh" --images data/face_crops/vinh/*.jpg

Drop face images into any folder and point --images at them.
Images can be:
  - Already-cropped face images (any size, any aspect ratio)
  - Full video frames (the script tries YOLO pose to locate the face)

Tips for good enrollment:
  - 3-10 images with clearly visible face (frontal or slight angle)
  - Avoid back-of-head, heavy blur, or extreme side profile
  - Different lighting / expressions → more robust embedding
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from backend.player_identity import FaceDB, FaceEmbedder, align_face_from_keypoints

FACE_MODEL_PATH = ROOT / "data" / "models" / "face" / "w600k_r50.onnx"
FACE_DB_PATH    = ROOT / "data" / "players" / "faces.json"
POSE_WEIGHTS    = ROOT / "backend" / "weights" / "yolov8x-pose.pt"

# If the image is small (< this threshold on shortest side), treat it as already a face crop
ALREADY_CROP_THRESHOLD_PX = 300


def _embed_from_full_frame(frame: np.ndarray, embedder: FaceEmbedder) -> list[np.ndarray]:
    """Try YOLO pose on a full frame to detect and align faces. Returns list of embeddings."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(POSE_WEIGHTS))
    except Exception:
        return []

    results = model.predict(frame, verbose=False, half=True, device=0)
    if not results or results[0].boxes is None:
        return []

    kpts_all = results[0].keypoints
    if kpts_all is None:
        return []

    kpts_xy   = kpts_all.xy.cpu().numpy()
    kpts_conf = kpts_all.conf.cpu().numpy()

    embeddings = []
    for i in range(len(kpts_xy)):
        aligned = align_face_from_keypoints(frame, kpts_xy[i], kpts_conf[i], out_size=112)
        if aligned is None:
            continue
        if aligned.std() < 8.0:
            continue
        emb = embedder.embed(aligned)
        embeddings.append(emb)
    return embeddings


def _embed_as_crop(img: np.ndarray, embedder: FaceEmbedder) -> np.ndarray:
    """Treat the whole image as a face crop: resize to 112x112 and embed directly."""
    resized = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    return embedder.embed(resized)


def embed_image(img_path: Path, embedder: FaceEmbedder, force_crop: bool = False) -> list[np.ndarray]:
    """Return one or more embeddings from an image file.

    Strategy:
    1. If the image is small (< ALREADY_CROP_THRESHOLD_PX on shortest side) or
       force_crop=True: treat as face crop, resize and embed directly.
    2. Otherwise: try YOLO pose detection on the full frame.
    3. If YOLO returns nothing: fall back to treating the whole image as a crop.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [SKIP] Cannot read: {img_path.name}")
        return []

    h, w = img.shape[:2]
    shortest = min(h, w)

    if force_crop or shortest < ALREADY_CROP_THRESHOLD_PX:
        emb = _embed_as_crop(img, embedder)
        return [emb]

    # Try full-frame YOLO detection first
    embeddings = _embed_from_full_frame(img, embedder)
    if embeddings:
        return embeddings

    # Fallback: treat as crop
    print(f"  [WARN] YOLO found no face in {img_path.name}, using full-image fallback")
    emb = _embed_as_crop(img, embedder)
    return [emb]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name",   required=True, help="Player full name (e.g. 'Trần Quang Vinh')")
    parser.add_argument("--images", required=True, nargs="+", help="Path(s) to face image files")
    parser.add_argument("--crop",   action="store_true",
                        help="Force treat all images as already-cropped faces (skip YOLO)")
    parser.add_argument("--update", action="store_true",
                        help="If player already exists in DB, average new embeddings into existing record")
    args = parser.parse_args()

    # --- Load model ---
    if not FACE_MODEL_PATH.exists():
        print(f"ERROR: ArcFace model not found: {FACE_MODEL_PATH}")
        print(f"Run: python scripts/download_face_models.py")
        return 1

    print(f"\nLoading ArcFace model...")
    embedder = FaceEmbedder(FACE_MODEL_PATH)

    # --- Collect images ---
    img_paths = []
    for pattern in args.images:
        p = Path(pattern)
        if p.is_file():
            img_paths.append(p)
        else:
            # Try glob
            matched = sorted(ROOT.glob(pattern)) or sorted(Path(".").glob(pattern))
            img_paths.extend(matched)

    img_paths = [p for p in img_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    if not img_paths:
        print(f"ERROR: No valid image files found in: {args.images}")
        return 1

    print(f"Processing {len(img_paths)} image(s) for '{args.name}'...\n")

    # --- Embed all images ---
    all_embeddings: list[np.ndarray] = []
    for img_path in img_paths:
        embs = embed_image(img_path, embedder, force_crop=args.crop)
        if embs:
            all_embeddings.extend(embs)
            print(f"  OK  {img_path.name}  ({len(embs)} embedding(s))")
        else:
            print(f"  FAIL  {img_path.name}")

    if not all_embeddings:
        print(f"\nERROR: No embeddings extracted. Check image quality.")
        return 1

    # --- Average embeddings ---
    stacked = np.stack(all_embeddings, axis=0)        # [N, 512]
    avg_emb = stacked.mean(axis=0)
    avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-9)
    print(f"\n  Averaged {len(all_embeddings)} embedding(s) into 1 record.")

    # --- Enroll ---
    db = FaceDB(FACE_DB_PATH)
    existing = next((r for r in db.records if r.name == args.name), None)

    if existing and args.update:
        db.update_embedding(existing.player_id, avg_emb)
        print(f"  Updated existing record for '{args.name}'")
    elif existing and not args.update:
        print(f"\n  '{args.name}' already in DB.")
        print(f"  Re-run with --update to merge new embeddings into existing record.")
        return 0
    else:
        record = db.enroll(args.name, avg_emb)
        print(f"  Enrolled: '{args.name}'  (id={record.player_id[:8]}...)")

    db.save()
    print(f"\nDB saved: {FACE_DB_PATH}")
    print(f"Total players in DB: {len(db)}")
    for r in db.records:
        print(f"  - {r.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
