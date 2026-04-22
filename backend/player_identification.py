"""Player identification pipeline — scan video to resolve NEAR/FAR player names.

Runs BEFORE the main rally-detection pipeline. Uses:
  - Signal 3 set boundaries (already detected) as primary face-capture windows.
    During the side-swap walk (~20–30 s after a set ends), players physically
    walk toward the fixed camera — guaranteed face-visible frames.
  - Jersey color (HSV histogram) as a per-session anchor for Set 2+ re-tracking.

Usage (standalone debug):
    python scripts/run_player_identification.py --video input.mp4 --job-id <id>

Usage (programmatic, after apply_set_numbers()):
    from backend.player_identification import run_player_identification
    result = run_player_identification(timeline, video_path, config, face_db)
    # result.near_name, result.far_name  (None = not identified)

Design:
  - Phase B: sample frames inside each set-boundary window (side-swap walk)
  - Phase C: extract jersey color from each detected body at calm between-rally moments
  - Phase D: bind face identity to jersey; re-track across sets by jersey alone
  - Phase E: collect UnknownFace entries → operator enrolls via Web UI
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.player_identity import (
    FaceDB,
    FaceEmbedder,
    PlayerRecord,
    align_face_from_keypoints,
    crop_face_from_keypoints,
    face_similarity,
    DEFAULT_MATCH_THRESHOLD,
)
from backend.rally_timeline_contract import RallyTimeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Window to sample after a set boundary (seconds).
# The side-swap walk typically takes 20–30 s; we scan a 35 s window to be safe.
BOUNDARY_WINDOW_SEC: float = 35.0

# Sampling rate inside the boundary window (frames per second).
BOUNDARY_SAMPLE_FPS: float = 2.0

# Minimum face detection confidence to use a detected face for embedding.
MIN_FACE_AREA_PX: int = 30 * 30  # face crop must be at least 30×30 before resize

# Jersey HSV histogram bins per channel
JERSEY_HIST_BINS: tuple[int, int, int] = (16, 8, 8)

# Jersey color distance threshold — if two jerseys are closer than this,
# they are flagged as "ambiguous" (too similar to distinguish).
JERSEY_AMBIGUOUS_THRESHOLD: float = 0.12

# Upper-torso fraction of body bbox used for jersey sampling (top portion of body)
JERSEY_TORSO_FRACTION_TOP: float = 0.12    # skip top 12% (head)
JERSEY_TORSO_FRACTION_BOTTOM: float = 0.45  # use down to 45% from top

# Minimum number of valid jersey samples to trust the histogram
JERSEY_MIN_SAMPLES: int = 3

# Confidence threshold for YOLO keypoints used in face alignment
_MIN_KPT_CONF = 0.4

# YOLO keypoint indices (COCO pose order)
_KPT_NOSE = 0
_KPT_LEYE = 1
_KPT_REYE = 2
_KPT_LEAR = 3
_KPT_REAR = 4


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class UnknownFace:
    """A face detected but not matched in the DB — needs operator enrollment."""
    body_role: str                      # "near" or "far"
    boundary_sec: float                 # timestamp of best crop
    face_embedding: np.ndarray          # averaged 512-dim embedding for DB matching
    best_crop_bgr: Optional[np.ndarray] = None   # single best 112x112 crop (highest face_score)
    top_crops: list = dataclasses.field(default_factory=list)
    # top_crops: list of (face_score, t_sec, crop_bgr) tuples, sorted best-first


@dataclasses.dataclass
class IdentificationResult:
    near_name: Optional[str]  # None = not identified
    far_name: Optional[str]
    near_jersey_hist: Optional[np.ndarray]
    far_jersey_hist: Optional[np.ndarray]
    status: str               # "identified" | "partial" | "failed"
    unknown_faces: list[UnknownFace] = dataclasses.field(default_factory=list)
    # Table ROI detected at the start of identification; reused downstream by
    # rally detection to avoid running YOLOv8x-table twice on the same video.
    table_roi: Optional[object] = None  # TableROI (avoid top-level import cycle)
    # Player-zone crop used by Step 2. Step 3.1 trusts this zone instead of
    # deriving a wider side-swap zone that may include adjacent tables.
    player_zone_xyxy: Optional[tuple[float, float, float, float]] = None

    @property
    def is_complete(self) -> bool:
        return self.near_name is not None and self.far_name is not None


# ---------------------------------------------------------------------------
# Jersey feature extraction (Phase C)
# ---------------------------------------------------------------------------

def extract_jersey_hist(
    frame: np.ndarray,
    body_bbox_xyxy: np.ndarray,
) -> Optional[np.ndarray]:
    """Extract a normalised HSV histogram from the upper-torso region of a body bbox.

    Args:
        frame: BGR image.
        body_bbox_xyxy: [x1, y1, x2, y2] body bounding box.

    Returns:
        Flat normalised histogram (H*S*V bins), or None if region is too small.
    """
    x1, y1, x2, y2 = body_bbox_xyxy.astype(int)
    bh = y2 - y1
    # Crop upper torso (skip head region at very top)
    ty1 = y1 + int(bh * JERSEY_TORSO_FRACTION_TOP)
    ty2 = y1 + int(bh * JERSEY_TORSO_FRACTION_BOTTOM)
    if ty2 <= ty1 or x2 <= x1:
        return None

    h, w = frame.shape[:2]
    tx1 = max(0, x1)
    tx2 = min(w, x2)
    ty1 = max(0, ty1)
    ty2 = min(h, ty2)

    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hb, sb, vb = JERSEY_HIST_BINS
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [hb, sb, vb],
        [0, 180, 0, 256, 0, 256],
    )
    hist = hist.flatten().astype(np.float32)
    total = hist.sum()
    if total < 1:
        return None
    return hist / total


def jersey_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Chi-squared distance between two normalised histograms (range [0, inf))."""
    denom = h1 + h2
    mask = denom > 1e-9
    diff = (h1[mask] - h2[mask]) ** 2 / denom[mask]
    return float(diff.sum())


def _accumulate_jersey(hists: list[np.ndarray]) -> Optional[np.ndarray]:
    """Average a list of jersey histograms."""
    valid = [h for h in hists if h is not None]
    if len(valid) < JERSEY_MIN_SAMPLES:
        return None
    return np.stack(valid, axis=0).mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Table ROI + player zone
# ---------------------------------------------------------------------------

# Player zone = Table ROI bbox expanded on each side by these fractions of the
# table bbox width/height.  The zone filters out players at adjacent tables.
# X: 25% on each side (total +50% width) — trims adjacent-table spill
# Y: 110% on each side (total +220% height) — extra headroom for tall players
#    above the table surface to capture full body + faces
PLAYER_ZONE_EXPAND_X: float = 0.25
PLAYER_ZONE_EXPAND_Y: float = 1.10


def detect_table_roi_and_player_zone(
    video_path: str,
    table_weights_path: str,
    device: str = "cuda",
    expand_x: float = PLAYER_ZONE_EXPAND_X,
    expand_y: float = PLAYER_ZONE_EXPAND_Y,
):
    """Detect the table ROI (YOLOv8x-table) and derive the player zone around it.

    The table ROI is the bounding box of the actual table surface.  The player
    zone is that bbox expanded by (expand_x, expand_y) on each side to cover
    where players stand and move; it is used to filter face captures to the
    main match and exclude players at adjacent tables.

    Args:
        video_path:         Path to the video file.
        table_weights_path: Path to YOLOv8x-table weights.
        device:             Torch device for table detection.
        expand_x, expand_y: Per-side expansion fractions of the table bbox.

    Returns:
        (table_roi, player_zone_xyxy):
          - table_roi: TableROI object (raw detection, pass-through to rally pipeline)
          - player_zone_xyxy: (x1, y1, x2, y2) pixel-coord bbox for identification filtering

        Returns (None, None) if detection fails.
    """
    import cv2 as _cv2
    from backend.ai_table_roi_dl import DLConfig, detect_table_roi_dl

    try:
        roi = detect_table_roi_dl(
            str(video_path),
            cfg=DLConfig(weights_path=str(table_weights_path), device=device),
        )
    except Exception:
        return None, None

    if roi is None or roi.w <= 0 or roi.h <= 0:
        return None, None

    cap = _cv2.VideoCapture(str(video_path))
    frame_w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frame_w <= 0 or frame_h <= 0:
        return None, None

    bw = float(roi.w)
    bh = float(roi.h)
    px1 = max(0.0, float(roi.x) - bw * expand_x)
    py1 = max(0.0, float(roi.y) - bh * expand_y)
    px2 = min(float(frame_w), float(roi.x + roi.w) + bw * expand_x)
    py2 = min(float(frame_h), float(roi.y + roi.h) + bh * expand_y)

    return roi, (px1, py1, px2, py2)


# ---------------------------------------------------------------------------
# Frame-level face + body detection helpers
# ---------------------------------------------------------------------------

def _detect_bodies_and_faces(
    frame: np.ndarray,
    yolo_model,
    roi_xyxy: Optional[tuple[float, float, float, float]] = None,
) -> list[dict]:
    """Run YOLO pose inference on a frame and return structured detections.

    Args:
        roi_xyxy: Optional (x1, y1, x2, y2) pixel-coordinate ROI.  Only
            detections whose bbox CENTER falls inside this region are kept.
            Use estimate_table_roi() to compute it once per video.

    Returns a list of dicts:
        {
            "bbox_xyxy": np.ndarray [4],   body bounding box
            "area": float,                  bounding box area
            "kpts_xy": np.ndarray [17, 2],
            "kpts_conf": np.ndarray [17],
        }
    Sorted by area descending (near player = largest bbox = first).
    """
    results = yolo_model.predict(frame, verbose=False, half=True, device=0)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    boxes = results[0].boxes.xyxy.cpu().numpy()     # [N, 4]
    kpts_all = results[0].keypoints
    if kpts_all is None:
        return []

    kpts_xy = kpts_all.xy.cpu().numpy()             # [N, 17, 2]
    kpts_conf = kpts_all.conf.cpu().numpy()         # [N, 17]

    detections = []
    for i in range(len(boxes)):
        b = boxes[i]
        area = (b[2] - b[0]) * (b[3] - b[1])

        # ROI filter: skip players whose bbox center is outside the table ROI.
        # This prevents picking up players on adjacent tables.
        if roi_xyxy is not None:
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            rx1, ry1, rx2, ry2 = roi_xyxy
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                continue

        detections.append({
            "bbox_xyxy": b,
            "area": area,
            "kpts_xy": kpts_xy[i],
            "kpts_conf": kpts_conf[i],
        })

    detections.sort(key=lambda d: -d["area"])
    return detections


def _face_visibility_score(kpts_conf: np.ndarray) -> float:
    """Return a face visibility score [0, 1] based on nose + eye keypoint confidences.

    Score = min(conf_nose, conf_leye, conf_reye).
    Only counts as a real face if all three are above MIN_FACE_CONF.
    Returns 0.0 when the face is not visible (back of head, looking away).
    """
    return float(min(kpts_conf[_KPT_NOSE], kpts_conf[_KPT_LEYE], kpts_conf[_KPT_REYE]))


# Minimum per-keypoint confidence to treat a crop as "face visible".
# Set at 0.50 so that small/far-side bodies with slightly lower YOLO confidence
# are still captured when the face is actually visible.
MIN_FACE_CONF: float = 0.50


def _face_display_crop(
    frame: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    out_size: int = 224,
    padding: float = 0.50,
) -> Optional[np.ndarray]:
    """Produce a natural-orientation face crop for human display (NOT aligned).

    Uses nose + eye keypoints to locate the face center, then takes a simple
    square crop from the original frame — no rotation or warping.
    Looks natural to the operator for enrollment confirmation.
    """
    idxs = [_KPT_NOSE, _KPT_LEYE, _KPT_REYE]
    valid = [i for i in idxs if kpts_conf[i] >= _MIN_KPT_CONF]
    if len(valid) < 2:
        return None
    pts = kpts_xy[valid]
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    span = max(
        abs(kpts_xy[_KPT_LEYE][0] - kpts_xy[_KPT_REYE][0]) * 2.5,
        40.0,
    )
    half = span * (0.5 + padding)
    h, w = frame.shape[:2]
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half * 1.2))   # slightly more room above (forehead)
    x2 = min(w, int(cx + half))
    y2 = min(h, int(cy + half * 0.8))   # slightly less below (chin)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def _face_kpts_in_head_region(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    bbox_xyxy: np.ndarray,
    head_fraction: float = 0.35,
) -> bool:
    """Return True if all visible face keypoints (nose, eyes) lie inside the
    expected head region of this body's bounding box.

    Prevents keypoint contamination: in crowded frames YOLO sometimes assigns
    facial keypoints from the large NEAR player (rank-0) to the small FAR
    player's detection (rank-1), causing a crop of the wrong person.

    head_fraction: fraction of bbox height counted as 'head' (top of bbox).
    """
    x1, y1, x2, y2 = bbox_xyxy
    bh = y2 - y1
    bw = x2 - x1
    # Head region: top head_fraction of bbox height, with small margins
    head_y_max = y1 + bh * head_fraction
    x_margin = bw * 0.20   # 20% bbox-width margin on each horizontal side
    y_top_margin = bh * 0.12  # allow keypoints slightly above bbox top (head sticks out)

    for idx in [_KPT_NOSE, _KPT_LEYE, _KPT_REYE]:
        if kpts_conf[idx] < _MIN_KPT_CONF:
            continue  # low-confidence keypoint — ignore
        kx, ky = kpts_xy[idx]
        in_x = (x1 - x_margin) <= kx <= (x2 + x_margin)
        in_y = (y1 - y_top_margin) <= ky <= head_y_max
        if not in_x or not in_y:
            return False
    return True


def _try_embed_face(
    frame: np.ndarray,
    det: dict,
    embedder: FaceEmbedder,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Try to extract and embed the face from a single body detection.

    Only succeeds when nose AND both eyes are clearly visible (conf >= MIN_FACE_CONF)
    AND the face keypoints are inside this body's head region (contamination guard).
    No fallback to back-of-head crops.

    Returns (embedding, display_crop_bgr, face_visibility_score) or None.
    display_crop_bgr is a natural-orientation crop (NOT the aligned 112x112 used for ArcFace).
    """
    kpts_xy = det["kpts_xy"]
    kpts_conf = det["kpts_conf"]

    score = _face_visibility_score(kpts_conf)
    if score < MIN_FACE_CONF:
        return None  # face not visible — skip this frame

    # Reject frames where YOLO's face keypoints have leaked into another person's
    # face region (common when rank-1 FAR body is small and rank-0 NEAR body is large).
    if not _face_kpts_in_head_region(kpts_xy, kpts_conf, det["bbox_xyxy"]):
        return None

    # Display crop (natural orientation, square crop centered on nose/eyes).
    # Used for both human inspection AND ArcFace embedding, keeping the embedding
    # method consistent with enroll_player.py --crop (which also uses simple resize).
    #
    # Why not align_face_from_keypoints here:
    #   In a side-camera setup, YOLO's affine alignment distorts the face at angle,
    #   producing embeddings that do NOT match the enrolled templates (which were
    #   created from simple-resize crops).  Using the same method for both enrollment
    #   and identification is more important than using "perfect" alignment.
    display = _face_display_crop(frame, kpts_xy, kpts_conf, out_size=224)
    if display is None:
        return None

    embed_crop = cv2.resize(display, (112, 112), interpolation=cv2.INTER_LINEAR)
    if embed_crop.std() < 8.0:
        return None

    emb = embedder.embed(embed_crop)
    return emb, display, score


# ---------------------------------------------------------------------------
# Phase B — set-boundary face capture
# ---------------------------------------------------------------------------

def _collect_face_embeddings_in_window(
    video_path: str,
    t_start: float,
    t_end: float,
    yolo_model,
    embedder: FaceEmbedder,
    sample_fps: float = BOUNDARY_SAMPLE_FPS,
    roi_xyxy: Optional[tuple[float, float, float, float]] = None,
) -> list[dict]:
    """Sample frames in [t_start, t_end] and extract face embeddings.

    Returns list of:
        {
            "t_sec": float,
            "body_rank": int,      0 = largest (near), 1 = second-largest (far)
            "embedding": np.ndarray [512],
            "face_crop": np.ndarray [112, 112, 3],
            "face_score": float,
        }
    """
    results_out = []
    cap = cv2.VideoCapture(str(video_path))
    step_sec = 1.0 / sample_fps
    t = t_start
    while t <= t_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break
        dets = _detect_bodies_and_faces(frame, yolo_model, roi_xyxy=roi_xyxy)
        for rank, det in enumerate(dets[:2]):  # top 2 bodies (near + far)
            result = _try_embed_face(frame, det, embedder)
            if result is not None:
                emb, face_crop, face_score = result
                results_out.append({
                    "t_sec": t,
                    "body_rank": rank,
                    "embedding": emb,
                    "face_crop": face_crop,
                    "face_score": face_score,  # min(conf_nose, conf_leye, conf_reye)
                })
        t += step_sec
    cap.release()
    return results_out


# ---------------------------------------------------------------------------
# Phase C — jersey sampling (between rallies)
# ---------------------------------------------------------------------------

def _collect_jersey_hists_for_set(
    video_path: str,
    rally_t_ends: list[float],
    yolo_model,
    idle_offset_sec: float = 2.0,
    n_per_rally: int = 2,
    roi_xyxy: Optional[tuple[float, float, float, float]] = None,
) -> dict[str, list[np.ndarray]]:
    """Sample jersey histograms at idle moments (t_end + offset) within a set.

    Returns {"near": [...histograms...], "far": [...histograms...]}.
    """
    near_hists: list[np.ndarray] = []
    far_hists: list[np.ndarray] = []

    cap = cv2.VideoCapture(str(video_path))
    for t_end in rally_t_ends:
        for k in range(n_per_rally):
            t = t_end + idle_offset_sec + k * 0.5
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret:
                continue
            dets = _detect_bodies_and_faces(frame, yolo_model, roi_xyxy=roi_xyxy)
            if len(dets) >= 1:
                h = extract_jersey_hist(frame, dets[0]["bbox_xyxy"])
                if h is not None:
                    near_hists.append(h)
            if len(dets) >= 2:
                h = extract_jersey_hist(frame, dets[1]["bbox_xyxy"])
                if h is not None:
                    far_hists.append(h)
    cap.release()
    return {"near": near_hists, "far": far_hists}


# ---------------------------------------------------------------------------
# Phase D — face-to-jersey binding and identity resolution
# ---------------------------------------------------------------------------

def _resolve_identity_from_embeddings(
    face_results: list[dict],
    face_db: FaceDB,
    threshold: float,
) -> tuple[dict[int, tuple[PlayerRecord, float]], list[dict]]:
    """For each body rank (0=near, 1=far), aggregate embeddings and match DB.

    Returns:
        matched: {rank: (PlayerRecord, best_similarity)}
        unknowns: list of {rank, best_embedding, best_crop}
    """
    # Group embeddings by body rank
    by_rank: dict[int, list[dict]] = {0: [], 1: []}
    for r in face_results:
        rank = r["body_rank"]
        if rank in by_rank:
            by_rank[rank].append(r)

    matched: dict[int, tuple[PlayerRecord, float]] = {}
    unknowns: list[dict] = []

    for rank, detections in by_rank.items():
        if not detections:
            continue

        # Average all embeddings for a stable query embedding.
        embs = np.stack([d["embedding"] for d in detections], axis=0)  # [N, 512]
        avg_emb = embs.mean(axis=0)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-9)

        record = face_db.match(avg_emb, threshold=threshold)
        if record is not None:
            sim = face_similarity(avg_emb, record.embedding_array())
            matched[rank] = (record, sim)
        else:
            # Sort detections by face_score descending → pick top 3 for display.
            sorted_dets = sorted(detections, key=lambda d: d.get("face_score", 0.0), reverse=True)
            top3 = [(d.get("face_score", 0.0), d["t_sec"], d["face_crop"]) for d in sorted_dets[:3]]
            best = sorted_dets[0]
            unknowns.append({
                "rank": rank,
                "embedding": avg_emb,
                "face_crop": best["face_crop"],
                "face_score": best.get("face_score", 0.0),
                "t_sec": best["t_sec"],
                "n_samples": len(detections),
                "top_crops": top3,
            })

    return matched, unknowns


# ---------------------------------------------------------------------------
# Main identification pipeline
# ---------------------------------------------------------------------------

def run_player_identification(
    timeline: RallyTimeline,
    video_path: str,
    pose_weights_path: str,
    face_db: FaceDB,
    face_model_path: Optional[Path] = None,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    table_weights_path: Optional[str] = None,
    log_fn=None,
) -> IdentificationResult:
    """Scan the video to identify who is NEAR and FAR.

    Strategy:
    1. Find all set-boundary timestamps from the timeline.
    2. For each boundary, sample the 35 s side-swap walk window.
    3. Detect bodies + extract face embeddings using YOLO pose + ArcFace.
    4. Match embeddings against face_db.
    5. Independently extract jersey histograms from between-rally idle frames.
    6. Bind face identity to jersey; report unknowns for operator enrollment.

    Args:
        timeline:          RallyTimeline with set_number assigned.
        video_path:        Path to the (trimmed) working video.
        pose_weights_path: Path to YOLO pose .pt weights.
        face_db:           Loaded FaceDB (may be empty).
        face_model_path:   Path to w600k_r50.onnx. Defaults to data/models/face/w600k_r50.onnx.
        match_threshold:   Cosine similarity threshold for face matching.
        log_fn:            Optional callable(str) for progress logging.

    Returns:
        IdentificationResult with near_name, far_name, histograms, unknowns.
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    if face_model_path is None:
        face_model_path = Path(__file__).resolve().parent.parent / "data" / "models" / "face" / "w600k_r50.onnx"

    # Load models
    _log("[player_id] Loading ArcFace model...")
    try:
        embedder = FaceEmbedder(face_model_path)
    except FileNotFoundError as exc:
        _log(f"[player_id] SKIP — {exc}")
        return IdentificationResult(
            near_name=None, far_name=None,
            near_jersey_hist=None, far_jersey_hist=None,
            status="failed",
        )

    _log("[player_id] Loading YOLO pose model...")
    try:
        from ultralytics import YOLO
        yolo = YOLO(str(pose_weights_path))
    except Exception as exc:
        _log(f"[player_id] SKIP — YOLO load failed: {exc}")
        return IdentificationResult(
            near_name=None, far_name=None,
            near_jersey_hist=None, far_jersey_hist=None,
            status="failed",
        )

    pts = timeline.points
    if not pts:
        return IdentificationResult(
            near_name=None, far_name=None,
            near_jersey_hist=None, far_jersey_hist=None,
            status="failed",
        )

    # --- Detect table ROI + derive player zone ---
    # The player zone is the table bbox expanded by X+30%, Y+110% on each side;
    # only bodies whose center falls inside it are considered.  Prevents picking
    # up players at adjacent tables.
    table_roi = None
    roi = None
    if table_weights_path:
        _log("[player_id] Detecting table ROI (YOLOv8x-table)...")
        table_roi, roi = detect_table_roi_and_player_zone(video_path, table_weights_path)
        if table_roi is not None:
            _log(
                f"[player_id]   table ROI: x={table_roi.x} y={table_roi.y} "
                f"w={table_roi.w} h={table_roi.h}"
            )
            _log(f"[player_id]   player zone: x=[{roi[0]:.0f},{roi[2]:.0f}] y=[{roi[1]:.0f},{roi[3]:.0f}]")
        else:
            _log("[player_id]   table ROI detection failed — scanning full frame (may pick up adjacent tables)")
    else:
        _log("[player_id] No table weights provided — scanning full frame (may pick up adjacent tables)")

    # --- Phase B: build sampling windows ---
    #
    # Two independent capture strategies (body_rank flips after swap, so they cannot share a pool):
    #
    # Strategy for FAR player (rank=1 in set 1):
    #   Early window t=1s–15s at 4fps. FAR player faces the camera from match start.
    #   rank=1 (smaller bbox = far side) is the ONLY relevant rank here.
    #
    # Strategy for NEAR player (rank=0 in set 1):
    #   First half of the side-swap walk (~15s), rank=0 only.
    #   The near player starts walking toward the camera first, large bbox, face visible.
    #   We limit to the FIRST 15s of the swap walk because after that the two players
    #   may have crossed and rank=0 becomes the former FAR player.

    # Set-boundary timestamps (t_end of last rally in each set)
    swap_boundary_times: list[float] = []
    for i in range(len(pts) - 1):
        if pts[i + 1].set_number > pts[i].set_number:
            swap_boundary_times.append(pts[i].t_end)

    _log(f"[player_id] Swap windows: t={[f'{t:.0f}s' for t in swap_boundary_times]}")

    # --- Phase B: face embedding collection (split by role) ---
    #
    # FAR player (rank=1 in set 1): faces camera from match start.
    #   → rank=1 from early window (t=1–15s).
    #
    # NEAR player (rank=0 in set 1): faces camera only when walking FAR→NEAR.
    #   During every side-swap walk, one player walks FROM the far side (rank=1 initially,
    #   face toward camera) and one player walks FROM the near side (rank=0 initially,
    #   face away from camera).  In alternating swaps these roles flip:
    #     Swap 1: FAR player walks home (rank=1 approaching) → skip for NEAR
    #     Swap 2: NEAR player walks home (rank=1 approaching) → capture for NEAR
    #   We identify which swap is which by checking whether rank=1 matches the
    #   already-identified FAR player.  If it does → skip; if it doesn't → use.
    far_candidates: list[dict] = []
    near_candidates: list[dict] = []

    # FAR player: rank=1 from early window (faces camera consistently in set 1).
    _log("[player_id] Scanning t=1s–40s for FAR player (rank=1)...")
    early_results = _collect_face_embeddings_in_window(
        video_path, 1.0, 40.0, yolo, embedder, sample_fps=4.0, roi_xyxy=roi
    )
    early_far = [r for r in early_results if r["body_rank"] == 1]
    far_candidates.extend(early_far)
    _log(f"[player_id]   → {len(early_far)} FAR face embeddings (of {len(early_results)} total)")

    # Quick-resolve FAR player embedding for exclusion filtering.
    far_quick_emb: Optional[np.ndarray] = None
    far_quick_record: Optional[PlayerRecord] = None
    if far_candidates:
        _embs = np.stack([r["embedding"] for r in far_candidates])
        _avg = _embs.mean(axis=0)
        _avg = _avg / (np.linalg.norm(_avg) + 1e-9)
        far_quick_emb = _avg
        far_quick_record = face_db.match(_avg, threshold=match_threshold)
        if far_quick_record:
            _log(f"[player_id]   FAR quick-match: {far_quick_record.name}")

    # NEAR player (set 1): rank=0 from the full first-set window.
    # The NEAR player faces the camera only briefly during set 1 (looking sideways,
    # walking to position, service returns).  We scan the whole first set at 1 fps,
    # then filter out rank-flip contamination: whenever the FAR player temporarily
    # becomes rank=0 (moves closer), his embedding is very similar to his DB entry.
    # We discard any rank=0 embedding with similarity to the known FAR player >= threshold.
    t_set1_end = pts[-1].t_end  # fallback: last rally
    for p in pts:
        if p.set_number > 1:
            # use end of last rally in set 1 as the upper boundary
            set1_pts = [q for q in pts if q.set_number == 1]
            t_set1_end = set1_pts[-1].t_end if set1_pts else pts[-1].t_end
            break
    _log(f"[player_id] Scanning t=1s–{t_set1_end:.0f}s for NEAR player (rank=0, 1 fps)...")
    near_early_results = _collect_face_embeddings_in_window(
        video_path, 1.0, t_set1_end, yolo, embedder, sample_fps=1.0, roi_xyxy=roi
    )
    near_early_raw = [r for r in near_early_results if r["body_rank"] == 0]
    if far_quick_record is not None and near_early_raw:
        far_emb_arr = far_quick_record.embedding_array()
        early_near = [
            r for r in near_early_raw
            if face_similarity(r["embedding"], far_emb_arr) < match_threshold
        ]
        excluded = len(near_early_raw) - len(early_near)
        _log(f"[player_id]   → {len(early_near)} NEAR embeddings kept ({excluded} FAR-similar excluded)")
    else:
        early_near = near_early_raw
        _log(f"[player_id]   → {len(early_near)} NEAR face embeddings (no FAR filter applied)")
    near_candidates.extend(early_near)

    # Fallback: if no rank=0 face found early, scan set-start windows after each swap.
    # In even-indexed swaps (swap 1, 3, …): NEAR player returns to FAR side → rank=1 approaching.
    # We use exclusion: if rank=1 matches the known FAR player → skip; else → NEAR candidate.
    if not near_candidates:
        _log("[player_id] No early NEAR faces found; trying set-start windows as fallback...")
        for swap_idx, t_swap in enumerate(swap_boundary_times):
            t_start = t_swap + 12.0
            t_end = t_swap + 27.0
            _log(f"[player_id] Scanning set-start window {t_start:.0f}s–{t_end:.0f}s for NEAR player (rank=1)...")
            swap_results = _collect_face_embeddings_in_window(
                video_path, t_start, t_end, yolo, embedder, sample_fps=4.0, roi_xyxy=roi
            )
            swap_rank1 = [r for r in swap_results if r["body_rank"] == 1]
            _log(f"[player_id]   → {len(swap_rank1)} rank=1 embeddings (of {len(swap_results)} total)")

            if not swap_rank1:
                continue

            _swap_embs = np.stack([r["embedding"] for r in swap_rank1])
            _swap_avg = _swap_embs.mean(axis=0)
            _swap_avg = _swap_avg / (np.linalg.norm(_swap_avg) + 1e-9)

            if far_quick_record is not None:
                sim = face_similarity(_swap_avg, far_quick_record.embedding_array())
                _log(f"[player_id]   Similarity to known FAR player: {sim:.3f}")
                if sim >= match_threshold:
                    _log(f"[player_id]   → Skipping (known FAR player on far side again)")
                    continue
                _log(f"[player_id]   → Using as NEAR player candidates ({len(swap_rank1)} embeddings)")
            else:
                if swap_idx % 2 != 0:
                    _log(f"[player_id]   → Skipping (odd swap, FAR player likely on far side; no DB match)")
                    continue
                _log(f"[player_id]   → Using as NEAR player candidates (parity fallback)")

            near_candidates.extend(swap_rank1)

    _log(f"[player_id] Total: far={len(far_candidates)} near={len(near_candidates)} embeddings")

    # Reconstruct all_face_results with corrected role tags for _resolve_identity_from_embeddings
    # We override body_rank: far_candidates → rank=1, near_candidates → rank=0
    all_face_results: list[dict] = []
    for r in far_candidates:
        all_face_results.append({**r, "body_rank": 1})
    for r in near_candidates:
        all_face_results.append({**r, "body_rank": 0})

    if not all_face_results:
        _log("[player_id] No faces detected in any window.")
        return IdentificationResult(
            near_name=None, far_name=None,
            near_jersey_hist=None, far_jersey_hist=None,
            status="failed",
        )

    # --- Phase D: resolve identity ---
    matched, unknowns = _resolve_identity_from_embeddings(
        all_face_results, face_db, match_threshold
    )

    near_name: Optional[str] = None
    far_name: Optional[str] = None
    if 0 in matched:
        near_name = matched[0][0].name
        _log(f"[player_id] NEAR identified: {near_name} (sim={matched[0][1]:.3f})")
    if 1 in matched:
        far_name = matched[1][0].name
        _log(f"[player_id] FAR  identified: {far_name} (sim={matched[1][1]:.3f})")

    unknown_faces: list[UnknownFace] = []
    for u in unknowns:
        role = "near" if u["rank"] == 0 else "far"
        _log(
            f"[player_id] Unknown face detected for {role.upper()} — "
            f"best crop at t={u.get('t_sec', 0):.1f}s  "
            f"face_score={u.get('face_score', 0):.2f}  "
            f"n_samples={u.get('n_samples', 0)}"
        )
        unknown_faces.append(UnknownFace(
            body_role=role,
            boundary_sec=u.get("t_sec", 0.0),
            face_embedding=u["embedding"],
            best_crop_bgr=u["face_crop"],
            top_crops=u.get("top_crops", []),
        ))

    # --- Phase C: jersey extraction ---
    _log("[player_id] Extracting jersey colors...")
    near_hists_all: list[np.ndarray] = []
    far_hists_all: list[np.ndarray] = []

    set_numbers = sorted(set(p.set_number for p in pts))
    for sn in set_numbers:
        set_pts = [p for p in pts if p.set_number == sn]
        rally_ends = [p.t_end for p in set_pts]
        hists = _collect_jersey_hists_for_set(video_path, rally_ends, yolo, roi_xyxy=roi)
        near_hists_all.extend(hists["near"])
        far_hists_all.extend(hists["far"])

    near_jersey = _accumulate_jersey(near_hists_all)
    far_jersey = _accumulate_jersey(far_hists_all)

    # Check jersey ambiguity
    if near_jersey is not None and far_jersey is not None:
        dist = jersey_distance(near_jersey, far_jersey)
        if dist < JERSEY_AMBIGUOUS_THRESHOLD:
            _log(f"[player_id] WARNING: jersey colors too similar (dist={dist:.3f}) — may cause re-tracking errors")

    if near_jersey is not None:
        _log("[player_id] Jersey histograms: near=OK, far=" + ("OK" if far_jersey is not None else "MISSING"))

    # Build status
    if near_name is not None and far_name is not None:
        status = "identified"
    elif near_name is not None or far_name is not None:
        status = "partial"
    else:
        status = "failed"

    _log(f"[player_id] Result: status={status} near={near_name!r} far={far_name!r}")

    return IdentificationResult(
        near_name=near_name,
        far_name=far_name,
        near_jersey_hist=near_jersey,
        far_jersey_hist=far_jersey,
        status=status,
        unknown_faces=unknown_faces,
        table_roi=table_roi,
    )


# ---------------------------------------------------------------------------
# Phase D — set re-tracking by jersey (after side swap)
# ---------------------------------------------------------------------------

def resolve_near_far_by_jersey(
    frame: np.ndarray,
    near_jersey_hist: np.ndarray,
    far_jersey_hist: np.ndarray,
    yolo_model,
    roi_xyxy: Optional[tuple[float, float, float, float]] = None,
) -> Optional[tuple[str, str]]:
    """Given a new frame after a set boundary, determine which body is now NEAR/FAR.

    Compares detected body jerseys against saved histograms and returns the
    role assignment: ("near_player_name", "far_player_name") or None.

    Note: This function returns role order ("near_body_is_player_a": bool).
    The caller maps roles to player names using the session binding.
    """
    dets = _detect_bodies_and_faces(frame, yolo_model, roi_xyxy=roi_xyxy)
    if len(dets) < 2:
        return None

    h0 = extract_jersey_hist(frame, dets[0]["bbox_xyxy"])
    h1 = extract_jersey_hist(frame, dets[1]["bbox_xyxy"])
    if h0 is None or h1 is None:
        return None

    # Near player (largest bbox) vs player_a jersey
    dist_0_near = jersey_distance(h0, near_jersey_hist)
    dist_0_far = jersey_distance(h0, far_jersey_hist)

    # Assign: if largest body is closer to near_jersey → player_a is still NEAR
    if dist_0_near <= dist_0_far:
        return ("near", "far")   # largest body = near player, assignment unchanged
    else:
        return ("far", "near")   # largest body = far player, roles have swapped


# ---------------------------------------------------------------------------
# Standalone identification (no rally timeline required)
# ---------------------------------------------------------------------------

_EARLY_STOP_SIM: float = 0.55   # confidence threshold to stop scanning early
_CHUNK_SEC: float = 20.0         # scan in 20-second chunks
_MIN_EMBS_FOR_MATCH: int = 5     # minimum face embeddings before attempting a match


def _match_embedding_group(
    embs: list[dict],
    face_db: "FaceDB",
    match_threshold: float,
) -> tuple[Optional["PlayerRecord"], Optional[float]]:
    """Average a group of embeddings and match it against the face DB."""
    if len(embs) < _MIN_EMBS_FOR_MATCH:
        return None, None

    emb_arr = np.stack([r["embedding"] for r in embs])
    avg = emb_arr.mean(axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-9)
    record = face_db.match(avg, threshold=match_threshold)
    if record is None:
        return None, None
    sim = face_similarity(avg, record.embedding_array())
    return record, sim


def _scan_player_chunked(
    video_path: str,
    t_start: float,
    t_end: float,
    rank: int,
    yolo,
    embedder: "FaceEmbedder",
    face_db: "FaceDB",
    sample_fps: float = 4.0,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    early_stop_sim: float = _EARLY_STOP_SIM,
    exclude_record: Optional["PlayerRecord"] = None,
    dont_stop_on: Optional["PlayerRecord"] = None,
    roi_xyxy: Optional[tuple[float, float, float, float]] = None,
    log_fn=None,
) -> tuple:
    """Scan one player window in chunks; stop early if a confident match is found.

    Args:
        exclude_record: If set, embeddings similar to this player (>= match_threshold)
                        are filtered out before matching. Used to prevent Player 2's
                        window from matching Player 1 before the side swap.
        dont_stop_on: If set, never early-stop when the match is this player. Used
                      for Player 2 scan so it skips past the pre-swap period where
                      the accumulated average falsely converges toward Player 1.

    Returns:
        (all_embeddings, matched_record_or_None, matched_sim_or_None)
        all_embeddings: list of raw embedding dicts collected before stopping
                        (excludes frames dominated by exclude_record).
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    excl_emb: Optional[np.ndarray] = (
        exclude_record.embedding_array() if exclude_record is not None else None
    )

    all_embs: list[dict] = []   # candidate embeddings (exclude_record filtered out)
    matched_record = None
    matched_sim: Optional[float] = None

    t = t_start
    while t < t_end:
        chunk_end = min(t + _CHUNK_SEC, t_end)
        results = _collect_face_embeddings_in_window(
            video_path, t, chunk_end, yolo, embedder, sample_fps=sample_fps,
            roi_xyxy=roi_xyxy,
        )
        chunk_embs = [r for r in results if r["body_rank"] == rank]

        # Filter out embeddings that look like the already-identified player
        if excl_emb is not None:
            n_before = len(chunk_embs)
            chunk_embs = [
                r for r in chunk_embs
                if face_similarity(r["embedding"], excl_emb) < match_threshold
            ]
            n_excl = n_before - len(chunk_embs)
            if n_excl > 0:
                _log(f"[quick_id]   t={chunk_end:.0f}s: excluded {n_excl} frames similar to {exclude_record.name}")

        all_embs.extend(chunk_embs)

        if len(all_embs) >= _MIN_EMBS_FOR_MATCH:
            record, sim = _match_embedding_group(all_embs, face_db, match_threshold)
            if record is not None and sim is not None:
                _log(f"[quick_id]   t={chunk_end:.0f}s: {len(all_embs)} embs → {record.name} (sim={sim:.3f})")
                blocked = (
                    dont_stop_on is not None
                    and record.player_id == dont_stop_on.player_id
                )
                if sim >= early_stop_sim and not blocked:
                    matched_record = record
                    matched_sim = sim
                    _log(f"[quick_id]   → early stop: confident match for {record.name}")
                    break
                elif sim >= early_stop_sim and blocked:
                    _log(f"[quick_id]   → skipping early stop (match is Player 1, continuing scan)")
                    # Reset accumulated embeddings so we don't carry the pre-swap noise forward
                    all_embs = []
            else:
                _log(f"[quick_id]   t={chunk_end:.0f}s: {len(all_embs)} embs → no match yet")
        else:
            _log(f"[quick_id]   t={chunk_end:.0f}s: {len(all_embs)} candidate embs (need {_MIN_EMBS_FOR_MATCH} to match)")

        t = chunk_end

    # Final match attempt with all collected embeddings (if no early stop)
    if matched_record is None and len(all_embs) >= _MIN_EMBS_FOR_MATCH:
        record, sim = _match_embedding_group(all_embs, face_db, match_threshold)
        if record is not None:
            matched_sim = sim
            matched_record = record

    return all_embs, matched_record, matched_sim


def _scan_player_best_chunk_match(
    video_path: str,
    t_start: float,
    t_end: float,
    rank: int,
    yolo,
    embedder: "FaceEmbedder",
    face_db: "FaceDB",
    sample_fps: float = 2.0,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    early_stop_sim: float = _EARLY_STOP_SIM,
    exclude_record: Optional["PlayerRecord"] = None,
    reject_record: Optional["PlayerRecord"] = None,
    roi_xyxy: Optional[tuple[float, float, float, float]] = None,
    log_fn=None,
) -> tuple[list[dict], Optional["PlayerRecord"], Optional[float]]:
    """Scan independent chunks and keep the best non-rejected chunk-level match."""
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    excl_emb: Optional[np.ndarray] = (
        exclude_record.embedding_array() if exclude_record is not None else None
    )

    all_embs: list[dict] = []
    best_record: Optional["PlayerRecord"] = None
    best_sim: Optional[float] = None

    t = t_start
    while t < t_end:
        chunk_end = min(t + _CHUNK_SEC, t_end)
        results = _collect_face_embeddings_in_window(
            video_path, t, chunk_end, yolo, embedder, sample_fps=sample_fps,
            roi_xyxy=roi_xyxy,
        )
        chunk_embs = [r for r in results if r["body_rank"] == rank]

        if excl_emb is not None:
            n_before = len(chunk_embs)
            chunk_embs = [
                r for r in chunk_embs
                if face_similarity(r["embedding"], excl_emb) < match_threshold
            ]
            n_excl = n_before - len(chunk_embs)
            if n_excl > 0 and exclude_record is not None:
                _log(f"[quick_id]   t={chunk_end:.0f}s: excluded {n_excl} frames similar to {exclude_record.name}")

        all_embs.extend(chunk_embs)

        if len(chunk_embs) < _MIN_EMBS_FOR_MATCH:
            _log(f"[quick_id]   t={chunk_end:.0f}s: {len(chunk_embs)} candidate embs in chunk (need {_MIN_EMBS_FOR_MATCH})")
            t = chunk_end
            continue

        record, sim = _match_embedding_group(chunk_embs, face_db, match_threshold)
        if record is None or sim is None:
            _log(f"[quick_id]   t={chunk_end:.0f}s: {len(chunk_embs)} embs in chunk -> no match")
            t = chunk_end
            continue

        if reject_record is not None and record.player_id == reject_record.player_id:
            _log(f"[quick_id]   t={chunk_end:.0f}s: chunk rejected -> still collapses to {record.name} (sim={sim:.3f})")
            t = chunk_end
            continue

        _log(f"[quick_id]   t={chunk_end:.0f}s: chunk match -> {record.name} (sim={sim:.3f})")
        if best_record is None or best_sim is None or sim > best_sim:
            best_record = record
            best_sim = sim
        if sim >= early_stop_sim:
            _log(f"[quick_id]   -> early stop: confident chunk match for {record.name}")
            break
        t = chunk_end

    return all_embs, best_record, best_sim


def _best_unknown_face(embs: list[dict], role: str) -> Optional["UnknownFace"]:
    """Extract best face crop from unmatched embeddings for enrollment UI."""
    if not embs:
        return None
    best = max(embs, key=lambda r: r.get("face_score", 0.0))
    return UnknownFace(
        body_role=role,
        boundary_sec=best.get("t_sec", 0.0),
        face_embedding=best["embedding"],
        best_crop_bgr=best.get("face_crop"),
        top_crops=best.get("top_crops", []),
    )


def quick_identify_players_standalone(
    video_path: str,
    pose_weights_path: str,
    face_db: "FaceDB",
    face_model_path: Optional[Path] = None,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    table_weights_path: Optional[str] = None,
    log_fn=None,
) -> "IdentificationResult":
    """Identify both players from video without a rally timeline.

    Strategy:
    - Player 1 (starts FAR, faces camera from kick-off):
        scan rank=1, t=1–120s at 4 fps.  Stop early once a confident match is found.
    - Player 2 (starts NEAR, then swaps to FAR after ~120s and faces camera):
        scan rank=1, t=120–400s at 4 fps.  Stop early on confident match.

    Both windows use chunked scanning (20 s chunks) with early stop at
    sim >= _EARLY_STOP_SIM so we don't waste time after a confident match.

    If a player is detected but not in the DB, an UnknownFace is returned with
    a face crop so the operator can enroll them via the Web UI.

    Args:
        video_path:       Path to the raw (or trimmed) video.
        pose_weights_path: Path to YOLO pose .pt weights.
        face_db:          Loaded FaceDB (may be empty).
        face_model_path:  Path to w600k_r50.onnx. Defaults to data/models/face/.
        match_threshold:  Cosine similarity threshold for face matching.
        log_fn:           Optional callable(str) for progress logging.
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    if face_model_path is None:
        face_model_path = Path(__file__).resolve().parent.parent / "data" / "models" / "face" / "w600k_r50.onnx"

    _log("[quick_id] Loading ArcFace model...")
    try:
        embedder = FaceEmbedder(face_model_path)
    except (FileNotFoundError, RuntimeError) as exc:
        _log(f"[quick_id] SKIP — {exc}")
        return IdentificationResult(
            near_name=None, far_name=None,
            near_jersey_hist=None, far_jersey_hist=None,
            status="failed",
        )

    _log("[quick_id] Loading YOLO pose model...")
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required: torch.cuda.is_available() returned False")
        from ultralytics import YOLO
        yolo = YOLO(str(pose_weights_path))
    except Exception as exc:
        _log(f"[quick_id] SKIP — YOLO load failed: {exc}")
        return IdentificationResult(
            near_name=None, far_name=None,
            near_jersey_hist=None, far_jersey_hist=None,
            status="failed",
        )

    # ── Detect table ROI + derive player zone ────────────────────────────────
    # Player zone = table bbox + X+30%/Y+110% on each side; filters out players
    # at adjacent tables.  Also returned in the result so Step 3 can reuse the
    # Table ROI instead of running YOLOv8x-table a second time on the same video.
    table_roi = None
    roi_xyxy = None
    if table_weights_path:
        _log("[quick_id] Detecting table ROI (YOLOv8x-table)...")
        table_roi, roi_xyxy = detect_table_roi_and_player_zone(video_path, table_weights_path)
        if table_roi is not None:
            _log(
                f"[quick_id]   table ROI: x={table_roi.x} y={table_roi.y} "
                f"w={table_roi.w} h={table_roi.h}"
            )
            _log(
                f"[quick_id]   player zone: x=[{roi_xyxy[0]:.0f},{roi_xyxy[2]:.0f}] "
                f"y=[{roi_xyxy[1]:.0f},{roi_xyxy[3]:.0f}]"
            )
        else:
            _log("[quick_id]   table ROI detection failed — scanning full frame (may pick up adjacent tables)")
    else:
        _log("[quick_id] No table weights provided — scanning full frame (may pick up adjacent tables)")

    # ── Player 1: FAR in set 1, rank=1, t=1–120s ─────────────────────────────
    _log("[quick_id] Player 1 — FAR position, scanning t=1–120s (rank=1, 4 fps)...")
    p1_embs, p1_match, p1_sim = _scan_player_chunked(
        video_path, 1.0, 120.0, rank=1,
        yolo=yolo, embedder=embedder, face_db=face_db,
        sample_fps=4.0, match_threshold=match_threshold,
        roi_xyxy=roi_xyxy,
        log_fn=log_fn,
    )
    _log(f"[quick_id] Player 1: {len(p1_embs)} embeddings collected")
    if p1_match:
        _log(f"[quick_id] Player 1 identified: {p1_match.name} (sim={p1_sim:.3f})")
    else:
        _log(f"[quick_id] Player 1: no match found in DB — {'face detected, needs enrollment' if p1_embs else 'no face detected'}")

    # ── Player 2: NEAR in set 1 → FAR in set 2, rank=1, t=120–400s ───────────
    # Exclude Player 1's face so pre-swap frames (still showing Player 1 at rank=1)
    # don't contaminate the Player 2 average embedding.
    _log("[quick_id] Player 2 — NEAR→FAR swap zone, scanning t=120–400s (rank=1, 2 fps, excluding Player 1)...")
    p2_embs, p2_match, p2_sim = _scan_player_best_chunk_match(
        video_path, 120.0, 400.0, rank=1,
        yolo=yolo, embedder=embedder, face_db=face_db,
        sample_fps=2.0, match_threshold=match_threshold,
        exclude_record=p1_match,
        reject_record=p1_match,
        roi_xyxy=roi_xyxy,
        log_fn=log_fn,
    )
    _log(f"[quick_id] Player 2: {len(p2_embs)} embeddings collected")
    if p2_match:
        _log(f"[quick_id] Player 2 identified: {p2_match.name} (sim={p2_sim:.3f})")
    else:
        _log(f"[quick_id] Player 2: no match found in DB — {'face detected, needs enrollment' if p2_embs else 'no face detected'}")

    # ── Fallback: if Player 2 scan failed or returned the same match as Player 1,
    #    and there are exactly 2 players in the DB, the other must be Player 2.
    #    This handles the case where face similarity is too close to distinguish via
    #    embedding averaging (e.g. cross-sim > 0.5 between the two enrolled players).
    _log("[quick_id] Player 2 refinement â€” evaluating early-window and clean post-swap evidence...")
    p2_legacy_embs = list(p2_embs)
    p2_legacy_match = p2_match
    p2_legacy_sim = p2_sim

    p2_early_embs, p2_early_match, p2_early_sim = _scan_player_chunked(
        video_path, 1.0, 40.0, rank=0,
        yolo=yolo, embedder=embedder, face_db=face_db,
        sample_fps=2.0, match_threshold=match_threshold,
        roi_xyxy=roi_xyxy,
        log_fn=log_fn,
    )
    if p2_early_match is not None:
        _log(f"[quick_id] Player 2 early-window candidate: {p2_early_match.name} (sim={p2_early_sim:.3f})")
    else:
        _log(f"[quick_id] Player 2 early-window candidate: none â€” {'face detected, needs enrollment' if p2_early_embs else 'no face detected'}")

    p2_late_embs = list(p2_legacy_embs)
    p2_late_match = p2_legacy_match
    p2_late_sim = p2_legacy_sim
    if p2_late_match is not None:
        _log(f"[quick_id] Player 2 post-swap candidate: {p2_late_match.name} (sim={p2_late_sim:.3f})")
    else:
        _log(f"[quick_id] Player 2 post-swap candidate: none â€” {'face detected, needs enrollment' if p2_late_embs else 'no face detected'}")

    p2_embs = list(p2_early_embs) + list(p2_late_embs)
    p2_candidates: list[tuple[str, "PlayerRecord", float]] = []
    if p2_early_match is not None and p2_early_sim is not None:
        if p1_match is None or p2_early_match.player_id != p1_match.player_id:
            p2_candidates.append(("early_window", p2_early_match, p2_early_sim))
        else:
            _log("[quick_id] Player 2 early-window candidate rejected â€” same identity as Player 1")
    if p2_late_match is not None and p2_late_sim is not None:
        if p1_match is None or p2_late_match.player_id != p1_match.player_id:
            p2_candidates.append(("post_swap", p2_late_match, p2_late_sim))
        else:
            _log("[quick_id] Player 2 post-swap candidate rejected â€” same identity as Player 1")

    if p2_candidates:
        source, p2_match, p2_sim = max(p2_candidates, key=lambda item: item[2])
        _log(f"[quick_id] Player 2 refined selection from {source}: {p2_match.name} (sim={p2_sim:.3f})")
    else:
        p2_match = p2_legacy_match
        p2_sim = p2_legacy_sim

    if p1_match is not None and (
        p2_match is None or p2_match.player_id == p1_match.player_id
    ):
        p2_match = None
        p2_sim = None
        _log("[quick_id] Player 2 unresolved after refinement - leaving as unknown (no fallback guessing)")

    # ── Map to near/far names (Player 1 = FAR in set 1, Player 2 = NEAR in set 1)
    far_name:  Optional[str] = p1_match.name if p1_match else None
    near_name: Optional[str] = p2_match.name if p2_match else None

    # ── Collect unknown faces for enrollment UI ───────────────────────────────
    unknown_faces: list[UnknownFace] = []
    if p1_match is None:
        uf = _best_unknown_face(p1_embs, role="far")
        if uf:
            unknown_faces.append(uf)
            _log(f"[quick_id] Unknown FAR player — face crop extracted for enrollment")
        else:
            _log(f"[quick_id] Unknown FAR player — no face crop available")
    if p2_match is None:
        uf = _best_unknown_face(p2_embs, role="near")
        if uf:
            unknown_faces.append(uf)
            _log(f"[quick_id] Unknown NEAR player — face crop extracted for enrollment")
        else:
            _log(f"[quick_id] Unknown NEAR player — no face crop available")

    status = (
        "identified" if (near_name is not None and far_name is not None)
        else ("partial" if (near_name is not None or far_name is not None)
        else "failed")
    )
    _log(f"[quick_id] Result: status={status}  FAR(player1)={far_name!r}  NEAR(player2)={near_name!r}")

    return IdentificationResult(
        near_name=near_name,
        far_name=far_name,
        near_jersey_hist=None,
        far_jersey_hist=None,
        status=status,
        unknown_faces=unknown_faces,
        table_roi=table_roi,
        player_zone_xyxy=roi_xyxy,
    )
