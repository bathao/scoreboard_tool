# ai_table_roi_dl.py
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional, Tuple, List

import cv2
import numpy as np

from backend.ai_table_roi import TableROI


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class DLConfig:
    """
    Table ROI detection for diagonal tripod table-tennis clips (multi-table scenes).

    Robust selection:
      1) YOLO proposes coarse "table vicinity" boxes.
      2) Refine inside each proposal using:
         - HSV blue segmentation (table top)
         - contour -> rotated rectangle (minAreaRect)
         - edge evidence (Canny) to favor tabletop with white lines
         - geometric checks (aspect, position, size)
         - EXTRA: shrink-top using horizontal edge profile to remove blue barriers
      3) Multi-frame consensus + strong "main table prior" (closest to camera center).

    Output stays backward-compatible: axis-aligned bbox TableROI(x,y,w,h).
    """
    weights_path: str
    conf_thres: float = 0.10
    iou_thres: float = 0.45
    device: str = "cuda"

    # Optional: if your YOLO model has a specific "table" class id(s), set it here.
    # Example: table_class_ids=(3,)  # if class 3 means table in your custom model
    table_class_ids: Optional[Tuple[int, ...]] = None

    # If using COCO-like model, person is class 0; we skip it if table_class_ids is None.
    person_class_id: int = 0

    # Vertical search band (avoid ceiling banners & too much floor)
    y_band_min_ratio: float = 0.18
    y_band_max_ratio: float = 0.92

    # Expand search region around YOLO box so refinement can recover edges
    # when YOLO only covers part of the tabletop.
    search_expand_left: float = 0.0
    search_expand_right: float = 0.0
    search_expand_up: float = 0.65
    search_expand_down: float = 0.65

    # HSV blue range (wide enough for lighting variations)
    hsv_lower: Tuple[int, int, int] = (80, 30, 20)
    hsv_upper: Tuple[int, int, int] = (145, 255, 255)

    # Morphology
    morph_kernel: int = 5
    morph_close_iter: int = 2
    morph_open_iter: int = 1

    # Candidate filters (relative to refined search region)
    min_contour_area_ratio: float = 0.05
    min_aspect: float = 1.25
    max_aspect: float = 8.0
    min_y_pos: float = 0.30  # tabletop shouldn't be too high in the search window

    # Scoring weights INSIDE refinement (blue+rect+edges)
    w_area: float = 0.50
    w_edge: float = 0.25
    w_center_x: float = 0.15
    w_center_y: float = 0.10

    # Main table prior target (normalized center expectation)
    # For diagonal tripod, main table often sits slightly right and lower-middle.
    target_cx: float = 0.52
    target_cy: float = 0.62

    # Limit aggressive top shrink so the far edge of the table is preserved.
    top_shrink_margin_ratio: float = 0.02
    top_shrink_max_ratio: float = 1.0

    # Final output expansion to recover a little more space above the table.
    final_expand_top_ratio: float = 0.02


# ----------------------------
# Utilities
# ----------------------------
def _try_import_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except ImportError:
        return None


def _clip_box(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a2x, a2y = ax + aw, ay + ah
    b2x, b2y = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(a2x, b2x), min(a2y, b2y)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _save_debug_image(debug: bool, path: str, img: np.ndarray) -> None:
    if not debug:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)
    except Exception:
        pass


def _normalize01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _side_continuation_score(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    *,
    cfg: DLConfig,
    side: str,
) -> float:
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    band_w = max(8, int(w * 0.15))
    y1 = max(0, y + int(h * 0.15))
    y2 = min(H, y + int(h * 0.75))
    if y2 <= y1:
        return 0.0

    if side == "left":
        x1 = max(0, x - band_w)
        x2 = x
    elif side == "right":
        x1 = x + w
        x2 = min(W, x + w + band_w)
    else:
        raise ValueError(f"Invalid side: {side}")

    if x2 <= x1:
        return 0.0

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    blue_density = float(np.count_nonzero(mask)) / float(mask.size)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    return blue_density * 0.8 + edge_density * 0.2


def _aspect_shape_prior_score(bbox: Tuple[int, int, int, int]) -> float:
    """
    Soft prior for axis-aligned tabletop bbox shape.

    Truncated table boxes tend to become too narrow after refinement. We keep
    this prior broad so it still generalizes across clips and small camera shifts.
    """
    _x, _y, w, h = bbox
    aspect = float(w) / max(1.0, float(h))
    target = 3.2
    tolerance = 0.85
    z = (aspect - target) / tolerance
    return float(np.exp(-(z * z)))


def _center_table_prior_score(
    bbox: Tuple[int, int, int, int],
    *,
    frame_w: int,
    frame_h: int,
    target_cx: float,
    target_cy: float,
) -> float:
    """
    Score 0..1: higher means bbox center matches "main table" location in camera view.
    Includes strong edge penalties to suppress side tables near borders.
    """
    x, y, w, h = bbox
    cx = (x + w / 2.0) / max(1.0, float(frame_w))
    cy = (y + h / 2.0) / max(1.0, float(frame_h))

    # Elliptical distance (normalized tolerances)
    dx = (cx - target_cx) / 0.35
    dy = (cy - target_cy) / 0.28
    dist2 = dx * dx + dy * dy

    score = float(np.exp(-dist2))  # 0..1

    # Edge penalty: side tables often touch borders
    left = x / max(1.0, float(frame_w))
    right = (x + w) / max(1.0, float(frame_w))
    if left < 0.03 or right > 0.97:
        score *= 0.20
    elif left < 0.06 or right > 0.94:
        score *= 0.55

    return float(_normalize01(score))


def _shrink_top_using_horizontal_edge(
    roi_gray: np.ndarray,
    bbox_local: Tuple[int, int, int, int],
    *,
    top_shrink_margin_ratio: float,
    top_shrink_max_ratio: float,
) -> Tuple[int, int, int, int]:
    """
    Reduce bbox top if it includes background (blue barriers).

    We detect the strongest horizontal edge row (table far edge) within the upper
    half of the bbox and move bbox.y downward to that edge (+small margin).
    """
    bx, by, bw, bh = bbox_local
    H, W = roi_gray.shape[:2]

    bx = max(0, min(bx, W - 1))
    by = max(0, min(by, H - 1))
    bw = max(1, min(bw, W - bx))
    bh = max(1, min(bh, H - by))

    crop = roi_gray[by : by + bh, bx : bx + bw]
    if crop.size == 0 or crop.shape[0] < 20:
        return bx, by, bw, bh

    # Vertical derivative (Sobel-Y) => strong at horizontal edges
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    abs_gy = np.abs(gy)

    row_sum = abs_gy.sum(axis=1)

    # Search only upper ~55% for the tabletop far edge
    search_end = int(len(row_sum) * 0.55)
    if search_end < 10:
        return bx, by, bw, bh

    segment = row_sum[:search_end]
    peak = int(np.argmax(segment))

    # If peak is too close to top, likely noise
    if peak < int(bh * 0.08):
        return bx, by, bw, bh

    margin = int(max(2, bh * float(top_shrink_margin_ratio)))
    new_by = by + peak + margin
    new_bh = (by + bh) - new_by

    # Prevent over-shrinking: keep at least ~50% height
    if new_bh < int(bh * 0.50):
        return bx, by, bw, bh
    if (new_by - by) > int(bh * float(top_shrink_max_ratio)):
        return bx, by, bw, bh

    return bx, new_by, bw, new_bh


# ----------------------------
# Refinement: tabletop inside YOLO proposal
# ----------------------------
class _Candidate:
    __slots__ = ("bbox", "score", "yolo_conf")

    def __init__(self, bbox: Tuple[int, int, int, int], score: float, yolo_conf: float):
        self.bbox = bbox
        self.score = score
        self.yolo_conf = yolo_conf


def _refine_tabletop_in_search(
    frame: np.ndarray,
    search_area: Tuple[int, int, int, int],
    *,
    cfg: DLConfig,
    debug: bool = False,
    debug_tag: str = "roi",
) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
    """
    Return (bbox, refine_score) in FULL-FRAME coords.
    refine_score is an evidence score from blue+geometry+edges (not probability).
    """
    H, W = frame.shape[:2]
    x, y, w, h = search_area
    x, y, w, h = _clip_box(int(x), int(y), int(w), int(h), W, H)

    y_band_min = int(H * cfg.y_band_min_ratio)
    y_band_max = int(H * cfg.y_band_max_ratio)

    search_x1 = max(0, x - int(w * cfg.search_expand_left))
    search_x2 = min(W, x + w + int(w * cfg.search_expand_right))
    search_y1 = max(y_band_min, y - int(h * cfg.search_expand_up))
    search_y2 = min(y_band_max, y + int(h * cfg.search_expand_down))
    if search_x2 <= search_x1 or search_y2 <= search_y1:
        return None

    roi_zone = frame[search_y1:search_y2, search_x1:search_x2]
    if roi_zone.size == 0:
        return None

    hsv = cv2.cvtColor(roi_zone, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    k = max(3, int(cfg.morph_kernel) | 1)  # make odd
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(cfg.morph_close_iter))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(cfg.morph_open_iter))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        _save_debug_image(debug, f"matches/_debug/{debug_tag}_mask_none.png", mask)
        return None

    roi_h, roi_w = roi_zone.shape[:2]
    roi_area = float(roi_w * roi_h)

    gray = cv2.cvtColor(roi_zone, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    best_score = -1.0
    best_bbox_local: Optional[Tuple[int, int, int, int]] = None
    best_poly: Optional[np.ndarray] = None

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < roi_area * cfg.min_contour_area_ratio:
            continue

        rect = cv2.minAreaRect(cnt)  # rotated rectangle
        (cx, cy), (rw, rh), _ang = rect
        rw, rh = float(rw), float(rh)
        if rw < 2 or rh < 2:
            continue

        long_side = max(rw, rh)
        short_side = min(rw, rh)
        aspect = long_side / max(short_side, 1e-6)
        if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
            continue

        y_pos = cy / float(roi_h)
        if y_pos < cfg.min_y_pos:
            continue

        poly = cv2.boxPoints(rect).astype(np.int32)
        bx, by, bw, bh = cv2.boundingRect(poly)
        if bw <= 0 or bh <= 0:
            continue

        bx = max(0, min(bx, roi_w - 1))
        by = max(0, min(by, roi_h - 1))
        bw = max(1, min(bw, roi_w - bx))
        bh = max(1, min(bh, roi_h - by))

        edge_roi = edges[by : by + bh, bx : bx + bw]
        edge_density = float(np.count_nonzero(edge_roi)) / float(edge_roi.size)

        # centrality inside roi_zone (weak prior)
        x_center_score = 1.0 - abs(cx - (roi_w / 2.0)) / (roi_w / 2.0)
        x_center_score = _normalize01(x_center_score)

        # slightly prefer lower-than-center to reduce barrier/background selection
        target_y = roi_h * 0.62
        y_center_score = 1.0 - abs(cy - target_y) / (roi_h * 0.62)
        y_center_score = _normalize01(y_center_score)

        area_score = area / roi_area
        score = (
            area_score * cfg.w_area
            + edge_density * cfg.w_edge
            + x_center_score * cfg.w_center_x
            + y_center_score * cfg.w_center_y
        )

        # Generic barrier-strip penalty (high + short)
        if by < int(roi_h * 0.15) and bh < int(roi_h * 0.30):
            score *= 0.80

        if score > best_score:
            best_score = score
            best_bbox_local = (bx, by, bw, bh)
            best_poly = poly

    if best_bbox_local is None:
        _save_debug_image(debug, f"matches/_debug/{debug_tag}_mask_no_candidate.png", mask)
        return None

    # --- EXTRA: shrink top to remove blue barrier region ---
    bx, by, bw, bh = best_bbox_local
    bx, by, bw, bh = _shrink_top_using_horizontal_edge(
        gray,
        (bx, by, bw, bh),
        top_shrink_margin_ratio=cfg.top_shrink_margin_ratio,
        top_shrink_max_ratio=cfg.top_shrink_max_ratio,
    )
    best_bbox_local = (bx, by, bw, bh)

    out_x = search_x1 + bx
    out_y = search_y1 + by
    out_w = bw
    out_h = bh
    out_x, out_y, out_w, out_h = _clip_box(out_x, out_y, out_w, out_h, W, H)

    if debug:
        dbg = roi_zone.copy()
        if best_poly is not None:
            cv2.drawContours(dbg, [best_poly], -1, (0, 255, 0), 2)
        cv2.rectangle(dbg, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
        cv2.putText(
            dbg,
            f"refine={best_score:.3f} (top_shrunk)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        _save_debug_image(debug, f"matches/_debug/{debug_tag}_roi_zone.png", dbg)
        _save_debug_image(debug, f"matches/_debug/{debug_tag}_mask.png", mask)
        _save_debug_image(debug, f"matches/_debug/{debug_tag}_edges.png", edges)

    return (out_x, out_y, out_w, out_h), float(best_score)


# ----------------------------
# Public API
# ----------------------------
def detect_table_roi_dl(
    video_path: str,
    *,
    cfg: Optional[DLConfig] = None,
    sample_time_sec: float = 2.0,
    max_frames: int = 5,
    debug: bool = False,
) -> TableROI:
    """
    Detect the MAIN tabletop ROI from a table tennis clip that may contain multiple tables.

    The "main table" is defined as:
      - closest to expected camera center (strong prior),
      - strong tabletop evidence (blue + edges + geometry),
      - stable across sampled frames (consensus).

    Returns: TableROI(x,y,w,h,confidence,method,notes)
    """
    YOLO = _try_import_ultralytics()
    if YOLO is None or cfg is None:
        raise RuntimeError("CRITICAL: ultralytics not installed or cfg is None.")

    model = YOLO(cfg.weights_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"CRITICAL: Cannot open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)

    # Sample multiple times: robust against occlusion / motion
    base = float(sample_time_sec)
    check_times = [max(0.0, base), base + 2.0, base + 4.0, base + 6.0, base + 8.0, base + 10.0]
    check_times = check_times[:max_frames]

    candidates: List[_Candidate] = []

    def maybe_recover_truncated_bbox(
        frame: np.ndarray,
        yolo_bbox: Tuple[int, int, int, int],
        bbox: Tuple[int, int, int, int],
        refine_score: float,
    ) -> Tuple[Tuple[int, int, int, int], float]:
        yx, yy, yw, yh = yolo_bbox
        bx, by, bw, bh = bbox
        aspect = bw / max(1.0, float(bh))
        left_touch = abs(bx - yx) <= max(6, int(yw * 0.02))
        right_touch = abs((bx + bw) - (yx + yw)) <= max(6, int(yw * 0.02))
        orig_shape = _aspect_shape_prior_score(bbox)

        def recovery_rank(candidate_bbox: Tuple[int, int, int, int], candidate_score: float) -> float:
            shape = _aspect_shape_prior_score(candidate_bbox)
            center = _center_table_prior_score(
                candidate_bbox,
                frame_w=W,
                frame_h=H,
                target_cx=cfg.target_cx,
                target_cy=cfg.target_cy,
            )
            return shape * 0.55 + candidate_score * 0.20 + center * 0.25

        # If refinement stays glued to a YOLO edge and the tabletop aspect is
        # unnaturally narrow, YOLO likely clipped that side of the table.
        repair_left = aspect < 2.95 and left_touch and not right_touch
        repair_right = aspect < 2.95 and right_touch and not left_touch
        if aspect < 2.95 and left_touch and right_touch:
            left_score = _side_continuation_score(frame, yolo_bbox, cfg=cfg, side="left")
            right_score = _side_continuation_score(frame, yolo_bbox, cfg=cfg, side="right")
            if left_score > right_score * 1.10:
                repair_left = True
            elif right_score > left_score * 1.10:
                repair_right = True
            else:
                repair_left = True
                repair_right = True
        if not repair_left and not repair_right:
            return bbox, refine_score

        candidates: List[Tuple[Tuple[int, int, int, int], float]] = [(bbox, refine_score)]
        if repair_left:
            repaired = _refine_tabletop_in_search(
                frame,
                yolo_bbox,
                cfg=replace(cfg, search_expand_left=max(cfg.search_expand_left, 0.15)),
                debug=False,
                debug_tag="repair_left",
            )
            if repaired is not None:
                candidates.append(repaired)
        if repair_right:
            repaired = _refine_tabletop_in_search(
                frame,
                yolo_bbox,
                cfg=replace(cfg, search_expand_right=max(cfg.search_expand_right, 0.15)),
                debug=False,
                debug_tag="repair_right",
            )
            if repaired is not None:
                candidates.append(repaired)

        best_bbox, best_score = bbox, refine_score
        best_rank = recovery_rank(bbox, refine_score)
        for candidate_bbox, candidate_score in candidates[1:]:
            candidate_aspect = candidate_bbox[2] / max(1.0, float(candidate_bbox[3]))
            candidate_shape = _aspect_shape_prior_score(candidate_bbox)
            improved_shape = candidate_aspect > max(2.95, aspect + 0.20)
            strong_shape_gain = candidate_shape >= (orig_shape + 0.08)
            not_much_worse = candidate_score >= (refine_score - 0.04)
            rank = recovery_rank(candidate_bbox, candidate_score)
            if improved_shape and strong_shape_gain and not_much_worse and rank > best_rank:
                best_bbox, best_score, best_rank = candidate_bbox, candidate_score, rank

        if best_bbox != bbox:
            return best_bbox, best_score
        return bbox, refine_score

    for ts in check_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(
            frame,
            conf=cfg.conf_thres,
            iou=cfg.iou_thres,
            device=cfg.device,
            verbose=False,
        )
        if not results or len(results[0].boxes) == 0:
            continue

        for bi, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            yolo_conf = float(box.conf[0])

            # Class filtering
            if cfg.table_class_ids is not None:
                if cls_id not in cfg.table_class_ids:
                    continue
            else:
                # fallback: only skip person
                if cls_id == cfg.person_class_id:
                    continue

            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            bw_ai = max(1, x2i - x1i)
            bh_ai = max(1, y2i - y1i)

            # quick reject tiny boxes
            if bw_ai < int(W * 0.15) or bh_ai < int(H * 0.06):
                continue

            refined = _refine_tabletop_in_search(
                frame,
                (x1i, y1i, bw_ai, bh_ai),
                cfg=cfg,
                debug=debug,
                debug_tag=f"t{int(ts)}_b{bi}",
            )
            if refined is None:
                continue

            bbox, refine_score = refined
            bbox, refine_score = maybe_recover_truncated_bbox(
                frame,
                (x1i, y1i, bw_ai, bh_ai),
                bbox,
                refine_score,
            )
            bx, by, bw, bh = bbox

            # Evidence score from refinement (soft-normalized)
            refine_soft = float(_normalize01(refine_score * 1.6))

            # Strong main-table prior (center of camera + edge penalty)
            center_prior = _center_table_prior_score(
                bbox,
                frame_w=W,
                frame_h=H,
                target_cx=cfg.target_cx,
                target_cy=cfg.target_cy,
            )

            # Size prior: main table usually bigger than side tables
            area_ratio = (bw * bh) / float(max(1, W * H))
            size_prior = float(_normalize01(area_ratio * 6.0))

            # FINAL combined score: center dominates (multi-table selection)
            combined = (
                refine_soft * 0.40 +
                center_prior * 0.40 +
                size_prior * 0.10 +
                yolo_conf * 0.10
            )

            candidates.append(_Candidate(bbox=bbox, score=combined, yolo_conf=yolo_conf))

        if debug and candidates:
            top = max(candidates, key=lambda c: c.score)
            dbg = frame.copy()
            x, y, w, h = top.bbox
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(
                dbg,
                f"TOP score={top.score:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            _save_debug_image(debug, f"matches/_debug/frame_t{int(ts)}.png", dbg)

    cap.release()

    if not candidates:
        raise RuntimeError(
            "CRITICAL: Failed to locate tabletop ROI. "
            "Check YOLO class filtering (table_class_ids), lighting (HSV), or conf thresholds."
        )

    # ----------------------------
    # Multi-frame consensus:
    #   1) compute median bbox
    #   2) choose candidate closest to median (stability) while honoring center prior
    # ----------------------------
    xs = np.array([c.bbox[0] for c in candidates], dtype=np.float32)
    ys = np.array([c.bbox[1] for c in candidates], dtype=np.float32)
    ws = np.array([c.bbox[2] for c in candidates], dtype=np.float32)
    hs = np.array([c.bbox[3] for c in candidates], dtype=np.float32)
    med_bbox = (int(np.median(xs)), int(np.median(ys)), int(np.median(ws)), int(np.median(hs)))

    best = None
    best_rank = -1.0
    for c in candidates:
        iou_to_med = _iou(c.bbox, med_bbox)
        center_prior = _center_table_prior_score(
            c.bbox, frame_w=W, frame_h=H, target_cx=cfg.target_cx, target_cy=cfg.target_cy
        )
        shape_prior = _aspect_shape_prior_score(c.bbox)
        # Final decision: stability + center + evidence + shape.
        # Shape prior helps reject candidates that look horizontally truncated.
        rank = (
            iou_to_med * 0.25 +
            center_prior * 0.25 +
            c.score * 0.25 +
            shape_prior * 0.25
        )
        if rank > best_rank:
            best_rank = rank
            best = c

    assert best is not None
    bx, by, bw, bh = _clip_box(best.bbox[0], best.bbox[1], best.bbox[2], best.bbox[3], W, H)
    expand_top = int(round(bh * cfg.final_expand_top_ratio))
    if expand_top > 0:
        new_y = max(0, by - expand_top)
        bh = min(H - new_y, bh + (by - new_y))
        by = new_y

    return TableROI(
        x=bx,
        y=by,
        w=bw,
        h=bh,
        confidence=float(best.yolo_conf),
        method="yolo+blue+rect+edge+center+consensus+topshrink",
        notes=(
            f"score={best.score:.3f}; "
            f"target_center=({cfg.target_cx:.2f},{cfg.target_cy:.2f}); "
            f"hsv={cfg.hsv_lower}-{cfg.hsv_upper}; "
            f"final_expand_top={cfg.final_expand_top_ratio:.2f}"
        ),
    )
