# backend/ai_table_roi.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class TableROI:
    """
    ROI represented as (x, y, w, h) in full-frame coordinates.
    """
    x: int
    y: int
    w: int
    h: int
    confidence: float
    method: str = "classical"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TableROI":
        return TableROI(
            x=int(d["x"]),
            y=int(d["y"]),
            w=int(d["w"]),
            h=int(d["h"]),
            confidence=float(d.get("confidence", 0.0)),
            method=str(d.get("method", "classical")),
            notes=str(d.get("notes", "")),
        )

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def save_table_roi(path: Path, roi: TableROI) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(roi.to_dict(), f, ensure_ascii=False, indent=2)


def load_table_roi(path: Path) -> TableROI:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return TableROI.from_dict(d)


def _largest_contour_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Return bbox of largest contour as (x,y,w,h,area_ratio).
    area_ratio = contour_area / image_area
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    areas = [cv2.contourArea(c) for c in contours]
    idx = int(np.argmax(areas))
    c = contours[idx]
    x, y, w, h = cv2.boundingRect(c)
    img_area = float(mask.shape[0] * mask.shape[1])
    area_ratio = float(areas[idx] / max(1.0, img_area))
    return (x, y, w, h, area_ratio)


def detect_table_roi_classical(
    video_path: str,
    *,
    sample_time_sec: float = 1.0,
    max_frames: int = 5,
    debug: bool = False,
) -> TableROI:
    """
    Heuristic table ROI detection for fixed camera:
    - Sample a few frames early in the video
    - Use color clustering-ish approach in HSV to find dominant 'table-like' region
    - Pick largest stable contour bbox

    This is a baseline fallback when DL is not available.
    Works best if table color contrasts with surroundings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 60.0

    frame_idx0 = int(round(sample_time_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx0)

    bboxes = []
    notes = []

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Downscale for speed
        scale = 0.5
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Try generic table color ranges (blue/green common).
        # We'll take union masks and choose largest contour.
        # You may refine later per your tournament table color.
        mask_blue1 = cv2.inRange(hsv, (80, 40, 40), (140, 255, 255))
        mask_green = cv2.inRange(hsv, (35, 35, 35), (85, 255, 255))
        mask = cv2.bitwise_or(mask_blue1, mask_green)

        # Morphology cleanup
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        bbox = _largest_contour_bbox(mask)
        if bbox is None:
            notes.append("no contour")
            continue

        x, y, bw, bh, area_ratio = bbox

        # Scale bbox back to full res
        x = int(round(x / scale))
        y = int(round(y / scale))
        bw = int(round(bw / scale))
        bh = int(round(bh / scale))

        bboxes.append((x, y, bw, bh, area_ratio))

        if debug:
            print(f"[roi-classical] frame {i}: bbox=({x},{y},{bw},{bh}) area_ratio={area_ratio:.3f}")

    cap.release()

    if not bboxes:
        return TableROI(x=0, y=0, w=0, h=0, confidence=0.0, method="classical", notes="failed: no bbox")

    # Aggregate: choose bbox with max area_ratio; you can also median-filter
    best = max(bboxes, key=lambda t: t[4])
    x, y, bw, bh, area_ratio = best

    # Confidence: simple heuristic based on area ratio (table should be non-trivial)
    # Typical view: table might occupy 10%~40% of frame depending on zoom.
    conf = float(np.clip((area_ratio - 0.03) / 0.20, 0.0, 1.0))

    # Clamp bbox within bounds
    # (We don't know original w/h here, so keep positive only)
    if bw <= 0 or bh <= 0:
        conf = 0.0

    note = "ok" if conf >= 0.5 else "low_confidence; consider DL or manual ROI"
    if notes:
        note += f"; samples_notes={notes[:3]}"

    return TableROI(x=x, y=y, w=bw, h=bh, confidence=conf, method="classical", notes=note)