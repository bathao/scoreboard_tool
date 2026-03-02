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
    Includes logic to expand into a Unified Play Zone for motion analysis.
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

    def get_unified_play_zone(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """
        Calculates an expanded ROI that includes the table and the players' 
        active movement area. This significantly stabilizes rally detection.
        
        Expansion Strategy:
        - UP: +100% of table height (to catch players' bodies/heads)
        - DOWN: +20% of table height (to catch footwork near the table)
        - SIDES: +25% of table width (to catch wide shots/lateral movement)
        """
        # Vertical expansion
        new_y = max(0, self.y - int(self.h * 1.0))
        new_h = min(frame_h - new_y, self.h + int(self.h * 1.2))
        
        # Horizontal expansion
        new_x = max(0, self.x - int(self.w * 0.25))
        new_w = min(frame_w - new_x, self.w + int(self.w * 0.5))
        
        return (int(new_x), int(new_y), int(new_w), int(new_h))


def save_table_roi(path: Path, roi: TableROI) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(roi.to_dict(), f, ensure_ascii=False, indent=2)


def load_table_roi(path: Path) -> TableROI:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return TableROI.from_dict(d)


def _largest_contour_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
    """Return bbox of largest contour as (x,y,w,h,area_ratio)."""
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
    """Baseline fallback using HSV color filtering for table detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frame_idx0 = int(round(sample_time_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx0)

    bboxes = []
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        scale = 0.5
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        mask_blue = cv2.inRange(hsv, (80, 40, 40), (140, 255, 255))
        mask_green = cv2.inRange(hsv, (35, 35, 35), (85, 255, 255))
        mask = cv2.bitwise_or(mask_blue, mask_green)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        bbox = _largest_contour_bbox(mask)
        if bbox:
            x, y, bw, bh, area_ratio = bbox
            bboxes.append((int(x/scale), int(y/scale), int(bw/scale), int(bh/scale), area_ratio))

    cap.release()
    if not bboxes:
        return TableROI(0, 0, 0, 0, 0.0, notes="failed: no bbox")

    best = max(bboxes, key=lambda t: t[4])
    x, y, bw, bh, area_ratio = best
    conf = float(np.clip((area_ratio - 0.03) / 0.20, 0.0, 1.0))
    return TableROI(x=x, y=y, w=bw, h=bh, confidence=conf, method="classical")