from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.ai_table_roi import TableROI
from backend.video_gpu_io import nvdec_bgr24_stream


@dataclass(frozen=True)
class BallTrackingSignals:
    timestamps: List[float]
    energies: List[float]
    crop_roi: Tuple[int, int, int, int]


def _expand_roi(roi: TableROI, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x, y, w, h = roi.as_tuple()
    pad_x = int(round(w * 0.12))
    pad_top = int(round(h * 0.80))
    pad_bottom = int(round(h * 0.30))

    ex1 = max(0, x - pad_x)
    ey1 = max(0, y - pad_top)
    ex2 = min(frame_w, x + w + pad_x)
    ey2 = min(frame_h, y + h + pad_bottom)
    return ex1, ey1, max(1, ex2 - ex1), max(1, ey2 - ey1)


def _extract_ball_candidates(diff_gray: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    if diff_gray.size == 0:
        return []

    blur = cv2.GaussianBlur(diff_gray, (3, 3), 0)
    thresh_val = max(10.0, float(np.percentile(blur, 99.2)) * 0.55)
    _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: List[Tuple[np.ndarray, float]] = []
    roi_area = float(diff_gray.shape[0] * diff_gray.shape[1])
    max_area = max(18, int(round(roi_area * 0.00042)))

    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < 2 or area > max_area:
            continue

        aspect = float(max(w, h) / max(1, min(w, h)))
        if aspect > 3.2:
            continue

        patch = blur[y : y + h, x : x + w]
        if patch.size == 0:
            continue

        mean_diff = float(patch.mean())
        compactness = float(area / max(1.0, w * h))
        score = (0.75 * (mean_diff / 255.0)) + (0.25 * compactness)
        center = np.array(centroids[idx], dtype=np.float32)
        candidates.append((center, float(score)))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def extract_ball_motion_energies(
    video_path: str,
    *,
    roi: TableROI,
    frame_w: int,
    frame_h: int,
    fps: float,
    stride: int = 2,
) -> BallTrackingSignals:
    v_path = Path(video_path).resolve()
    if not v_path.exists():
        raise FileNotFoundError(f"Video not found: {v_path}")

    crop_roi = _expand_roi(roi, frame_w, frame_h)
    timestamps: List[float] = []
    energies: List[float] = []
    prev_gray: Optional[np.ndarray] = None
    prev_center: Optional[np.ndarray] = None
    prev_velocity = np.zeros(2, dtype=np.float32)
    prev_energy = 0.0
    missing_count = 0

    frame_iter = nvdec_bgr24_stream(
        str(v_path),
        int(frame_w),
        int(frame_h),
        crop_roi=crop_roi,
    )

    for idx, frame_np in enumerate(frame_iter):
        if idx % stride != 0:
            continue

        timestamps.append(float(idx / fps))
        gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            energies.append(0.0)
            prev_gray = gray
            continue

        diff_gray = cv2.absdiff(gray, prev_gray)
        candidates = _extract_ball_candidates(diff_gray)

        chosen_center: Optional[np.ndarray] = None
        chosen_score = 0.0
        if candidates:
            if prev_center is None or missing_count > 2:
                chosen_center, chosen_score = candidates[0]
            else:
                predicted = prev_center + prev_velocity
                best_metric: Optional[float] = None
                for center, score in candidates:
                    dist = float(np.linalg.norm(center - predicted))
                    max_jump = 52.0 + (14.0 * float(missing_count))
                    if dist > max_jump:
                        continue
                    metric = dist - (score * 95.0)
                    if best_metric is None or metric < best_metric:
                        best_metric = metric
                        chosen_center = center
                        chosen_score = score
                if chosen_center is None:
                    chosen_center, chosen_score = candidates[0]

        if chosen_center is not None:
            speed = 0.0
            if prev_center is not None:
                delta = chosen_center - prev_center
                speed = float(np.linalg.norm(delta))
                prev_velocity = (0.45 * prev_velocity) + (0.55 * delta)
            else:
                prev_velocity = np.zeros(2, dtype=np.float32)
            prev_center = chosen_center
            missing_count = 0
            prev_energy = float(min(1.5, (0.85 * chosen_score) + (speed / 42.0)))
            energies.append(prev_energy)
        else:
            missing_count += 1
            if missing_count <= 2:
                prev_energy *= 0.62
                energies.append(float(prev_energy))
            else:
                prev_energy = 0.0
                prev_center = None
                prev_velocity = np.zeros(2, dtype=np.float32)
                energies.append(0.0)

        prev_gray = gray

    return BallTrackingSignals(
        timestamps=timestamps,
        energies=energies,
        crop_roi=crop_roi,
    )
