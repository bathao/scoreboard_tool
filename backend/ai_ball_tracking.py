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


@dataclass(frozen=True)
class BallTrackingProfile:
    pad_x_ratio: float
    pad_top_ratio: float
    pad_bottom_ratio: float
    min_start_score: float
    min_continue_score: float
    min_confirm_hits: int
    min_confirm_motion_px: float
    proposal_match_px: float
    max_jump_px: float
    max_jump_missing_gain: float
    allow_top_fallback: bool
    hold_misses: int
    hold_decay: float
    min_continue_motion_px: float
    strong_score: float
    score_weight: float
    speed_divisor: float


def _get_ball_tracking_profile(profile: str) -> BallTrackingProfile:
    if profile == "support":
        return BallTrackingProfile(
            pad_x_ratio=0.12,
            pad_top_ratio=0.80,
            pad_bottom_ratio=0.30,
            min_start_score=0.0,
            min_continue_score=0.0,
            min_confirm_hits=1,
            min_confirm_motion_px=0.0,
            proposal_match_px=26.0,
            max_jump_px=52.0,
            max_jump_missing_gain=14.0,
            allow_top_fallback=True,
            hold_misses=2,
            hold_decay=0.62,
            min_continue_motion_px=0.0,
            strong_score=0.0,
            score_weight=0.85,
            speed_divisor=42.0,
        )
    if profile == "standalone":
        return BallTrackingProfile(
            pad_x_ratio=0.08,
            pad_top_ratio=0.55,
            pad_bottom_ratio=0.18,
            min_start_score=0.17,
            min_continue_score=0.12,
            min_confirm_hits=2,
            min_confirm_motion_px=3.0,
            proposal_match_px=18.0,
            max_jump_px=38.0,
            max_jump_missing_gain=10.0,
            allow_top_fallback=False,
            hold_misses=1,
            hold_decay=0.35,
            min_continue_motion_px=2.6,
            strong_score=0.26,
            score_weight=0.50,
            speed_divisor=22.0,
        )
    raise ValueError(f"Invalid ball tracking profile: {profile}")


def _expand_roi(
    roi: TableROI,
    frame_w: int,
    frame_h: int,
    *,
    profile: BallTrackingProfile,
) -> Tuple[int, int, int, int]:
    x, y, w, h = roi.as_tuple()
    pad_x = int(round(w * profile.pad_x_ratio))
    pad_top = int(round(h * profile.pad_top_ratio))
    pad_bottom = int(round(h * profile.pad_bottom_ratio))

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


def _pick_best_candidate(
    candidates: List[Tuple[np.ndarray, float]],
    *,
    min_score: float,
    predicted_center: Optional[np.ndarray] = None,
    max_jump_px: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], float]:
    best_center: Optional[np.ndarray] = None
    best_score = 0.0
    best_metric: Optional[float] = None

    for center, score in candidates:
        if score < min_score:
            continue
        if predicted_center is None:
            return center, float(score)

        dist = float(np.linalg.norm(center - predicted_center))
        if max_jump_px is not None and dist > max_jump_px:
            continue
        metric = dist - (score * 95.0)
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_center = center
            best_score = float(score)

    return best_center, best_score


def extract_ball_motion_energies(
    video_path: str,
    *,
    roi: TableROI,
    frame_w: int,
    frame_h: int,
    fps: float,
    stride: int = 2,
    profile: str = "support",
) -> BallTrackingSignals:
    v_path = Path(video_path).resolve()
    if not v_path.exists():
        raise FileNotFoundError(f"Video not found: {v_path}")

    cfg = _get_ball_tracking_profile(profile)
    crop_roi = _expand_roi(roi, frame_w, frame_h, profile=cfg)
    timestamps: List[float] = []
    energies: List[float] = []
    prev_gray: Optional[np.ndarray] = None
    prev_center: Optional[np.ndarray] = None
    prev_velocity = np.zeros(2, dtype=np.float32)
    prev_energy = 0.0
    missing_count = 0
    proposal_center: Optional[np.ndarray] = None
    proposal_anchor: Optional[np.ndarray] = None
    proposal_velocity = np.zeros(2, dtype=np.float32)
    proposal_hits = 0
    proposal_best_score = 0.0

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
            if prev_center is None:
                starter_center, starter_score = _pick_best_candidate(
                    candidates,
                    min_score=cfg.min_start_score,
                )
                if starter_center is not None:
                    if proposal_center is None or float(np.linalg.norm(starter_center - proposal_center)) > cfg.proposal_match_px:
                        proposal_center = starter_center
                        proposal_anchor = starter_center
                        proposal_velocity = np.zeros(2, dtype=np.float32)
                        proposal_hits = 1
                        proposal_best_score = float(starter_score)
                    else:
                        delta = starter_center - proposal_center
                        proposal_velocity = (0.35 * proposal_velocity) + (0.65 * delta)
                        proposal_center = starter_center
                        proposal_hits += 1
                        proposal_best_score = max(float(proposal_best_score), float(starter_score))

                    confirm_motion = 0.0
                    if proposal_anchor is not None and proposal_center is not None:
                        confirm_motion = float(np.linalg.norm(proposal_center - proposal_anchor))
                    if proposal_hits >= cfg.min_confirm_hits and (
                        confirm_motion >= cfg.min_confirm_motion_px or proposal_best_score >= cfg.strong_score
                    ):
                        prev_center = proposal_center
                        prev_velocity = proposal_velocity.copy()
                        chosen_center = proposal_center
                        chosen_score = float(max(proposal_best_score, starter_score))
                        proposal_center = None
                        proposal_anchor = None
                        proposal_velocity = np.zeros(2, dtype=np.float32)
                        proposal_hits = 0
                        proposal_best_score = 0.0
                        missing_count = 0
                else:
                    proposal_center = None
                    proposal_anchor = None
                    proposal_velocity = np.zeros(2, dtype=np.float32)
                    proposal_hits = 0
                    proposal_best_score = 0.0
            else:
                predicted = prev_center + prev_velocity
                max_jump = cfg.max_jump_px + (cfg.max_jump_missing_gain * float(missing_count))
                chosen_center, chosen_score = _pick_best_candidate(
                    candidates,
                    min_score=cfg.min_continue_score,
                    predicted_center=predicted,
                    max_jump_px=max_jump,
                )
                if chosen_center is None and cfg.allow_top_fallback:
                    chosen_center, chosen_score = _pick_best_candidate(
                        candidates,
                        min_score=max(cfg.min_start_score, cfg.min_continue_score),
                    )

        if chosen_center is not None:
            speed = 0.0
            if prev_center is not None:
                delta = chosen_center - prev_center
                speed = float(np.linalg.norm(delta))
                if speed < cfg.min_continue_motion_px and chosen_score < cfg.strong_score:
                    chosen_center = None
                else:
                    prev_velocity = (0.45 * prev_velocity) + (0.55 * delta)
            else:
                prev_velocity = np.zeros(2, dtype=np.float32)
            if chosen_center is not None:
                prev_center = chosen_center
                missing_count = 0
                prev_energy = float(min(1.5, (cfg.score_weight * chosen_score) + (speed / cfg.speed_divisor)))
                energies.append(prev_energy)
            else:
                missing_count += 1
                if missing_count <= cfg.hold_misses:
                    prev_energy *= cfg.hold_decay
                    energies.append(float(prev_energy))
                else:
                    prev_energy = 0.0
                    prev_center = None
                    prev_velocity = np.zeros(2, dtype=np.float32)
                    energies.append(0.0)
        else:
            missing_count += 1
            if missing_count <= cfg.hold_misses:
                prev_energy *= cfg.hold_decay
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
