from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

from backend.ai_table_roi import TableROI


def _clip_box(box: Sequence[float], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))
    return x1, y1, x2, y2


def _xywh_to_xyxy(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    return x, y, x + w, y + h


def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_area(box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return float(max(1, x2 - x1) * max(1, y2 - y1))


def _intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _point_in_box(point: Tuple[float, float], box: Tuple[int, int, int, int]) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return inter / max(1.0, union)


def _safe_mean_hsv(frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(3, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean = hsv.reshape(-1, 3).mean(axis=0)
    return np.array(
        [float(mean[0]) / 179.0, float(mean[1]) / 255.0, float(mean[2]) / 255.0],
        dtype=np.float32,
    )


def _hsv_hist_signature(
    frame_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    *,
    h_bins: int = 8,
    s_bins: int = 4,
    v_bins: int = 4,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(h_bins + s_bins + v_bins, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180]).astype(np.float32).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256]).astype(np.float32).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256]).astype(np.float32).flatten()
    feat = np.concatenate([h_hist, s_hist, v_hist], axis=0)
    total = float(feat.sum())
    if total <= 0:
        return np.zeros_like(feat, dtype=np.float32)
    return (feat / total).astype(np.float32)


def _combined_color_signature(frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    return np.concatenate([_safe_mean_hsv(frame_bgr, box), _hsv_hist_signature(frame_bgr, box)], axis=0).astype(np.float32)


def _sub_box(box: Tuple[int, int, int, int], *, y0_ratio: float, y1_ratio: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    h = max(1, y2 - y1)
    sy1 = y1 + int(round(h * y0_ratio))
    sy2 = y1 + int(round(h * y1_ratio))
    sy1 = max(y1, min(sy1, y2 - 1))
    sy2 = max(sy1 + 1, min(sy2, y2))
    return x1, sy1, x2, sy2


def _norm_point(kpts: np.ndarray, idx: int) -> Optional[np.ndarray]:
    if idx >= len(kpts):
        return None
    x, y = kpts[idx]
    if x <= 0 or y <= 0:
        return None
    return np.array([float(x), float(y)], dtype=np.float32)


def _pair_distance(kpts: np.ndarray, idx_a: int, idx_b: int, scale: float) -> float:
    pa = _norm_point(kpts, idx_a)
    pb = _norm_point(kpts, idx_b)
    if pa is None or pb is None or scale <= 0:
        return 0.0
    return float(np.linalg.norm(pa - pb) / scale)


def _mean_point(kpts: np.ndarray, indices: Sequence[int]) -> Optional[np.ndarray]:
    pts = [_norm_point(kpts, idx) for idx in indices]
    valid = [p for p in pts if p is not None]
    if not valid:
        return None
    return np.mean(np.stack(valid, axis=0), axis=0)


def _body_signature(kpts: np.ndarray, box_h: float, box_w: float) -> np.ndarray:
    scale_h = max(1.0, float(box_h))
    aspect = float(box_w) / scale_h
    shoulder_w = _pair_distance(kpts, 5, 6, scale_h)
    hip_w = _pair_distance(kpts, 11, 12, scale_h)
    hip_center = _mean_point(kpts, [11, 12])
    shoulder_center = _mean_point(kpts, [5, 6])
    nose = _norm_point(kpts, 0)

    torso = 0.0
    if shoulder_center is not None and hip_center is not None:
        torso = float(np.linalg.norm(shoulder_center - hip_center) / scale_h)

    head_to_hip = 0.0
    if nose is not None and hip_center is not None:
        head_to_hip = float(np.linalg.norm(nose - hip_center) / scale_h)

    leg_l = _pair_distance(kpts, 11, 15, scale_h)
    leg_r = _pair_distance(kpts, 12, 16, scale_h)
    leg = max(leg_l, leg_r)
    return np.array([aspect, shoulder_w, hip_w, torso, head_to_hip, leg], dtype=np.float32)


def _box_only_body_signature(
    box: Tuple[int, int, int, int],
    *,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx, cy = _box_center(box)
    return np.array(
        [
            w / h,
            h / max(1.0, float(frame_h)),
            w / max(1.0, float(frame_w)),
            cx / max(1.0, float(frame_w)),
            cy / max(1.0, float(frame_h)),
            (w * h) / max(1.0, float(frame_w * frame_h)),
        ],
        dtype=np.float32,
    )


def _norm_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.linalg.norm(a - b))


@dataclass(frozen=True)
class OfflinePlayerDetection:
    frame_idx: int
    box: Tuple[int, int, int, int]
    center: Tuple[float, float]
    keypoints: np.ndarray
    detection_confidence: float
    near_table: bool
    in_player_zone: bool
    table_distance: float
    body_signature: np.ndarray
    upper_body_signature: np.ndarray
    head_signature: np.ndarray
    lower_body_signature: np.ndarray
    shoe_signature: np.ndarray


@dataclass(frozen=True)
class TrackletObservation:
    frame_idx: int
    box: Tuple[int, int, int, int]
    center: Tuple[float, float]
    keypoints: np.ndarray
    confidence: float


@dataclass
class PlayerTracklet:
    tracklet_id: int
    observations: List[TrackletObservation] = field(default_factory=list)
    near_table_frames: int = 0
    table_distance_sum: float = 0.0
    confidence_sum: float = 0.0
    body_signature: Optional[np.ndarray] = None
    upper_body_signature: Optional[np.ndarray] = None
    head_signature: Optional[np.ndarray] = None
    lower_body_signature: Optional[np.ndarray] = None
    shoe_signature: Optional[np.ndarray] = None
    assigned_role: Optional[str] = None

    @property
    def start_frame(self) -> int:
        return self.observations[0].frame_idx

    @property
    def end_frame(self) -> int:
        return self.observations[-1].frame_idx

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / max(1, len(self.observations))

    @property
    def near_table_ratio(self) -> float:
        return self.near_table_frames / max(1, len(self.observations))

    @property
    def mean_table_distance(self) -> float:
        return self.table_distance_sum / max(1, len(self.observations))

    @property
    def mean_center_x(self) -> float:
        return float(np.mean([obs.center[0] for obs in self.observations]))

    @property
    def mean_center_y(self) -> float:
        return float(np.mean([obs.center[1] for obs in self.observations]))

    @property
    def first_box(self) -> Tuple[int, int, int, int]:
        return self.observations[0].box

    @property
    def last_box(self) -> Tuple[int, int, int, int]:
        return self.observations[-1].box

    @property
    def last_center(self) -> Tuple[float, float]:
        return self.observations[-1].center

    @property
    def first_center(self) -> Tuple[float, float]:
        return self.observations[0].center

    def overlaps_with(self, other: "PlayerTracklet") -> bool:
        return self.start_frame <= other.end_frame and other.start_frame <= self.end_frame

    def get_observation(self, frame_idx: int) -> Optional[TrackletObservation]:
        for obs in self.observations:
            if obs.frame_idx == frame_idx:
                return obs
        return None


@dataclass(frozen=True)
class OfflineTrackingResult:
    tracklets: List[PlayerTracklet]
    role_frames: Dict[int, Dict[str, TrackletObservation]]


class OfflinePlayerTracker:
    def __init__(
        self,
        table_roi: TableROI,
        *,
        frame_w: int,
        frame_h: int,
        max_link_gap_frames: int = 10,
        max_center_jump_norm: float = 0.12,
        max_interpolate_gap_frames: int = 12,
    ) -> None:
        self.table_roi = table_roi
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        self.max_link_gap_frames = int(max_link_gap_frames)
        self.max_center_jump_norm = float(max_center_jump_norm)
        self.max_interpolate_gap_frames = int(max_interpolate_gap_frames)
        self.play_zone_xyxy = _xywh_to_xyxy(self.table_roi.get_unified_play_zone(self.frame_w, self.frame_h))
        self.player_zones = self._build_player_zones()
        self.table_center = np.array(
            [self.table_roi.x + self.table_roi.w / 2.0, self.table_roi.y + self.table_roi.h / 2.0],
            dtype=np.float32,
        )
        self._next_tracklet_id = 1
        self._active_tracklets: List[PlayerTracklet] = []
        self._finalized_tracklets: List[PlayerTracklet] = []

    def build_detections(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_idx: int,
        boxes_xyxy: np.ndarray,
        keypoints_xy: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
    ) -> List[OfflinePlayerDetection]:
        detections: List[OfflinePlayerDetection] = []
        for i in range(len(boxes_xyxy)):
            box = _clip_box(boxes_xyxy[i], frame_bgr.shape[1], frame_bgr.shape[0])
            x1, y1, x2, y2 = box
            center = _box_center(box)
            table_distance = float(np.linalg.norm(np.array(center, dtype=np.float32) - self.table_center))
            near_table = _intersects(box, self.play_zone_xyxy)
            in_player_zone = self._is_in_main_player_zone(box, center)
            conf = float(confidences[i]) if confidences is not None and i < len(confidences) else 0.0
            upper_box = _sub_box(box, y0_ratio=0.10, y1_ratio=0.52)
            head_box = _sub_box(box, y0_ratio=0.00, y1_ratio=0.24)
            lower_box = _sub_box(box, y0_ratio=0.45, y1_ratio=0.90)
            shoe_box = _sub_box(box, y0_ratio=0.84, y1_ratio=1.00)
            box_h = max(1, y2 - y1)
            box_w = max(1, x2 - x1)
            if keypoints_xy is not None and i < len(keypoints_xy):
                kpts = np.asarray(keypoints_xy[i], dtype=np.float32)
            else:
                kpts = np.zeros((17, 2), dtype=np.float32)
            has_pose = bool(np.any(kpts > 0))
            detections.append(
                OfflinePlayerDetection(
                    frame_idx=int(frame_idx),
                    box=box,
                    center=(float(center[0]), float(center[1])),
                    keypoints=kpts,
                    detection_confidence=conf,
                    near_table=near_table,
                    in_player_zone=in_player_zone,
                    table_distance=table_distance,
                    body_signature=_body_signature(kpts, box_h, box_w) if has_pose else _box_only_body_signature(
                        box,
                        frame_w=self.frame_w,
                        frame_h=self.frame_h,
                    ),
                    upper_body_signature=_combined_color_signature(frame_bgr, upper_box),
                    head_signature=_combined_color_signature(frame_bgr, head_box),
                    lower_body_signature=_combined_color_signature(frame_bgr, lower_box),
                    shoe_signature=_combined_color_signature(frame_bgr, shoe_box),
                )
            )
        detections.sort(key=lambda d: (not d.in_player_zone, not d.near_table, d.table_distance))
        return detections

    def add_frame_detections(self, detections: Sequence[OfflinePlayerDetection]) -> None:
        if not detections:
            return
        frame_idx = detections[0].frame_idx
        self._retire_stale_tracklets(frame_idx)
        matches, unmatched_tracklets, unmatched_detections = self._match_frame(detections)
        for tracklet, det in matches:
            self._append_detection(tracklet, det)
        for det in unmatched_detections:
            if self._should_start_tracklet(det):
                self._active_tracklets.append(self._new_tracklet(det))
        for stale in unmatched_tracklets:
            if frame_idx - stale.end_frame > self.max_link_gap_frames:
                self._finalized_tracklets.append(stale)
                self._active_tracklets.remove(stale)

    def finish(self) -> OfflineTrackingResult:
        self._finalized_tracklets.extend(self._active_tracklets)
        self._active_tracklets = []
        merged = self._merge_tracklets(self._finalized_tracklets)
        assigned = self._assign_roles(merged)
        role_frames = self._build_role_frames(assigned)
        return OfflineTrackingResult(tracklets=assigned, role_frames=role_frames)

    def _build_role_frames(
        self,
        tracklets: Sequence[PlayerTracklet],
    ) -> Dict[int, Dict[str, TrackletObservation]]:
        role_frames: Dict[int, Dict[str, TrackletObservation]] = {}
        role_tracklets: Dict[str, List[PlayerTracklet]] = {"A": [], "B": []}
        for tracklet in tracklets:
            if tracklet.assigned_role in role_tracklets:
                role_tracklets[tracklet.assigned_role].append(tracklet)

        for role, assigned_tracklets in role_tracklets.items():
            for tracklet in sorted(assigned_tracklets, key=lambda t: (t.start_frame, t.tracklet_id)):
                dense_obs = self._filter_role_observations(tracklet, self._densify_tracklet_observations(tracklet))
                for obs in dense_obs:
                    role_frames.setdefault(obs.frame_idx, {})
                    existing = role_frames[obs.frame_idx].get(role)
                    if existing is None or existing.confidence < obs.confidence:
                        role_frames[obs.frame_idx][role] = obs
        self._resolve_role_conflicts(role_frames, role_tracklets)
        return role_frames

    def _resolve_role_conflicts(
        self,
        role_frames: Dict[int, Dict[str, TrackletObservation]],
        role_tracklets: Dict[str, List[PlayerTracklet]],
    ) -> None:
        role_center_x = {
            role: float(np.mean([t.mean_center_x for t in tracklets])) if tracklets else None
            for role, tracklets in role_tracklets.items()
        }
        role_center_y = {
            role: float(np.mean([t.mean_center_y for t in tracklets])) if tracklets else None
            for role, tracklets in role_tracklets.items()
        }
        for _, roles in role_frames.items():
            if "A" not in roles or "B" not in roles:
                continue
            obs_a = roles["A"]
            obs_b = roles["B"]
            if _box_iou(obs_a.box, obs_b.box) < 0.85 and np.hypot(obs_a.center[0] - obs_b.center[0], obs_a.center[1] - obs_b.center[1]) > 25.0:
                continue
            score_a_for_a = self._role_observation_consistency(obs_a, role_center_x["A"], role_center_y["A"])
            score_a_for_b = self._role_observation_consistency(obs_a, role_center_x["B"], role_center_y["B"])
            score_b_for_a = self._role_observation_consistency(obs_b, role_center_x["A"], role_center_y["A"])
            score_b_for_b = self._role_observation_consistency(obs_b, role_center_x["B"], role_center_y["B"])
            if (score_a_for_a + score_b_for_b) <= (score_a_for_b + score_b_for_a):
                continue
            if score_a_for_a <= score_b_for_a:
                del roles["B"]
            else:
                del roles["A"]

    def _role_observation_consistency(
        self,
        obs: TrackletObservation,
        center_x: Optional[float],
        center_y: Optional[float],
    ) -> float:
        if center_x is None or center_y is None:
            return 999.0
        dx = abs(obs.center[0] - center_x) / max(1.0, float(self.frame_w))
        dy = abs(obs.center[1] - center_y) / max(1.0, float(self.frame_h))
        return (1.4 * dx) + (1.2 * dy) - (0.1 * obs.confidence)

    def _filter_role_observations(
        self,
        tracklet: PlayerTracklet,
        observations: Sequence[TrackletObservation],
    ) -> List[TrackletObservation]:
        if not observations:
            return []
        core_window = observations[: min(len(observations), 120)]
        core_center_x = float(np.median([obs.center[0] for obs in core_window]))
        core_center_y = float(np.median([obs.center[1] for obs in core_window]))
        core_area = float(np.median([_box_area(obs.box) for obs in core_window]))
        max_shift_x_px = max(220.0, self.frame_w * 0.13)
        max_shift_y_px = max(220.0, self.frame_h * 0.18)
        filtered: List[TrackletObservation] = []
        for obs in observations:
            if abs(obs.center[0] - core_center_x) > max_shift_x_px:
                continue
            if abs(obs.center[1] - core_center_y) > max_shift_y_px:
                continue
            area_ratio = _box_area(obs.box) / max(1.0, core_area)
            if area_ratio < 0.40 or area_ratio > 2.20:
                continue
            filtered.append(obs)
        return filtered

    def _densify_tracklet_observations(
        self,
        tracklet: PlayerTracklet,
    ) -> List[TrackletObservation]:
        if len(tracklet.observations) <= 1:
            return list(tracklet.observations)
        dense: List[TrackletObservation] = []
        obs_list = tracklet.observations
        for idx, current in enumerate(obs_list[:-1]):
            dense.append(current)
            nxt = obs_list[idx + 1]
            gap = nxt.frame_idx - current.frame_idx
            if gap <= 1 or gap > self.max_interpolate_gap_frames:
                continue
            for missing_frame in range(current.frame_idx + 1, nxt.frame_idx):
                alpha = float(missing_frame - current.frame_idx) / float(gap)
                dense.append(self._interpolate_observation(current, nxt, missing_frame, alpha))
        dense.append(obs_list[-1])
        return dense

    def _interpolate_observation(
        self,
        left: TrackletObservation,
        right: TrackletObservation,
        frame_idx: int,
        alpha: float,
    ) -> TrackletObservation:
        left_box = np.array(left.box, dtype=np.float32)
        right_box = np.array(right.box, dtype=np.float32)
        interp_box = ((1.0 - alpha) * left_box + alpha * right_box).astype(np.float32)
        box = _clip_box(interp_box, self.frame_w, self.frame_h)
        left_kpts = np.asarray(left.keypoints, dtype=np.float32)
        right_kpts = np.asarray(right.keypoints, dtype=np.float32)
        if left_kpts.shape == right_kpts.shape:
            keypoints = ((1.0 - alpha) * left_kpts + alpha * right_kpts).astype(np.float32)
        else:
            keypoints = left_kpts
        confidence = max(0.15, min(left.confidence, right.confidence) * 0.55)
        return TrackletObservation(
            frame_idx=int(frame_idx),
            box=box,
            center=_box_center(box),
            keypoints=keypoints,
            confidence=float(confidence),
        )

    def _retire_stale_tracklets(self, current_frame_idx: int) -> None:
        stale = [
            t
            for t in self._active_tracklets
            if current_frame_idx - t.end_frame > self.max_link_gap_frames
        ]
        for tracklet in stale:
            self._finalized_tracklets.append(tracklet)
            self._active_tracklets.remove(tracklet)

    def _should_start_tracklet(self, det: OfflinePlayerDetection) -> bool:
        diag = float(np.hypot(self.frame_w, self.frame_h))
        return det.in_player_zone or (det.near_table and (det.table_distance / max(1.0, diag)) <= 0.30)

    def _new_tracklet(self, det: OfflinePlayerDetection) -> PlayerTracklet:
        tracklet = PlayerTracklet(tracklet_id=self._next_tracklet_id)
        self._next_tracklet_id += 1
        self._append_detection(tracklet, det)
        return tracklet

    def _append_detection(self, tracklet: PlayerTracklet, det: OfflinePlayerDetection) -> None:
        tracklet.observations.append(
            TrackletObservation(
                frame_idx=det.frame_idx,
                box=det.box,
                center=det.center,
                keypoints=det.keypoints,
                confidence=det.detection_confidence,
            )
        )
        tracklet.near_table_frames += int(det.near_table)
        tracklet.table_distance_sum += det.table_distance
        tracklet.confidence_sum += det.detection_confidence
        tracklet.body_signature = self._ema(tracklet.body_signature, det.body_signature, alpha=0.25)
        tracklet.upper_body_signature = self._ema(tracklet.upper_body_signature, det.upper_body_signature, alpha=0.22)
        tracklet.head_signature = self._ema(tracklet.head_signature, det.head_signature, alpha=0.20)
        tracklet.lower_body_signature = self._ema(tracklet.lower_body_signature, det.lower_body_signature, alpha=0.20)
        tracklet.shoe_signature = self._ema(tracklet.shoe_signature, det.shoe_signature, alpha=0.18)

    def _match_frame(
        self,
        detections: Sequence[OfflinePlayerDetection],
    ) -> Tuple[List[Tuple[PlayerTracklet, OfflinePlayerDetection]], List[PlayerTracklet], List[OfflinePlayerDetection]]:
        candidates: List[Tuple[float, PlayerTracklet, OfflinePlayerDetection]] = []
        for tracklet in self._active_tracklets:
            for det in detections:
                cost = self._link_cost(tracklet, det)
                if cost is None:
                    continue
                candidates.append((cost, tracklet, det))
        candidates.sort(key=lambda item: item[0])

        matched_tracklet_ids: set[int] = set()
        matched_detection_ids: set[int] = set()
        matches: List[Tuple[PlayerTracklet, OfflinePlayerDetection]] = []
        for cost, tracklet, det in candidates:
            det_id = id(det)
            if tracklet.tracklet_id in matched_tracklet_ids or det_id in matched_detection_ids:
                continue
            matched_tracklet_ids.add(tracklet.tracklet_id)
            matched_detection_ids.add(det_id)
            matches.append((tracklet, det))

        unmatched_tracklets = [t for t in self._active_tracklets if t.tracklet_id not in matched_tracklet_ids]
        unmatched_detections = [d for d in detections if id(d) not in matched_detection_ids]
        return matches, unmatched_tracklets, unmatched_detections

    def _link_cost(self, tracklet: PlayerTracklet, det: OfflinePlayerDetection) -> Optional[float]:
        gap = det.frame_idx - tracklet.end_frame
        if gap <= 0 or gap > self.max_link_gap_frames:
            return None
        diag = float(np.hypot(self.frame_w, self.frame_h))
        predicted_center = np.array(tracklet.last_center, dtype=np.float32)
        if len(tracklet.observations) >= 2:
            prev = np.array(tracklet.observations[-2].center, dtype=np.float32)
            velocity = predicted_center - prev
            predicted_center = predicted_center + velocity * min(2.0, float(gap))
        center_norm = float(np.linalg.norm(np.array(det.center, dtype=np.float32) - predicted_center) / max(1.0, diag))
        max_jump = self.max_center_jump_norm + (0.018 * max(0, gap - 1))
        if center_norm > max_jump:
            return None

        iou = _box_iou(tracklet.last_box, det.box)
        appearance = self._appearance_cost(tracklet, det)
        zone_penalty = 0.0 if det.in_player_zone else 0.55
        near_table_bonus = -0.08 if det.near_table and tracklet.near_table_ratio >= 0.45 else 0.0
        return (0.60 * center_norm) + (0.20 * (1.0 - iou)) + (0.22 * appearance) + near_table_bonus + zone_penalty

    def _appearance_cost(self, tracklet: PlayerTracklet, det: OfflinePlayerDetection) -> float:
        return (
            1.20 * _norm_distance(tracklet.body_signature, det.body_signature)
            + 0.90 * _norm_distance(tracklet.upper_body_signature, det.upper_body_signature)
            + 0.80 * _norm_distance(tracklet.head_signature, det.head_signature)
            + 0.75 * _norm_distance(tracklet.lower_body_signature, det.lower_body_signature)
            + 0.50 * _norm_distance(tracklet.shoe_signature, det.shoe_signature)
        )

    def _merge_tracklets(self, tracklets: Sequence[PlayerTracklet]) -> List[PlayerTracklet]:
        merged = sorted([t for t in tracklets if t.duration_frames >= 1], key=lambda t: (t.start_frame, t.tracklet_id))
        max_safe_merge_gap = min(2, self.max_link_gap_frames)
        changed = True
        while changed:
            changed = False
            for idx in range(len(merged)):
                if changed:
                    break
                left = merged[idx]
                for jdx in range(idx + 1, len(merged)):
                    right = merged[jdx]
                    if right.start_frame <= left.end_frame:
                        continue
                    gap = right.start_frame - left.end_frame
                    if gap > max_safe_merge_gap:
                        break
                    cost = self._merge_cost(left, right)
                    if cost > 0.24:
                        continue
                    merged[idx] = self._combine_tracklets(left, right)
                    del merged[jdx]
                    changed = True
                    break
        return merged

    def _merge_cost(self, left: PlayerTracklet, right: PlayerTracklet) -> float:
        gap = right.start_frame - left.end_frame
        diag = float(np.hypot(self.frame_w, self.frame_h))
        center_norm = float(
            np.linalg.norm(np.array(right.first_center, dtype=np.float32) - np.array(left.last_center, dtype=np.float32))
            / max(1.0, diag)
        )
        appearance = (
            1.10 * _norm_distance(left.body_signature, right.body_signature)
            + 0.90 * _norm_distance(left.upper_body_signature, right.upper_body_signature)
            + 0.75 * _norm_distance(left.head_signature, right.head_signature)
            + 0.70 * _norm_distance(left.lower_body_signature, right.lower_body_signature)
        )
        gap_cost = min(1.0, gap / max(1.0, float(self.max_link_gap_frames * 2)))
        return (0.55 * center_norm) + (0.28 * appearance) + (0.17 * gap_cost)

    def _combine_tracklets(self, left: PlayerTracklet, right: PlayerTracklet) -> PlayerTracklet:
        combined = PlayerTracklet(tracklet_id=left.tracklet_id)
        combined.observations = [*left.observations, *right.observations]
        combined.near_table_frames = left.near_table_frames + right.near_table_frames
        combined.table_distance_sum = left.table_distance_sum + right.table_distance_sum
        combined.confidence_sum = left.confidence_sum + right.confidence_sum
        combined.body_signature = self._ema(left.body_signature, right.body_signature, alpha=0.5)
        combined.upper_body_signature = self._ema(left.upper_body_signature, right.upper_body_signature, alpha=0.5)
        combined.head_signature = self._ema(left.head_signature, right.head_signature, alpha=0.5)
        combined.lower_body_signature = self._ema(left.lower_body_signature, right.lower_body_signature, alpha=0.5)
        combined.shoe_signature = self._ema(left.shoe_signature, right.shoe_signature, alpha=0.5)
        return combined

    def _assign_roles(self, tracklets: Sequence[PlayerTracklet]) -> List[PlayerTracklet]:
        solved = [self._clone_tracklet(t) for t in tracklets]
        primary = [t for t in solved if self._tracklet_priority_score(t) >= 0.18]
        seed_pair = self._pick_seed_pair(primary)
        if seed_pair is None:
            return solved

        left_seed, right_seed = seed_pair
        left_seed.assigned_role = "A"
        right_seed.assigned_role = "B"
        role_prototypes: Dict[str, List[PlayerTracklet]] = {"A": [left_seed], "B": [right_seed]}

        remaining = [t for t in primary if t.tracklet_id not in {left_seed.tracklet_id, right_seed.tracklet_id}]
        remaining.sort(key=lambda t: (t.start_frame, -self._tracklet_priority_score(t)))
        for tracklet in remaining:
            cost_a = self._role_cost(tracklet, role_prototypes["A"], role="A", other_role_tracklets=role_prototypes["B"])
            cost_b = self._role_cost(tracklet, role_prototypes["B"], role="B", other_role_tracklets=role_prototypes["A"])
            best_role = "A" if cost_a <= cost_b else "B"
            best_cost = min(cost_a, cost_b)
            margin = abs(cost_a - cost_b)
            if best_cost > 1.35 or margin < 0.12:
                continue
            tracklet.assigned_role = best_role
            role_prototypes[best_role].append(tracklet)

        for tracklet in remaining:
            if tracklet.assigned_role is not None:
                continue
            sim_a = min(self._tracklet_similarity_cost(tracklet, other) for other in role_prototypes["A"])
            sim_b = min(self._tracklet_similarity_cost(tracklet, other) for other in role_prototypes["B"])
            best_role = "A" if sim_a <= sim_b else "B"
            best_sim = min(sim_a, sim_b)
            sim_margin = abs(sim_a - sim_b)
            role_center_mean = float(np.mean([other.mean_center_x for other in role_prototypes[best_role]]))
            role_center_y_mean = float(np.mean([other.mean_center_y for other in role_prototypes[best_role]]))
            diag = float(np.hypot(self.frame_w, self.frame_h))
            center_shift_norm = abs(tracklet.mean_center_x - role_center_mean) / max(1.0, diag)
            center_shift_y_norm = abs(tracklet.mean_center_y - role_center_y_mean) / max(1.0, float(self.frame_h))
            if self._tracklet_priority_score(tracklet) < 0.55:
                continue
            if best_sim > 0.85 or sim_margin < 0.18:
                continue
            if center_shift_norm > 0.16:
                continue
            if center_shift_y_norm > 0.14:
                continue
            tracklet.assigned_role = best_role
            role_prototypes[best_role].append(tracklet)
        return solved

    def _pick_seed_pair(self, tracklets: Sequence[PlayerTracklet]) -> Optional[Tuple[PlayerTracklet, PlayerTracklet]]:
        best_pair: Optional[Tuple[PlayerTracklet, PlayerTracklet]] = None
        best_score = -1.0
        ordered = sorted(tracklets, key=lambda t: (t.start_frame, -self._tracklet_priority_score(t)))
        for idx, left in enumerate(ordered):
            if self._tracklet_priority_score(left) < 0.22:
                continue
            for right in ordered[idx + 1 :]:
                if self._tracklet_priority_score(right) < 0.22:
                    continue
                overlap_start = max(left.start_frame, right.start_frame)
                overlap_end = min(left.end_frame, right.end_frame)
                if overlap_end < overlap_start:
                    continue
                left_obs = left.get_observation(overlap_start)
                right_obs = right.get_observation(overlap_start)
                if left_obs is None or right_obs is None:
                    continue
                if abs(left_obs.center[0] - right_obs.center[0]) < (self.frame_w * 0.06):
                    continue
                score = (
                    self._tracklet_priority_score(left)
                    + self._tracklet_priority_score(right)
                    - (overlap_start / max(1.0, left.end_frame + 1))
                )
                if score > best_score:
                    best_score = score
                    best_pair = (
                        left if left_obs.center[0] <= right_obs.center[0] else right,
                        right if left_obs.center[0] <= right_obs.center[0] else left,
                    )
        return best_pair

    def _role_cost(
        self,
        tracklet: PlayerTracklet,
        role_tracklets: Sequence[PlayerTracklet],
        *,
        role: str,
        other_role_tracklets: Sequence[PlayerTracklet],
    ) -> float:
        if not role_tracklets:
            return 999.0
        appearance = min(self._tracklet_similarity_cost(tracklet, other) for other in role_tracklets)
        overlap_penalty = 0.0
        for other in role_tracklets:
            if other.overlaps_with(tracklet):
                overlap_penalty += 0.55
        for other in other_role_tracklets:
            if other.overlaps_with(tracklet):
                overlap_penalty -= 0.08

        side_cost = 0.0
        seed = role_tracklets[0]
        table_x = float(self.table_center[0])
        role_center_mean = float(np.mean([other.mean_center_x for other in role_tracklets]))
        role_center_y_mean = float(np.mean([other.mean_center_y for other in role_tracklets]))
        diag = float(np.hypot(self.frame_w, self.frame_h))
        center_shift_norm = abs(tracklet.mean_center_x - role_center_mean) / max(1.0, diag)
        center_shift_y_norm = abs(tracklet.mean_center_y - role_center_y_mean) / max(1.0, float(self.frame_h))
        if seed.mean_center_x <= table_x and role == "A":
            side_cost = 0.0 if tracklet.mean_center_x <= table_x else 0.10
        if seed.mean_center_x > table_x and role == "B":
            side_cost = 0.0 if tracklet.mean_center_x > table_x else 0.10
        center_shift_penalty = 0.0
        if center_shift_norm > 0.18:
            center_shift_penalty = 0.65
        elif center_shift_norm > 0.12:
            center_shift_penalty = 0.25
        center_shift_y_penalty = 0.0
        if center_shift_y_norm > 0.16:
            center_shift_y_penalty = 0.75
        elif center_shift_y_norm > 0.10:
            center_shift_y_penalty = 0.30
        return appearance + overlap_penalty + side_cost + center_shift_penalty + center_shift_y_penalty - (0.20 * self._tracklet_priority_score(tracklet))

    def _tracklet_similarity_cost(self, a: PlayerTracklet, b: PlayerTracklet) -> float:
        return (
            1.10 * _norm_distance(a.body_signature, b.body_signature)
            + 0.95 * _norm_distance(a.upper_body_signature, b.upper_body_signature)
            + 0.75 * _norm_distance(a.head_signature, b.head_signature)
            + 0.75 * _norm_distance(a.lower_body_signature, b.lower_body_signature)
            + 0.45 * _norm_distance(a.shoe_signature, b.shoe_signature)
        )

    def _tracklet_priority_score(self, tracklet: PlayerTracklet) -> float:
        diag = float(np.hypot(self.frame_w, self.frame_h))
        distance_score = 1.0 - min(1.0, tracklet.mean_table_distance / max(1.0, diag * 0.40))
        duration_score = min(1.0, tracklet.duration_frames / 60.0)
        return (0.45 * tracklet.near_table_ratio) + (0.30 * duration_score) + (0.20 * distance_score) + (0.05 * tracklet.mean_confidence)

    def _build_player_zones(self) -> Dict[str, Tuple[int, int, int, int]]:
        x, y, w, h = self.table_roi.as_tuple()
        left = max(0, x - int(w * 0.45))
        right = min(self.frame_w, x + w + int(w * 0.45))
        top_far = max(0, y - int(h * 1.75))
        bottom_far = min(self.frame_h, y + int(h * 0.28))
        top_near = max(0, y + int(h * 0.08))
        bottom_near = min(self.frame_h, y + int(h * 2.80))
        side_pad_far = int(w * 0.06)
        side_pad_near = int(w * 0.18)
        return {
            "far": (
                max(0, x - side_pad_far),
                top_far,
                min(self.frame_w, x + w + side_pad_far),
                bottom_far,
            ),
            "near": (
                max(0, x - side_pad_near),
                top_near,
                min(self.frame_w, x + w + side_pad_near),
                bottom_near,
            ),
            "union": (left, top_far, right, bottom_near),
        }

    def _is_in_main_player_zone(
        self,
        box: Tuple[int, int, int, int],
        center: Tuple[float, float],
    ) -> bool:
        if _point_in_box(center, self.player_zones["far"]) or _point_in_box(center, self.player_zones["near"]):
            return True
        return _intersects(box, self.player_zones["far"]) or _intersects(box, self.player_zones["near"])

    def _clone_tracklet(self, tracklet: PlayerTracklet) -> PlayerTracklet:
        cloned = PlayerTracklet(tracklet_id=tracklet.tracklet_id)
        cloned.observations = list(tracklet.observations)
        cloned.near_table_frames = tracklet.near_table_frames
        cloned.table_distance_sum = tracklet.table_distance_sum
        cloned.confidence_sum = tracklet.confidence_sum
        cloned.body_signature = None if tracklet.body_signature is None else np.array(tracklet.body_signature, dtype=np.float32)
        cloned.upper_body_signature = None if tracklet.upper_body_signature is None else np.array(tracklet.upper_body_signature, dtype=np.float32)
        cloned.head_signature = None if tracklet.head_signature is None else np.array(tracklet.head_signature, dtype=np.float32)
        cloned.lower_body_signature = None if tracklet.lower_body_signature is None else np.array(tracklet.lower_body_signature, dtype=np.float32)
        cloned.shoe_signature = None if tracklet.shoe_signature is None else np.array(tracklet.shoe_signature, dtype=np.float32)
        cloned.assigned_role = tracklet.assigned_role
        return cloned

    def _ema(self, current: Optional[np.ndarray], new_value: np.ndarray, *, alpha: float) -> np.ndarray:
        if current is None:
            return np.array(new_value, dtype=np.float32)
        return ((1.0 - alpha) * current + alpha * new_value).astype(np.float32)
