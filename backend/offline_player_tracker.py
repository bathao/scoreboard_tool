from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

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
    role_state_frames: Dict[int, Dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class RoleProfile:
    role: str
    preferred_zone: str
    seed_center_x: float
    seed_center_y: float
    seed_anchor_x: float
    seed_anchor_y: float
    min_tracklet_zone_ratio: float
    max_center_depth_shift_norm: float
    max_anchor_depth_shift_norm: float
    min_observation_ownership_score: float


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
        max_role_occlusion_gap_frames: int = 90,
    ) -> None:
        self.table_roi = table_roi
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        self.max_link_gap_frames = int(max_link_gap_frames)
        self.max_center_jump_norm = float(max_center_jump_norm)
        self.max_interpolate_gap_frames = int(max_interpolate_gap_frames)
        self.max_role_occlusion_gap_frames = int(max_role_occlusion_gap_frames)
        self.play_zone_xyxy = _xywh_to_xyxy(self.table_roi.get_unified_play_zone(self.frame_w, self.frame_h))
        self.player_zones = self._build_player_zones()
        self.table_center = np.array(
            [self.table_roi.x + self.table_roi.w / 2.0, self.table_roi.y + self.table_roi.h / 2.0],
            dtype=np.float32,
        )
        self._next_tracklet_id = 1
        self._active_tracklets: List[PlayerTracklet] = []
        self._finalized_tracklets: List[PlayerTracklet] = []
        self._role_profiles: Dict[str, RoleProfile] = {}

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
        role_frames, role_state_frames = self._build_role_frames(assigned)
        return OfflineTrackingResult(
            tracklets=assigned,
            role_frames=role_frames,
            role_state_frames=role_state_frames,
        )

    def _build_role_frames(
        self,
        tracklets: Sequence[PlayerTracklet],
    ) -> Tuple[Dict[int, Dict[str, TrackletObservation]], Dict[int, Dict[str, str]]]:
        role_frames: Dict[int, Dict[str, TrackletObservation]] = {}
        role_state_frames: Dict[int, Dict[str, str]] = {}
        role_tracklets: Dict[str, List[PlayerTracklet]] = {"A": [], "B": []}
        for tracklet in tracklets:
            if tracklet.assigned_role in role_tracklets:
                role_tracklets[tracklet.assigned_role].append(tracklet)

        for role, assigned_tracklets in role_tracklets.items():
            selected_frames = self._select_role_observations(
                role,
                sorted(assigned_tracklets, key=lambda t: (t.start_frame, t.tracklet_id)),
                self._role_profiles.get(role),
            )
            for frame_idx, obs in selected_frames.items():
                role_frames.setdefault(frame_idx, {})
                role_frames[frame_idx][role] = obs
            self._mark_role_occlusions(
                role,
                sorted(assigned_tracklets, key=lambda t: (t.start_frame, t.tracklet_id)),
                role_frames,
                role_state_frames,
            )
        self._resolve_role_conflicts(role_frames, role_tracklets)
        for frame_idx, roles in role_frames.items():
            for role in roles:
                role_state_frames.setdefault(frame_idx, {})
                role_state_frames[frame_idx][role] = "visible"
        return role_frames, role_state_frames

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

    def _mark_role_occlusions(
        self,
        role: str,
        assigned_tracklets: Sequence[PlayerTracklet],
        role_frames: Dict[int, Dict[str, TrackletObservation]],
        role_state_frames: Dict[int, Dict[str, str]],
    ) -> None:
        ordered = [t for t in assigned_tracklets if t.assigned_role == role]
        for left, right in zip(ordered, ordered[1:]):
            gap = right.start_frame - left.end_frame - 1
            if gap <= 0 or gap > self.max_role_occlusion_gap_frames:
                continue
            if self._role_gap_bridge_cost(role, left, right) is None:
                continue
            for frame_idx in range(left.end_frame + 1, right.start_frame):
                if role in role_frames.get(frame_idx, {}):
                    continue
                role_state_frames.setdefault(frame_idx, {})
                role_state_frames[frame_idx][role] = "occluded"

    def _tracklet_exits_frame_edge(
        self,
        role: str,
        tracklet: PlayerTracklet,
    ) -> bool:
        if len(tracklet.observations) < 2:
            return False
        prev_obs = tracklet.observations[-2]
        last_obs = tracklet.observations[-1]
        prev_bottom_x = (prev_obs.box[0] + prev_obs.box[2]) / 2.0
        last_bottom_x = (last_obs.box[0] + last_obs.box[2]) / 2.0
        if role == "A":
            border_distance = float(last_obs.box[0])
            outward_motion = prev_bottom_x - last_bottom_x
        else:
            border_distance = float(self.frame_w - last_obs.box[2])
            outward_motion = last_bottom_x - prev_bottom_x
        outside_main_zone = not self._is_in_main_player_zone(last_obs.box, last_obs.center)
        return (border_distance <= (self.frame_w * 0.10) or outside_main_zone) and outward_motion >= (self.frame_w * 0.012)

    def _tracklet_enters_from_frame_edge(
        self,
        role: str,
        tracklet: PlayerTracklet,
    ) -> bool:
        if len(tracklet.observations) < 2:
            return False
        first_obs = tracklet.observations[0]
        next_obs = tracklet.observations[1]
        first_bottom_x = (first_obs.box[0] + first_obs.box[2]) / 2.0
        next_bottom_x = (next_obs.box[0] + next_obs.box[2]) / 2.0
        if role == "A":
            border_distance = float(first_obs.box[0])
            inward_motion = next_bottom_x - first_bottom_x
        else:
            border_distance = float(self.frame_w - first_obs.box[2])
            inward_motion = first_bottom_x - next_bottom_x
        outside_main_zone = not self._is_in_main_player_zone(first_obs.box, first_obs.center)
        return (border_distance <= (self.frame_w * 0.10) or outside_main_zone) and inward_motion >= (self.frame_w * 0.012)

    def _role_gap_bridge_cost(
        self,
        role: str,
        left: PlayerTracklet,
        right: PlayerTracklet,
    ) -> Optional[float]:
        gap = right.start_frame - left.end_frame - 1
        if gap < 0 or gap > self.max_role_occlusion_gap_frames:
            return None
        if self._tracklet_exits_frame_edge(role, left) or self._tracklet_enters_from_frame_edge(role, right):
            return None
        diag = float(np.hypot(self.frame_w, self.frame_h))
        dx = abs(right.first_center[0] - left.last_center[0]) / max(1.0, float(self.frame_w))
        dy = abs(right.first_center[1] - left.last_center[1]) / max(1.0, float(self.frame_h))
        center_norm = float(
            np.linalg.norm(np.array(right.first_center, dtype=np.float32) - np.array(left.last_center, dtype=np.float32))
            / max(1.0, diag)
        )
        area_ratio = _box_area(right.first_box) / max(1.0, _box_area(left.last_box))
        if dx > 0.14 or dy > 0.18 or center_norm > 0.18:
            return None
        if area_ratio < 0.45 or area_ratio > 2.40:
            return None
        appearance = self._tracklet_similarity_cost(left, right)
        gap_cost = gap / max(1.0, float(self.max_role_occlusion_gap_frames))
        cost = (0.46 * appearance) + (0.22 * center_norm) + (0.18 * dy) + (0.14 * gap_cost)
        if cost > 0.72:
            return None
        return cost

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

    def _tracklet_anchor_stats(
        self,
        tracklet: PlayerTracklet,
    ) -> Tuple[float, float, float]:
        bottoms_x = [((obs.box[0] + obs.box[2]) / 2.0) for obs in tracklet.observations]
        bottoms_y = [float(obs.box[3]) for obs in tracklet.observations]
        areas = [_box_area(obs.box) for obs in tracklet.observations]
        return float(np.median(bottoms_x)), float(np.median(bottoms_y)), float(np.median(areas))

    def _tracklet_center_stats(
        self,
        tracklet: PlayerTracklet,
    ) -> Tuple[float, float]:
        centers_x = [obs.center[0] for obs in tracklet.observations]
        centers_y = [obs.center[1] for obs in tracklet.observations]
        return float(np.median(centers_x)), float(np.median(centers_y))

    def _zone_membership_ratio(
        self,
        observations: Sequence[TrackletObservation],
        zone_name: str,
    ) -> float:
        if not observations:
            return 0.0
        hits = sum(1 for obs in observations if self._point_in_depth_band(obs.center, zone_name))
        return hits / max(1, len(observations))

    def _point_in_depth_band(
        self,
        point: Tuple[float, float],
        zone_name: str,
    ) -> bool:
        _x1, y1, _x2, y2 = self.player_zones[zone_name]
        return y1 <= point[1] <= y2

    def _preferred_zone_for_tracklet(self, tracklet: PlayerTracklet) -> str:
        near_ratio = self._zone_membership_ratio(tracklet.observations, "near")
        far_ratio = self._zone_membership_ratio(tracklet.observations, "far")
        if near_ratio > far_ratio:
            return "near"
        if far_ratio > near_ratio:
            return "far"
        center_x, center_y = self._tracklet_center_stats(tracklet)
        table_y_mid = float(self.table_roi.y + (self.table_roi.h * 0.55))
        if center_y >= table_y_mid:
            return "near"
        return "far"

    def _build_role_profile(
        self,
        role: str,
        seed: PlayerTracklet,
    ) -> RoleProfile:
        preferred_zone = self._preferred_zone_for_tracklet(seed)
        seed_anchor_x, seed_anchor_y, _seed_anchor_area = self._tracklet_anchor_stats(seed)
        seed_center_x, seed_center_y = self._tracklet_center_stats(seed)
        if preferred_zone == "near":
            return RoleProfile(
                role=role,
                preferred_zone=preferred_zone,
                seed_center_x=seed_center_x,
                seed_center_y=seed_center_y,
                seed_anchor_x=seed_anchor_x,
                seed_anchor_y=seed_anchor_y,
                min_tracklet_zone_ratio=0.34,
                max_center_depth_shift_norm=0.16,
                max_anchor_depth_shift_norm=0.24,
                min_observation_ownership_score=-0.06,
            )
        return RoleProfile(
            role=role,
            preferred_zone=preferred_zone,
            seed_center_x=seed_center_x,
            seed_center_y=seed_center_y,
            seed_anchor_x=seed_anchor_x,
            seed_anchor_y=seed_anchor_y,
            min_tracklet_zone_ratio=0.26,
            max_center_depth_shift_norm=0.20,
            max_anchor_depth_shift_norm=0.18,
            min_observation_ownership_score=-0.12,
        )

    def _role_side_ok(
        self,
        role: str,
        bottom_x: float,
    ) -> bool:
        table_x = float(self.table_center[0])
        if role == "A":
            return bottom_x <= table_x + (self.frame_w * 0.08)
        return bottom_x >= table_x - (self.frame_w * 0.08)

    def _depth_shift_norm(
        self,
        profile: RoleProfile,
        *,
        center_y: float,
        anchor_y: float,
    ) -> Tuple[float, float]:
        if profile.preferred_zone == "near":
            center_shift = max(0.0, profile.seed_center_y - center_y) / max(1.0, float(self.frame_h))
            anchor_shift = max(0.0, profile.seed_anchor_y - anchor_y) / max(1.0, float(self.frame_h))
        else:
            center_shift = max(0.0, center_y - profile.seed_center_y) / max(1.0, float(self.frame_h))
            anchor_shift = max(0.0, anchor_y - profile.seed_anchor_y) / max(1.0, float(self.frame_h))
        return float(center_shift), float(anchor_shift)

    def _tracklet_ownership_score(
        self,
        tracklet: PlayerTracklet,
        profile: RoleProfile,
    ) -> float:
        center_x, center_y = self._tracklet_center_stats(tracklet)
        anchor_x, anchor_y, _anchor_area = self._tracklet_anchor_stats(tracklet)
        if not self._role_side_ok(profile.role, anchor_x):
            return -999.0
        zone_ratio = self._zone_membership_ratio(tracklet.observations, profile.preferred_zone)
        center_shift, anchor_shift = self._depth_shift_norm(profile, center_y=center_y, anchor_y=anchor_y)
        return (
            (2.10 * zone_ratio)
            - (2.60 * center_shift)
            - (2.10 * anchor_shift)
            - (0.40 * abs(anchor_x - profile.seed_anchor_x) / max(1.0, float(self.frame_w)))
        )

    def _tracklet_matches_role_profile(
        self,
        tracklet: PlayerTracklet,
        profile: RoleProfile,
    ) -> bool:
        zone_ratio = self._zone_membership_ratio(tracklet.observations, profile.preferred_zone)
        if zone_ratio < profile.min_tracklet_zone_ratio:
            return False
        center_x, center_y = self._tracklet_center_stats(tracklet)
        anchor_x, anchor_y, _anchor_area = self._tracklet_anchor_stats(tracklet)
        if not self._role_side_ok(profile.role, anchor_x):
            return False
        center_shift, anchor_shift = self._depth_shift_norm(profile, center_y=center_y, anchor_y=anchor_y)
        if center_shift > profile.max_center_depth_shift_norm:
            return False
        if anchor_shift > profile.max_anchor_depth_shift_norm:
            return False
        return True

    def _observation_ownership_score(
        self,
        obs: TrackletObservation,
        profile: RoleProfile,
    ) -> float:
        in_zone = self._point_in_depth_band(obs.center, profile.preferred_zone)
        bottom_x = (obs.box[0] + obs.box[2]) / 2.0
        if not self._role_side_ok(profile.role, bottom_x):
            return -999.0
        center_shift, anchor_shift = self._depth_shift_norm(profile, center_y=float(obs.center[1]), anchor_y=float(obs.box[3]))
        return (
            (0.55 if in_zone else -0.22)
            - (1.75 * center_shift)
            - (1.50 * anchor_shift)
            - (0.25 * abs(bottom_x - profile.seed_anchor_x) / max(1.0, float(self.frame_w)))
        )

    def _tracklet_mode_signature(self, tracklet: PlayerTracklet) -> np.ndarray:
        anchor_x, anchor_y, anchor_area = self._tracklet_anchor_stats(tracklet)
        return np.array(
            [
                anchor_x / max(1.0, float(self.frame_w)),
                anchor_y / max(1.0, float(self.frame_h)),
                np.sqrt(anchor_area / max(1.0, float(self.frame_w * self.frame_h))),
                float(tracklet.mean_center_x) / max(1.0, float(self.frame_w)),
                float(tracklet.mean_center_y) / max(1.0, float(self.frame_h)),
            ],
            dtype=np.float32,
        )

    def _role_mode_cost(
        self,
        tracklet: PlayerTracklet,
        prototype: PlayerTracklet,
        *,
        role: str,
    ) -> float:
        mode_cost = _norm_distance(self._tracklet_mode_signature(tracklet), self._tracklet_mode_signature(prototype))
        lower_cost = _norm_distance(tracklet.lower_body_signature, prototype.lower_body_signature)
        shoe_cost = _norm_distance(tracklet.shoe_signature, prototype.shoe_signature)
        body_cost = _norm_distance(tracklet.body_signature, prototype.body_signature)
        side_cost = 0.0
        table_x = float(self.table_center[0])
        anchor_x, _anchor_y, _anchor_area = self._tracklet_anchor_stats(tracklet)
        if role == "A" and anchor_x > table_x + (self.frame_w * 0.14):
            side_cost = 0.22
        if role == "B" and anchor_x < table_x - (self.frame_w * 0.14):
            side_cost = 0.22
        return (0.44 * mode_cost) + (0.34 * lower_cost) + (0.18 * shoe_cost) + (0.12 * body_cost) + side_cost

    def _same_role_overlap_compatibility(
        self,
        tracklet: PlayerTracklet,
        other: PlayerTracklet,
        *,
        role: str,
    ) -> bool:
        if not tracklet.overlaps_with(other):
            return False
        overlap_start = max(tracklet.start_frame, other.start_frame)
        overlap_end = min(tracklet.end_frame, other.end_frame)
        if overlap_end < overlap_start:
            return False
        tracklet_obs = tracklet.get_observation(overlap_start)
        other_obs = other.get_observation(overlap_start)
        if tracklet_obs is None or other_obs is None:
            return False

        bottom_dx = abs(((tracklet_obs.box[0] + tracklet_obs.box[2]) / 2.0) - ((other_obs.box[0] + other_obs.box[2]) / 2.0)) / max(1.0, float(self.frame_w))
        bottom_dy = abs(tracklet_obs.box[3] - other_obs.box[3]) / max(1.0, float(self.frame_h))
        lower_cost = _norm_distance(tracklet.lower_body_signature, other.lower_body_signature)
        shoe_cost = _norm_distance(tracklet.shoe_signature, other.shoe_signature)
        table_x = float(self.table_center[0])
        anchor_x, _anchor_y, _anchor_area = self._tracklet_anchor_stats(tracklet)
        if role == "A" and anchor_x > table_x + (self.frame_w * 0.14):
            return False
        if role == "B" and anchor_x < table_x - (self.frame_w * 0.14):
            return False
        return bottom_dx <= 0.12 and bottom_dy <= 0.18 and lower_cost <= 1.15 and shoe_cost <= 1.05

    def _observation_unary_score(
        self,
        role: str,
        tracklet: PlayerTracklet,
        obs: TrackletObservation,
    ) -> float:
        score = 1.30 * float(obs.confidence)
        if _intersects(obs.box, self.play_zone_xyxy):
            score += 0.06
        if self._is_in_main_player_zone(obs.box, obs.center):
            score += 0.08
        table_x = float(self.table_center[0])
        bottom_x = (obs.box[0] + obs.box[2]) / 2.0
        if role == "A":
            if bottom_x <= table_x + (self.frame_w * 0.08):
                score += 0.10
            else:
                score -= 0.22
        else:
            if bottom_x >= table_x - (self.frame_w * 0.08):
                score += 0.10
            else:
                score -= 0.22
        score += 0.08 * self._tracklet_priority_score(tracklet)
        return score

    def _observation_transition_score(
        self,
        prev_tracklet_id: int,
        prev_obs: TrackletObservation,
        next_tracklet_id: int,
        next_obs: TrackletObservation,
    ) -> float:
        dx = abs(next_obs.center[0] - prev_obs.center[0]) / max(1.0, float(self.frame_w))
        dy = abs(next_obs.center[1] - prev_obs.center[1]) / max(1.0, float(self.frame_h))
        prev_bottom_x = (prev_obs.box[0] + prev_obs.box[2]) / 2.0
        next_bottom_x = (next_obs.box[0] + next_obs.box[2]) / 2.0
        bottom_dx = abs(next_bottom_x - prev_bottom_x) / max(1.0, float(self.frame_w))
        prev_area = _box_area(prev_obs.box)
        next_area = _box_area(next_obs.box)
        area_delta = abs(np.log(max(1e-6, next_area / max(1.0, prev_area))))
        switch_penalty = 0.14 if prev_tracklet_id != next_tracklet_id else 0.0
        return -((1.45 * dx) + (0.85 * dy) + (0.95 * bottom_dx) + (0.18 * area_delta) + switch_penalty)

    def _select_role_observations(
        self,
        role: str,
        assigned_tracklets: Sequence[PlayerTracklet],
        profile: Optional[RoleProfile],
    ) -> Dict[int, TrackletObservation]:
        frame_candidates: Dict[int, List[Tuple[int, PlayerTracklet, TrackletObservation, float]]] = {}
        for tracklet in assigned_tracklets:
            for obs in self._densify_tracklet_observations(tracklet):
                ownership_score = 0.0 if profile is None else self._observation_ownership_score(obs, profile)
                if profile is not None and ownership_score < profile.min_observation_ownership_score:
                    continue
                frame_candidates.setdefault(obs.frame_idx, []).append((tracklet.tracklet_id, tracklet, obs, ownership_score))

        if not frame_candidates:
            return {}

        selected: Dict[int, TrackletObservation] = {}
        frame_groups: List[List[int]] = []
        for frame_idx in sorted(frame_candidates):
            if not frame_groups or frame_idx > (frame_groups[-1][-1] + 1):
                frame_groups.append([frame_idx])
            else:
                frame_groups[-1].append(frame_idx)

        for frames in frame_groups:
            if not frames:
                continue
            scores_by_frame: List[List[float]] = []
            backptr_by_frame: List[List[int]] = []
            first_candidates = frame_candidates[frames[0]]
            scores_by_frame.append([
                self._observation_unary_score(role, tracklet, obs) + ownership_score
                for _tracklet_id, tracklet, obs, ownership_score in first_candidates
            ])
            backptr_by_frame.append([-1] * len(first_candidates))

            for frame_idx in frames[1:]:
                candidates = frame_candidates[frame_idx]
                prev_candidates = frame_candidates[frames[len(scores_by_frame) - 1]]
                prev_scores = scores_by_frame[-1]
                curr_scores: List[float] = []
                curr_backptr: List[int] = []
                for curr_idx, (curr_tracklet_id, curr_tracklet, curr_obs, curr_ownership) in enumerate(candidates):
                    unary = self._observation_unary_score(role, curr_tracklet, curr_obs) + curr_ownership
                    best_score = None
                    best_prev = -1
                    for prev_idx, (prev_tracklet_id, _prev_tracklet, prev_obs, _prev_ownership) in enumerate(prev_candidates):
                        transition = self._observation_transition_score(prev_tracklet_id, prev_obs, curr_tracklet_id, curr_obs)
                        candidate_score = prev_scores[prev_idx] + unary + transition
                        if best_score is None or candidate_score > best_score:
                            best_score = candidate_score
                            best_prev = prev_idx
                    curr_scores.append(float(best_score if best_score is not None else unary))
                    curr_backptr.append(best_prev)
                scores_by_frame.append(curr_scores)
                backptr_by_frame.append(curr_backptr)

            last_scores = scores_by_frame[-1]
            best_idx = max(range(len(last_scores)), key=lambda idx: last_scores[idx])
            for rev_idx in range(len(frames) - 1, -1, -1):
                frame_idx = frames[rev_idx]
                tracklet_id, tracklet, obs, ownership_score = frame_candidates[frame_idx][best_idx]
                if profile is not None:
                    unary = self._observation_unary_score(role, tracklet, obs) + ownership_score
                    if unary < 0.72:
                        prev_idx = backptr_by_frame[rev_idx][best_idx]
                        if prev_idx < 0:
                            break
                        best_idx = prev_idx
                        continue
                selected[frame_idx] = obs
                prev_idx = backptr_by_frame[rev_idx][best_idx]
                if prev_idx < 0:
                    break
                best_idx = prev_idx
        return selected

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

    def _recent_observations(
        self,
        tracklet: PlayerTracklet,
        *,
        window: int = 10,
    ) -> List[TrackletObservation]:
        if len(tracklet.observations) <= window:
            return list(tracklet.observations)
        return list(tracklet.observations[-window:])

    def _tracklet_history_center(
        self,
        tracklet: PlayerTracklet,
        *,
        window: int = 10,
    ) -> np.ndarray:
        obs = self._recent_observations(tracklet, window=window)
        if not obs:
            return np.array(tracklet.last_center, dtype=np.float32)
        xs = [item.center[0] for item in obs]
        ys = [item.center[1] for item in obs]
        return np.array([float(np.median(xs)), float(np.median(ys))], dtype=np.float32)

    def _tracklet_history_area(
        self,
        tracklet: PlayerTracklet,
        *,
        window: int = 10,
    ) -> float:
        obs = self._recent_observations(tracklet, window=window)
        if not obs:
            return _box_area(tracklet.last_box)
        return float(np.median([_box_area(item.box) for item in obs]))

    def _match_frame(
        self,
        detections: Sequence[OfflinePlayerDetection],
    ) -> Tuple[List[Tuple[PlayerTracklet, OfflinePlayerDetection]], List[PlayerTracklet], List[OfflinePlayerDetection]]:
        if not self._active_tracklets or not detections:
            return [], list(self._active_tracklets), list(detections)

        tracklets = list(self._active_tracklets)
        dets = list(detections)
        n_tracklets = len(tracklets)
        n_dets = len(dets)
        size = n_tracklets + n_dets
        huge = 1e6
        keep_unmatched_cost = 0.58
        start_new_cost = 0.62
        cost_matrix = np.full((size, size), huge, dtype=np.float32)

        for i, tracklet in enumerate(tracklets):
            for j, det in enumerate(dets):
                cost = self._link_cost(tracklet, det)
                if cost is not None:
                    cost_matrix[i, j] = cost
            cost_matrix[i, n_dets + i] = keep_unmatched_cost

        for j in range(n_dets):
            cost_matrix[n_tracklets + j, j] = start_new_cost

        cost_matrix[n_tracklets:, n_dets:] = 0.0
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracklet_ids: set[int] = set()
        matched_detection_ids: set[int] = set()
        matches: List[Tuple[PlayerTracklet, OfflinePlayerDetection]] = []
        for row, col in zip(row_ind.tolist(), col_ind.tolist()):
            if row >= n_tracklets or col >= n_dets:
                continue
            if cost_matrix[row, col] >= keep_unmatched_cost:
                continue
            tracklet = tracklets[row]
            det = dets[col]
            matched_tracklet_ids.add(tracklet.tracklet_id)
            matched_detection_ids.add(id(det))
            matches.append((tracklet, det))

        unmatched_tracklets = [t for t in tracklets if t.tracklet_id not in matched_tracklet_ids]
        unmatched_detections = [d for d in dets if id(d) not in matched_detection_ids]
        return matches, unmatched_tracklets, unmatched_detections

    def _link_cost(self, tracklet: PlayerTracklet, det: OfflinePlayerDetection) -> Optional[float]:
        gap = det.frame_idx - tracklet.end_frame
        if gap <= 0 or gap > self.max_link_gap_frames:
            return None
        diag = float(np.hypot(self.frame_w, self.frame_h))
        predicted_center = np.array(tracklet.last_center, dtype=np.float32)
        history_center = self._tracklet_history_center(tracklet)
        history_area = self._tracklet_history_area(tracklet)
        if len(tracklet.observations) >= 2:
            prev = np.array(tracklet.observations[-2].center, dtype=np.float32)
            velocity = predicted_center - prev
            predicted_center = predicted_center + velocity * min(2.0, float(gap))
        det_center = np.array(det.center, dtype=np.float32)
        center_norm = float(np.linalg.norm(det_center - predicted_center) / max(1.0, diag))
        history_center_norm = float(np.linalg.norm(det_center - history_center) / max(1.0, diag))
        max_jump = self.max_center_jump_norm + (0.018 * max(0, gap - 1))
        if center_norm > max_jump:
            return None
        if len(tracklet.observations) >= 6 and history_center_norm > (max_jump + 0.055):
            return None

        vertical_shift_norm = abs(det.center[1] - float(history_center[1])) / max(1.0, float(self.frame_h))
        if len(tracklet.observations) >= 6 and vertical_shift_norm > (0.18 + (0.01 * max(0, gap - 1))):
            return None

        det_area = _box_area(det.box)
        area_ratio = det_area / max(1.0, history_area)
        if len(tracklet.observations) >= 6 and (area_ratio < 0.38 or area_ratio > 2.75):
            return None

        iou = _box_iou(tracklet.last_box, det.box)
        appearance = self._appearance_cost(tracklet, det)
        shape_cost = min(1.0, abs(np.log(max(1e-6, area_ratio))))
        zone_penalty = 0.0 if det.in_player_zone else 0.55
        near_table_bonus = -0.08 if det.near_table and tracklet.near_table_ratio >= 0.45 else 0.0
        return (
            (0.42 * center_norm)
            + (0.18 * history_center_norm)
            + (0.14 * (1.0 - iou))
            + (0.17 * appearance)
            + (0.09 * shape_cost)
            + near_table_bonus
            + zone_penalty
        )

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
        self._role_profiles = {}
        primary = [t for t in solved if self._tracklet_priority_score(t) >= 0.18]
        seed_pair = self._pick_seed_pair(primary)
        if seed_pair is None:
            return solved

        left_seed, right_seed = seed_pair
        left_seed.assigned_role = "A"
        right_seed.assigned_role = "B"
        self._role_profiles = {
            "A": self._build_role_profile("A", left_seed),
            "B": self._build_role_profile("B", right_seed),
        }
        role_prototypes: Dict[str, List[PlayerTracklet]] = {"A": [left_seed], "B": [right_seed]}

        remaining = [t for t in primary if t.tracklet_id not in {left_seed.tracklet_id, right_seed.tracklet_id}]
        remaining.sort(key=lambda t: (t.start_frame, -self._tracklet_priority_score(t)))
        for tracklet in remaining:
            if "A" in self._role_profiles and not self._tracklet_matches_role_profile(tracklet, self._role_profiles["A"]):
                cost_a = 999.0
            else:
                cost_a = self._role_cost(tracklet, role_prototypes["A"], role="A", other_role_tracklets=role_prototypes["B"])
            if "B" in self._role_profiles and not self._tracklet_matches_role_profile(tracklet, self._role_profiles["B"]):
                cost_b = 999.0
            else:
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
            if "A" in self._role_profiles and not self._tracklet_matches_role_profile(tracklet, self._role_profiles["A"]):
                sim_a = 999.0
            else:
                sim_a = min(self._role_mode_cost(tracklet, other, role="A") for other in role_prototypes["A"])
            if "B" in self._role_profiles and not self._tracklet_matches_role_profile(tracklet, self._role_profiles["B"]):
                sim_b = 999.0
            else:
                sim_b = min(self._role_mode_cost(tracklet, other, role="B") for other in role_prototypes["B"])
            best_role = "A" if sim_a <= sim_b else "B"
            best_sim = min(sim_a, sim_b)
            sim_margin = abs(sim_a - sim_b)
            if best_sim >= 999.0:
                continue
            best_proto = min(
                role_prototypes[best_role],
                key=lambda other: self._role_mode_cost(tracklet, other, role=best_role),
            )
            role_center_mean = float(best_proto.mean_center_x)
            role_center_y_mean = float(best_proto.mean_center_y)
            diag = float(np.hypot(self.frame_w, self.frame_h))
            center_shift_norm = abs(tracklet.mean_center_x - role_center_mean) / max(1.0, diag)
            center_shift_y_norm = abs(tracklet.mean_center_y - role_center_y_mean) / max(1.0, float(self.frame_h))
            if self._tracklet_priority_score(tracklet) < 0.55:
                continue
            if best_sim > 0.88 or sim_margin < 0.10:
                continue
            if center_shift_norm > 0.16:
                continue
            if center_shift_y_norm > 0.22:
                continue
            tracklet.assigned_role = best_role
            role_prototypes[best_role].append(tracklet)
        return solved

    def _pick_seed_pair(self, tracklets: Sequence[PlayerTracklet]) -> Optional[Tuple[PlayerTracklet, PlayerTracklet]]:
        best_pair: Optional[Tuple[PlayerTracklet, PlayerTracklet]] = None
        best_score = -1.0
        ordered = sorted(tracklets, key=lambda t: (-self._tracklet_priority_score(t), t.start_frame, t.tracklet_id))
        for idx, first in enumerate(ordered):
            for second in ordered[idx + 1 :]:
                left, right = self._order_seed_pair(first, second)
                score = self._seed_pair_score(left, right)
                if score is None:
                    continue
                if score > best_score:
                    best_score = score
                    best_pair = (left, right)
        return best_pair

    def _order_seed_pair(
        self,
        first: PlayerTracklet,
        second: PlayerTracklet,
    ) -> Tuple[PlayerTracklet, PlayerTracklet]:
        return (
            (first, second)
            if first.mean_center_x <= second.mean_center_x
            else (second, first)
        )

    def _seed_tracklet_ok(
        self,
        tracklet: PlayerTracklet,
        *,
        role: str,
    ) -> bool:
        if self._tracklet_priority_score(tracklet) < 0.22:
            return False

        anchor_x, anchor_y, _anchor_area = self._tracklet_anchor_stats(tracklet)
        if not self._role_side_ok(role, anchor_x):
            return False

        table_bottom = float(self.table_roi.y + self.table_roi.h)
        if role == "A":
            return anchor_y >= (table_bottom + (self.table_roi.h * 0.04))
        return anchor_y <= (table_bottom + (self.table_roi.h * 0.30))

    def _seed_pair_score(
        self,
        left: PlayerTracklet,
        right: PlayerTracklet,
    ) -> Optional[float]:
        if not self._seed_tracklet_ok(left, role="A"):
            return None
        if not self._seed_tracklet_ok(right, role="B"):
            return None

        overlap_start = max(left.start_frame, right.start_frame)
        overlap_end = min(left.end_frame, right.end_frame)
        if overlap_end < overlap_start:
            return None
        overlap_frames = overlap_end - overlap_start + 1
        if overlap_frames < 2:
            return None

        left_obs = left.get_observation(overlap_start)
        right_obs = right.get_observation(overlap_start)
        if left_obs is None or right_obs is None:
            return None

        center_gap_norm = abs(left_obs.center[0] - right_obs.center[0]) / max(1.0, float(self.frame_w))
        if center_gap_norm < 0.06:
            return None

        left_anchor_x, left_anchor_y, _left_area = self._tracklet_anchor_stats(left)
        right_anchor_x, right_anchor_y, _right_area = self._tracklet_anchor_stats(right)
        depth_gap_norm = (left_anchor_y - right_anchor_y) / max(1.0, float(self.frame_h))
        left_zone = self._preferred_zone_for_tracklet(left)
        right_zone = self._preferred_zone_for_tracklet(right)
        if depth_gap_norm < -0.04:
            return None
        if left_zone != "near" and depth_gap_norm < 0.08:
            return None

        left_priority = self._tracklet_priority_score(left)
        right_priority = self._tracklet_priority_score(right)
        overlap_score = min(1.0, overlap_frames / 45.0)
        start_penalty = min(0.10, overlap_start / 600.0)
        table_x = float(self.table_center[0])
        side_balance = (
            abs(left_anchor_x - table_x) + abs(right_anchor_x - table_x)
        ) / max(1.0, float(self.frame_w))
        zone_bonus = 0.0
        if left_zone == "near":
            zone_bonus += 0.55
        else:
            zone_bonus -= 0.35
        if right_zone == "far":
            zone_bonus += 0.18

        return (
            (1.15 * left_priority)
            + (1.10 * right_priority)
            + (0.65 * overlap_score)
            + (0.90 * max(0.0, depth_gap_norm))
            + (0.30 * center_gap_norm)
            + zone_bonus
            - (0.18 * side_balance)
            - start_penalty
        )

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
        profile = self._role_profiles.get(role)
        if profile is not None and not self._tracklet_matches_role_profile(tracklet, profile):
            return 999.0
        prototype_costs = [self._role_mode_cost(tracklet, other, role=role) for other in role_tracklets]
        appearance = min(prototype_costs)
        overlap_penalty = 0.0
        for other in role_tracklets:
            if other.overlaps_with(tracklet):
                if self._same_role_overlap_compatibility(tracklet, other, role=role):
                    overlap_penalty += 0.06
                else:
                    overlap_penalty += 0.55
        for other in other_role_tracklets:
            if other.overlaps_with(tracklet):
                overlap_penalty -= 0.08

        side_cost = 0.0
        seed = role_tracklets[0]
        table_x = float(self.table_center[0])
        best_idx = int(np.argmin(np.asarray(prototype_costs, dtype=np.float32)))
        best_proto = role_tracklets[best_idx]
        role_center_mean = float(best_proto.mean_center_x)
        role_center_y_mean = float(best_proto.mean_center_y)
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
        if center_shift_y_norm > 0.24:
            center_shift_y_penalty = 0.75
        elif center_shift_y_norm > 0.16:
            center_shift_y_penalty = 0.30
        ownership_penalty = 0.0
        if profile is not None:
            ownership_score = self._tracklet_ownership_score(tracklet, profile)
            ownership_penalty = max(0.0, 0.55 - ownership_score)
        return (
            appearance
            + overlap_penalty
            + side_cost
            + center_shift_penalty
            + center_shift_y_penalty
            + ownership_penalty
            - (0.20 * self._tracklet_priority_score(tracklet))
        )

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
