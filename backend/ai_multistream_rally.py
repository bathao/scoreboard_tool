from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from backend.ai_ball_tracking import extract_ball_motion_energies
from backend.offline_player_tracker import OfflinePlayerTracker, TrackletObservation
from backend.ai_rally_segmentation import RallySegment, detect_rally_segments_advanced_gpu
from backend.ai_table_roi import TableROI
from backend.ai_table_roi_dl import DLConfig, detect_table_roi_dl
from backend.video_gpu_io import nvdec_bgr24_stream, probe_video_ffprobe


@dataclass(frozen=True)
class MultiStreamSignals:
    roi: TableROI
    timestamps: List[float]
    table_energies: List[float]
    ball_energies: List[float]
    player_a_energies: List[float]
    player_b_energies: List[float]
    player_energies: List[float]
    fused_energies: List[float]
    effective_fps: float
    player_signal_source: str
    ball_signal_source: str
    player_a_crouch_scores: List[float] = field(default_factory=list)
    player_b_crouch_scores: List[float] = field(default_factory=list)
    player_a_serve_scores: List[float] = field(default_factory=list)
    player_b_serve_scores: List[float] = field(default_factory=list)
    player_a_upper_body_scores: List[float] = field(default_factory=list)
    player_b_upper_body_scores: List[float] = field(default_factory=list)
    player_a_footwork_scores: List[float] = field(default_factory=list)
    player_b_footwork_scores: List[float] = field(default_factory=list)
    player_a_reach_scores: List[float] = field(default_factory=list)
    player_b_reach_scores: List[float] = field(default_factory=list)
    player_a_net_approach_scores: List[float] = field(default_factory=list)
    player_b_net_approach_scores: List[float] = field(default_factory=list)
    player_a_face_hidden_scores: List[float] = field(default_factory=list)
    player_b_face_hidden_scores: List[float] = field(default_factory=list)
    player_a_face_touch_scores: List[float] = field(default_factory=list)
    player_b_face_touch_scores: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class RoleTrackerSeries:
    timestamps: List[float]
    table_energies: List[float]
    player_a_energies: List[float]
    player_b_energies: List[float]
    player_a_crouch_scores: List[float]
    player_b_crouch_scores: List[float]
    player_a_serve_scores: List[float]
    player_b_serve_scores: List[float]
    player_a_upper_body_scores: List[float]
    player_b_upper_body_scores: List[float]
    player_a_footwork_scores: List[float]
    player_b_footwork_scores: List[float]
    player_a_reach_scores: List[float]
    player_b_reach_scores: List[float]
    player_a_net_approach_scores: List[float]
    player_b_net_approach_scores: List[float]
    player_a_face_hidden_scores: List[float]
    player_b_face_hidden_scores: List[float]
    player_a_face_touch_scores: List[float]
    player_b_face_touch_scores: List[float]


@dataclass(frozen=True)
class PlayerStateMachineDiagnostics:
    timestamps: List[float]
    segments: List[RallySegment]
    phase_by_frame: List[str]
    server_role_by_frame: List[str]
    ready_recent_flags: List[bool]
    live_now_flags: List[bool]
    dead_now_flags: List[bool]
    catch_proxy_scores: List[float]
    quiet_after_catch_scores: List[float]
    ready_pair: List[float]
    live_pair: List[float]
    casual_pair: List[float]
    stand_pair: List[float]
    motion_a: List[float]
    motion_b: List[float]
    crouch_a: List[float]
    crouch_b: List[float]
    serve_a: List[float]
    serve_b: List[float]
    upper_a: List[float]
    upper_b: List[float]
    foot_a: List[float]
    foot_b: List[float]
    reach_a: List[float]
    reach_b: List[float]
    approach_a: List[float]
    approach_b: List[float]
    start_events: List["PlayerRallyStartEvent"]


@dataclass(frozen=True)
class PlayerRallyStartEvent:
    trigger_sample_idx: int
    segment_start_sample_idx: int
    trigger_timestamp: float
    segment_start_timestamp: float
    server_role: str
    reason: str
    score: float
    serve_driver_a: float
    serve_driver_b: float
    react_to_a: float
    react_to_b: float


@dataclass(frozen=True)
class PlayerRallyStartCandidate:
    sample_idx: int
    timestamp: float
    role: str
    score: float
    prep_score: float
    launch_score: float
    opponent_ready_score: float
    dominance_ratio: float
    episode_start_sample_idx: int
    episode_end_sample_idx: int
    episode_peak_sample_idx: int
    episode_peak_score: float
    crouch_score: float
    reach_score: float
    serve_score: float
    upper_body_score: float
    footwork_score: float
    ready_pair_score: float = 0.0
    pre_ready_mean: float = 0.0
    pre_live_peak: float = 0.0
    server_action_score: float = 0.0
    server_peak_score: float = 0.0
    server_growth_score: float = 0.0
    server_peak_delay_sec: float = 0.0
    receiver_peak_score: float = 0.0
    live_peak_score: float = 0.0


def _merge_player_start_candidates(
    timestamps: np.ndarray,
    candidates: List[PlayerRallyStartCandidate],
    *,
    same_role_merge_window_sec: float = 1.20,
    cross_role_merge_window_sec: float = 0.70,
) -> List[PlayerRallyStartCandidate]:
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda item: item.timestamp)

    def cluster_gap_seconds(prev: PlayerRallyStartCandidate, curr: PlayerRallyStartCandidate) -> float:
        prev_end_idx = max(0, min(int(prev.episode_end_sample_idx), len(timestamps) - 1))
        curr_start_idx = max(0, min(int(curr.episode_start_sample_idx), len(timestamps) - 1))
        return float(timestamps[curr_start_idx] - timestamps[prev_end_idx])

    def merge_cluster(
        representative: PlayerRallyStartCandidate,
        other: PlayerRallyStartCandidate,
    ) -> PlayerRallyStartCandidate:
        peak_idx = representative.episode_peak_sample_idx
        peak_score = representative.episode_peak_score
        if other.episode_peak_score > peak_score:
            peak_idx = other.episode_peak_sample_idx
            peak_score = other.episode_peak_score
        return replace(
            representative,
            episode_start_sample_idx=min(representative.episode_start_sample_idx, other.episode_start_sample_idx),
            episode_end_sample_idx=max(representative.episode_end_sample_idx, other.episode_end_sample_idx),
            episode_peak_sample_idx=int(peak_idx),
            episode_peak_score=float(peak_score),
        )

    def keep_earlier_prep_anchor(
        prev: PlayerRallyStartCandidate,
        curr: PlayerRallyStartCandidate,
    ) -> bool:
        peak_lead = int(prev.episode_peak_sample_idx) - int(prev.sample_idx)
        prev_prep_margin = float(prev.prep_score - prev.launch_score)
        prev_action = max(prev.launch_score, prev.serve_score, prev.upper_body_score, prev.footwork_score)
        curr_action = max(curr.launch_score, curr.serve_score, curr.upper_body_score, curr.footwork_score)
        return bool(
            peak_lead >= 1
            and prev.prep_score >= 0.72
            and prev.crouch_score >= 0.68
            and prev.reach_score >= 0.58
            and prev_prep_margin >= 0.18
            and prev.launch_score <= 0.46
            and prev.serve_score <= 0.45
            and prev.upper_body_score <= 0.45
            and prev.footwork_score <= 0.78
            and curr.timestamp > prev.timestamp
            and curr_action >= 0.56
            and curr_action >= (prev_action + 0.12)
        )

    merged: List[PlayerRallyStartCandidate] = []
    for candidate in sorted_candidates:
        if merged:
            prev = merged[-1]
            if candidate.role == prev.role:
                timestamp_gap_sec = float(candidate.timestamp - prev.timestamp)
                extend_merge_for_prep_anchor = bool(
                    timestamp_gap_sec >= same_role_merge_window_sec
                    and cluster_gap_seconds(prev, candidate) < same_role_merge_window_sec
                    and keep_earlier_prep_anchor(prev, candidate)
                )
                should_merge = timestamp_gap_sec < same_role_merge_window_sec or extend_merge_for_prep_anchor
                if should_merge:
                    if keep_earlier_prep_anchor(prev, candidate):
                        merged[-1] = merge_cluster(prev, candidate)
                        continue
                    prev_rank = (prev.episode_peak_score, prev.score, prev.launch_score)
                    curr_rank = (candidate.episode_peak_score, candidate.score, candidate.launch_score)
                    representative = candidate if curr_rank > prev_rank else prev
                    merged[-1] = merge_cluster(representative, candidate if representative is prev else prev)
                    continue
            else:
                if (candidate.timestamp - prev.timestamp) < cross_role_merge_window_sec:
                    prev_rank = (prev.episode_peak_score, prev.score, prev.launch_score)
                    curr_rank = (candidate.episode_peak_score, candidate.score, candidate.launch_score)
                    if curr_rank > prev_rank:
                        merged[-1] = candidate
                    continue
        merged.append(candidate)

    return merged


def _dedupe_player_sandwich_start_candidates(
    timestamps: np.ndarray,
    confirmed_candidates: List[PlayerRallyStartCandidate],
) -> List[PlayerRallyStartCandidate]:
    if not confirmed_candidates:
        return []

    deduped_candidates: List[PlayerRallyStartCandidate] = []
    for candidate in confirmed_candidates:
        if deduped_candidates:
            prev = deduped_candidates[-1]
            gap_sec = float(candidate.timestamp - prev.timestamp)
            prev_end_idx = max(0, min(int(prev.episode_end_sample_idx), len(timestamps) - 1))
            curr_start_idx = max(0, min(int(candidate.episode_start_sample_idx), len(timestamps) - 1))
            episode_gap_sec = float(timestamps[curr_start_idx] - timestamps[prev_end_idx])
            followup_duplicate_pattern = bool(
                gap_sec <= 3.0
                and candidate.pre_live_peak >= 0.70
                and candidate.pre_ready_mean <= 0.25
                and candidate.live_peak_score >= 0.95
            )
            upright_duplicate_pattern = bool(
                gap_sec <= 4.0
                and candidate.pre_live_peak <= 0.45
                and candidate.pre_ready_mean >= 0.40
                and candidate.receiver_peak_score >= 0.90
                and candidate.live_peak_score >= 0.95
                and candidate.crouch_score <= 0.75
            )
            low_dominance_duplicate_pattern = bool(
                gap_sec <= 2.5
                and candidate.pre_live_peak >= 0.75
                and candidate.launch_score >= 0.50
                and candidate.dominance_ratio <= 1.30
                and candidate.live_peak_score >= 0.78
            )
            same_role_live_followup_duplicate_pattern = bool(
                prev.role == candidate.role
                and gap_sec <= 2.2
                and episode_gap_sec <= 0.35
                and prev.live_peak_score >= 0.95
                and prev.server_peak_score >= 0.85
                and candidate.pre_live_peak >= 0.95
                and candidate.pre_ready_mean <= 0.40
                and candidate.ready_pair_score >= 0.60
                and candidate.server_action_score >= 0.44
                and candidate.receiver_peak_score >= 0.95
                and candidate.live_peak_score >= 0.95
                and candidate.dominance_ratio <= 1.30
            )
            if (
                followup_duplicate_pattern
                or upright_duplicate_pattern
                or low_dominance_duplicate_pattern
                or same_role_live_followup_duplicate_pattern
            ):
                continue
        deduped_candidates.append(candidate)

    return deduped_candidates


def _infer_player_serve_mode_from_starter_roles(starter_roles: Sequence[str]) -> Literal["double", "single"]:
    runs: List[int] = []
    prev_role: Optional[str] = None
    current_len = 0
    for role in starter_roles:
        if role not in {"A", "B"}:
            continue
        if role == prev_role:
            current_len += 1
            continue
        if current_len > 0:
            runs.append(current_len)
        prev_role = role
        current_len = 1
    if current_len > 0:
        runs.append(current_len)
    if not runs:
        return "double"

    double_score = 0.0
    single_score = 0.0
    for run_len in runs:
        if run_len == 2:
            double_score += 1.0
            single_score -= 1.0
            continue
        if run_len == 1:
            double_score += 0.20
            single_score += 1.0
            continue
        double_score -= 0.65 * float(run_len - 2)
        single_score -= 1.10 * float(run_len - 1)

    return "double" if double_score >= single_score else "single"


def _infer_forced_let_indices_from_starter_roles(
    starter_roles: Sequence[str],
    point_likelihoods: Sequence[float],
    *,
    serve_mode: Literal["double", "single"],
) -> Set[int]:
    let_indices: Set[int] = set()
    legal_limit = 2 if serve_mode == "double" else 1

    run_start = 0
    sample_count = min(len(starter_roles), len(point_likelihoods))
    while run_start < sample_count:
        role = starter_roles[run_start]
        run_end = run_start + 1
        while run_end < sample_count and starter_roles[run_end] == role:
            run_end += 1

        run_len = run_end - run_start
        forced_let_count = max(0, run_len - legal_limit)
        if forced_let_count > 0:
            if legal_limit == 1:
                let_indices.update(range(run_start, run_end - 1))
            else:
                prefix_indices = list(range(run_start, run_end - 1))
                if prefix_indices:
                    # Under the exact-starter assumption, overflow starts in a same-server run
                    # are more likely to be early LET replays, while the later starters are the
                    # legal serves that complete the turn. Bias the kept legal start toward the
                    # latest strong prefix candidate instead of the strongest raw point score.
                    latest_prefix_idx = prefix_indices[-1]
                    best_prefix_score = max(float(point_likelihoods[idx]) for idx in prefix_indices)
                    legal_candidates = [
                        idx for idx in prefix_indices
                        if float(point_likelihoods[idx]) >= (best_prefix_score - 1.25)
                    ]
                    best_legal_idx = max(legal_candidates or [latest_prefix_idx])
                    let_indices.update(idx for idx in prefix_indices if idx != best_legal_idx)

        run_start = run_end

    return let_indices


def _repair_double_serve_role_singletons(
    candidates: Sequence[PlayerRallyStartCandidate],
) -> List[PlayerRallyStartCandidate]:
    if len(candidates) < 5:
        return list(candidates)

    repaired = list(candidates)

    def is_suspicious_edge(candidate: PlayerRallyStartCandidate) -> bool:
        return bool(
            candidate.pre_live_peak >= 0.80
            and candidate.launch_score <= 0.52
            and candidate.server_peak_score <= 0.55
            and candidate.server_growth_score <= 0.12
            and candidate.receiver_peak_score >= 0.55
            and candidate.live_peak_score <= 0.80
        )

    while True:
        roles = [candidate.role for candidate in repaired]
        if _infer_player_serve_mode_from_starter_roles(roles) != "double":
            break

        runs: List[Tuple[str, int, int]] = []
        run_start = 0
        while run_start < len(repaired):
            role = repaired[run_start].role
            run_end = run_start + 1
            while run_end < len(repaired) and repaired[run_end].role == role:
                run_end += 1
            runs.append((role, run_start, run_end))
            run_start = run_end

        flipped = False
        for idx in range(len(runs) - 2):
            left_role, left_start, left_end = runs[idx]
            mid_role, mid_start, mid_end = runs[idx + 1]
            right_role, right_start, right_end = runs[idx + 2]
            left_len = left_end - left_start
            mid_len = mid_end - mid_start
            right_len = right_end - right_start
            if left_role != right_role or left_role == mid_role or mid_len != 1:
                continue

            flip_idx: Optional[int] = None
            if left_len == 3 and right_len == 2:
                flip_idx = left_end - 1
            elif left_len == 2 and right_len == 3:
                flip_idx = right_start
            else:
                continue

            candidate = repaired[flip_idx]
            if not is_suspicious_edge(candidate):
                continue

            repaired[flip_idx] = replace(candidate, role=str(mid_role))
            flipped = True
            break

        if not flipped:
            break

    return repaired


def _calc_wrist_velocity(
    curr_kpts: np.ndarray,
    prev_kpts: np.ndarray | None,
    player_h: float,
    *,
    gap_frames: int = 1,
) -> float:
    if prev_kpts is None or player_h <= 0:
        return 0.0

    norm_gap = max(1.0, float(gap_frames))
    valid_vels: List[float] = []
    for wrist_idx in (9, 10):
        cx, cy = curr_kpts[wrist_idx]
        px, py = prev_kpts[wrist_idx]
        if cx <= 0 or cy <= 0 or px <= 0 or py <= 0:
            continue
        dist = float(np.linalg.norm(curr_kpts[wrist_idx] - prev_kpts[wrist_idx]))
        norm_v = dist / (float(player_h) * norm_gap)
        if norm_v < 0.75:
            valid_vels.append(norm_v)
    return max(valid_vels) if valid_vels else 0.0


def _mean_valid_keypoint(kpts: np.ndarray, indices: Tuple[int, ...]) -> Optional[np.ndarray]:
    points: List[np.ndarray] = []
    for idx in indices:
        if idx >= len(kpts):
            continue
        x, y = kpts[idx]
        if x <= 0 or y <= 0:
            continue
        points.append(np.array([float(x), float(y)], dtype=np.float32))
    if not points:
        return None
    return np.mean(np.stack(points, axis=0), axis=0)


def _pair_distance_norm(kpts: np.ndarray, idx_a: int, idx_b: int, scale_h: float) -> float:
    if idx_a >= len(kpts) or idx_b >= len(kpts) or scale_h <= 0:
        return 0.0
    ax, ay = kpts[idx_a]
    bx, by = kpts[idx_b]
    if ax <= 0 or ay <= 0 or bx <= 0 or by <= 0:
        return 0.0
    return float(np.linalg.norm(np.array([ax, ay], dtype=np.float32) - np.array([bx, by], dtype=np.float32)) / scale_h)


def _calc_keypoint_group_velocity(
    curr_kpts: np.ndarray,
    prev_kpts: np.ndarray | None,
    indices: Tuple[int, ...],
    player_h: float,
    *,
    gap_frames: int = 1,
) -> float:
    if prev_kpts is None or player_h <= 0:
        return 0.0

    norm_gap = max(1.0, float(gap_frames))
    valid_vels: List[float] = []
    for idx in indices:
        if idx >= len(curr_kpts) or idx >= len(prev_kpts):
            continue
        cx, cy = curr_kpts[idx]
        px, py = prev_kpts[idx]
        if cx <= 0 or cy <= 0 or px <= 0 or py <= 0:
            continue
        dist = float(np.linalg.norm(curr_kpts[idx] - prev_kpts[idx]))
        valid_vels.append(dist / (float(player_h) * norm_gap))
    return max(valid_vels) if valid_vels else 0.0


def _fallback_lower_anchor(obs: TrackletObservation) -> np.ndarray:
    return np.array(
        [
            (float(obs.box[0]) + float(obs.box[2])) / 2.0,
            float(obs.box[3]),
        ],
        dtype=np.float32,
    )


def _lower_anchor(obs: TrackletObservation) -> np.ndarray:
    anchor = _mean_valid_keypoint(obs.keypoints, (15, 16))
    if anchor is not None:
        return anchor
    anchor = _mean_valid_keypoint(obs.keypoints, (13, 14))
    if anchor is not None:
        return anchor
    return _fallback_lower_anchor(obs)


def _calc_box_center_velocity(
    curr_box: Tuple[int, int, int, int],
    prev_box: Tuple[int, int, int, int],
    player_h: float,
    *,
    gap_frames: int = 1,
) -> float:
    if player_h <= 0:
        return 0.0

    curr_center = np.array([(curr_box[0] + curr_box[2]) / 2.0, (curr_box[1] + curr_box[3]) / 2.0], dtype=np.float32)
    prev_center = np.array([(prev_box[0] + prev_box[2]) / 2.0, (prev_box[1] + prev_box[3]) / 2.0], dtype=np.float32)
    norm_gap = max(1.0, float(gap_frames))
    return float(np.linalg.norm(curr_center - prev_center) / (float(player_h) * norm_gap))


def _calc_role_crouch_raw(obs: TrackletObservation) -> float:
    box_h = max(1.0, float(obs.box[3] - obs.box[1]))
    box_w = max(1.0, float(obs.box[2] - obs.box[0]))
    aspect = float(box_w / box_h)
    ankle_span = _pair_distance_norm(obs.keypoints, 15, 16, box_h)
    if ankle_span <= 0:
        ankle_span = _pair_distance_norm(obs.keypoints, 13, 14, box_h)

    hip_center = _mean_valid_keypoint(obs.keypoints, (11, 12))
    ankle_center = _mean_valid_keypoint(obs.keypoints, (15, 16))
    knee_center = _mean_valid_keypoint(obs.keypoints, (13, 14))
    lower_center = ankle_center if ankle_center is not None else knee_center
    leg_extension = 0.0
    if hip_center is not None and lower_center is not None:
        leg_extension = float(max(0.0, lower_center[1] - hip_center[1]) / box_h)
    crouch_compactness = max(0.0, 0.82 - leg_extension)
    return float((0.55 * aspect) + (0.25 * ankle_span) + (0.20 * crouch_compactness))


def _calc_role_serve_raw(
    curr_obs: TrackletObservation,
    prev_obs: Optional[TrackletObservation],
    *,
    gap_frames: int = 1,
) -> float:
    box_h = max(1.0, float(curr_obs.box[3] - curr_obs.box[1]))
    wrist_v = _calc_wrist_velocity(
        curr_obs.keypoints,
        None if prev_obs is None else prev_obs.keypoints,
        box_h,
        gap_frames=max(1, gap_frames),
    )
    shoulders = _mean_valid_keypoint(curr_obs.keypoints, (5, 6))
    nose = _mean_valid_keypoint(curr_obs.keypoints, (0,))
    arm_lift = 0.0
    for wrist_idx in (9, 10):
        if wrist_idx >= len(curr_obs.keypoints):
            continue
        wx, wy = curr_obs.keypoints[wrist_idx]
        if wx <= 0 or wy <= 0:
            continue
        if shoulders is not None:
            arm_lift = max(arm_lift, float(max(0.0, shoulders[1] - wy) / box_h))
        if nose is not None:
            arm_lift = max(arm_lift, float(max(0.0, nose[1] - wy) / box_h) * 1.15)
    return float((0.55 * wrist_v) + (0.45 * arm_lift))


def _calc_role_upper_body_raw(
    curr_obs: TrackletObservation,
    prev_obs: Optional[TrackletObservation],
    *,
    gap_frames: int = 1,
) -> float:
    box_h = max(1.0, float(curr_obs.box[3] - curr_obs.box[1]))
    serve_like = _calc_role_serve_raw(curr_obs, prev_obs, gap_frames=max(1, gap_frames))
    elbow_v = _calc_keypoint_group_velocity(
        curr_obs.keypoints,
        None if prev_obs is None else prev_obs.keypoints,
        (7, 8),
        box_h,
        gap_frames=max(1, gap_frames),
    )
    shoulders = _mean_valid_keypoint(curr_obs.keypoints, (5, 6))
    prev_shoulders = None if prev_obs is None else _mean_valid_keypoint(prev_obs.keypoints, (5, 6))
    shoulder_shift = 0.0
    if shoulders is not None and prev_shoulders is not None:
        shoulder_shift = float(np.linalg.norm(shoulders - prev_shoulders) / (box_h * max(1.0, float(gap_frames))))
    wrist_span = _pair_distance_norm(curr_obs.keypoints, 9, 10, box_h)
    prev_wrist_span = 0.0 if prev_obs is None else _pair_distance_norm(prev_obs.keypoints, 9, 10, box_h)
    span_change = abs(wrist_span - prev_wrist_span) / max(1.0, float(gap_frames))
    return float((0.50 * serve_like) + (0.22 * elbow_v) + (0.18 * span_change) + (0.10 * shoulder_shift))


def _calc_role_footwork_raw(
    curr_obs: TrackletObservation,
    prev_obs: Optional[TrackletObservation],
    *,
    gap_frames: int = 1,
) -> float:
    if prev_obs is None:
        return 0.0

    box_h = max(1.0, float(curr_obs.box[3] - curr_obs.box[1]))
    norm_gap = max(1.0, float(gap_frames))
    curr_anchor = _lower_anchor(curr_obs)
    prev_anchor = _lower_anchor(prev_obs)
    anchor_delta = np.abs(curr_anchor - prev_anchor) / (box_h * norm_gap)
    anchor_motion = float(anchor_delta[0] + (0.65 * anchor_delta[1]))
    stance_span = _pair_distance_norm(curr_obs.keypoints, 15, 16, box_h)
    if stance_span <= 0:
        stance_span = _pair_distance_norm(curr_obs.keypoints, 13, 14, box_h)
    prev_stance_span = _pair_distance_norm(prev_obs.keypoints, 15, 16, box_h)
    if prev_stance_span <= 0:
        prev_stance_span = _pair_distance_norm(prev_obs.keypoints, 13, 14, box_h)
    stance_change = abs(stance_span - prev_stance_span) / norm_gap
    center_v = _calc_box_center_velocity(
        curr_obs.box,
        prev_obs.box,
        box_h,
        gap_frames=max(1, gap_frames),
    )
    return float((0.45 * anchor_motion) + (0.30 * stance_change) + (0.25 * center_v))


def _calc_role_reach_raw(
    curr_obs: TrackletObservation,
    _prev_obs: Optional[TrackletObservation],
    *,
    role: str,
    gap_frames: int = 1,
) -> float:
    _ = gap_frames
    box_h = max(1.0, float(curr_obs.box[3] - curr_obs.box[1]))
    box_w = max(1.0, float(curr_obs.box[2] - curr_obs.box[0]))
    shoulders = _mean_valid_keypoint(curr_obs.keypoints, (5, 6))
    hips = _mean_valid_keypoint(curr_obs.keypoints, (11, 12))
    if shoulders is None:
        shoulders = np.array(
            [
                (float(curr_obs.box[0]) + float(curr_obs.box[2])) / 2.0,
                float(curr_obs.box[1]) + (0.30 * box_h),
            ],
            dtype=np.float32,
        )
    if hips is None:
        hips = np.array(
            [
                shoulders[0],
                float(curr_obs.box[1]) + (0.63 * box_h),
            ],
            dtype=np.float32,
        )

    best = 0.0
    for wrist_idx in (9, 10):
        if wrist_idx >= len(curr_obs.keypoints):
            continue
        wx, wy = curr_obs.keypoints[wrist_idx]
        if wx <= 0 or wy <= 0:
            continue
        toward_table = (float(wx) - float(shoulders[0])) / box_w if role == "A" else (float(shoulders[0]) - float(wx)) / box_w
        if toward_table <= 0:
            continue
        torso_mid_y = float((shoulders[1] + hips[1]) / 2.0)
        vertical_align = 1.0 - min(1.0, abs(float(wy) - torso_mid_y) / max(1.0, 0.80 * box_h))
        best = max(best, float(max(0.0, toward_table) * max(0.0, vertical_align)))
    return best


def _calc_role_net_approach_raw(
    curr_obs: TrackletObservation,
    prev_obs: Optional[TrackletObservation],
    *,
    roi: TableROI,
    gap_frames: int = 1,
) -> float:
    if prev_obs is None:
        return 0.0

    table_cx = float(roi.x + (roi.w / 2.0))
    dist_prev = abs(float(prev_obs.center[0]) - table_cx)
    dist_curr = abs(float(curr_obs.center[0]) - table_cx)
    toward_net = max(0.0, dist_prev - dist_curr) / max(1.0, float(roi.w) * max(1.0, float(gap_frames)))
    return float(toward_net)


def _calc_role_face_hidden_raw(
    curr_obs: TrackletObservation,
    _prev_obs: Optional[TrackletObservation],
    *,
    gap_frames: int = 1,
) -> float:
    _ = gap_frames
    box_w = max(1.0, float(curr_obs.box[2] - curr_obs.box[0]))
    box_h = max(1.0, float(curr_obs.box[3] - curr_obs.box[1]))
    shoulders = _mean_valid_keypoint(curr_obs.keypoints, (5, 6))
    hips = _mean_valid_keypoint(curr_obs.keypoints, (11, 12))
    if shoulders is None or hips is None:
        return 0.0

    face_pts = [_mean_valid_keypoint(curr_obs.keypoints, (idx,)) for idx in (0, 1, 2)]
    visible_face_pts = sum(1 for pt in face_pts if pt is not None)
    missing_face = 1.0 - (float(visible_face_pts) / 3.0)

    nose = face_pts[0]
    head_down = 0.0
    if nose is not None:
        head_down = float(np.clip((float(nose[1]) - float(shoulders[1])) / (0.22 * box_h), 0.0, 1.0))

    shoulder_left = _mean_valid_keypoint(curr_obs.keypoints, (5,))
    shoulder_right = _mean_valid_keypoint(curr_obs.keypoints, (6,))
    hip_left = _mean_valid_keypoint(curr_obs.keypoints, (11,))
    hip_right = _mean_valid_keypoint(curr_obs.keypoints, (12,))

    profile_turn = 0.0
    if shoulder_left is not None and shoulder_right is not None and hip_left is not None and hip_right is not None:
        shoulder_span = abs(float(shoulder_left[0]) - float(shoulder_right[0])) / box_w
        hip_span = abs(float(hip_left[0]) - float(hip_right[0])) / box_w
        # A strongly collapsed shoulder/hip span is a better proxy for "turned away"
        # than missing face keypoints on this footage, where the pose model often still
        # hallucinates nose/eye points after the player has already disengaged.
        profile_turn = float(
            np.clip(
                1.0 - max(shoulder_span / 0.24, hip_span / 0.20),
                0.0,
                1.0,
            )
        )

    face_hidden = (0.72 * missing_face) + (0.28 * head_down)
    return float(np.clip(max(face_hidden, profile_turn), 0.0, 1.0))


def _calc_role_face_touch_raw(
    curr_obs: TrackletObservation,
    _prev_obs: Optional[TrackletObservation],
    *,
    gap_frames: int = 1,
) -> float:
    _ = gap_frames
    box_h = max(1.0, float(curr_obs.box[3] - curr_obs.box[1]))
    nose = _mean_valid_keypoint(curr_obs.keypoints, (0,))
    if nose is None:
        eyes = _mean_valid_keypoint(curr_obs.keypoints, (1, 2))
        nose = eyes
    if nose is None:
        shoulders = _mean_valid_keypoint(curr_obs.keypoints, (5, 6))
        if shoulders is None:
            return 0.0
        nose = np.array([float(shoulders[0]), float(shoulders[1] - (0.16 * box_h))], dtype=np.float32)

    best = 0.0
    for wrist_idx in (9, 10):
        if wrist_idx >= len(curr_obs.keypoints):
            continue
        wx, wy = curr_obs.keypoints[wrist_idx]
        if wx <= 0 or wy <= 0:
            continue
        dist = float(np.linalg.norm(np.array([float(wx), float(wy)], dtype=np.float32) - nose) / box_h)
        touch = float(np.clip(1.0 - (dist / 0.26), 0.0, 1.0))
        best = max(best, touch)
    return float(best)


def _is_player_near_table(box_xyxy: np.ndarray, roi: TableROI, margin_px: int) -> bool:
    x1, _y1, x2, _y2 = [float(v) for v in box_xyxy]
    tx, _ty, tw, _th = roi.as_tuple()
    return (x1 < tx + tw + margin_px) and (x2 > tx - margin_px)


def _smooth_and_normalize(values: List[float]) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float32)

    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 3:
        p10, p95 = float(arr.min()), float(arr.max())
        return np.clip((arr - p10) / (p95 - p10 + 1e-6), 0.0, 1.0)

    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    smooth = np.convolve(arr, kernel, mode="same")
    p10, p95 = np.percentile(smooth, 10), np.percentile(smooth, 95)
    return np.clip((smooth - p10) / (p95 - p10 + 1e-6), 0.0, 1.0)


def _calc_role_motion_energy(
    curr_obs: TrackletObservation,
    prev_obs: Optional[TrackletObservation],
    *,
    gap_frames: int = 1,
) -> float:
    if prev_obs is None:
        return 0.0

    player_h = max(1.0, float(curr_obs.box[3] - curr_obs.box[1]))
    wrist_v = _calc_wrist_velocity(
        curr_obs.keypoints,
        prev_obs.keypoints,
        player_h,
        gap_frames=max(1, gap_frames),
    )
    center_v = _calc_box_center_velocity(
        curr_obs.box,
        prev_obs.box,
        player_h,
        gap_frames=max(1, gap_frames),
    )
    return max(wrist_v, min(center_v, 1.0))


def _build_role_energy_series(
    frame_indices: List[int],
    role_frames: Dict[int, Dict[str, TrackletObservation]],
    role_state_frames: Dict[int, Dict[str, str]],
    *,
    role: str,
    occluded_hold_samples: int = 3,
    occluded_decay: float = 0.55,
) -> List[float]:
    energies: List[float] = []
    prev_obs: Optional[TrackletObservation] = None
    prev_visible_frame_idx: Optional[int] = None
    prev_energy = 0.0

    for frame_idx in frame_indices:
        obs = role_frames.get(frame_idx, {}).get(role)
        state = role_state_frames.get(frame_idx, {}).get(role)

        if obs is not None:
            gap_frames = 1 if prev_visible_frame_idx is None else max(1, frame_idx - prev_visible_frame_idx)
            energy = _calc_role_motion_energy(obs, prev_obs, gap_frames=gap_frames)
            prev_obs = obs
            prev_visible_frame_idx = frame_idx
            prev_energy = energy
        elif state == "occluded" and prev_visible_frame_idx is not None:
            sample_gap = max(1, frame_idx - prev_visible_frame_idx)
            if sample_gap <= occluded_hold_samples:
                energy = prev_energy * float(occluded_decay ** max(0, sample_gap - 1))
            else:
                energy = 0.0
        else:
            energy = 0.0
            if state != "occluded":
                prev_energy = 0.0

        energies.append(float(energy))

    return energies


def _build_role_feature_series(
    frame_indices: List[int],
    role_frames: Dict[int, Dict[str, TrackletObservation]],
    role_state_frames: Dict[int, Dict[str, str]],
    *,
    role: str,
    feature_fn,
    occluded_hold_samples: int = 2,
    occluded_decay: float = 0.65,
) -> List[float]:
    values: List[float] = []
    prev_obs: Optional[TrackletObservation] = None
    prev_visible_frame_idx: Optional[int] = None
    prev_value = 0.0

    for frame_idx in frame_indices:
        obs = role_frames.get(frame_idx, {}).get(role)
        state = role_state_frames.get(frame_idx, {}).get(role)

        if obs is not None:
            gap_frames = 1 if prev_visible_frame_idx is None else max(1, frame_idx - prev_visible_frame_idx)
            value = float(feature_fn(obs, prev_obs, gap_frames=max(1, gap_frames)))
            prev_obs = obs
            prev_visible_frame_idx = frame_idx
            prev_value = value
        elif state == "occluded" and prev_visible_frame_idx is not None:
            sample_gap = max(1, frame_idx - prev_visible_frame_idx)
            if sample_gap <= occluded_hold_samples:
                value = prev_value * float(occluded_decay ** max(0, sample_gap - 1))
            else:
                value = 0.0
        else:
            value = 0.0
            if state != "occluded":
                prev_value = 0.0

        values.append(float(value))

    return values


_POSE_BATCH_SIZE = 16


def _collect_role_tracker_energies(
    cap: cv2.VideoCapture,
    person_model: YOLO,
    *,
    roi: TableROI,
    fps: float,
    stride: int,
    device: str,
    player_margin_px: int,
    frame_w: int,
    frame_h: int,
) -> RoleTrackerSeries:
    tx, ty, tw, th = roi.as_tuple()
    tracker = OfflinePlayerTracker(roi, frame_w=frame_w, frame_h=frame_h)

    timestamps: List[float] = []
    frame_indices: List[int] = []
    table_energies: List[float] = []
    prev_table_gray: Optional[np.ndarray] = None
    frame_idx = 0

    # Batch buffers
    batch_frames: List[np.ndarray] = []
    batch_fidx: List[int] = []
    batch_ts: List[float] = []
    batch_te: List[float] = []

    def _flush_batch() -> None:
        if not batch_frames:
            return
        results = person_model.predict(
            batch_frames,
            classes=[0],
            device=device,
            verbose=False,
            half=True,
        )
        for result, fidx, frame in zip(results, batch_fidx, batch_frames):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
                keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
                if boxes.size > 0:
                    near_mask = [_is_player_near_table(box, roi, player_margin_px) for box in boxes]
                    mask_arr = np.asarray(near_mask, dtype=bool)
                    boxes = boxes[mask_arr]
                    if confs is not None:
                        confs = confs[mask_arr]
                    if keypoints is not None:
                        keypoints = keypoints[mask_arr]
                if boxes.size > 0:
                    detections = tracker.build_detections(
                        frame,
                        frame_idx=fidx,
                        boxes_xyxy=boxes,
                        keypoints_xy=keypoints,
                        confidences=confs,
                    )
                    tracker.add_frame_detections(detections)
        timestamps.extend(batch_ts)
        frame_indices.extend(batch_fidx)
        table_energies.extend(batch_te)
        batch_frames.clear()
        batch_fidx.clear()
        batch_ts.clear()
        batch_te.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        table_crop = frame[ty : ty + th, tx : tx + tw]
        table_gray = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)
        te = 0.0 if prev_table_gray is None else float(np.mean(cv2.absdiff(table_gray, prev_table_gray)))
        prev_table_gray = table_gray

        batch_frames.append(frame.copy())
        batch_fidx.append(int(frame_idx))
        batch_ts.append(float(frame_idx / fps))
        batch_te.append(te)

        if len(batch_frames) >= _POSE_BATCH_SIZE:
            _flush_batch()

        frame_idx += 1

    _flush_batch()

    tracking_result = tracker.finish()
    role_hold_samples = max(2, int(round(6 / max(1, stride))))
    player_a_energies = _build_role_energy_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        occluded_hold_samples=role_hold_samples,
    )
    player_b_energies = _build_role_energy_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        occluded_hold_samples=role_hold_samples,
    )
    player_a_crouch = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, _prev, gap_frames=1: _calc_role_crouch_raw(obs),
        occluded_hold_samples=role_hold_samples,
        occluded_decay=0.72,
    )
    player_b_crouch = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, _prev, gap_frames=1: _calc_role_crouch_raw(obs),
        occluded_hold_samples=role_hold_samples,
        occluded_decay=0.72,
    )
    player_a_serve = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_serve_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.55,
    )
    player_b_serve = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_serve_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.55,
    )
    player_a_upper = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_upper_body_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.58,
    )
    player_b_upper = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_upper_body_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.58,
    )
    player_a_footwork = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_footwork_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=role_hold_samples,
        occluded_decay=0.62,
    )
    player_b_footwork = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_footwork_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=role_hold_samples,
        occluded_decay=0.62,
    )
    player_a_reach = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_reach_raw(obs, prev, role="A", gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.55,
    )
    player_b_reach = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_reach_raw(obs, prev, role="B", gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.55,
    )
    player_a_net_approach = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_net_approach_raw(obs, prev, roi=roi, gap_frames=gap_frames),
        occluded_hold_samples=role_hold_samples,
        occluded_decay=0.68,
    )
    player_b_net_approach = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_net_approach_raw(obs, prev, roi=roi, gap_frames=gap_frames),
        occluded_hold_samples=role_hold_samples,
        occluded_decay=0.68,
    )
    player_a_face_hidden = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_face_hidden_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.55,
    )
    player_b_face_hidden = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_face_hidden_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.55,
    )
    player_a_face_touch = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_face_touch_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.58,
    )
    player_b_face_touch = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_face_touch_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.58,
    )
    return RoleTrackerSeries(
        timestamps=timestamps,
        table_energies=table_energies,
        player_a_energies=player_a_energies,
        player_b_energies=player_b_energies,
        player_a_crouch_scores=player_a_crouch,
        player_b_crouch_scores=player_b_crouch,
        player_a_serve_scores=player_a_serve,
        player_b_serve_scores=player_b_serve,
        player_a_upper_body_scores=player_a_upper,
        player_b_upper_body_scores=player_b_upper,
        player_a_footwork_scores=player_a_footwork,
        player_b_footwork_scores=player_b_footwork,
        player_a_reach_scores=player_a_reach,
        player_b_reach_scores=player_b_reach,
        player_a_net_approach_scores=player_a_net_approach,
        player_b_net_approach_scores=player_b_net_approach,
        player_a_face_hidden_scores=player_a_face_hidden,
        player_b_face_hidden_scores=player_b_face_hidden,
        player_a_face_touch_scores=player_a_face_touch,
        player_b_face_touch_scores=player_b_face_touch,
    )


def _compute_player_state_machine_diagnostics(signals: MultiStreamSignals) -> PlayerStateMachineDiagnostics:
    empty = PlayerStateMachineDiagnostics(
        timestamps=list(signals.timestamps),
        segments=[],
        phase_by_frame=[],
        server_role_by_frame=[],
        ready_recent_flags=[],
        live_now_flags=[],
        dead_now_flags=[],
        catch_proxy_scores=[],
        quiet_after_catch_scores=[],
        ready_pair=[],
        live_pair=[],
        casual_pair=[],
        stand_pair=[],
        motion_a=[],
        motion_b=[],
        crouch_a=[],
        crouch_b=[],
        serve_a=[],
        serve_b=[],
        upper_a=[],
        upper_b=[],
        foot_a=[],
        foot_b=[],
        reach_a=[],
        reach_b=[],
        approach_a=[],
        approach_b=[],
        start_events=[],
    )
    if not signals.timestamps:
        return empty
    if not signals.player_a_crouch_scores or not signals.player_b_crouch_scores:
        return empty

    ts = np.asarray(signals.timestamps, dtype=np.float32)
    sample_count = len(ts)

    def select_raw_feature(*candidates: List[float]) -> np.ndarray:
        for series in candidates:
            if series and len(series) == sample_count:
                return np.asarray(series, dtype=np.float32)
        return np.zeros(sample_count, dtype=np.float32)

    def select_feature(*candidates: List[float]) -> np.ndarray:
        for series in candidates:
            if series and len(series) == sample_count:
                return _smooth_and_normalize(series)
        return np.zeros(sample_count, dtype=np.float32)

    motion_a_raw = select_raw_feature(signals.player_a_energies)
    motion_b_raw = select_raw_feature(signals.player_b_energies)
    upper_a_raw = select_raw_feature(
        signals.player_a_upper_body_scores,
        signals.player_a_serve_scores,
        signals.player_a_energies,
    )
    upper_b_raw = select_raw_feature(
        signals.player_b_upper_body_scores,
        signals.player_b_serve_scores,
        signals.player_b_energies,
    )
    foot_a_raw = select_raw_feature(signals.player_a_footwork_scores, signals.player_a_energies)
    foot_b_raw = select_raw_feature(signals.player_b_footwork_scores, signals.player_b_energies)
    reach_a_raw = select_raw_feature(signals.player_a_reach_scores)
    reach_b_raw = select_raw_feature(signals.player_b_reach_scores)
    approach_a_raw = select_raw_feature(signals.player_a_net_approach_scores)
    approach_b_raw = select_raw_feature(signals.player_b_net_approach_scores)

    motion_a = select_feature(signals.player_a_energies)
    motion_b = select_feature(signals.player_b_energies)
    crouch_a = select_feature(signals.player_a_crouch_scores)
    crouch_b = select_feature(signals.player_b_crouch_scores)
    serve_a = select_feature(signals.player_a_serve_scores, signals.player_a_energies)
    serve_b = select_feature(signals.player_b_serve_scores, signals.player_b_energies)
    upper_a = select_feature(
        signals.player_a_upper_body_scores,
        signals.player_a_serve_scores,
        signals.player_a_energies,
    )
    upper_b = select_feature(
        signals.player_b_upper_body_scores,
        signals.player_b_serve_scores,
        signals.player_b_energies,
    )
    foot_a = select_feature(signals.player_a_footwork_scores, signals.player_a_energies)
    foot_b = select_feature(signals.player_b_footwork_scores, signals.player_b_energies)
    reach_a = select_feature(signals.player_a_reach_scores)
    reach_b = select_feature(signals.player_b_reach_scores)
    approach_a = select_feature(signals.player_a_net_approach_scores)
    approach_b = select_feature(signals.player_b_net_approach_scores)

    ready_a = np.clip(crouch_a * (1.0 - (0.82 * np.maximum(motion_a, foot_a))), 0.0, 1.0)
    ready_b = np.clip(crouch_b * (1.0 - (0.82 * np.maximum(motion_b, foot_b))), 0.0, 1.0)
    ready_pair = np.minimum(ready_a, ready_b)

    competitive_a = np.maximum.reduce(
        [
            motion_a * (0.35 + (0.65 * crouch_a)),
            upper_a * (0.45 + (0.55 * crouch_a)),
            foot_a * (0.42 + (0.58 * crouch_a)),
        ]
    )
    competitive_b = np.maximum.reduce(
        [
            motion_b * (0.35 + (0.65 * crouch_b)),
            upper_b * (0.45 + (0.55 * crouch_b)),
            foot_b * (0.42 + (0.58 * crouch_b)),
        ]
    )
    live_pair = np.maximum(competitive_a, competitive_b)
    stand_a = np.clip((1.0 - crouch_a) * (1.0 - (0.35 * np.maximum(upper_a, foot_a))), 0.0, 1.0)
    stand_b = np.clip((1.0 - crouch_b) * (1.0 - (0.35 * np.maximum(upper_b, foot_b))), 0.0, 1.0)
    stand_pair = np.minimum(stand_a, stand_b)
    casual_a = np.clip(
        1.0 - np.maximum.reduce([0.90 * upper_a, 0.80 * foot_a, 0.70 * crouch_a, 0.55 * serve_a]),
        0.0,
        1.0,
    )
    casual_b = np.clip(
        1.0 - np.maximum.reduce([0.90 * upper_b, 0.80 * foot_b, 0.70 * crouch_b, 0.55 * serve_b]),
        0.0,
        1.0,
    )
    casual_pair = np.minimum(casual_a, casual_b)

    def _window_bounds(start_idx: int, secs: float) -> Tuple[int, int]:
        end_t = float(ts[start_idx] + secs)
        end_idx = int(np.searchsorted(ts, end_t, side="right"))
        end_idx = max(start_idx + 1, min(end_idx, len(ts)))
        return start_idx, end_idx

    def window_max(arr: np.ndarray, start_idx: int, secs: float) -> float:
        a, b = _window_bounds(start_idx, secs)
        return float(arr[a:b].max()) if b > a else float(arr[start_idx])

    def window_mean(arr: np.ndarray, start_idx: int, secs: float) -> float:
        a, b = _window_bounds(start_idx, secs)
        return float(arr[a:b].mean()) if b > a else float(arr[start_idx])

    def append_segment(
        out: List[RallySegment],
        start_idx: Optional[int],
        end_idx: Optional[int],
        *,
        label: str,
    ) -> None:
        if start_idx is None or end_idx is None or end_idx <= start_idx:
            return
        seg_start_t = float(ts[start_idx])
        seg_end_t = float(ts[end_idx])
        min_dur = 0.35 if label == "let" else 1.0
        if (seg_end_t - seg_start_t) < min_dur:
            return
        conf_window = live_pair[start_idx : end_idx + 1]
        conf_floor = 0.42 if label == "let" else 0.50
        flags = ["player_state_machine", "rally_label_point"]
        if label == "let":
            flags = ["player_state_machine", "rally_label_let", "let_no_score"]
        out.append(
            RallySegment(
                t_start=seg_start_t,
                t_end=seg_end_t,
                confidence=float(np.clip(np.median(conf_window), conf_floor, 1.0)),
                flags=flags,
            )
        )

    segments: List[RallySegment] = []
    phase = "search_ready"
    last_ready_idx: Optional[int] = None
    seg_start_idx: Optional[int] = None
    last_live_idx: Optional[int] = None
    casual_run_start_idx: Optional[int] = None
    server_role: Optional[str] = None

    ready_memory_sec = 1.15
    reaction_window_sec = 0.75
    end_casual_sec = 1.00
    let_short_sec = 1.50
    let_walk_window_sec = 0.95
    let_quiet_window_sec = 0.60

    phase_by_frame: List[str] = []
    server_role_by_frame: List[str] = []
    ready_recent_flags: List[bool] = []
    live_now_flags: List[bool] = []
    dead_now_flags: List[bool] = []
    catch_proxy_scores: List[float] = []
    quiet_after_catch_scores: List[float] = []
    start_events: List[PlayerRallyStartEvent] = []

    def record_state(
        *,
        ready_recent: bool,
        live_now: bool,
        dead_now: bool,
        catch_proxy: float = 0.0,
        quiet_after_catch: float = 0.0,
    ) -> None:
        phase_by_frame.append(str(phase))
        server_role_by_frame.append("" if server_role is None else str(server_role))
        ready_recent_flags.append(bool(ready_recent))
        live_now_flags.append(bool(live_now))
        dead_now_flags.append(bool(dead_now))
        catch_proxy_scores.append(float(catch_proxy))
        quiet_after_catch_scores.append(float(quiet_after_catch))

    for idx in range(sample_count):
        if ready_pair[idx] >= 0.24:
            last_ready_idx = idx
            if phase == "search_ready":
                phase = "ready"
        elif phase == "ready" and last_ready_idx is not None:
            if float(ts[idx] - ts[last_ready_idx]) > ready_memory_sec:
                phase = "search_ready"
                last_ready_idx = None

        ready_recent = bool(last_ready_idx is not None and float(ts[idx] - ts[last_ready_idx]) <= ready_memory_sec)
        live_now = False
        dead_now = False
        catch_proxy = 0.0
        quiet_after_catch = 0.0

        if phase in {"search_ready", "ready"} and ready_recent:
            serve_candidates: List[Tuple[float, Optional[str], str, float]] = []
            react_to_a = max(window_max(competitive_b, idx, reaction_window_sec), window_max(motion_b, idx, reaction_window_sec))
            react_to_b = max(window_max(competitive_a, idx, reaction_window_sec), window_max(motion_a, idx, reaction_window_sec))
            serve_driver_a = float((0.52 * serve_a[idx]) + (0.30 * upper_a[idx]) + (0.18 * foot_a[idx]))
            serve_driver_b = float((0.52 * serve_b[idx]) + (0.30 * upper_b[idx]) + (0.18 * foot_b[idx]))
            if serve_driver_a >= 0.33 and react_to_a >= 0.12:
                serve_candidates.append((serve_driver_a + (0.35 * react_to_a), "A", "serve_a_reaction", float(serve_driver_a + (0.35 * react_to_a))))
            if serve_driver_b >= 0.33 and react_to_b >= 0.12:
                serve_candidates.append((serve_driver_b + (0.35 * react_to_b), "B", "serve_b_reaction", float(serve_driver_b + (0.35 * react_to_b))))
            if not serve_candidates and live_pair[idx] >= 0.30 and competitive_a[idx] >= 0.13 and competitive_b[idx] >= 0.12:
                serve_candidates.append((float(live_pair[idx]), None, "dual_live_fallback", float(live_pair[idx])))

            if serve_candidates:
                _score, server_role, start_reason, start_score = max(serve_candidates, key=lambda item: item[0])
                seg_start_idx = idx
                if last_ready_idx is not None and float(ts[idx] - ts[last_ready_idx]) <= 0.40:
                    seg_start_idx = max(last_ready_idx, idx - 1)
                start_events.append(
                    PlayerRallyStartEvent(
                        trigger_sample_idx=int(idx),
                        segment_start_sample_idx=int(seg_start_idx),
                        trigger_timestamp=float(ts[idx]),
                        segment_start_timestamp=float(ts[seg_start_idx]),
                        server_role="" if server_role is None else str(server_role),
                        reason=str(start_reason),
                        score=float(start_score),
                        serve_driver_a=float(serve_driver_a),
                        serve_driver_b=float(serve_driver_b),
                        react_to_a=float(react_to_a),
                        react_to_b=float(react_to_b),
                    )
                )
                last_live_idx = idx
                casual_run_start_idx = None
                phase = "active"
                record_state(ready_recent=ready_recent, live_now=False, dead_now=False)
                continue

        if phase != "active":
            record_state(ready_recent=ready_recent, live_now=False, dead_now=False)
            continue

        live_now = bool(
            live_pair[idx] >= 0.14
            or competitive_a[idx] >= 0.11
            or competitive_b[idx] >= 0.11
            or serve_a[idx] >= 0.24
            or serve_b[idx] >= 0.24
            or (min(crouch_a[idx], crouch_b[idx]) >= 0.22 and max(competitive_a[idx], competitive_b[idx]) >= 0.09)
        )
        if live_now:
            last_live_idx = idx
            casual_run_start_idx = None

        if seg_start_idx is not None and server_role is not None:
            rally_age = float(ts[idx] - ts[seg_start_idx])
            if rally_age <= let_short_sec:
                if server_role == "A":
                    receive_reach = reach_b
                    receive_approach = approach_b
                    receive_reach_raw = reach_b_raw
                    receive_upper_raw = upper_b_raw
                    receive_foot_raw = foot_b_raw
                    receive_approach_raw = approach_b_raw
                    receive_motion_raw = motion_b_raw
                else:
                    receive_reach = reach_a
                    receive_approach = approach_a
                    receive_reach_raw = reach_a_raw
                    receive_upper_raw = upper_a_raw
                    receive_foot_raw = foot_a_raw
                    receive_approach_raw = approach_a_raw
                    receive_motion_raw = motion_a_raw
                catch_proxy = float(
                    receive_reach_raw[idx]
                    - max(
                        receive_upper_raw[idx],
                        0.90 * receive_foot_raw[idx],
                        0.65 * receive_motion_raw[idx],
                    )
                )
                quiet_after_catch = float(
                    window_mean(
                        np.maximum.reduce(
                            [
                                motion_a_raw,
                                motion_b_raw,
                                upper_a_raw,
                                upper_b_raw,
                                foot_a_raw,
                                foot_b_raw,
                            ]
                        ),
                        idx,
                        let_quiet_window_sec,
                    )
                )
                if (
                    receive_reach[idx] >= 0.18
                    and receive_reach_raw[idx] >= 0.12
                    and catch_proxy >= 0.05
                    and window_max(receive_approach, idx, let_walk_window_sec) >= 0.12
                    and window_max(receive_approach_raw, idx, let_walk_window_sec) >= 0.08
                    and quiet_after_catch <= 0.22
                ):
                    append_segment(
                        segments,
                        seg_start_idx,
                        max(idx, last_live_idx or idx),
                        label="let",
                    )
                    phase = "search_ready"
                    last_ready_idx = None
                    seg_start_idx = None
                    last_live_idx = None
                    casual_run_start_idx = None
                    server_role = None
                    record_state(
                        ready_recent=ready_recent,
                        live_now=live_now,
                        dead_now=False,
                        catch_proxy=catch_proxy,
                        quiet_after_catch=quiet_after_catch,
                    )
                    continue

        dead_now = bool(stand_pair[idx] >= 0.56 or casual_pair[idx] >= 0.64)
        if dead_now:
            if casual_run_start_idx is None:
                casual_run_start_idx = idx
            if float(ts[idx] - ts[casual_run_start_idx]) >= end_casual_sec:
                append_segment(segments, seg_start_idx, last_live_idx, label="point")
                phase = "search_ready"
                last_ready_idx = None
                seg_start_idx = None
                last_live_idx = None
                casual_run_start_idx = None
                server_role = None
            record_state(
                ready_recent=ready_recent,
                live_now=live_now,
                dead_now=dead_now,
                catch_proxy=catch_proxy,
                quiet_after_catch=quiet_after_catch,
            )
            continue

        casual_run_start_idx = None
        record_state(
            ready_recent=ready_recent,
            live_now=live_now,
            dead_now=dead_now,
            catch_proxy=catch_proxy,
            quiet_after_catch=quiet_after_catch,
        )

    if phase == "active":
        append_segment(segments, seg_start_idx, last_live_idx, label="point")

    return PlayerStateMachineDiagnostics(
        timestamps=ts.astype(np.float32).tolist(),
        segments=segments,
        phase_by_frame=phase_by_frame,
        server_role_by_frame=server_role_by_frame,
        ready_recent_flags=ready_recent_flags,
        live_now_flags=live_now_flags,
        dead_now_flags=dead_now_flags,
        catch_proxy_scores=catch_proxy_scores,
        quiet_after_catch_scores=quiet_after_catch_scores,
        ready_pair=ready_pair.astype(np.float32).tolist(),
        live_pair=live_pair.astype(np.float32).tolist(),
        casual_pair=casual_pair.astype(np.float32).tolist(),
        stand_pair=stand_pair.astype(np.float32).tolist(),
        motion_a=motion_a.astype(np.float32).tolist(),
        motion_b=motion_b.astype(np.float32).tolist(),
        crouch_a=crouch_a.astype(np.float32).tolist(),
        crouch_b=crouch_b.astype(np.float32).tolist(),
        serve_a=serve_a.astype(np.float32).tolist(),
        serve_b=serve_b.astype(np.float32).tolist(),
        upper_a=upper_a.astype(np.float32).tolist(),
        upper_b=upper_b.astype(np.float32).tolist(),
        foot_a=foot_a.astype(np.float32).tolist(),
        foot_b=foot_b.astype(np.float32).tolist(),
        reach_a=reach_a.astype(np.float32).tolist(),
        reach_b=reach_b.astype(np.float32).tolist(),
        approach_a=approach_a.astype(np.float32).tolist(),
        approach_b=approach_b.astype(np.float32).tolist(),
        start_events=start_events,
    )


def _detect_player_state_machine_rallies(signals: MultiStreamSignals) -> List[RallySegment]:
    return _compute_player_state_machine_diagnostics(signals).segments


def _compute_player_rally_start_candidates(
    diagnostics: PlayerStateMachineDiagnostics,
    *,
    trigger_score_thresh: float = 0.58,
    release_score_thresh: float = 0.50,
    same_role_merge_window_sec: float = 1.20,
    cross_role_merge_window_sec: float = 0.70,
) -> List[PlayerRallyStartCandidate]:
    if not diagnostics.timestamps:
        return []

    ts = np.asarray(diagnostics.timestamps, dtype=np.float32)
    sample_count = len(ts)

    def arr(values: List[float]) -> np.ndarray:
        if values and len(values) == sample_count:
            return np.asarray(values, dtype=np.float32)
        return np.zeros(sample_count, dtype=np.float32)

    motion_a = arr(diagnostics.motion_a)
    motion_b = arr(diagnostics.motion_b)
    crouch_a = arr(diagnostics.crouch_a)
    crouch_b = arr(diagnostics.crouch_b)
    serve_a = arr(diagnostics.serve_a)
    serve_b = arr(diagnostics.serve_b)
    upper_a = arr(diagnostics.upper_a)
    upper_b = arr(diagnostics.upper_b)
    foot_a = arr(diagnostics.foot_a)
    foot_b = arr(diagnostics.foot_b)
    reach_a = arr(diagnostics.reach_a)
    reach_b = arr(diagnostics.reach_b)

    def build_candidate(
        *,
        role: str,
        start_idx: int,
        end_idx: int,
        score: np.ndarray,
        prep: np.ndarray,
        launch: np.ndarray,
        opp_ready: np.ndarray,
        dominance_ratio: np.ndarray,
        crouch: np.ndarray,
        reach: np.ndarray,
        serve: np.ndarray,
        upper: np.ndarray,
        foot: np.ndarray,
        server_action: np.ndarray,
    ) -> PlayerRallyStartCandidate:
        peak_slice = score[start_idx : end_idx + 1]
        peak_idx = int(start_idx + int(np.argmax(peak_slice)))
        peak_score = float(score[peak_idx])
        server_peak = float(server_action[peak_idx])
        onset_floor = max(float(release_score_thresh), peak_score - 0.24)
        relaxed_onset_floor = max(float(release_score_thresh), peak_score - 0.34)
        early_onset_cap = min(0.76, max(0.54, server_peak - 0.08))
        chosen_idx = peak_idx
        calm_prep_idx: Optional[int] = None
        for idx in range(start_idx, peak_idx + 1):
            prep_margin = float(prep[idx] - launch[idx])
            if (
                prep[idx] >= 0.68
                and crouch[idx] >= 0.62
                and reach[idx] >= 0.50
                and score[idx] >= relaxed_onset_floor
                and prep_margin >= 0.02
                and server_action[idx] <= early_onset_cap
            ):
                calm_prep_idx = idx
                break

        if calm_prep_idx is not None:
            chosen_idx = calm_prep_idx
        else:
            best_rank: Optional[Tuple[float, float, float, float, float]] = None
            for idx in range(start_idx, peak_idx + 1):
                prep_margin = float(prep[idx] - launch[idx])
                action_gap = float(server_peak - server_action[idx])
                if (
                    prep[idx] < 0.70
                    or crouch[idx] < 0.64
                    or reach[idx] < 0.54
                    or score[idx] < onset_floor
                    or prep_margin < 0.10
                    or action_gap < 0.06
                ):
                    continue
                rank = (
                    prep_margin - (0.28 * float(launch[idx])) - (0.12 * float(upper[idx])) - (0.08 * float(foot[idx])),
                    float(prep[idx]),
                    float(action_gap),
                    float(-launch[idx]),
                    float(-idx),
                )
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    chosen_idx = idx

        if calm_prep_idx is None and best_rank is None:
            fallback_idx = chosen_idx
            onset_floor = max(float(trigger_score_thresh), peak_score - 0.18)
            for idx in range(start_idx, end_idx + 1):
                if (
                    prep[idx] >= 0.72
                    and launch[idx] >= 0.44
                    and score[idx] >= onset_floor
                ):
                    fallback_idx = idx
                    break
            chosen_idx = fallback_idx

        return PlayerRallyStartCandidate(
            sample_idx=int(chosen_idx),
            timestamp=float(ts[chosen_idx]),
            role=str(role),
            score=float(score[chosen_idx]),
            prep_score=float(prep[chosen_idx]),
            launch_score=float(launch[chosen_idx]),
            opponent_ready_score=float(opp_ready[chosen_idx]),
            dominance_ratio=float(dominance_ratio[chosen_idx]),
            episode_start_sample_idx=int(start_idx),
            episode_end_sample_idx=int(end_idx),
            episode_peak_sample_idx=int(peak_idx),
            episode_peak_score=float(peak_score),
            crouch_score=float(crouch[chosen_idx]),
            reach_score=float(reach[chosen_idx]),
            serve_score=float(serve[chosen_idx]),
            upper_body_score=float(upper[chosen_idx]),
            footwork_score=float(foot[chosen_idx]),
        )

    def mine_role(
        *,
        role: str,
        motion: np.ndarray,
        crouch: np.ndarray,
        serve: np.ndarray,
        upper: np.ndarray,
        foot: np.ndarray,
        reach: np.ndarray,
        opp_motion: np.ndarray,
        opp_crouch: np.ndarray,
        opp_serve: np.ndarray,
        opp_upper: np.ndarray,
        opp_foot: np.ndarray,
    ) -> List[PlayerRallyStartCandidate]:
        prep = np.clip((0.62 * crouch) + (0.38 * reach), 0.0, 1.0)
        launch = np.maximum(
            (0.45 * serve) + (0.20 * upper) + (0.10 * foot) + (0.25 * reach),
            (0.40 * serve) + (0.30 * upper) + (0.15 * foot) + (0.15 * reach),
        )
        server_action = np.maximum.reduce(
            [
                serve,
                upper,
                0.78 * foot,
                0.58 * motion,
            ]
        )
        opp_comp = np.maximum.reduce([opp_motion, opp_upper, opp_foot])
        opp_ready = np.clip(opp_crouch * (1.0 - (0.70 * opp_comp)), 0.0, 1.0)
        opp_serve_comp = np.maximum.reduce([opp_serve, opp_upper, opp_foot])
        dominance_ratio = launch / np.maximum(0.15, opp_serve_comp)
        start_cue = (reach >= 0.46) | (serve >= 0.22) | (upper >= 0.30) | (foot >= 0.46)
        gate = (crouch >= 0.48) & start_cue & (launch >= 0.32) & (dominance_ratio >= 1.02)
        score = np.clip((0.68 * prep) + (0.20 * launch) + (0.12 * opp_ready), 0.0, 1.0)

        out: List[PlayerRallyStartCandidate] = []
        active = False
        start_idx: Optional[int] = None
        for idx in range(sample_count):
            active_now = bool(gate[idx] and score[idx] >= (release_score_thresh if active else trigger_score_thresh))
            if not active:
                if active_now:
                    active = True
                    start_idx = idx
                continue

            if active_now:
                continue

            if start_idx is not None:
                out.append(
                    build_candidate(
                        role=role,
                        start_idx=start_idx,
                        end_idx=max(start_idx, idx - 1),
                        score=score,
                        prep=prep,
                        launch=launch,
                        opp_ready=opp_ready,
                        dominance_ratio=dominance_ratio,
                        crouch=crouch,
                        reach=reach,
                        serve=serve,
                        upper=upper,
                        foot=foot,
                        server_action=server_action,
                    )
                )
            active = False
            start_idx = None

        if active and start_idx is not None:
            out.append(
                build_candidate(
                    role=role,
                    start_idx=start_idx,
                    end_idx=sample_count - 1,
                    score=score,
                    prep=prep,
                    launch=launch,
                    opp_ready=opp_ready,
                    dominance_ratio=dominance_ratio,
                    crouch=crouch,
                    reach=reach,
                    serve=serve,
                    upper=upper,
                    foot=foot,
                    server_action=server_action,
                )
            )

        return out

    candidates = mine_role(
        role="A",
        motion=motion_a,
        crouch=crouch_a,
        serve=serve_a,
        upper=upper_a,
        foot=foot_a,
        reach=reach_a,
        opp_motion=motion_b,
        opp_crouch=crouch_b,
        opp_serve=serve_b,
        opp_upper=upper_b,
        opp_foot=foot_b,
    )
    candidates.extend(
        mine_role(
            role="B",
            motion=motion_b,
            crouch=crouch_b,
            serve=serve_b,
            upper=upper_b,
            foot=foot_b,
            reach=reach_b,
            opp_motion=motion_a,
            opp_crouch=crouch_a,
            opp_serve=serve_a,
            opp_upper=upper_a,
            opp_foot=foot_a,
        )
    )
    candidates.sort(key=lambda item: item.timestamp)

    return _merge_player_start_candidates(
        ts,
        candidates,
        same_role_merge_window_sec=same_role_merge_window_sec,
        cross_role_merge_window_sec=cross_role_merge_window_sec,
    )


def _select_player_sandwich_start_candidates(
    diagnostics: PlayerStateMachineDiagnostics,
    *,
    swing_confirm_window_sec: float = 0.70,
    swing_confirm_peak_thresh: float = 0.42,
    post_reaction_window_sec: float = 1.05,
    pre_ready_window_sec: float = 0.55,
    min_candidate_score: float = 0.58,
    min_crouch_score: float = 0.65,
    min_reach_score: float = 0.56,
    min_opponent_ready_score: float = 0.30,
    min_ready_pair_score: float = 0.05,
    min_prep_launch_margin: float = 0.14,
    max_launch_score: float = 0.78,
    max_server_action_over_reach: float = 0.10,
    min_pre_ready_mean: float = 0.06,
    max_pre_live_peak: float = 0.92,
    min_server_growth: float = 0.08,
    min_server_peak: float = 0.44,
    min_server_peak_delay_sec: float = 0.0,
    min_receiver_peak: float = 0.06,
    min_live_peak: float = 0.78,
    max_serve_over_reach: float = 0.18,
    max_upper_over_reach: float = 0.20,
    max_foot_over_reach: float = 0.14,
) -> List[PlayerRallyStartCandidate]:
    if not diagnostics.timestamps:
        return []

    ts = np.asarray(diagnostics.timestamps, dtype=np.float32)
    sample_count = len(ts)

    def arr(values: List[float]) -> np.ndarray:
        if values and len(values) == sample_count:
            return np.asarray(values, dtype=np.float32)
        return np.zeros(sample_count, dtype=np.float32)

    motion_a = arr(diagnostics.motion_a)
    motion_b = arr(diagnostics.motion_b)
    crouch_a = arr(diagnostics.crouch_a)
    crouch_b = arr(diagnostics.crouch_b)
    reach_a = arr(diagnostics.reach_a)
    reach_b = arr(diagnostics.reach_b)
    serve_a = arr(diagnostics.serve_a)
    serve_b = arr(diagnostics.serve_b)
    upper_a = arr(diagnostics.upper_a)
    upper_b = arr(diagnostics.upper_b)
    foot_a = arr(diagnostics.foot_a)
    foot_b = arr(diagnostics.foot_b)

    ready_a = np.clip(crouch_a * (1.0 - (0.82 * np.maximum(motion_a, foot_a))), 0.0, 1.0)
    ready_b = np.clip(crouch_b * (1.0 - (0.82 * np.maximum(motion_b, foot_b))), 0.0, 1.0)
    ready_pair = np.minimum(ready_a, ready_b)
    competitive_a = np.maximum.reduce(
        [
            motion_a * (0.35 + (0.65 * crouch_a)),
            upper_a * (0.45 + (0.55 * crouch_a)),
            foot_a * (0.42 + (0.58 * crouch_a)),
        ]
    )
    competitive_b = np.maximum.reduce(
        [
            motion_b * (0.35 + (0.65 * crouch_b)),
            upper_b * (0.45 + (0.55 * crouch_b)),
            foot_b * (0.42 + (0.58 * crouch_b)),
        ]
    )
    live_pair = np.maximum(competitive_a, competitive_b)

    def search_end_idx(start_idx: int, secs: float, *, stop_exclusive: Optional[int] = None) -> int:
        end_t = float(ts[start_idx] + secs)
        end_idx = int(np.searchsorted(ts, end_t, side="right"))
        if stop_exclusive is not None:
            end_idx = min(end_idx, int(stop_exclusive))
        min_end_idx = start_idx + 2 if sample_count > (start_idx + 1) else start_idx + 1
        return max(min_end_idx, min(end_idx, sample_count))

    def search_start_idx(end_idx: int, secs: float) -> int:
        start_t = float(ts[end_idx] - secs)
        start_idx = int(np.searchsorted(ts, start_t, side="left"))
        return max(0, min(start_idx, end_idx))

    def window_max(values: np.ndarray, start_idx: int, secs: float, *, stop_exclusive: Optional[int] = None) -> float:
        end_idx = search_end_idx(start_idx, secs, stop_exclusive=stop_exclusive)
        return float(values[start_idx:end_idx].max()) if end_idx > start_idx else float(values[start_idx])

    def range_max(values: np.ndarray, start_idx: int, end_exclusive: int) -> float:
        end_exclusive = max(start_idx + 1, min(end_exclusive, sample_count))
        return float(values[start_idx:end_exclusive].max()) if end_exclusive > start_idx else float(values[start_idx])

    def range_mean(values: np.ndarray, start_idx: int, end_exclusive: int) -> float:
        end_exclusive = max(start_idx + 1, min(end_exclusive, sample_count))
        return float(values[start_idx:end_exclusive].mean()) if end_exclusive > start_idx else float(values[start_idx])

    def enrich_candidate(candidate: PlayerRallyStartCandidate) -> PlayerRallyStartCandidate:
        idx = int(candidate.sample_idx)
        if idx < 0 or idx >= sample_count:
            return candidate

        pre_start_idx = search_start_idx(idx, pre_ready_window_sec)
        if pre_start_idx < idx:
            pre_ready_mean = range_mean(ready_pair, pre_start_idx, idx)
            pre_live_peak = range_max(live_pair, pre_start_idx, idx)
        else:
            pre_ready_mean = float(ready_pair[idx])
            pre_live_peak = float(live_pair[idx - 1]) if idx > 0 else float(live_pair[idx])

        if candidate.role == "A":
            server_motion = motion_a
            server_serve = serve_a
            server_upper = upper_a
            server_foot = foot_a
            receiver_motion = motion_b
            receiver_upper = upper_b
            receiver_foot = foot_b
        else:
            server_motion = motion_b
            server_serve = serve_b
            server_upper = upper_b
            server_foot = foot_b
            receiver_motion = motion_a
            receiver_upper = upper_a
            receiver_foot = foot_a

        server_action = np.maximum.reduce(
            [
                server_serve,
                server_upper,
                0.78 * server_foot,
                0.58 * server_motion,
            ]
        )
        receiver_action = np.maximum.reduce(
            [
                receiver_upper,
                0.92 * receiver_foot,
                0.65 * receiver_motion,
            ]
        )

        stop_exclusive = min(sample_count, int(candidate.episode_end_sample_idx) + 2)
        confirm_end_idx = search_end_idx(idx, swing_confirm_window_sec, stop_exclusive=stop_exclusive)
        confirm_slice = server_action[idx:confirm_end_idx]
        if confirm_slice.size == 0:
            confirm_peak_idx = idx
            confirm_peak = float(server_action[idx])
        else:
            confirm_peak_idx = int(idx + int(np.argmax(confirm_slice)))
            confirm_peak = float(confirm_slice.max())
        server_now = float(server_action[idx])
        server_growth = float(confirm_peak - server_now)
        server_peak_delay_sec = float(ts[confirm_peak_idx] - ts[idx]) if confirm_peak_idx >= idx else 0.0

        reaction_end_idx = search_end_idx(confirm_peak_idx, post_reaction_window_sec)
        receiver_peak = range_max(receiver_action, confirm_peak_idx, reaction_end_idx)
        live_peak = range_max(live_pair, idx, reaction_end_idx)

        return replace(
            candidate,
            ready_pair_score=float(ready_pair[idx]),
            pre_ready_mean=float(pre_ready_mean),
            pre_live_peak=float(pre_live_peak),
            server_action_score=server_now,
            server_peak_score=float(confirm_peak),
            server_growth_score=float(server_growth),
            server_peak_delay_sec=float(server_peak_delay_sec),
            receiver_peak_score=float(receiver_peak),
            live_peak_score=float(live_peak),
        )

    def start_is_confirmed(candidate: PlayerRallyStartCandidate) -> bool:
        clean_prep_rescue = bool(
            candidate.prep_score >= 0.78
            and candidate.launch_score <= 0.48
            and candidate.serve_score <= 0.40
            and candidate.upper_body_score <= 0.46
            and candidate.footwork_score <= 0.50
            and candidate.dominance_ratio >= 1.75
            and candidate.pre_live_peak <= 0.80
            and candidate.server_growth_score >= 0.24
            and candidate.server_peak_delay_sec >= 0.08
            and candidate.receiver_peak_score >= 0.80
            and candidate.live_peak_score >= 0.82
        )
        strong_live_context_start_rescue = bool(
            candidate.prep_score >= 0.88
            and candidate.launch_score <= 0.38
            and candidate.serve_score <= 0.22
            and candidate.upper_body_score <= 0.30
            and candidate.footwork_score <= 0.45
            and candidate.server_action_score <= 0.35
            and candidate.dominance_ratio >= 2.0
            and candidate.ready_pair_score >= 0.45
            and candidate.opponent_ready_score >= 0.45
            and candidate.pre_ready_mean >= 0.40
            and candidate.pre_live_peak >= 0.95
            and candidate.server_peak_score >= 0.95
            and candidate.server_growth_score >= 0.60
            and candidate.server_peak_delay_sec >= 0.40
            and candidate.receiver_peak_score >= 0.28
            and candidate.live_peak_score >= 0.98
        )
        quick_abort_start_rescue = bool(
            candidate.prep_score >= 0.80
            and candidate.crouch_score >= 0.90
            and candidate.launch_score <= 0.52
            and candidate.serve_score <= 0.40
            and candidate.upper_body_score <= 0.46
            and candidate.footwork_score <= 0.58
            and candidate.ready_pair_score >= 0.45
            and candidate.opponent_ready_score >= 0.55
            and candidate.pre_ready_mean >= 0.20
            and candidate.pre_live_peak >= 0.80
            and candidate.server_action_score <= 0.46
            and candidate.server_peak_score >= 0.46
            and candidate.server_peak_delay_sec >= 0.06
            and candidate.receiver_peak_score >= 0.55
            and 0.50 <= candidate.live_peak_score <= 0.78
        )
        high_prelive_context_start_rescue = bool(
            candidate.prep_score >= 0.80
            and candidate.launch_score <= 0.40
            and candidate.serve_score <= 0.33
            and candidate.upper_body_score <= 0.35
            and candidate.footwork_score <= 0.30
            and candidate.dominance_ratio >= 1.75
            and candidate.ready_pair_score >= 0.52
            and candidate.pre_ready_mean >= 0.19
            and candidate.pre_live_peak >= 0.95
            and candidate.server_action_score <= 0.35
            and candidate.server_peak_score >= 0.85
            and candidate.server_growth_score >= 0.50
            and candidate.server_peak_delay_sec >= 0.30
            and candidate.live_peak_score >= 0.95
        )
        low_live_floor_rescue = bool(
            candidate.prep_score >= 0.75
            and candidate.launch_score <= 0.42
            and candidate.ready_pair_score >= 0.55
            and candidate.opponent_ready_score >= 0.55
            and candidate.pre_live_peak <= 0.35
            and candidate.server_action_score <= 0.37
            and candidate.server_peak_score >= 0.60
            and candidate.server_growth_score >= 0.40
            and candidate.server_peak_delay_sec >= 0.13
            and candidate.receiver_peak_score >= 0.40
            and 0.74 <= candidate.live_peak_score <= 0.78
        )
        marginal_reach_start_rescue = bool(
            candidate.prep_score >= 0.70
            and candidate.launch_score <= 0.42
            and candidate.reach_score >= 0.54
            and candidate.opponent_ready_score >= 0.45
            and candidate.dominance_ratio >= 2.0
            and candidate.ready_pair_score >= 0.40
            and candidate.pre_ready_mean >= 0.40
            and candidate.pre_live_peak <= 0.50
            and candidate.server_action_score <= 0.46
            and candidate.server_peak_score >= 0.75
            and candidate.server_growth_score >= 0.30
            and candidate.server_peak_delay_sec >= 0.20
            and candidate.live_peak_score >= 0.95
        )
        opening_live_context_rescue = bool(
            candidate.prep_score >= 0.82
            and candidate.crouch_score >= 0.98
            and candidate.reach_score >= 0.53
            and candidate.ready_pair_score >= 0.65
            and candidate.pre_ready_mean >= 0.35
            and candidate.pre_live_peak >= 0.98
            and candidate.launch_score <= 0.44
            and candidate.server_action_score <= 0.50
            and candidate.server_peak_score >= 0.95
            and candidate.server_growth_score >= 0.50
            and candidate.server_peak_delay_sec >= 0.40
            and candidate.receiver_peak_score >= 0.95
            and candidate.live_peak_score >= 0.95
        )
        dynamic_footwork_start_rescue = bool(
            candidate.prep_score >= 0.84
            and candidate.crouch_score >= 0.95
            and candidate.launch_score <= 0.48
            and candidate.footwork_score >= 0.70
            and 0.50 <= candidate.server_action_score <= 0.60
            and candidate.pre_ready_mean >= 0.30
            and candidate.pre_live_peak >= 0.70
            and candidate.dominance_ratio >= 1.20
            and candidate.receiver_peak_score <= 0.45
            and candidate.live_peak_score >= 0.98
            and candidate.server_peak_delay_sec >= 0.60
        )
        # Some real serves briefly lose receiver-ready confidence exactly at trigger time
        # even though the receiver was ready just beforehand and the rally confirms cleanly.
        receiver_ready_dropout_rescue = bool(
            candidate.prep_score >= 0.78
            and candidate.launch_score <= 0.40
            and candidate.serve_score <= 0.40
            and candidate.upper_body_score <= 0.40
            and candidate.footwork_score <= 0.48
            and candidate.server_action_score <= 0.42
            and candidate.dominance_ratio >= 1.75
            and candidate.ready_pair_score < min_ready_pair_score
            and candidate.opponent_ready_score < min_opponent_ready_score
            and candidate.pre_ready_mean >= max(0.16, min_pre_ready_mean)
            and candidate.pre_live_peak <= 0.80
            and candidate.server_peak_score >= 0.90
            and candidate.server_growth_score >= 0.24
            and candidate.server_peak_delay_sec >= 0.08
            and candidate.receiver_peak_score >= 0.55
            and candidate.live_peak_score >= 0.88
        )
        late_reset_dropout_rescue = bool(
            candidate.prep_score >= 0.80
            and candidate.launch_score <= 0.38
            and candidate.serve_score <= 0.32
            and candidate.upper_body_score <= 0.32
            and candidate.footwork_score <= 0.34
            and candidate.dominance_ratio >= 2.0
            and candidate.ready_pair_score < min_ready_pair_score
            and candidate.opponent_ready_score < min_opponent_ready_score
            and candidate.pre_ready_mean >= 0.16
            and candidate.pre_live_peak >= 0.95
            and candidate.server_action_score <= 0.30
            and candidate.server_peak_score >= 0.52
            and candidate.server_growth_score >= 0.20
            and candidate.server_peak_delay_sec >= 0.20
            and candidate.receiver_peak_score >= 0.85
            and candidate.live_peak_score >= 0.92
        )
        strong_followup_start_exception = bool(
            candidate.prep_score >= 0.85
            and candidate.opponent_ready_score >= 0.50
            and candidate.ready_pair_score >= 0.45
            and candidate.dominance_ratio >= 1.45
            and candidate.footwork_score <= 0.40
        )
        if candidate.score < min_candidate_score or candidate.crouch_score < min_crouch_score:
            return False
        if clean_prep_rescue:
            if candidate.reach_score < 0.50:
                return False
        elif receiver_ready_dropout_rescue or late_reset_dropout_rescue:
            if candidate.reach_score < 0.50:
                return False
        elif marginal_reach_start_rescue or opening_live_context_rescue:
            if candidate.reach_score < 0.53:
                return False
        elif quick_abort_start_rescue:
            if candidate.reach_score < min_reach_score:
                return False
        else:
            if candidate.reach_score < min_reach_score or candidate.ready_pair_score < min_ready_pair_score:
                return False
        aggressive_stroke_pattern = bool(
            candidate.launch_score >= 0.54
            and candidate.dominance_ratio < 1.70
            and (candidate.upper_body_score >= 0.46 or candidate.footwork_score >= 0.46)
        )
        already_live_exchange_pattern = bool(
            candidate.pre_live_peak >= 0.48
            and candidate.live_peak_score >= 0.92
            and candidate.pre_ready_mean <= 0.40
            and candidate.dominance_ratio <= 1.55
            and candidate.server_action_score >= 0.34
            and max(candidate.launch_score, candidate.upper_body_score, candidate.footwork_score) >= 0.36
            and not strong_followup_start_exception
            and not dynamic_footwork_start_rescue
            and not opening_live_context_rescue
        )
        already_live_attack_pattern = bool(
            candidate.pre_live_peak >= 0.84
            and candidate.pre_ready_mean <= 0.24
            and candidate.opponent_ready_score <= 0.42
            and candidate.launch_score >= 0.56
            and candidate.server_action_score >= 0.62
            and candidate.upper_body_score >= 0.60
        )
        weak_followup_live_pattern = bool(
            candidate.pre_live_peak >= 0.48
            and candidate.live_peak_score >= 0.86
            and candidate.dominance_ratio <= 1.15
            and candidate.ready_pair_score <= 0.32
            and candidate.server_action_score <= 0.32
            and candidate.receiver_peak_score <= 0.40
            and candidate.prep_score <= 0.82
        )
        upright_live_reset_pattern = bool(
            candidate.crouch_score <= 0.70
            and candidate.prep_score <= 0.70
            and candidate.launch_score >= 0.34
            and candidate.pre_live_peak >= 0.60
            and candidate.ready_pair_score >= 0.30
            and candidate.server_action_score <= 0.38
            and candidate.live_peak_score >= 0.90
            and candidate.receiver_peak_score >= 0.35
        )
        low_dominance_attack_pattern = bool(
            candidate.dominance_ratio <= 1.15
            and candidate.launch_score >= 0.50
            and candidate.footwork_score >= 0.55
            and candidate.pre_live_peak >= 0.48
            and candidate.server_action_score >= 0.44
            and candidate.live_peak_score >= 0.80
        )
        live_reattack_pattern = bool(
            candidate.pre_live_peak >= 0.85
            and candidate.pre_ready_mean <= 0.36
            and candidate.launch_score >= 0.50
            and candidate.server_action_score >= 0.44
            and candidate.live_peak_score >= 0.90
            and candidate.receiver_peak_score >= 0.75
        )
        if (
            aggressive_stroke_pattern
            or already_live_exchange_pattern
            or already_live_attack_pattern
            or weak_followup_live_pattern
            or upright_live_reset_pattern
            or low_dominance_attack_pattern
            or live_reattack_pattern
        ):
            return False
        if candidate.opponent_ready_score < min_opponent_ready_score and not (
            clean_prep_rescue
            or receiver_ready_dropout_rescue
            or late_reset_dropout_rescue
        ):
            return False
        if (candidate.prep_score - candidate.launch_score) < min_prep_launch_margin:
            return False
        if candidate.launch_score > max_launch_score:
            return False
        if candidate.server_action_score > (candidate.reach_score + max_server_action_over_reach):
            return False
        if candidate.serve_score > (candidate.reach_score + max_serve_over_reach):
            return False
        if candidate.upper_body_score > (candidate.reach_score + max_upper_over_reach):
            return False
        if candidate.footwork_score > (candidate.reach_score + max_foot_over_reach):
            return False

        ready_context = max(candidate.pre_ready_mean, 0.88 * candidate.ready_pair_score)
        if not (
            clean_prep_rescue
            or receiver_ready_dropout_rescue
            or strong_live_context_start_rescue
            or quick_abort_start_rescue
            or late_reset_dropout_rescue
            or high_prelive_context_start_rescue
            or low_live_floor_rescue
            or marginal_reach_start_rescue
            or opening_live_context_rescue
            or dynamic_footwork_start_rescue
        ) and (ready_context < min_pre_ready_mean or candidate.pre_live_peak > max_pre_live_peak):
            return False
        if candidate.server_peak_score < max(swing_confirm_peak_thresh, min_server_peak):
            return False
        if not quick_abort_start_rescue and candidate.server_growth_score < min_server_growth:
            return False
        if candidate.server_peak_delay_sec <= min_server_peak_delay_sec:
            return False
        if not (quick_abort_start_rescue or low_live_floor_rescue) and (candidate.receiver_peak_score < min_receiver_peak or candidate.live_peak_score < min_live_peak):
            return False
        if quick_abort_start_rescue and candidate.receiver_peak_score < 0.55:
            return False

        return True

    start_candidates = _compute_player_rally_start_candidates(diagnostics)
    if not start_candidates:
        return []

    enriched_candidates = [enrich_candidate(candidate) for candidate in start_candidates]
    confirmed_candidates = [candidate for candidate in enriched_candidates if start_is_confirmed(candidate)]
    if not confirmed_candidates:
        return []
    deduped_candidates = _dedupe_player_sandwich_start_candidates(diagnostics.timestamps, confirmed_candidates)
    return _repair_double_serve_role_singletons(deduped_candidates)


def _detect_player_sandwich_rallies_from_diagnostics(
    diagnostics: PlayerStateMachineDiagnostics,
    *,
    swing_confirm_window_sec: float = 0.70,
    swing_confirm_peak_thresh: float = 0.42,
    reset_hold_sec: float = 0.55,
    reset_live_thresh: float = 0.18,
    stand_reset_thresh: float = 0.56,
    casual_reset_thresh: float = 0.64,
    min_point_sec: float = 0.90,
    let_max_sec: float = 1.50,
    let_quiet_window_sec: float = 0.60,
) -> List[RallySegment]:
    if not diagnostics.timestamps:
        return []

    ts = np.asarray(diagnostics.timestamps, dtype=np.float32)
    sample_count = len(ts)

    def arr(values: List[float]) -> np.ndarray:
        if values and len(values) == sample_count:
            return np.asarray(values, dtype=np.float32)
        return np.zeros(sample_count, dtype=np.float32)

    motion_a = arr(diagnostics.motion_a)
    motion_b = arr(diagnostics.motion_b)
    crouch_a = arr(diagnostics.crouch_a)
    crouch_b = arr(diagnostics.crouch_b)
    serve_a = arr(diagnostics.serve_a)
    serve_b = arr(diagnostics.serve_b)
    upper_a = arr(diagnostics.upper_a)
    upper_b = arr(diagnostics.upper_b)
    foot_a = arr(diagnostics.foot_a)
    foot_b = arr(diagnostics.foot_b)
    reach_a = arr(diagnostics.reach_a)
    reach_b = arr(diagnostics.reach_b)
    approach_a = arr(diagnostics.approach_a)
    approach_b = arr(diagnostics.approach_b)

    competitive_a = np.maximum.reduce(
        [
            motion_a * (0.35 + (0.65 * crouch_a)),
            upper_a * (0.45 + (0.55 * crouch_a)),
            foot_a * (0.42 + (0.58 * crouch_a)),
        ]
    )
    competitive_b = np.maximum.reduce(
        [
            motion_b * (0.35 + (0.65 * crouch_b)),
            upper_b * (0.45 + (0.55 * crouch_b)),
            foot_b * (0.42 + (0.58 * crouch_b)),
        ]
    )
    live_pair = np.maximum(competitive_a, competitive_b)
    stand_a = np.clip((1.0 - crouch_a) * (1.0 - (0.35 * np.maximum(upper_a, foot_a))), 0.0, 1.0)
    stand_b = np.clip((1.0 - crouch_b) * (1.0 - (0.35 * np.maximum(upper_b, foot_b))), 0.0, 1.0)
    stand_pair = np.minimum(stand_a, stand_b)
    casual_a = np.clip(
        1.0 - np.maximum.reduce([0.90 * upper_a, 0.80 * foot_a, 0.70 * crouch_a, 0.55 * serve_a]),
        0.0,
        1.0,
    )
    casual_b = np.clip(
        1.0 - np.maximum.reduce([0.90 * upper_b, 0.80 * foot_b, 0.70 * crouch_b, 0.55 * serve_b]),
        0.0,
        1.0,
    )
    casual_pair = np.minimum(casual_a, casual_b)
    reset_pair = np.maximum(stand_pair, 0.92 * casual_pair)
    shared_activity = np.maximum.reduce([motion_a, motion_b, upper_a, upper_b, foot_a, foot_b])

    def search_end_idx(start_idx: int, secs: float, *, stop_exclusive: Optional[int] = None) -> int:
        end_t = float(ts[start_idx] + secs)
        end_idx = int(np.searchsorted(ts, end_t, side="right"))
        if stop_exclusive is not None:
            end_idx = min(end_idx, int(stop_exclusive))
        return max(start_idx + 1, min(end_idx, sample_count))

    def window_max(values: np.ndarray, start_idx: int, secs: float, *, stop_exclusive: Optional[int] = None) -> float:
        end_idx = search_end_idx(start_idx, secs, stop_exclusive=stop_exclusive)
        return float(values[start_idx:end_idx].max()) if end_idx > start_idx else float(values[start_idx])

    def window_mean(values: np.ndarray, start_idx: int, secs: float, *, stop_exclusive: Optional[int] = None) -> float:
        end_idx = search_end_idx(start_idx, secs, stop_exclusive=stop_exclusive)
        return float(values[start_idx:end_idx].mean()) if end_idx > start_idx else float(values[start_idx])

    def append_segment(
        out: List[RallySegment],
        *,
        start_idx: int,
        end_idx: int,
        label: str,
        start_score: float,
        server_role: str,
        extra_flags: Optional[Sequence[str]] = None,
    ) -> None:
        if end_idx <= start_idx or start_idx < 0 or end_idx >= sample_count:
            return
        seg_start_t = float(ts[start_idx])
        seg_end_t = float(ts[end_idx])
        min_dur = 0.35 if label == "let" else min_point_sec
        if (seg_end_t - seg_start_t) < min_dur:
            return

        active_window = live_pair[start_idx : end_idx + 1]
        active_peak = float(active_window.max()) if active_window.size else 0.0
        active_mean = float(active_window.mean()) if active_window.size else 0.0
        reset_peak = float(reset_pair[end_idx : min(sample_count, end_idx + 2)].max())
        conf_floor = 0.42 if label == "let" else 0.50
        confidence = float(
            np.clip(
                0.30 + (0.32 * float(start_score)) + (0.22 * active_peak) + (0.10 * active_mean) + (0.08 * reset_peak),
                conf_floor,
                0.96,
            )
        )

        flags = ["player_sandwich", "rally_label_point"]
        if label == "let":
            flags = ["player_sandwich", "rally_label_let", "let_no_score"]
        if extra_flags:
            flags.extend(str(flag) for flag in extra_flags if flag)
        out.append(
            RallySegment(
                t_start=seg_start_t,
                t_end=seg_end_t,
                confidence=confidence,
                flags=flags,
                server_role=str(server_role) if server_role in {"A", "B"} else None,
            )
        )

    def find_preliminary_end(start_idx: int, stop_exclusive: int) -> Optional[int]:
        reset_run_start_idx: Optional[int] = None
        earliest_end_idx = int(np.searchsorted(ts, float(ts[start_idx] + 0.45), side="left"))
        for idx in range(start_idx + 1, max(start_idx + 1, stop_exclusive)):
            if idx < earliest_end_idx:
                continue
            reset_now = bool(
                ((stand_pair[idx] >= stand_reset_thresh) or (casual_pair[idx] >= casual_reset_thresh))
                and live_pair[idx] <= reset_live_thresh
            )
            if reset_now:
                if reset_run_start_idx is None:
                    reset_run_start_idx = idx
                if float(ts[idx] - ts[reset_run_start_idx]) >= reset_hold_sec:
                    return int(reset_run_start_idx)
                continue
            reset_run_start_idx = None
        return None

    def compute_local_let_signal(
        candidate: PlayerRallyStartCandidate,
        start_idx: int,
        end_idx: int,
        stop_exclusive: int,
    ) -> Tuple[bool, float]:
        duration = float(ts[end_idx] - ts[start_idx])
        if candidate.role == "A":
            receive_reach = reach_b
            receive_approach = approach_b
            receive_upper = upper_b
            receive_foot = foot_b
            receive_motion = motion_b
        else:
            receive_reach = reach_a
            receive_approach = approach_a
            receive_upper = upper_a
            receive_foot = foot_a
            receive_motion = motion_a

        receive_reach_peak = window_max(receive_reach, start_idx, let_max_sec, stop_exclusive=stop_exclusive)
        receive_approach_peak = window_max(receive_approach, start_idx, let_max_sec, stop_exclusive=stop_exclusive)
        receive_return_peak = max(
            window_max(receive_upper, start_idx, let_max_sec, stop_exclusive=stop_exclusive),
            0.90 * window_max(receive_foot, start_idx, let_max_sec, stop_exclusive=stop_exclusive),
            0.75 * window_max(receive_motion, start_idx, let_max_sec, stop_exclusive=stop_exclusive),
        )
        quiet_after = window_mean(shared_activity, end_idx, let_quiet_window_sec, stop_exclusive=stop_exclusive)
        duration_support = float(np.clip((let_max_sec - duration) / max(let_max_sec - 0.35, 1e-6), 0.0, 1.0))
        reach_support = float(np.clip((receive_reach_peak - 0.30) / 0.55, 0.0, 1.0))
        approach_support = float(np.clip((receive_approach_peak - 0.08) / 0.36, 0.0, 1.0))
        quiet_support = float(np.clip((0.30 - quiet_after) / 0.30, 0.0, 1.0))
        return_suppression = float(np.clip((0.42 - receive_return_peak) / 0.42, 0.0, 1.0))
        let_score = float(
            np.clip(
                (0.22 * duration_support)
                + (0.22 * reach_support)
                + (0.18 * approach_support)
                + (0.18 * quiet_support)
                + (0.20 * return_suppression),
                0.0,
                1.0,
            )
        )
        is_let = bool(
            duration <= let_max_sec
            and
            receive_reach_peak >= 0.40
            and receive_approach_peak >= 0.14
            and receive_return_peak <= 0.32
            and quiet_after <= 0.22
        )
        return is_let, let_score

    start_candidates = _select_player_sandwich_start_candidates(
        diagnostics,
        swing_confirm_window_sec=swing_confirm_window_sec,
        swing_confirm_peak_thresh=swing_confirm_peak_thresh,
    )
    if not start_candidates:
        return []

    provisional_records: List[Dict[str, object]] = []
    for idx, candidate in enumerate(start_candidates):
        start_idx = int(candidate.sample_idx)
        if start_idx >= sample_count - 1:
            continue

        next_start_idx: Optional[int] = None
        if idx + 1 < len(start_candidates):
            next_start_idx = int(start_candidates[idx + 1].sample_idx)
            if next_start_idx <= start_idx:
                continue

        stop_exclusive = sample_count if next_start_idx is None else max(start_idx + 1, next_start_idx)
        preliminary_end_idx = find_preliminary_end(start_idx, stop_exclusive)
        if preliminary_end_idx is None:
            if next_start_idx is not None:
                preliminary_end_idx = max(start_idx + 1, next_start_idx - 1)
            else:
                trailing_live = np.flatnonzero(live_pair[start_idx:] >= 0.10)
                if trailing_live.size:
                    preliminary_end_idx = int(start_idx + trailing_live[-1])
                else:
                    preliminary_end_idx = min(sample_count - 1, start_idx + 1)

        if next_start_idx is not None:
            preliminary_end_idx = min(preliminary_end_idx, next_start_idx - 1)
        if preliminary_end_idx <= start_idx:
            continue

        local_let, local_let_score = compute_local_let_signal(candidate, start_idx, preliminary_end_idx, stop_exclusive)
        duration_sec = float(ts[preliminary_end_idx] - ts[start_idx])
        active_window = live_pair[start_idx : preliminary_end_idx + 1]
        active_peak = float(active_window.max()) if active_window.size else 0.0
        active_mean = float(active_window.mean()) if active_window.size else 0.0
        point_likelihood = max(
            0.0,
            float(
                duration_sec
                + (1.25 * active_peak)
                + (0.70 * active_mean)
                + (0.55 * float(candidate.live_peak_score))
                + (0.40 * float(candidate.receiver_peak_score))
                + (0.25 * float(candidate.server_peak_score))
                - (2.50 * float(local_let_score))
            ),
        )
        provisional_records.append(
            {
                "candidate": candidate,
                "start_idx": start_idx,
                "end_idx": preliminary_end_idx,
                "duration_sec": duration_sec,
                "local_let": bool(local_let),
                "local_let_score": float(local_let_score),
                "point_likelihood": float(point_likelihood),
            }
        )

    if not provisional_records:
        return []

    starter_roles = [str(record["candidate"].role) for record in provisional_records]  # type: ignore[union-attr]
    point_likelihoods = [float(record["point_likelihood"]) for record in provisional_records]
    serve_mode = _infer_player_serve_mode_from_starter_roles(starter_roles)
    legal_limit = 2 if serve_mode == "double" else 1
    forced_let_indices = _infer_forced_let_indices_from_starter_roles(
        starter_roles,
        point_likelihoods,
        serve_mode=serve_mode,
    )

    overflow_run_indices: Set[int] = set()
    run_start = 0
    while run_start < len(starter_roles):
        role = starter_roles[run_start]
        run_end = run_start + 1
        while run_end < len(starter_roles) and starter_roles[run_end] == role:
            run_end += 1
        if (run_end - run_start) > legal_limit:
            overflow_run_indices.update(range(run_start, run_end))
        run_start = run_end

    segments: List[RallySegment] = []
    for idx, record in enumerate(provisional_records):
        candidate = record["candidate"]  # type: ignore[assignment]
        assert isinstance(candidate, PlayerRallyStartCandidate)
        label = "point"
        extra_flags: List[str] = []
        if idx in forced_let_indices:
            label = "let"
            extra_flags.extend(
                [
                    "let_inferred_forced_serve_order",
                    f"let_inferred_serve_mode_{serve_mode}",
                ]
            )
        elif idx not in overflow_run_indices and bool(record["local_let"]):
            label = "let"
            extra_flags.append("let_inferred_local_abort")
        append_segment(
            segments,
            start_idx=int(record["start_idx"]),
            end_idx=int(record["end_idx"]),
            label=label,
            start_score=float(candidate.score),
            server_role=str(candidate.role),
            extra_flags=extra_flags,
        )

    return segments


def _detect_player_sandwich_rallies(signals: MultiStreamSignals) -> List[RallySegment]:
    diagnostics = _compute_player_state_machine_diagnostics(signals)
    return _detect_player_sandwich_rallies_from_diagnostics(diagnostics)


def _collect_nearest_two_energies(
    cap: cv2.VideoCapture,
    person_model: YOLO,
    *,
    roi: TableROI,
    fps: float,
    stride: int,
    device: str,
    player_margin_px: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    tx, ty, tw, th = roi.as_tuple()
    timestamps: List[float] = []
    table_energies: List[float] = []
    player_energies: List[float] = []
    prev_table_gray: Optional[np.ndarray] = None
    prev_player_kpts: Dict[int, np.ndarray] = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        timestamp = float(frame_idx / fps)
        timestamps.append(timestamp)

        table_crop = frame[ty : ty + th, tx : tx + tw]
        table_gray = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)
        if prev_table_gray is None:
            table_energies.append(0.0)
        else:
            table_energies.append(float(np.mean(cv2.absdiff(table_gray, prev_table_gray))))
        prev_table_gray = table_gray

        results = person_model.track(
            frame,
            persist=True,
            classes=[0],
            device=device,
            verbose=False,
            half=True,
        )

        frame_player_vels: List[float] = []
        if results and results[0].boxes.id is not None and results[0].keypoints is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            kpts = results[0].keypoints.xy.cpu().numpy()
            candidates: List[Tuple[float, int]] = []
            table_center = np.array([tx + tw / 2.0, ty + th / 2.0], dtype=np.float32)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
                dist = float(np.linalg.norm(center - table_center))
                candidates.append((dist, i))
            candidates.sort(key=lambda x: x[0])

            for _, idx in candidates[:2]:
                player_id = int(ids[idx])
                box = boxes[idx]
                if not _is_player_near_table(box, roi, player_margin_px):
                    continue
                player_h = max(1.0, float(box[3] - box[1]))
                vel = _calc_wrist_velocity(kpts[idx], prev_player_kpts.get(player_id), player_h)
                frame_player_vels.append(vel)
                prev_player_kpts[player_id] = kpts[idx]

        player_energies.append(max(frame_player_vels) if frame_player_vels else 0.0)
        frame_idx += 1

    return timestamps, table_energies, player_energies, player_energies


def _extract_production_table_energies(
    video_path: str,
    *,
    roi: TableROI,
    width: int,
    height: int,
    fps: float,
    stride: int,
    device: str,
) -> Tuple[List[float], List[float]]:
    tx, ty, tw, th = roi.as_tuple()
    energies: List[float] = []
    timestamps: List[float] = []
    prev_frame_gpu = None

    frame_iter = nvdec_bgr24_stream(
        str(video_path),
        int(width),
        int(height),
        crop_roi=(tx, ty, tw, th),
    )

    for idx, frame_np in enumerate(frame_iter):
        if idx % stride != 0:
            continue

        curr_gpu = torch.from_numpy(frame_np).to(device).float()
        if prev_frame_gpu is not None:
            diff = torch.abs(curr_gpu - prev_frame_gpu)
            diff_max = F.max_pool2d(
                diff.permute(2, 0, 1).unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1,
            )
            energies.append(float(diff_max.mean().item()))
            timestamps.append(float(idx / fps))
        prev_frame_gpu = curr_gpu

    return timestamps, energies


def _refine_table_segments_with_role_support(
    segments: List[RallySegment],
    *,
    timestamps: List[float],
    table_energies: List[float],
    player_a_energies: List[float],
    player_b_energies: List[float],
    min_segment_sec: float = 8.0,
    quiet_table_thresh: float = 0.28,
    quiet_role_thresh: float = 0.16,
    quiet_run_sec: float = 0.75,
    boundary_guard_sec: float = 0.9,
    min_piece_sec: float = 1.0,
) -> List[RallySegment]:
    if not segments or not timestamps:
        return list(segments)

    ts = np.asarray(timestamps, dtype=np.float32)
    table_norm = _smooth_and_normalize(table_energies)
    player_a_norm = _smooth_and_normalize(player_a_energies)
    player_b_norm = _smooth_and_normalize(player_b_energies)
    role_norm = np.maximum(player_a_norm, player_b_norm)

    refined: List[RallySegment] = []
    for seg in segments:
        seg_dur = float(seg.t_end - seg.t_start)
        if seg_dur < min_segment_sec:
            refined.append(seg)
            continue

        start_idx = int(np.searchsorted(ts, float(seg.t_start), side="left"))
        end_idx = int(np.searchsorted(ts, float(seg.t_end), side="right")) - 1
        start_idx = max(0, min(start_idx, len(ts) - 1))
        end_idx = max(start_idx, min(end_idx, len(ts) - 1))
        if end_idx <= start_idx:
            refined.append(seg)
            continue

        inner_start_t = float(seg.t_start + boundary_guard_sec)
        inner_end_t = float(seg.t_end - boundary_guard_sec)
        inner_start_idx = int(np.searchsorted(ts, inner_start_t, side="left"))
        inner_end_idx = int(np.searchsorted(ts, inner_end_t, side="right")) - 1
        inner_start_idx = max(start_idx + 1, inner_start_idx)
        inner_end_idx = min(end_idx - 1, inner_end_idx)
        if inner_end_idx <= inner_start_idx:
            refined.append(seg)
            continue

        quiet_cuts: List[int] = []
        run_start: Optional[int] = None
        for idx in range(inner_start_idx, inner_end_idx + 1):
            is_quiet = bool(table_norm[idx] < quiet_table_thresh and role_norm[idx] < quiet_role_thresh)
            if is_quiet:
                if run_start is None:
                    run_start = idx
                continue

            if run_start is not None:
                run_end = idx - 1
                if float(ts[run_end] - ts[run_start]) >= quiet_run_sec:
                    quiet_cuts.append(int((run_start + run_end) // 2))
                run_start = None

        if run_start is not None:
            run_end = inner_end_idx
            if float(ts[run_end] - ts[run_start]) >= quiet_run_sec:
                quiet_cuts.append(int((run_start + run_end) // 2))

        if not quiet_cuts:
            refined.append(seg)
            continue

        cut_indices = sorted(set(quiet_cuts))
        current_start_idx = start_idx
        split_segments: List[RallySegment] = []
        for cut_idx in cut_indices:
            left_dur = float(ts[cut_idx] - ts[current_start_idx])
            right_dur = float(ts[end_idx] - ts[cut_idx])
            if left_dur < min_piece_sec or right_dur < min_piece_sec:
                continue

            piece_conf = float(np.clip(np.median(table_norm[current_start_idx:cut_idx + 1]), 0.5, 1.0))
            split_segments.append(
                RallySegment(
                    t_start=float(ts[current_start_idx]),
                    t_end=float(ts[cut_idx]),
                    confidence=piece_conf,
                    flags=sorted(set(list(seg.flags) + ["role_quiet_gap_split"])),
                )
            )
            current_start_idx = cut_idx

        final_dur = float(ts[end_idx] - ts[current_start_idx])
        if not split_segments or final_dur < min_piece_sec:
            refined.append(seg)
            continue

        piece_conf = float(np.clip(np.median(table_norm[current_start_idx:end_idx + 1]), 0.5, 1.0))
        split_segments.append(
            RallySegment(
                t_start=float(ts[current_start_idx]),
                t_end=float(ts[end_idx]),
                confidence=piece_conf,
                flags=sorted(set(list(seg.flags) + ["role_quiet_gap_split"])),
            )
        )
        refined.extend(split_segments)

    return refined


def _merge_segments_with_ball_support(
    segments: List[RallySegment],
    *,
    timestamps: List[float],
    ball_energies: List[float],
    merge_max_gap_sec: float = 1.2,
    max_merged_duration_sec: float = 8.5,
    boundary_window_sec: float = 0.65,
    active_peak_thresh: float = 0.34,
    active_mean_thresh: float = 0.18,
) -> List[RallySegment]:
    if len(segments) <= 1 or not timestamps or not ball_energies:
        return list(segments)

    ts = np.asarray(timestamps, dtype=np.float32)
    ball_norm = _smooth_and_normalize(ball_energies)
    merged: List[RallySegment] = []
    current = segments[0]

    for nxt in segments[1:]:
        gap = float(nxt.t_start - current.t_end)
        should_merge = False
        candidate_duration = float(nxt.t_end - current.t_start)
        if gap <= merge_max_gap_sec and candidate_duration <= max_merged_duration_sec and (
            gap <= 0.08 or "split_long" in current.flags or "split_long" in nxt.flags
        ):
            win_start = max(float(current.t_start), float(current.t_end - boundary_window_sec))
            win_end = min(float(nxt.t_end), float(nxt.t_start + boundary_window_sec))
            start_idx = int(np.searchsorted(ts, win_start, side="left"))
            end_idx = int(np.searchsorted(ts, win_end, side="right")) - 1
            start_idx = max(0, min(start_idx, len(ts) - 1))
            end_idx = max(start_idx, min(end_idx, len(ts) - 1))
            if end_idx >= start_idx:
                window = ball_norm[start_idx : end_idx + 1]
                peak = float(window.max())
                mean = float(window.mean())
                if peak >= active_peak_thresh and mean >= active_mean_thresh:
                    should_merge = True

        if should_merge:
            merged_flags = sorted(set(list(current.flags) + list(nxt.flags) + ["ball_gap_merge"]))
            merged_conf = float(np.clip(np.median([current.confidence, nxt.confidence]), 0.5, 1.0))
            current = RallySegment(
                t_start=float(current.t_start),
                t_end=float(nxt.t_end),
                confidence=merged_conf,
                flags=merged_flags,
            )
            continue

        merged.append(current)
        current = nxt

    merged.append(current)
    return merged


def _merge_ball_split_pair_artifacts(
    segments: List[RallySegment],
    *,
    contiguous_eps_sec: float = 0.05,
    short_piece_sec: float = 3.8,
    max_pair_sec: float = 10.0,
) -> List[RallySegment]:
    if len(segments) <= 1:
        return list(segments)

    merged: List[RallySegment] = []
    i = 0
    while i < len(segments):
        if i + 1 < len(segments):
            current = segments[i]
            nxt = segments[i + 1]
            gap = float(nxt.t_start - current.t_end)
            current_dur = float(current.t_end - current.t_start)
            next_dur = float(nxt.t_end - nxt.t_start)
            if (
                gap <= contiguous_eps_sec
                and "split_long" in current.flags
                and "split_long" in nxt.flags
                and min(current_dur, next_dur) < short_piece_sec
                and (current_dur + next_dur) <= max_pair_sec
            ):
                merged.append(
                    RallySegment(
                        t_start=float(current.t_start),
                        t_end=float(nxt.t_end),
                        confidence=float(max(current.confidence, nxt.confidence)),
                        flags=sorted(set(list(current.flags) + list(nxt.flags) + ["ball_pair_merge"])),
                    )
                )
                i += 2
                continue

        merged.append(segments[i])
        i += 1

    return merged


def extract_multistream_signals(
    video_path: str,
    table_weights_path: str,
    *,
    pose_weights_path: str = "weights/yolov8x-pose.pt",
    stride: int = 2,
    player_margin_px: int = 220,
    player_fuse_gain: float = 1.0,
    player_signal_source: str = "role_tracker",
    ball_fuse_gain: float = 1.15,
    ball_signal_source: str = "none",
    ball_tracking_profile: str = "support",
    device: str = "cuda",
) -> MultiStreamSignals:
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA GPU is required for multi-stream extraction.")

    v_path = Path(video_path).resolve()
    table_w_path = Path(table_weights_path).resolve()
    pose_w_path = Path(pose_weights_path).resolve()
    if not v_path.exists():
        raise FileNotFoundError(f"Video not found: {v_path}")
    if not table_w_path.exists():
        raise FileNotFoundError(f"Table weights not found: {table_w_path}")
    if player_signal_source not in {"role_tracker", "nearest_two", "none"}:
        raise ValueError(f"Invalid player_signal_source: {player_signal_source}")
    if ball_signal_source not in {"none", "classical"}:
        raise ValueError(f"Invalid ball_signal_source: {ball_signal_source}")
    if player_signal_source != "none" and not pose_w_path.exists():
        raise FileNotFoundError(f"Pose weights not found: {pose_w_path}")

    info = probe_video_ffprobe(str(v_path))
    roi = detect_table_roi_dl(
        str(v_path),
        cfg=DLConfig(weights_path=str(table_w_path), device=device),
    )
    if player_signal_source == "none":
        timestamps, table_energies = _extract_production_table_energies(
            str(v_path),
            roi=roi,
            width=int(info.width),
            height=int(info.height),
            fps=float(info.fps),
            stride=max(1, int(stride)),
            device=device,
        )
        player_a_energies = [0.0 for _ in timestamps]
        player_b_energies = [0.0 for _ in timestamps]
        player_a_crouch_scores = [0.0 for _ in timestamps]
        player_b_crouch_scores = [0.0 for _ in timestamps]
        player_a_serve_scores = [0.0 for _ in timestamps]
        player_b_serve_scores = [0.0 for _ in timestamps]
        player_a_upper_body_scores = [0.0 for _ in timestamps]
        player_b_upper_body_scores = [0.0 for _ in timestamps]
        player_a_footwork_scores = [0.0 for _ in timestamps]
        player_b_footwork_scores = [0.0 for _ in timestamps]
        player_a_reach_scores = [0.0 for _ in timestamps]
        player_b_reach_scores = [0.0 for _ in timestamps]
        player_a_net_approach_scores = [0.0 for _ in timestamps]
        player_b_net_approach_scores = [0.0 for _ in timestamps]
        player_a_face_hidden_scores = [0.0 for _ in timestamps]
        player_b_face_hidden_scores = [0.0 for _ in timestamps]
        player_a_face_touch_scores = [0.0 for _ in timestamps]
        player_b_face_touch_scores = [0.0 for _ in timestamps]
    else:
        person_model = YOLO(str(pose_w_path))
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {v_path}")

        if player_signal_source == "role_tracker":
            role_series = _collect_role_tracker_energies(
                cap,
                person_model,
                roi=roi,
                fps=info.fps,
                stride=max(1, int(stride)),
                device=device,
                player_margin_px=int(player_margin_px),
                frame_w=int(info.width),
                frame_h=int(info.height),
            )
            timestamps = role_series.timestamps
            table_energies = role_series.table_energies
            player_a_energies = role_series.player_a_energies
            player_b_energies = role_series.player_b_energies
            player_a_crouch_scores = role_series.player_a_crouch_scores
            player_b_crouch_scores = role_series.player_b_crouch_scores
            player_a_serve_scores = role_series.player_a_serve_scores
            player_b_serve_scores = role_series.player_b_serve_scores
            player_a_upper_body_scores = role_series.player_a_upper_body_scores
            player_b_upper_body_scores = role_series.player_b_upper_body_scores
            player_a_footwork_scores = role_series.player_a_footwork_scores
            player_b_footwork_scores = role_series.player_b_footwork_scores
            player_a_reach_scores = role_series.player_a_reach_scores
            player_b_reach_scores = role_series.player_b_reach_scores
            player_a_net_approach_scores = role_series.player_a_net_approach_scores
            player_b_net_approach_scores = role_series.player_b_net_approach_scores
            player_a_face_hidden_scores = role_series.player_a_face_hidden_scores
            player_b_face_hidden_scores = role_series.player_b_face_hidden_scores
            player_a_face_touch_scores = role_series.player_a_face_touch_scores
            player_b_face_touch_scores = role_series.player_b_face_touch_scores
        else:
            timestamps, table_energies, player_a_energies, player_b_energies = _collect_nearest_two_energies(
                cap,
                person_model,
                roi=roi,
                fps=info.fps,
                stride=max(1, int(stride)),
                device=device,
                player_margin_px=int(player_margin_px),
            )
            player_a_crouch_scores = [0.0 for _ in timestamps]
            player_b_crouch_scores = [0.0 for _ in timestamps]
            player_a_serve_scores = [0.0 for _ in timestamps]
            player_b_serve_scores = [0.0 for _ in timestamps]
            player_a_upper_body_scores = [0.0 for _ in timestamps]
            player_b_upper_body_scores = [0.0 for _ in timestamps]
            player_a_footwork_scores = [0.0 for _ in timestamps]
            player_b_footwork_scores = [0.0 for _ in timestamps]
            player_a_reach_scores = [0.0 for _ in timestamps]
            player_b_reach_scores = [0.0 for _ in timestamps]
            player_a_net_approach_scores = [0.0 for _ in timestamps]
            player_b_net_approach_scores = [0.0 for _ in timestamps]
            player_a_face_hidden_scores = [0.0 for _ in timestamps]
            player_b_face_hidden_scores = [0.0 for _ in timestamps]
            player_a_face_touch_scores = [0.0 for _ in timestamps]
            player_b_face_touch_scores = [0.0 for _ in timestamps]

    if player_signal_source != "none":
        cap.release()
        prod_timestamps, prod_table_energies = _extract_production_table_energies(
            str(v_path),
            roi=roi,
            width=int(info.width),
            height=int(info.height),
            fps=float(info.fps),
            stride=max(1, int(stride)),
            device=device,
        )

        if prod_timestamps and prod_table_energies:
            aligned_len = min(len(prod_timestamps), max(0, len(player_a_energies) - 1), max(0, len(player_b_energies) - 1))
            timestamps = prod_timestamps[:aligned_len]
            table_energies = prod_table_energies[:aligned_len]
            player_a_energies = player_a_energies[1 : 1 + aligned_len]
            player_b_energies = player_b_energies[1 : 1 + aligned_len]
            player_a_crouch_scores = player_a_crouch_scores[1 : 1 + aligned_len]
            player_b_crouch_scores = player_b_crouch_scores[1 : 1 + aligned_len]
            player_a_serve_scores = player_a_serve_scores[1 : 1 + aligned_len]
            player_b_serve_scores = player_b_serve_scores[1 : 1 + aligned_len]
            player_a_upper_body_scores = player_a_upper_body_scores[1 : 1 + aligned_len]
            player_b_upper_body_scores = player_b_upper_body_scores[1 : 1 + aligned_len]
            player_a_footwork_scores = player_a_footwork_scores[1 : 1 + aligned_len]
            player_b_footwork_scores = player_b_footwork_scores[1 : 1 + aligned_len]
            player_a_reach_scores = player_a_reach_scores[1 : 1 + aligned_len]
            player_b_reach_scores = player_b_reach_scores[1 : 1 + aligned_len]
            player_a_net_approach_scores = player_a_net_approach_scores[1 : 1 + aligned_len]
            player_b_net_approach_scores = player_b_net_approach_scores[1 : 1 + aligned_len]
            player_a_face_hidden_scores = player_a_face_hidden_scores[1 : 1 + aligned_len]
            player_b_face_hidden_scores = player_b_face_hidden_scores[1 : 1 + aligned_len]
            player_a_face_touch_scores = player_a_face_touch_scores[1 : 1 + aligned_len]
            player_b_face_touch_scores = player_b_face_touch_scores[1 : 1 + aligned_len]
        elif timestamps:
            timestamps = timestamps[1:]
            table_energies = table_energies[1:]
            player_a_energies = player_a_energies[1:]
            player_b_energies = player_b_energies[1:]
            player_a_crouch_scores = player_a_crouch_scores[1:]
            player_b_crouch_scores = player_b_crouch_scores[1:]
            player_a_serve_scores = player_a_serve_scores[1:]
            player_b_serve_scores = player_b_serve_scores[1:]
            player_a_upper_body_scores = player_a_upper_body_scores[1:]
            player_b_upper_body_scores = player_b_upper_body_scores[1:]
            player_a_footwork_scores = player_a_footwork_scores[1:]
            player_b_footwork_scores = player_b_footwork_scores[1:]
            player_a_reach_scores = player_a_reach_scores[1:]
            player_b_reach_scores = player_b_reach_scores[1:]
            player_a_net_approach_scores = player_a_net_approach_scores[1:]
            player_b_net_approach_scores = player_b_net_approach_scores[1:]
            player_a_face_hidden_scores = player_a_face_hidden_scores[1:]
            player_b_face_hidden_scores = player_b_face_hidden_scores[1:]
            player_a_face_touch_scores = player_a_face_touch_scores[1:]
            player_b_face_touch_scores = player_b_face_touch_scores[1:]

    if ball_signal_source == "classical":
        ball_signals = extract_ball_motion_energies(
            str(v_path),
            roi=roi,
            frame_w=int(info.width),
            frame_h=int(info.height),
            fps=float(info.fps),
            stride=max(1, int(stride)),
            profile=ball_tracking_profile,
        )
        aligned_len = min(len(timestamps), len(ball_signals.timestamps), len(ball_signals.energies))
        timestamps = timestamps[:aligned_len]
        table_energies = table_energies[:aligned_len]
        player_a_energies = player_a_energies[:aligned_len]
        player_b_energies = player_b_energies[:aligned_len]
        player_a_crouch_scores = player_a_crouch_scores[:aligned_len]
        player_b_crouch_scores = player_b_crouch_scores[:aligned_len]
        player_a_serve_scores = player_a_serve_scores[:aligned_len]
        player_b_serve_scores = player_b_serve_scores[:aligned_len]
        player_a_upper_body_scores = player_a_upper_body_scores[:aligned_len]
        player_b_upper_body_scores = player_b_upper_body_scores[:aligned_len]
        player_a_footwork_scores = player_a_footwork_scores[:aligned_len]
        player_b_footwork_scores = player_b_footwork_scores[:aligned_len]
        player_a_reach_scores = player_a_reach_scores[:aligned_len]
        player_b_reach_scores = player_b_reach_scores[:aligned_len]
        player_a_net_approach_scores = player_a_net_approach_scores[:aligned_len]
        player_b_net_approach_scores = player_b_net_approach_scores[:aligned_len]
        player_a_face_hidden_scores = player_a_face_hidden_scores[:aligned_len]
        player_b_face_hidden_scores = player_b_face_hidden_scores[:aligned_len]
        player_a_face_touch_scores = player_a_face_touch_scores[:aligned_len]
        player_b_face_touch_scores = player_b_face_touch_scores[:aligned_len]
        ball_energies = ball_signals.energies[:aligned_len]
    else:
        ball_energies = [0.0 for _ in timestamps]

    player_energies = [
        max(float(a), float(b))
        for a, b in zip(player_a_energies, player_b_energies)
    ]

    table_norm = _smooth_and_normalize(table_energies)
    ball_norm = _smooth_and_normalize(ball_energies)
    player_a_norm = _smooth_and_normalize(player_a_energies)
    player_b_norm = _smooth_and_normalize(player_b_energies)
    player_norm = np.maximum(player_a_norm, player_b_norm)
    fused_norm = np.maximum(
        np.maximum(table_norm, player_norm * float(player_fuse_gain)),
        ball_norm * float(ball_fuse_gain),
    )

    return MultiStreamSignals(
        roi=roi,
        timestamps=timestamps,
        table_energies=table_energies,
        ball_energies=ball_energies,
        player_a_energies=player_a_energies,
        player_b_energies=player_b_energies,
        player_energies=player_energies,
        fused_energies=fused_norm.tolist(),
        effective_fps=float(info.fps / max(1, stride)),
        player_signal_source=player_signal_source,
        ball_signal_source=ball_signal_source,
        player_a_crouch_scores=player_a_crouch_scores,
        player_b_crouch_scores=player_b_crouch_scores,
        player_a_serve_scores=player_a_serve_scores,
        player_b_serve_scores=player_b_serve_scores,
        player_a_upper_body_scores=player_a_upper_body_scores,
        player_b_upper_body_scores=player_b_upper_body_scores,
        player_a_footwork_scores=player_a_footwork_scores,
        player_b_footwork_scores=player_b_footwork_scores,
        player_a_reach_scores=player_a_reach_scores,
        player_b_reach_scores=player_b_reach_scores,
        player_a_net_approach_scores=player_a_net_approach_scores,
        player_b_net_approach_scores=player_b_net_approach_scores,
        player_a_face_hidden_scores=player_a_face_hidden_scores,
        player_b_face_hidden_scores=player_b_face_hidden_scores,
        player_a_face_touch_scores=player_a_face_touch_scores,
        player_b_face_touch_scores=player_b_face_touch_scores,
    )


def detect_multistream_rallies(
    signals: MultiStreamSignals,
    *,
    mode: str = "fused",
) -> List[RallySegment]:
    if mode not in {"table", "player", "ball", "fused", "table_refined", "table_ball_refined"}:
        raise ValueError(f"Invalid mode: {mode}")
    if mode == "player" and signals.player_signal_source == "none":
        raise ValueError("Player-only mode requires a real player signal source.")
    if mode == "ball" and signals.ball_signal_source == "none":
        raise ValueError("Ball-only mode requires a real ball signal source.")

    if mode in {"table", "table_refined", "table_ball_refined"}:
        energies = signals.table_energies
    elif mode == "player":
        energies = signals.player_energies
    elif mode == "ball":
        energies = signals.ball_energies
    else:
        energies = signals.fused_energies
    detect_kwargs = {}
    if mode == "ball":
        detect_kwargs = {
            "high_thresh": 0.28,
            "low_thresh": 0.08,
            "max_gap_sec": 1.15,
            "long_segment_sec": 10.0,
            "split_gap_sec": 0.40,
            "min_split_dur_sec": 1.5,
            "artifact_min_dur_sec": 1.2,
        }
    elif mode == "player":
        if signals.player_signal_source == "role_tracker" and signals.player_a_crouch_scores and signals.player_b_crouch_scores:
            return _detect_player_sandwich_rallies(signals)
        detect_kwargs = {
            "high_thresh": 0.22,
            "low_thresh": 0.09,
            "max_gap_sec": 1.35,
            "long_segment_sec": 10.0,
            "split_gap_sec": 0.45,
            "min_split_dur_sec": 1.4,
            "artifact_min_dur_sec": 1.1,
        }
    segments = detect_rally_segments_advanced_gpu(
        list(energies),
        list(signals.timestamps),
        effective_fps=signals.effective_fps,
        **detect_kwargs,
    )
    if mode in {"table", "player", "ball"}:
        if mode == "ball":
            return _merge_ball_split_pair_artifacts(segments)
        return segments
    if mode == "table_refined":
        return _refine_table_segments_with_role_support(
            segments,
            timestamps=signals.timestamps,
            table_energies=signals.table_energies,
            player_a_energies=signals.player_a_energies,
            player_b_energies=signals.player_b_energies,
        )
    if mode == "table_ball_refined":
        return _merge_segments_with_ball_support(
            segments,
            timestamps=signals.timestamps,
            ball_energies=signals.ball_energies,
        )
    return segments
