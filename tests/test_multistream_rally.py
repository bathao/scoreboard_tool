import numpy as np

import scripts.generate_draft_multistream as generate_draft_multistream
from backend.ai_multistream_rally import MultiStreamSignals, detect_multistream_rallies
from backend.ai_multistream_rally import (
    _dedupe_player_sandwich_start_candidates,
    _detect_player_sandwich_rallies,
    _detect_player_sandwich_rallies_from_diagnostics,
    _calc_role_face_hidden_raw,
    _infer_forced_let_indices_from_starter_roles,
    _infer_player_serve_mode_from_starter_roles,
    _merge_player_start_candidates,
    _repair_double_serve_role_singletons,
    PlayerStateMachineDiagnostics,
    PlayerRallyStartCandidate,
    _build_role_energy_series,
    _compute_player_rally_start_candidates,
    _compute_player_state_machine_diagnostics,
    _detect_player_state_machine_rallies,
    _merge_ball_split_pair_artifacts,
    _merge_segments_with_ball_support,
    _refine_table_segments_with_role_support,
    _select_player_sandwich_start_candidates,
)
from backend.ai_rally_segmentation import RallySegment
from backend.ai_table_roi import TableROI
from backend.offline_player_tracker import TrackletObservation


def _obs(frame_idx: int, *, cx: float, cy: float, wrist_shift: float = 0.0) -> TrackletObservation:
    box = (int(cx - 25), int(cy - 50), int(cx + 25), int(cy + 50))
    keypoints = np.zeros((17, 2), dtype=np.float32)
    keypoints[9] = (cx - 5 + wrist_shift, cy)
    keypoints[10] = (cx + 5 + wrist_shift, cy)
    return TrackletObservation(
        frame_idx=frame_idx,
        box=box,
        center=(cx, cy),
        keypoints=keypoints,
        confidence=0.95,
    )


def _obs_with_pose(
    frame_idx: int,
    *,
    box,
    nose,
    left_eye,
    right_eye,
    left_shoulder,
    right_shoulder,
    left_hip,
    right_hip,
) -> TrackletObservation:
    keypoints = np.zeros((17, 2), dtype=np.float32)
    keypoints[0] = nose
    keypoints[1] = left_eye
    keypoints[2] = right_eye
    keypoints[5] = left_shoulder
    keypoints[6] = right_shoulder
    keypoints[11] = left_hip
    keypoints[12] = right_hip
    x1, y1, x2, y2 = box
    return TrackletObservation(
        frame_idx=frame_idx,
        box=box,
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        keypoints=keypoints,
        confidence=0.95,
    )


def _player_diagnostics(
    timestamps,
    *,
    motion_a=None,
    motion_b=None,
    crouch_a=None,
    crouch_b=None,
    serve_a=None,
    serve_b=None,
    upper_a=None,
    upper_b=None,
    foot_a=None,
    foot_b=None,
    reach_a=None,
    reach_b=None,
    approach_a=None,
    approach_b=None,
):
    timestamps = list(timestamps)
    n = len(timestamps)

    def fill(values):
        return list(values) if values is not None else [0.0] * n

    return PlayerStateMachineDiagnostics(
        timestamps=timestamps,
        segments=[],
        phase_by_frame=["search_ready"] * n,
        server_role_by_frame=[""] * n,
        ready_recent_flags=[False] * n,
        live_now_flags=[False] * n,
        dead_now_flags=[False] * n,
        catch_proxy_scores=[0.0] * n,
        quiet_after_catch_scores=[0.0] * n,
        ready_pair=[0.0] * n,
        live_pair=[0.0] * n,
        casual_pair=[0.0] * n,
        stand_pair=[0.0] * n,
        motion_a=fill(motion_a),
        motion_b=fill(motion_b),
        crouch_a=fill(crouch_a),
        crouch_b=fill(crouch_b),
        serve_a=fill(serve_a),
        serve_b=fill(serve_b),
        upper_a=fill(upper_a),
        upper_b=fill(upper_b),
        foot_a=fill(foot_a),
        foot_b=fill(foot_b),
        reach_a=fill(reach_a),
        reach_b=fill(reach_b),
        approach_a=fill(approach_a),
        approach_b=fill(approach_b),
        start_events=[],
    )


def test_role_energy_series_uses_visible_motion():
    frame_indices = [0, 2, 4]
    role_frames = {
        0: {"A": _obs(0, cx=100, cy=200, wrist_shift=0.0)},
        2: {"A": _obs(2, cx=110, cy=198, wrist_shift=6.0)},
        4: {"A": _obs(4, cx=124, cy=196, wrist_shift=12.0)},
    }
    role_state_frames = {0: {"A": "visible"}, 2: {"A": "visible"}, 4: {"A": "visible"}}

    energies = _build_role_energy_series(
        frame_indices,
        role_frames,
        role_state_frames,
        role="A",
    )

    assert energies[0] == 0.0
    assert energies[1] > 0.0
    assert energies[2] > 0.0


def test_role_energy_series_short_occlusion_keeps_decay_signal():
    frame_indices = [0, 2, 4]
    role_frames = {
        0: {"B": _obs(0, cx=300, cy=120, wrist_shift=0.0)},
        2: {"B": _obs(2, cx=312, cy=122, wrist_shift=8.0)},
    }
    role_state_frames = {
        0: {"B": "visible"},
        2: {"B": "visible"},
        4: {"B": "occluded"},
    }

    energies = _build_role_energy_series(
        frame_indices,
        role_frames,
        role_state_frames,
        role="B",
        occluded_hold_samples=3,
    )

    assert energies[1] > 0.0
    assert energies[2] > 0.0
    assert energies[2] < energies[1]


def test_face_hidden_raw_uses_profile_turn_when_face_keypoints_still_visible():
    profile_obs = _obs_with_pose(
        0,
        box=(100, 100, 200, 320),
        nose=(151, 132),
        left_eye=(153, 128),
        right_eye=(149, 128),
        left_shoulder=(154, 170),
        right_shoulder=(148, 171),
        left_hip=(153, 248),
        right_hip=(149, 249),
    )
    frontal_obs = _obs_with_pose(
        0,
        box=(100, 100, 200, 320),
        nose=(150, 132),
        left_eye=(156, 128),
        right_eye=(144, 128),
        left_shoulder=(170, 170),
        right_shoulder=(130, 171),
        left_hip=(166, 248),
        right_hip=(134, 249),
    )

    assert _calc_role_face_hidden_raw(profile_obs, None) >= 0.75
    assert _calc_role_face_hidden_raw(frontal_obs, None) <= 0.15


def test_role_refine_splits_long_table_segment_on_quiet_gap():
    timestamps = [float(i) for i in range(13)]
    segments = [RallySegment(t_start=0.0, t_end=12.0, confidence=0.7, flags=[])]
    table_energies = [1.0, 1.0, 0.95, 0.92, 0.88, 0.18, 0.12, 0.16, 0.86, 0.92, 0.97, 1.0, 0.95]
    player_a = [0.6, 0.64, 0.62, 0.58, 0.54, 0.03, 0.02, 0.04, 0.52, 0.57, 0.6, 0.63, 0.61]
    player_b = [0.58, 0.62, 0.59, 0.55, 0.5, 0.03, 0.02, 0.03, 0.48, 0.55, 0.58, 0.6, 0.57]

    refined = _refine_table_segments_with_role_support(
        segments,
        timestamps=timestamps,
        table_energies=table_energies,
        player_a_energies=player_a,
        player_b_energies=player_b,
        quiet_run_sec=1.0,
        boundary_guard_sec=1.0,
    )

    assert len(refined) == 2
    assert refined[0].t_end < refined[1].t_start + 0.01
    assert "role_quiet_gap_split" in refined[0].flags
    assert "role_quiet_gap_split" in refined[1].flags


def test_role_refine_does_not_split_when_players_stay_active():
    timestamps = [float(i) for i in range(13)]
    segments = [RallySegment(t_start=0.0, t_end=12.0, confidence=0.7, flags=[])]
    table_energies = [1.0, 1.0, 0.95, 0.92, 0.88, 0.18, 0.12, 0.16, 0.86, 0.92, 0.97, 1.0, 0.95]
    player_a = [0.6, 0.64, 0.62, 0.58, 0.54, 0.42, 0.38, 0.41, 0.52, 0.57, 0.6, 0.63, 0.61]
    player_b = [0.58, 0.62, 0.59, 0.55, 0.5, 0.4, 0.36, 0.39, 0.48, 0.55, 0.58, 0.6, 0.57]

    refined = _refine_table_segments_with_role_support(
        segments,
        timestamps=timestamps,
        table_energies=table_energies,
        player_a_energies=player_a,
        player_b_energies=player_b,
        quiet_run_sec=1.0,
        boundary_guard_sec=1.0,
    )

    assert len(refined) == 1


def test_ball_refine_merges_contiguous_split_when_ball_stays_active():
    timestamps = [float(i) for i in range(12)]
    segments = [
        RallySegment(t_start=0.0, t_end=4.0, confidence=0.6, flags=["split_long"]),
        RallySegment(t_start=4.0, t_end=8.0, confidence=0.6, flags=["split_long"]),
    ]
    ball = [0.0, 0.2, 0.45, 0.6, 0.7, 0.62, 0.5, 0.18, 0.05, 0.0, 0.0, 0.0]

    merged = _merge_segments_with_ball_support(
        segments,
        timestamps=timestamps,
        ball_energies=ball,
        active_peak_thresh=0.12,
        active_mean_thresh=0.08,
    )

    assert len(merged) == 1
    assert merged[0].t_start == 0.0
    assert merged[0].t_end == 8.0
    assert "ball_gap_merge" in merged[0].flags


def test_ball_refine_keeps_split_when_ball_is_quiet():
    timestamps = [float(i) for i in range(12)]
    segments = [
        RallySegment(t_start=0.0, t_end=4.0, confidence=0.6, flags=["split_long"]),
        RallySegment(t_start=4.0, t_end=8.0, confidence=0.6, flags=["split_long"]),
    ]
    ball = [0.0 for _ in range(12)]

    merged = _merge_segments_with_ball_support(
        segments,
        timestamps=timestamps,
        ball_energies=ball,
        active_peak_thresh=0.12,
        active_mean_thresh=0.08,
    )

    assert len(merged) == 2


def test_ball_pair_merge_merges_short_contiguous_split_pair():
    segments = [
        RallySegment(t_start=0.0, t_end=3.2, confidence=0.5, flags=["split_long"]),
        RallySegment(t_start=3.2, t_end=9.0, confidence=0.6, flags=["split_long"]),
        RallySegment(t_start=12.0, t_end=18.0, confidence=0.7, flags=["split_long"]),
    ]

    merged = _merge_ball_split_pair_artifacts(
        segments,
        short_piece_sec=3.8,
        max_pair_sec=10.0,
    )

    assert len(merged) == 2
    assert merged[0].t_start == 0.0
    assert merged[0].t_end == 9.0
    assert "ball_pair_merge" in merged[0].flags
    assert merged[1].t_start == 12.0


def test_detect_multistream_rallies_ball_mode_uses_ball_signal(monkeypatch):
    captured = {}

    def fake_detect(energies, timestamps, effective_fps, **kwargs):
        captured["energies"] = list(energies)
        captured["timestamps"] = list(timestamps)
        captured["effective_fps"] = effective_fps
        captured["kwargs"] = dict(kwargs)
        return [RallySegment(t_start=1.0, t_end=3.0, confidence=0.75, flags=["ball_only"])]

    monkeypatch.setattr("backend.ai_multistream_rally.detect_rally_segments_advanced_gpu", fake_detect)

    signals = MultiStreamSignals(
        roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
        timestamps=[0.0, 1.0, 2.0, 3.0],
        table_energies=[0.9, 0.8, 0.7, 0.6],
        ball_energies=[0.0, 0.35, 0.7, 0.2],
        player_a_energies=[0.0, 0.0, 0.0, 0.0],
        player_b_energies=[0.0, 0.0, 0.0, 0.0],
        player_energies=[0.0, 0.0, 0.0, 0.0],
        fused_energies=[0.9, 0.8, 0.7, 0.6],
        effective_fps=30.0,
        player_signal_source="role_tracker",
        ball_signal_source="classical",
    )

    segments = detect_multistream_rallies(signals, mode="ball")

    assert captured["energies"] == signals.ball_energies
    assert captured["timestamps"] == signals.timestamps
    assert captured["effective_fps"] == signals.effective_fps
    assert captured["kwargs"]["max_gap_sec"] == 1.15
    assert captured["kwargs"]["high_thresh"] == 0.28
    assert captured["kwargs"]["artifact_min_dur_sec"] == 1.2
    assert len(segments) == 1
    assert segments[0].flags == ["ball_only"]


def test_detect_multistream_rallies_player_mode_uses_player_signal(monkeypatch):
    captured = {}

    def fake_detect(energies, timestamps, effective_fps, **kwargs):
        captured["energies"] = list(energies)
        captured["timestamps"] = list(timestamps)
        captured["effective_fps"] = effective_fps
        captured["kwargs"] = dict(kwargs)
        return [RallySegment(t_start=0.5, t_end=2.5, confidence=0.72, flags=["player_only"])]

    monkeypatch.setattr("backend.ai_multistream_rally.detect_rally_segments_advanced_gpu", fake_detect)

    signals = MultiStreamSignals(
        roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
        timestamps=[0.0, 1.0, 2.0, 3.0],
        table_energies=[0.9, 0.8, 0.7, 0.6],
        ball_energies=[0.0, 0.35, 0.7, 0.2],
        player_a_energies=[0.0, 0.2, 0.5, 0.1],
        player_b_energies=[0.0, 0.1, 0.4, 0.3],
        player_energies=[0.0, 0.2, 0.5, 0.3],
        fused_energies=[0.9, 0.8, 0.7, 0.6],
        effective_fps=30.0,
        player_signal_source="role_tracker",
        ball_signal_source="none",
    )

    segments = detect_multistream_rallies(signals, mode="player")

    assert captured["energies"] == signals.player_energies
    assert captured["timestamps"] == signals.timestamps
    assert captured["effective_fps"] == signals.effective_fps
    assert captured["kwargs"]["max_gap_sec"] == 1.35
    assert captured["kwargs"]["high_thresh"] == 0.22
    assert captured["kwargs"]["artifact_min_dur_sec"] == 1.1
    assert len(segments) == 1
    assert segments[0].flags == ["player_only"]


def test_player_state_machine_ignores_motion_without_ready_or_serve():
    signals = MultiStreamSignals(
        roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
        timestamps=[float(i) for i in range(8)],
        table_energies=[0.0] * 8,
        ball_energies=[0.0] * 8,
        player_a_energies=[0.0, 0.05, 0.18, 0.22, 0.16, 0.04, 0.0, 0.0],
        player_b_energies=[0.0, 0.04, 0.11, 0.10, 0.08, 0.03, 0.0, 0.0],
        player_energies=[0.0, 0.05, 0.18, 0.22, 0.16, 0.04, 0.0, 0.0],
        fused_energies=[0.0] * 8,
        effective_fps=1.0,
        player_signal_source="role_tracker",
        ball_signal_source="none",
        player_a_crouch_scores=[0.0] * 8,
        player_b_crouch_scores=[0.0] * 8,
        player_a_serve_scores=[0.0] * 8,
        player_b_serve_scores=[0.0] * 8,
    )

    segments = _detect_player_state_machine_rallies(signals)
    assert segments == []


def test_player_state_machine_detects_one_ready_serve_rally():
    timestamps = [float(i) for i in range(11)]
    signals = MultiStreamSignals(
        roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
        timestamps=timestamps,
        table_energies=[0.0] * len(timestamps),
        ball_energies=[0.0] * len(timestamps),
        player_a_energies=[0.02, 0.02, 0.03, 0.55, 0.32, 0.26, 0.22, 0.06, 0.03, 0.02, 0.02],
        player_b_energies=[0.02, 0.02, 0.03, 0.08, 0.33, 0.28, 0.24, 0.05, 0.03, 0.02, 0.02],
        player_energies=[0.02, 0.02, 0.03, 0.55, 0.33, 0.28, 0.24, 0.06, 0.03, 0.02, 0.02],
        fused_energies=[0.0] * len(timestamps),
        effective_fps=1.0,
        player_signal_source="role_tracker",
        ball_signal_source="none",
        player_a_crouch_scores=[0.82, 0.85, 0.88, 0.84, 0.76, 0.70, 0.65, 0.10, 0.05, 0.04, 0.04],
        player_b_crouch_scores=[0.80, 0.83, 0.86, 0.82, 0.78, 0.72, 0.64, 0.10, 0.05, 0.04, 0.04],
        player_a_serve_scores=[0.02, 0.03, 0.08, 0.78, 0.28, 0.18, 0.08, 0.02, 0.02, 0.02, 0.02],
        player_b_serve_scores=[0.02, 0.03, 0.06, 0.05, 0.18, 0.15, 0.06, 0.02, 0.02, 0.02, 0.02],
    )

    segments = _detect_player_state_machine_rallies(signals)

    assert len(segments) == 1
    assert segments[0].t_start <= 3.0
    assert segments[0].t_end >= 6.0
    assert "player_state_machine" in segments[0].flags


def test_player_state_machine_diagnostics_exports_one_start_event():
    timestamps = [float(i) for i in range(11)]
    signals = MultiStreamSignals(
        roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
        timestamps=timestamps,
        table_energies=[0.0] * len(timestamps),
        ball_energies=[0.0] * len(timestamps),
        player_a_energies=[0.02, 0.02, 0.03, 0.55, 0.32, 0.26, 0.22, 0.06, 0.03, 0.02, 0.02],
        player_b_energies=[0.02, 0.02, 0.03, 0.08, 0.33, 0.28, 0.24, 0.05, 0.03, 0.02, 0.02],
        player_energies=[0.02, 0.02, 0.03, 0.55, 0.33, 0.28, 0.24, 0.06, 0.03, 0.02, 0.02],
        fused_energies=[0.0] * len(timestamps),
        effective_fps=1.0,
        player_signal_source="role_tracker",
        ball_signal_source="none",
        player_a_crouch_scores=[0.82, 0.85, 0.88, 0.84, 0.76, 0.70, 0.65, 0.10, 0.05, 0.04, 0.04],
        player_b_crouch_scores=[0.80, 0.83, 0.86, 0.82, 0.78, 0.72, 0.64, 0.10, 0.05, 0.04, 0.04],
        player_a_serve_scores=[0.02, 0.03, 0.08, 0.78, 0.28, 0.18, 0.08, 0.02, 0.02, 0.02, 0.02],
        player_b_serve_scores=[0.02, 0.03, 0.06, 0.05, 0.18, 0.15, 0.06, 0.02, 0.02, 0.02, 0.02],
    )

    diagnostics = _compute_player_state_machine_diagnostics(signals)

    assert len(diagnostics.start_events) == 1
    assert diagnostics.start_events[0].trigger_timestamp >= 2.0
    assert diagnostics.start_events[0].server_role == "A"


def test_player_start_candidate_miner_detects_multiple_independent_onsets():
    timestamps = [float(i) for i in range(40)]
    n = len(timestamps)
    motion_a = [0.08] * n
    motion_b = [0.08] * n
    crouch_a = [0.42] * n
    crouch_b = [0.42] * n
    serve_a = [0.05] * n
    serve_b = [0.05] * n
    upper_a = [0.05] * n
    upper_b = [0.05] * n
    foot_a = [0.05] * n
    foot_b = [0.05] * n
    reach_a = [0.18] * n
    reach_b = [0.18] * n

    def inject(role_values, start_idx: int) -> None:
        role_values["crouch"][start_idx : start_idx + 3] = [1.0, 0.96, 0.82]
        role_values["reach"][start_idx : start_idx + 3] = [0.82, 0.76, 0.58]
        role_values["serve"][start_idx : start_idx + 3] = [0.42, 0.86, 0.36]
        role_values["upper"][start_idx : start_idx + 3] = [0.46, 0.92, 0.32]
        role_values["foot"][start_idx : start_idx + 3] = [0.48, 0.78, 0.28]
        role_values["opp_crouch"][start_idx : start_idx + 3] = [0.72, 0.70, 0.62]

    inject(
        {
            "crouch": crouch_b,
            "reach": reach_b,
            "serve": serve_b,
            "upper": upper_b,
            "foot": foot_b,
            "opp_crouch": crouch_a,
        },
        3,
    )
    inject(
        {
            "crouch": crouch_b,
            "reach": reach_b,
            "serve": serve_b,
            "upper": upper_b,
            "foot": foot_b,
            "opp_crouch": crouch_a,
        },
        12,
    )
    inject(
        {
            "crouch": crouch_a,
            "reach": reach_a,
            "serve": serve_a,
            "upper": upper_a,
            "foot": foot_a,
            "opp_crouch": crouch_b,
        },
        25,
    )
    inject(
        {
            "crouch": crouch_a,
            "reach": reach_a,
            "serve": serve_a,
            "upper": upper_a,
            "foot": foot_a,
            "opp_crouch": crouch_b,
        },
        34,
    )

    diagnostics = _player_diagnostics(
        timestamps,
        motion_a=motion_a,
        motion_b=motion_b,
        crouch_a=crouch_a,
        crouch_b=crouch_b,
        serve_a=serve_a,
        serve_b=serve_b,
        upper_a=upper_a,
        upper_b=upper_b,
        foot_a=foot_a,
        foot_b=foot_b,
        reach_a=reach_a,
        reach_b=reach_b,
    )

    candidates = _compute_player_rally_start_candidates(diagnostics)

    assert [(candidate.role, candidate.sample_idx) for candidate in candidates] == [
        ("B", 3),
        ("B", 12),
        ("A", 25),
        ("A", 34),
    ]


def test_player_start_candidate_miner_prefers_early_onset_over_late_peak():
    timestamps = [float(i) for i in range(10)]
    diagnostics = _player_diagnostics(
        timestamps,
        crouch_a=[0.30, 0.30, 0.30, 0.30, 0.40, 0.98, 0.95, 0.70, 0.30, 0.30],
        crouch_b=[0.30, 0.30, 0.30, 0.30, 0.65, 0.68, 0.62, 0.40, 0.30, 0.30],
        serve_a=[0.05, 0.05, 0.05, 0.05, 0.10, 0.46, 0.92, 0.24, 0.05, 0.05],
        serve_b=[0.05] * 10,
        upper_a=[0.05, 0.05, 0.05, 0.05, 0.10, 0.50, 0.95, 0.22, 0.05, 0.05],
        upper_b=[0.05] * 10,
        foot_a=[0.05, 0.05, 0.05, 0.05, 0.08, 0.48, 0.84, 0.20, 0.05, 0.05],
        foot_b=[0.05] * 10,
        reach_a=[0.15, 0.15, 0.15, 0.15, 0.22, 0.84, 0.76, 0.44, 0.15, 0.15],
        reach_b=[0.15] * 10,
        motion_a=[0.05] * 10,
        motion_b=[0.05] * 10,
    )

    candidates = _compute_player_rally_start_candidates(diagnostics)

    assert len(candidates) == 1
    assert candidates[0].role == "A"
    assert candidates[0].sample_idx == 5
    assert candidates[0].episode_peak_sample_idx == 6


def test_player_sandwich_selector_rejects_stroke_like_start_and_keeps_real_serve_chain():
    timestamps = [i * 0.2 for i in range(18)]
    n = len(timestamps)
    motion_a = [0.02] * n
    motion_b = [0.02] * n
    crouch_a = [0.12] * n
    crouch_b = [0.12] * n
    serve_a = [0.02] * n
    serve_b = [0.02] * n
    upper_a = [0.02] * n
    upper_b = [0.02] * n
    foot_a = [0.02] * n
    foot_b = [0.02] * n
    reach_a = [0.14] * n
    reach_b = [0.14] * n

    crouch_a[0:4] = [0.86, 0.88, 0.96, 0.82]
    crouch_b[0:4] = [0.84, 0.86, 0.82, 0.74]
    reach_a[2:4] = [0.86, 0.72]
    serve_a[2:4] = [0.42, 0.92]
    upper_a[2:4] = [0.48, 0.88]
    foot_a[2:4] = [0.36, 0.72]
    motion_a[2:5] = [0.10, 0.26, 0.18]
    upper_b[3:6] = [0.12, 0.44, 0.28]
    foot_b[3:6] = [0.08, 0.38, 0.24]
    motion_b[3:6] = [0.08, 0.34, 0.24]

    crouch_a[12:15] = [0.72, 0.66, 0.48]
    crouch_b[12:15] = [0.58, 0.46, 0.36]
    reach_b[12:15] = [0.74, 0.68, 0.40]
    serve_b[12:15] = [0.92, 0.86, 0.30]
    upper_b[12:15] = [0.96, 0.88, 0.28]
    foot_b[12:15] = [0.88, 0.72, 0.22]
    motion_b[11:15] = [0.30, 0.42, 0.34, 0.20]
    motion_a[11:15] = [0.26, 0.30, 0.22, 0.14]
    upper_a[11:15] = [0.22, 0.30, 0.24, 0.12]
    foot_a[11:15] = [0.18, 0.28, 0.22, 0.10]

    diagnostics = _player_diagnostics(
        timestamps,
        motion_a=motion_a,
        motion_b=motion_b,
        crouch_a=crouch_a,
        crouch_b=crouch_b,
        serve_a=serve_a,
        serve_b=serve_b,
        upper_a=upper_a,
        upper_b=upper_b,
        foot_a=foot_a,
        foot_b=foot_b,
        reach_a=reach_a,
        reach_b=reach_b,
    )

    raw_candidates = _compute_player_rally_start_candidates(diagnostics)
    selected = _select_player_sandwich_start_candidates(diagnostics)

    assert len(raw_candidates) == 2
    assert [(candidate.role, candidate.sample_idx) for candidate in raw_candidates] == [("A", 2), ("B", 12)]
    assert [(candidate.role, candidate.sample_idx) for candidate in selected] == [("A", 2)]
    assert selected[0].server_peak_delay_sec >= 0.08
    assert selected[0].live_peak_score >= 0.78


def test_player_sandwich_selector_rescues_true_start_when_receiver_ready_drops_at_trigger_frame():
    timestamps = [i * 0.2 for i in range(18)]
    n = len(timestamps)
    motion_a = [0.02] * n
    motion_b = [0.02] * n
    crouch_a = [0.08] * n
    crouch_b = [0.08] * n
    serve_a = [0.02] * n
    serve_b = [0.02] * n
    upper_a = [0.02] * n
    upper_b = [0.02] * n
    foot_a = [0.02] * n
    foot_b = [0.02] * n
    reach_a = [0.12] * n
    reach_b = [0.12] * n

    crouch_a[3:6] = [0.74, 0.76, 0.78]
    reach_a[3:6] = [0.46, 0.48, 0.50]
    serve_a[3:6] = [0.10, 0.10, 0.12]
    upper_a[3:6] = [0.10, 0.10, 0.12]
    foot_a[3:6] = [0.12, 0.12, 0.14]

    crouch_b[2:6] = [0.92, 0.90, 0.88, 0.86]
    reach_b[2:6] = [0.30, 0.28, 0.26, 0.24]
    foot_b[2:6] = [0.08, 0.08, 0.06, 0.06]
    motion_b[2:6] = [0.04, 0.04, 0.03, 0.03]

    crouch_a[5:10] = [0.92, 0.95, 1.00, 0.86, 0.72]
    reach_a[5:10] = [0.60, 0.64, 0.70, 0.66, 0.52]
    serve_a[5:10] = [0.18, 0.28, 0.56, 1.00, 0.44]
    upper_a[5:10] = [0.18, 0.24, 0.34, 0.98, 0.46]
    foot_a[5:10] = [0.38, 0.44, 0.50, 0.20, 0.10]
    motion_a[5:10] = [0.16, 0.22, 0.28, 0.42, 0.20]

    crouch_b[5:8] = [0.0, 0.0, 0.0]
    reach_b[5:8] = [0.0, 0.0, 0.0]
    serve_b[5:8] = [0.0, 0.0, 0.0]
    upper_b[5:8] = [0.0, 0.0, 0.0]
    foot_b[5:8] = [0.0, 0.0, 0.0]
    motion_b[5:8] = [0.0, 0.0, 0.0]

    crouch_b[8:11] = [0.70, 0.66, 0.54]
    reach_b[8:11] = [0.52, 0.62, 0.50]
    upper_b[8:11] = [0.66, 0.40, 0.18]
    foot_b[8:11] = [0.62, 0.28, 0.12]
    motion_b[8:11] = [0.32, 0.18, 0.10]

    diagnostics = _player_diagnostics(
        timestamps,
        motion_a=motion_a,
        motion_b=motion_b,
        crouch_a=crouch_a,
        crouch_b=crouch_b,
        serve_a=serve_a,
        serve_b=serve_b,
        upper_a=upper_a,
        upper_b=upper_b,
        foot_a=foot_a,
        foot_b=foot_b,
        reach_a=reach_a,
        reach_b=reach_b,
    )

    raw_candidates = _compute_player_rally_start_candidates(diagnostics)
    selected = _select_player_sandwich_start_candidates(diagnostics)

    assert [(candidate.role, candidate.sample_idx) for candidate in raw_candidates] == [("A", 6)]
    assert [(candidate.role, candidate.sample_idx) for candidate in selected] == [("A", 6)]
    assert selected[0].ready_pair_score < 0.05
    assert selected[0].opponent_ready_score < 0.30
    assert selected[0].pre_ready_mean >= 0.16
    assert selected[0].server_peak_score >= 0.90
    assert selected[0].receiver_peak_score >= 0.55


def test_player_start_candidate_merge_keeps_clean_prep_anchor_over_later_same_role_burst():
    timestamps = np.asarray([i * 0.2 for i in range(24)], dtype=np.float32)
    candidates = [
        PlayerRallyStartCandidate(
            sample_idx=4,
            timestamp=float(timestamps[4]),
            role="B",
            score=0.596,
            prep_score=0.657,
            launch_score=0.380,
            opponent_ready_score=0.420,
            dominance_ratio=1.180,
            episode_start_sample_idx=4,
            episode_end_sample_idx=4,
            episode_peak_sample_idx=4,
            episode_peak_score=0.620,
            crouch_score=0.670,
            reach_score=0.560,
            serve_score=0.286,
            upper_body_score=0.303,
            footwork_score=0.486,
        ),
        PlayerRallyStartCandidate(
            sample_idx=7,
            timestamp=float(timestamps[7]),
            role="B",
            score=0.669,
            prep_score=0.756,
            launch_score=0.409,
            opponent_ready_score=0.410,
            dominance_ratio=1.360,
            episode_start_sample_idx=7,
            episode_end_sample_idx=10,
            episode_peak_sample_idx=8,
            episode_peak_score=0.752,
            crouch_score=0.790,
            reach_score=0.770,
            serve_score=0.369,
            upper_body_score=0.282,
            footwork_score=0.220,
        ),
        PlayerRallyStartCandidate(
            sample_idx=12,
            timestamp=float(timestamps[12]),
            role="B",
            score=0.733,
            prep_score=0.742,
            launch_score=0.848,
            opponent_ready_score=0.300,
            dominance_ratio=1.920,
            episode_start_sample_idx=12,
            episode_end_sample_idx=14,
            episode_peak_sample_idx=13,
            episode_peak_score=0.900,
            crouch_score=0.740,
            reach_score=0.860,
            serve_score=0.915,
            upper_body_score=0.851,
            footwork_score=1.000,
        ),
        PlayerRallyStartCandidate(
            sample_idx=15,
            timestamp=float(timestamps[15]),
            role="B",
            score=0.709,
            prep_score=0.724,
            launch_score=0.581,
            opponent_ready_score=0.320,
            dominance_ratio=1.600,
            episode_start_sample_idx=15,
            episode_end_sample_idx=16,
            episode_peak_sample_idx=15,
            episode_peak_score=0.932,
            crouch_score=0.720,
            reach_score=0.740,
            serve_score=0.611,
            upper_body_score=0.621,
            footwork_score=0.423,
        ),
    ]

    merged = _merge_player_start_candidates(timestamps, candidates)

    assert [(candidate.role, candidate.sample_idx) for candidate in merged] == [("B", 7)]
    assert merged[0].episode_end_sample_idx == 16
    assert merged[0].episode_peak_sample_idx == 15
    assert merged[0].episode_peak_score == 0.932


def test_player_sandwich_dedupe_rejects_same_role_live_followup_after_true_start():
    timestamps = np.asarray([i * 0.2 for i in range(40)], dtype=np.float32)
    confirmed_candidates = [
        PlayerRallyStartCandidate(
            sample_idx=15,
            timestamp=float(timestamps[15]),
            role="B",
            score=0.684,
            prep_score=0.794,
            launch_score=0.328,
            opponent_ready_score=0.649,
            dominance_ratio=2.187,
            episode_start_sample_idx=12,
            episode_end_sample_idx=23,
            episode_peak_sample_idx=22,
            episode_peak_score=0.925,
            crouch_score=0.872,
            reach_score=0.668,
            serve_score=0.184,
            upper_body_score=0.171,
            footwork_score=0.442,
            ready_pair_score=0.556,
            pre_ready_mean=0.571,
            pre_live_peak=0.627,
            server_action_score=0.345,
            server_peak_score=0.925,
            server_growth_score=0.580,
            server_peak_delay_sec=0.200,
            receiver_peak_score=0.665,
            live_peak_score=1.000,
        ),
        PlayerRallyStartCandidate(
            sample_idx=24,
            timestamp=float(timestamps[24]),
            role="B",
            score=0.729,
            prep_score=0.824,
            launch_score=0.418,
            opponent_ready_score=0.710,
            dominance_ratio=1.189,
            episode_start_sample_idx=24,
            episode_end_sample_idx=27,
            episode_peak_sample_idx=26,
            episode_peak_score=1.000,
            crouch_score=1.000,
            reach_score=0.537,
            serve_score=0.365,
            upper_body_score=0.482,
            footwork_score=0.229,
            ready_pair_score=0.670,
            pre_ready_mean=0.377,
            pre_live_peak=1.000,
            server_action_score=0.482,
            server_peak_score=1.000,
            server_growth_score=0.518,
            server_peak_delay_sec=0.500,
            receiver_peak_score=1.000,
            live_peak_score=1.000,
        ),
    ]

    deduped = _dedupe_player_sandwich_start_candidates(timestamps, confirmed_candidates)

    assert [(candidate.role, candidate.sample_idx) for candidate in deduped] == [("B", 15)]


def test_player_sandwich_selector_rejects_already_live_exchange_false_positive():
    timestamps = [i * 0.2 for i in range(20)]
    n = len(timestamps)
    motion_a = [0.02] * n
    motion_b = [0.02] * n
    crouch_a = [0.12] * n
    crouch_b = [0.12] * n
    serve_a = [0.02] * n
    serve_b = [0.02] * n
    upper_a = [0.02] * n
    upper_b = [0.02] * n
    foot_a = [0.02] * n
    foot_b = [0.02] * n
    reach_a = [0.14] * n
    reach_b = [0.14] * n

    crouch_a[0:4] = [0.86, 0.90, 0.96, 0.82]
    crouch_b[0:4] = [0.84, 0.86, 0.82, 0.74]
    reach_a[2:4] = [0.86, 0.74]
    serve_a[2:4] = [0.42, 0.92]
    upper_a[2:4] = [0.48, 0.88]
    foot_a[2:4] = [0.36, 0.72]
    motion_a[2:5] = [0.10, 0.26, 0.18]
    upper_b[3:6] = [0.12, 0.44, 0.28]
    foot_b[3:6] = [0.08, 0.38, 0.24]
    motion_b[3:6] = [0.08, 0.34, 0.24]

    crouch_a[9:14] = [0.58, 0.60, 0.62, 0.58, 0.44]
    crouch_b[9:15] = [0.64, 0.66, 0.72, 0.76, 0.70, 0.48]
    motion_a[9:15] = [0.28, 0.34, 0.38, 0.40, 0.36, 0.22]
    motion_b[9:15] = [0.30, 0.34, 0.38, 0.42, 0.34, 0.20]
    upper_a[9:15] = [0.22, 0.28, 0.34, 0.40, 0.36, 0.18]
    upper_b[9:15] = [0.24, 0.30, 0.34, 0.44, 0.86, 0.26]
    foot_a[9:15] = [0.18, 0.24, 0.30, 0.34, 0.30, 0.16]
    foot_b[9:15] = [0.20, 0.26, 0.30, 0.38, 0.66, 0.22]
    serve_a[9:15] = [0.10, 0.14, 0.20, 0.22, 0.18, 0.08]
    serve_b[9:15] = [0.12, 0.16, 0.24, 0.36, 0.74, 0.18]
    reach_a[9:15] = [0.22, 0.28, 0.34, 0.40, 0.36, 0.20]
    reach_b[9:15] = [0.24, 0.30, 0.38, 0.86, 0.74, 0.36]

    diagnostics = _player_diagnostics(
        timestamps,
        motion_a=motion_a,
        motion_b=motion_b,
        crouch_a=crouch_a,
        crouch_b=crouch_b,
        serve_a=serve_a,
        serve_b=serve_b,
        upper_a=upper_a,
        upper_b=upper_b,
        foot_a=foot_a,
        foot_b=foot_b,
        reach_a=reach_a,
        reach_b=reach_b,
    )

    raw_candidates = _compute_player_rally_start_candidates(diagnostics)
    selected = _select_player_sandwich_start_candidates(diagnostics)

    assert [(candidate.role, candidate.sample_idx) for candidate in raw_candidates] == [("A", 2), ("B", 12)]
    assert [(candidate.role, candidate.sample_idx) for candidate in selected] == [("A", 2)]


def test_player_sandwich_detector_uses_reset_to_close_rallies():
    timestamps = [float(i) for i in range(16)]
    n = len(timestamps)
    motion_a = [0.03] * n
    motion_b = [0.03] * n
    crouch_a = [0.08] * n
    crouch_b = [0.08] * n
    serve_a = [0.02] * n
    serve_b = [0.02] * n
    upper_a = [0.02] * n
    upper_b = [0.02] * n
    foot_a = [0.02] * n
    foot_b = [0.02] * n
    reach_a = [0.12] * n
    reach_b = [0.12] * n

    def inject_start(role_values, start_idx: int) -> None:
        role_values["crouch"][start_idx : start_idx + 3] = [1.0, 0.96, 0.80]
        role_values["reach"][start_idx : start_idx + 3] = [0.84, 0.74, 0.52]
        role_values["serve"][start_idx : start_idx + 3] = [0.42, 0.90, 0.34]
        role_values["upper"][start_idx : start_idx + 3] = [0.48, 0.88, 0.30]
        role_values["foot"][start_idx : start_idx + 3] = [0.44, 0.74, 0.26]

    inject_start(
        {
            "crouch": crouch_a,
            "reach": reach_a,
            "serve": serve_a,
            "upper": upper_a,
            "foot": foot_a,
        },
        2,
    )
    inject_start(
        {
            "crouch": crouch_b,
            "reach": reach_b,
            "serve": serve_b,
            "upper": upper_b,
            "foot": foot_b,
        },
        10,
    )
    crouch_b[2:5] = [0.86, 0.84, 0.72]
    crouch_a[10:13] = [0.86, 0.84, 0.72]

    motion_a[2:6] = [0.18, 0.42, 0.38, 0.24]
    motion_b[2:6] = [0.10, 0.28, 0.35, 0.22]
    motion_a[10:14] = [0.10, 0.24, 0.34, 0.22]
    motion_b[10:14] = [0.18, 0.44, 0.36, 0.24]

    diagnostics = _player_diagnostics(
        timestamps,
        motion_a=motion_a,
        motion_b=motion_b,
        crouch_a=crouch_a,
        crouch_b=crouch_b,
        serve_a=serve_a,
        serve_b=serve_b,
        upper_a=upper_a,
        upper_b=upper_b,
        foot_a=foot_a,
        foot_b=foot_b,
        reach_a=reach_a,
        reach_b=reach_b,
    )

    segments = _detect_player_sandwich_rallies_from_diagnostics(diagnostics)

    assert len(segments) == 2
    assert segments[0].t_start == 2.0
    assert segments[0].t_end == 5.0
    assert segments[1].t_start == 10.0
    assert segments[1].t_end == 13.0
    assert "player_sandwich" in segments[0].flags


def test_player_sandwich_detector_force_closes_at_next_start():
    timestamps = [float(i) for i in range(12)]
    n = len(timestamps)
    motion_a = [0.03] * n
    motion_b = [0.03] * n
    crouch_a = [0.10] * n
    crouch_b = [0.10] * n
    serve_a = [0.02] * n
    serve_b = [0.02] * n
    upper_a = [0.02] * n
    upper_b = [0.02] * n
    foot_a = [0.02] * n
    foot_b = [0.02] * n
    reach_a = [0.12] * n
    reach_b = [0.12] * n

    crouch_a[2:5] = [1.0, 0.95, 0.82]
    reach_a[2:5] = [0.84, 0.76, 0.58]
    serve_a[2:5] = [0.44, 0.90, 0.36]
    upper_a[2:5] = [0.48, 0.86, 0.32]
    foot_a[2:5] = [0.44, 0.72, 0.28]

    crouch_b[7:10] = [1.0, 0.96, 0.82]
    reach_b[7:10] = [0.84, 0.76, 0.58]
    serve_b[7:10] = [0.44, 0.90, 0.36]
    upper_b[7:10] = [0.48, 0.86, 0.32]
    foot_b[7:10] = [0.44, 0.72, 0.28]
    crouch_b[2:5] = [0.86, 0.82, 0.72]
    crouch_a[7:10] = [0.86, 0.82, 0.72]

    motion_a[2:7] = [0.16, 0.42, 0.38, 0.34, 0.28]
    motion_b[2:7] = [0.10, 0.26, 0.30, 0.26, 0.22]
    motion_a[7:11] = [0.10, 0.24, 0.30, 0.22]
    motion_b[7:11] = [0.16, 0.40, 0.34, 0.24]
    crouch_a[5:7] = [0.55, 0.42]
    crouch_b[5:7] = [0.48, 0.40]
    upper_a[5:7] = [0.34, 0.26]
    upper_b[5:7] = [0.26, 0.22]
    foot_a[5:7] = [0.30, 0.24]
    foot_b[5:7] = [0.26, 0.22]

    diagnostics = _player_diagnostics(
        timestamps,
        motion_a=motion_a,
        motion_b=motion_b,
        crouch_a=crouch_a,
        crouch_b=crouch_b,
        serve_a=serve_a,
        serve_b=serve_b,
        upper_a=upper_a,
        upper_b=upper_b,
        foot_a=foot_a,
        foot_b=foot_b,
        reach_a=reach_a,
        reach_b=reach_b,
    )

    segments = _detect_player_sandwich_rallies_from_diagnostics(diagnostics)

    assert len(segments) == 2
    assert segments[0].t_start == 2.0
    assert segments[0].t_end == 6.0
    assert segments[1].t_start == 7.0


def test_player_sandwich_detector_marks_short_catch_and_walk_as_let():
    timestamps = [i * 0.5 for i in range(12)]
    n = len(timestamps)
    motion_a = [0.02] * n
    motion_b = [0.02] * n
    crouch_a = [0.08] * n
    crouch_b = [0.08] * n
    serve_a = [0.02] * n
    serve_b = [0.02] * n
    upper_a = [0.02] * n
    upper_b = [0.02] * n
    foot_a = [0.02] * n
    foot_b = [0.02] * n
    reach_a = [0.10] * n
    reach_b = [0.10] * n
    approach_a = [0.0] * n
    approach_b = [0.0] * n

    crouch_a[3:6] = [1.0, 0.92, 0.74]
    reach_a[3:6] = [0.86, 0.78, 0.50]
    serve_a[3:6] = [0.42, 0.90, 0.28]
    upper_a[3:6] = [0.46, 0.86, 0.24]
    foot_a[3:6] = [0.40, 0.70, 0.22]
    motion_a[3:6] = [0.14, 0.36, 0.10]
    motion_b[3:6] = [0.08, 0.10, 0.04]
    crouch_b[3:5] = [0.84, 0.78]

    reach_b[4:7] = [0.70, 0.88, 0.46]
    approach_b[4:7] = [0.18, 0.62, 0.30]
    upper_b[4:7] = [0.08, 0.10, 0.04]
    foot_b[4:7] = [0.08, 0.10, 0.04]

    diagnostics = _player_diagnostics(
        timestamps,
        motion_a=motion_a,
        motion_b=motion_b,
        crouch_a=crouch_a,
        crouch_b=crouch_b,
        serve_a=serve_a,
        serve_b=serve_b,
        upper_a=upper_a,
        upper_b=upper_b,
        foot_a=foot_a,
        foot_b=foot_b,
        reach_a=reach_a,
        reach_b=reach_b,
        approach_a=approach_a,
        approach_b=approach_b,
    )

    segments = _detect_player_sandwich_rallies_from_diagnostics(diagnostics)

    assert len(segments) == 1
    assert "rally_label_let" in segments[0].flags
    assert "let_no_score" in segments[0].flags
    assert segments[0].t_end <= 3.0


def test_player_state_machine_marks_short_catch_and_walk_to_net_as_let():
    timestamps = [i * 0.5 for i in range(12)]
    signals = MultiStreamSignals(
        roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
        timestamps=timestamps,
        table_energies=[0.0] * len(timestamps),
        ball_energies=[0.0] * len(timestamps),
        player_a_energies=[0.02, 0.03, 0.03, 0.58, 0.12, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02],
        player_b_energies=[0.02, 0.03, 0.03, 0.10, 0.20, 0.08, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02],
        player_energies=[0.02, 0.03, 0.03, 0.58, 0.20, 0.08, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02],
        fused_energies=[0.0] * len(timestamps),
        effective_fps=2.0,
        player_signal_source="role_tracker",
        ball_signal_source="none",
        player_a_crouch_scores=[0.82, 0.85, 0.88, 0.78, 0.20, 0.08, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04],
        player_b_crouch_scores=[0.80, 0.83, 0.86, 0.80, 0.35, 0.10, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04],
        player_a_serve_scores=[0.02, 0.03, 0.06, 0.92, 0.16, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        player_b_serve_scores=[0.02, 0.03, 0.05, 0.04, 0.12, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        player_a_upper_body_scores=[0.02, 0.03, 0.04, 0.88, 0.18, 0.06, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02],
        player_b_upper_body_scores=[0.02, 0.03, 0.04, 0.08, 0.18, 0.06, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02],
        player_a_footwork_scores=[0.02, 0.03, 0.04, 0.28, 0.08, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        player_b_footwork_scores=[0.02, 0.03, 0.04, 0.08, 0.12, 0.08, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02],
        player_a_reach_scores=[0.0] * len(timestamps),
        player_b_reach_scores=[0.02, 0.02, 0.03, 0.05, 0.88, 0.18, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02],
        player_a_net_approach_scores=[0.0] * len(timestamps),
        player_b_net_approach_scores=[0.02, 0.02, 0.02, 0.04, 0.42, 0.82, 0.18, 0.04, 0.02, 0.02, 0.02, 0.02],
    )

    segments = _detect_player_state_machine_rallies(signals)

    assert len(segments) == 1
    assert "rally_label_let" in segments[0].flags
    assert "let_no_score" in segments[0].flags
    assert segments[0].t_end <= 2.5


def test_detect_multistream_rallies_player_mode_uses_sandwich_detector_for_role_tracker(monkeypatch):
    signals = MultiStreamSignals(
        roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
        timestamps=[0.0, 1.0, 2.0],
        table_energies=[0.0, 0.0, 0.0],
        ball_energies=[0.0, 0.0, 0.0],
        player_a_energies=[0.0, 0.0, 0.0],
        player_b_energies=[0.0, 0.0, 0.0],
        player_energies=[0.0, 0.0, 0.0],
        fused_energies=[0.0, 0.0, 0.0],
        effective_fps=30.0,
        player_signal_source="role_tracker",
        ball_signal_source="none",
        player_a_crouch_scores=[0.5, 0.5, 0.5],
        player_b_crouch_scores=[0.5, 0.5, 0.5],
    )

    def fake_detect(_signals):
        return [RallySegment(t_start=1.0, t_end=2.0, confidence=0.8, flags=["player_sandwich"])]

    monkeypatch.setattr("backend.ai_multistream_rally._detect_player_sandwich_rallies", fake_detect)

    segments = detect_multistream_rallies(signals, mode="player")

    assert len(segments) == 1
    assert segments[0].flags == ["player_sandwich"]


def test_detect_multistream_rallies_ball_mode_requires_real_ball_source():
    signals = MultiStreamSignals(
        roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
        timestamps=[0.0, 1.0],
        table_energies=[0.2, 0.1],
        ball_energies=[0.0, 0.0],
        player_a_energies=[0.0, 0.0],
        player_b_energies=[0.0, 0.0],
        player_energies=[0.0, 0.0],
        fused_energies=[0.2, 0.1],
        effective_fps=30.0,
        player_signal_source="role_tracker",
        ball_signal_source="none",
    )

    try:
        detect_multistream_rallies(signals, mode="ball")
    except ValueError as exc:
        assert "Ball-only mode requires" in str(exc)
    else:
        raise AssertionError("Expected ball-only mode without ball source to fail")


def test_detect_multistream_rallies_player_mode_requires_real_player_source():
    signals = MultiStreamSignals(
        roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
        timestamps=[0.0, 1.0],
        table_energies=[0.2, 0.1],
        ball_energies=[0.0, 0.0],
        player_a_energies=[0.0, 0.0],
        player_b_energies=[0.0, 0.0],
        player_energies=[0.0, 0.0],
        fused_energies=[0.2, 0.1],
        effective_fps=30.0,
        player_signal_source="none",
        ball_signal_source="none",
    )

    try:
        detect_multistream_rallies(signals, mode="player")
    except ValueError as exc:
        assert "Player-only mode requires" in str(exc)
    else:
        raise AssertionError("Expected player-only mode without player source to fail")


def test_build_draft_table_mode_disables_non_table_streams(monkeypatch):
    captured = {}

    def fake_extract_multistream_signals(video_path, table_weights_path, **kwargs):
        captured["player_signal_source"] = kwargs["player_signal_source"]
        captured["ball_signal_source"] = kwargs["ball_signal_source"]
        captured["ball_tracking_profile"] = kwargs["ball_tracking_profile"]
        return MultiStreamSignals(
            roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
            timestamps=[0.0, 1.0, 2.0],
            table_energies=[0.1, 0.7, 0.2],
            ball_energies=[0.0, 0.0, 0.0],
            player_a_energies=[0.0, 0.0, 0.0],
            player_b_energies=[0.0, 0.0, 0.0],
            player_energies=[0.0, 0.0, 0.0],
            fused_energies=[0.1, 0.7, 0.2],
            effective_fps=30.0,
            player_signal_source=kwargs["player_signal_source"],
            ball_signal_source=kwargs["ball_signal_source"],
        )

    def fake_detect_multistream_rallies(signals, *, mode):
        assert mode == "table"
        return [RallySegment(t_start=1.0, t_end=2.0, confidence=0.85, flags=[])]

    monkeypatch.setattr(generate_draft_multistream.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generate_draft_multistream, "extract_multistream_signals", fake_extract_multistream_signals)
    monkeypatch.setattr(generate_draft_multistream, "detect_multistream_rallies", fake_detect_multistream_rallies)

    draft = generate_draft_multistream.build_draft(
        "demo.mp4",
        "weights/yolov8x_table.pt",
        mode="table",
        player_signal_source="role_tracker",
        ball_signal_source="classical",
    )

    assert captured["player_signal_source"] == "none"
    assert captured["ball_signal_source"] == "none"
    assert captured["ball_tracking_profile"] == "support"
    assert len(draft.points) == 1
    assert "table_only" in draft.points[0].flags
    assert "player_signal_none" in draft.points[0].flags
    assert "ball_signal_none" in draft.points[0].flags
    assert draft.analysis_metadata["detector_mode"] == "table"
    assert draft.to_dict()["summary"]["total_rallies"] == 1


def test_build_draft_ball_mode_disables_player_streams(monkeypatch):
    captured = {}

    def fake_extract_multistream_signals(video_path, table_weights_path, **kwargs):
        captured["player_signal_source"] = kwargs["player_signal_source"]
        captured["ball_signal_source"] = kwargs["ball_signal_source"]
        captured["ball_tracking_profile"] = kwargs["ball_tracking_profile"]
        return MultiStreamSignals(
            roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
            timestamps=[0.0, 1.0, 2.0],
            table_energies=[0.1, 0.1, 0.1],
            ball_energies=[0.0, 0.5, 0.0],
            player_a_energies=[0.0, 0.0, 0.0],
            player_b_energies=[0.0, 0.0, 0.0],
            player_energies=[0.0, 0.0, 0.0],
            fused_energies=[0.1, 0.5, 0.1],
            effective_fps=30.0,
            player_signal_source=kwargs["player_signal_source"],
            ball_signal_source=kwargs["ball_signal_source"],
        )

    def fake_detect_multistream_rallies(signals, *, mode):
        assert mode == "ball"
        return [RallySegment(t_start=1.0, t_end=2.0, confidence=0.8, flags=["ball_only"])]

    monkeypatch.setattr(generate_draft_multistream.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generate_draft_multistream, "extract_multistream_signals", fake_extract_multistream_signals)
    monkeypatch.setattr(generate_draft_multistream, "detect_multistream_rallies", fake_detect_multistream_rallies)

    draft = generate_draft_multistream.build_draft(
        "demo.mp4",
        "weights/yolov8x_table.pt",
        mode="ball",
        ball_signal_source="classical",
        player_signal_source="role_tracker",
    )

    assert captured["player_signal_source"] == "none"
    assert captured["ball_signal_source"] == "classical"
    assert captured["ball_tracking_profile"] == "standalone"
    assert len(draft.points) == 1
    assert "ball_only" in draft.points[0].flags
    assert "player_signal_none" in draft.points[0].flags
    assert draft.analysis_metadata["detector_mode"] == "ball"
    assert draft.to_dict()["summary"]["total_rallies"] == 1


def test_build_draft_player_mode_allows_ball_streams_for_endpoint_refinement(monkeypatch):
    captured = {}

    def fake_extract_multistream_signals(video_path, table_weights_path, **kwargs):
        captured["player_signal_source"] = kwargs["player_signal_source"]
        captured["ball_signal_source"] = kwargs["ball_signal_source"]
        captured["ball_tracking_profile"] = kwargs["ball_tracking_profile"]
        return MultiStreamSignals(
            roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
            timestamps=[0.0, 1.0, 2.0],
            table_energies=[0.1, 0.1, 0.1],
            ball_energies=[0.0, 0.0, 0.0],
            player_a_energies=[0.0, 0.4, 0.2],
            player_b_energies=[0.0, 0.3, 0.1],
            player_energies=[0.0, 0.4, 0.2],
            fused_energies=[0.1, 0.4, 0.2],
            effective_fps=30.0,
            player_signal_source=kwargs["player_signal_source"],
            ball_signal_source=kwargs["ball_signal_source"],
        )

    def fake_detect_multistream_rallies(signals, *, mode):
        assert mode == "player"
        return [RallySegment(t_start=0.5, t_end=2.0, confidence=0.78, flags=["player_only"], server_role="B")]

    monkeypatch.setattr(generate_draft_multistream.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generate_draft_multistream, "extract_multistream_signals", fake_extract_multistream_signals)
    monkeypatch.setattr(generate_draft_multistream, "detect_multistream_rallies", fake_detect_multistream_rallies)

    draft = generate_draft_multistream.build_draft(
        "demo.mp4",
        "weights/yolov8x_table.pt",
        mode="player",
        player_signal_source="role_tracker",
        ball_signal_source="classical",
    )

    assert captured["player_signal_source"] == "role_tracker"
    assert captured["ball_signal_source"] == "classical"
    assert captured["ball_tracking_profile"] == "support"
    assert len(draft.points) == 1
    assert "player_only" in draft.points[0].flags
    assert "player_signal_role_tracker" in draft.points[0].flags
    assert "ball_signal_classical" in draft.points[0].flags
    assert draft.points[0].starter_role == "B"
    assert draft.analysis_metadata["detector_mode"] == "player"
    assert draft.to_dict()["summary"]["total_rallies"] == 1


def test_infer_player_serve_mode_prefers_double_for_set4_like_role_sequence():
    starter_roles = [
        "B", "B",
        "A", "A",
        "B", "B",
        "A", "A",
        "B", "B", "B", "B",
        "A", "A",
        "B", "B",
        "A", "A",
        "B", "B", "B",
        "A", "A",
    ]

    assert _infer_player_serve_mode_from_starter_roles(starter_roles) == "double"


def test_infer_forced_let_indices_double_mode_prefers_latest_strong_prefix_point():
    starter_roles = ["B", "B", "B", "B", "A", "A", "B", "B", "B"]
    point_likelihoods = [0.20, 0.74, 0.31, 0.92, 0.55, 0.58, 0.26, 0.81, 0.95]

    let_indices = _infer_forced_let_indices_from_starter_roles(
        starter_roles,
        point_likelihoods,
        serve_mode="double",
    )

    assert let_indices == {0, 1, 6}


def test_repair_double_serve_role_singletons_flips_suspicious_run_edge():
    def make_candidate(timestamp: float, role: str) -> PlayerRallyStartCandidate:
        return PlayerRallyStartCandidate(
            sample_idx=int(timestamp * 10),
            timestamp=timestamp,
            role=role,
            score=0.72,
            prep_score=0.84,
            launch_score=0.40,
            opponent_ready_score=0.58,
            dominance_ratio=1.40,
            episode_start_sample_idx=int(timestamp * 10),
            episode_end_sample_idx=int(timestamp * 10) + 2,
            episode_peak_sample_idx=int(timestamp * 10) + 1,
            episode_peak_score=0.80,
            crouch_score=0.90,
            reach_score=0.60,
            serve_score=0.32,
            upper_body_score=0.30,
            footwork_score=0.28,
            ready_pair_score=0.56,
            pre_ready_mean=0.44,
            pre_live_peak=0.92,
            server_action_score=0.38,
            server_peak_score=0.50,
            server_growth_score=0.10,
            server_peak_delay_sec=0.08,
            receiver_peak_score=0.66,
            live_peak_score=0.62,
        )

    candidates = [
        make_candidate(2.0, "B"),
        make_candidate(10.0, "B"),
        make_candidate(18.0, "B"),
        make_candidate(26.0, "A"),
        make_candidate(34.0, "B"),
        make_candidate(42.0, "B"),
    ]

    repaired = _repair_double_serve_role_singletons(candidates)

    assert [candidate.role for candidate in repaired] == ["B", "B", "A", "A", "B", "B"]


def test_build_draft_excludes_let_segments_from_points(monkeypatch):
    def fake_extract_multistream_signals(video_path, table_weights_path, **kwargs):
        return MultiStreamSignals(
            roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
            timestamps=[0.0, 1.0, 2.0, 3.0],
            table_energies=[0.1, 0.1, 0.1, 0.1],
            ball_energies=[0.0, 0.0, 0.0, 0.0],
            player_a_energies=[0.0, 0.2, 0.1, 0.0],
            player_b_energies=[0.0, 0.2, 0.1, 0.0],
            player_energies=[0.0, 0.2, 0.1, 0.0],
            fused_energies=[0.1, 0.2, 0.1, 0.1],
            effective_fps=30.0,
            player_signal_source=kwargs["player_signal_source"],
            ball_signal_source=kwargs["ball_signal_source"],
        )

    def fake_detect_multistream_rallies(signals, *, mode):
        assert mode == "player"
        return [
            RallySegment(
                t_start=0.5,
                t_end=0.9,
                confidence=0.62,
                flags=["player_sandwich", "rally_label_let", "let_no_score"],
                server_role="B",
            ),
            RallySegment(
                t_start=1.2,
                t_end=2.8,
                confidence=0.85,
                flags=["player_sandwich", "rally_label_point"],
                server_role="B",
            ),
        ]

    monkeypatch.setattr(generate_draft_multistream.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generate_draft_multistream, "extract_multistream_signals", fake_extract_multistream_signals)
    monkeypatch.setattr(generate_draft_multistream, "detect_multistream_rallies", fake_detect_multistream_rallies)

    draft = generate_draft_multistream.build_draft(
        "demo.mp4",
        "weights/yolov8x_table.pt",
        mode="player",
        player_signal_source="role_tracker",
    )

    assert len(draft.points) == 1
    assert draft.points[0].id == "pt_0001"
    assert draft.points[0].starter_role == "B"
    assert draft.points[0].t_start == 1.2
    assert draft.points[0].active_start == 1.2
    assert draft.points[0].active_end == 3.0
    assert draft.points[0].search_upper_bound == 3.0
    assert draft.points[0].preceding_let_count == 1
    assert draft.points[0].preceding_let_starts == [0.5]
    assert draft.points[0].service_attempt_index == 2
    assert draft.points[0].boundary_mode == "video_end_open_tail"
    assert draft.points[0].endpoint_mode in {"detector_end_clamped", "last_exchange_support", "dead_reset_run_start"}
    assert 0.0 <= draft.points[0].endpoint_confidence <= 1.0
    assert draft.analysis_metadata["excluded_let_count"] == 1
    assert len(draft.analysis_metadata["excluded_let_starts"]) == 1
    assert draft.analysis_metadata["excluded_let_starts"][0]["t_start"] == 0.5
    assert draft.analysis_metadata["active_window_mode"] == "accepted_start_to_next_accepted_start"
    assert draft.analysis_metadata["endpoint_refine_mode"] == "roi_plus_ball_bounded_search"
    assert draft.analysis_metadata["unattached_trailing_let_count"] == 0
    assert draft.to_dict()["summary"]["total_rallies"] == 1


def test_build_draft_attaches_multiple_preceding_lets_and_active_windows(monkeypatch):
    def fake_extract_multistream_signals(video_path, table_weights_path, **kwargs):
        return MultiStreamSignals(
            roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            table_energies=[0.1] * 8,
            ball_energies=[0.0] * 8,
            player_a_energies=[0.0] * 8,
            player_b_energies=[0.0] * 8,
            player_energies=[0.0] * 8,
            fused_energies=[0.1] * 8,
            effective_fps=30.0,
            player_signal_source=kwargs["player_signal_source"],
            ball_signal_source=kwargs["ball_signal_source"],
        )

    def fake_detect_multistream_rallies(signals, *, mode):
        assert mode == "player"
        return [
            RallySegment(
                t_start=0.5,
                t_end=0.9,
                confidence=0.60,
                flags=["player_sandwich", "rally_label_let", "let_no_score"],
                server_role="B",
            ),
            RallySegment(
                t_start=1.0,
                t_end=1.3,
                confidence=0.61,
                flags=["player_sandwich", "rally_label_let", "let_no_score"],
                server_role="B",
            ),
            RallySegment(
                t_start=2.0,
                t_end=3.4,
                confidence=0.84,
                flags=["player_sandwich", "rally_label_point"],
                server_role="B",
            ),
            RallySegment(
                t_start=5.0,
                t_end=6.2,
                confidence=0.88,
                flags=["player_sandwich", "rally_label_point"],
                server_role="A",
            ),
        ]

    monkeypatch.setattr(generate_draft_multistream.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generate_draft_multistream, "extract_multistream_signals", fake_extract_multistream_signals)
    monkeypatch.setattr(generate_draft_multistream, "detect_multistream_rallies", fake_detect_multistream_rallies)

    draft = generate_draft_multistream.build_draft(
        "demo.mp4",
        "weights/yolov8x_table.pt",
        mode="player",
        player_signal_source="role_tracker",
    )

    assert len(draft.points) == 2
    assert draft.points[0].t_start == 2.0
    assert draft.points[0].starter_role == "B"
    assert draft.points[0].active_start == 2.0
    assert draft.points[0].active_end == 5.0
    assert draft.points[0].search_upper_bound == 5.0
    assert draft.points[0].preceding_let_count == 2
    assert draft.points[0].preceding_let_starts == [0.5, 1.0]
    assert draft.points[0].service_attempt_index == 3
    assert draft.points[0].boundary_mode == "next_start_exclusive"
    assert draft.points[1].t_start == 5.0
    assert draft.points[1].starter_role == "A"
    assert draft.points[1].active_start == 5.0
    assert draft.points[1].active_end == 7.0
    assert draft.points[1].search_upper_bound == 7.0
    assert draft.points[1].preceding_let_count == 0
    assert draft.points[1].preceding_let_starts == []
    assert draft.points[1].service_attempt_index == 1
    assert draft.points[1].boundary_mode == "video_end_open_tail"
    assert draft.analysis_metadata["excluded_let_count"] == 2
    assert draft.analysis_metadata["unattached_trailing_let_count"] == 0


def test_build_draft_refines_endpoint_before_search_upper_bound(monkeypatch):
    def fake_extract_multistream_signals(video_path, table_weights_path, **kwargs):
        return MultiStreamSignals(
            roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            table_energies=[0.0, 0.9, 0.85, 0.10, 0.05, 0.02, 0.01],
            ball_energies=[0.0, 0.85, 0.80, 0.08, 0.03, 0.01, 0.0],
            player_a_energies=[0.0] * 7,
            player_b_energies=[0.0] * 7,
            player_energies=[0.0] * 7,
            fused_energies=[0.0] * 7,
            effective_fps=30.0,
            player_signal_source=kwargs["player_signal_source"],
            ball_signal_source=kwargs["ball_signal_source"],
        )

    def fake_detect_multistream_rallies(signals, *, mode):
        return [
            RallySegment(
                t_start=1.0,
                t_end=5.2,
                confidence=0.82,
                flags=["player_sandwich", "rally_label_point"],
                server_role="B",
            ),
            RallySegment(
                t_start=6.0,
                t_end=6.5,
                confidence=0.70,
                flags=["player_sandwich", "rally_label_point"],
                server_role="A",
            ),
        ]

    monkeypatch.setattr(generate_draft_multistream.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generate_draft_multistream, "extract_multistream_signals", fake_extract_multistream_signals)
    monkeypatch.setattr(generate_draft_multistream, "detect_multistream_rallies", fake_detect_multistream_rallies)

    draft = generate_draft_multistream.build_draft(
        "demo.mp4",
        "weights/yolov8x_table.pt",
        mode="player",
        player_signal_source="role_tracker",
        ball_signal_source="classical",
    )

    assert len(draft.points) == 2
    assert draft.points[0].search_upper_bound == 6.0
    assert draft.points[0].t_end < 6.0
    assert draft.points[0].endpoint_mode in {"last_exchange_support", "dead_reset_run_start"}


def test_build_draft_endpoint_prefers_dead_reset_run_over_lingering_table_motion(monkeypatch):
    def fake_extract_multistream_signals(video_path, table_weights_path, **kwargs):
        return MultiStreamSignals(
            roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            table_energies=[0.0, 0.92, 0.88, 0.36, 0.34, 0.30, 0.27, 0.22, 0.18],
            ball_energies=[0.0, 0.86, 0.82, 0.05, 0.03, 0.02, 0.01, 0.0, 0.0],
            player_a_energies=[0.05, 0.54, 0.50, 0.28, 0.30, 0.26, 0.22, 0.18, 0.12],
            player_b_energies=[0.05, 0.48, 0.46, 0.24, 0.26, 0.23, 0.19, 0.16, 0.10],
            player_energies=[0.05, 0.54, 0.50, 0.28, 0.30, 0.26, 0.22, 0.18, 0.12],
            fused_energies=[0.0] * 9,
            effective_fps=30.0,
            player_signal_source=kwargs["player_signal_source"],
            ball_signal_source=kwargs["ball_signal_source"],
            player_a_crouch_scores=[0.82, 0.84, 0.78, 0.14, 0.12, 0.10, 0.08, 0.08, 0.08],
            player_b_crouch_scores=[0.80, 0.82, 0.76, 0.16, 0.13, 0.10, 0.08, 0.08, 0.08],
            player_a_serve_scores=[0.04, 0.12, 0.08, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            player_b_serve_scores=[0.04, 0.10, 0.06, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            player_a_upper_body_scores=[0.10, 0.62, 0.56, 0.10, 0.08, 0.06, 0.05, 0.05, 0.05],
            player_b_upper_body_scores=[0.10, 0.56, 0.52, 0.10, 0.08, 0.06, 0.05, 0.05, 0.05],
            player_a_footwork_scores=[0.10, 0.58, 0.52, 0.09, 0.08, 0.06, 0.05, 0.05, 0.05],
            player_b_footwork_scores=[0.10, 0.54, 0.50, 0.09, 0.08, 0.06, 0.05, 0.05, 0.05],
        )

    def fake_detect_multistream_rallies(signals, *, mode):
        return [
            RallySegment(
                t_start=1.0,
                t_end=7.2,
                confidence=0.84,
                flags=["player_sandwich", "rally_label_point"],
                server_role="B",
            ),
            RallySegment(
                t_start=8.0,
                t_end=8.5,
                confidence=0.70,
                flags=["player_sandwich", "rally_label_point"],
                server_role="A",
            ),
        ]

    monkeypatch.setattr(generate_draft_multistream.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generate_draft_multistream, "extract_multistream_signals", fake_extract_multistream_signals)
    monkeypatch.setattr(generate_draft_multistream, "detect_multistream_rallies", fake_detect_multistream_rallies)

    draft = generate_draft_multistream.build_draft(
        "demo.mp4",
        "weights/yolov8x_table.pt",
        mode="player",
        player_signal_source="role_tracker",
        ball_signal_source="classical",
    )

    assert len(draft.points) == 2
    assert draft.points[0].search_upper_bound == 8.0
    assert draft.points[0].t_end <= 5.0
    assert draft.points[0].endpoint_mode == "dead_reset_run_start"


def test_build_draft_endpoint_scores_dead_runs_and_skips_false_reset_resume(monkeypatch):
    def fake_extract_multistream_signals(video_path, table_weights_path, **kwargs):
        return MultiStreamSignals(
            roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            table_energies=[0.0, 0.82, 0.78, 0.12, 0.70, 0.72, 0.14, 0.08, 0.04, 0.02],
            ball_energies=[0.0, 0.74, 0.68, 0.05, 0.62, 0.64, 0.06, 0.02, 0.01, 0.0],
            player_a_energies=[0.05, 0.48, 0.44, 0.12, 0.40, 0.42, 0.16, 0.12, 0.10, 0.08],
            player_b_energies=[0.05, 0.44, 0.42, 0.12, 0.38, 0.40, 0.15, 0.12, 0.10, 0.08],
            player_energies=[0.05, 0.48, 0.44, 0.12, 0.40, 0.42, 0.16, 0.12, 0.10, 0.08],
            fused_energies=[0.0] * 10,
            effective_fps=30.0,
            player_signal_source=kwargs["player_signal_source"],
            ball_signal_source=kwargs["ball_signal_source"],
            player_a_crouch_scores=[0.84, 0.86, 0.80, 0.14, 0.70, 0.68, 0.10, 0.08, 0.08, 0.08],
            player_b_crouch_scores=[0.82, 0.84, 0.78, 0.14, 0.68, 0.66, 0.10, 0.08, 0.08, 0.08],
            player_a_serve_scores=[0.06, 0.12, 0.08, 0.02, 0.04, 0.04, 0.02, 0.02, 0.02, 0.02],
            player_b_serve_scores=[0.06, 0.10, 0.06, 0.02, 0.04, 0.04, 0.02, 0.02, 0.02, 0.02],
            player_a_upper_body_scores=[0.10, 0.54, 0.50, 0.08, 0.46, 0.48, 0.08, 0.05, 0.05, 0.05],
            player_b_upper_body_scores=[0.10, 0.50, 0.46, 0.08, 0.44, 0.46, 0.08, 0.05, 0.05, 0.05],
            player_a_footwork_scores=[0.10, 0.50, 0.46, 0.08, 0.42, 0.44, 0.08, 0.05, 0.05, 0.05],
            player_b_footwork_scores=[0.10, 0.48, 0.44, 0.08, 0.40, 0.42, 0.08, 0.05, 0.05, 0.05],
        )

    def fake_detect_multistream_rallies(signals, *, mode):
        return [
            RallySegment(
                t_start=1.0,
                t_end=8.2,
                confidence=0.82,
                flags=["player_sandwich", "rally_label_point"],
                server_role="A",
            ),
            RallySegment(
                t_start=9.0,
                t_end=9.5,
                confidence=0.70,
                flags=["player_sandwich", "rally_label_point"],
                server_role="B",
            ),
        ]

    monkeypatch.setattr(generate_draft_multistream.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generate_draft_multistream, "extract_multistream_signals", fake_extract_multistream_signals)
    monkeypatch.setattr(generate_draft_multistream, "detect_multistream_rallies", fake_detect_multistream_rallies)

    draft = generate_draft_multistream.build_draft(
        "demo.mp4",
        "weights/yolov8x_table.pt",
        mode="player",
        player_signal_source="role_tracker",
        ball_signal_source="classical",
    )

    assert len(draft.points) == 2
    assert draft.points[0].search_upper_bound == 9.0
    assert draft.points[0].t_end >= 5.0
    assert draft.points[0].endpoint_mode in {"dead_reset_run_start", "last_exchange_support"}


def test_build_draft_endpoint_ignores_one_sided_pickup_motion(monkeypatch):
    def fake_extract_multistream_signals(video_path, table_weights_path, **kwargs):
        return MultiStreamSignals(
            roi=TableROI(x=10, y=20, w=100, h=50, confidence=0.9),
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            table_energies=[0.0, 0.82, 0.76, 0.34, 0.30, 0.22, 0.16, 0.10, 0.06],
            ball_energies=[0.0, 0.76, 0.70, 0.08, 0.05, 0.03, 0.02, 0.01, 0.0],
            player_a_energies=[0.05, 0.46, 0.44, 0.54, 0.58, 0.44, 0.28, 0.16, 0.08],
            player_b_energies=[0.05, 0.44, 0.42, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03],
            player_energies=[0.05, 0.46, 0.44, 0.54, 0.58, 0.44, 0.28, 0.16, 0.08],
            fused_energies=[0.0] * 9,
            effective_fps=30.0,
            player_signal_source=kwargs["player_signal_source"],
            ball_signal_source=kwargs["ball_signal_source"],
            player_a_crouch_scores=[0.82, 0.84, 0.80, 0.12, 0.10, 0.08, 0.08, 0.08, 0.08],
            player_b_crouch_scores=[0.80, 0.82, 0.78, 0.12, 0.10, 0.08, 0.08, 0.08, 0.08],
            player_a_serve_scores=[0.04, 0.10, 0.06, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            player_b_serve_scores=[0.04, 0.10, 0.06, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            player_a_upper_body_scores=[0.10, 0.52, 0.48, 0.12, 0.10, 0.08, 0.06, 0.05, 0.05],
            player_b_upper_body_scores=[0.10, 0.50, 0.46, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05],
            player_a_footwork_scores=[0.10, 0.50, 0.46, 0.14, 0.12, 0.08, 0.06, 0.05, 0.05],
            player_b_footwork_scores=[0.10, 0.48, 0.44, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05],
        )

    def fake_detect_multistream_rallies(signals, *, mode):
        return [
            RallySegment(
                t_start=1.0,
                t_end=7.2,
                confidence=0.84,
                flags=["player_sandwich", "rally_label_point"],
                server_role="B",
            ),
            RallySegment(
                t_start=8.0,
                t_end=8.5,
                confidence=0.70,
                flags=["player_sandwich", "rally_label_point"],
                server_role="A",
            ),
        ]

    monkeypatch.setattr(generate_draft_multistream.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generate_draft_multistream, "extract_multistream_signals", fake_extract_multistream_signals)
    monkeypatch.setattr(generate_draft_multistream, "detect_multistream_rallies", fake_detect_multistream_rallies)

    draft = generate_draft_multistream.build_draft(
        "demo.mp4",
        "weights/yolov8x_table.pt",
        mode="player",
        player_signal_source="role_tracker",
        ball_signal_source="classical",
    )

    assert len(draft.points) == 2
    assert draft.points[0].search_upper_bound == 8.0
    assert draft.points[0].t_end <= 5.0
    assert draft.points[0].endpoint_mode == "dead_reset_run_start"


def test_refine_endpoint_reopens_open_tail_when_late_live_tail_is_strong():
    timestamps = np.arange(0.0, 2.5, 0.1, dtype=np.float32)
    table = np.full_like(timestamps, 0.08)
    ball = np.full_like(timestamps, 0.04)
    live = np.full_like(timestamps, 0.10)
    interaction = np.full_like(timestamps, 0.04)
    one_sided = np.full_like(timestamps, 0.08)
    reset = np.full_like(timestamps, 0.62)
    shared = np.full_like(timestamps, 0.12)
    terminal = np.full_like(timestamps, 0.18)

    # Early real rally.
    early = (timestamps >= 0.2) & (timestamps <= 0.8)
    table[early] = 0.70
    ball[early] = 0.78
    live[early] = 0.62
    interaction[early] = 0.28
    one_sided[early] = 0.18
    reset[early] = 0.42
    shared[early] = 0.68
    terminal[early] = 0.16

    # False dead pocket that should not terminate the last rally.
    dead = (timestamps >= 0.9) & (timestamps <= 1.1)
    table[dead] = 0.10
    ball[dead] = 0.06
    live[dead] = 0.22
    interaction[dead] = 0.08
    one_sided[dead] = 0.14
    reset[dead] = 0.66
    shared[dead] = 0.16
    terminal[dead] = 0.24

    # Strong late live tail near the open end of the clip.
    late = (timestamps >= 1.4) & (timestamps <= 2.4)
    table[late] = 0.26
    ball[late] = 0.74
    live[late] = 0.58
    interaction[late] = 0.34
    one_sided[late] = 0.16
    reset[late] = 0.46
    shared[late] = 0.60
    terminal[late] = 0.18

    refined_end, endpoint_mode, _conf = generate_draft_multistream._refine_endpoint_from_signals(
        timestamps,
        table,
        ball,
        live,
        interaction,
        one_sided,
        reset,
        shared,
        terminal,
        t_start=0.2,
        detector_end=1.0,
        search_upper_bound=2.4,
        is_open_tail=True,
    )

    assert refined_end >= 2.0
    assert endpoint_mode == "last_exchange_support"
