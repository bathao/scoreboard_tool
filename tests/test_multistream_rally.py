import numpy as np

import scripts.generate_draft_multistream as generate_draft_multistream
from backend.ai_multistream_rally import MultiStreamSignals, detect_multistream_rallies
from backend.ai_multistream_rally import (
    _build_role_energy_series,
    _merge_ball_split_pair_artifacts,
    _merge_segments_with_ball_support,
    _refine_table_segments_with_role_support,
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


def test_build_draft_player_mode_disables_ball_streams(monkeypatch):
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
        return [RallySegment(t_start=0.5, t_end=2.0, confidence=0.78, flags=["player_only"])]

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
    assert captured["ball_signal_source"] == "none"
    assert captured["ball_tracking_profile"] == "support"
    assert len(draft.points) == 1
    assert "player_only" in draft.points[0].flags
    assert "player_signal_role_tracker" in draft.points[0].flags
    assert "ball_signal_none" in draft.points[0].flags
    assert draft.analysis_metadata["detector_mode"] == "player"
    assert draft.to_dict()["summary"]["total_rallies"] == 1
