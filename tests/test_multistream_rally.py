import numpy as np

from backend.ai_multistream_rally import _build_role_energy_series, _refine_table_segments_with_role_support
from backend.ai_rally_segmentation import RallySegment
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
