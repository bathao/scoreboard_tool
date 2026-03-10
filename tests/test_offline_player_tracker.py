import numpy as np

from backend.ai_table_roi import TableROI
from backend.offline_player_tracker import OfflinePlayerTracker


def _frame(width: int = 640, height: int = 360) -> np.ndarray:
    return np.full((height, width, 3), 127, dtype=np.uint8)


def _kpts(cx: float, cy: float) -> np.ndarray:
    pts = np.zeros((17, 2), dtype=np.float32)
    for idx in range(17):
        pts[idx] = (cx + (idx % 3) * 2.0, cy + idx * 1.5)
    return pts


def _build_detection(
    tracker: OfflinePlayerTracker,
    *,
    frame_idx: int,
    box: tuple[int, int, int, int],
    conf: float = 0.9,
) -> list:
    boxes = np.asarray([box], dtype=np.float32)
    centers = _kpts((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    kpts = np.asarray([centers], dtype=np.float32)
    confs = np.asarray([conf], dtype=np.float32)
    return tracker.build_detections(
        _frame(),
        frame_idx=frame_idx,
        boxes_xyxy=boxes,
        keypoints_xy=kpts,
        confidences=confs,
    )


def test_links_same_person_into_single_tracklet():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360, max_link_gap_frames=5)

    tracker.add_frame_detections(_build_detection(tracker, frame_idx=0, box=(120, 90, 180, 250)))
    tracker.add_frame_detections(_build_detection(tracker, frame_idx=1, box=(124, 92, 184, 252)))
    tracker.add_frame_detections(_build_detection(tracker, frame_idx=2, box=(128, 94, 188, 254)))

    result = tracker.finish()
    assert len(result.tracklets) == 1
    assert result.tracklets[0].duration_frames == 3


def test_assigns_roles_for_disconnected_tracklets_after_gap():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360, max_link_gap_frames=3)
    frame = _frame()

    early_boxes = np.asarray(
        [
            (140, 88, 192, 260),
            (348, 108, 432, 252),
        ],
        dtype=np.float32,
    )
    early_kpts = np.asarray(
        [
            _kpts(166, 166),
            _kpts(390, 180),
        ],
        dtype=np.float32,
    )
    early_confs = np.asarray([0.95, 0.96], dtype=np.float32)
    tracker.add_frame_detections(
        tracker.build_detections(
            frame,
            frame_idx=0,
            boxes_xyxy=early_boxes,
            keypoints_xy=early_kpts,
            confidences=early_confs,
        )
    )
    tracker.add_frame_detections(
        tracker.build_detections(
            frame,
            frame_idx=1,
            boxes_xyxy=early_boxes,
            keypoints_xy=early_kpts,
            confidences=early_confs,
        )
    )

    late_boxes = np.asarray(
        [
            (350, 88, 402, 260),
            (140, 108, 224, 252),
        ],
        dtype=np.float32,
    )
    late_kpts = np.asarray(
        [
            _kpts(166, 166),
            _kpts(390, 180),
        ],
        dtype=np.float32,
    )
    late_confs = np.asarray([0.94, 0.93], dtype=np.float32)
    tracker.add_frame_detections(
        tracker.build_detections(
            frame,
            frame_idx=8,
            boxes_xyxy=late_boxes,
            keypoints_xy=late_kpts,
            confidences=late_confs,
        )
    )
    tracker.add_frame_detections(
        tracker.build_detections(
            frame,
            frame_idx=9,
            boxes_xyxy=late_boxes,
            keypoints_xy=late_kpts,
            confidences=late_confs,
        )
    )

    result = tracker.finish()
    assigned = [t for t in result.tracklets if t.assigned_role is not None]
    roles = {t.assigned_role for t in assigned}
    assert roles == {"A", "B"}
    role_a = next(t for t in assigned if t.assigned_role == "A")
    role_b = next(t for t in assigned if t.assigned_role == "B")
    assert role_a.mean_center_x != role_b.mean_center_x
    assert 8 in result.role_frames
    assert set(result.role_frames[8].keys()) == {"A", "B"}


def test_rejects_short_far_spectator_tracklet():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360, max_link_gap_frames=4)
    frame = _frame()

    boxes = np.asarray(
        [
            (150, 95, 210, 255),
            (360, 100, 420, 260),
            (20, 20, 70, 150),
        ],
        dtype=np.float32,
    )
    kpts = np.asarray(
        [
            _kpts(180, 170),
            _kpts(390, 175),
            _kpts(45, 90),
        ],
        dtype=np.float32,
    )
    confs = np.asarray([0.95, 0.95, 0.90], dtype=np.float32)

    for frame_idx in range(3):
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=boxes,
                keypoints_xy=kpts,
                confidences=confs,
            )
        )

    result = tracker.finish()
    assigned = [t for t in result.tracklets if t.assigned_role is not None]
    assert {t.assigned_role for t in assigned} == {"A", "B"}
    assert all(t.mean_center_x >= 100 for t in assigned)


def test_fills_short_internal_gap_inside_assigned_tracklet():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360, max_link_gap_frames=6, max_interpolate_gap_frames=6)
    frame = _frame()

    boxes_early = np.asarray(
        [
            (150, 95, 210, 255),
            (360, 100, 420, 260),
        ],
        dtype=np.float32,
    )
    kpts_early = np.asarray(
        [
            _kpts(180, 170),
            _kpts(390, 175),
        ],
        dtype=np.float32,
    )
    confs = np.asarray([0.95, 0.95], dtype=np.float32)

    tracker.add_frame_detections(
        tracker.build_detections(
            frame,
            frame_idx=0,
            boxes_xyxy=boxes_early,
            keypoints_xy=kpts_early,
            confidences=confs,
        )
    )
    tracker.add_frame_detections(
        tracker.build_detections(
            frame,
            frame_idx=1,
            boxes_xyxy=boxes_early,
            keypoints_xy=kpts_early,
            confidences=confs,
        )
    )

    boxes_late = np.asarray(
        [
            (156, 98, 216, 258),
            (365, 102, 425, 262),
        ],
        dtype=np.float32,
    )
    kpts_late = np.asarray(
        [
            _kpts(186, 174),
            _kpts(395, 178),
        ],
        dtype=np.float32,
    )
    tracker.add_frame_detections(
        tracker.build_detections(
            frame,
            frame_idx=4,
            boxes_xyxy=boxes_late,
            keypoints_xy=kpts_late,
            confidences=confs,
        )
    )
    tracker.add_frame_detections(
        tracker.build_detections(
            frame,
            frame_idx=5,
            boxes_xyxy=boxes_late,
            keypoints_xy=kpts_late,
            confidences=confs,
        )
    )

    result = tracker.finish()
    assert 2 in result.role_frames
    assert 3 in result.role_frames
    assert "B" in result.role_frames[2]
    assert "B" in result.role_frames[3]


def test_build_detections_accepts_bbox_only_without_pose():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360)
    frame = _frame()
    boxes = np.asarray([(150, 95, 210, 255)], dtype=np.float32)
    confs = np.asarray([0.95], dtype=np.float32)

    detections = tracker.build_detections(
        frame,
        frame_idx=0,
        boxes_xyxy=boxes,
        keypoints_xy=None,
        confidences=confs,
    )

    assert len(detections) == 1
    assert detections[0].in_player_zone
    assert detections[0].body_signature.shape == (6,)


def test_spectator_outside_main_player_zone_is_not_started():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360)
    frame = _frame()
    boxes = np.asarray([(20, 20, 70, 150)], dtype=np.float32)
    confs = np.asarray([0.90], dtype=np.float32)

    detections = tracker.build_detections(
        frame,
        frame_idx=0,
        boxes_xyxy=boxes,
        keypoints_xy=None,
        confidences=confs,
    )
    tracker.add_frame_detections(detections)
    result = tracker.finish()

    assert len(result.tracklets) == 0


def test_keeps_tracklet_on_consistent_player_instead_of_jumping_to_stranger():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360, max_link_gap_frames=3)
    frame = _frame()

    early_boxes = np.asarray(
        [
            (150, 90, 220, 300),
            (360, 100, 420, 260),
        ],
        dtype=np.float32,
    )
    early_kpts = np.asarray(
        [
            _kpts(185, 195),
            _kpts(390, 180),
        ],
        dtype=np.float32,
    )
    confs = np.asarray([0.95, 0.95], dtype=np.float32)

    for frame_idx in range(3):
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=early_boxes,
                keypoints_xy=early_kpts,
                confidences=confs,
            )
        )

    ambiguous_boxes = np.asarray(
        [
            (154, 92, 224, 302),
            (0, 120, 70, 230),
            (360, 100, 420, 260),
        ],
        dtype=np.float32,
    )
    ambiguous_kpts = np.asarray(
        [
            _kpts(189, 197),
            _kpts(35, 175),
            _kpts(390, 180),
        ],
        dtype=np.float32,
    )
    tracker.add_frame_detections(
        tracker.build_detections(
            frame,
            frame_idx=3,
            boxes_xyxy=ambiguous_boxes,
            keypoints_xy=ambiguous_kpts,
            confidences=np.asarray([0.94, 0.96, 0.95], dtype=np.float32),
        )
    )

    result = tracker.finish()
    long_tracklets = [t for t in result.tracklets if t.duration_frames >= 4]
    assert len(long_tracklets) == 2
    left_tracklet = min(long_tracklets, key=lambda t: t.mean_center_x)
    assert left_tracklet.last_center[0] > 150.0


def test_marks_short_gap_as_occluded_when_role_returns():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360, max_link_gap_frames=2, max_role_occlusion_gap_frames=12)
    frame = _frame()

    early_boxes = np.asarray(
        [
            (150, 95, 210, 255),
            (360, 100, 420, 260),
        ],
        dtype=np.float32,
    )
    early_kpts = np.asarray(
        [
            _kpts(180, 170),
            _kpts(390, 175),
        ],
        dtype=np.float32,
    )
    confs = np.asarray([0.95, 0.95], dtype=np.float32)

    for frame_idx in range(2):
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=early_boxes,
                keypoints_xy=early_kpts,
                confidences=confs,
            )
        )

    for frame_idx in range(5, 7):
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=early_boxes,
                keypoints_xy=early_kpts,
                confidences=confs,
            )
        )

    result = tracker.finish()
    assert result.role_state_frames[2]["A"] == "occluded"
    assert result.role_state_frames[3]["A"] == "occluded"
    assert result.role_state_frames[4]["A"] == "occluded"
    assert result.role_state_frames[5]["A"] == "visible"


def test_true_leave_does_not_create_fake_occlusion_timeline():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360, max_link_gap_frames=2, max_role_occlusion_gap_frames=12)
    frame = _frame()

    both_boxes = np.asarray(
        [
            (150, 95, 210, 255),
            (360, 100, 420, 260),
        ],
        dtype=np.float32,
    )
    both_kpts = np.asarray(
        [
            _kpts(180, 170),
            _kpts(390, 175),
        ],
        dtype=np.float32,
    )
    for frame_idx in range(2):
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=both_boxes,
                keypoints_xy=both_kpts,
                confidences=np.asarray([0.95, 0.95], dtype=np.float32),
            )
        )

    b_only_boxes = np.asarray([(360, 100, 420, 260)], dtype=np.float32)
    b_only_kpts = np.asarray([_kpts(390, 175)], dtype=np.float32)
    for frame_idx in range(5, 7):
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=b_only_boxes,
                keypoints_xy=b_only_kpts,
                confidences=np.asarray([0.95], dtype=np.float32),
            )
        )

    result = tracker.finish()
    for frame_idx in range(2, 7):
        assert result.role_state_frames.get(frame_idx, {}).get("A") != "occluded"


def test_role_frames_do_not_drop_valid_player_after_large_box_growth():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360, max_link_gap_frames=2)
    frame = _frame()

    for frame_idx in range(3):
        boxes = np.asarray(
            [
                (140, 100, 210, 250),
                (360, 100, 430, 255),
            ],
            dtype=np.float32,
        )
        kpts = np.asarray(
            [
                _kpts(175, 175),
                _kpts(395, 178),
            ],
            dtype=np.float32,
        )
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=boxes,
                keypoints_xy=kpts,
                confidences=np.asarray([0.95, 0.95], dtype=np.float32),
            )
        )

    for frame_idx in range(3, 8):
        boxes = np.asarray(
            [
                (150, 80, 300, 340),
                (360, 100, 430, 255),
            ],
            dtype=np.float32,
        )
        kpts = np.asarray(
            [
                _kpts(225, 210),
                _kpts(395, 178),
            ],
            dtype=np.float32,
        )
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=boxes,
                keypoints_xy=kpts,
                confidences=np.asarray([0.93, 0.95], dtype=np.float32),
            )
        )

    result = tracker.finish()
    for frame_idx in range(3, 8):
        assert "A" in result.role_frames[frame_idx]


def test_compact_overlap_representation_can_continue_same_role():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360, max_link_gap_frames=2)
    frame = _frame()

    for frame_idx in range(4):
        boxes = np.asarray(
            [
                (120, 95, 215, 320),
                (360, 100, 430, 255),
            ],
            dtype=np.float32,
        )
        kpts = np.asarray(
            [
                _kpts(168, 207),
                _kpts(395, 178),
            ],
            dtype=np.float32,
        )
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=boxes,
                keypoints_xy=kpts,
                confidences=np.asarray([0.95, 0.95], dtype=np.float32),
            )
        )

    for frame_idx in range(4, 6):
        boxes = np.asarray(
            [
                (124, 96, 220, 322),
                (132, 120, 208, 252),
                (360, 100, 430, 255),
            ],
            dtype=np.float32,
        )
        kpts = np.asarray(
            [
                _kpts(172, 209),
                _kpts(170, 186),
                _kpts(395, 178),
            ],
            dtype=np.float32,
        )
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=boxes,
                keypoints_xy=kpts,
                confidences=np.asarray([0.94, 0.91, 0.95], dtype=np.float32),
            )
        )

    for frame_idx in range(6, 10):
        boxes = np.asarray(
            [
                (132, 120, 208, 252),
                (360, 100, 430, 255),
            ],
            dtype=np.float32,
        )
        kpts = np.asarray(
            [
                _kpts(170, 186),
                _kpts(395, 178),
            ],
            dtype=np.float32,
        )
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=boxes,
                keypoints_xy=kpts,
                confidences=np.asarray([0.91, 0.95], dtype=np.float32),
            )
        )

    result = tracker.finish()
    assert "A" in result.role_frames[8]
    assert result.role_frames[8]["A"].center[0] < 240.0


def test_seed_pair_prefers_near_player_over_far_spectator_from_start():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360, max_link_gap_frames=2)
    frame = _frame()

    for frame_idx in range(8):
        boxes = np.asarray(
            [
                (100, 126, 200, 326),
                (128, 40, 188, 188),
                (392, 58, 458, 208),
            ],
            dtype=np.float32,
        )
        kpts = np.asarray(
            [
                _kpts(150, 226),
                _kpts(158, 114),
                _kpts(425, 133),
            ],
            dtype=np.float32,
        )
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=boxes,
                keypoints_xy=kpts,
                confidences=np.asarray([0.84, 0.96, 0.95], dtype=np.float32),
            )
        )

    result = tracker.finish()
    assert tracker._role_profiles["A"].preferred_zone == "near"
    assert tracker._role_profiles["B"].preferred_zone == "far"
    assert "A" in result.role_frames[0]
    assert "B" in result.role_frames[0]
    assert result.role_frames[0]["A"].center[1] > 180.0
    assert result.role_frames[0]["B"].center[1] < 180.0


def test_deferred_seeding_waits_for_near_player_and_keeps_early_a_missing():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360, max_link_gap_frames=2)
    frame = _frame()

    for frame_idx in range(4):
        boxes = np.asarray(
            [
                (128, 40, 188, 188),
                (392, 58, 458, 208),
            ],
            dtype=np.float32,
        )
        kpts = np.asarray(
            [
                _kpts(158, 114),
                _kpts(425, 133),
            ],
            dtype=np.float32,
        )
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=boxes,
                keypoints_xy=kpts,
                confidences=np.asarray([0.96, 0.95], dtype=np.float32),
            )
        )

    for frame_idx in range(4, 10):
        boxes = np.asarray(
            [
                (100, 126, 200, 326),
                (128, 40, 188, 188),
                (392, 58, 458, 208),
            ],
            dtype=np.float32,
        )
        kpts = np.asarray(
            [
                _kpts(150, 226),
                _kpts(158, 114),
                _kpts(425, 133),
            ],
            dtype=np.float32,
        )
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=boxes,
                keypoints_xy=kpts,
                confidences=np.asarray([0.84, 0.96, 0.95], dtype=np.float32),
            )
        )

    result = tracker.finish()
    assert tracker._role_profiles["A"].preferred_zone == "near"
    assert tracker._role_profiles["B"].preferred_zone == "far"
    for frame_idx in range(4):
        assert "A" not in result.role_frames.get(frame_idx, {})
        assert "B" in result.role_frames.get(frame_idx, {})
    for frame_idx in range(4, 10):
        assert "A" in result.role_frames.get(frame_idx, {})
        assert result.role_frames[frame_idx]["A"].center[1] > 180.0
        assert "B" in result.role_frames.get(frame_idx, {})


def test_true_leave_prefers_missing_over_neighboring_far_tracklet_and_reacquires_on_return():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360, max_link_gap_frames=2, max_role_occlusion_gap_frames=8)
    frame = _frame()

    early_a_boxes = [
        (120, 120, 220, 330),
        (78, 120, 178, 330),
        (40, 120, 140, 330),
    ]
    for frame_idx, a_box in enumerate(early_a_boxes):
        both_boxes = np.asarray(
            [
                a_box,
                (390, 55, 455, 205),
            ],
            dtype=np.float32,
        )
        both_kpts = np.asarray(
            [
                _kpts((a_box[0] + a_box[2]) / 2.0, (a_box[1] + a_box[3]) / 2.0),
                _kpts(422, 130),
            ],
            dtype=np.float32,
        )
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=both_boxes,
                keypoints_xy=both_kpts,
                confidences=np.asarray([0.95, 0.95], dtype=np.float32),
            )
        )

    spectator_boxes = np.asarray(
        [
            (88, 38, 148, 186),
            (392, 58, 458, 208),
        ],
        dtype=np.float32,
    )
    spectator_kpts = np.asarray(
        [
            _kpts(118, 112),
            _kpts(425, 133),
        ],
        dtype=np.float32,
    )
    for frame_idx in range(3, 7):
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=spectator_boxes,
                keypoints_xy=spectator_kpts,
                confidences=np.asarray([0.94, 0.95], dtype=np.float32),
            )
        )

    return_a_boxes = [
        (56, 122, 156, 332),
        (92, 122, 192, 332),
        (128, 122, 228, 332),
    ]
    for offset, a_box in enumerate(return_a_boxes, start=7):
        return_boxes = np.asarray(
            [
                a_box,
                (92, 42, 152, 188),
                (394, 62, 460, 210),
            ],
            dtype=np.float32,
        )
        return_kpts = np.asarray(
            [
                _kpts((a_box[0] + a_box[2]) / 2.0, (a_box[1] + a_box[3]) / 2.0),
                _kpts(122, 115),
                _kpts(427, 136),
            ],
            dtype=np.float32,
        )
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=offset,
                boxes_xyxy=return_boxes,
                keypoints_xy=return_kpts,
                confidences=np.asarray([0.95, 0.94, 0.95], dtype=np.float32),
            )
        )

    result = tracker.finish()
    for frame_idx in range(3, 7):
        assert "A" not in result.role_frames.get(frame_idx, {})
        assert result.role_state_frames.get(frame_idx, {}).get("A") != "occluded"
        assert "B" in result.role_frames.get(frame_idx, {})
    for frame_idx in range(8, 10):
        assert "A" in result.role_frames.get(frame_idx, {})
        assert result.role_frames[frame_idx]["A"].center[1] > 180.0
        assert "B" in result.role_frames.get(frame_idx, {})


def test_far_role_does_not_drop_when_role_specific_ownership_is_enabled():
    roi = TableROI(220, 110, 200, 120, 1.0)
    tracker = OfflinePlayerTracker(roi, frame_w=640, frame_h=360, max_link_gap_frames=2)
    frame = _frame()

    for frame_idx in range(3):
        boxes = np.asarray(
            [
                (122, 118, 222, 328),
                (392, 52, 454, 196),
            ],
            dtype=np.float32,
        )
        kpts = np.asarray(
            [
                _kpts(172, 223),
                _kpts(423, 124),
            ],
            dtype=np.float32,
        )
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=boxes,
                keypoints_xy=kpts,
                confidences=np.asarray([0.95, 0.95], dtype=np.float32),
            )
        )

    for frame_idx in range(3, 8):
        boxes = np.asarray(
            [
                (126, 120, 226, 330),
                (388, 72, 448, 246),
            ],
            dtype=np.float32,
        )
        kpts = np.asarray(
            [
                _kpts(176, 225),
                _kpts(418, 159),
            ],
            dtype=np.float32,
        )
        tracker.add_frame_detections(
            tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=boxes,
                keypoints_xy=kpts,
                confidences=np.asarray([0.95, 0.92], dtype=np.float32),
            )
        )

    result = tracker.finish()
    for frame_idx in range(3, 8):
        assert "B" in result.role_frames.get(frame_idx, {})
