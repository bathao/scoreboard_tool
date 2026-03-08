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


def test_assigns_role_by_global_identity_not_current_side():
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
    assert result.role_frames[8]["A"].center[0] > result.role_frames[8]["B"].center[0]


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
