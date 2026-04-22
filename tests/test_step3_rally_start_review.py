from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.rally_timeline_contract import RallyTimeline, RallyTimelinePoint
from backend.step3_rally_start_review import (
    Step3LogicAuditConfig,
    Step3SideIdentificationConfig,
    Step3PlayerContext,
    _side_identification_scan_plan,
    _side_identification_retry_scan_window,
    _promote_strong_candidate_if_safe,
    annotate_serve_order_rule_reviews,
    apply_single_player_side_evidence,
    audit_step3_side_state_logic,
    fill_unknown_side_state_by_continuity,
    infer_current_sides_from_single_player,
    server_identity_for_starter_role,
    select_single_player_side_evidence,
    serve_order_review_markers,
    summarize_rally_start_events,
    timeline_total_rally_start_events,
)


def test_server_identity_maps_role_a_b_to_step2_names() -> None:
    context = Step3PlayerContext(
        player_a_name="Near Player",
        player_b_name="Far Player",
        player_a_starts_near=True,
    )

    assert server_identity_for_starter_role("A", context)["server_player_name"] == "Near Player"
    assert server_identity_for_starter_role("B", context)["server_player_name"] == "Far Player"
    assert server_identity_for_starter_role("", context)["server_player_name"] == "unknown"


def test_timeline_total_rally_start_events_merges_lets_and_server_names() -> None:
    timeline = RallyTimeline(
        points=[
            RallyTimelinePoint(
                id="pt_0001",
                t_start=1.2,
                t_end=2.8,
                starter_role="B",
                flags=["player_sandwich", "rally_label_point"],
            ),
            RallyTimelinePoint(
                id="pt_0002",
                t_start=8.0,
                t_end=9.5,
                starter_role="A",
                flags=["player_sandwich", "rally_label_point"],
            ),
        ],
        analysis_metadata={
            "excluded_let_starts": [
                {
                    "t_start": 4.0,
                    "t_end": 4.8,
                    "starter_role": "B",
                    "flags": ["player_sandwich", "rally_label_let", "let_no_score"],
                }
            ],
            "unattached_trailing_let_starts": [],
        },
    )
    context = Step3PlayerContext(
        player_a_name="Near Player",
        player_b_name="Far Player",
        player_a_starts_near=True,
    )

    events = timeline_total_rally_start_events(timeline, player_context=context)

    assert [event["id"] for event in events] == ["rally_0001", "rally_0002", "rally_0003"]
    assert [event["kind"] for event in events] == ["scoring", "let", "scoring"]
    assert [event["server_player_name"] for event in events] == ["Far Player", "Far Player", "Near Player"]
    assert events[1]["source"] == "excluded_let_starts"


def test_summarize_rally_start_events_exposes_first_server() -> None:
    events = [
        {
            "id": "rally_0001",
            "kind": "scoring",
            "t_start": 1.2,
            "t_end": 2.8,
            "starter_role": "B",
            "server_initial_side": "far",
            "server_player_key": "player_b",
            "server_player_name": "Far Player",
            "server_identity_source": "step2_initial_role_map",
        },
        {
            "id": "rally_0002",
            "kind": "let",
            "t_start": 4.0,
            "t_end": 4.8,
            "starter_role": "B",
            "server_initial_side": "far",
            "server_player_key": "player_b",
            "server_player_name": "Far Player",
            "server_identity_source": "step2_initial_role_map",
        },
    ]

    summary = summarize_rally_start_events(
        events,
        timeline_path=Path("timeline.json"),
        events_json_path=Path("events.json"),
        export_info={"start_frames_dir": "frames", "csv_path": "frames/rally_start_times.csv", "image_count": 2},
    )

    assert summary["total"] == 2
    assert summary["scoring"] == 1
    assert summary["lets"] == 1
    assert summary["needs_review"] == 0
    assert summary["first_server"]["server_player_name"] == "Far Player"


def test_serve_order_review_markers_flag_double_serve_singleton_gap() -> None:
    context = Step3PlayerContext(
        player_a_name="Near Player",
        player_b_name="Far Player",
        player_a_starts_near=True,
    )
    events = [
        {"id": "rally_0001", "kind": "scoring", "t_start": 1.0, "t_end": 2.0, "starter_role": "B"},
        {"id": "rally_0002", "kind": "scoring", "t_start": 8.0, "t_end": 9.0, "starter_role": "B"},
        {"id": "rally_0003", "kind": "scoring", "t_start": 20.0, "t_end": 24.0, "starter_role": "A"},
        {"id": "rally_0004", "kind": "scoring", "t_start": 40.0, "t_end": 42.0, "starter_role": "B"},
        {"id": "rally_0005", "kind": "scoring", "t_start": 50.0, "t_end": 52.0, "starter_role": "B"},
    ]

    markers = serve_order_review_markers(events, player_context=context)

    assert len(markers) == 1
    assert markers[0]["kind"] == "needs_review"
    assert markers[0]["starter_role"] == "A"
    assert markers[0]["server_player_name"] == "Near Player"
    assert markers[0]["t_start"] == 32.0
    assert markers[0]["review_reason"] == "double_serve_singleton_gap"


def test_serve_order_audit_flags_let_from_wrong_server() -> None:
    context = Step3PlayerContext(
        player_a_name="Near Player",
        player_b_name="Far Player",
        player_a_starts_near=True,
    )
    events = [
        {"id": "rally_0001", "kind": "scoring", "t_start": 1.0, "t_end": 2.0, "starter_role": "B"},
        {"id": "rally_0002", "kind": "scoring", "t_start": 8.0, "t_end": 9.0, "starter_role": "B"},
        {"id": "rally_0003", "kind": "scoring", "t_start": 20.0, "t_end": 21.0, "starter_role": "A"},
        {"id": "rally_0004", "kind": "let", "t_start": 24.0, "t_end": 24.8, "starter_role": "B"},
        {"id": "rally_0005", "kind": "needs_review", "t_start": 28.0, "t_end": 28.0, "starter_role": "A"},
        {"id": "rally_0006", "kind": "scoring", "t_start": 35.0, "t_end": 36.0, "starter_role": "B"},
        {"id": "rally_0007", "kind": "scoring", "t_start": 44.0, "t_end": 45.0, "starter_role": "B"},
    ]

    annotate_serve_order_rule_reviews(events, player_context=context)

    assert events[3]["serve_order_expected_role"] == "A"
    assert events[3]["serve_order_expected_server_name"] == "Near Player"
    assert events[3]["serve_order_ok"] is False
    assert events[3]["review_reason"] == "let_server_conflicts_with_expected_turn"


class _FakeRecord:
    def __init__(self, name: str, embedding: np.ndarray) -> None:
        self.name = name
        self._embedding = embedding.astype(np.float32)

    def embedding_array(self) -> np.ndarray:
        return self._embedding


def test_single_player_side_evidence_infers_other_player_by_exclusion() -> None:
    side_map = infer_current_sides_from_single_player(
        identified_player_key="player_b",
        identified_side="NEAR",
    )

    assert side_map == {
        "player_a_current_side": "FAR",
        "player_b_current_side": "NEAR",
    }


def test_single_player_side_evidence_remaps_server_without_swap_cutoff() -> None:
    context = Step3PlayerContext(
        player_a_name="Near Player",
        player_b_name="Far Player",
        player_a_starts_near=True,
    )
    event = {"id": "rally_0001", "starter_role": "B"}

    apply_single_player_side_evidence(
        event,
        {
            "side_evidence_source": "single_player_face_id_at_start",
            "side_evidence_status": "identified",
            "side_identified_player_key": "player_b",
            "side_identified_player_name": "Far Player",
            "side_identified_current_side": "NEAR",
        },
        player_context=context,
    )

    assert event["player_a_current_side"] == "FAR"
    assert event["player_b_current_side"] == "NEAR"
    assert event["current_side"] == "FAR"
    assert event["server_player_name"] == "Near Player"
    assert event["server_identity_source"] == "per_rally_single_player_side_id"


def test_select_single_player_side_evidence_uses_trusted_step2_records_only() -> None:
    player_a_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    player_b_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    trusted = {
        "player_a": _FakeRecord("Near Player", player_a_emb),
        "player_b": _FakeRecord("Far Player", player_b_emb),
    }
    face_results = [
        {
            "body_rank": 1,
            "embedding": np.array([0.02, 0.999, 0.0], dtype=np.float32),
            "face_score": 0.8,
            "t_sec": 20.0,
        }
    ]

    evidence = select_single_player_side_evidence(
        face_results,
        trusted,
        match_threshold=0.35,
        match_margin=0.04,
        min_face_score=0.35,
        min_accepted_samples=1,
    )

    assert evidence["side_evidence_status"] == "identified"
    assert evidence["side_identified_player_key"] == "player_b"
    assert evidence["side_identified_current_side"] == "FAR"


def test_weak_single_player_side_evidence_stays_unknown() -> None:
    player_a_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    player_b_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    trusted = {
        "player_a": _FakeRecord("Near Player", player_a_emb),
        "player_b": _FakeRecord("Far Player", player_b_emb),
    }
    face_results = [
        {
            "body_rank": 1,
            "embedding": np.array([0.02, 0.999, 0.0], dtype=np.float32),
            "face_score": 0.8,
            "t_sec": 20.0,
        },
        {
            "body_rank": 1,
            "embedding": np.array([0.01, 0.999, 0.0], dtype=np.float32),
            "face_score": 0.8,
            "t_sec": 20.25,
        },
    ]

    evidence = select_single_player_side_evidence(
        face_results,
        trusted,
        match_threshold=0.35,
        match_margin=0.04,
        min_best_similarity=0.45,
        min_avg_similarity=0.38,
        min_face_score=0.35,
        min_accepted_samples=4,
    )

    assert evidence["side_evidence_status"] == "unknown"
    assert "not_enough_accepted_samples" in evidence["side_evidence_reason"]
    assert evidence["side_evidence_candidate_player_key"] == "player_b"


def test_side_id_scan_plan_blocks_post_rally_extension_on_long_gap() -> None:
    config = Step3SideIdentificationConfig(
        window_before_sec=1.0,
        window_after_sec=4.0,
        break_gap_sec=12.0,
        next_event_guard_sec=0.25,
    )
    event = {
        "id": "rally_0015",
        "source_t_start": 143.176,
        "source_t_end": 148.815,
    }

    plan = _side_identification_scan_plan(
        event,
        next_start=172.188,
        config=config,
        time_field="source_t_start",
        end_time_field="source_t_end",
    )

    assert plan["primary_lo"] == pytest.approx(142.176)
    assert plan["primary_hi"] == pytest.approx(147.176)
    assert plan["fallback_hi"] == pytest.approx(152.815)
    assert plan["fallback_allowed"] is False
    assert plan["blocks_post_rally"] is True
    assert plan["block_reason"] == "long_gap_blocks_post_rally_extension"


def test_side_id_scan_plan_allows_guarded_extension_when_no_break_gap() -> None:
    config = Step3SideIdentificationConfig(
        window_before_sec=1.0,
        window_after_sec=4.0,
        break_gap_sec=12.0,
        next_event_guard_sec=0.25,
    )
    event = {
        "id": "rally_0016",
        "source_t_start": 172.188,
        "source_t_end": 175.324,
    }

    plan = _side_identification_scan_plan(
        event,
        next_start=180.0,
        config=config,
        time_field="source_t_start",
        end_time_field="source_t_end",
    )

    assert plan["primary_hi"] == pytest.approx(176.188)
    assert plan["fallback_hi"] == pytest.approx(179.324)
    assert plan["fallback_allowed"] is True
    assert plan["blocks_post_rally"] is False


def test_side_id_retry_window_is_start_anchored_and_break_safe() -> None:
    config = Step3SideIdentificationConfig(
        window_before_sec=1.0,
        window_after_sec=4.0,
        retry_window_after_sec=8.0,
        break_gap_sec=12.0,
        next_event_guard_sec=0.25,
    )
    event = {
        "id": "rally_0015",
        "source_t_start": 143.176,
        "source_t_end": 148.815,
    }
    plan = _side_identification_scan_plan(
        event,
        next_start=172.188,
        config=config,
        time_field="source_t_start",
        end_time_field="source_t_end",
    )

    retry_window = _side_identification_retry_scan_window(plan, config=config)

    assert retry_window == pytest.approx((142.176, 148.815))
    assert retry_window[1] < 172.188


def test_strong_candidate_promotion_accepts_only_sample_count_miss() -> None:
    config = Step3SideIdentificationConfig()
    evidence = {
        "side_evidence_status": "unknown",
        "side_evidence_reason": "not_enough_accepted_samples",
        "side_evidence_candidate_player_key": "player_b",
        "side_evidence_candidate_player_name": "Far Player",
        "side_evidence_candidate_current_side": "FAR",
        "side_evidence_candidate_similarity": 0.58,
        "side_evidence_candidate_avg_similarity": 0.54,
        "side_evidence_candidate_margin": 0.46,
        "side_evidence_candidate_sample_count": 3,
    }

    promoted = _promote_strong_candidate_if_safe(evidence, config=config)

    assert promoted["side_evidence_status"] == "identified"
    assert promoted["side_identified_player_key"] == "player_b"
    assert promoted["side_identified_current_side"] == "FAR"
    assert promoted["side_evidence_reason"] == "promoted_strong_candidate"


def test_continuity_fill_infers_between_matching_known_side_maps() -> None:
    context = Step3PlayerContext("Near Player", "Far Player")
    events = [
        {
            "id": "rally_0001",
            "source_t_start": 10.0,
            "starter_role": "B",
            "player_a_current_side": "NEAR",
            "player_b_current_side": "FAR",
            "side_evidence_status": "identified",
        },
        {
            "id": "rally_0002",
            "source_t_start": 20.0,
            "starter_role": "B",
            "player_a_current_side": "unknown",
            "player_b_current_side": "unknown",
            "side_evidence_status": "unknown",
        },
        {
            "id": "rally_0003",
            "source_t_start": 40.0,
            "starter_role": "A",
            "player_a_current_side": "NEAR",
            "player_b_current_side": "FAR",
            "side_evidence_status": "identified",
        },
    ]

    filled = fill_unknown_side_state_by_continuity(
        events,
        player_context=context,
        time_field="source_t_start",
        max_terminal_gap_sec=12.0,
    )

    assert filled == 1
    assert events[1]["side_evidence_status"] == "inferred"
    assert events[1]["player_a_current_side"] == "NEAR"
    assert events[1]["player_b_current_side"] == "FAR"
    assert events[1]["server_player_name"] == "Far Player"


def _logic_event(
    idx: int,
    server_key: str,
    *,
    kind: str = "scoring",
    t: float | None = None,
    player_a_side: str = "NEAR",
    player_b_side: str = "FAR",
) -> dict:
    t_start = float(idx * 10 if t is None else t)
    return {
        "id": f"rally_{idx:04d}",
        "kind": kind,
        "source_t_start": t_start,
        "source_t_end": t_start + 1.0,
        "server_player_key": server_key,
        "server_player_name": "Near Player" if server_key == "player_a" else "Far Player",
        "player_a_current_side": player_a_side,
        "player_b_current_side": player_b_side,
        "side_evidence_status": "identified",
    }


def test_step3_logic_audit_accepts_double_serve_order_with_let_replay() -> None:
    context = Step3PlayerContext("Near Player", "Far Player")
    events = [
        _logic_event(1, "player_b"),
        _logic_event(2, "player_b"),
        _logic_event(3, "player_a"),
        _logic_event(4, "player_a", kind="let"),
        _logic_event(5, "player_a"),
        _logic_event(6, "player_b"),
        _logic_event(7, "player_b"),
    ]

    audit = audit_step3_side_state_logic(events, player_context=context)

    assert audit["ok"] is True
    assert audit["blocking_issue_count"] == 0
    assert events[3]["step3_3_expected_server_key"] == "player_a"
    assert events[3]["step3_3_logic_ok"] is True


def test_step3_logic_audit_flags_singleton_server_run_for_step3_2_rescan() -> None:
    context = Step3PlayerContext("Near Player", "Far Player")
    events = [
        _logic_event(1, "player_b"),
        _logic_event(2, "player_b"),
        _logic_event(3, "player_a"),
        _logic_event(4, "player_b"),
        _logic_event(5, "player_b"),
    ]

    audit = audit_step3_side_state_logic(events, player_context=context)

    assert audit["ok"] is False
    assert "rally_0003" in audit["rescan_event_ids"]
    assert any(issue["type"] == "serve_run_length_violation" for issue in audit["issues"])
    assert any(issue["type"] == "serve_order_conflict" for issue in audit["issues"])


def test_step3_logic_audit_resets_expected_server_at_likely_set_boundary() -> None:
    context = Step3PlayerContext("Near Player", "Far Player")
    events = []
    servers = ["player_b", "player_b", "player_a", "player_a", "player_b", "player_b"]
    servers += ["player_a", "player_a", "player_b", "player_b", "player_a"]
    for idx, server in enumerate(servers, start=1):
        events.append(_logic_event(idx, server))
    events.append(
        _logic_event(
            12,
            "player_a",
            t=150.0,
            player_a_side="FAR",
            player_b_side="NEAR",
        )
    )
    events.append(
        _logic_event(
            13,
            "player_a",
            t=160.0,
            player_a_side="FAR",
            player_b_side="NEAR",
        )
    )

    audit = audit_step3_side_state_logic(
        events,
        player_context=context,
        config=Step3LogicAuditConfig(set_boundary_gap_sec=12.0, min_scoring_before_set_boundary=11),
    )

    assert audit["ok"] is True
    assert audit["segments"] == 2
    assert events[11]["step3_3_set_boundary_before"] is True
