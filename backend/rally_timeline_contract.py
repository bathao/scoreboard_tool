# backend/rally_timeline_contract.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal, Tuple
import json
import math
from pathlib import Path

from backend.models import RallyEvent

# --- CONSTANTS ---
RallyWinner = Literal["player_a", "player_b", "unknown"]
EventSource = Literal["ai", "human"]
WinnerDecision = Literal["auto", "review", "blocked"]
SCHEMA_VERSION = "rally_timeline_v1"

@dataclass(frozen=True)
class Correction:
    """Audit log for human or automated corrections."""
    at: str
    by: str
    changes: Dict[str, Dict[str, Any]]
    note: str = ""

@dataclass
class RallyTimelinePoint:
    """Contract for a single rally segment in the frozen rally timeline."""
    id: str
    t_start: float
    t_end: float
    active_start: Optional[float] = None
    active_end: Optional[float] = None
    search_upper_bound: Optional[float] = None
    starter_role: Optional[str] = None
    preceding_let_count: int = 0
    preceding_let_starts: List[float] = field(default_factory=list)
    service_attempt_index: int = 1
    boundary_mode: Optional[str] = None
    endpoint_mode: Optional[str] = None
    endpoint_confidence: float = 0.0
    point_end_event: Optional[str] = None
    winner_candidate: RallyWinner = "unknown"
    winner_confidence: float = 0.0
    winner_decision: Optional[WinnerDecision] = None
    winner_reason: Optional[str] = None
    winner_model: Optional[str] = None
    winner_score_a: float = 0.0
    winner_score_b: float = 0.0
    winner_end_category: Optional[str] = None
    winner_loser_candidate: RallyWinner = "unknown"
    winner_last_hitter_candidate: RallyWinner = "unknown"
    winner: RallyWinner = "unknown"
    confidence: float = 0.0
    flags: List[str] = field(default_factory=list)
    source: EventSource = "ai"
    corrections: List[Correction] = field(default_factory=list)
    # Set assignment — populated by set boundary detection after winner prediction
    set_number: int = 1
    # Optional YOLO-derived side positions (populated by pipeline for Signal 3)
    near_mean_x: Optional[float] = None
    far_mean_x: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RallyTimelinePoint":
        # STRICT KEY CHECK
        for key in ["id", "t_start", "t_end"]:
            if key not in d:
                raise KeyError(f"CRITICAL: RallyTimelinePoint missing mandatory key: '{key}'")
        
        corrections_raw = d.get("corrections", []) or []
        corrections = [
            Correction(
                at=str(c.get("at", "")),
                by=str(c.get("by", "")),
                changes=dict(c.get("changes", {})),
                note=str(c.get("note", ""))
            ) for c in corrections_raw
        ]

        return RallyTimelinePoint(
            id=str(d["id"]),
            t_start=float(d["t_start"]),
            t_end=float(d["t_end"]),
            active_start=(None if d.get("active_start") is None else float(d.get("active_start"))),
            active_end=(None if d.get("active_end") is None else float(d.get("active_end"))),
            search_upper_bound=(None if d.get("search_upper_bound") is None else float(d.get("search_upper_bound"))),
            starter_role=(None if d.get("starter_role") in (None, "") else str(d.get("starter_role"))),
            preceding_let_count=int(d.get("preceding_let_count", 0)),
            preceding_let_starts=[float(x) for x in (d.get("preceding_let_starts", []) or [])],
            service_attempt_index=int(d.get("service_attempt_index", 1)),
            boundary_mode=(None if d.get("boundary_mode") in (None, "") else str(d.get("boundary_mode"))),
            endpoint_mode=(None if d.get("endpoint_mode") in (None, "") else str(d.get("endpoint_mode"))),
            endpoint_confidence=float(d.get("endpoint_confidence", 0.0)),
            point_end_event=(None if d.get("point_end_event") in (None, "") else str(d.get("point_end_event"))),
            winner_candidate=str(d.get("winner_candidate", "unknown")),  # type: ignore
            winner_confidence=float(d.get("winner_confidence", 0.0)),
            winner_decision=(None if d.get("winner_decision") in (None, "") else str(d.get("winner_decision"))),  # type: ignore
            winner_reason=(None if d.get("winner_reason") in (None, "") else str(d.get("winner_reason"))),
            winner_model=(None if d.get("winner_model") in (None, "") else str(d.get("winner_model"))),
            winner_score_a=float(d.get("winner_score_a", 0.0)),
            winner_score_b=float(d.get("winner_score_b", 0.0)),
            winner_end_category=(None if d.get("winner_end_category") in (None, "") else str(d.get("winner_end_category"))),
            winner_loser_candidate=str(d.get("winner_loser_candidate", "unknown")),  # type: ignore
            winner_last_hitter_candidate=str(d.get("winner_last_hitter_candidate", "unknown")),  # type: ignore
            winner=str(d.get("winner", "unknown")),  # type: ignore
            confidence=float(d.get("confidence", 0.0)),
            flags=list(d.get("flags", [])),
            source=str(d.get("source", "ai")),  # type: ignore
            corrections=corrections,
            set_number=int(d.get("set_number", 1)),
            near_mean_x=(None if d.get("near_mean_x") is None else float(d["near_mean_x"])),
            far_mean_x=(None if d.get("far_mean_x") is None else float(d["far_mean_x"])),
        )


NON_SCORING_POINT_FLAGS = {"rally_label_let", "let_no_score"}


def counts_toward_score(point: RallyTimelinePoint) -> bool:
    return not any(flag in NON_SCORING_POINT_FLAGS for flag in point.flags)

@dataclass
class RallyTimeline:
    """Root container for frozen rally timeline analysis data."""
    schema_version: str = SCHEMA_VERSION
    sport: str = "table_tennis"
    video_path: str = ""
    video_fps: Optional[float] = None
    best_of: int = 5
    created_at: str = ""
    roi: Dict[str, int] = field(default_factory=dict) # Strict ROI storage
    points: List[RallyTimelinePoint] = field(default_factory=list)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    score_validation: Dict[str, Any] = field(default_factory=dict)

    def build_summary(self) -> Dict[str, Any]:
        total_rallies = len(self.points)
        unknown_winner_rallies = sum(1 for p in self.points if p.winner == "unknown")
        candidate_unknown_rallies = sum(1 for p in self.points if p.winner_candidate == "unknown")
        auto_winner_rallies = sum(1 for p in self.points if p.winner_decision == "auto" and p.winner != "unknown")
        review_winner_rallies = sum(1 for p in self.points if p.winner_decision == "review")
        blocked_winner_rallies = sum(1 for p in self.points if p.winner_decision == "blocked")
        non_scoring_rallies = sum(1 for p in self.points if not counts_toward_score(p))
        return {
            "total_rallies": total_rallies,
            "scoring_rallies": total_rallies - non_scoring_rallies,
            "non_scoring_rallies": non_scoring_rallies,
            "winner_known_rallies": total_rallies - unknown_winner_rallies,
            "winner_unknown_rallies": unknown_winner_rallies,
            "winner_candidate_known_rallies": total_rallies - candidate_unknown_rallies,
            "winner_candidate_unknown_rallies": candidate_unknown_rallies,
            "winner_auto_rallies": auto_winner_rallies,
            "winner_review_rallies": review_winner_rallies,
            "winner_blocked_rallies": blocked_winner_rallies,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sport": self.sport,
            "video_path": self.video_path,
            "video_fps": self.video_fps,
            "best_of": self.best_of,
            "created_at": self.created_at,
            "roi": self.roi,
            "points": [p.to_dict() for p in self.points],
            "summary": self.build_summary(),
            "analysis_metadata": self.analysis_metadata,
            "score_validation": self.score_validation,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RallyTimeline":
        # 1. TOP-LEVEL STRICT VALIDATION
        required = ["schema_version", "video_path", "video_fps", "roi"]
        for key in required:
            if key not in d or d[key] is None:
                raise ValueError(f"CRITICAL DATA ERROR: Mandatory field '{key}' is missing or null.")

        # 2. ROI STRUCTURE STRICT VALIDATION
        roi = d["roi"]
        for k in ["x", "y", "w", "h"]:
            if k not in roi or not isinstance(roi[k], int):
                raise KeyError(f"CRITICAL ROI ERROR: ROI field '{k}' is missing or not an integer.")

        # 3. RECONSTRUCTION
        points = [RallyTimelinePoint.from_dict(x) for x in (d.get("points", []) or [])]
        return RallyTimeline(
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
            sport=str(d.get("sport", "table_tennis")),
            video_path=str(d.get("video_path", "")),
            video_fps=float(d["video_fps"]),
            best_of=int(d.get("best_of", 5)),
            created_at=str(d.get("created_at", "")),
            roi=dict(roi),
            points=points,
            analysis_metadata=dict(d.get("analysis_metadata", {})),
            score_validation=dict(d.get("score_validation", {})),
        )

# --- SEMANTIC VALIDATORS ---
def validate_rally_timeline(timeline: RallyTimeline) -> List[str]:
    errors = []
    if not timeline.roi or timeline.roi.get('w', 0) <= 0:
        errors.append("ROI is missing or has zero width")
    if timeline.video_fps is None or timeline.video_fps <= 0:
        errors.append("Invalid video_fps")
    
    last_t = -1.0
    for i, p in enumerate(timeline.points):
        if p.t_start < 0 or p.t_end <= p.t_start:
            errors.append(f"Point {i}: Invalid time range ({p.t_start} -> {p.t_end})")
        if p.t_start < last_t:
            errors.append(f"Point {i}: Non-monotonic timestamps (start before previous end)")
        last_t = p.t_start
    return errors

# --- UI & LOGIC HELPERS ---
def classify_review_bucket(confidence: float) -> str:
    if confidence >= 0.85: return "auto"
    if confidence >= 0.60: return "review"
    return "block"

def needs_human_review(p: RallyTimelinePoint) -> bool:
    if p.winner == "unknown" or classify_review_bucket(p.confidence) != "auto":
        return True
    return False

# --- CONVERSION & IO ---
def to_core_rally_events(timeline: RallyTimeline, timestamp_mode: Literal["end", "start"] = "end") -> List[RallyEvent]:
    core = []
    for p in timeline.points:
        if p.winner == "unknown":
            continue
        if not counts_toward_score(p):
            continue
        ts = p.t_end if timestamp_mode == "end" else p.t_start
        core.append(RallyEvent(winner=str(p.winner), timestamp=float(ts)))
    core.sort(key=lambda e: e.timestamp)
    return core

def save_rally_timeline(path: Path, timeline: RallyTimeline) -> None:
    errors = validate_rally_timeline(timeline)
    if errors:
        raise ValueError(f"STRICT SAVE FAILED: {errors}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timeline.to_dict(), f, ensure_ascii=False, indent=2)

def load_rally_timeline(path: Path) -> RallyTimeline:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return RallyTimeline.from_dict(data)
