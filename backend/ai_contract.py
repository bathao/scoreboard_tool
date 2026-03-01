# backend/ai_contract.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal, Tuple
import json
from pathlib import Path

from backend.models import RallyEvent


DraftWinner = Literal["player_a", "player_b", "unknown"]
EventSource = Literal["ai", "human"]


@dataclass(frozen=True)
class Correction:
    """
    Audit log for human (or automated) corrections.
    """
    at: str  # ISO string recommended, e.g. "2026-03-01T12:10:05+07:00"
    by: str  # username / machine tag
    changes: Dict[str, Dict[str, Any]]  # {"winner": {"from": "unknown", "to": "player_a"}}
    note: str = ""


@dataclass
class DraftPointEvent:
    """
    AI Draft contract for a single point/rally.

    - Uses time in *seconds* from start of (normalized) video.
    - winner can be "unknown" in draft.
    - confidence in [0,1]
    - flags are machine-readable hints for UI routing
    """
    id: str
    t_start: float
    t_end: float
    winner: DraftWinner = "unknown"
    confidence: float = 0.0
    flags: List[str] = field(default_factory=list)
    source: EventSource = "ai"
    corrections: List[Correction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # dataclasses -> dict already converts nested dataclasses, OK
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DraftPointEvent":
        corrections_raw = d.get("corrections", []) or []
        corrections: List[Correction] = []
        for c in corrections_raw:
            corrections.append(
                Correction(
                    at=str(c.get("at", "")),
                    by=str(c.get("by", "")),
                    changes=dict(c.get("changes", {})),
                    note=str(c.get("note", "")),
                )
            )

        return DraftPointEvent(
            id=str(d["id"]),
            t_start=float(d["t_start"]),
            t_end=float(d["t_end"]),
            winner=str(d.get("winner", "unknown")),  # type: ignore
            confidence=float(d.get("confidence", 0.0)),
            flags=list(d.get("flags", [])),
            source=str(d.get("source", "ai")),  # type: ignore
            corrections=corrections,
        )


@dataclass
class DraftMatch:
    """
    Draft container for a match video analysis run.
    """
    schema_version: str = "draft_match_v1"
    sport: str = "table_tennis"
    video_path: str = ""
    video_fps: Optional[float] = None
    best_of: int = 5
    created_at: str = ""
    points: List[DraftPointEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sport": self.sport,
            "video_path": self.video_path,
            "video_fps": self.video_fps,
            "best_of": self.best_of,
            "created_at": self.created_at,
            "points": [p.to_dict() for p in self.points],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DraftMatch":
        points = [DraftPointEvent.from_dict(x) for x in (d.get("points", []) or [])]
        return DraftMatch(
            schema_version=str(d.get("schema_version", "draft_match_v1")),
            sport=str(d.get("sport", "table_tennis")),
            video_path=str(d.get("video_path", "")),
            video_fps=(float(d["video_fps"]) if d.get("video_fps") is not None else None),
            best_of=int(d.get("best_of", 5)),
            created_at=str(d.get("created_at", "")),
            points=points,
        )


# =============================================================================
# Validation (shape + semantics)
# =============================================================================

def _is_finite_number(x: float) -> bool:
    return isinstance(x, (int, float)) and x == x and x not in (float("inf"), float("-inf"))


def validate_point_event(p: DraftPointEvent) -> List[str]:
    """
    Return list of problems (empty == valid).
    This is a semantic validator for local pipeline safety.
    """
    problems: List[str] = []

    if not p.id:
        problems.append("id is empty")

    if not _is_finite_number(p.t_start) or not _is_finite_number(p.t_end):
        problems.append("t_start or t_end not finite number")
    else:
        if p.t_start < 0:
            problems.append("t_start < 0")
        if p.t_end < 0:
            problems.append("t_end < 0")
        if p.t_end <= p.t_start:
            problems.append("t_end must be > t_start")

    if p.winner not in ("player_a", "player_b", "unknown"):
        problems.append(f"invalid winner: {p.winner}")

    if not _is_finite_number(p.confidence):
        problems.append("confidence not finite number")
    else:
        if p.confidence < 0 or p.confidence > 1:
            problems.append("confidence must be in [0,1]")

    if p.source not in ("ai", "human"):
        problems.append(f"invalid source: {p.source}")

    # flags should be list[str]
    if not isinstance(p.flags, list) or any(not isinstance(f, str) for f in p.flags):
        problems.append("flags must be list[str]")

    return problems


def validate_draft_match(m: DraftMatch) -> List[str]:
    problems: List[str] = []
    if m.schema_version != "draft_match_v1":
        problems.append(f"unsupported schema_version: {m.schema_version}")

    if m.sport != "table_tennis":
        problems.append(f"unexpected sport: {m.sport}")

    if m.best_of <= 0 or (m.best_of % 2 == 0):
        problems.append("best_of must be positive odd number")

    # Validate each point
    for i, p in enumerate(m.points):
        pp = validate_point_event(p)
        problems.extend([f"points[{i}]: {msg}" for msg in pp])

    # Sort/order sanity: non-overlapping is NOT required,
    # but monotonic time is recommended for UI and later conversion.
    # We enforce non-decreasing t_start.
    last_start: Optional[float] = None
    for i, p in enumerate(m.points):
        if last_start is not None and p.t_start < last_start:
            problems.append(f"points[{i}]: t_start not non-decreasing")
        last_start = p.t_start

    return problems


# =============================================================================
# Confidence routing helpers
# =============================================================================

def classify_review_bucket(confidence: float) -> str:
    """
    Bucket used for UI routing:
    - auto: >= 0.85
    - review: [0.60, 0.85)
    - block: < 0.60
    """
    if confidence >= 0.85:
        return "auto"
    if confidence >= 0.60:
        return "review"
    return "block"


def needs_human_review(p: DraftPointEvent) -> bool:
    """
    Review if:
    - winner unknown, OR
    - low confidence, OR
    - flags indicate problems
    """
    if p.winner == "unknown":
        return True
    if classify_review_bucket(p.confidence) != "auto":
        return True
    heavy_flags = {"TIME_INCONSISTENT", "SEGMENT_UNSTABLE", "POSSIBLE_MISSED_POINT"}
    if any(f in heavy_flags for f in p.flags):
        return True
    return False


# =============================================================================
# Conversion: Draft -> Core (RallyEvent list)
# =============================================================================

def to_core_rally_events(
    draft: DraftMatch,
    *,
    timestamp_mode: Literal["end", "start"] = "end",
    require_resolved_winner: bool = True,
) -> List[RallyEvent]:
    """
    Convert DraftMatch points into core RallyEvent list for ScoreEngine.

    timestamp_mode:
      - "end": RallyEvent.timestamp = t_end
      - "start": RallyEvent.timestamp = t_start

    require_resolved_winner:
      - True: raise ValueError if any winner == "unknown"
      - False: drop unknown events (or you can keep them out for now)

    Notes:
    - Output events are sorted by timestamp and monotonic by design.
    """
    core: List[RallyEvent] = []

    for p in draft.points:
        if p.winner == "unknown":
            if require_resolved_winner:
                raise ValueError(f"Unresolved winner in point id={p.id}")
            else:
                continue

        ts = p.t_end if timestamp_mode == "end" else p.t_start
        core.append(RallyEvent(winner=p.winner, timestamp=float(ts)))  # type: ignore

    core.sort(key=lambda e: e.timestamp)

    # Ensure monotonic non-decreasing timestamps
    last: Optional[float] = None
    for e in core:
        if last is not None and e.timestamp < last:
            raise ValueError("Converted core events are not monotonic (unexpected)")
        last = e.timestamp

    return core


# =============================================================================
# IO
# =============================================================================

def save_draft_match(path: Path, draft: DraftMatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(draft.to_dict(), f, ensure_ascii=False, indent=2)


def load_draft_match(path: Path) -> DraftMatch:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    draft = DraftMatch.from_dict(data)
    problems = validate_draft_match(draft)
    if problems:
        msg = "DraftMatch validation failed:\n" + "\n".join(f"- {p}" for p in problems[:50])
        raise ValueError(msg)
    return draft


def save_draft_points_json(path: Path, points: List[DraftPointEvent]) -> None:
    """
    Save only points list (lightweight).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in points], f, ensure_ascii=False, indent=2)


def load_draft_points_json(path: Path) -> List[DraftPointEvent]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Draft points JSON must be a list")
    points = [DraftPointEvent.from_dict(x) for x in data]
    # lightweight validation
    for i, p in enumerate(points):
        probs = validate_point_event(p)
        if probs:
            raise ValueError(f"DraftPointEvent[{i}] invalid: {probs}")
    # sort by time
    points.sort(key=lambda p: p.t_start)
    return points