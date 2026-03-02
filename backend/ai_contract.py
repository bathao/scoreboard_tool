# backend/ai_contract.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal, Tuple
import json
import math
from pathlib import Path

from backend.models import RallyEvent

# --- CONSTANTS ---
DraftWinner = Literal["player_a", "player_b", "unknown"]
EventSource = Literal["ai", "human"]
SCHEMA_VERSION = "draft_match_v1"

@dataclass(frozen=True)
class Correction:
    """Audit log for human or automated corrections."""
    at: str
    by: str
    changes: Dict[str, Dict[str, Any]]
    note: str = ""

@dataclass
class DraftPointEvent:
    """Contract for a single rally segment."""
    id: str
    t_start: float
    t_end: float
    winner: DraftWinner = "unknown"
    confidence: float = 0.0
    flags: List[str] = field(default_factory=list)
    source: EventSource = "ai"
    corrections: List[Correction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DraftPointEvent":
        # STRICT KEY CHECK
        for key in ["id", "t_start", "t_end"]:
            if key not in d:
                raise KeyError(f"CRITICAL: DraftPointEvent missing mandatory key: '{key}'")
        
        corrections_raw = d.get("corrections", []) or []
        corrections = [
            Correction(
                at=str(c.get("at", "")),
                by=str(c.get("by", "")),
                changes=dict(c.get("changes", {})),
                note=str(c.get("note", ""))
            ) for c in corrections_raw
        ]

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
    """Root container for match analysis draft data."""
    schema_version: str = SCHEMA_VERSION
    sport: str = "table_tennis"
    video_path: str = ""
    video_fps: Optional[float] = None
    best_of: int = 5
    created_at: str = ""
    roi: Dict[str, int] = field(default_factory=dict) # Strict ROI storage
    points: List[DraftPointEvent] = field(default_factory=list)

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
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DraftMatch":
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
        points = [DraftPointEvent.from_dict(x) for x in (d.get("points", []) or [])]
        return DraftMatch(
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
            sport=str(d.get("sport", "table_tennis")),
            video_path=str(d.get("video_path", "")),
            video_fps=float(d["video_fps"]),
            best_of=int(d.get("best_of", 5)),
            created_at=str(d.get("created_at", "")),
            roi=dict(roi),
            points=points,
        )

# --- SEMANTIC VALIDATORS ---
def validate_draft_match(m: DraftMatch) -> List[str]:
    errors = []
    if not m.roi or m.roi.get('w', 0) <= 0:
        errors.append("ROI is missing or has zero width")
    if m.video_fps is None or m.video_fps <= 0:
        errors.append("Invalid video_fps")
    
    last_t = -1.0
    for i, p in enumerate(m.points):
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

def needs_human_review(p: DraftPointEvent) -> bool:
    if p.winner == "unknown" or classify_review_bucket(p.confidence) != "auto":
        return True
    return False

# --- CONVERSION & IO ---
def to_core_rally_events(draft: DraftMatch, timestamp_mode: Literal["end", "start"] = "end") -> List[RallyEvent]:
    core = []
    for p in draft.points:
        if p.winner == "unknown": continue
        ts = p.t_end if timestamp_mode == "end" else p.t_start
        core.append(RallyEvent(winner=str(p.winner), timestamp=float(ts)))
    core.sort(key=lambda e: e.timestamp)
    return core

def save_draft_match(path: Path, draft: DraftMatch) -> None:
    errors = validate_draft_match(draft)
    if errors:
        raise ValueError(f"STRICT SAVE FAILED: {errors}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(draft.to_dict(), f, ensure_ascii=False, indent=2)

def load_draft_match(path: Path) -> DraftMatch:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DraftMatch.from_dict(data)