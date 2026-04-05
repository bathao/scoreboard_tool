from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal

import ollama


WinnerLabel = Literal["player_a", "player_b", "unknown"]
WinnerDecision = Literal["auto", "review", "blocked"]


@dataclass(frozen=True)
class WinnerPrediction:
    winner: WinnerLabel
    confidence: float
    decision: WinnerDecision
    score_a: float
    score_b: float
    reason: str
    raw: str
    thinking: str
    model: str


def _extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _extract_tagged_json_block(text: str, tag: str = "FINAL_JSON:") -> Dict[str, Any]:
    raw = str(text or "")
    if not raw:
        return {}
    idx = raw.rfind(tag)
    if idx < 0:
        return {}
    tail = raw[idx + len(tag) :].strip()
    return _extract_json_block(tail)


def _normalize_winner(value: Any) -> WinnerLabel:
    raw = str(value or "").strip().lower()
    if raw in {"player_a", "a", "near", "near_side", "near-side"}:
        return "player_a"
    if raw in {"player_b", "b", "far", "far_side", "far-side"}:
        return "player_b"
    return "unknown"


def _normalize_decision(
    winner: WinnerLabel,
    confidence: float,
    proposed: Any,
) -> WinnerDecision:
    if winner == "unknown":
        return "blocked"

    raw = str(proposed or "").strip().lower()
    if confidence >= 0.86 and raw == "auto":
        return "auto"
    if confidence >= 0.58 and raw in {"auto", "review"}:
        return "review" if raw != "auto" else "auto"
    if confidence >= 0.58:
        return "review"
    return "blocked"


def _extract_winner_from_free_text(text: str) -> WinnerLabel:
    raw = str(text or "").strip().lower()
    if not raw:
        return "unknown"

    direct = re.findall(r"\b(player_a|player_b|unknown)\b", raw)
    if direct:
        return _normalize_winner(direct[-1])

    patterns = [
        (r"\bplayer a\b.*\b(win|wins|won)\b", "player_a"),
        (r"\bplayer b\b.*\b(win|wins|won)\b", "player_b"),
        (r"\b(win|wins|won)\b.*\bplayer a\b", "player_a"),
        (r"\b(win|wins|won)\b.*\bplayer b\b", "player_b"),
        (r"\bnear(?:-side| side)?\b.*\b(win|wins|won)\b", "player_a"),
        (r"\bfar(?:-side| side)?\b.*\b(win|wins|won)\b", "player_b"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, raw):
            return _normalize_winner(label)
    return "unknown"


def _extract_confidence_from_free_text(text: str) -> float | None:
    raw = str(text or "").strip().lower()
    if not raw:
        return None

    percent_matches = re.findall(r"(\d{2,3})\s*%", raw)
    if percent_matches:
        try:
            value = float(percent_matches[-1]) / 100.0
            return max(0.0, min(1.0, value))
        except ValueError:
            pass

    decimal_matches = re.findall(r"\b(?:confidence\s*[:=]\s*)?(0?\.\d+)\b", raw)
    if decimal_matches:
        try:
            value = float(decimal_matches[-1])
            return max(0.0, min(1.0, value))
        except ValueError:
            pass
    return None


def _extract_side_scores_from_free_text(text: str) -> tuple[float | None, float | None]:
    raw = str(text or "").strip().lower()
    if not raw:
        return None, None

    def _pct(pattern: str) -> float | None:
        match = re.search(pattern, raw)
        if not match:
            return None
        try:
            value = float(match.group(1))
        except ValueError:
            return None
        if value > 1.0:
            value = value / 100.0
        return max(0.0, min(1.0, value))

    score_a = _pct(r"(?:player_a_win_score|score_a|player a score|player a win score)\s*[:=]\s*(\d{1,3}(?:\.\d+)?)")
    score_b = _pct(r"(?:player_b_win_score|score_b|player b score|player b win score)\s*[:=]\s*(\d{1,3}(?:\.\d+)?)")
    return score_a, score_b


def _scores_from_winner_and_confidence(winner: WinnerLabel, confidence: float) -> tuple[float, float]:
    conf = float(max(0.0, min(1.0, confidence)))
    if winner == "player_a":
        return conf, 1.0 - conf
    if winner == "player_b":
        return 1.0 - conf, conf
    return 0.5, 0.5


def _confidence_from_scores(score_a: float, score_b: float, winner: WinnerLabel) -> float:
    a = float(max(0.0, min(1.0, score_a)))
    b = float(max(0.0, min(1.0, score_b)))
    if winner == "player_a":
        return a
    if winner == "player_b":
        return b
    return max(a, b)


class OllamaVisionClient:
    def __init__(
        self,
        model_name: str = "qwen3-vl:8b",
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.temperature = float(temperature)

    def _summarize_thinking_to_json(self, thinking: str) -> Dict[str, Any]:
        raw_thinking = str(thinking or "").strip()
        if not raw_thinking:
            return {}
        prompt = (
            "Condense the analysis below into one compact JSON object only. Do not explain. "
            "Choose player_a or player_b. Do not answer unknown. Output exactly:\n"
            "{\"winner\":\"player_a or player_b\",\"player_a_win_score\":0-100,"
            "\"player_b_win_score\":0-100,\"reason\":\"one short sentence\"}\n"
            "ANALYSIS:\n"
            f"{raw_thinking}"
        )
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            options={
                "temperature": self.temperature,
                "num_predict": 160,
            },
        )
        message = getattr(response, "message", None)
        raw = str(getattr(message, "content", "") or "").strip()
        thinking2 = str(getattr(message, "thinking", "") or "").strip()
        data = _extract_json_block(raw)
        if not data:
            data = _extract_tagged_json_block(raw)
        if not data:
            data = _extract_tagged_json_block(thinking2)
        if not data:
            data = _extract_json_block(thinking2)
        return data

    def predict_winner(self, image_path: str) -> str:
        prediction = self.predict_winner_structured([image_path])
        return prediction.winner

    def predict_winner_structured(
        self,
        image_paths: Iterable[str | Path],
        *,
        prompt: str | None = None,
    ) -> WinnerPrediction:
        image_bytes: List[bytes] = []
        for image_path in image_paths:
            with open(image_path, "rb") as f:
                image_bytes.append(f.read())

        is_qwen3_vl = self.model_name.startswith("qwen3-vl")

        if prompt is None:
            if is_qwen3_vl:
                prompt = (
                    "You are given ordered images from the same table-tennis rally. For each time step there are two "
                    "images: first a full view, then a zoomed table-centered view. The time steps go from earlier to "
                    "later. Player A is the near-side player. Player B is the far-side player. You MUST choose the "
                    "winner of this rally as player_a or player_b. Do not answer unknown. Judge mainly from the last "
                    "clear shot, whether the opponent had a real return, and the immediate aftermath. Do not decide "
                    "only from who walks away or picks up the ball. Keep your reasoning brief. End with exactly one line "
                    "that starts with FINAL_JSON: followed by one compact JSON object:\n"
                    "{\"winner\":\"player_a or player_b\",\"player_a_win_score\":0-100,"
                    "\"player_b_win_score\":0-100,\"reason\":\"one short sentence\"}\n"
                    "The two scores must add to 100. Give higher score to the side more likely to have won this rally."
                )
            else:
                prompt = (
                    "These 4 images are ordered from earlier to later in the second half of one table-tennis rally. "
                    "The top player is player_b (far side). The bottom player is player_a (near side). "
                    "Do not guess based on which player is closer to the camera. "
                    "Choose the player who most likely WON the point after the final exchange and immediate aftermath. "
                    "If the evidence is weak or ambiguous, answer unknown. "
                    "Reply with exactly one label and nothing else: player_a or player_b or unknown."
                )

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": image_bytes,
                }
            ],
            think=False,
            options={
                "temperature": self.temperature,
                "num_predict": 384 if is_qwen3_vl else 8,
            },
        )

        message = getattr(response, "message", None)
        raw = str(getattr(message, "content", "") or "").strip()
        thinking = str(getattr(message, "thinking", "") or "").strip()
        data = _extract_json_block(raw)
        if not data:
            data = _extract_tagged_json_block(raw)
        if not data:
            data = _extract_tagged_json_block(thinking)
        if not data:
            data = _extract_json_block(thinking)
        if not data and is_qwen3_vl and thinking:
            data = self._summarize_thinking_to_json(thinking)

        if data:
            winner = _normalize_winner(data.get("winner"))
            score_a = data.get("player_a_win_score")
            score_b = data.get("player_b_win_score")
            try:
                score_a_f = float(score_a if score_a is not None else 0.0)
            except (TypeError, ValueError):
                score_a_f = 0.0
            try:
                score_b_f = float(score_b if score_b is not None else 0.0)
            except (TypeError, ValueError):
                score_b_f = 0.0
            if score_a_f > 1.0 or score_b_f > 1.0:
                score_a_f /= 100.0
                score_b_f /= 100.0
            score_a_f = float(max(0.0, min(1.0, score_a_f)))
            score_b_f = float(max(0.0, min(1.0, score_b_f)))
            if score_a_f <= 0.0 and score_b_f <= 0.0 and winner != "unknown":
                score_a_f, score_b_f = _scores_from_winner_and_confidence(winner, 0.64)
            confidence = _confidence_from_scores(score_a_f, score_b_f, winner)
            reason = str(data.get("reason", "")).strip()
            if len(reason) > 120:
                reason = reason[:120].rstrip()
            decision = _normalize_decision(winner, confidence, data.get("decision"))
        else:
            winner = _normalize_winner(raw)
            if winner == "unknown":
                winner = _extract_winner_from_free_text(raw)
            if winner == "unknown":
                winner = _extract_winner_from_free_text(thinking)
            score_a_f, score_b_f = _extract_side_scores_from_free_text(raw)
            if score_a_f is None and score_b_f is None:
                score_a_f, score_b_f = _extract_side_scores_from_free_text(thinking)
            confidence = _extract_confidence_from_free_text(raw)
            if confidence is None:
                confidence = _extract_confidence_from_free_text(thinking)
            if score_a_f is not None or score_b_f is not None:
                if score_a_f is None:
                    score_a_f = 1.0 - float(score_b_f)
                if score_b_f is None:
                    score_b_f = 1.0 - float(score_a_f)
                score_a_f = float(max(0.0, min(1.0, score_a_f)))
                score_b_f = float(max(0.0, min(1.0, score_b_f)))
                if winner == "unknown":
                    winner = "player_a" if score_a_f >= score_b_f else "player_b"
                confidence = _confidence_from_scores(score_a_f, score_b_f, winner)
            else:
                if confidence is None:
                    confidence = 0.58 if winner != "unknown" else 0.0
                confidence = float(max(0.0, min(1.0, confidence)))
                score_a_f, score_b_f = _scores_from_winner_and_confidence(winner, confidence)
            if winner == "unknown" and is_qwen3_vl:
                winner = "player_a"
                confidence = max(confidence, 0.50)
                score_a_f, score_b_f = _scores_from_winner_and_confidence(winner, confidence)
            if winner == "unknown":
                decision = "blocked"
            else:
                decision = "review"
            reason = ""

        return WinnerPrediction(
            winner=winner,
            confidence=confidence,
            decision=decision,
            score_a=score_a_f,
            score_b=score_b_f,
            reason=reason,
            raw=raw,
            thinking=thinking,
            model=self.model_name,
        )
