from backend.ai_ollama_client import _extract_json_block, _normalize_decision, _normalize_winner


def test_extract_json_block_parses_embedded_object():
    payload = 'noise {"winner":"player_b","confidence":0.72,"decision":"review","reason":"late miss"} tail'
    parsed = _extract_json_block(payload)

    assert parsed["winner"] == "player_b"
    assert parsed["confidence"] == 0.72


def test_normalize_winner_accepts_near_far_aliases():
    assert _normalize_winner("near") == "player_a"
    assert _normalize_winner("far-side") == "player_b"
    assert _normalize_winner("unclear") == "unknown"


def test_normalize_decision_blocks_unknown_candidate():
    assert _normalize_decision("unknown", 0.95, "auto") == "blocked"
    assert _normalize_decision("player_a", 0.62, "review") == "review"
    assert _normalize_decision("player_b", 0.91, "auto") == "auto"
