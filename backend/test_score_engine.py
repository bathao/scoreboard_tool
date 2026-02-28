from score_engine import (
    MatchEngine,
    MatchAlreadyFinishedError,
    InvalidWinnerCodeError,
    MatchNotFinishedError
)


def run_test(name, data, expected_exception=None):
    print(f"Running test: {name}")

    try:
        engine = MatchEngine(data)
        engine.process()

        if expected_exception:
            print("  ❌ FAILED (expected exception not raised)\n")
        else:
            print("  ✅ PASSED\n")

    except Exception as e:
        if expected_exception and isinstance(e, expected_exception):
            print(f"  ✅ PASSED ({expected_exception.__name__} caught)\n")
        else:
            print(f"  ❌ FAILED (unexpected exception: {e})\n")


if __name__ == "__main__":

    # --- VALID MATCH (A wins 3-0) ---
    valid_match = {
        "schema_version": "1.0",
        "match_id": "valid-1",
        "best_of": 5,
        "players": {"A": "P1", "B": "P2"},
        "points": ["A"] * 11 +
                  ["A"] * 11 +
                  ["A"] * 11
    }

    # --- EXTRA POINT AFTER MATCH FINISHED ---
    extra_point_match = {
        "schema_version": "1.0",
        "match_id": "invalid-1",
        "best_of": 5,
        "players": {"A": "P1", "B": "P2"},
        "points": ["A"] * 11 +
                  ["A"] * 11 +
                  ["A"] * 11 +
                  ["A"]
    }

    # --- INVALID POINT CODE ---
    invalid_point_code = {
        "schema_version": "1.0",
        "match_id": "invalid-2",
        "best_of": 5,
        "players": {"A": "P1", "B": "P2"},
        "points": ["A"] * 10 + ["C"]
    }

    # --- MATCH NOT FINISHED ---
    unfinished_match = {
        "schema_version": "1.0",
        "match_id": "invalid-3",
        "best_of": 5,
        "players": {"A": "P1", "B": "P2"},
        "points": ["A"] * 11
    }

    run_test("Valid match", valid_match, None)
    run_test("Extra point after finish", extra_point_match, MatchAlreadyFinishedError)
    run_test("Invalid point code", invalid_point_code, InvalidWinnerCodeError)
    run_test("Unfinished match", unfinished_match, MatchNotFinishedError)