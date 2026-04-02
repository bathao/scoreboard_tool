from __future__ import annotations

from typing import Any


def compare_endpoint_suite(
    draft_points: list[dict[str, Any]],
    suite_spec: dict[str, Any],
) -> dict[str, Any]:
    point_by_id = {
        str(point["id"]): point
        for point in draft_points
        if isinstance(point, dict) and point.get("id")
    }

    default_tolerance = float(suite_spec.get("default_tolerance_sec", 0.25))
    point_results: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    pass_count = 0
    fail_count = 0
    missing_count = 0

    for target in suite_spec.get("points", []):
        point_id = str(target["id"])
        expected_t_end = float(target["t_end"])
        tolerance = float(target.get("tolerance_sec", default_tolerance))
        expected_mode = target.get("endpoint_mode")
        actual = point_by_id.get(point_id)

        if actual is None:
            point_results.append(
                {
                    "id": point_id,
                    "status": "missing",
                    "expected_t_end": expected_t_end,
                    "tolerance_sec": tolerance,
                }
            )
            fail_count += 1
            missing_count += 1
            continue

        actual_t_end = float(actual["t_end"])
        diff_sec = actual_t_end - expected_t_end
        abs_diff_sec = abs(diff_sec)
        max_abs_diff = max(max_abs_diff, abs_diff_sec)
        actual_mode = actual.get("endpoint_mode")

        ok = abs_diff_sec <= tolerance
        mode_ok = True
        if expected_mode is not None:
            mode_ok = actual_mode == expected_mode
            ok = ok and mode_ok

        point_results.append(
            {
                "id": point_id,
                "status": "ok" if ok else "diff",
                "expected_t_end": expected_t_end,
                "actual_t_end": actual_t_end,
                "diff_sec": diff_sec,
                "abs_diff_sec": abs_diff_sec,
                "tolerance_sec": tolerance,
                "expected_mode": expected_mode,
                "actual_mode": actual_mode,
                "mode_ok": mode_ok,
            }
        )
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    return {
        "name": suite_spec.get("name"),
        "enforcement": suite_spec.get("enforcement", "report_only"),
        "point_count": len(point_results),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "missing_count": missing_count,
        "max_abs_diff_sec": max_abs_diff,
        "ok": fail_count == 0,
        "point_results": point_results,
    }
