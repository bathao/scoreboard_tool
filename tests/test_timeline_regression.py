from backend.timeline_regression import compare_timeline_suite


def test_compare_timeline_suite_reports_ok_and_diff():
    timeline_points = [
        {"id": "pt_0001", "t_end": 7.48, "endpoint_mode": "dead_reset_run_start"},
        {"id": "pt_0002", "t_end": 12.80, "endpoint_mode": "dead_reset_run_start"},
    ]
    suite_spec = {
        "name": "demo",
        "enforcement": "required",
        "default_tolerance_sec": 0.05,
        "points": [
            {"id": "pt_0001", "t_end": 7.474, "endpoint_mode": "dead_reset_run_start"},
            {"id": "pt_0002", "t_end": 12.90, "endpoint_mode": "dead_reset_run_start"},
        ],
    }

    result = compare_timeline_suite(timeline_points, suite_spec)

    assert result["name"] == "demo"
    assert result["pass_count"] == 1
    assert result["fail_count"] == 1
    assert result["ok"] is False
    assert result["point_results"][0]["status"] == "ok"
    assert result["point_results"][1]["status"] == "diff"


def test_compare_timeline_suite_reports_missing():
    suite_spec = {
        "name": "demo",
        "points": [
            {"id": "pt_0001", "t_end": 7.474},
        ],
    }

    result = compare_timeline_suite([], suite_spec)

    assert result["missing_count"] == 1
    assert result["fail_count"] == 1
    assert result["point_results"][0]["status"] == "missing"
