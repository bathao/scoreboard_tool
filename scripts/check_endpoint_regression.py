from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.endpoint_regression import compare_endpoint_suite


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _suite_override_map(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --draft override: {item}")
        name, path = item.split("=", 1)
        overrides[name.strip()] = path.strip()
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description="Check endpoint regression targets against draft JSON files.")
    parser.add_argument(
        "--suite-file",
        default="matches/ground_truth/endpoint_regression_suite.json",
        help="Path to the endpoint regression suite JSON",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only the named suite (can be repeated)",
    )
    parser.add_argument(
        "--draft",
        action="append",
        default=[],
        help="Override draft path as suite_name=path (can be repeated)",
    )
    args = parser.parse_args()

    suite_file = Path(args.suite_file)
    suite_manifest = _load_json(suite_file)
    suite_specs = list(suite_manifest.get("suites", []))
    only = set(args.only)
    draft_overrides = _suite_override_map(args.draft)

    required_failed = False
    printed_any = False

    for suite_spec in suite_specs:
        suite_name = str(suite_spec["name"])
        if only and suite_name not in only:
            continue
        printed_any = True

        draft_path = Path(draft_overrides.get(suite_name, suite_spec["draft_json"]))
        if not draft_path.exists():
            print(f"[{suite_name}] missing draft: {draft_path}")
            if suite_spec.get("enforcement", "report_only") == "required":
                required_failed = True
            continue

        draft = _load_json(draft_path)
        result = compare_endpoint_suite(list(draft.get("points", [])), suite_spec)
        print(
            f"[{suite_name}] "
            f"ok={result['ok']} "
            f"pass={result['pass_count']}/{result['point_count']} "
            f"fail={result['fail_count']} "
            f"missing={result['missing_count']} "
            f"max_abs_diff={result['max_abs_diff_sec']:.3f}s "
            f"enforcement={result['enforcement']}"
        )
        for point_result in result["point_results"]:
            if point_result["status"] == "ok":
                continue
            if point_result["status"] == "missing":
                print(
                    f"  - {point_result['id']}: missing "
                    f"(expected {point_result['expected_t_end']:.3f}s)"
                )
                continue
            actual_mode = point_result.get("actual_mode")
            expected_mode = point_result.get("expected_mode")
            mode_msg = ""
            if expected_mode is not None and actual_mode != expected_mode:
                mode_msg = f", mode {actual_mode!r} != {expected_mode!r}"
            print(
                f"  - {point_result['id']}: "
                f"actual={point_result['actual_t_end']:.3f}s "
                f"target={point_result['expected_t_end']:.3f}s "
                f"diff={point_result['diff_sec']:+.3f}s "
                f"tol={point_result['tolerance_sec']:.3f}s{mode_msg}"
            )

        if result["enforcement"] == "required" and not result["ok"]:
            required_failed = True

    if not printed_any:
        print("No suites selected.")
        return 1

    return 1 if required_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
