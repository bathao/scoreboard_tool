from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def _load_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = str(line).strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _group_rows_by_record(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["record_id"])].append(row)
    return dict(grouped)


def _bucket_key(record_rows: list[dict[str, object]]) -> tuple[str, str]:
    first = record_rows[0]
    return (str(first.get("taxonomy", "")), str(first.get("winner", "")))


def _compute_split_counts(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    test_count = int(round(n * test_ratio))
    val_count = int(round(n * val_ratio))
    if n >= 6 and test_count == 0:
        test_count = 1
    if n >= 6 and val_count == 0:
        val_count = 1
    while test_count + val_count >= n:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break
    train_count = n - val_count - test_count
    return train_count, val_count, test_count


def _build_split_assignments(
    grouped_rows: dict[str, list[dict[str, object]]],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    rng = random.Random(seed)
    bucket_to_records: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record_id, rows in grouped_rows.items():
        bucket_to_records[_bucket_key(rows)].append(record_id)

    assignments: dict[str, str] = {}
    for _, record_ids in sorted(bucket_to_records.items()):
        shuffled = list(record_ids)
        rng.shuffle(shuffled)
        train_count, val_count, test_count = _compute_split_counts(len(shuffled), val_ratio, test_ratio)
        train_ids = shuffled[:train_count]
        val_ids = shuffled[train_count : train_count + val_count]
        test_ids = shuffled[train_count + val_count : train_count + val_count + test_count]
        for record_id in train_ids:
            assignments[record_id] = "train"
        for record_id in val_ids:
            assignments[record_id] = "val"
        for record_id in test_ids:
            assignments[record_id] = "test"
    return assignments


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    unique_records = {str(row["record_id"]) for row in rows}
    taxonomy_counts = Counter(str(row.get("taxonomy", "")) for row in rows if str(row.get("view_variant", "")) == "original")
    winner_counts = Counter(str(row.get("winner", "")) for row in rows if str(row.get("view_variant", "")) == "original")
    view_variant_counts = Counter(str(row.get("view_variant", "")) for row in rows)
    return {
        "sample_count": len(rows),
        "unique_record_count": len(unique_records),
        "view_variant_counts": dict(sorted(view_variant_counts.items())),
        "winner_counts_from_original_views": dict(sorted(winner_counts.items())),
        "taxonomy_counts_from_original_views": dict(sorted(taxonomy_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create grouped train/val/test splits for the winner fine-tune dataset.")
    parser.add_argument(
        "--manifest",
        default="dataset/collections/finetune_dataset/manifest.jsonl",
        help="Rolling fine-tune manifest JSONL.",
    )
    parser.add_argument(
        "--out-dir",
        default="dataset/collections/finetune_dataset/splits/v1",
        help="Output directory for train/val/test JSONL splits.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--val-ratio", type=float, default=0.12, help="Validation ratio over unique record_ids.")
    parser.add_argument("--test-ratio", type=float, default=0.12, help="Test ratio over unique record_ids.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    rows = _load_manifest(manifest_path)
    grouped = _group_rows_by_record(rows)
    assignments = _build_split_assignments(
        grouped,
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )

    split_rows: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split_name = assignments[str(row["record_id"])]
        split_rows[split_name].append(row)

    for split_name, rows_for_split in split_rows.items():
        _write_jsonl(out_dir / f"{split_name}.jsonl", rows_for_split)

    summary = {
        "schema": "winner_finetune_split_summary_v1",
        "source_manifest": str(manifest_path).replace("\\", "/"),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "splits": {name: _summarize_rows(rows_for_split) for name, rows_for_split in split_rows.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(f"Created grouped splits in {out_dir}")
    for split_name in ("train", "val", "test"):
        info = summary["splits"][split_name]
        print(
            f"{split_name}: records={info['unique_record_count']} samples={info['sample_count']} "
            f"views={info['view_variant_counts']}"
        )


if __name__ == "__main__":
    main()
