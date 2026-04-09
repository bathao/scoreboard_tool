from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from winner_finetune_common import (
    build_training_prompt,
    load_manifest_rows,
    observed_taxonomies,
    parse_prediction_json,
    resolve_cached_clip_path,
    resolve_clip_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local Qwen3-VL winner adapter on held-out reviewed rallies.")
    parser.add_argument("--manifest", default="dataset/collections/finetune_dataset/splits/v1/test.jsonl")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument(
        "--cache-clips-dir",
        default="dataset/collections/finetune_dataset/cache/qwen3vl4b_4f384_v1",
        help="Optional directory containing prebuilt lightweight cache clips named by sample_id.",
    )
    parser.add_argument("--base-model-dir", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter-dir", default="models/adapters/qwen3vl4b_table_tennis_pilot")
    parser.add_argument("--skip-adapter", action="store_true", help="Evaluate the base prompt-only model without loading a LoRA adapter.")
    parser.add_argument("--out-jsonl", default="matches/checks/qwen3vl4b_table_tennis_pilot_test_predictions.jsonl")
    parser.add_argument("--summary-json", default="matches/checks/qwen3vl4b_table_tennis_pilot_test_summary.json")
    parser.add_argument(
        "--view-variant",
        default="original",
        help="Optional view_variant filter for eval rows; set empty string to evaluate all rows.",
    )
    parser.add_argument("--fps-sample", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--size-shortest-edge", type=int, default=448)
    parser.add_argument("--size-longest-edge", type=int, default=1048576)
    parser.add_argument("--max-pixels", type=int, default=589824)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest)
    dataset_root = Path(args.dataset_root)
    cache_clips_dir = Path(args.cache_clips_dir) if str(args.cache_clips_dir).strip() else None
    out_jsonl = Path(args.out_jsonl)
    summary_json = Path(args.summary_json)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows = load_manifest_rows(manifest_path)
    if str(args.view_variant).strip():
        rows = [row for row in rows if str(row.get("view_variant", "")).strip() == str(args.view_variant).strip()]
    active_taxonomies = observed_taxonomies(rows)
    prompt_text = build_training_prompt(active_taxonomies)

    processor = AutoProcessor.from_pretrained(args.base_model_dir)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map="auto",
    )
    model = base_model
    if (not bool(args.skip_adapter)) and str(args.adapter_dir).strip():
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    results: list[dict[str, object]] = []
    field_correct = Counter()

    for row in rows:
        clip_path = resolve_clip_path(dataset_root, row)
        if cache_clips_dir is not None:
            cached_path = resolve_cached_clip_path(cache_clips_dir, row)
            if cached_path.exists():
                clip_path = cached_path
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(clip_path.resolve())},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processor_kwargs = {
            "text": [text],
            "videos": [str(clip_path.resolve())],
            "return_tensors": "pt",
            "fps": float(args.fps_sample),
            "min_frames": int(args.min_frames),
            "max_frames": int(args.max_frames),
            "size": {
                "shortest_edge": int(args.size_shortest_edge),
                "longest_edge": int(args.size_longest_edge),
            },
        }
        if int(args.max_pixels) > 0:
            processor_kwargs["max_pixels"] = int(args.max_pixels)
        inputs = processor(**processor_kwargs)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=int(args.max_new_tokens), do_sample=False)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        parsed = parse_prediction_json(output_text)

        record = {
            "sample_id": row["sample_id"],
            "record_id": row["record_id"],
            "view_variant": row["view_variant"],
            "winner_gold": row["winner"],
            "loser_gold": row["loser"],
            "taxonomy_gold": row["taxonomy"],
            "last_hitter_gold": row["last_hitter"],
            "winner_pred": parsed.get("winner", ""),
            "loser_pred": parsed.get("loser", ""),
            "taxonomy_pred": parsed.get("taxonomy", ""),
            "last_hitter_pred": parsed.get("last_hitter", ""),
            "raw_output": output_text,
        }
        results.append(record)

        field_correct["winner_total"] += 1
        field_correct["loser_total"] += 1
        field_correct["taxonomy_total"] += 1
        field_correct["last_hitter_total"] += 1
        if str(record["winner_pred"]) == str(record["winner_gold"]):
            field_correct["winner_correct"] += 1
        if str(record["loser_pred"]) == str(record["loser_gold"]):
            field_correct["loser_correct"] += 1
        if str(record["taxonomy_pred"]) == str(record["taxonomy_gold"]):
            field_correct["taxonomy_correct"] += 1
        if str(record["last_hitter_pred"]) == str(record["last_hitter_gold"]):
            field_correct["last_hitter_correct"] += 1

    with out_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary = {
        "schema": "winner_lora_eval_summary_v1",
        "manifest": str(manifest_path).replace("\\", "/"),
        "adapter_dir": (
            ""
            if bool(args.skip_adapter) or (not str(args.adapter_dir).strip())
            else str(Path(args.adapter_dir)).replace("\\", "/")
        ),
        "cache_clips_dir": (str(cache_clips_dir).replace("\\", "/") if cache_clips_dir is not None else ""),
        "skip_adapter": bool(args.skip_adapter),
        "view_variant": str(args.view_variant),
        "sample_count": len(results),
        "winner_accuracy": field_correct["winner_correct"] / max(1, field_correct["winner_total"]),
        "loser_accuracy": field_correct["loser_correct"] / max(1, field_correct["loser_total"]),
        "taxonomy_accuracy": field_correct["taxonomy_correct"] / max(1, field_correct["taxonomy_total"]),
        "last_hitter_accuracy": field_correct["last_hitter_correct"] / max(1, field_correct["last_hitter_total"]),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
