from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from winner_finetune_common import (
    ACTIVE_PILOT_TAXONOMY_ORDER,
    build_training_prompt,
    load_manifest_rows,
    observed_taxonomies,
    parse_prediction_json,
    resolve_cached_clip_path,
    resolve_clip_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run winner/taxonomy inference with a local Qwen3-VL table-tennis adapter.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Absolute or relative path to a single rally clip.")
    source.add_argument("--manifest", help="JSONL manifest containing clip_relpath rows.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument(
        "--cache-clips-dir",
        default="",
        help="Optional cache-clip directory. When provided for manifest rows, cached clips are preferred if present.",
    )
    parser.add_argument("--base-model-dir", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter-dir", default="models/adapters/qwen3vl4b_table_tennis_pilot_4ep_cache_v2")
    parser.add_argument("--skip-adapter", action="store_true", help="Run prompt-only base model without loading the adapter.")
    parser.add_argument(
        "--taxonomy-list",
        default="",
        help="Optional comma-separated taxonomy list. If omitted, use manifest-observed taxonomies or the active pilot list.",
    )
    parser.add_argument("--out-json", default="", help="Optional JSON output file for single-video mode.")
    parser.add_argument("--out-jsonl", default="matches/checks/qwen3vl4b_table_tennis_adapter_predictions.jsonl")
    parser.add_argument("--fps-sample", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--size-shortest-edge", type=int, default=384)
    parser.add_argument("--size-longest-edge", type=int, default=1048576)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def _active_taxonomies(args: argparse.Namespace, rows: list[dict[str, object]]) -> list[str]:
    if str(args.taxonomy_list).strip():
        return [part.strip() for part in str(args.taxonomy_list).split(",") if part.strip()]
    manifest_taxonomies = observed_taxonomies(rows)
    if manifest_taxonomies:
        return manifest_taxonomies
    return list(ACTIVE_PILOT_TAXONOMY_ORDER)


def _run_prediction(
    *,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    clip_path: Path,
    prompt_text: str,
    args: argparse.Namespace,
) -> tuple[dict[str, str], str]:
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
    return parsed, output_text


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    cache_clips_dir = Path(args.cache_clips_dir) if str(args.cache_clips_dir).strip() else None

    rows: list[dict[str, object]] = []
    single_video = None
    if str(args.manifest or "").strip():
        rows = load_manifest_rows(Path(args.manifest))
    else:
        single_video = Path(str(args.video)).resolve()
        rows = [{"sample_id": single_video.stem}]

    prompt_text = build_training_prompt(_active_taxonomies(args, rows))

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

    if single_video is not None:
        parsed, output_text = _run_prediction(model=model, processor=processor, clip_path=single_video, prompt_text=prompt_text, args=args)
        record = {
            "schema": "winner_adapter_single_prediction_v1",
            "video": str(single_video).replace("\\", "/"),
            "adapter_dir": ("" if bool(args.skip_adapter) else str(Path(args.adapter_dir)).replace("\\", "/")),
            "prediction": parsed,
            "raw_output": output_text,
        }
        if str(args.out_json).strip():
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(record, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, ensure_ascii=True, indent=2))
        return

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for row in rows:
        clip_path = resolve_clip_path(dataset_root, row)
        if cache_clips_dir is not None:
            cached_path = resolve_cached_clip_path(cache_clips_dir, row)
            if cached_path.exists():
                clip_path = cached_path
        parsed, output_text = _run_prediction(model=model, processor=processor, clip_path=clip_path, prompt_text=prompt_text, args=args)
        results.append(
            {
                "sample_id": row.get("sample_id", ""),
                "record_id": row.get("record_id", ""),
                "view_variant": row.get("view_variant", ""),
                "clip_path": str(clip_path).replace("\\", "/"),
                "winner_pred": parsed.get("winner", ""),
                "loser_pred": parsed.get("loser", ""),
                "taxonomy_pred": parsed.get("taxonomy", ""),
                "last_hitter_pred": parsed.get("last_hitter", ""),
                "raw_output": output_text,
            }
        )

    with out_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(
        json.dumps(
            {
                "schema": "winner_adapter_batch_prediction_v1",
                "manifest": str(Path(args.manifest)).replace("\\", "/"),
                "sample_count": len(results),
                "out_jsonl": str(out_jsonl).replace("\\", "/"),
                "adapter_dir": ("" if bool(args.skip_adapter) else str(Path(args.adapter_dir)).replace("\\", "/")),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
