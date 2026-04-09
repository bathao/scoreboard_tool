from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments

from winner_finetune_common import (
    build_target_json,
    build_training_prompt,
    load_manifest_rows,
    observed_taxonomies,
    resolve_cached_clip_path,
    resolve_clip_path,
)


class WinnerManifestDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, object]],
        dataset_root: Path,
        prompt_text: str,
        cache_dir: Path | None = None,
    ) -> None:
        self.rows = rows
        self.dataset_root = dataset_root
        self.prompt_text = prompt_text
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        clip_path = resolve_clip_path(self.dataset_root, row)
        if self.cache_dir is not None:
            cached_path = resolve_cached_clip_path(self.cache_dir, row)
            if cached_path.exists():
                clip_path = cached_path
        return {
            "sample_id": str(row["sample_id"]),
            "record_id": str(row["record_id"]),
            "clip_path": str(clip_path),
            "prompt_text": self.prompt_text,
            "target_text": build_target_json(row),
        }


@dataclass
class WinnerVideoDataCollator:
    processor: AutoProcessor
    fps_sample: float
    min_frames: int
    max_frames: int
    size_shortest_edge: int
    size_longest_edge: int
    max_pixels: int

    def _messages(self, clip_path: str, prompt_text: str, target_text: str | None) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(Path(clip_path).resolve())},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        if target_text is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": target_text}]})
        return messages

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        full_texts: list[str] = []
        prompt_only_texts: list[str] = []
        videos: list[str] = []

        for feature in features:
            clip_path = str(feature["clip_path"])
            prompt_text = str(feature["prompt_text"])
            target_text = str(feature["target_text"])

            full_messages = self._messages(clip_path, prompt_text, target_text)
            prompt_messages = self._messages(clip_path, prompt_text, None)

            full_texts.append(self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False))
            prompt_only_texts.append(self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True))
            videos.append(clip_path)

        processor_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "fps": float(self.fps_sample),
            "min_frames": int(self.min_frames),
            "max_frames": int(self.max_frames),
            "size": {
                "shortest_edge": int(self.size_shortest_edge),
                "longest_edge": int(self.size_longest_edge),
            },
        }
        if int(self.max_pixels) > 0:
            processor_kwargs["max_pixels"] = int(self.max_pixels)

        full_inputs = self.processor(text=full_texts, videos=videos, **processor_kwargs)
        prompt_inputs = self.processor(text=prompt_only_texts, videos=videos, **processor_kwargs)

        labels = full_inputs["input_ids"].clone()
        labels[full_inputs["attention_mask"] == 0] = -100

        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for idx, prompt_len in enumerate(prompt_lengths):
            labels[idx, : min(int(prompt_len), labels.shape[1])] = -100

        batch = {k: v for k, v in full_inputs.items()}
        batch["labels"] = labels
        return batch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local LoRA adapter for Qwen3-VL winner prediction.")
    parser.add_argument("--train-manifest", default="dataset/collections/finetune_dataset/splits/v1/train.jsonl")
    parser.add_argument("--val-manifest", default="dataset/collections/finetune_dataset/splits/v1/val.jsonl")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument(
        "--cache-clips-dir",
        default="dataset/collections/finetune_dataset/cache/qwen3vl4b_4f384_v1",
        help="Optional directory containing prebuilt lightweight cache clips named by sample_id.",
    )
    parser.add_argument("--model-dir", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-dir", default="models/adapters/qwen3vl4b_table_tennis_pilot")
    parser.add_argument(
        "--eval-view-variant",
        default="original",
        help="Optional view_variant filter for eval rows; set empty string to evaluate all rows.",
    )
    parser.add_argument("--fps-sample", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--size-shortest-edge", type=int, default=448)
    parser.add_argument("--size-longest-edge", type=int, default=1048576)
    parser.add_argument("--max-pixels", type=int, default=589824)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument(
        "--eval-strategy",
        choices=["no", "steps", "epoch"],
        default="epoch",
        help="Evaluation schedule for pilot training.",
    )
    parser.add_argument(
        "--save-strategy",
        choices=["steps", "epoch"],
        default="epoch",
        help="Checkpoint save schedule for pilot training.",
    )
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-persistent-workers", action="store_true")
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_manifest = Path(args.train_manifest)
    val_manifest = Path(args.val_manifest)
    dataset_root = Path(args.dataset_root)
    cache_clips_dir = Path(args.cache_clips_dir) if str(args.cache_clips_dir).strip() else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_manifest_rows(train_manifest)
    val_rows = load_manifest_rows(val_manifest) if val_manifest.exists() else []
    if str(args.eval_view_variant).strip():
        val_rows = [row for row in val_rows if str(row.get("view_variant", "")).strip() == str(args.eval_view_variant).strip()]
    active_taxonomies = observed_taxonomies(train_rows + val_rows)
    prompt_text = build_training_prompt(active_taxonomies)

    train_dataset = WinnerManifestDataset(
        train_rows,
        dataset_root=dataset_root,
        prompt_text=prompt_text,
        cache_dir=cache_clips_dir,
    )
    eval_dataset = (
        WinnerManifestDataset(
            val_rows,
            dataset_root=dataset_root,
            prompt_text=prompt_text,
            cache_dir=cache_clips_dir,
        )
        if val_rows
        else None
    )

    processor = AutoProcessor.from_pretrained(args.model_dir)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=torch_dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = [part.strip() for part in str(args.lora_target_modules).split(",") if str(part).strip()]
    lora_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = WinnerVideoDataCollator(
        processor=processor,
        fps_sample=float(args.fps_sample),
        min_frames=int(args.min_frames),
        max_frames=int(args.max_frames),
        size_shortest_edge=int(args.size_shortest_edge),
        size_longest_edge=int(args.size_longest_edge),
        max_pixels=int(args.max_pixels),
    )

    effective_eval_strategy = (str(args.eval_strategy) if eval_dataset is not None else "no")
    enable_best_model = bool(eval_dataset) and effective_eval_strategy != "no" and str(args.save_strategy) == effective_eval_strategy

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=bool(eval_dataset),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        num_train_epochs=float(args.num_train_epochs),
        max_steps=int(args.max_steps),
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=int(args.logging_steps),
        logging_first_step=True,
        eval_strategy=effective_eval_strategy,
        eval_steps=int(args.eval_steps),
        save_strategy=str(args.save_strategy),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        save_only_model=True,
        dataloader_num_workers=int(args.dataloader_num_workers),
        dataloader_persistent_workers=(bool(args.dataloader_persistent_workers) and int(args.dataloader_num_workers) > 0),
        dataloader_prefetch_factor=(int(args.dataloader_prefetch_factor) if int(args.dataloader_num_workers) > 0 else None),
        run_name=output_dir.name,
        seed=int(args.seed),
        load_best_model_at_end=enable_best_model,
        metric_for_best_model=("eval_loss" if enable_best_model else None),
        greater_is_better=(False if enable_best_model else None),
        bf16_full_eval=torch.cuda.is_available(),
        skip_memory_metrics=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )

    summary = {
        "schema": "winner_lora_train_run_v1",
        "train_manifest": str(train_manifest).replace("\\", "/"),
        "val_manifest": str(val_manifest).replace("\\", "/") if val_manifest.exists() else "",
        "model_dir": str(Path(args.model_dir)).replace("\\", "/"),
        "output_dir": str(output_dir).replace("\\", "/"),
        "cache_clips_dir": (str(cache_clips_dir).replace("\\", "/") if cache_clips_dir is not None else ""),
        "train_sample_count": len(train_rows),
        "val_sample_count": len(val_rows),
        "eval_view_variant": str(args.eval_view_variant),
        "effective_eval_strategy": effective_eval_strategy,
        "enable_best_model": enable_best_model,
        "active_taxonomies": active_taxonomies,
        "video_config": {
            "fps_sample": float(args.fps_sample),
            "min_frames": int(args.min_frames),
            "max_frames": int(args.max_frames),
            "size_shortest_edge": int(args.size_shortest_edge),
            "size_longest_edge": int(args.size_longest_edge),
            "max_pixels": int(args.max_pixels),
        },
        "lora": {
            "r": int(args.lora_r),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "target_modules": target_modules,
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    train_result = trainer.train()
    trainer.save_model()
    processor.save_pretrained(output_dir)

    metrics = dict(train_result.metrics)
    metrics["log_history_length"] = len(getattr(trainer.state, "log_history", []))

    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    (output_dir / "log_history.json").write_text(
        json.dumps(getattr(trainer.state, "log_history", []), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
