# Rolling Fine-Tune Dataset

This folder is reserved for the rolling active-learning collection used for future winner-model training.

Intended flow:
1. run the current pipeline on a new match
2. review rally winners in the Web UI
3. keep correct AI picks or fix wrong picks with one click
4. auto-save the reviewed rally into:
   - `dataset/reviewed_matches/<match_id>/set_<nn>/`
   - and enroll it into this rolling fine-tune collection
5. once the collection reaches about `200-500` reviewed rally examples, launch local winner-model `SFT` / adapter tuning
6. use the newly adapted model as the next pre-labeler

Expected future contents:
- `manifest.jsonl`
  - rows that reference canonical reviewed assets
- optional split files or snapshots for specific training runs

Current pilot status:
- current rolling manifest size:
  - `142` training views
- current unique reviewed rally count behind that manifest:
  - `71`
- current intended next step:
  - start a first local `LoRA / QLoRA` pilot on `Qwen3-VL-4B`
  - keep splits grouped by `record_id`
  - use held-out reviewed rallies to compare against the prompt-only baseline

Current augmentation rule:
- `flip_h` is allowed as a training view
- it does not count as a new reviewed rally
- for this project, `flip_h` keeps:
  - `winner`
  - `loser`
  - `taxonomy`
  - `last_hitter`
  unchanged
- only image-space left/right wording in free-text fields should be adjusted when needed

Important:
- this folder should reference canonical reviewed data
- avoid copying duplicate clips here unless a specific training tool requires it
