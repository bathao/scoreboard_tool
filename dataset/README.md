# Dataset Layout

Root-level reviewed data and future train/eval collections live here.

See also:
- `DATASET_CONVENTIONS.md`
  - naming rules
  - folder layout
  - reviewed label schema
  - how to add more matches / sets

Recommended structure:
- `reviewed_matches/`
  - canonical reviewed assets grouped by `match_id` and `set_id`
- `collections/`
  - train / val / test manifests that reference reviewed assets
  - also hosts the rolling active-learning fine-tune queue:
    - `collections/finetune_dataset/`
- `registry.json`
  - lightweight index of reviewed matches currently tracked

Recommended source-video naming outside `dataset/`:
- raw full match:
  - `inputs/raw_matches/<match_id>__full.mp4`
- debug split sets:
  - `inputs/debug_sets/<match_id>/set_01.mp4`
  - `inputs/debug_sets/<match_id>/set_02.mp4`
  - ...

Intended long-term loop:
1. run the current pipeline
2. review winner predictions in the Web UI
3. keep correct rallies or fix wrong rallies with one click
4. auto-save those reviewed rallies into:
   - `reviewed_matches/` as canonical assets
   - `collections/finetune_dataset/` as the rolling training queue
5. once the queue reaches about `200-500` reviewed rallies, launch `Train Now`
6. use the newly adapted winner model to pre-label the next matches

Current dataset readiness:
- canonical reviewed dataset:
  - `71` unique rallies
- rolling fine-tune queue:
  - `142` training views
  - `71` original
  - `71` `flip_h`
- current intended next step:
  - start the first local `Qwen3-VL-4B` adapter-training pilot
  - keep train / val / test grouped by `record_id`
  - evaluate only on held-out reviewed rallies

Do not store reviewed training/eval assets under pipeline output folders such as:
- `matches/`
- `debug_report/`
