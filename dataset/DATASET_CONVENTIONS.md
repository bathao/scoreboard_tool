# Dataset Conventions

This file defines how reviewed rally datasets should be stored and extended over time.

## Purpose
- keep reviewed data reusable for:
  - benchmark
  - prompt few-shot
  - retrieval experiments
  - future model training / evaluation
- keep reviewed data separate from temporary pipeline outputs

## Root Layout
Use this structure under `dataset/`:

```text
dataset/
  README.md
  DATASET_CONVENTIONS.md
  registry.json
  reviewed_matches/
    <match_id>/
      match_meta.json
      set_01/
        README.md
        labels.jsonl
        fewshot_seed.jsonl
        clips/
          pt_0001.mp4
      set_02/
      ...
  collections/
    <collection_name>/
      train.jsonl
      val.jsonl
      test.jsonl
```

## Separation Of Concerns
- `dataset/reviewed_matches/`
  - canonical reviewed assets
  - the source of truth for human-labeled rally data
- `dataset/collections/`
  - train / val / test manifests
  - should reference reviewed assets, not duplicate them by default
- `matches/`
  - pipeline outputs and regression artifacts
  - not the long-term home for reviewed training/eval data
- `debug_report/`
  - temporary review exports
  - not canonical dataset storage

## Naming Rules
- `match_id`
  - stable and globally unique within the repo
  - format:
    - `match_<short_name>_<nnn>`
  - example:
    - `match_vinh_001`
- `set_id`
  - fixed-width set label
  - format:
    - `set_01`
    - `set_02`
    - `set_03`
    - `set_04`
    - `set_05`
- `point_id`
  - local inside a set
  - format:
    - `pt_0001`
- `record_id`
  - global reviewed rally id
  - format:
    - `<match_id>__<set_id>__<point_id>`
  - example:
    - `match_vinh_001__set_04__pt_0001`

## Required Files Per Reviewed Set
- `labels.jsonl`
  - full reviewed labels for all rallies in the set
- `fewshot_seed.jsonl`
  - curated reviewed subset for prompt-time few-shot experiments
- `clips/`
  - full frozen rally clips
- `README.md`
  - local notes for that reviewed set bundle

## Required Files Per Reviewed Match
- `match_meta.json`
  - match-level metadata and source-video notes

## Clip Rules
- use full frozen rally clips, not temporary model-view crops
- clip naming inside a set bundle should stay simple:
  - `clips/pt_0001.mp4`
- clip-to-label mapping should be explicit in JSONL:
  - `clip_relpath = clips/pt_0001.mp4`

## Labels Schema
Each `labels.jsonl` row should include at least:
- `schema`
- `match_id`
- `set_id`
- `point_id`
- `record_id`
- `clip_relpath`
- `source_video_relpath`
- `t_start`
- `t_end`
- `winner`
- `loser`
- `taxonomy`
- `last_hitter`
- `winner_label_status`
- `taxonomy_label_status`
- `source`
- `note`

## Status Rules
- `winner_label_status`
  - `confirmed` or `pending`
- `taxonomy_label_status`
  - `confirmed` or `pending`
- do not silently mix reviewed and guessed labels without a status field

## Few-Shot Seed Rules
- `fewshot_seed.jsonl` should be a curated subset, not automatically every reviewed point
- prefer coverage across:
  - taxonomy types
  - near-win and far-win
  - different visible endings
- when possible, avoid duplicating too many highly similar examples

## Collections Rules
- `collections/` manifests should reference reviewed assets using:
  - `record_id`
  - `clip_relpath` or absolute-resolvable dataset path
- do not copy video files into each collection unless there is a real need
- prefer manifest-based splits first

## Versioning Rules
- reviewed data should be append-only or explicitly versioned
- when a set is relabeled materially, create a new reviewed bundle version or record the change clearly
- avoid overwriting historical labels without a note

## Adding A New Reviewed Match
1. Create:
   - `dataset/reviewed_matches/<match_id>/`
2. Add:
   - `match_meta.json`
3. For each reviewed set, create:
   - `set_nn/labels.jsonl`
   - `set_nn/fewshot_seed.jsonl`
   - `set_nn/clips/`
   - `set_nn/README.md`
4. Update:
   - `dataset/registry.json`

## Do Not
- do not store reviewed training/eval assets only under `matches/`
- do not depend on `debug_report/` as canonical storage
- do not use temporary experiment names as long-term dataset ids
- do not hide global identity only in folder names; keep it in `record_id`
