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
    README.md
    finetune_dataset/
      README.md
      manifest.jsonl
    <collection_name>/
      train.jsonl
      val.jsonl
      test.jsonl
```

Recommended source-video layout outside the dataset itself:

```text
inputs/
  raw_matches/
    <match_id>__full.mp4
  debug_sets/
    <match_id>/
      set_01.mp4
      set_02.mp4
      ...
```

## Separation Of Concerns
- `dataset/reviewed_matches/`
  - canonical reviewed assets
  - the source of truth for human-labeled rally data
- `dataset/collections/`
  - train / val / test manifests
  - should reference reviewed assets, not duplicate them by default
- `dataset/collections/finetune_dataset/`
  - rolling active-learning collection
  - stores reviewed examples that are ready to feed future winner-model training
  - should reference canonical reviewed assets under `reviewed_matches/`
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

## Source Video Naming Rules
- Use the `match_id` as the anchor for all related assets:
  - raw full match
  - debug split sets
  - reviewed dataset
  - future render outputs
- Recommended full-match source filename:
  - `inputs/raw_matches/<match_id>__full.mp4`
  - example:
    - `inputs/raw_matches/match_vinh_001__full.mp4`
- Recommended debug split-set filenames:
  - store them under:
    - `inputs/debug_sets/<match_id>/`
  - use:
    - `set_01.mp4`
    - `set_02.mp4`
    - `set_03.mp4`
    - `set_04.mp4`
    - `set_05.mp4`
  - example:
    - `inputs/debug_sets/match_vinh_001/set_04.mp4`
- Avoid naming future files in the old ad-hoc style:
  - `inputs/raw_matches/match_vinh_001__full.mp4`
  - `inputs/debug_sets/match_vinh_001/set_01.mp4`
  - `inputs/debug_sets/match_vinh_001/set_02.mp4`
- Keep the original camera filename in metadata instead of forcing it into the canonical dataset id:
  - store it in:
    - `match_meta.json`
    - or another source manifest field such as `source_original_filename`

## Recommended Output Naming
- Reviewed dataset clips remain simple inside a reviewed set:
  - `clips/pt_0001.mp4`
- Future scoreboard renders should also anchor to `match_id`, for example:
  - `<match_id>__preview.mp4`
  - `<match_id>__final.mp4`
- Future training/eval manifests should reference:
  - `match_id`
  - `set_id`
  - `point_id`
  rather than depending only on raw filenames

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

## Taxonomy Consistency Rules
- Treat taxonomy as a canonical cross-dataset label set, not ad-hoc wording per set.
- If two rallies end by the same underlying losing event, they should reuse the same taxonomy label even if the surface description is different.
- Prefer reusing an existing taxonomy before creating a new one.
- Use the `note` field for point-specific nuance:
  - net clip
  - edge-like feel
  - dead tail
  - unusual body contact
  - operator wording
  instead of inventing a near-duplicate taxonomy.
- Only create a new taxonomy when the rally-ending event is materially different from all existing labels.
- When a new taxonomy is introduced:
  - document it in the active taxonomy list
  - explain why existing labels were not enough
  - apply it consistently to all future reviewed datasets

Current canonical winner-ending taxonomy set:
- `clean_winner_no_touch`
- `touched_but_out`
- `touched_but_no_net_cross`
- `attacker_direct_out`
- `attacker_into_net`
- `double_bounce_before_return`
- `ball_hits_player_or_body`
- `ball_hits_non_racket_object`
- `illegal_or_mishit_return`
- `blocked_by_visibility`
- `ambiguous_review`

Canonical mapping guidance:
- `opponent touched the ball but sent it out`:
  - `touched_but_out`
- `opponent touched the ball but the return did not cross the net`:
  - `touched_but_no_net_cross`
- `the last attacker directly hit out`:
  - `attacker_direct_out`
- `the last attacker directly hit into the net / failed over the net on the attack`:
  - `attacker_into_net`
- `winner hit a legal shot and the loser never touched it`:
  - `clean_winner_no_touch`

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
- reserve:
  - `dataset/collections/finetune_dataset/`
  as the rolling collection fed by the review UI for future winner-model `SFT`

## Active Learning Loop
The intended long-term winner-data loop is:

1. Run the current pipeline on a new match.
2. Review rally winners in the Web UI.
3. For each reviewed rally:
   - if the AI result is correct, keep it with one click
   - if the AI result is wrong, fix it with one click
4. After review, automatically save the canonical reviewed result into:
   - `dataset/reviewed_matches/<match_id>/set_<nn>/`
   - and enroll the reviewed rally into:
     - `dataset/collections/finetune_dataset/manifest.jsonl`
5. When the rolling fine-tune collection reaches about:
   - `200-500` reviewed rally examples
   a future `Train Now` flow may launch local winner-model `SFT` / adapter tuning.
6. The newly adapted winner model becomes the next pre-labeler for later matches.

Important:
- `reviewed_matches/` stays the canonical source of truth
- `finetune_dataset/` is the rolling training collection derived from those canonical reviewed assets
- keep held-out reviewed sets outside the training collection for honest evaluation

## Horizontal Flip Augmentation Rules
- Horizontal flip may be used as a training augmentation.
- Horizontal flip does **not** create a new reviewed rally.
- Count it as:
  - one additional training view
  - not one additional canonical reviewed label
- For this table-tennis project, the canonical supervision fields stay unchanged under `flip_h`:
  - `winner`
  - `loser`
  - `taxonomy`
  - `last_hitter`
- Reason:
  - `player_a` and `player_b` are role labels anchored to near-side / far-side semantics
  - horizontal flip changes image left/right only
  - it does not swap near/far roles
- What may need adjustment under `flip_h`:
  - free-text notes that mention:
    - `left`
    - `right`
    - `left side`
    - `right side`
    - similar image-space wording
- Preferred labeling rule:
  - avoid left/right wording in canonical notes when possible
  - prefer:
    - `player_a`
    - `player_b`
    - `near`
    - `far`
- If a flip-specific manifest stores text fields, image-space left/right wording should be swapped for the flipped variant.
- Keep train / val / test splits grouped by canonical `record_id`:
  - original and flipped variants of the same rally must stay in the same split
  - never let original land in train while flipped lands in val/test

## First Training Pilot Rules
- The current reviewed dataset seed may be used to start the first local winner-model training pilot when it has:
  - `71` unique reviewed rallies
  - plus `flip_h` training augmentation
- For the current project state, that means:
  - `71` canonical reviewed rallies
  - `71` flipped training views
  - `142` total training views in the rolling fine-tune manifest
- Treat this first run as:
  - `LoRA / QLoRA` pilot validation
  - not a final production model milestone
- Preferred first supervision target:
  - strict JSON output with:
    - `winner`
    - `loser`
    - `taxonomy`
    - `last_hitter`
- Preferred first split rule:
  - split by canonical `record_id`
  - keep all view variants of the same rally in the same split
- Preferred first evaluation rule:
  - compare the adapted model on held-out reviewed rallies
  - against the current prompt-only winner baseline
- Do not describe `flip_h` as new gold data:
  - it is augmentation only
  - the count of unique reviewed rallies remains unchanged

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

