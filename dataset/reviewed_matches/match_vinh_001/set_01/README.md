# Reviewed Set Bundle

Canonical reviewed data for:
- `match_id = match_vinh_001`
- `set_id = set_01`

Files:
- `labels.jsonl`
  - full reviewed labels for all rallies in this set
- `fewshot_seed.jsonl`
  - curated reviewed subset for prompt-time few-shot experiments
- `clips/`
  - full frozen rally clips

Key conventions:
- `point_id` is local within the set:
  - `pt_0001`
- `record_id` is globally unique:
  - `match_vinh_001__set_01__pt_0001`
- `clip_relpath` points to the matching clip inside this set bundle
