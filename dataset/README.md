# Dataset Layout

Root-level reviewed data and future train/eval collections live here.

Recommended structure:
- `reviewed_matches/`
  - canonical reviewed assets grouped by `match_id` and `set_id`
- `collections/`
  - train / val / test manifests that reference reviewed assets
- `registry.json`
  - lightweight index of reviewed matches currently tracked

Do not store reviewed training/eval assets under pipeline output folders such as:
- `matches/`
- `debug_report/`
