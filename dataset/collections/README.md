# Dataset Collections

This folder holds manifest-style dataset collections derived from the canonical reviewed assets under:
- `dataset/reviewed_matches/`

Typical uses:
- train / val / test splits
- retrieval subsets
- prompt few-shot subsets
- rolling fine-tune queues

Important:
- do not treat `collections/` as the source of truth for reviewed labels
- canonical reviewed clips and labels must remain under:
  - `dataset/reviewed_matches/`
