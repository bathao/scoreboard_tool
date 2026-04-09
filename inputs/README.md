# Input Video Conventions

Recommended source-video layout for this repo:

```text
inputs/
  raw_matches/
    <match_id>__full.mp4
  debug_sets/
    <match_id>/
      set_01.mp4
      set_02.mp4
      set_03.mp4
      set_04.mp4
      set_05.mp4
```

Examples:
- `inputs/raw_matches/match_vinh_001__full.mp4`
- `inputs/debug_sets/match_vinh_001/set_04.mp4`

Why this convention helps:
- one stable `match_id` can connect:
  - raw input
  - debug set clips
  - reviewed dataset bundles
  - future scoreboard renders
  - training/eval manifests

Prefer this over older ad-hoc names such as:
- `inputs/raw_matches/match_vinh_001__full.mp4`
- `inputs/debug_sets/match_vinh_001/set_01.mp4`
- `inputs/debug_sets/match_vinh_001/set_02.mp4`

Keep the original camera/source filename in metadata, not as the canonical dataset id.

