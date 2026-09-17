# W2 current pickup

## Bottom line

The bounded two-pair pixel export completed on 17 September 2026. It replayed four
fixed intervals for false Am2 `184:4123` and approved Am2 `30:33`. The pinned CSV
reproduced all 48 rows exactly. The output does not assign physical ownership.

## Evidence

- `w2_fresh_review/w2_pair_atlas/atlas.html`: raw and overlay views for both pairs
- `w2_fresh_review/w2_pair_atlas/184_4123/`: false-candidate trace and arrays
- `w2_fresh_review/w2_pair_atlas/30_33/`: approved-candidate trace and arrays
- `w2_fresh_review/w2_pair_atlas/manifest.json`: input identities and replay checks
- `w2_fresh_review/w2_pair_atlas/pair_summary.json`: four interval summaries

The manifest reports `complete`, zero contrast-replay differences, equal availability
masks and equal passing-offset masks. It records physical ownership as unresolved.

## Reproduction

From the repository root, use the approved CPU environment:

```bash
REPO=/absolute/path/to/checkout \
PYTHON=/absolute/path/to/environment/bin/python \
OUT=/absolute/path/to/checkout/scratch/court_det_fix/worklog/webui_further_followups_16092026/w2_handover/w2_fresh_review/w2_pair_atlas \
bash /absolute/path/to/checkout/scratch/court_det_fix/worklog/webui_further_followups_16092026/w2_handover/w2_fresh_review/run_followup.sh
```

The output directory must be new. The runner also creates a transient ZIP. The commit
keeps the unpacked evidence only.

## Limits

This is a diagnostic export. It does not prove court-paint ownership, create a scorer,
change a ruling or justify a threshold. The pre-export reasoning is in
`w2_fresh_review/REVIEW.md`.
