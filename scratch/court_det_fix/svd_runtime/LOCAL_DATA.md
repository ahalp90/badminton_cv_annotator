# Retained SVD timing data

The nine-case timing experiment completed on Carmack on 23 September 2026.
[RESULTS.md](../archive/20260926/experiments/svd_runtime/RESULTS.md), `results.json.gz`, `history_check.json.gz`,
`integration_smoke.json.gz`, and [WORKLOG.md](../archive/20260926/experiments/svd_runtime/WORKLOG.md) are tracked summaries.
The small `receipts/`, `partial_history.json.gz` and `saved_input_check.json.gz`
records are also tracked. The partial history receipt is superseded by the
complete history result, but remains useful run history.

| Local-only path | Producer and purpose |
| --- | --- |
| `full/` (95.68 MiB) | `run_carmack.sh` runs `run_benchmark.py`; raw per-case matching, timing and summary records |
| `integration_smoke_raw/` (3.16 MiB) | `smoke_integration.py` writes the three raw integration comparisons at this fixed path |
| `smoke/` (0.45 MiB) | Pair-limited `run_benchmark.py` preparation checks; full findings remain in WORKLOG |

These exact directories are ignored by Git and remain at their original paths.
They are preserved evidence, not declared disposable or bit-identically
regenerable. The original Carmack run root was
`/scratch/ahalperi/court_det_fix/svd_runtime_20260923`; local copies are complete.
Remote work must follow `~/.codex/remote_hpc.md`.

A fresh clone contains the compact results but not these raw directories. Copy
them explicitly from the retained checkout/archive when needed; do not rerun the
completed benchmark merely to reorient. Existing CLI consumers use their paths.
For example, to rebuild the compact timing summary without rerunning matching:

```bash
~/.venvs/badminton-cicd/bin/python \
  scratch/court_det_fix/svd_runtime/summarise.py \
  --input scratch/court_det_fix/svd_runtime/full \
  --output /tmp/court-svd-summary.json.gz
```

The [23 September inventory](../archive/20260923_top_level/untracked_before.json.gz)
records filenames, sizes and hashes before the storage-policy refresh.
