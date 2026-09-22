# W5 directional admission: completed packet

All three arms completed on the fixed 27-case development panel. Each has
27 case records and successful pilot and gallery receipts. The saved
[comparison](w5_directional_20260921_r5_comparison/comparison.md) exactly
matches a fresh validation with the local comparator on 22 September 2026.
Visual rulings remain pending; no global admission floor has been selected.

| Floor (lengthwise, cross-court) | Saved arm |
| --- | --- |
| (3,3) | [Results](w5_directional_20260921_r5_33/result.md) |
| (4,3) | [Results](w5_directional_20260921_r5_43/result.md) |
| (5,3) | [Results](w5_directional_20260921_r5_53/result.md) |

Each arm retains its full arrays, compressed case records, gallery, manifest,
rankings, sensitivity records, logs and receipts. The packet was moved intact
from `scratch/court_det_fix/worklog/w5_carmack/`. Original absolute paths in
manifests and logs describe the producing checkout; they are provenance.
For `render_gallery.py`, pass an absolute arm-directory path as `--run` to
select a filed packet. Pass the investigation root as `--root` for its inputs.

## Completion and comparison

The [supervisor log](w5_directional_20260921_r5_supervisor.log) ends after all
three arms completed. The final comparison command rejected the configured
six workers because it required exactly ten. The existing local repair
accepts any positive worker count. With that repair, all three arms pass
the comparator's contract checks and reproduce the saved comparison JSON.
The supervisor log is preserved unchanged, including its failure message.

Filing checks also confirmed the same 27 case IDs in every arm's per-view
table, case records and arrays. All per-view rows report completion and
matching determinism checks. Every compressed JSON/CSV file passes `gzip -t`.

The [dispatch snapshots](dispatch/) preserve the local `run_three_arm_sweep.sh`,
`run_remote.sh` and repaired `compare_directional_runs.py` as they stood at
filing. These are local source snapshots, not a newly recovered remote hash
receipt. Arm manifests retain the recorded producer/helper provenance.
The earlier r4 launcher and monitoring records are in the preparation archive.

## Review still owed

Inspect every wrong or unclear view, prior-ruling conflict and regression
case excluded from the contact sheets at full-frame scale. Check the useful
GX5 admission and the earlier weak GX0 and Am2 winners against saved rulings.
Choose one global floor only if the evidence supports it; otherwise reject it.

The earlier handover also requires a Claude Opus 5 High review of the W5
result and final historical-impact accounting. That review has not been run
as part of filing. The [whole-court evidence account](../README.md) and
[box-provenance account](../box_provenance.md) retain the scientific context.

The preparation archive at `../../../.recovery/w5-preparation-20260922.tar.gz`
preserves the old campaign, worktree, handover, check-ins and earlier monitoring
records under their original repository-relative names. Remote data remains
untouched.
