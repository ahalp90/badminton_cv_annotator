# Evidence: provenance and replays

## Source and input identity

Checked 2026-09-15 before any remote computation. MD5 of every shared module the experiment imports is identical between the local tree `L` and the remote root `R` (43 files: the independent-court package, `courtkeynet/court_corners.py`, the vp_pruning, marking_diagnosis, axis_matching and automatic_axes helpers, and `svd_fixed/run_svd_fixed.py`). The legacy `smoke/legacy/zone_net.py` matches the local copy under the 20260908 tree. The three input packs match their local copies. The run manifest (`runs/<run>/manifest.json.gz`) lists every hash, size and version.

Git: branch `fix/court-det`, HEAD `d0c9a12`, tracked tree clean at start. The packet's scientific checkpoint is `9fcec94`; the four later commits changed documentation only (`git diff --stat 9fcec94 HEAD -- experiments src` lists no Python files).

## Controls

| Case | Control record | Source string | Visually approved |
| --- | --- | --- | --- |
| gxBQ_window_00_frame_0 | `R/automatic_axes_20260914/gx0_control/bank_diagnosis.json.gz` | filled from record at E3 | filled at E3 |
| gxBQ_window_00_frame_5 | `R/automatic_axes_20260914/gx5_bank_diagnosis.json.gz` | | |
| am2_window_01_frame_28019 | `R/automatic_axes_20260914/am2_far_bank_diagnosis.json.gz` | | |
| other six | `R/axis_matching_20260914/given_finite/<case>.json.gz`, corners divided by native/working | | |

## Replays

Filled per stage from the saved records.
