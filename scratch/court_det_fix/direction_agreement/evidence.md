# Evidence: provenance and replays

## Source and input identity

Checked 2026-09-15 before any remote computation. MD5 of every shared module the experiment imports is identical between the local tree `L` and the remote root `R` (43 files: the independent-court package, `courtkeynet/court_corners.py`, the vp_pruning, marking_diagnosis, axis_matching and automatic_axes helpers, and `svd_fixed/run_svd_fixed.py`). The legacy `smoke/legacy/zone_net.py` matches the local copy under the 20260908 tree. The three input packs match their local copies. The run manifest (`runs/<run>/manifest.json.gz`) lists every hash, size, version and per-case image.

Git: branch `fix/court-det`, HEAD `d0c9a12`, tracked tree clean at start. The packet's scientific checkpoint is `9fcec94`; the four later commits changed documentation only (`git diff --stat 9fcec94 HEAD -- experiments src` lists no Python files).

Remote environment: Python 3.11.13, NumPy 2.4.6, OpenCV 5.0.0, SciPy 1.17.1, one OpenCV thread and single-thread BLAS in every stage script.

## Population

Nine cases, native 1920×1080 (GX and amateur) or 960×540 (ShuttleSet), all at working size 960×540. Saved settings match the packet's expected values in every case (1.5 degrees, 128-line cap, 16 retained directions, IoU 0.8, coverage selection). The merged line count is below the cap for GX0 (106), GX5 (108) and SS03-19 (124).

## Controls (read only at E3, after the E2 records were frozen)

| Case | Control record | Source string | Visually approved |
| --- | --- | --- | --- |
| GX0 | `R/automatic_axes_20260914/gx0_control/bank_diagnosis.json.gz` | `visually_approved_GX0_supplied_direction_line_winner_89` | yes |
| GX5 | `R/automatic_axes_20260914/gx5_bank_diagnosis.json.gz` | `visually_approved_GX5_generated_court` | yes |
| Am2-150 | `R/axis_matching_20260914/given_finite/am2_window_00_frame_150.json.gz` | `visually_approved_VP_diagnostic_court` | yes |
| Am2-28019 | `R/automatic_axes_20260914/am2_far_bank_diagnosis.json.gz` | `manual_reference_directions_only` | no |
| Am3-0 | `R/axis_matching_20260914/given_finite/am3_window_00_frame_0.json.gz` | `manual_reference_directions_only` | no |
| SS03-17 | `R/axis_matching_20260914/given_finite/shuttleset_03_scene_0017.json.gz` | `manual_reference_directions_only` | no |
| SS03-19 | `R/axis_matching_20260914/given_finite/shuttleset_03_scene_0019.json.gz` | `visually_approved_automatic_marking_refit` | yes |
| SS03-16 | `R/axis_matching_20260914/given_finite/shuttleset_03_scene_0016.json.gz` | `visually_approved_automatic_marking_refit` | yes |
| SS21-20 | `R/axis_matching_20260914/given_finite/shuttleset_21_scene_0020.json.gz` | `visually_approved_automatic_marking_refit` | yes |

The three bank-diagnosis controls are used in working pixels as saved. The other six divide the given record's native corners by the native/working scale. Each E3 record carries the control's corners, source string, record path and MD5.

## E0 replays (all nine cases, `runs/<run>/e0/<case>.json.gz`, field `checks`)

1. `merge_lines_with_membership` returns coefficients and order equal to `detector._merge_lines` (`np.array_equal`).
2. The first 128 merged lines equal the saved `direction_lines` (rtol 0, atol 1e-12).
3. `vp_pruning.estimate` replayed with the saved settings matches the saved estimator field by field: exact for IDs, counts, statuses, masks, provenance lists and integer counts; rtol 0, atol 1e-12 for lines, points and the transform.
4. `angular_residuals_at` with the foot anchor equals `vp_pruning.angular_residuals` on the full bank (`np.array_equal`).
5. Support counts equal the saved `support_counts`.
6. `retain_pencils_with_buckets` returns the same retained rows and statuses as `vp_pruning.retain_pencils`, and its retained IDs, masks, statuses and points equal the saved record.
7. Infinity candidates have identical residuals under both anchors.
8. Arm B's representatives equal the saved retained IDs.
9. The packet's E0 step 5: `automatic_axes_20260914/check_pool_replays.py` run from `R` printed nine "exact unfiltered selection replay passed" messages, exit 0 (`runs/<run>/logs/pool_replay.log`).

R sharing B's leaders and buckets, and MR sharing M's, hold by construction (both arms are built from the same allocation object), so they are not listed as evidence. The bank is rebuilt with `diagnose_direction_bank.reconstruct_bank`, which asserts the saved candidate IDs. Membership masks are `residuals <= 1.5` on the saved `.npy.xz` matrices and are not stored separately; member fragment lengths can be rebuilt from the E0 member IDs and the input pack.

## E3 replays

For GX0, GX5 and Am2-28019 the B and B+SVD fits reproduce `R/automatic_axes_20260914/svd_fixed/records/<case>.json.gz`: all 240 `max_corner_working_px` values per set differ by 0.0 (tolerance 1e-8), the SVD refit points agree within 1e-8, and the control corners and source strings match.

## E4 replays and reuse

All 18 case-arms (nine views under M and under R) were generated; no arm was empty and none reused the baseline by identity. For every case-arm, `rescore_camera_pool.rescore` ran with `replay=True` at both rescoring stages: it re-evaluated the saved final entries and required equality, and reconstructed the previous stage's selection from saved corners and required the same candidate IDs in the same order. That is four replay checks per case-arm, 72 in total; each stream log carries the corresponding "replay" lines (`runs/<run>/logs/e4_*.log`) and every stream exited 0.

The accounting (`diag`, exit 0) diagnosed all 81 case-arm stages with `diagnose_automatic.diagnose` against the frozen controls: no missing or empty stage, `basis_failed` 0 and `identity_reused` false in every row, 240 pairs attempted per case-arm. `runs/<run>/e4/diagnosis.json.gz` keeps each diagnosis with its control and record path; `accounting.csv.gz` is the flat table and `summary.md` its readable form. Corner errors come from `projective_seed.corner_errors` on native corners scaled to working pixels, as the packet specifies.

Checks made while writing the report: B's and M's SS21-20 all-camera winners have identical corners (maximum difference 0.0 native px, read from `e4/diagnosis.json.gz`). The local baseline all-camera copies the gallery joins against (`L/automatic_axes/collected/all_camera/`) are byte-identical by MD5 to the remote records the accounting read (`R/automatic_axes_20260914/all_camera/`) for all nine views. The one all-camera entry counted as camera-ineligible (Am2-28019 under M, candidate 74:4750) passed the prefilter at 0.065 but failed the geometry check, which leaves its final camera error unset. `code_md5` hashes only this folder's `.py` files, so `sync.sh` changes do not appear in it. One difference from the baseline pipeline: the first rescoring replays against the generation result held in memory rather than read back from its JSON file. JSON round-trips floats exactly and the writer refuses NaN, so the two are equivalent, but the baseline was replayed from disk. No case-arm reused the baseline by direction identity (R changes 5-13 representatives per view and M shares at most 5 leaders), so the identity-reuse path never ran. No resume was needed; the resume identity covers the experiment code, the saved estimator and the E2 record but not the shared helper sources, input packs, legacy zone module or frames, which the manifest hashes separately.

## What is not evidence

Membership is the merge group assignment; it is not ground truth. Control fits are least-squares diagnostics against a control; they are not generated courts. Nothing in E0-E3 makes a usability claim.
