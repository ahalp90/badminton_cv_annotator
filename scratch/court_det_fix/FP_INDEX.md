# Court detector file map

Use this map to locate code, input packs and compact results. [INDEX.md](INDEX.md) is the short entry point, [pickup.md](pickup.md) owns current actions, and [DETECTOR_DECISIONS.md](DETECTOR_DECISIONS.md) explains the decisions and historical limits. Paths in this file are relative to `scratch/court_det_fix/` unless marked repository-relative.

## Active experimental code

| Need | Code and role |
| --- | --- |
| Generate fresh G0/G1 proposals with SVD12 | [automatic_generation.py](w5_holistic/automatic_generation.py) contains the direction-family screen and matcher path; [line_template_source.py](w5_holistic/line_template_source.py) supplies line-template proposals; [run_w5.py](w5_holistic/run_w5.py) runs the W5 whole-court experiment. This is experimental code, not the integrated detector. |
| Run the six-case search-depth comparison | [svd_search/run.py](svd_search/run.py) runs baseline 512/256, deeper 640/256 and wider shortlist 512/512 arms. [SVD search README](svd_search/README.md) maps the experiment. |
| Generate Am1 seeds and inspect net selection | [am1_recovery_trial.py](colour_consistency/am1_recovery_trial.py) adds automatic line-group seeds; [am1_net_selection_trial.py](colour_consistency/am1_net_selection_trial.py) evaluates the recovery selection. [AM1_RECOVERY.md](colour_consistency/AM1_RECOVERY.md) explains the result. |
| Measure saved-pool net behaviour | [scan_saved.py](net_recovery/scan_saved.py) scans the frozen 71 views; [bounded_trial.py](net_recovery/bounded_trial.py) implements the bounded post-support rule and saved-pool trial. [test_bounded_trial.py](net_recovery/test_bounded_trial.py) covers that rule. |
| Correct a selected court | [edge_auto_trial.py](colour_consistency/edge_auto_trial.py) tests automatic stripe polarity; [refit_selected.py](net_recovery/refit_selected.py) applies it to requested selections. Selection precedes correction in the accepted gallery. |
| Replay seeded Am1 with net preference | [run_combined.py](net_recovery/run_combined.py) reuses the frozen G0/G1 comparison, generates additional Am1 line-group seeds and mixes historical with fresh paint measurements. It is **not** a fresh full G0/G1/template/SVD12 combined runner. Its score drift cannot isolate the seed's ranking effect. |
| Rebuild paired reference statistics | [paired_reference_analysis.py](net_recovery/statistics/paired_reference_analysis.py) reads the local bounded-trial packet, writes results JSON and prints summaries. The prose report is a separate record. Run from repository root: `~/.venvs/badminton-cicd/bin/python scratch/court_det_fix/net_recovery/statistics/paired_reference_analysis.py`. |

## Code, inputs and big working packets

### Inputs, results and visual checks

| Need | Location and interpretation |
| --- | --- |
| Shared pixels and controls | [frozen_views/README.md](frozen_views/README.md) describes saved frames and input packs. [71-view manifest](wider_evaluation/runs/20260922/manifest.json.gz), [frozen comparison](wider_evaluation/runs/20260922/comparison.json.gz) and [numeric fits](wider_evaluation/runs/20260922/numeric_fit.json.gz) are historical inputs; [wider evaluation](archive/20260922/wider_evaluation_20260922.md) explains coverage. |
| Bounded net cohorts and choices | [bounded_split.json.gz](net_recovery/bounded_split.json.gz) fixes cohort membership and seeded Am1; [saved_net_scan.json.gz](net_recovery/saved_net_scan.json.gz) retains the unrestricted scan. [Paired report](net_recovery/statistics/paired_reference_report.md) and [results JSON](net_recovery/statistics/paired_reference_results.json.gz) contain compact reference comparisons. |
| Accepted net and stripe visuals | [bounded gallery](net_recovery/bounded_gallery/index.html) and [polarity gallery](net_recovery/polarity_gallery/index.html) use tracked images from [colour_consistency/gallery/](colour_consistency/gallery/). The 20-case page is gallery-level user feedback, not 20 per-case labels. [Net assessment](net_recovery/ASSESSMENT.md) explains the bounded cue and caveats. |
| Search timing and choices | [SVD matcher benchmark](svd_runtime/RESULTS.md) measures the 52.9% matcher saving; [six-case search worklog](svd_search/WORKLOG.md) reports full-trial timings and rulings. The 18 tracked records in [run_20260923/cases/](svd_search/run_20260923/cases/) support replay and timing inspection. `svd_search/run_20260923/generation/` holds larger local-only stage records. See [search local-data note](svd_search/LOCAL_DATA.md). |
| Colour and fitting trials | [colour worklog](colour_consistency/PLAN.md), [polarity assessment](edge_polarity/local_audit/ASSESSMENT.md) and [diagnostic measurements](colour_consistency/measurements.json.gz). The saved W5 and G1/template colour columns share old ranking; they are not colour before/after arms. |
| Historical proposal and temporal evidence | [evidence/](evidence/) groups investigations by mechanism; [old experiment package](../../experiments/annotator/independent_court/README.md) contains reusable historical code. [Review packet](evidence/review_20260922/README.md) identifies what was published. |

## Local data and recovery

The bounded statistical rerun requires the ignored `local_scratch/net_recovery/20260923/bounded/bounded_trial.json.gz` (about 17 MB). The accepted corrected fits and exact requests are in `local_scratch/net_recovery/20260923/selected_polarity/{bounded_results,bounded_requests}.json.gz`. The Am1 seeded pool is `local_scratch/external_delegate/20260923-am1-recovery-trial/seeded_pool.json.gz`. These are local-only inputs, not substitutes for similarly named frozen pools. Check they exist before moving to a new worktree or remote checkout. The [SVD runtime local-data note](svd_runtime/LOCAL_DATA.md) and [search local-data note](svd_search/LOCAL_DATA.md) distinguish ignored raw packets from tracked checks and receipts.

## Integrated scene path and saved contract

The current annotator first finds raw cut intervals in repository-relative [composition_mask.py](../../src/annotator/composition_mask.py) (`detect_cuts`, PySceneDetect `ContentDetector`). [court_evidence.py](../../src/annotator/court_evidence.py) samples up to ten frames per interval in `detect_scene_evidence`; `matching_view_groups` in [court_views.py](../../src/annotator/court_views.py) later compares perceptual hashes and alignment. Only initially valid court scenes join groups, and a group needs at least three members. `_share_scene_corners` then shares geometry. Grouping happens **after** first-pass court detection, so it does not reduce that search count.

The dataset stage is [vision.py](../../src/dataset_builder/vision.py) (`build_detected_court_stage`, `persist_court_vision`). Scene records keep raw cut intervals, sample indices, validity, active court corners and optional `view_group_index`; [court codec](../../src/dataset_builder/_court_codec.py) validates them. Raw cuts and distinct compatible views are different units. [Launch handover](handover_20260923/01_LAUNCH_OPTIMISATION.md) has the code anchors and design constraint.

## Old experiment names

`G0` means original-fragment proposals; `G1` means paint-filtered proposals. `S0` and `S1` name original and filtered scoring fragments. `GX`, `Am1`–`Am4`, `SS03` and `SS21` identify source videos. A source-qualified candidate such as `G1:16:44` is unique only within its case. `/child` marks a separately refined candidate; a parent ruling does not automatically apply to the child. **Reference-best** is chosen retrospectively using annotations and is never an automatic selection. [G0/G1 explanation](evidence/g0_g1/README.md); [candidate and gate account](evidence/holistic_admission/README.md).

The [archived idea map](archive/20260923_top_level/FP_INDEX.md) retains older experiment tags and numeric descriptions, including L1–L3, W2/W4, E0–E4, C2 and the 33/43/53 visibility arms. Recovery archives are for superseded material; [INDEX.md](INDEX.md#recovery) explains their route.
