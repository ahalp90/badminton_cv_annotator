# Where the code and evidence live

This file owns paths and data requirements. It does not own results or the
work queue. [INDEX.md](INDEX.md) maps the live documents;
[pickup.md](pickup.md) is the place to resume.

Paths below are relative to `scratch/court_det_fix/`, unless stated otherwise.

## Run or change the detector

| Need | Start here |
| --- | --- |
| API, settings and prepared-view command | [Detector guide](court_detector/README.md) |
| GPU, parallel work, Numba, cheap scoring and court reuse | [Speed-up design](court_detector/PERFORMANCE.md) |
| Historical research runner and comparison protocol | [run_d17.py](archive/20260927_code/d17_timing/run_d17.py), [wiring and checks](archive/20260927_code/d17_timing/WIRING.md#how-to-check-an-integrated-detector) |
| Scripts and saved speed-up measurements | [Measurement index](court_detector_optimisation_handover/claude_evidence/README.md) |
| Compare saved research runs exactly | [compare_exact_runs.py](court_detector_optimisation_handover/claude_evidence/exact_rewrites/compare_exact_runs.py) |
| Rebuild the older paired net statistics | [paired_reference_analysis.py](archive/20260927_code/net_recovery/statistics/paired_reference_analysis.py); archived source, with historical paths |

## Code and input paths to keep stable

All maintained detector code is in [court_detector/](court_detector/README.md#code-map).
The old script folders now hold data and historical notes. Their source moved
to the [27 September code archive](archive/20260927_code/README.md).

| Folder or file | Why it stays |
| --- | --- |
| [frozen_views/](frozen_views/README.md) | Source images, packs and baseline records |
| [wider_evaluation/](wider_evaluation/) | Saved view manifest and control packs used by the view runner |
| [w5_holistic/](w5_holistic/), [next_steps_20260916/](next_steps_20260916/) | Saved candidate courts and trial records |
| [direction_agreement/runs/](direction_agreement/runs/) | Saved direction records |
| [worklog/remote_records_20260921/](worklog/remote_records_20260921/README.md) | Saved paint-line inputs; this is data despite the folder name |
| [fresh_feet/](court_detector_optimisation_handover/claude_evidence/fresh_feet/) | Prepared 28-view list, feet records and independent reference scripts used by tests |
| [net_recovery/](net_recovery/), [colour_consistency/](colour_consistency/), [edge_polarity/](edge_polarity/) | Historical measurements and galleries |

Archived scripts keep their old internal paths. Read the
[archive notice](archive/20260927_code/README.md#rerunning-old-code) before a rerun.

## Checks and measurements

These are dated records. Their old plans and approvals remain historical.

| Question | Record |
| --- | --- |
| Did the joined detector match the research chain? | [25 September check](court_detector/check_20260925/README.md) |
| What changed with the upright-camera filter? | [Camera-filter check](court_detector/check_20260926_upright/README.md) |
| Which paint-test change failed? | [Per-sample paint check](court_detector/check_20260926_paint_test/README.md) |
| Why keep the paint/line blend and discard the top-15 refit? | [Court-choice check](court_detector/check_20260926_court_choice/README.md#decision), [completed Carmack run](court_detector/check_20260926_court_choice/README.md#carmack-run) |
| Why discard averaged line paint? | [Line-paint check](court_detector/check_20260926_line_paint/README.md) |
| Why discard player-size filtering and later score changes? | [Player-size check](court_detector/check_20260926_player_size/README.md) |
| Where are old net-weight measurements? | [Paired report](net_recovery/statistics/paired_reference_report.md), [results](net_recovery/statistics/paired_reference_results.json.gz) |
| Where are old direction-screen and search-depth results? | [Direction-screen benchmark](archive/20260926/experiments/svd_runtime/RESULTS.md), [search worklog](archive/20260926/experiments/svd_search/WORKLOG.md), [compute audit](archive/20260926/experiments/svd_runtime/COMPUTE_AUDIT.md) |
| Where are the accepted net/stripe galleries? | [Net choices](net_recovery/bounded_gallery/index.html), [adjusted stripes](net_recovery/polarity_gallery/index.html) |
| Where are earlier experiments and retired documents? | [Evidence](evidence/), [archive map](archive/README.md) |

## Data that is not in git

Keep these inputs distinct from similarly named saved pools:

- Repository-relative `local_scratch/net_recovery/20260923/bounded/bounded_trial.json.gz`
  is required by the paired statistics script
- Repository-relative `local_scratch/net_recovery/20260923/selected_polarity/{bounded_results,bounded_requests}.json.gz`
  records the earlier corrected fits and exact requests
- Repository-relative `local_scratch/external_delegate/20260923-am1-recovery-trial/seeded_pool.json.gz`
  holds the seeded Am1 pool
- [SVD runtime data](svd_runtime/LOCAL_DATA.md) and
  [search data](svd_search/LOCAL_DATA.md) identify larger local run records
- The joined detector's people records and source videos are on Carmack.
  The [25 September check](court_detector/check_20260925/README.md) records
  its inputs and runner commands
- The [camera-filter check](court_detector/check_20260926_upright/README.md)
  records its remote saved stages. The
  [final choice check](court_detector/check_20260926_court_choice/README.md#carmack-run)
  identifies `court_detector_20260926_final/`, with 378 MB of remote artefacts

Confirm files exist before attempting a replay. Nothing in the tidy reruns a
remote job or assumes that old remote paths still exist.

## Existing video pipeline

These paths are repository-relative:

| Code | Role |
| --- | --- |
| [composition_mask.py](../../src/annotator/composition_mask.py) | Finds cuts using PySceneDetect's `ContentDetector` |
| [court_evidence.py](../../src/annotator/court_evidence.py) | Samples frames inside each interval and detects court evidence |
| [court_views.py](../../src/annotator/court_views.py) | Groups compatible views by image hashes and alignment after initial searches |
| [vision.py](../../src/dataset_builder/vision.py) | Builds and saves the detected-court stage |
| [_court_codec.py](../../src/dataset_builder/_court_codec.py) | Checks the saved court-record format |

## Read old identifiers

`S0` and `S1` identify original and paint-filtered fragments used for scoring,
whereas G0 and G1 identify which fragments were searched. A label such as
`G1:16:44` names a court only within its own case. `/child` means a separate
refit. A judgement about the original court does not automatically cover it.

“Reference-best” means retrospectively closest to annotations. It is not an
automatic detector choice. `Am1`–`Am4`, `GX`, `SS03` and `SS21` name source
videos. The [archived file map](archive/20260923_top_level/FP_INDEX.md) preserves
older experiment labels when needed.
