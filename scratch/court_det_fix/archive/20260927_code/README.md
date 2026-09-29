# Former detector source and experiment runners

Archived on 27 September 2026 after the detector code moved into one package.
Use [the detector guide](../../court_detector/README.md) for maintained code and
[pickup](../../pickup.md) for current work.

The original files are retained byte-for-byte. Search and scoring functions
were extracted from research drivers; their report writers and trial launchers
stay here. Tests and the remaining independent-court experiment runners now
import the maintained package. Large data and saved outputs stayed in place.

## Maintained replacements

Paths in the left column were repository-relative before the move.

| Former source | Maintained module |
| --- | --- |
| [scratch/court_det_fix/w5_holistic/run_w5.py](w5_holistic/run_w5.py) | [scoring.py](../../court_detector/scoring.py) |
| [scratch/court_det_fix/w5_holistic/verifier.py](w5_holistic/verifier.py) | [measurements.py](../../court_detector/measurements.py) |
| [scratch/court_det_fix/wider_evaluation/generation.py](wider_evaluation/generation.py) | [search_records.py](../../court_detector/search_records.py) |
| [scratch/court_det_fix/w5_holistic/automatic_generation.py](w5_holistic/automatic_generation.py) | [generation.py](../../court_detector/generation.py) |
| [scratch/court_det_fix/wider_evaluation/measurement.py](wider_evaluation/measurement.py) | [sampling.py](../../court_detector/sampling.py) |
| [scratch/court_det_fix/next_steps_20260916/webui_seed/source/run_automatic.py](next_steps_20260916/webui_seed/source/run_automatic.py) | [candidate_pool.py](../../court_detector/candidate_pool.py) |
| [scratch/court_det_fix/next_steps_20260916/webui_seed/source/run_given.py](next_steps_20260916/webui_seed/source/run_given.py) | [proposals.py](../../court_detector/proposals.py) |
| [scratch/court_det_fix/next_steps_20260916/webui_seed/source/projective_seed.py](next_steps_20260916/webui_seed/source/projective_seed.py) | [line_matching.py](../../court_detector/line_matching.py) |
| [scratch/court_det_fix/next_steps_20260916/webui_seed/source/run_population.py](next_steps_20260916/webui_seed/source/run_population.py) | [prepare_lines.py](../../court_detector/prepare_lines.py) |
| [scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis/scan_population.py](frozen_helpers_20260914/marking_diagnosis/scan_population.py) | [candidate_geometry.py](../../court_detector/candidate_geometry.py) |
| [scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis/run_diagnosis.py](frozen_helpers_20260914/marking_diagnosis/run_diagnosis.py) | [court_checks.py](../../court_detector/court_checks.py) |
| [scratch/court_det_fix/frozen_helpers_20260914/legacy/zone_net.py](frozen_helpers_20260914/legacy/zone_net.py) | [players.py](../../court_detector/players.py) |
| [scratch/court_det_fix/frozen_helpers_20260914/legacy/camera_diagnostic.py](frozen_helpers_20260914/legacy/camera_diagnostic.py) | [camera.py](../../court_detector/camera.py) |
| [scratch/court_det_fix/w5_holistic/line_template_source.py](w5_holistic/line_template_source.py) | [line_templates.py](../../court_detector/line_templates.py) |
| [scratch/court_det_fix/frozen_helpers_20260914/vp_pruning/vp_pruning.py](frozen_helpers_20260914/vp_pruning/vp_pruning.py) | [directions.py](../../court_detector/directions.py) |
| [scratch/court_det_fix/next_steps_20260916/webui_seed/source/inspect_appearance.py](next_steps_20260916/webui_seed/source/inspect_appearance.py) | [paint_profiles.py](../../court_detector/paint_profiles.py) |
| [experiments/annotator/independent_court/detector.py](experiments/annotator/independent_court/detector.py) | [geometry.py](../../court_detector/geometry.py) |
| [experiments/annotator/independent_court/assignment.py](experiments/annotator/independent_court/assignment.py) | [line_observations.py](../../court_detector/line_observations.py) |
| [experiments/annotator/independent_court/case_provenance.py](experiments/annotator/independent_court/case_provenance.py) | [image_sources.py](../../court_detector/image_sources.py) |
| [experiments/annotator/independent_court/fixed_stripe_refit.py](experiments/annotator/independent_court/fixed_stripe_refit.py) | [stripe_fitting.py](../../court_detector/stripe_fitting.py) |
| [experiments/annotator/independent_court/stripe_observations.py](experiments/annotator/independent_court/stripe_observations.py) | [stripe_measurements.py](../../court_detector/stripe_measurements.py) |
| [experiments/annotator/independent_court/junction_observations.py](experiments/annotator/independent_court/junction_observations.py) | [junctions.py](../../court_detector/junctions.py) |
| [experiments/annotator/independent_court/net_geometry.py](experiments/annotator/independent_court/net_geometry.py) | [net_geometry.py](../../court_detector/net_geometry.py) |
| [experiments/annotator/independent_court/paint_geometry.py](experiments/annotator/independent_court/paint_geometry.py) | [paint_geometry.py](../../court_detector/paint_geometry.py) |

## Other archived code

The original relative trees are retained for `w5_holistic`, `wider_evaluation`,
`line_identity`, `frozen_helpers_20260914`, `next_steps_20260916`,
`colour_consistency`, `edge_polarity`, `net_recovery` and `d17_timing`.
These include reference calculations, experiment launchers and their tests.
They remain useful for checking an old finding or reopening an experiment.
Dated `court_detector/check_*` drivers are also archived here; their reports
and data remain beside the saved results.

## Rerunning old code

Internal imports, relative data paths, shell commands and remote paths are
historical and have not been repaired. Moving a script back alone may not make
it runnable. Use the matching historical checkout or adapt a copy with the
inputs named in its original report. The maintained detector does not import
these scripts. Reference feet scripts still used by tests remain beside their
saved evidence.

## Recovery and checks

The source snapshot is `.recovery/before-code-move-20260927.tar.gz`, relative to
`court_det_fix/`. It also preserves the document tidy as it stood before this
code move. The [completed worklog](WORKLOG.md) records the scope and checks.
