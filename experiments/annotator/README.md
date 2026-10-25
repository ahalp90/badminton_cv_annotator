# Auto-annotator: experiments and handover

The auto-annotator turns badminton footage into rallies with contact times and
labels for which court half each hit came from. It reduces the manual labelling
needed to build a dataset for studying player performance. A useful rally must be
correct from beginning to end: one missed hit, extra hit, wrong player or bad
clip boundary can make it unsuitable for that dataset.

This work grew over several months from hand-written motion and geometry rules
into a pipeline of contact detection, rally repair and review ranking. The
experiments explain why those parts exist, which approaches earned their place,
and where the system still falls short.

**The current system finds most contacts, but complete rallies still need human
review.** The current three-model XGBoost system recovers a mean of 1,822.4 of
3,327 labelled rallies across five fits in the 46-video ShuttleSet22 comparison.
That is 54.78% under the report's standard scoring rules. The
[tree comparison](reports/tree_simplification.md) gives the results and definitions.

## Reading guide

| Purpose | Guide |
|---|---|
| Understand the build and avoid repeating failed approaches | [Development: what we tried and learned](development.md) |
| Understand the results and their limits | [Evaluation](evaluation.md) |
| Find a detailed experiment report | [Reports by research question](reports/README.md) |
| Check a result or run an experiment | [Reproducing the experiments](reproducing.md) |
| Run the annotator on saved video extracts | [Quickstart](../../docs/annotator/quickstart.md) |
| Understand or change the implementation | [How it works](../../docs/annotator/how_it_works.md), then the [code map](../../docs/annotator/code_map.md) |

## Current system

The working pipeline uses saved court geometry, shuttle tracks and player poses.
Rules find plausible contact frames, a learned classifier scores them, and
one rally selector compares possible contact sequences for each rally. A separate model ranks the
finished rallies for review. The normal annotation and retraining commands live
in `src/annotator`; the code here runs experiments around that implementation.

The default is the **raw-baseline XGBoost bundle, seed 0**, in
[the committed model directory](tree_simplification/results/boundary_retrain_v7/2_3/models/raw_baseline-seed-0/).
All three models were retrained with the validation-selected 2-second / 3-second
boundary policy. It uses a contact classifier, one rally selector and a review
ranker. The [reproduction guide](tree_simplification/README.md) covers the fixed split and
five seeded fits. The former eight-model bundles are historical records; the
current annotator no longer loads or fits them. Heuristic-only annotation remains
available.

Three substantial problems remain:

- **Whole-rally reliability.** Missing opening hits, extra contacts, wrong sides
  and clip boundaries interact. The development experiments distinguish these
  errors and show which repairs helped.
- **Transfer to new footage.** End-to-end performance on unfamiliar broadcasts
  and club recordings is unmeasured. The amateur court-fitting trials cover
  only part of that question.
- **Review quality.** The confidence model ranks clips, but the number of correct
  rallies recovered varies with queue size. Automatic acceptance of its labels
  remains unsupported.

The [development guide](development.md#where-to-continue) connects these directions
to the experiments that motivate them.

## Where things live

| Code or evidence | Purpose |
|---|---|
| `heuristic_tuning/` | Calibration fixtures and parameter sweeps for the early rules |
| `contact_attribution_comparison_20260814/` | Player-distance experiment code and evidence |
| `court_geometry_repair/`, `independent_court/`, `court_scene_sampling/` | Court repairs, alternative fitting methods and frame-sampling comparisons |
| `tree_simplification/` | Current XGBoost fitting, evaluation and saved results |
| `development.py` | Shared fixed splits, sampling order and experiment helpers |
| `old_court_regression/`, `good_court_refit/` | Historical pinned inputs and comparison evidence |
| `annotator_closeout_new_courts/` | Closeout of the new-court model: where it fails, footage checks, detector audit and shuttle guard addendum |
| `measurement.py`, `records.py`, `runs/` | End-to-end measurement, result packaging and saved runs |

Earlier contact-model and sequence-repair research remains in tracked
`scratch/contact_det*` directories. Their local run paths and status notes
describe the experiments at the time. Large caches and temporary logs remain
with the run outputs.
