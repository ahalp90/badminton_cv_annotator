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
review.** The selected model recovers 1,744 of 3,327 labelled rallies in the
46-video ShuttleSet22 comparison. That is 52.4% under the recorded evaluation
rules. These videos have been examined repeatedly during development; the result
does not establish performance on unfamiliar broadcasts or club footage.

## Reading guide

| Purpose | Guide |
|---|---|
| Understand the build and avoid repeating failed approaches | [Development: what we tried and learned](development.md) |
| Understand the results and their limits | [Evaluation](evaluation.md) |
| Check a result or run an experiment | [Reproducing the experiments](reproducing.md) |
| Run the annotator on saved video extracts | [Quickstart](../../docs/annotator/quickstart.md) |
| Understand or change the implementation | [How it works](../../docs/annotator/how_it_works.md), then the [code map](../../docs/annotator/code_map.md) |

The development guide covers the main experiments and their consequences in one
place. The detailed reports supply the measurements and reproduction details.

## Current system

The working pipeline uses saved court geometry, shuttle tracks and player poses.
Rules find plausible contact frames, a learned classifier scores them, and
later models compare small repairs to each rally. A separate model ranks the
finished rallies for review. The normal annotation and retraining commands live
in `src/annotator`; the code here runs experiments around that implementation.

The selected model is the **new-court base**, recorded on 3 October 2026. Its
files are in [the committed model directory](../../data/annotator/sset_and_sset22_trained_20261003T041112Z/).
The quickstart covers its environment and input requirements. The latest refit
is one comparison within the larger build: it recovered more contacts than the
historical model, but fewer complete rallies.

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

The four guides at this level provide the project overview, development history,
evaluation and reproduction routes. The [report map](reports/README.md) groups
specialist reading by research question. Results and their rerun instructions
share a document.

The other directories contain code and saved outputs:

| Code or evidence | Purpose |
|---|---|
| `heuristic_tuning/` | Calibration fixtures and parameter sweeps for the early rules |
| `contact_attribution_comparison_20260814/` | Player-distance experiment code and evidence |
| `court_geometry_repair/`, `independent_court/`, `court_scene_sampling/` | Court repairs, alternative fitting methods and frame-sampling comparisons |
| `old_court_regression/`, `good_court_refit/` | Refit runners, pinned inputs and comparison evidence |
| `measurement.py`, `records.py`, `runs/` | End-to-end measurement, result packaging and saved runs |

Earlier contact-model and sequence-repair research remains in tracked
`scratch/contact_det*` directories. The development guide links specific reports
when their detail is useful. Their local run paths and historical status notes
describe the experiment at the time.

The development guide records each substantial experiment through its question,
result and implication. Detailed reports hold the evidence and rerun instructions;
large caches and temporary logs belong with the run outputs.
