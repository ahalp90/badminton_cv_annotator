# Selected annotator for the next benchmark

The selected development bundle reduces eight fitted classifiers to three and
improves mean rally recovery and attributed contact F1 on the existing validation
videos. It uses a contact detector, one 31-leaf XGBoost rally selector with 400
boosting rounds, and a small review ranker. Its current inference stage is slower.

The experiment asked whether fewer fitted models could recover more complete
badminton rallies while preserving contact quality. The selected bundle improves
both measures over the matched old XGBoost stack and the best earlier histogram
gradient boosting (HGB) grid setting. It is the recommended configuration for the
next benchmark; transfer to amateur or 60 fps footage remains untested.

## What the numbers mean

All comparisons use 32 training videos and the same eight development-validation videos. Each run covers 668 labelled rallies and 5,696 contacts. The five seed offsets (0, 100, 200, 300 and 400) vary fitting and sampling on those same videos; they are not independent validation samples. Seed ranges describe observed changes, not confidence intervals. No new untouched-test evaluation or raw-test inspection was performed.

The primary measures are completely correct rally recovery and player-attributed contact F1 at ±5 and ±10 frames. These tolerances are defined at 30 frames per second (fps) and scaled to each video's fps. A rally counts when one predicted section contains its complete labelled contact sequence, with every contact matched within tolerance, correct player sides and no extra contacts. Exact start/end frame equality is not required. Contact F1 is twice the number of correctly attributed matched contacts divided by predicted plus labelled contacts. Mean F1 is the arithmetic mean of the five per-seed pooled F1 scores. F1 changes below use percentage points (pp). `sset_30` is 30 fps; `sset_18` is the only 25 fps video among the eight validation videos.

The matched old XGBoost (XGB) reference is an eight-model stack, including a confidence model used only to rank review work. The reduced stack keeps one contact model and one or two whole-rally selectors; review ranking is separate. Grid-017, tied with grid-021, is the best earlier HGB validation setting. The table shows mean recovered rallies out of 668 and attributed F1 at each tolerance.

| System | Rallies ±10 | F1 ±10 | Rallies ±5 | F1 ±5 |
|---|---:|---:|---:|---:|
| Earlier HGB grid-017 (tie: grid-021) | 282.4 | 84.9174% | 216.0 | 82.7971% |
| Matched old XGB stack | 284.8 | 85.00% | 226.2 | 83.02% |
| Selected 31-leaf, 400-round, one-selector XGB | 296.0 | 85.53% | 243.6 | 83.72% |

Against the matched old XGB stack, the selected setting gains 11.2 rallies and 0.53 pp F1 at ±10, and 17.4 rallies and 0.70 pp at ±5. Its paired changes are positive for both metrics in all five seeds: rally changes are +4 to +15 and +9 to +24; F1 changes are +0.19 to +1.06 pp and +0.22 to +1.30 pp, respectively. Against grid-017, the mean differences are +13.6 and +27.6 rallies and about +0.61 and +0.92 pp F1. The retained HGB comparison record contains aggregate means only, so these are descriptive differences rather than paired seed results.

Both comparisons cover whole systems. Proposal coverage and target selection changed alongside the selector stack, so the gains do not isolate tree size, layer removal or backend. “Contact-32” means each contact-model fit used all 32 training videos; the HGB and XGB contact models were separate fits, not the same fitted model. The current validation results do not establish population significance or domain robustness.

## Why one selector at 400 rounds

The 31-leaf, 400-round, one-selector setting improves over its matched old XGB reference in all five seed comparisons for both measures and both tolerances. Against the same-depth 15-leaf baseline, its mean gains are 7.2/6.8 rallies and 0.60/0.65 pp F1 at ±10/±5. Moving from 200 to 400 rounds raises the one-selector means by 2.2/1.6 rallies and 0.25/0.30 pp F1, but some seed-paired changes are negative. This is evidence for the selected setting on these validation videos, not a guarantee for new footage.

The four retained settings have these five-seed means. Each row uses the same
validation population as the first table.

| Rally selectors | Rounds each | Rallies ±10 | F1 ±10 | Rallies ±5 | F1 ±5 |
|---|---:|---:|---:|---:|---:|
| One, 31 leaves | 200 | 293.8 | 85.28% | 242.0 | 83.42% |
| Two, 31 leaves | 200 | 293.8 | 85.17% | 241.4 | 83.28% |
| **One, 31 leaves** | **400** | **296.0** | **85.53%** | **243.6** | **83.72%** |
| Two, 31 leaves | 400 | 296.8 | 85.27% | 244.8 | 83.43% |

A second selector adds only 0.8/1.2 mean rallies over the one-selector 400-round setting at ±10/±5, while mean F1 falls by 0.26/0.29 pp. The one-selector option also keeps the annotation model count to two: one contact model and one rally selector. The retained review ranker makes three models in total. The selected contact fit uses 31 leaves, 180 rounds, learning rate 0.06, child weight 5, seed 20260824 and score cutoff 0.9. The selected rally selector uses 31 leaves, 400 rounds, learning rate 0.05, child weight 5, L2 regularisation strength 1 and seed 20260905. The search stopped after the five-seed repeats. The exported bundle uses the original seed-zero fit; its seed was not selected by validation performance.

Training uses four fixed groups. Each video's rally-training inputs use contact predictions from a model trained on the other 24 videos; the final contact model is then fit on all 32. Later selectors train from earlier held-group outputs and confidence features, with group exclusion repeated downstream. Contact labels can still pass indirectly between groups through these inputs. For example, a contact model supplying group B may have trained on group A, even when a later selector fit excludes group A. Internal training scores are therefore not unbiased evaluation estimates. Validation labels do not enter fitting. A separate eight-fold contact-28 sensitivity check changed only the simplified training inputs, kept final contact-32 fitting and validation pools fixed, and had worse initial-seed recovery; contact-24 remains the primary training-input setup.

For rough-rally targets, use ±5 labels when the candidate pool contains a known correct ±5 option; otherwise keep ±10 labels. Exclude unknowns. Each judged rough rally receives equal total weight, regardless of how many candidate options it has. A scale based on the legacy pool preserves the original training-weight magnitude when proposal coverage expands.

## Regression evidence

At ±10, the selected model loses a mean 0.4 and 1.4 rallies on `sset_18` and `sset_22`, respectively. Their F1 changes are −0.55 and −0.54 pp. On `sset_30`, it gains 0.4 rallies and 0.36 pp F1 at ±10 against old XGB, but loses two rallies at ±5 while gaining 0.16 pp F1. Against the same-depth 15-leaf baseline, it gains 2.0 rallies/1.11 pp at ±10 and 2.4 rallies/1.09 pp at ±5. These remaining per-video losses matter even with positive overall means; saved streams do not establish that a detector candidate was absent or identify a feature-level cause.



## Transfer to different footage

The validation videos are professional footage at 25 or 30 fps, not 60 fps amateur footage. Rally durations are measured in seconds and frame windows scale with fps, but motion features such as shuttle velocity and ankle speed retain raw per-frame values. Scaling the windows therefore does not make those features frame-rate invariant. No transfer comparison has been run; changing the motion units would require retraining. Five seeds on the same eight videos measure fitting variation, not domain robustness. These experiments do not show whether smaller trees transfer better. A target-domain sample could guide development, but choosing a model on it would require separate footage for final evaluation.

## Production behaviour and review ranking

Research and runtime now share label-free proposals and features, targets and
layer fitting. Bundle metadata checks feature layout, proposal limits, layer
count, review mode and package versions. Older bundles still load.

The selected bundle passes exact fresh-pool and annotation-stream checks on all
eight validation videos. The matched legacy streams also reproduce exactly.
These checks run after saving and reloading the bundle. The original seed-zero
bundle recovers **301/668 rallies at ±10 and 252/668 at ±5**, with F1 **85.23% and
83.70%**. Those are this bundle's results, distinct from the five-seed means above.
The 1.7 MiB bundle also loads successfully through the laptop's production API.

Retain the seven-leaf, 100-round review classifier. It ranks the same selected
annotations as the selector-only alternative. Its overall ordering improves at
both tolerances, while its benefit near the top of the queue is mixed.

The comparison contains 613 judged sections and excludes 141 unknown sections.
There are 312 wrong judged sections at ±10 and 361 at ±5. The first 10%, 25% and
50% below mean the first 62, 154 and 307 **judged** sections after ordering.
They do not describe those fractions of the full unfiltered queue.

| Tolerance | Ranking | Correct-section AUROC | Wrong-section average precision | Wrong in first 10% / 25% / 50% |
|---|---|---:|---:|---:|
| ±10 | Selector score | 0.7804 | 0.7766 | 55 / 123 / 222 |
| ±10 | Fitted review ranker | 0.8013 | 0.7901 | 53 / 126 / 228 |
| ±5 | Selector score | 0.7954 | 0.8398 | 57 / 136 / 247 |
| ±5 | Fitted review ranker | 0.8166 | 0.8533 | 58 / 137 / 252 |

AUROC measures whether correct sections tend to receive higher scores than
wrong sections. Average precision measures how well the queue retrieves errors
as the inspected fraction grows. Higher is better for both. The fitted ranker
finds two fewer wrong sections in the first 10% at ±10, but six more by halfway.
Its per-video AUROC falls on `sset_18`, `sset_30` and `sset_39` at both tolerances.
This is one fitted ranker evaluated on eight development videos. Its benefit is
modest, and the scores are not calibrated automatic-acceptance probabilities.
Review scores do not change annotations. The legacy confidence model scores
different annotations, so it is not a matched ranker comparison.

The current smaller stack is slower at inference. Two runs over all eight videos
through the same production entry point took 116.51/116.47 seconds, compared
with 31.02/31.34 seconds for the old XGBoost stack: about 3.74 times as long.
Both were limited to two CPU threads, with model order reversed on the second repetition.
Every output stream matched its saved reference. The timed stage includes
contact scoring, fresh proposal construction, selection and review scoring.
It excludes vision processing, initial contact-feature construction, file reads
and metric scoring. These are descriptive measurements on one host. Fewer
fitted classifiers have not produced a faster inference path.

A profile of `sset_30` attributes most of the time to constructing and describing
the expanded rally candidates. These are broader than the old stack's options.
The unprofiled selected replay takes about 22.7 seconds for this video's 91.25
minutes of footage. The profile is diagnostic: its instrumentation adds overhead,
so its timings are not substituted for the paired measurements above.

For the next development benchmark, the recommendation prioritises recovery,
contact quality and fewer fitted roles over the extra 85 seconds of this CPU
stage across eight videos. This is an engineering judgement. Total video-pipeline
time and manual correction time have not been measured here, so the benchmark
should check that practical trade-off before production adoption.

The latest targeted tests pass: 25 tests, including real one- and two-selector
fits, CLI settings and export boundaries. Changed Python files pass Ruff and the
whole-project Pyrefly check reports zero errors. Earlier integration checks
included exact HGB trainer parity and independent Opus reviews. The broader
suite recorded 3,135 passed, 35 skipped and one failure because `python` was
absent from PATH. That test passed with the virtual environment on PATH; the
original full-suite run still exited 1.

The [five-seed motion-normalisation follow-up](fps_motion_normalisation.md)
retained raw features: neither tested transformation improved mean validation
recovery or attributed F1. The selected structure and bundle remain unchanged.

The selected bundle is ready for the next MonoTrack benchmark. That benchmark and a separate amateur/60 fps
evaluation can test whether the development gains reduce manual correction in
the intended footage. No redesign merge to `main` has been performed.

## Evidence and reproduction

Historical published test summaries were incidentally visible while locating the old HGB validation record. They did not inform selection or any comparison here.

The [baseline evidence](../tree_simplification/results/five-seed-evidence.json.gz)
retains all five old-XGBoost and 7/15-leaf repeats, plus the complete six-setting
initial-seed screen. The [31-leaf evidence](../tree_simplification/results/retained-setting-evidence.json.gz)
retains both round settings and depths across all five seeds, including per-video
counts and paired differences. The [earlier HGB record](../tree_simplification/results/previous-hgb-validation.json.gz)
identifies the source branch and commit for grid-017. See the
[evidence notes](../tree_simplification/results/README.md) for metric aggregation
and provenance.

The [experiment instructions](../tree_simplification/README.md) describe fixed
inputs, pool construction, target generation and repeated fitting. The selected
run is `xgb-followup/longer31`, depth 1, under the Carmack run root
`/scratch/ahalperi/annotator-tree-simplification-20261020`.

Export the retained fit with the [production export command](../tree_simplification/README.md#production-export-and-review-ranking).
For a fresh fit through the production API, use
[`--rally-layers 1`](../../../docs/annotator/retuning.md#rally-layer-bundles).
The legacy fitting path remains available without that flag. Loading a saved
bundle selects its inference path automatically.

The [selected production result](../tree_simplification/results/selected-production.json.gz)
records per-video annotation and review metrics. The [review rows](../tree_simplification/results/selected-review-rows.json.gz)
retain the individual scores and correctness labels used for the ranking table.

The [timing result](../tree_simplification/results/selected-timing.json.gz)
retains all 32 measurements and stream-parity checks. The
[profiling result](../tree_simplification/results/selected-profile.json.gz)
records the single-video diagnostic. Reproduce the comparison with the
[timing command](../tree_simplification/README.md#production-inference-timing).
