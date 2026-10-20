# Preprocessing offers a small rally-recovery trade-off

Keep baseline preprocessing as the default and filtering disabled. The input
search found a small rally-recovery trade-off, while filtered test annotations
still have substantial judged error.

This experiment tested whether input settings chosen for an older annotator still
suit the [simplified raw-feature stack](tree_simplification.md). It assessed
whether to change the default preprocessing and whether saved review scores can
help select annotations for a dataset.

Every fit uses the same 32 training videos. Development validation has eight
videos, 668 labelled rallies and 5,696 labelled contacts. The ShuttleSet22
test set has 46 videos, 3,327 labelled rallies and 37,184 labelled contacts.

The primary metric is player-attributed contact F1 at ±10 equivalent 30 fps
frames. A correct contact needs both a timing match and the correct player side;
F1 is twice the correct-contact count divided by predicted plus labelled contacts.
Correct-rally recovery counts wholly correct predicted rallies against the labels,
with 668 labelled rallies on validation and 3,327 on test. The ±5 tolerance is a secondary check. Counts are
pooled across videos within each seed, then results are averaged over five matched
seed offsets: 0, 100, 200, 300 and 400. These fits share videos; they are not five
independent generalisation trials.

The raw-feature baseline has the highest mean validation F1. Increasing the body-size half-window from 12 to 36 frames recovers
five more correct validation rallies on average, at a cost of 0.31 percentage
points of F1. Its test gains are small and vary across seeds. Keep it as
an optional fitted alternative. Review filtering also remains optional: the most
aggressive threshold chosen on validation still leaves about 30% error among judged retained test
spans at ±10.

## What changed, and how it was selected

The search changed one input at a time: baseline plus 13 initial challengers,
then six refinements. Values below are expressed at 30 fps; frame counts and
speed thresholds scale with each video's frame rate.

| Input | Baseline | Initial alternatives | Refinements |
|---|---:|---|---|
| Rest-speed threshold | 0.002 | 0.001, 0.0125 | — |
| Frames of rest before rally end | 90 | 30, 150 | 120, 180 |
| Gap-state demotion bound | 75 | 15, 150 | — |
| Body-size half-window | 12 | 1, 24 | 18, 36 |
| Trajectory smoothing window | 3 | 1, 5 | — |
| Start-speed threshold | 0.015 | 0.01, 0.0225 | 0.0075, 0.03 |
| Span opening rule | Back-fill | Region start | — |

The body-size half-window smooths player bounding-box height when expressing
shuttle-to-wrist distance in body-height units. The 36-frame value is a
half-window, rather than the total window length.

All models use XGBoost. Contact settings stay at 31 leaves, 180 rounds and
learning rate 0.06; selector settings stay at 31 leaves, 400 rounds and rate 0.05.
The selector considers up to three contact edits and six deletion candidates.
The optional reviewer has seven leaves and 100 rounds. Minimum child weight is 5
and L2 regularisation is 1. Raw features and the selected target policy stay
fixed. Four contact fits train on 24 videos each to supply training scores; the
final contact fit uses all 32. Each candidate rebuilds its training pools and
span weights. There is no refit on all 40 development videos.

The seed-0 screen shortlisted baseline and the union of the top two candidates
under F1 and recovery. That shortlist was fixed before repeat scores were opened.
Four further seeds per shortlisted configuration brought the total to 40
validation fits from 20 preparations. Repeats reused preparation and rebuilt the
learned stages and export. No combinations were searched.

The F1 ranking uses mean F1 at ±10, then recovery at ±10 and F1 at ±5. The
recovery ranking uses recovery at ±10, then F1 at ±10 and recovery at ±5.
Remaining ties favour baseline, fewer changed settings and a stable identifier.
No first-metric tie affected either winner. Validation selected both fitted
finalists and review thresholds before test access; the test run measured those
choices unchanged.

## Validation favours baseline F1 and the longer body window for recovery

| Configuration | F1 ±10 | Correct rallies ±10 / 668 | F1 ±5 | Correct rallies ±5 / 668 |
|---|---:|---:|---:|---:|
| Baseline | **85.53%** | 296.0 | 83.72% | 243.6 |
| End rest 150 | 85.45% | 261.6 | **83.76%** | 217.4 |
| End rest 180 | 85.26% | 231.4 | 83.60% | 190.2 |
| Body half-window 36 | 85.22% | **301.0** | 83.44% | 244.8 |
| Body half-window 24 | 85.20% | 300.2 | 83.49% | **248.4** |

Longer end-rest settings led seed-0 F1, but lost that lead over five seeds and
reduced ±10 recovery in every paired seed. Body half-window 36 improves recovery
in three seeds, ties once and loses once; it loses F1 in four. The gap-demotion
alternatives exactly matched baseline's seed-0 metrics.

Excluding seed 0 preserves both winners. Leaving out one validation video at a
time gives body half-window 36 the recovery win in seven of eight cases. Baseline
wins F1 in five, end rest 150 in two and body half-window 36 in one. These are
sensitivity checks within the same small validation population.

## Test gains are small and mixed

| Configuration chosen on validation | F1 ±10 | Correct rallies ±10 / 3,327 | F1 ±5 | Correct rallies ±5 / 3,327 |
|---|---:|---:|---:|---:|
| Baseline | 86.24% | 1,822.4 | 84.62% | 1,470.2 |
| Body half-window 36 | 86.32% | 1,830.0 | 84.70% | 1,474.6 |

| Body 36 minus baseline | Mean | Range across five seeds | Better / worse seeds |
|---|---:|---:|---:|
| F1 ±10, percentage points | +0.083 | −0.157 to +0.499 | 2 / 3 |
| Correct rallies ±10 | +7.6 | −29 to +38 | 3 / 2 |
| F1 ±5, percentage points | +0.081 | −0.255 to +0.457 | 4 / 1 |
| Correct rallies ±5 | +4.4 | −43 to +44 | 3 / 2 |

Across test videos, body half-window 36 improves mean ±10 F1 in 27 of 46 and
reduces it in 19; recovery improves in 24, falls in 21 and ties in one. The seed
spread exceeds the mean gains. This supports a small fitted-model trade-off,
without establishing statistical significance, equivalence or transfer to amateur
footage. Test results were not used to select a deployment seed.

## Review filtering reduces coverage and judged error

Filtering has a different denominator from labelled-rally recovery. It operates
on **predicted spans**. Test coverage counts 20,105 span–seed instances per
configuration: 4,021 predictions per seed over the same 46 videos. Of these,
5,020 have unknown correctness. Error is wrong / (correct + wrong) among retained
judged spans; unknown spans count towards coverage but stay outside that error
denominator. There are no missing test scores.

For each model and ranking, validation selected cutoffs closest to 50%, 70%, 80%,
90% and 95% retained coverage, with ties favouring more coverage. Cutoffs use all
five validation seeds at ±10 and apply unchanged to both test tolerances. A
validation coverage target guarantees neither test coverage nor error.

The baseline's fitted reviewer gives the following test results. Its unfiltered
±10 judged population has 9,112 correct and 5,973 wrong span–seed instances.

| Validation coverage target | Test coverage | Retained instances | Unknown retained | Error ±10 | Error ±5 |
|---|---:|---:|---:|---:|---:|
| Unfiltered | 100.00% | 20,105 | 5,020 | 39.60% | 51.27% |
| 95% | 95.80% | 19,260 | 4,202 | 39.49% | 51.18% |
| 90% | 90.85% | 18,266 | 3,329 | 39.04% | 50.82% |
| 80% | 82.45% | 16,577 | 2,394 | 36.58% | 48.71% |
| 70% | 75.76% | 15,232 | 1,945 | 34.47% | 46.63% |
| 50% | 58.75% | 11,812 | 1,018 | 29.90% | 40.99% |

The validation-80% cutoff retains 8,995 / 9,112 known-correct test instances
(98.7%). It removes 17.5% of predictions, but reduces judged error by only about
three percentage points. The 2,394 unknown retained instances remain unassessed.
Of the 3,528 removed instances, 2,626 have unknown correctness, 785 are wrong
and 117 are correct. Reduced coverage therefore does not imply an equivalent
reduction in known annotation errors.
Even the validation-50% cutoff leaves about 30% judged error at ±10.

The pooled 90% and 95% cutoffs are less stable across seeds. Some fitted models
assign the same low score to many spans, so a cutoff near those scores can retain
or remove the whole group. Baseline's validation-90% cutoff retains 100% of test
spans for seed 0 but 85.9% for seed 300. The validation-80% cutoff ranges from
81.2% to 83.7% across test seeds. Inspect the per-seed coverage before using a
pooled cutoff with one fitted bundle; the target is not a per-bundle guarantee.

![Test coverage and judged error for baseline fitted-reviewer and selector scores](../tree_simplification/results/input-review-tradeoff.svg)

Each marker uses a threshold chosen on validation; lines only connect measured points.
Labels identify the fitted reviewer's validation coverage targets.
The figure pools five fits of the same test videos. Unknown correctness counts
on the coverage axis and is excluded from the error axis. Selector-score fallback
and fitted reviewer offer similar trade-offs at these points. Their coverages
differ, so neither has demonstrated superiority at matched coverage. Body
half-window 36 is similar: its fitted reviewer retains 83.67% at the
validation-80% cutoff, with 36.64% judged error at ±10.

## Using the result for dataset construction

Keep baseline preprocessing as the default, with review filtering disabled. Bundles
carry preprocessing settings; annotation loads them and rejects conflicting
overrides. Using body half-window 36 therefore requires its fitted bundle.

Optional filtering writes a separate checksum-linked decision file over saved
`contacts.review_score`. Scores below the lower threshold are marked for
filtering; scores at or above the upper threshold are marked for acceptance.
The middle interval and missing scores go to manual review. Original annotations
and scores remain available.

```bash
python -m dataset_builder review-policy \
  --run-dir RUN --output REVIEW_POLICY.json.gz \
  --low-filter-threshold LOW --high-accept-threshold HIGH
```

Use exact cutoffs from `review_thresholds.candidate_rankings` in the
[evidence record](../tree_simplification/results/input-preprocessing-evidence.json.gz).
All evaluated bundles contain the fitted reviewer, so their exported scores need
the fitted-reviewer cutoffs. The selector curves are diagnostic comparisons here;
production also supports selector scores when a bundle has no review model.
Scores are not calibrated probabilities.
This experiment measures single-cutoff retained sets; separate acceptance and
filtering thresholds have no validated acceptance-error guarantee.

For building a usable annotation dataset, the useful next step is to examine
annotation errors and review-score usefulness on independent videos. Small
preprocessing gains leave substantial judged error after filtering. Further
preprocessing search has less immediate support than improving the annotations
and checking how much review effort the scores can save.

## Cost, checks and reproduction

The 20 preparations and 40 validation fits produced 220 timed commands. Summed
elapsed time is 21.08 hours; summed reported user CPU time is 35.17 hours. These
are accumulated costs across parallel work. Pool construction and export/replay
cost 5.99 and 6.25 elapsed hours; contact fitting costs 1.86 hours. Maximum
resident memory is 18.07 GiB per command. Validation allocations used at most 28
CPU workers. Ten test lanes used 20 workers and completed about 39 minutes after
launch, including completion-detection delay.

All 40 validation exports reproduced selector metrics and prediction streams.
All ten test lanes passed bundle, source, input and 46-video completeness
checks; all 460 saved streams were readable. An initial test launch stopped
before evaluation because four input paths were missing. The retry changed only
those paths and preserved failed records.
Independent recomputation reproduced all ten test summaries from per-video counts
and all 24 threshold reports from saved review rows and unchanged cutoffs.

Implementation commits are `b9c2c62b`, `ab5f3bfe`, `370506cd` and `94099e16`.
Baseline fits use the first revision; challengers use the last. Intervening
training changes add provenance guards without changing fit calculations.
Scoped tests passed 41 cases and whole-project Pyrefly passed. Whole-project
Ruff retains 789 unrelated findings. Full pytest passed 3,177 tests with 35 skips
and one missing-`python` PATH failure; that case passed with the environment's
executable directory on PATH.

The [portable evidence](../tree_simplification/results/input-preprocessing-evidence.json.gz)
contains candidate specifications, per-seed/per-video counts, paired changes,
selection identities, thresholds chosen on validation and quality/coverage reports. Models and
machine-specific paths are excluded. See the
[experiment commands](../tree_simplification/README.md#refit-changed-preprocessing-with-raw-features)
for preparation and fitting, and
[test-run instructions](../tree_simplification/README.md#test-run-and-review-thresholds)
for evaluation and threshold application. Test paths need `test_inputs`,
`test_courts`, `inpaint_root` and `pose_root` alongside development paths.
