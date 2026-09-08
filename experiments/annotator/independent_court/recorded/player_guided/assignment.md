# First frozen marking-assignment test, 9 September 2026

**The first partial assignment matcher selects 5 accurate courts in 20 frames.**
The recorded existing and bidirectional scores select 7/20 and 8/20. This result limits the tested raw-line
representation and objective. It does not settle whether a fuller court-graph
matcher is useful. Production keeps CourtKeyNet.

The larger goal is reliable detection of the main playing court in amateur
video, using lines and player evidence instead of CourtKeyNet, the current
learned court detector. Court geometry supports the video auto-annotation
pipeline. This experiment makes no change to that pipeline.

The question was whether assigning observed line groups to named court markings
could resolve wrong court interpretations. No graph matcher had been run before
this experiment. The [protocol](assignment_protocol.md) was committed before the
full run; the earlier centred-stripe ledger result was a different experiment.

## What was held fixed

The comparison uses twenty development frames from seven source videos. Nearby
frames are not independent validation. All 714 original/refitted geometries
from the previous replay remain unchanged; 682 pass the shared saved gates.
An extra gallery probe is excluded. One frame has no candidates and stays in
every denominator. Reference labels enter only after ranking.

Accuracy means worst-corner error at most **15 pixels in 1280×720 coordinates**,
including off-screen corners. Visible-landmark root mean square (RMS) error is
reported separately in the [results](assignment_results.json.gz). It measures
the size of reprojection errors at the manually clicked markings. Neither metric alone
establishes that every stripe received the correct marking identity.

The new matcher groups collinear DeepLSD fragments once per image. Each group
can match at most one named marking, and each marking can match at most one
group. The centre is one identity with two finite painted intervals. A common
frozen homography projects all eleven marking identities consistently.

Matching maximises **floor support**, the mean of two measurements. **Forward
support** measures how well each visible marking is covered by its assigned
group's fragments. **Reverse support** measures how much observed group length
the assigned markings explain. Both range from zero to one. Distances use a
two-pixel Gaussian scale in working images capped at 960 pixels. All visible
observations remain in the comparison, including those outside a candidate's
court.

The final score is `(3 × floor support + net support) / 4`. Net support measures
alignment of the predicted net with observed lines; its saved value is reused.

This is a small bipartite assignment problem, solved directly. There is no
partial-state search or search-budget cutoff. All eligible frozen alternatives
receive complete scores. The experiment does not recover proposals discarded
before the replay, enumerate alternative assignments within one geometry, or
test junction recognition and stripe-centre uncertainty.

## Results

The recorded existing score measures support near projected court markings in
broad angle groups. The bidirectional score also measures detected line length
explained inside the proposed court. Both are reused from the preceding replay.

The independent-support control uses the same groups, distances and objective.
It lets markings and groups independently reuse their best counterpart.
The raw ledger control compares explained groups and missing visible markings.
It omits the morning experiment's paint filter, centring, new refits and posts.
Its result therefore does not reproduce or overturn that experiment's 10/20.

| Selector | Accurate leading fits / 20 |
|---|---:|
| Recorded existing score | 7 |
| Recorded bidirectional score | 8 |
| Independent group support | 5 |
| Partial assignment | 5 |
| Raw differential ledger control | 4 |

Assignment selects the same winner as independent support in **17/19 non-empty
pools**. Enforcing exclusivity changes two winners relative to that control:
yellow frame 90 improves from 694.26 to 21.92 pixels, and letterboxed frame 78
worsens from 16.35 to 31.94 pixels. Both still miss
the accuracy cutoff. Most failures of the new scoring setup already occur
before exclusivity is imposed.

The three successes lost relative to the bidirectional score have assignment
errors between 15.96 and 17.86 pixels. Their counts are sensitive to the
historical cutoff; the full errors remain visible in the table below.

An accurate geometry exists in **13/20 pools**, both before and after the shared
gates. Seven pools therefore need better available geometries to pass this
metric. That says nothing about whether their raw images contain sufficient
evidence: this experiment does not trace earlier proposal losses.

The table gives worst-corner errors in pixels. Full frame IDs, visible errors,
complete rankings, assignments and unmatched groups are in the results file.

| Frame | Existing | Bidirectional | Independent | Assignment | Raw ledger |
|---|---:|---:|---:|---:|---:|
| Yellow 14 | 187.26 | 696.24 | 696.24 | 696.24 | 152.65 |
| Yellow 90 | 21.92 | 21.95 | 694.26 | 21.92 | 164.96 |
| Yellow 156 | 641.40 | 653.42 | 653.42 | 653.42 | 144.08 |
| Letterboxed 45 | 17.86 | 11.88 | 17.86 | 17.86 | 401.87 |
| Letterboxed 58 | 24.48 | 23.23 | 23.23 | 23.23 | 401.95 |
| Letterboxed 78 | 31.94 | 79.26 | 16.35 | 31.94 | 16.35 |
| Centre 36 | 3.00 | 3.13 | 3.13 | 3.13 | 130.80 |
| Centre 64 | 5.67 | 4.59 | 15.96 | 15.96 | 4.59 |
| Centre 71 | 4.92 | 4.20 | 17.13 | 17.13 | 130.35 |
| Amateur 1, frame 54 | 60.67 | 81.44 | 81.44 | 81.44 | 79.56 |
| Amateur 1, frame 5352 | 58.10 | 48.99 | 48.99 | 48.99 | 48.99 |
| Amateur 2, frame 150 | 12.44 | 12.44 | 12.44 | 12.44 | 12.44 |
| Amateur 2, frame 28019 | Empty | Empty | Empty | Empty | Empty |
| Amateur 3, frame 0 | 11.54 | 13.87 | 13.87 | 13.87 | 37.04 |
| Amateur 3, frame 10514 | 27.83 | 27.83 | 27.83 | 27.83 | 27.83 |
| Amateur 3, frame 17174 | 38.06 | 38.06 | 38.06 | 38.06 | 29.14 |
| Amateur 3, frame 24515 | 124.87 | 124.87 | 124.87 | 124.87 | 70.53 |
| Amateur 4, frame 0 | 1662.55 | 136.01 | 136.01 | 136.01 | 136.01 |
| Amateur 4, frame 319 | 13.62 | 13.62 | 13.62 | 13.62 | 13.62 |
| Amateur 4, frame 13782 | 5.21 | 5.21 | 5.21 | 5.21 | 5.21 |

## What the assignments explain

**Yellow frame 156 retains the baseline/service-line alias.** The matcher picks
the same wrong 653.42-pixel geometry shown in the previous failure figure.
It assigns the visible near-baseline evidence to the near long-service marking
and leaves the interior near long-service evidence unmatched. Its predicted
baseline is off-screen. The accurate original remains available at 14.89 pixels.

![The accurate original above and wrong selected refit below, with the unused interior marking indicated](marking_refit_failure.jpg)

This figure comes from the preceding experiment; the new matcher selects the
same lower-panel geometry. Magenta shows the candidate and orange the manual
reference. The arrow indicates the unused yellow court line.

The wrong fit has forward/reverse support of 0.481/0.072, compared with
0.390/0.071 for the accurate original. Their net scores are nearly identical
(0.907/0.906). The objective still rewards stronger alignment elsewhere despite
the unexplained interior marking. Keeping observations outside the candidate
footprint fixes an omission in bookkeeping; it does not make this objective
discriminate the alias.

The observed baseline also splits into two long groups, about 958 and 948
working pixels long. Their separation is 4.2–4.8 pixels, above the three-pixel
grouping tolerance. The accurate fit assigns one to the baseline; the wrong fit
assigns the other to the long-service marking. These nearby responses show why
raw groups need not correspond to distinct physical stripes.

**Letterboxed frame 58 and centre frame 64 retain poorer geometric variants.**
Their assignment winners have 23.23 and 15.96-pixel corner errors, versus
available best errors of 6.36 and 4.59 pixels. Selected visible-landmark RMS
errors are 3.53 and 4.98 pixels, versus 1.19 and 1.74 for those best candidates.
Thus the difference is visible as well as extrapolated. Independent support
selects these same poorer variants.

A post-run diagnostic compared support from all raw fragments with support
from the best single group. It examined only these three named frames and used
labels to identify the best available comparator. For centre frame 64's best
geometry, mean forward support falls from 0.704 to 0.566 under the single-group
restriction. For its selected geometry, it falls from 0.713 to 0.613.
Grouping therefore loses useful evidence. Even the all-fragment forward score
favours the poorer geometry in each of the three inspected cases. This check
isolates a measurement loss. It does not test an improved selector.

## Next useful test

Keep geometry frozen while testing observations that preserve plausible stripe
positions and fragment membership alternatives. Raw collinear groups are too
strong an assumption to stand in for physical stripes without that check.
The yellow counterexample also needs evidence that distinguishes marking
identity despite stronger residual alignment elsewhere. Finite junction or
termination evidence is one hypothesis to test where it is observable.

These results do not justify widening search around the current objective.
They also do not justify abandoning the graph approach. Later proposal search
should record unique states, duplicate visits and budget cutoffs. Exact state
keys are a reasonable starting point; hashed keys only become useful if their
cost is material. No acceptance or production-readiness claim follows from
the present development comparison.

## Replay and verification

Run from the repository root in the existing NumPy/OpenCV/SciPy environment:

```bash
PYTHONPATH=src:. python -m experiments.annotator.independent_court.run_assignment \
  --replay experiments/annotator/independent_court/recorded/player_guided/marking_refit_replay.zip \
  --output /tmp/assignment_results.json.gz
```

The full local CPU comparison took 132 seconds for 714 geometries, including
682 eligible geometries scored by the new selectors. This is one descriptive
runtime measurement. No video decoding, model inference or remote job is needed.

[Post-run diagnostics](assignment_diagnostics.zip) include their script, frozen
results and output. Extract the archive, then run `diagnose.py` with the same
`PYTHONPATH` from the repository root. Labels are used there only to choose
the best available geometry for inspection.

The eleven new tests pass, including a real-replay smoke check. The combined
assignment, detector and player-guided tests pass (32 tests). Scoped Ruff and
Pyrefly pass. Whole-project Pyrefly still reports the three known missing
optional vision-language model imports. All 714 geometries and saved baseline
scores match the source replay. Shuffling candidate order preserves all twenty
rankings. A fresh 24-geometry replay exactly reproduces the saved results.
Independent Fable review reproduced the headline counts and a sampled case's
assignment scores. It found no verified implementation defect within its scope.
