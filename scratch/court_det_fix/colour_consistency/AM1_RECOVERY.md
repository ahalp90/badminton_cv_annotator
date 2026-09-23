# Am1 recovery: extra line groups and net support

**The expanded Am1 pool contains a promising court that net support selects
automatically.** Its maximum visible-landmark disagreement is 7.2 working
pixels, versus 45.0 for the expanded pool's paint-score winner. Saved GX5 and
Letterboxed45 selections stay unchanged. Visual review is pending.

The net-ranking trial covers three known development views and four pool
configurations. A separate generation check covers GX5, Am4-319 and SS21-10. Am1 geometry and earlier net measurements on Am1 and GX5
informed the design, so these views are not independent validation. The result
could help the scene-level auto-annotator, but production generation, fitting
and ranking are unchanged. Colour trials remain closed; automatic stripe
polarity remains preferred for later fitting integration.

## What was tested

The question was whether existing line and net evidence could fix Am1, a
difficult amateur court view, without colour handling or a new detector.

The existing W5 detector combines three proposal sources, refits their courts,
and ranks them by paint support. Am1's saved selection mistakes part of the net
for a court boundary. The investigation separated two possible failures:
missing useful proposals, and choosing badly among proposals that exist.

1. **Proposal coverage:** add the three pairwise intersections of the three
   longest merged lengthwise lines as extra vanishing points. These define
   additional groups of lines that can form candidate rectangles. Keep the
   existing 4,096-rectangle and 256-template caps. Reuse unchanged saved
   proposals and refits; score and refit new proposals with unchanged W5 code.
2. **Selection:** project each eligible court's regulation-height net into the
   image. Compare its two tape halves and two posts with all generic detected
   image fragments. Prefer the first court in the existing ranking with strong
   support for both tape halves and at least one post. Otherwise keep the
   existing selection. Camera and full-court player gates remain unchanged.

The net rule was fixed before scoring the expanded pool: 24 equally spaced samples per piece;
4 working pixels distance tolerance; 8 degrees direction tolerance; 2 pixels
extent margin; at least 75% support per required piece. Only in-frame samples
enter the denominator. No manual paint colour, post annotation or player
selection is needed. Low or missing net support does not reject a court.

Am1 was evaluated with both its saved and expanded pools. GX5 and Letterboxed45
used their saved pools as retention controls for the net preference. These two
controls therefore do not test the combined generation-and-ranking change.
Am1's annotated geometry guided the seed investigation. Earlier reference-net
measurements on Am1 and GX5 informed the net rule, including allowing one
missing post. These are development cases, not independent validation.

## Results

Reference points are used only after automatic selection. Am1 has 21 annotated
visible landmarks. The working image is 960 × 540; its native image is twice
that size in each dimension. Smaller reference disagreement is a diagnostic,
not visual approval. Some amateur references mark paint centres and others
mark outside edges, so small drift is inconclusive.

| Am1 automatic selection | Maximum visible-landmark disagreement | Median visible-landmark disagreement | Maximum corner disagreement |
| --- | ---: | ---: | ---: |
| Extra groups, existing paint ranking | 45.0 working px | 11.3 working px | 98.5 working px |
| Extra groups, net preference | 7.2 working px | 1.2 working px | 32.7 working px |

The net preference chooses the original rank-2 court. Its two tape halves have
23/24 and 22/24 supported samples; both posts have 24/24. All samples are inside
the image, so this result is not caused by a small visible denominator. The
paint-score winner has tape support of 0/24 and 3/24. Its left post alone is
strong, which cannot satisfy the rule.

There are 15 strongly net-supported courts among the expanded pool's 142
full-court eligible candidates. The preference selects the first in the
existing paint ranking; it does not optimise reference error. A rank-3 court
has smaller extrapolated corner error but worse visible-landmark agreement.
The trial does not use that hindsight to change its choice.

The saved Am1 pool has no strongly supported full-court candidate and keeps its
wrong selection. Net preference cannot recover a court absent from that pool.
GX5 keeps its existing selection despite a missing right post. Letterboxed45
has no qualifying alternative and also keeps its existing selection. Thus one
of four evaluated pool configurations changes its output.

The large remaining corner error is mainly outside the image. Review visible
court markings in the [three-case gallery](am1_gallery/index.html), rather than
using extrapolated corners alone to decide whether the result is useful.

## Why this differs from the earlier net no-go

The observed Am1 service lines and sidelines already survived line extraction.
Their useful rectangle was omitted because the two sidelines belonged to
different selected vanishing-point groups. Adding more of the original ranked
groups did not recover that pair. The three automatic intersections do.

The earlier net review measured the old proposal pool. Its negative result
remains valid for that pool. The expanded pool changes the question: net
support can now distinguish useful alternatives already present. It uses
observed image fragments to check predicted net geometry. A separate post
assembly detector is unnecessary for this trial.

## Decision and limits

**Keep this as a promising experiment for visual review.** If the Am1 overlay
is useful, the next check is final selection on a small set of existing good
courts after applying both changes together. Generation retention alone cannot
establish that newly added candidates will not outrank those courts.

The three-view result does not measure false acceptance on background controls.
It also lacks a correct but net-obscured winner competing against a strongly
supported wrong court. Deeper candidates in Am1 can pass the net rule while
fitting landmarks poorly; preserving the existing paint ordering matters.

Generic fragments can be walls, stands or other court markings. Net projection
assumes a centred pinhole camera, square pixels and regulation heights. There
is no occlusion mask. Missing support remains neutral, so the rule also leaves
existing false acceptances unresolved. No threshold sweep or new net detector
is justified by this result.

The bounded generation checks preserve the existing selected source geometry
on GX5, Am4-319 and SS21-10. GX5's template parent has exactly the same corners
and homography. Its camera-error value differs by about 3 × 10⁻¹⁷, which changes
neither eligibility nor geometry. Am4-319 and SS21-10 use untouched G0/G1 source
parents. This establishes availability, not final selection after the new
candidates are scored. The [retention record](am1_seed_retention.json.gz)
separates exact geometry from exact gate-metadata equality.

## Records and reproduction

- [Proposal replay code](am1_recovery_trial.py) and [compact results](am1_recovery_trial.json.gz).
- [Net preference code](am1_net_selection_trial.py) and [all scored support rows](am1_net_selection_trial.json.gz).
- [Shared-template gallery builder](build_am1_gallery.py).
- The full expanded pool is local at `local_scratch/external_delegate/20260923-am1-recovery-trial/seeded_pool.json.gz` (16 MB compressed). Preserve it; the compact summary is not a replacement for that pool.

From the repository root, set numerical threads to one:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/src"
~/.venvs/badminton-cicd/bin/python \
  scratch/court_det_fix/colour_consistency/am1_recovery_trial.py \
  scratch/court_det_fix/colour_consistency/am1_recovery_trial.json.gz \
  --pool-output local_scratch/external_delegate/20260923-am1-recovery-trial/seeded_pool.json.gz
~/.venvs/badminton-cicd/bin/python \
  scratch/court_det_fix/colour_consistency/am1_net_selection_trial.py \
  --am1-pool local_scratch/external_delegate/20260923-am1-recovery-trial/seeded_pool.json.gz \
  --output scratch/court_det_fix/colour_consistency/am1_net_selection_trial.json.gz
```

The expanded replay reused 564 parents and evaluated 204 changed parents.
It produced 768 parents and 443 valid children. Its final diagnostic run took
267.8 seconds; a separate pool-preservation replay reproduced its choices and
selected geometry in 293.6 seconds. The net trial took 30.6 seconds for four
pool configurations across three views: 16.4 preparation, 3.3 projection and
10.8 support scoring. This includes reranking assertions and all camera-ranked
candidates; it is not production runtime.

Saved comparator identity, candidate reuse, source geometry, gate order and
coordinate checks passed. The seed checks cover finite/infinite intersections
and restoration of the temporary generator hook. Scoped Ruff and syntax checks
passed; whole-project Pyrefly exited 0 with 39 existing suppressions. Final
review reproduced the chosen geometry, 21-landmark errors and all four support
fractions. None of the 12 fragments supporting the chosen net also supports
that court's painted lines under the same tolerances. Gallery checks are
recorded in the [worklog](PLAN.md).
