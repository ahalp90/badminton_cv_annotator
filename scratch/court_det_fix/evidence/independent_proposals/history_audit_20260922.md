# Earlier fits and the current detector

## Resume

Eight previously approved automatic gallery fits survive with exactly unchanged
coordinates. All eight pass the current camera and player gates, but rank
between 6 and 50. The approved GX5 physical refit is still selected by both
current arms. The old broadcast winners already show the inward pattern.

The historical trace therefore prioritises ranking over recovering lost fits
for these examples. Next compare scoring rules on these fixed candidates.
Keep the distinct GX0 direction-selection loss in scope. No detector changes
or new visual judgements are part of this audit. The image-edge polarity
experiment is parked; no script or results were created.

## Question and session boundaries

The user recalls better fits in HTML galleries before the two orchestration
sessions and suspects a later regression. The session boundaries come from
the original handovers, preserved in
`../../.recovery/court-before-cleanup-20260921.tar.gz`:

- `scratch/court_det_fix/worklog/claude_session_15092026_14h32m/HANDOVER.md`:
  direction experiments, commits `afc23ff` through `9e0dec8`, 15 September.
- `scratch/court_det_fix/worklog/claude_session_16092026/HANDOVER.md`:
  line-identity work, commits `771578b` through `8429bd2`, 16 September.

Both were read directly from the archive. Commit messages are not used to
infer model authorship. The 14 September checkpoint `82d766d` predates both.

## Approved automatic fits remain available

The [original visual rulings](../../../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/automatic_axes_visual_judgements.md)
identify eight usable winners across six views. These combine the user's
judgements of different rankings; they are not one automatic selection rule.
A six-worker comparison finds all eight unchanged in the current G0 population.
All eight pass the current camera and player gates.

| View | Earlier candidate | Current full-pool rank |
| --- | --- | ---: |
| Amateur-2 frame 150 | `30:33` | 26 |
| Amateur-3 frame 0 | `43:22603` | 32 |
| Amateur-3 frame 0 | `43:22627` | 50 |
| ShuttleSet-03 scene 16 | `1:60` | 27 |
| ShuttleSet-03 scene 16 | `1:90` | 21 |
| ShuttleSet-03 scene 17 | `1:80` | 25 |
| ShuttleSet-03 scene 19 | `1:60` | 45 |
| ShuttleSet-21 scene 20 | `0:2` | 6 |

The [comparison records](history_audit_20260922/approved_winner_retention.json.gz)
preserve their exact corners, gate status and current selected geometries.
The selected Amateur-2 fit differs from the previously approved `30:33` by
9.252 working pixels at the most displaced corner. That is a concrete scoring
comparison to investigate. Displacement alone does not prove a visual
regression in every case.

## GX5: the approved fit survives

Commit `1b75435` records the user's judgement that the refitted courts
“perfectly hug the outer boundary of the white lines”. The saved start is
complete-search generation `6476322`, selected diagnostically using reference
geometry. The source is
`experiments/annotator/independent_court/recorded/player_guided/gx_close_refit.zip`.

The unchanged close-refit wrapper was rerun against today's core fitter and
the frozen GX input pack. Both starts match exactly. Maximum native-coordinate
differences from the saved results are 2.59e-10 px for the legacy refit and
7.83e-11 px for the physical-paint refit. The run exits 0.

The current W5 `(4,3)` case record contains the start within 4.42e-8 native px as
`line_template:rectangle_85795:template_72`. Its child is the old physical
refit to numerical precision. Both full-source and G1-plus-template arms select
that child, with gates applied. This example provides no evidence of a lost
fit or a fitter regression.

## GX0: generation and ranking are separate losses

The 14 September `gx0_control_measurements.json.gz` records approved candidate
89 and the two observed-direction controls, 1864 and 5144. The latter two
were judged essentially ideal. Their directions were chosen using the approved
control; they were not automatic winners. The same day's diagnostic already
reports a 7.662-working-pixel gap between candidate 89 and the closest automatic
candidate, `22:4588`. That loss predates both orchestration sessions.

The old automatic paint winner was `22:4588`; the old line winner was
`22:4579`. Both parents survive in the current full-source pool. Today's refit
of `22:4588` is 3.778 working pixels from approved candidate 89, while the
selected `22:4579/child` is 12.328 pixels away. Current paint scores favour the
latter: 0.43798 versus 0.41334. These are disagreements with a saved approved
fit, not new visual ratings or errors against independent ground truth.

Thus the better surviving fit is not lost by refitting in this example. The
current score orders it below a more displaced fit. The current span-weighted
paint score was introduced in the later W5 revision `25a4d3e` on 20 September.
The initial W5 score in `a1fcc9d` already multiplied image paint support by
exclusive fragment support. Both equal-weight and span-weighted versions favour
`22:4579` over `22:4588` on these two saved parents and children. The reversal
therefore cannot be attributed to span weighting alone.

## Core-code check

There is no diff between `82d766d` and current HEAD for `detector.py`,
`assignment.py`, `stripe_observations.py`, `fixed_stripe_refit.py` or
`paint_geometry.py` under `experiments/annotator/independent_court/`.
This does not clear callers, input selection or ranking. The GX5 replay checks
one real fitted result in addition to the source comparison.

## Broadcast bias predates both sessions

The broadcast extension archive is byte-identical to its `82d766d` version.
Its 20-case input pack equals the current frozen pack. All 20 current images
match the original line-extraction image hashes.

The [geometry comparison](history_audit_20260922/broadcast_geometry_history.json.gz)
uses the same 18 views as the wider numeric summary. It excludes the two
unverified views. Positive distances point inward; units are working pixels.

| Median edge displacement | Old physical stripe winner | Current full-source winner |
| --- | ---: | ---: |
| Left | +3.32 | +3.83 |
| Right | +3.52 | +2.23 |
| Far | +0.19 | +0.40 |
| Near | +2.10 | +1.33 |

The old legacy-stripe winners show the same sign pattern. These measurements
use one static grid per video, not independent scene annotations. They show
that the historical geometry already had the inward pattern against the same
reference. They do not establish exact paint-boundary accuracy.

For SS03-34, the old gallery selects the same geometry under both stripe
models: upper-left corner `(317.849, 215.273)`. The current corner is
`(318.744, 215.940)`, a movement of approximately `(+0.894, +0.667)` pixels.
The older corner was already inward of the shared grid. The recurring bias
is therefore not first introduced by the two later sessions.

## Other completed work

The [WebUI follow-ups reproduce locally](../webui_followups_20260922/local_replay/README.md).
All 19 numerical inputs match the pinned revision; discrete outcomes match,
with numerical differences below 5.83e-11. Original return files are unchanged.
These returns are supplementary evidence, not the first evidence of attainable
good fits.

## Replay records

The [verification record](history_audit_20260922/verification.json.gz) lists the
unchanged core files and GX5 coordinate comparisons. The
[GX5 replay](history_audit_20260922/gx5_close_refit_current.json.gz) preserves
both paint models; the [GX0 comparison](history_audit_20260922/gx0_old_to_current.json.gz)
preserves the three old geometries and their nearest/current selected fits.
Corner comparison allows a whole-court 180-degree rotation and reports the
maximum corresponding-corner distance in 960×540 working pixels.
The six-view comparison is reproducible with
[compare_approved_winners.py](history_audit_20260922/compare_approved_winners.py),
run from the repository root with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1`. It uses six processes. Both executions exit 0; the second
adds the gate and rank check. Scoped Ruff passes, exit 0.

GX5 replay command, from the repository root:

```bash
court_history=scratch/court_det_fix/evidence/independent_proposals/development/player_guided
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  ~/.venvs/badminton-cicd/bin/python \
  "$court_history/20260913/close_refit/run_close_refit.py" \
  --inputs scratch/court_det_fix/frozen_views/packs/gx_extension_inputs.json.gz \
  --caps "$court_history/20260909/gx_trace/rectangle_cap.json.gz" \
  --trace-directory "$court_history/20260909/gx_trace" \
  --legacy "$court_history/20260908/marking_refit" \
  --output /tmp/gx5_close_refit_current.json.gz
```
