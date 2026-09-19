# W5 executor prompt — Luna Max implementation contract

You are implementing and running the W5 holistic court-detector pilot. The frontier steering model owns research decisions. Your job is to follow this contract exactly, make ordinary implementation choices locally, and return enough **raw and derived evidence** for the steering model to decide what happens next. The first pass is an evidence census with provisional rankings, not a fixed-threshold acceptance test.

If a local detail is missing, inspect the checkout and follow the existing `scratch/court_det_fix` conventions. Do not ask the human for routine path, launcher or formatting choices. If you find a contradiction that changes the meaning of the experiment, stop that affected comparison and report it to the steering model.

Commit as you go. Beyond the natural commit contents, these artefacts must be committed: ../w5_webui_checkins/commit_for_webui.md

## 1. What this run is testing

Run three arms over the same bounded automatic proposal union:

| Arm | What it means | What it isolates |
| --- | --- | --- |
| **A — legacy** | Existing nominal geometry and historical line/paint ranking | The saved baseline on the same proposal union |
| **B — evidence + provisional rank** | Keep each parent geometry fixed and measure the complete court using physical paint, fragment ownership, both directions, junctions, players and camera diagnostics | Whether the combined evidence contains a cross-view ranking signal without prematurely hard-gating it |
| **C — refit + evidence** | Attempt one existing fixed-identity whole-court refit for each parent; measure valid children alongside parents with B's evidence pass | Whether that local refit adds value without erasing a good parent |

This is research code only. Do not change production court behaviour, saved approvals, manual annotations or historical experiment records. Do not run CourtKeyNet inference.

Base repository state: `ahalp90/badminton_cv_annotator`, branch `fix/court-det`, reviewed at commit `82d3b871bfb784f91b283761148eaa0d6501c732`.

Record the actual checkout commit and any local source changes that matter.

## 2. Where new work lives

Use this scratch area unless the local checkout already has an equivalent W5 directory:

```text
scratch/court_det_fix/w5_holistic/
```

Suggested layout:

```text
w5_holistic/
  run_w5.py
  verifier.py
  render_gallery.py
  test_w5.py
  runs/<run_id>/
```

Large candidate arrays and response tensors stay under the run directory on the execution host. The compact return packet lives in the same run directory.

Experiments expected to take more than a few minutes should run on Carmack using the existing project environment. Do not install system packages or build a new distributed execution setup. Use ordinary CPU parallelism; up to 10 workers and one BLAS/OpenCV thread per worker is enough for the first pass.

A small reusable change is allowed in:

```text
experiments/annotator/independent_court/junction_observations.py
```

Add an optional `centres=` argument, defaulting to the current `detector.SEGMENTS_M`, so W5 can pass `paint_geometry.CENTRE_SEGMENTS_M`. Keep existing default behaviour and tests unchanged.

## 3. Names used below

Aliases are relative to the repository root:

```text
S = scratch/court_det_fix
D = S/direction_agreement
I = S/line_identity
N = S/next_steps_20260916
H = S/frozen_helpers_20260914
```

Proposal sources:

- **G0** — original generation population
- **G1** — paint-filtered direction-generation population

Court variants:

- **parent** — saved G0/G1 court geometry
- **child** — one valid W5 refit of a parent

Evidence models:

- **legacy profile** — historical coarse pixel paint test used by L2
- **exclusive stripe support** — fragment-to-marking evidence from `stripe_observations.py`
- **W5 evidence pass** — the sample-level fragment, photometric, junction, player and camera readouts defined below; the first-run reduction is provisional

## 4. The nine views

Use `I/shared.py::CASES` and these exact case IDs:

```text
gxBQ_window_00_frame_0
gxBQ_window_00_frame_5
am2_window_00_frame_150
am2_window_01_frame_28019
am3_window_00_frame_0
shuttleset_03_scene_0017
shuttleset_03_scene_0019
shuttleset_03_scene_0016
shuttleset_21_scene_0020
```

Load the automatic observations from the frozen packs:

```text
S/frozen_views/packs/gx_extension_inputs.json.gz
S/frozen_views/packs/marking_refit_inputs.json.gz
S/frozen_views/packs/broadcast_extension_inputs.json.gz
```

Resolve native frames through the existing loaders. Record, per view:

- native image path and dimensions
- whether the image is a source frame or cached/median thumbnail
- player-foot source
- whether reliable **same-image** person boxes exist for photometric masking

Use cached fragments and cached player evidence. Boxes from another frame are not valid photometric masks. The expected provenance from the frozen records is:

- reliable same-image boxes: GX0, Am2-150, Am2-28019, Am3-0
- GX5: anchor image is frame 5 but the chosen boxes are from frame 6 — not a valid mask
- all four broadcast views: cached images are median thumbnails while boxes come from a source frame — not valid masks

If the local frozen records disagree, record the actual provenance and use only genuinely same-image boxes.

## 5. Build the bounded parent population

### G0

Prefer:

```text
S/frozen_views/baseline_generation/<case>.json.gz
```

When that compact generation record is absent, reconstruct the unchanged global generation selection from:

```text
S/worklog/checks/independent/player_guided/20260914/automatic_axes/collected/all_camera/<case>.json.gz
```

Use the behaviour of:

```text
N/L2_scoring/run_l2_scoring.py::load_generation_population
N/L2_scoring/run_l2_scoring.py::reconstruct_generation_entries
```

as the specification. Record `g0_source = direct | replayed` per view.

Do not substitute the uncapped camera pool and call it G0.

### G1

Use the saved generation-stage `paint_observations` population from:

```text
I/runs/line_identity_20260915_222437/matcher/paint_observations
```

Use the exact generation-stage loading logic from the L2 script. Preserve the filtered-observation source for legacy replay, but W5 B/C must measure against the **original raw observations** so a line rejected by the earlier paint filter is not permanently lost.

### Union

Concatenate G0 then G1. Each source normally contributes at most 256 parents, so the ordinary W5 union is at most 512 parent origins per view.

Do not impose another score cap. Exact duplicate geometries may share expensive computation, but every origin remains addressable in the outputs.

Use this deterministic origin order:

```text
source_order: G0 = 0, G1 = 1
origin_index: saved order within that source
kind_order:   parent = 0, child = 1
```

This order is the final tie-break after W5 scores. Preserve it even when duplicate geometries share cached responses.

## 6. References stay outside automatic ranking

Evaluation may use:

```text
D/runs/direction_agreement_20260915_144900/e3/<case>.json.gz["control"]
```

plus the existing manual landmarks/corners recorded by the frozen cases.

Keep provenance explicit:

- manual visible landmark / manual corner
- approved supplied-direction control
- extrapolated corner

GX0's supplied-direction winner 89 is an approved control, not a manual corner annotation.

Automatic code must not receive reference coordinates, reference errors, human rulings or direction-fit witnesses.

For diagnostics only, also load these existing automatic candidates from the uncapped records when available:

```text
Am2-150    30:33    approved / very good
SS03-19    1:60     usable
Am2-28019  184:4123 rejected false paint winner
SS03-19    165:6702 rejected false/hallucinated paint winner
```

Score them with W5, but do not insert them into the bounded automatic union unless G0/G1 already contains them.

## 7. Replay preflight

Before interpreting W5, reproduce the saved L2 G0/S0 and G1/S1 winner identities on the four L2 comparison views.

The purpose is simple: establish that the intended populations and legacy scorer are loaded. Do not rebuild the whole historical experiment chain.

Record actual imported module paths for ambiguous helper basenames. In particular, preserve the known distinctions between the L2 seed snapshot, the frozen helper versions used by L1/L3, and the older instrumented producers described in `SOURCE_LEDGER.md`.

A small floating-point difference is not interesting if population membership, eligibility and winner identity are unchanged. An unexplained population or winner change is interesting; stop that affected comparison and report the source/path mismatch.

Run the W5 ranker once more after permuting parent input order. After applying the explicit tie key above, the ordered IDs must match. Compare the ID sequence directly.

## 8. Arm A — faithful legacy baseline

A must preserve L2's historical behaviour on the G0+G1 union.

Legacy eligibility is exactly:

```text
camera_error is not None
camera_error <= 0.1
legacy profile.score is not None
```

Report both:

- line winner: maximise `stripe.exclusive.score`
- paint-first winner: maximise `(profile.score, stripe.exclusive.score)`

The **paint-first winner is A's primary comparator** for every view. The line winner is secondary context. Never choose whichever legacy rule looks better after seeing the reference.

A uses the old nominal `detector.SEGMENTS_M` convention exactly as the replayed historical code expects.

## 9. B/C geometry and physical paint convention

Use 960×540 working coordinates. Scale to/from native images explicitly. Output outside-boundary corners in TL/TR/BR/BL order.

For B and C pass:

```text
paint_geometry.CENTRE_SEGMENTS_M
```

through stripe measurement, junction measurement and refitting.

There are 12 finite intervals representing 11 named floor markings. The centre line is one marking split into two finite intervals by the unpainted net gap. The net itself is not a floor marking.

Use the 40 mm paint convention and the existing centre/edge offsets from `paint_geometry.py`. Do not mix the physical centres with the old nominal template inside one B/C calculation.

## 10. B/C hard validity and historical diagnostics

For B/C, do not turn the earlier full-court predicate into a new admission gate.

A parent remains in the W5 evidence table whenever its court geometry is valid enough for the existing downstream measurements to run. A child has the additional numerical/refit validity requirements in section 14. Hard rejection is for things such as non-finite projection, invalid quadrilateral/projective geometry, failed source provenance or an invalid refit — not for weak empirical evidence.

For every candidate, compute and save the historical full-court fields:

```text
geometry_valid
one_player_fraction
two_player_fraction
camera_error
```

and the historical predicate:

```text
geometry_valid
one_player_fraction == 1.0
two_player_fraction >= 0.5
camera_error is not None
camera_error <= 0.1
```

Also save the weaker historical geometry+camera subset:

```text
geometry_valid
camera_error is not None
camera_error <= 0.1
```

These subsets are diagnostic comparisons. They let the steering model see whether the old player/camera assumptions would have removed a useful court. They do **not** remove the candidate from the initial B/C evidence census.

Do not require the legacy pixel profile for B/C.

The camera error comes from a simplified camera model. Player fractions and camera error are clues about a complete court, not physical certification. Keep their raw values visible.

## 11. Compute the W5 evidence pass

Prepare original raw-fragment observations once with stable fragment IDs and the existing union-length weights from `stripe_observations.fragment_weights`.

For each fixed geometry, call:

```python
stripe_observations.measure(
    homography,
    observations,
    size,
    centres=paint_geometry.CENTRE_SEGMENTS_M,
)
```

then:

```python
stripe_observations.score_model(evidence, weights, position_count=3)
```

Use its exclusive fragment assignment. Preserve the existing aggregate stripe outputs as useful context; do not make them the only W5 signal.

### 11.1 `C(s)` — exclusive fragment support at one marking sample

For a marking `m` and one of its sampled physical-centre locations `s`, let:

```text
C_m(s) = max forward response among fragments whose exclusive assignment is
         (marking = m, position = the tested centre/edge position),
         maximised over the three allowed positions.
```

Equivalent pseudocode:

```python
allowed[p, f] = (
    assignment.marking[f] == m
    and assignment.position[f] == p
)

C[s] = max(
    evidence.forward[m][p, s, f]
    for p in 0..2
    for f with allowed[p, f]
)
```

If no assigned fragment supports that sample, `C(s) = 0`.

Do not reuse one fragment for several incompatible marking identities.

### 11.2 Photometric evidence — keep the raw contrast, not just the old vote

Use the physical **marking centreline sample** as the photometric anchor. Let the projected marking's image-space normal define the sampling direction.

At each along-marking sample:

- try centre offsets `[-4, -2, 0, 2, 4]` working pixels along the normal;
- for each offset, compare the centre intensity with pixels 6 working pixels to either side along the same normal;
- for every fully observable offset, calculate `min(centre - side1, centre - side2)`;
- save `ridge_contrast(s)` as the maximum of those values;
- if no offset has all three required pixels available, mark the sample photometry `unknown`.

Use grayscale bilinear sampling, matching L2's existing ridge test.

For backwards comparison also derive:

```text
P10(s) = 1 if ridge_contrast(s) >= 10
       = 0 otherwise
```

The value `10` is the historical L2 contrast probe. It is **not** a W5 acceptance threshold. Retaining `ridge_contrast(s)` is what lets the steering model see whether that old value is sensible across cameras and change the reduction without re-reading pixels.

A sample is unavailable when a required pixel is outside the image or lies inside a reliable same-image person mask. When no reliable same-image mask exists, do not mask people; record `photometry_occlusion_aware = false` for that view.

Use the stripe model's 64 along-interval sample locations. That count is the inherited sampling resolution, not a minimum-evidence gate.

### 11.3 Per-marking readouts

For each named marking, save at least:

```text
q_geom_m    = mean(C(s)) over its geometrically visible sample positions
q_paint10_m = mean(C(s) * P10(s)) over samples with known photometry
```

For the split centre marking, concatenate its two visible finite intervals before taking each marking summary.

If a marking has no known photometry, `q_paint10_m` is missing rather than zero. Do not require 8 known samples or any other new count before saving a marking score.

Also save:

- number of geometrically visible sample positions;
- number and fraction with known photometry;
- projected visible span;
- exclusive fragment count / support summary;
- compact raw `ridge_contrast` summaries, including enough quantiles or the retained sample array to inspect the distribution later.

### 11.4 Directional balance and provisional `Q` readouts

Lengthwise named markings are indices 0–4 in `assignment.MARKINGS`:

```text
left_doubles, left_singles, centre, right_singles, right_doubles
```

Transverse markings are indices 5–10:

```text
far_baseline, far_long_service, far_short_service,
near_short_service, near_long_service, near_baseline
```

For each direction, calculate the mean of the available per-marking values and record how many markings/samples contributed.

Derive two transparent readouts:

```text
Q_geom    = min(mean_lengthwise_q_geom, mean_transverse_q_geom)
Q_paint10 = min(mean_lengthwise_q_paint10, mean_transverse_q_paint10)
```

A `Q` readout exists when **each court direction contributes at least one named marking with that readout**. This is a structural requirement — both projective directions must say something about a court — not a tuned count threshold.

Do not require 2 markings, 32 samples or any other empirical observability count in the first pass. Save those counts as data so the steering model can see whether sparse evidence is genuinely unstable.

`Q_paint10` is a useful historical/provisional paint-backed ordering, not the final W5 rule. `Q_geom` shows what the same exclusive assignments say before the inherited photometric cutoff is applied. The steering model may choose a different global photometric reduction after inspecting the five-view evidence.

## 12. Junction/termination evidence

Update `junction_observations.py` so the existing geometry can be evaluated at supplied physical centres:

```python
measure(..., centres=paint_geometry.CENTRE_SEGMENTS_M)
```

The parameter must replace every template lookup that defines the six sites: use `centres[6:]` for the transverse markings, `centres[2, 0, 0]` for the centre-line x position, and the supplied centre-line intervals when deciding whether the far/near continuation is physically painted. Do not leave a hidden `SEGMENTS_M` lookup inside the physical W5 path.

The default `centres=detector.SEGMENTS_M` must preserve old behaviour.

The existing helper contains historical classification constants such as `MIN_VISIBLE_SAMPLES = 8`, `PRESENT_SUPPORT = 0.55` and `ABSENT_SUPPORT = 0.20`. Preserve its old outputs for comparison, but **do not use those cuts as a W5 veto in the first pass**.

For W5, expose or compute a raw arm record at each of the six physical sites. Use the existing physical arm geometry (`ARM_START_M`, `ARM_END_M`, `ARM_SAMPLES = 16`) and save, for every far/near/left/right arm:

- fragment-support response on the usable samples;
- usable sample count and projected span;
- raw per-sample ridge contrast using the same photometric sampling as section 11;
- historical `P10` fraction where photometry is known;
- whether reliable same-image person masks were available.

Do not discard the arm merely because it has fewer than the historical eight usable samples; report the amount of evidence instead. If no usable sample exists, the arm is unknown.

For a far/near continuation that the **physical template says should be unpainted**, save a contradiction-evidence record containing the fragment support and the photometric continuation evidence. This is the clue the steering model cares about: does a candidate place a real-looking painted continuation through a place where that court identity says the centre line should stop?

Missing expected paint is not negative evidence by itself. Weak, clipped or masked evidence stays unknown.

There is **no hard junction veto in the initial W5 pass**. If the five-view controls show a clean, cross-scene separation, the steering model may later promote some form of this cue into the global ranker and request an appropriate sensitivity check.

## 13. B provisional rankings and statuses

For each hard-valid parent, save the complete evidence record before reducing it to a single order.

The first run needs one deterministic B pick for the gallery, but that pick is explicitly provisional:

1. candidates with `Q_paint10` available in both directions sort by `Q_paint10` descending;
2. if no candidate in the view has `Q_paint10`, fall back to candidates with `Q_geom`, sorted by `Q_geom` descending;
3. use existing exclusive reverse support descending as the next tie-break;
4. then `source_order`, `origin_index`, `kind_order` ascending.

This ordering exists so Luna can render and compare candidates without waiting for a steering decision. Do not call it an accepted detector rule.

Player fractions, camera error and junction continuation evidence do **not** veto candidates in this provisional order. Save the historical gated subsets separately so the steering model can see what those earlier rules would have done.

Parents have `kind_order = 0`.

Return one of:

- `provisional_for_review` — a hard-valid candidate has a two-direction `Q_paint10` or `Q_geom` readout;
- `evidence_sparse` — hard-valid candidates exist, but none has usable evidence from both court directions;
- `no_valid_candidate` — no candidate is geometrically/numerically valid enough for the evidence pass.

Never emit `approved` automatically.

For `evidence_sparse`, still save the strongest one-direction and per-cue candidates so the steering model can decide whether the problem is missing evidence, a poor reduction or a missing proposal.

## 14. Arm C — one fixed-identity refit

For every parent, use its physical stripe evidence and exclusive assignment to prepare:

```python
fixed_stripe_refit.prepare(..., centres=paint_geometry.CENTRE_SEGMENTS_M)
```

Then attempt exactly one initial refit:

```python
fixed_stripe_refit.refine(
    ..., use_positions=True,
    centres=paint_geometry.CENTRE_SEGMENTS_M,
)
```

Keep the existing `MAX_EVALUATIONS = 100` for the first probe. A budget exhaustion is evidence about this configuration, not proof that refitting is useless; if a promising fit repeatedly runs into the budget, the steering model may raise it globally.

The inherited `prepare()` path also uses the branch's existing fragment-support cutoff when choosing fitting samples. Record the full assignment-strength distribution and exactly which fragments entered the fit. Treat this as the **existing refit implementation under test**, not as a newly justified W5 threshold. If refit behaviour looks promising but sample admission is clearly the bottleneck, the steering model may request a global sensitivity pass later.

Do not run assignment/refit loops, focal-length search or broad parameter sweeps before the first evidence review.

A child can enter the W5 evidence table only when:

- the refitter reports `successful`;
- Jacobian rank is 8;
- all projected values are finite;
- minimum corner homogeneous denominator is `> 1e-6`;
- the TL/TR/BR/BL quadrilateral is strictly convex with consistent positive turns;
- the existing downstream `geometry_valid` check passes after recomputation.

These are numerical/geometric validity checks, not empirical quality gates. Do not add a new minimum-area rule beyond the existing geometry validity logic.

Save every attempted child, including invalid ones, with:

- parent origin;
- fitting sample count and fragment IDs;
- assignment strengths for selected and near-selected fragments;
- objective before/after;
- solver status and evaluations;
- Jacobian rank and condition;
- minimum denominator;
- attempted corners when finite;
- exact rejection reason.

For a valid child, recompute from scratch:

- geometry validity;
- player fractions;
- camera error;
- stripe assignments and all W5 evidence readouts;
- junction/termination evidence.

Do not inherit the parent's scores, diagnostics or human ruling.

Rank parents and valid children together using the same provisional B ordering. A parent is never replaced merely because its child converged.

## 15. Diagnostic rankings that must be saved

For B and C, save enough alternative orderings that the steering model can tell whether a result is driven by geometry, the historical paint cutoff or an inherited whole-court heuristic without rerunning the expensive evidence calculation.

At minimum save, per view:

1. `q_geom_rank` — hard-valid candidates with two-direction `Q_geom`;
2. `q_paint10_rank` — hard-valid candidates with two-direction `Q_paint10`;
3. `provisional_rank` — the deterministic gallery ordering from section 13;
4. `exclusive_reverse_rank` — hard-valid candidates ordered by existing exclusive reverse support;
5. `historical_fullcourt_subset_rank` — the provisional order restricted to candidates satisfying the old player/camera predicate;
6. `historical_camera_subset_rank` — the provisional order restricted to the old geometry+camera predicate.

For junctions, players and camera diagnostics, save the raw candidate fields and a compact top/bottom summary rather than inventing a new hard rank if the ordering is not naturally defined yet.

Also save the nearest/reference-best candidate **after** all label-free evidence and provisional rankings are complete, solely for failure diagnosis.

This should let the steering model distinguish:

- missing candidate;
- ranking/reduction failure;
- historical player-rule false negative;
- historical camera-rule false negative;
- photometric-threshold sensitivity;
- junction clue that is informative but not yet calibrated;
- refit damage or refit sample-admission failure;

without rerunning proposal generation.

## 16. Visual/evaluation packet

Complete and save all label-free rankings before joining any references.

Then generate one compact gallery with one row per view:

- A primary paint-first winner
- B provisional winner/status
- C provisional winner/status

Show:

- outside court boundary
- physical finite paint template
- useful tight crop around the court paint
- uncluttered prediction by default
- reference overlay available separately/toggleably

Where A/B/C select the same geometry, reuse the image instead of creating duplicate panels.

For failed/ambiguous rows, provide top-three diagnostic links rather than showing every candidate.

Include the four known diagnostic controls and their W5 cue breakdown even when they are not in the automatic pool.

Evaluation metrics remain separate by provenance. Report where available:

- visible manual-landmark reprojection error
- approved-control corner errors with the existing 180° relabelling
- each corner's error
- signed normal offsets along named visible markings

Do not merge manual and supplied-control metrics into one accuracy number.

The frontier model writes the new visual rulings to:

```text
visual_rulings.json
```

with vocabulary:

```text
usable
needs_correction
wrong_court
unclear
```

## 17. Pilot sequence

The steering model will first ask for these five views:

```text
gxBQ_window_00_frame_0
am2_window_00_frame_150
am2_window_01_frame_28019
am3_window_00_frame_0
shuttleset_03_scene_0019
```

Make the driver accept an explicit case list so the exact same implementation can later run all nine.

If the steering model changes how evidence is reduced after inspecting the pilot, record the changed global parameter/rule in the new run's manifest. Recompute from saved evidence when possible; only rerun expensive geometry if the changed rule actually requires new measurements. Do not carry hidden case-specific exceptions.

## 18. Stop conditions

Stop only the affected view/comparison when:

- required source input is missing
- image/coordinate provenance is inconsistent enough that the evidence is not comparable
- replay loads a materially different population/winner and the source mismatch cannot be explained
- a coding fault invalidates the algorithm

A solver budget exhaustion, rank-deficient child or invalid child is an ordinary recorded outcome, not a reason to stop the parent or the whole case.

If a coding bug is fixed, rerun the same global configuration. Do not smuggle a scoring change into a bug fix.

## 19. Return packet

Under `runs/<run_id>/`, return:

```text
result.md
manifest.json
per_view.csv
rankings.json
fit_attempts.csv   # or compact equivalent
visual_rulings.json  # may be pending until steering review
gallery/index.html   # or another compact browsable gallery
```

`result.md` should be short. State:

- which cases completed
- A/B/C selected IDs and statuses
- whether known positive/negative controls behaved as expected
- obvious missing/misranking/threshold-sensitivity/refit failures
- no claim beyond this development corpus

`manifest.json` should contain the exact global parameters, case list, source-stage provenance, imported helper paths, mask availability and a short record of any steering-approved global rule revision. Historical probe values and newly promoted decision thresholds must be distinguishable.

Keep bulky sample arrays and response tensors on the execution host under the run directory. They need not be committed or included in the compact handback.

## 20. Deferred branches are separate episodes

The active W5 run ends with the return packet above. Do **not** pre-implement or automatically launch the later fresh-scene or integration exercises.

Those optional stages live in [DEFERRED_BRANCHES.md](DEFERRED_BRANCHES.md). The frontier model must explicitly activate one of them. A generic outcome such as `carry forward` or `W5 looks promising` is not activation.

If a deferred branch is activated later:

- keep the final W5 implementation/rule as the named baseline for that episode;
- follow only the branch that was activated;
- create a separate compact run packet rather than overwriting W5 results;
- keep the same division of labour: Luna executes the defined experiment, the frontier model evaluates and chooses the next step;
- do not add new labels, verification machinery or broad test matrices unless that branch's actual evidence gives a concrete reason.

The purpose of the hook is preservation, not extra process.
