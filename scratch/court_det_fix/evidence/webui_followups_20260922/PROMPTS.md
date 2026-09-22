> Original WebUI task packet, moved from `court_det_fix/WEBUI_FOLLOWUPS.md`.
> Tasks 1/2 have a verified local replay. [SVD returns](../webui_followup3_20260922/README.md)
> are filed but unreviewed. Historical launch instructions below are preserved;
> [pickup](../../pickup.md) owns the next actions.

# Useful WebUI follow-ups

Tasks 1 and 2 have a [filed return](README.md).
The original report, runner, tables and figures are preserved.
[Local numerical replay](local_replay/README.md)
passes; detector integration is untested. Task 3 is still separate.

These tasks turn committed evidence into small executable results for the
local agent. They are not another general review of the detector. Start with
task 1. Task 2 can reuse its measurement helper, but does not require its
conclusions. Neither task requires another detector run.
Task 3 can run independently: it tests a cheap SVD search-reduction idea before
any matcher integration. These are the three useful lanes; a further broad
review or graph-search redesign would add little at this stage.

## How to launch

Give a fresh WebUI session this document and say:

> Follow the shared contract and complete task N only. Download the named
> committed inputs into your container and do the calculation. Return the
> runnable script, small results table and concise recommendation as files.
> Do not substitute an essay for an executable result. If an essential input
> cannot be retrieved, name it and stop rather than guessing.

## Shared contract

- Repository: `ahalp90/badminton_cv_annotator`. Pin project files to
  `b47eee0ff43ec7cbebcc2d84eb9e5e98075d4eea` on `fix/court-det`.
  This prompt document was added later; the evidence version above is fixed.
- Fetch only named files and necessary small dependencies through GitHub.
  Do not clone the full history or download every gallery. Gzip JSON can be
  read with Python's standard library; NumPy and Pillow suffice for the first
  two tasks. Check the actual container tools before promising execution.
- Project access is read-only. Container scratch scripts and downloadable
  results are wanted. No commits, PRs, model downloads, remote jobs or external
  delegates. Do not import the full detector merely to parse its outputs.
- Read `.github/AGENTS.md` once, then the selected task's named inputs. The
  local `.codex/context.md` is not published; this contract supplies the needed
  project context. Other tasks and reports are on-demand, not compulsory
  orientation reading.
- Keep case and source identity: `origin_case::G0/G1::candidate_id`. A W5
  `/child` is a different fitted court from its parent. Record coordinate
  systems before projecting anything. Read the producer's relevant functions
  when the saved field semantics are unclear.
- Verified context: the 27-case panel is development data. W5's 24/27 means
  qualitatively usable for development, not 24 clean fits. GX's 7/7 means
  correct-court identification, not seven accurate fits. Do not repeat those
  broad reviews or turn familiar-video tests into held-out accuracy.
- The target is a CourtKeyNet-free scene-level detector: sparse frame sampling,
  multi-frame agreement and shared search, not full search on every frame.
  Far-end clipping matters even when the near end looks excellent.
- Return one short `return.md`, one runnable script and its small data outputs.
  Include input paths, commands, checks, findings and the exact next local
  change or test. Preserve a rejected hypothesis if that is the useful result.
  No new provenance framework or multi-document worklog. If pausing, put the
  resume state at the top of `return.md`.
- Keep work bounded to the named cases and one proposed change per comparison.
  Use at most six CPU threads. New measurements from annotations are diagnostic
  oracles, not deployable selectors. Local verification is required before any
  returned code or recommendation is integrated.

## Task 1 — Measure the far-end problem and locate better existing courts

**Decision this unlocks:** should the next effort improve proposal geometry,
ranking, or both? Produce the visible-fit measurement helper needed for the
wider sample test instead of another subjective overlay review.

### Inputs

Paths below are repository-relative. Under
`scratch/court_det_fix/evidence/pixel_temporal/evaluation_20260922/`, fetch:

- `results/gx/candidates.json.gz`
- `results/gx/manifest.json`
- `results/gx/selection.json`
- the seven `results/gx/scores/*.json.gz` files
- `run_temporal_union.py`, for coordinate and candidate semantics only

Also fetch:

- `data/amateur_court_corners/2026-09-09/hand_corners_landmarks.csv`
- `data/amateur_court_corners/2026-09-09/hand_corners.csv`
- `scratch/court_det_fix/frozen_views/packs/gx_extension_inputs.json.gz`
  for frame dimensions and case IDs
- The raw GX86088 input and its far-end sheet, linked from
  `scratch/court_det_fix/evidence/review_20260922/temporal.md`.
  Fetch other raw frames only for the bounded visual check below.

### Starting facts and question

All seven GX frames have committed clicked landmarks. The candidate matrices
are in anchor **working** pixels at 960×540; saved registration matrices map
target **native** pixels to anchor native pixels. These are different spaces.
The replay's `working_alignment` shows the required scale conjugation.

User review finds frame-0 G1 `143:158` a skewed but tolerable difficult-case
fallback. GX86088 G1 `16:44` is only an exceptional last resort: its near end
looks good, but it loses much of the far long-service-to-baseline strip.
Neither is a clean-fit example. Do the measurements support that distinction,
and is materially better geometry already available in the saved pool?

### Work

1. Build a small standalone projector and measurement script. Match annotations
   by frame number and court-metre coordinates, not row order or corner-name
   guesses. Validate target/anchor and native/working conversion by overlaying
   known saved selections. Resolve court-symmetry relabelling consistently for
   the whole candidate and record the convention; do not match each landmark
   independently to whichever line is closest. Region names follow the
   annotation's court orientation. Do not refit or alter candidate geometry.
2. Measure the published native, common-union per-frame and shared **paint**
   winners on all seven frames. Report maximum and median clicked-landmark
   error separately for the far backcourt, remaining far half and near half,
   with point counts and explicit region definitions. Report native and
   working-pixel units. Do not count extrapolated corners as clicked truth.
3. On GX86088, quantify the two named candidates' far-strip loss if the
   annotations support it. Use a clearly defined overlap/missing-area measure
   against the annotation-derived court plane, clipped to the visible region.
   State the model dependence and fit residual; do not call this exact physical
   ground truth. If area is ill-conditioned, retain the landmark measurements
   and explain the failure rather than inventing a precise percentage.
4. As a separate **reference-guided diagnostic**, scan the 278 all-frame-
   eligible candidate occurrences for better GX86088 visible-landmark fits.
   Keep source IDs and separate geometrically distinct alternatives. Inspect
   at most three promising alternatives on the raw image, including the far
   end. A low corner error alone is not a visual approval.
5. Say whether this pool has a better attainable fit than the paint winners.
   Keep that answer scoped to this pool and view. Propose the smallest useful
   quality-report columns for the wider test. Numerical acceptance thresholds
   are proposals for user approval, not thresholds validated by this exercise.

### Return and stop

Return the script, per-frame metrics, candidate IDs for at most three better
alternatives, and no more than four annotated figures. Include projection and
identity checks. End with one of: **ranking lead**, **geometry/coverage lead**,
**both**, or **insufficient evidence**, with the next local action.

Do not tune a new scoring function against these annotations, regenerate
proposals, investigate all 27 views, or reconstruct missing W5 child geometry.
The complete small GX dataset is sufficient; stop if its coordinate mapping
cannot be verified.

## Task 2 — Test how much sparse multi-frame access actually buys

**Decision this unlocks:** a concrete first scene-sampling experiment and the
smallest replay test the local agent can reuse when implementing it.

### Inputs

Use the same GX candidate, manifest, selection and seven score files as task 1.
Read only `eligible_ids`, `frame_winners`, `temporal_winners` and the relevant
coordinate helpers in `run_temporal_union.py`; do not execute its full replay.
The committed temporal numbers occupy only a few megabytes.

For a contrasting control, fetch the corresponding `results/am3/` candidate,
manifest, selection and two score files. Use the original amateur clicked
landmarks in `data/amateur_court_corners/hand_corners_landmarks.csv` if numerical
fit checks are needed. Do not confuse those native coordinates with GX's.

### Starting facts and question

The completed GX replay uses 3,584 source-qualified occurrences; 278 are
eligible on all seven frames. Common-pool paint ranking identifies the court
on 7/7 frames, but residual skew remains. Shared median line ranking chooses
a wall. That does not remove the requirement for multi-frame scene agreement.

How much of the demonstrated benefit survives when proposals and decisions
can use only a small chosen set of sampled frames, without seeing the others?

### Work

1. Reproduce the published seven-frame common-pool counts and shared winner
   IDs exactly from cached scalar scores, including tie-breaks. Reproduce the
   native single-frame winners too. This is a small loader/selection check,
   not another scientific audit.
2. Enumerate the 7 singletons, 21 pairs and 35 triples of GX frames. For each
   subset, admit only candidates whose `origin_case` belongs to that subset.
   Require saved `status == 'ok'` on its observed frames. Do **not** reuse the
   278-candidate all-seven-eligible mask: that would consult unobserved frames.
3. Hold access fixed within each subset. Compare per-frame paint selection
   with shared median paint selection using the saved tie-break rule. Retain
   shared line selection as a known negative control, not a proposed fix.
   Evaluate each observed frame's choice separately; do not pick the best one
   using the unseen frames or annotations.
   Lock chosen IDs before consulting unobserved scores or annotations.
4. Evaluate the locked choices on the remaining frames without changing the
   pool, winner or policy. Report empty pools, unseen-frame eligibility
   failures and fitting errors separately. Reuse task 1's measurement script
   if supplied; otherwise implement only its basic clicked-landmark projection.
   Inspect no more than six distinct selected geometries when visuals are
   needed. Do not classify an unseen geometry as correct from a scalar score.
5. Distinguish the nearly adjacent GX0/GX5 pair from temporally spread subsets.
   Do not pick the single best subset after looking at annotations. Report
   distributions by sample count and expose severe failures, not just means.
   Repeat the limited one-versus-two-frame calculation on Am3 as a contrast.
6. Recommend the first **testable** sample budget and escalation condition for
   a scene-level prototype. Explain which choice the cache supports and which
   needs a real clip test. Candidate/score counts are work proxies, not measured
   latency. Existing sparse frames cannot validate change detection or the
   sampling interval between them.

### Return and stop

Return one standalone cached replay, a subset-level results table, and a short
proposal for the local scene-level experiment. Preserve every chosen ID and
the eligibility counts so it can become a regression fixture.

No new image scoring, tracking, video downloads, learned selector, threshold
search or camera-change implementation. This is offline sparse-access evidence
on familiar views, not a validated streaming detector. A finding that three
samples still fail is useful; do not conceal it by adding more mechanisms.

## Task 3 — Can SVD safely cut direction-pair searches?

**Decision this unlocks:** whether a cheap line-family pre-screen deserves a
local matcher trial. SVD refitting alone does not reduce search. Test a concrete
pruning rule, including the cases where it throws away useful directions.

### Inputs

Under `scratch/court_det_fix/`, fetch:

- All nine `direction_agreement/runs/direction_agreement_20260915_144900/e2/*.json.gz`
  and corresponding `e3/*.json.gz` records.
- The matching nine `frozen_views/baseline_directions/*.json.gz` records.
- `frozen_helpers_20260914/automatic_axes/svd_fixed/run_svd_fixed.py`, for
  `svd_direction` and `fit_groups` semantics only.
- `direction_agreement/run_fits.py`, for the coordinate transform and the
  distinction between diagnostic fits and generated courts.
- `evidence/direction_search/README.md`, for the established failure modes.

These are nine frozen development views, not nine independent held-out scenes.
Use baseline arm B only. Each line family (called a pencil in the records)
has fixed support membership. E3 saves its SVD residuals and all ordered-pair
control-fit errors. A control fit uses reference geometry; it is not a court
produced by the automatic matcher.

### Work

1. Reproduce the B-family SVD diagnostics from the saved homogeneous line
   equations and E2 support masks. Follow `run_fits.py`'s coordinate transform
   before normalising line normals. Compare with
   `sets.B_svd.groups` in E3 using a stated numerical tolerance. Do not rerun
   the control optimiser or matcher.
2. Test this single, deliberately simple screen: rank families by ascending
   `algebraic_rms`, then descending `line_count`, then `group_index`; keep the
   first 12 of 16. Also report 8, 10, 14 and 16 as sensitivity checks. Do not
   choose a different budget per case or use reference errors in the ranking.
   Preserve ordered pairs and original group IDs. Handle an unexpected family
   count explicitly rather than quietly assuming 16 everywhere.
3. Keep B's original direction estimates unchanged. For every budget and case,
   report ordered-pair counts, retention of B's best converged diagnostic pair,
   and best converged corner error before/after pruning. From
   `sets.B.fits.records`, also report how many pairs within one working pixel
   of the original best survive. That band measures diagnostic diversity; it
   is not a court-acceptance threshold. Keep failed and unconverged fits apart.
   These counts are potential pair launches before camera filtering, not
   measured runtime or guaranteed matcher savings.
4. Challenge the ranking's meaning. In particular, two-line supports can have
   a trivial zero residual. Check whether low residual merely rewards sparse
   or incidental line families. Test equivalent pixel coordinate descriptions
   (translation and uniform scale), transforming both lines and the
   normalised-to-working map consistently. The selected original group IDs
   should remain unchanged apart from documented numerical ties. Do not
   compare raw pixel-space algebraic residuals across different frames.
   For row-form lines and a working-coordinate transform `A`, use
   `lines_new = lines @ inv(A)` and `map_new = A @ map`. Copy the small NumPy
   formulas into the standalone script; do not import the experiment runners.
5. Give a go/no-go for a local trial of this exact pre-screen. If it loses a
   good pair, name the case and group IDs and preserve that counterexample.
   A rejection applies to this residual-ranked screen, not SVD-based search
   reduction in general. Suggest at most one principled next alternative;
   do not launch a threshold or heuristic search to rescue this one.

### Return and stop

Return a NumPy-only script, per-case/budget table, selected group IDs and a
small regression fixture containing any lost promising pair. State exactly
what a local automatic-matcher trial would still need to check: generated
candidate coverage, ranked visible fit and actual work saved.

Do not change support membership, merge similar directions, replace B with
B_svd, regenerate courts, or design graph search. Keeping a label-guided best
pair is necessary evidence for this screen, not proof of detector quality.
Stop after the fixed comparison and numerical checks, including a negative
result. No performance claim from pair counts alone.
