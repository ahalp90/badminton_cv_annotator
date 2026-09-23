# Colour consistency: implementation plan and worklog

## Resume

**Gallery correction, 23 September 2026.** The colour builder now fills the
existing SVD gallery template. Saved W5 and G1/templates remain side by side;
six SVD cases add detector and retrospective reference-best views. All 71
records retain their measured geometry and colour summaries below the images.
Headless Chromium could not start in the worker sandbox. The user subsequently
checked the replacement and confirmed it works. The original six-case SVD
gallery still builds separately.

**Corrected native-resolution diagnostic complete, 23 September 2026.** The
current `measurements.json.gz` and gallery supersede the initial batch figures
in the earlier execution record below. [Open the colour gallery](http://127.0.0.1:8880/)
or use [its local index](gallery/index.html). The user handles visual review.
The implementation audit and bounded corrections are complete; relevant checks
passed. No detector selection or fitting rule was changed.

## What this gallery can establish

**Colour has not changed selection, rejection or fitting.** This is a
measurement diagnostic on preserved geometry, not a before/after intervention.
Saved W5 and saved G1/templates are two accesses to the same earlier ranking.
Across the 71 records, 50 of 64 available fits have identical geometry, 14 differ,
and seven have no saved fit. Selected IDs and generated gallery coordinates were
checked against the earlier saved records with no mapping mismatch. The user
correctly found the repeated columns unhelpful for judging colour's benefit.

The shared-template replacement was confirmed working by the user. That ruling
covers the UI; it does not demonstrate that colour improves or fails to improve
the detector. The next experiment must apply a bounded cue to an actual decision
and show its effect against unchanged good controls. [pickup](../pickup.md)
now owns that next trial and the subsequent final net-evidence lead.

## What the completed diagnostic establishes

- All 71 preserved views were processed: 47 court views and 24 controls. There
  are 64 saved W5 candidates; seven records have no saved candidate. Six SVD
  cases also contribute 36 labelled selected/reference-best source records.
- Saved W5 supplies 169 chromatic marking comparisons; 73 carry the exploratory
  ambiguity flag. Greyscale footage contributes no independent chroma evidence.
  Missing support remains distinct from colour contradiction.
- The original observed-fragment probe reproduces Am1's net-versus-yellow-paint
  difference. The broader projected-stripe test has only one supported marking
  on saved Am1, so it cannot independently reject its net baseline. Narrow
  predicted stripes remain a coverage limit even at native resolution.
- Colour similarity alone does not separate the GX5 failure. Saved W5 has five
  supported markings, with median nearest raw-paint distance 3.61 Lab units.
  The hallucinated baseline-SVD court has only two supported markings, whose
  distance is 2.24. Its greater apparent colour similarity does not make the
  geometry better. Support and geometry must remain part of selection.
- Paint and floor colour need separate reporting. On saved GX5, left-doubles
  versus centre differs by 5.95 raw-paint units but 15.07 floor-relative units.
  Floor-relative increments are useful local measurements, not invariants of
  paint identity.

**Next trials:** [pickup](../pickup.md) puts Am2's floor-region contradiction
first, followed by Am1's observed-fragment colour question. The latter asks
whether tape can be distinguished from paint without relying on a wrong court's
predicted stripe width. Both need genuine-paint controls. This diagnostic does
not yet justify a colour-based rejection or placement rule.

The first deliverable is a cheap diagnostic across preserved court views. It
should reveal whether colour helps identify and position genuine court markings.
It must cover all supported visible markings, including partial courts such as
GX. The initial experiment produces measurements and a gallery; detector
selection and fitting changes are a later decision.

## Rejoinder for every delegate

> Do useful work, not provenance theatre. Reuse existing data, code and successful
> checks. Keep one concise execution record and report results, failures and
> material uncertainty. Do not create parallel manifests, extra audit rounds,
> exhaustive checklists or new frameworks unless a concrete failure requires
> them. Test the boundary that could invalidate the result, then move on.
> Stop speculative investigation when it no longer changes an implementation
> choice. Escalate a specific problem rather than commissioning a second version
> of the whole task. The user handles visual review; do not spend orchestrator
> tokens on browser screenshots or repeated visual checking.

## What is known

The [WebUI return](../edge_polarity/webui_return_colour_consistency/README.md)
reports a useful Am1-54 colour mismatch. Assigned net-tape fragments have nearly
neutral local colour increments; the assigned yellow paint has a strong yellow
increment. Its reported separation is 64.35 in OpenCV Lab chroma coordinates,
or 5.97 times its chosen robust scale. Those are two-case exploratory results,
not calibrated rejection thresholds. Source, results and overlays were read;
the local replay now gives separation 64.33, close to the returned 64.35.
The replay is retained separately from the broader diagnostic.

GX5 has no supported far-baseline fragments, so the original boundary veto is
untestable there. Its other visible lines can still inform the broader
diagnostic. All four Am1 reference fragments belong to the single
`near_short_service` interval. There is no genuine supported-baseline positive
control in the return. Its attached-mesh test failed to distinguish the cases.

The user proposes colour consistency across high-confidence court lines,
accommodating lighting and perspective. Start with local floor-relative colour
and projected stripe widths. Distance around the perimeter alone does not
describe lighting. A smooth spatial colour model is a possible later extension
if measured residuals justify it.

The user's SVD review separates detector selections from reference-selected
refits. On two easy broadcast cases, deeper gives the preferred detector choices;
all three arms have visually perfect best refitted candidates by reference.
Baseline provides that candidate quality at lower runtime. This motivates
examining evidence for selection and placement using preserved candidates,
without assuming the same result on challenging courts.

## Scope and measurement contract

- Reuse frozen images, court fits and fragment assignments. Reconstruct missing
  assignments with existing helpers only when necessary. Do not rerun direction
  search, SVD ablations or homography fitting for this diagnostic.
- SVD12 remains the default generation path. Use preserved W5 selections and
  completed SVD candidates as distinct sources. Hold candidate geometry and
  search depth fixed when comparing colour measurements.
- Establish confidence from existing geometric agreement, support and image
  ridge evidence before consulting the colour being evaluated. Reference
  annotations and colour agreement must not select the confidence set.
- Compare each target marking with other supported markings. Exclude the entire
  target marking from its own references. Group fragments by marking interval;
  four fragments of one service line do not provide four independent lines.
- Sample paint and adjacent floor using the projected stripe width. Record
  narrow, unresolved or occluded samples explicitly. Convert/cache image colour
  once per view, not once per fragment or candidate.
- Preserve the distribution of colour signatures. Mixed paint colours or an
  unstable reference should produce an ambiguity flag rather than a forced
  single-colour rejection. Start simply; add clustering or a spatial model only
  to address an observed failure.
- Store raw differences and support counts. Distinguish evaluated/no
  contradiction, mismatch, missing target support and missing reference
  support. Missing evidence never implies rejection. Greyscale-only imagery
  supplies no independent chromatic evidence.
- Keep the original WebUI probe as a named comparator. Its engineering constants
  are not a production rule. Avoid a broad threshold sweep or case-specific
  tuning during the first pass.

## Data and useful source paths

The frozen collection has 47 court views and a separate 24-control evaluation.
Do not assume every item has colour, a usable court assignment or an authoritative
quality label. Report actual eligible counts and exclusions once. Unlabelled
controls cannot establish a false-positive rate.

- `../frozen_views/packs/`: broadcast, GX and amateur source packs
- `../wider_evaluation/runs/20260922/manifest.json.gz`: view/image inventory
- `../evidence/holistic_admission/directional_20260921_r5/w5_directional_20260921_r5_43/case_records/`:
  saved W5 selections; verify each selection's G0/G1/template source
- `../w5_holistic/run_w5.py`, `verifier.py`: preparation, assignment, evidence
  and selection contracts
- `../edge_polarity/webui_return_colour_consistency/scripts/boundary_paint_followup.py`:
  original colour probe; preserve the received script
- `../svd_search/build_gallery.py` and `gallery_template.html`: current reusable
  stripe projection, corner crops and controls

Start with Am1-54, GX5 and a genuine supported boundary from a differently
coloured court. Include white, yellow and mixed markings, cropped courts,
occlusion, faded paint and negative controls where preserved data supports them.
Prioritise difficult amateur views over more easy broadcast examples.

## Execution and ownership

1. **One GPT-6 Sol medium worker** owns this directory's implementation, local
   batch execution, result summary and gallery. It should inspect the named
   contracts, identify usable preserved inputs, and expose any substantive
   ambiguity before choosing a new measurement. Keep edits isolated here;
   frozen helpers, received returns and detector defaults remain untouched.
2. Implement the original probe replay and the broader diagnostic. Run a small
   smoke sample before the batch. Check target/reference separation, independent
   confidence and perspective-scaled sampling. Include a genuine positive
   control, not just Am1's failure and GX's unsupported boundary.
3. **One Opus 5-5 high audit**, bounded to the diagnostic implementation and
   smoke evidence. Focus on leakage, correlated fragments, coordinate/colour
   units and missing-support behaviour. The parent verifies material findings
   and makes the final decision. No second broad audit to seek agreement.
4. Run the eligible preserved collection locally, then build the diagnostic
   gallery. Use Carmack only if measured cost warrants it; the existing SVD job
   already occupies its six-worker allocation. Save compact `.json.gz` results.
5. Summarise which failure types colour separates, where it is ambiguous and
   whether a bounded colour-guided placement trial is justified. The user does
   the visual review. Do not claim improved fitting from diagnostic separation.

Luna max/priority may do tightly bounded inventory or numeric extraction when
it saves real work. It does not own measurement design or interpretation.
Escalate Sol to high for a named unresolved implementation problem, not by
default. Keep tightly coupled measurement/gallery work with one owner.

## Checks and deliverables

Use scoped lint/types and small runtime checks at the measurement boundary.
Check repeated fragments of one marking, target exclusion, unresolved widths,
empty references and a supported genuine line. Reuse unchanged successful
checks; do not run the whole repository suite for an isolated diagnostic.

The gallery renderer expects corners `(4, 2)`, centre segments `(12, 2, 2)` and
edge segments `(24, 2, 2)`. Flat projected point arrays previously broke redraws.
Check producer/consumer shapes rather than only point counts. Reuse the fixed
renderer and let the user inspect the visuals.

Deliver a runnable diagnostic, compressed results, one gallery and a short
finding summary linked here. Append actual commands, exits, exclusions and
material limitations to this file as work completes. No per-agent report tree
inside the experiment directory.

## Delegate setup and authority

Use GPT-6 Sol medium/default service tier via the existing headless launcher;
native Luna uses GPT-6 Luna max/priority. Never substitute GPT-5.6. Use explicit
`claude-opus-5-5` at high effort for the audit. Codex, Anthropic and Carmack
project-data sharing are already authorised. Existing branch authority is
`fix/court-det`; no commits to `main`.

The Serena/Pyrefly endpoint is `http://127.0.0.1:9121/mcp`; pass it to workers
when available. Headless Sol could not resolve Carmack from its sandbox, so
delegate local work to it and keep remote transport with the existing native
monitor/parent. Read `~/.codex/remote_hpc.md` before any remote action. Maintain
one Carmack connection at a time and at most six compute workers.

## Execution record

- 23 September 2026: plan and source assessment recorded before compaction.
  No diagnostic implementation, colour replay or fitting change has started.
- 23 September 2026: implemented `run.py` and `build_gallery.py`. The received
  WebUI probe replay gave Am1-54 chroma separation 64.33 and GX5 an untestable
  far baseline, matching the received statuses (received Am1 separation 64.35).
  The replay is in `original_probe_replay.json.gz`.
- The wider batch measured all 71 inventory records: 45 ordinary views, two
  unverified views, 16 unlabelled controls and eight known non-court controls.
  Each saved source had 64 measurable candidates and seven absent candidates.
  The seven absent candidates are non-court controls. The one non-court candidate
  had no supported markings. Unlabelled controls remain unlabelled.
- On saved W5, supported chromatic comparisons numbered 85 amateur, 28 GX and
  44 unlabelled-control marking targets. The 20 broadcast views provided 81
  greyscale-only supported targets, with no independent chromatic evidence.
  Across amateur comparisons the median nearest other-marking Lab chroma
  distance was 2.01; GX was 1.87. These are descriptive, not rejection rates.
  `yellow_short_frame_156` had a 69.44 nearest-marking floor-relative distance
  for two
  supported targets, but only one reference marking for each; its paint identity
  needs visual review. Am1's broader saved candidate had only one supported
  marking, so its reference status is missing rather than a veto. GX5 had four
  supported markings for local comparison, including the near baseline.
- `measurements.json.gz` retains each physical marking's geometric support,
  exclusive fragment count, sample states, Lab increments and all target-to-
  reference distances. Five completed SVD cases contribute 15 separately
  labelled selected-candidate sources (`svd_baseline`, `svd_deeper`,
  `svd_shortlist`), measured with each candidate's own fixed geometry. The
  self-contained local review page is
  `gallery/index.html` with 71 nearby JPEG assets. It shows the two saved
  candidate sources separately, fixed geometry, support and ambiguous references.
  Colour has some large descriptive separations, but this pass does not establish
  a calibrated mismatch rule or a placement improvement. A bounded colour-guided
  placement trial would need visual review of the high-distance examples first.

### Run and checks

From the repository root, with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1`, run:

```bash
~/.venvs/badminton-cicd/bin/python scratch/court_det_fix/colour_consistency/run.py
~/.venvs/badminton-cicd/bin/python scratch/court_det_fix/colour_consistency/build_gallery.py
```

The original replay used `boundary_paint_followup.py` with its supplied packet
and the frozen Am1/GX images. The three-case smoke selected Am1-54, GX5 and
`shuttleset_21_scene_0000`. Scoped Ruff and `py_compile` both exited 0.
The final full runner and gallery builder exited 0. The runner included 15
completed-SVD sources in the 71-record result. A data
check exited 0 for all
candidate geometry shapes `(4,2)`, `(12,2,2)`, `(24,2,2)`, target exclusion,
unique physical-marking references and 71 gallery images.
Synthetic checks for missing references, target exclusion, greyscale-only
status and unresolved projected widths also exited 0.

The thresholds for projected width, ridge contrast and geometric support are
measurement eligibility settings, not calibrated court rejection criteria.
Player boxes mark occluded sample locations. Saved geometry can miss a true
stripe; that causes missing evidence here. The local signatures do not yet
resolve all mixtures of white and yellow paint or strong spatial lighting shifts.

## Bounded correction pass, 23 September 2026

The Opus audit's two measurement defects were verified and corrected. Paint
Lab, both adjacent-floor Lab values and local paint-minus-floor Lab now remain
per sample and as marking medians. Raw-paint and relative-colour ab distances
are reported separately, with each marking's 90th-percentile distance from
its median. GX5 saved W5 left-doubles versus centre is 5.95 raw-paint ab units
and 15.07 relative ab units. Floor variation can change the apparent ordering;
neither reading proves paint identity. A high relative b increment alone is
not evidence of yellow paint.

Projected width now intersects each edge line with the centre-line normal at
each sample point. The old corresponding-endpoint distance overstated widths
under perspective. Colour and player boxes use native pixels by default;
frozen fragment assignment still uses working coordinates. Supported sample
points are explicitly converted to working coordinates for overlays. The
`--sampling-space working|native` switch compares coverage without changing
geometry or the 1.5-pixel unresolved-width floor. Three-view saved-W5 smoke:

| View | Working supported / unresolved | Native supported / unresolved |
| --- | ---: | ---: |
| GX5 | 110 / 257 | 120 / 225 |
| Am1-54 | 7 / 96 | 8 / 64 |
| SS21-0 | 113 / 192 | 113 / 192 |

The SS21 source is native sized already and remains greyscale-only. The
greyscale check now counts pixels whose BGR channel range exceeds 4; at least
1% makes a view chromatic. This avoids rejecting a coloured court merely
because most pixels are neutral. It can still classify near-grey colourised
imagery or tiny colour regions conservatively.

The exploratory ambiguity display flag applies at 20 OpenCV Lab ab units to
within-marking raw-paint or relative spread and to between-reference spread.
A single reference is always flagged. Identical target/reference medians no
longer trigger ambiguity from a zero-versus-zero comparison. The 20-unit value
is illustrative (as in the WebUI probe), never a court acceptance rule.

The corrected native batch has 71 records. Saved W5 has 64 measured candidates,
6,900 supported sample points and 11,386 unresolved-width points. Its marking
statuses are 169 chromatic evaluated, 81 greyscale-only, 452 missing target
support and 2 missing reference support; 73 evaluated targets have the
exploratory ambiguity flag. Across 169 evaluated targets, median nearest
raw-paint and relative distances are 2.08 and 2.20; 18 differ by at least five
units. All three SVD G0 arms supplied six completed cases
each, and both detector selections and reference-best refits are separate
gallery sources. Reference-best is strictly retrospective. The sixth case
completed during the batch, so only that case was refreshed and inserted into
the 71-record output. Saved W5 uses G1/templates; SVD arms use G0. The user's
visual ruling is that GX5 and Am1-54 SVD examples are deranged hallucinations;
some GX hallucinations still follow a real long-edge line. Saved W5 is great on
GX5. Saved Am1 may be good except that net tape is mistaken for the far
baseline. Same-colour genuine lines can support wrong geometry, so colour
does not solve geometry selection by itself. No detector, fit or search changed.

Correction commands, from the repository root, all exited 0:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/.venvs/badminton-cicd/bin/python scratch/court_det_fix/colour_consistency/run.py --case am1_window_00_frame_54 --case gxBQ_window_00_frame_5 --case shuttleset_21_scene_0000 --sampling-space native --output scratch/court_det_fix/colour_consistency/smoke.json.gz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/.venvs/badminton-cicd/bin/python scratch/court_det_fix/colour_consistency/run.py --case am1_window_00_frame_54 --case gxBQ_window_00_frame_5 --case shuttleset_21_scene_0000 --sampling-space working --output local_scratch/external_delegate/20260923-colour-corrections/working_smoke.json.gz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/.venvs/badminton-cicd/bin/python scratch/court_det_fix/colour_consistency/run.py --sampling-space native --output scratch/court_det_fix/colour_consistency/measurements.json.gz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/.venvs/badminton-cicd/bin/python scratch/court_det_fix/colour_consistency/run.py --case am2_window_01_frame_28019 --sampling-space native --output local_scratch/external_delegate/20260923-colour-corrections/am2_refresh.json.gz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/.venvs/badminton-cicd/bin/python scratch/court_det_fix/colour_consistency/build_gallery.py
```

The one-case refresh was inserted by case ID and asserted unique. Scoped Ruff,
`py_compile`, gallery JavaScript syntax and 71 JPEG checks passed (exit 0).
Serena/Pyrefly returned no diagnostics for either changed Python file.
Targeted runtime checks passed (exit 0) for a synthetic perspective edge-line
intersection, projected shapes, native-to-working overlay points, target
exclusion, reference ambiguity including identical signatures, and greyscale
status. All 36 SVD source records matched saved native-to-working corners.
Earlier full-batch numerical findings above are superseded by the
corrected native measurements; they should not be compared as fit improvements.

### Am2 floor-region lead from user review

The user points out that baseline Am2 extends into and skews towards the blue
surrounding region. This is a candidate for floor-region consistency rather
than only comparing supported paint. The current diagnostic flags exploratory
ambiguity on four of baseline's five supported markings and none of deeper's
five. That is a lead, not a validated rejection rule.

A useful next test is to sample court-side and outside floor strips along the
preserved boundaries, including stretches without strong paint support. Anchor
floor appearance to independently reliable visible court regions and compare
baseline, deeper and saved W5. Blue intruding into the proposed interior could
contradict the fit. Painted lines must still determine exact placement: a floor
colour transition need not coincide with the badminton boundary. No floor-
region classifier or new fitting rule has been implemented.

### Final integration review

The parent confirmed the floor-colour confounding with bilinear samples and
verified the perpendicular-width defect numerically before the correction.
The corrected source uses centre-line normals, separate paint/floor values,
within-marking spreads and target-excluded references. Existing geometry and
confidence remain independent of the inspected colour. Worker runtime checks,
scoped lint, JavaScript syntax and empty Serena/Pyrefly diagnostics were reused;
no second broad audit or browser image review was run.

The completed SVD gallery received a further user assessment: GX0 refits are
near-perfect across arms and improve on saved W5; Am2 is suitable only for saved
W5 and deeper axes. Deeper Am2 is acceptable before refitting and perfect after
it. These findings are recorded in [the SVD worklog](../svd_search/WORKLOG.md).
