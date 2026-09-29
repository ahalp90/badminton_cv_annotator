# Colour consistency: implementation plan and worklog

## Resume

**Decision trials started, 23 September 2026.** Work through the Am2 floor cue,
then the Am1 observed-fragment cue, then decide the centre-to-edge policy.
Use saved choices and geometry. A rejected choice becomes an abstention;
do not insert an annotation-selected replacement. Sol owns implementation and
galleries, with a bounded Opus review of the substantive result. The user has
authorised feature-branch commits at suitable checkpoints.

The user now confirms that amateur reference courts mix paint-centre and
outside-edge annotations. The convention is unknown case by case. Small
signed reference drift cannot establish an improvement or regression.
Semantic mistakes, invalid fits and substantial geometry failures remain
meaningful. Limited image viewing is authorised to resolve concrete ambiguity.

**Automation requirement:** the user explicitly rules out manual handling,
including choosing paint colour or polarity. Floor references and paint-side
decisions must therefore be inferred automatically. Ambiguous evidence leaves
the cue unavailable. The initial manual-patch diagnostic is archived locally
and cannot qualify a rule for adoption.

**Colour decisions complete:** the automatic floor cue rejected 0/9 saved
choices. The final hue-only cue rejected 0/12: one comparison was usable and
11 were inconclusive. Neither cue qualifies for integration. A hue-grouped
raw-colour comparison rejected Am1, but still rejects paler examples of the
same hue; that intermediate result does not establish hue-only efficacy.

The player-floor idea is closed. Saved `sticky_anchor` pairs depend on court
presence. Do not introduce a new player-selection stage here. The automatic
edge trial is complete. **Carry automatic polarity forward as the preferred
fitting integration candidate.** The user finds its six displayed comparisons
practically identical to the bright-paint rule and both improved over the
original. It handles synthetic polarity inversion with modest added cost.
Real dark-court robustness is still unproven. Am1 remains a separate assignment
and acceptance failure. The current production fitter is unchanged; see the
[final policy and timing](../edge_polarity/local_audit/ASSESSMENT.md).
No further colour search is planned.

### Completed trial scope

1. **Floor contradiction:** test saved W5 and selected SVD baseline/deeper
   courts for Am2-28019, GX0 and GX5. Infer floor reference colours from the
   two sides of observed internal-marking fragments. Require at least two
   distinct supported internal markings; exclude outer boundaries. This uses
   each fixed candidate's assignments, without manually selected regions.
   Compare paired
   inward/outward strips at 0.15, 0.35 and 0.60 court metres. Use nearest-patch
   Lab chroma distance, a predeclared threshold and a contiguous contradiction
   requirement. Report sensitivity without tuning for reference agreement.
   If it changes no useful decision, stop this formulation.
2. **Observed paint:** sample actual fragments without using a fit's predicted
   stripe width. Separate raw paint, floor and relative colour. Count distinct
   reference markings, not fragments, and exclude the target marking. Include
   real-paint controls, partial views, missing evidence and synthetic dark
   stripes. Insufficient independent evidence must remain inconclusive.
3. **Centre-to-edge:** reuse the completed four-case and amateur comparisons.
   Check the rule's brightness assumptions against dark-stripe counterexamples.
   Decide between retaining the rule, limiting it to automatically inferred paint polarity,
   or leaving it experimental. Do not use small reference offsets as the vote.
4. **Review and close-out:** one substantive Opus 5-5 high audit without a time
   limit, source checks of material findings, and Sol-built galleries using the
   shared template. Record a go/no-go decision for each cue and commit the
   coherent result on `fix/court-det`.

Out of scope: manual colour, polarity or floor selection; new proposal searches;
changed candidate geometry during the
colour trials, learned net models, full scene integration, threshold searches,
and treating a floor-colour transition as the exact painted boundary.

Planned checkpoints: `Record the colour trials and annotation limits`,
`Test colour contradictions on saved court choices`, and
`Set the court paint and edge policy from the trials`. The user's current
commit authorisation covers these feature-branch checkpoints; messages may
be tightened to describe the actual completed change.

Verification: focused synthetic and real-input checks for sampling, masks,
missing evidence and unchanged geometry; scoped Ruff, syntax and types for new
scripts; gallery payload/JavaScript checks and preserved template controls.
Reuse earlier successful checks for unchanged code. No full search or broad
test suite is justified by these standalone experiments.

### Current concerns and module state

- `floor_trial.py` uses automatic observed-fragment floor references.
  `observed_colour.py` shares native sampling with `fragment_trial.py`. All three use unchanged geometry and same-frame masks.
  GX5 has no such mask; its subtitle may affect image evidence.
- The existing projected-stripe diagnostic remains historical and unchanged.
  Its GX5 box mask did not enforce the same-frame requirement; new trials use
  the verifier's provenance check. Any effect on old measurements is unmeasured.
- The observed-stripe sampler must handle bright and dark paint symmetrically.
  Recognising a local stripe does not by itself certify that it is court paint.
- The shared SVD template and existing galleries remain the renderer baseline.
  The final gallery must show changed decisions and reasons, with good controls.

### Execution record — completed trials

- `75ff070 Record the colour trials and annotation limits`: documented the net
  pre-evaluation, experiment scope and mixed annotation conventions. Link and
  whitespace checks passed (exit 0).
- The initial manual floor diagnostic completed on nine saved choices. It kept
  Am2's bad baseline, retained all W5 selections and rejected the two GX5 SVD
  hallucinations at the primary threshold. Those rejections disappeared at
  20 ab units. No threshold was tuned. This result does not establish automatic
  efficacy. Its code, data and checks are retained only under
  `local_scratch/external_delegate/20260923-floor-trial/`; the live runner now uses the automatic formulation. Scoped lint, syntax, synthetic
  boundaries, source-geometry checks and real execution passed (exit 0).
- Completed Sol task: automatic fragment-colour and floor trials, artefacts under
  `local_scratch/external_delegate/20260923-automatic-colour/`, completed with
  exit 0. Paint decisions: eight keep, four no-decision, zero rejections. Am1's
  references were ambiguous under raw-ab grouping; the three ShuttleSet views
  are greyscale. Floor decisions: nine keep. Both fixed sensitivity checks
  remain reported, not adopted as alternative thresholds. Scoped lint, syntax,
  synthetic boundaries and real-input checks passed (exit 0); Serena reported
  no diagnostics for the three new scripts.
- The synthetic existing-edge check passed (exit 0). Under identical geometry,
  bright contrasts `[200,-200,200]` become `[-200,200,-200]` after inversion.
  The historical rule swaps both edge labels and reverses the centre choice.
  Weak or masked evidence remains unchanged. Results are under
  `local_scratch/external_delegate/20260923-edge-polarity-check/`.
- `f4c6bf6 Test automatic colour evidence on saved court choices`: committed
  the shared sampler, first paint trial and automatic floor trial with their
  saved outputs.
- Completed Sol task: `paint_grouping_trial.py` and `edge_auto_trial.py`, with
  checks under `local_scratch/external_delegate/20260923-paint-edge-decision/`.
  The edge trial infers stripe polarity automatically; unresolved polarity
  preserves the original label. All 12 fits converge; Yellow14 fails the
  camera/full-court gate. Three label sets exactly reuse historical arms;
  nine differ. Of 382 fragments, 109 have unresolved polarity and 111 change
  labels from the original. Neither convergence nor camera eligibility proves
  correct geometry. Scoped lint, syntax and behavioural checks exited 0;
  Serena returned no diagnostics.
- Completed Sol task: hue-only target comparison and shared-template gallery,
  under `local_scratch/external_delegate/20260923-colour-gallery/`.
  The hue-only result is 0 abstain, 1 keep, 11 no-decision. Low chroma is
  inconclusive; it is not a reason to reject a neutral target against coloured
  references. The intermediate hue-grouped raw comparison remains labelled
  separately for reproducibility.
- Opus 5-5 high completed the substantive source audit without a time limit.
  Shell access failed, so saved results were reconciled locally afterwards.
  Record and findings are under
  `local_scratch/external_delegate/20260923-colour-edge-audit/`.

### Earlier diagnostic resume

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

### Final comparison and integration checks — 23 September 2026

The final hue-only arm uses angular distance, with chroma only establishing
whether hue is reliable. Reference markings have equal total weight. Targets
are excluded from their references. The 15°, 20° and 25° sensitivity arms each
produce zero rejections, one keep and eleven no-decisions. Neutral Am1 tape is
inconclusive; no extra material classifier was introduced to force rejection.
Same-hue pale paint, different-hue paint, angle wrap, missing/weak evidence,
target exclusion and fragment-replication checks pass (exit 0).

Sol reused the shared SVD template for a six-case comparison: SS03-34, SS03-29,
SS03-19, GX5, Am1 and Am4. The gallery distinguishes original labels, the earlier
bright-paint rule and automatic polarity. Am1 includes the separate saved W5
choice and observed colour evidence. It is also the largest edge-fit movement,
about 7,245 working pixels. An unavailable or invalid fit has no accepted outline.
Colour trials do not change any displayed geometry.

Integration found and corrected a gallery unit error: edge fit `corners_px`
are already working pixels; saved W5 corners and observed sample locations are
native pixels. The first gallery worker scaled both. Checking against the
explicit saved `measurement.corners_working_px` now passes for all 17 accepted
fits in the six-case gallery. The first four original fits also match the old
gallery geometry within 0.001 working pixels. Camera-ineligible copies produce
no accepted outline. Saved W5 and all 13 observed sample centres separately
match their native-to-working conversion. JavaScript and template checks pass
(exit 0); final builder Serena diagnostics are empty.
This does not affect numeric trial results or the edge-policy decision.

Whole-project Pyrefly completed with exit 0: zero errors, 39 suppressed.
This project check is separate from scoped checks on the scratch scripts.
The current documentation's local links and Git whitespace checks pass.

The old colour gallery now identifies itself as a historical diagnostic. Its
saved source comparisons must not be read as before/after colour intervention.
The current review is [the decision gallery](decision_gallery/index.html), served
locally at <http://127.0.0.1:8881/> (HTTP check 200). The image overlays were built
and corrected by Sol; visual judgement remains with the user.

### Independent audit and final ruling

Opus 5-5 high completed one read-only source review. Its shell calls hung, so it
could not open compressed numeric results or reproduce reported counts. Its
three Serena checks were clean. Model usage confirms the requested model.
The local reconciliation script then verified the saved counts, both fit gates,
working-pixel coordinates and a reproducible seed-23 case draw (Am1).

- **Floor evidence gaps:** the reported reproduction is real: masking the
  middle of a 14-location contradiction leaves runs of seven and six, changing
  abstain to keep with 39 paired locations remaining. This follows the declared
  contiguous-location rule. `keep` means no demonstrated veto, not verified
  floor agreement. Joining observations across unseen gaps would change that
  rule; it is not an automatic correctness fix. Missing evidence may weaken a
  rejection, and must never strengthen one. The closed negative trial remains
  reported with this limitation.
- **Original paint dispersion:** the first arm pooled fragments for its scale.
  That allows one marking with more fragments to dominate the threshold. The
  preserved original is diagnostic only. Both balanced arms and final hue-only
  give each reference marking total weight one; they are the relevant results.
- **Fit gates:** all eleven `automatic_valid` cases also pass the stored
  historical full-court gate; Yellow14 fails both. The runner's flag itself
  only tests camera eligibility. Do not assume the two gates are equivalent
  for future data. Am1 demonstrates that passing both does not certify a court.
- **Dark local profiles:** SS03-19 fragment 178 is the only dark inference in
  the three ShuttleSet views. It retains its original label; it does not meet
  the reviewer's proposed trigger of a dark inference changing that label.
  Its sampled centre is darker than both flanks on visibly bright-painted
  footage. This is a local image profile, not proof of physical dark paint.
  Possible confusion with nearby structures remains a limit of the method.
- **Hue under colour casts:** no cast-invariance claim is established. Hue-only
  tolerates colour-strength changes at fixed hue; lighting can also change hue.

The user subsequently reviewed the six-case decision gallery and judged the
automatic and earlier bright-paint corrections practically identical, with
both improving on the original. This supports choosing automatic polarity for
the next integration, while preserving uncertainty on real dark-taped courts
and the six unreviewed automatic fits. The known wrong Am1 assignment remains
unresolved by either correction. No production change is made in this phase.

A bounded local timing probe used one warm-up and three measured repeats per
case, one OpenCV thread and one BLAS thread. Median bright-rule evidence time
was 64 ms; adding native image preparation and automatic sampling gives about
154 ms (71–220 ms across case medians). This excludes proposal generation,
context preparation, fitting and scoring. The additional sampling median is
43 ms; native read/conversion median is 48 ms. The standalone probe does not
use `prepared_measurements`; integration should reuse that existing cache.
There is no need to create another greyscale-cache mechanism.

Reconciliation and timing scripts/results are retained locally under
`local_scratch/external_delegate/20260923-colour-edge-audit/` and
`local_scratch/external_delegate/20260923-edge-cost/`. The saved numeric checks,
scoped Ruff, syntax, gallery payload and JavaScript checks all exited 0.
The gallery correction is recorded under `20260923-gallery-scale-fix/`.

The user confirmed retaining automatic polarity because black court markings
are plausible. The general colour question is closed for this phase. Am1's
remaining failure is a separate recovery/abstention question. The current
follow-up examines existing options before choosing another experiment.

Final checkpoint: `Record the colour outcome and preferred edge correction`.
It includes the final trials, corrected shared-template gallery, current
handover and policy. The earlier diagnostic gallery is labelled historical.
Unrelated frozen images, raw searches and local review artefacts remain local.

### Am1 proposal-recovery follow-up — visually accepted, integration checks remain

Current outcome: the bounded trial found a promising automatic Am1 recovery.
See [the report](AM1_RECOVERY.md) and [gallery](am1_gallery/index.html).
The entries below preserve the investigation sequence.

The user approved one bounded investigation of Am1 recovery before introducing
net-post detection. The general colour-veto trials remain closed. This follow-up
asks whether existing service-line templates and automatically colour-consistent
fragments can produce a useful proposal that the saved pool lacked.

Luna owns saved-record extraction only. Opus 5-5 high traces template generation,
pruning and retention with source/evidence checks. Sol owns any justified bounded
runner or gallery. No manual colour, annotation-selected replacement, broad
search, production change or new post detector is in scope. Inspect actual
candidate survival before designing a new colour branch. Artefacts are under
`local_scratch/external_delegate/20260923-am1-{records,causal-trace}/`.

Claude shell execution is restored after the user unlocked the login keyring.
A native Opus probe returned `shell-ok`; memory was disabled and normal
permissions were retained. No shell configuration was changed.

The user updated delegation preferences: use Sol high/default tier for
substantial coding, experiment runners and galleries. Luna remains limited to
linear mechanical tasks. Opus 5-5 high remains the independent evaluator.

The user authorised continued AFK work and suitable commits. A second Opus
5-5 high delegate now evaluates the distinct net/post feasibility question
using existing image fragments and saved Am1/GX evidence. It does not repeat
the service-line template trace or build a new production detector. Record a
practical no-go if neither route justifies a bounded follow-up; update the
top-level handover and stop rather than expand the search indefinitely.

Opus findings are investigation leads, not final rulings. The final decision
requires source or numerical confirmation of each decisive claim. If the net
proposal is worthwhile, the user permits Claude to build a narrowly scoped
trial under explicit simplicity constraints. Its implementation must receive
final audit and tidy before the trial runs. Sol high remains responsible for
substantial Sol work and gallery building.

Provisional source-trace lead (awaiting final verification): the observed lines
nearest the reference service-line rectangle survive extraction and merging,
but rectangle 103151 is outside the direction-group rectangle union. Its two
sidelines belong to different selected groups. This identification uses the
reference only to diagnose missing coverage; inserting that rectangle manually
would not be an automatic recovery. The next decision depends on a label-free
way to retain useful geometry and on its downstream ranking.

The template trace completed. Its unchanged 256 proposal IDs reproduce the
saved pool exactly. Three additional vanishing points from pairs of the three
longest lengthwise merged lines admit a new leading template proposal with
13.1 working-pixel maximum retrospective landmark disagreement (old pool:
97.7). This establishes generation coverage only. The trace's claim of
recovery is premature until refitting and final W5 ranking are checked.

Sol high is implementing the bounded next trial under
`local_scratch/external_delegate/20260923-am1-recovery-trial/`: Am1 first, with
unchanged scoring/ranking and reuse of unchanged saved pools/refits. If useful,
check GX5, Am4-319 and SS21-10 retention; no 47-case sweep. Only the standalone
`colour_consistency/am1_recovery_trial.py` and its output are in scope.

The separate net feasibility review found strong support for the actual Am1
net when using all generic fragments, but many false matches from naive
candidate-free post/tape assembly. Its no-go applies to recovery from the
old candidate pool. A candidate-conditioned match against independent image
fragments is not inherently circular; no net decision is final until the
new-proposal ranking result is known. Low or missing support cannot itself
reject a court. Raw review and diagnostic scripts are under
`local_scratch/external_delegate/20260923-am1-net-feasibility/`.

The seeded W5 replay completed. It keeps the three source populations and
unchanged scoring/refitting/ranking. Actual selection changes to
`rectangle_103167:template_42/child`; maximum corner-reference disagreement is
197 native pixels (98 working), still material. Reference-near alternatives
survive both gates at camera ranks 2 and 3. The rank-3 alternative has 34 native
pixels maximum corner disagreement and 18.6 native pixels maximum visible
landmark disagreement. These references enter only after automatic selection.
No retention controls were run under the Am1-first stop condition.

A single net-ranking trial is now justified by the changed candidate coverage.
Its rule is fixed before outcomes: both projected tape halves must have at
least 75% aligned generic-fragment support, plus at least one post at 75%.
Prefer the first such court in the existing eligible paint-score ordering;
otherwise retain the existing choice. Missing support never rejects a court.
The comparison uses Am1's new pool, saved Am1, GX5 and Letterboxed45. It does
not assemble posts, refit nets or sweep thresholds. Opus builds the standalone
runner; it must be audited before real-data execution. Preserve all seeded
candidates so any follow-up can replay scores without repeating fitting.

The audited net trial completed with exit 0. It changes only seeded Am1 among
four evaluated pool configurations. The new choice is
`rectangle_98207:template_12/child`, original camera rank 2. Maximum/median
visible-landmark disagreement over 21 points changes from 90.06/22.58 to
14.44/2.40 native pixels (working coordinates are half those values). All four
net pieces have 24 visible samples: tape support 23/24 and 22/24; both posts
24/24. The old Am1 pool has no strong gated alternative and remains wrong.
GX5 and Letterboxed45 preserve their saved choices. Runtime is 30.6 seconds
for the four records, including preparation and reranking assertions.

The independent final audit reproduced the numbers and candidate identity.
It found no code-level reference leakage or reuse/order bug. Its important
qualification is retained: seed design used reference-labelled Am1 geometry,
and Am1/GX feasibility results informed the net condition. The rule was fixed
before scoring the expanded pool, not before seeing these development views.
None of the current controls tests a correct obscured winner against a strong
wrong alternative. Strong wrong candidates exist outside the full-court gate;
that gate remains part of the tested ordering. Even among seeded gated courts,
strong net support alone does not guarantee the best visible fit.

The full seeded pool is preserved locally (16 MB compressed), and a separate
replay reproduced the earlier choices and selected geometry exactly. The
optional `--pool-output` export was added afterwards for future runs; the
preserved pool itself was captured by `preserve_pool.py` during that replay.
Whole-project Pyrefly: exit 0, 39 existing suppressions. Seed/hook checks,
scoped Ruff and syntax: exit 0. Source-to-gallery ID/coordinate and JavaScript checks passed; the page and all three images return HTTP 200.

Three control generations completed at unchanged caps. GX5's selected template
was initially flagged as unretained because the check required exact equality
of every gate diagnostic. A targeted parent replay showed identical corners
and homography; only camera error changed from 0.009531846882642437 to
0.00953184688264241. This does not alter eligibility. The retention runner now
reports geometry equality separately and records differing gate values.
Am4-319 and SS21-10 retain their selected G0/G1 parents in the frozen sources.
No combined-change ranking/refitting control was run. The next useful check,
after visual review, is final selection safety on known good courts, including
a correct court with obscured net evidence facing a strong wrong alternative.

The gallery reuses the shared SVD template and existing image assets. All nine
displayed choices match the source IDs and working-pixel corners (tolerance
1e-6); native-to-working conversion occurs once. Homography corner agreement
uses 0.02 working pixels to allow stored floating-point precision. Image sizes,
relative links, JavaScript syntax and local HTTP checks passed. No image was
visually judged during gallery construction. Review URL:
http://127.0.0.1:8882/am1_gallery/ . The original six-case edge gallery remains
available separately.

The report received a fresh prose edit and a separate cold read. The reader
recovered all six experiment questions without guessing. Mechanical prose
warnings were checked; several were false positives on technical labels and
an explicit opening outcome. Numerical facts and qualifiers were checked
against the saved outputs. The checkpoint keeps the result experimental.

Final retention rerun: exit 0. Matching IDs with identical corners/homographies
are 177/256 on GX5, 181/256 on Am4-319 and 141/256 on SS21-10. Exact equality
including every gate float is lower (86, 110 and 99); do not interpret those
smaller counts as geometry loss. GX5's selected parent is preserved exactly in
geometry, with only the camera-error rounding difference above. The other two
selected source parents remain in unchanged G0/G1 pools.

Final checks: four-script Ruff, four Python syntax parses, generated JavaScript
syntax, local documentation links and `git diff --check` all exit 0.
Whole-project Pyrefly exits 0 (39 existing suppressions). Relevant behaviour was
exercised by the replay, fixed-rule selection, generation checks and targeted
numerical audits; no broad pytest run was needed for this scratch-only change.
The report audit scores 94/100 after source fidelity checks; all six cold-reader
questions were answered. No production default, manual colour handling or new
post detector was introduced. This closes the bounded experiment; visual review
and combined-change selection checks remain before integration.

### User visual review

The user finds the net-preference Am1 result a substantial improvement. The
right boundary follows the inner edge of the yellow stripe rather than the
outer edge, but the user considers that residual inset acceptable given the
recovery. Record Am1 as visually accepted for this experiment. Combined-change
selection checks remain before integration; this review does not establish
general net-rule accuracy or resolve the remaining edge convention.
