# Centre-assignment follow-up

## Resume

The user requested one sample from every amateur court, including GX.
[The eight-source gallery](gallery/amateur/index.html) is available at
http://127.0.0.1:8768/amateur/. The user reports one marginal, insignificant
regression (case not named). The user also judges Am1 frame 54 very bad and
worse than the saved selected court, with both in the same broad quality range.
It uses the same template and the earliest saved frame per source.
For future bounded experiment runners and galleries, delegate to GPT-6 Sol
at medium effort, non-priority, as the user requested.

The user prefers centre-to-edge across
[the four-case gallery](gallery/index.html), after the outlines were made
thicker and brighter. The initial ruling was "marginally better"; the user
then confirmed it as "the winner" across this small sample. On SS03-19,
the user reports that polarity-only marginally undershoots the back line,
while centre-to-edge hugs the inner edge of the back white line.
The default overlay represents the outside boundary, so this observation
supports relative improvement without establishing exact boundary alignment.
Keep the rule as a candidate partial improvement.
Production fitting is unchanged. The residual diagnostics and independent
audit are integrated in [the assessment](ASSESSMENT.md). The saved automatic
SVD check is also complete; next is actual 12-versus-16 matcher work and timing.

## Concerns

- Strong brightness asymmetry alone may not establish a physical paint edge.
- The two-line proxy omits the other fitted constraints.
- Objective values from different label assignments are different objectives.

## Scope and readiness

Branch: `fix/court-det`, initial tip `bee8584`. Existing untracked raw frames
and the supplied return are preserved. Serena is active at port 9121.
Inputs: `../webui_return/`, `../run_probe.py`, saved polarity results and exact
local candidate records. No detector edits or wider search are planned.

## Current module state

The existing probe swaps contradicted edge labels and preserves centre labels.
The fixed-stripe fitter freezes points, intervals and weights before fitting.
Only diagnostic artefacts and the current handover may change in this pass.

## Execution

- Read the pickup, polarity result and new reports. The prediction to test is
  a leftward upper-left movement after fragment 111 changes from centre to its
  polarity-compatible edge. The undiluted proxy is about −1.154 px in x.
- Requested two bounded Luna Max source/script extractions and one independent
  Opus 5.5 High read-only audit. Results remain leads until locally checked.
- Both supplied analytic scripts reproduce, exit 0. The exact fragment-111
  refit moves SS03-34's upper-left corner (−0.497, −0.070) working pixels.
- `diagnose.py` tests all strong centre fragments on SS03-34, SS03-29,
  SS03-19 and GX5. All four polarity baselines reproduce exactly; frozen
  array identity assertions pass. All arms converge and retain existing gates.
  Paint scores rise on SS03-34 and GX5 and fall on SS03-29 and SS03-19.
  GX5's largest corner movement is 2.348 px. These numbers alone did not
  establish a visual regression; the subsequent user ruling favours the change.
- The diagnostics also preserve SS03-34 fragment residuals, cross-line
  brightness profiles and projected stripe widths. The saved preferred court
  and movement of only the corrected upper-left corner are separate arms.
- Scoped Ruff and whole-project Pyrefly pass, exit 0 (0 errors, 39 suppressed).
  The four-case run passes after the residual-arm clarification, exit 0.
- Opus 5.5 High returned an independent audit under
  `local_scratch/external_delegate/20260923-court-centre-audit/result.md`.
  Its exact 111/179/both refits agree with the local run. Remaining claims
  need integration; no production change follows from the audit alone.
- At the user's request, `build_gallery.py` and `gallery_template.html` build
  raw/polarity/centre-label comparisons with synchronised crops and optional
  stripe-edge overlays. The SS03-34 panel also offers individual 111/179 arms.
  Build and scoped Ruff pass, exit 0. Browser rendering and JavaScript syntax
  checks pass, exit 0. The user then judged centre-to-edge marginally better.
- The numerical-only recommendation against general use was too strong.
  Retain centre-to-edge as a candidate partial improvement on these four
  development cases. The user's overall ruling does not establish wider
  generalisation or resolve the remaining inset.
- The user subsequently confirmed centre-to-edge as the winner across this
  small sample and specifically preferred its SS03-19 back-line placement.
  The SS03-19 paint-score decrease does not agree with that visual preference.
- Amateur extension: `run_amateur.py` fixes the earliest saved frame for each
  source before fitting: Am1 54, Am2 150, Am3 0, Am4 0, Centre 36, GX 0,
  Letterboxed 45 and Yellow 14. Results are in `amateur_results.json.gz`;
  individual records are under `amateur_cases/`. Points, intervals, weights
  and sample identities remain fixed across all three arms.
- Seven successful original-label refits reproduce their saved attempts within
  1.4e-9 native pixels. Am1's original and polarity-only fits have invalid
  projections. Its centre-to-edge fit converges with large off-image corners
  and passes the existing camera gate. Yellow's two corrected fits fail that
  gate. Am4 has zero strong centre fragments, so the corrected arms coincide.
  These outcomes remain visible; no case was replaced after seeing results.
- The initial amateur run stopped on the Am1 invalid-fit assertion (exit 1).
  The runner now retains unsuccessful arms as explicit diagnostic results.
  Rerunning Am1 and assembling the seven unaffected saved results passes
  (exit 0). No invalid geometry is drawn. Am1's dropdown includes the saved
  selected court for context.
- The gallery builder now accepts input, output and title arguments. It uses
  the same HTML template for both galleries. Build, scoped Ruff, whole-project
  Pyrefly (0 errors, 39 suppressed), JavaScript syntax and Chromium smoke all
  pass, exit 0. The rendered amateur page contains eight cases and 48 canvases.
  OpenCV projection reconstruction stays within 0.01 working pixels.
- Amateur visual ruling: one marginal, insignificant regression was reported
  without a case ID. Am1 frame 54 was judged very bad and a regression against
  the also-bad saved selected court. Do not infer individual rulings for the
  other cases or use score changes to supply the missing case ID.
- The user raised yellow paint on green flooring as a possible polarity
  failure mechanism. Saved evidence does not establish that explanation:
  Am1's original-label refit already has an invalid projection, and the saved
  selected court already has distant off-image corners. Twelve of 14 retained
  fragments have strong greyscale polarity; all four centre candidates have
  contrasts of roughly 26–52 levels at ±1 px. The retained assignments contain
  no right sideline or near-baseline markings. Fragment 302 changes contrast
  sign between ±1 and ±2 px, so local photometric ambiguity remains plausible.

## Completed fitting assessment and SVD follow-up

- The user identifies Am1's hallucinated far baseline as the net's bottom
  white band. Strong greyscale polarity cannot identify the object as court
  paint. The existing camera/full-court gates accept the bad corrected fit.
- On SS03-34, centre-to-edge moves the original upper-left corner 1.584 px
  left and 0.589 px up. The preferred diagnostic position remains 1.416 px
  left and 1.411 px up from that fit. Residual attribution uses polarity-only
  labels; it is not an objective comparison between different label arms.
- Cross-line profiles show fragment coordinates displaced from brightness
  transitions. These observations do not uniquely separate blur, rasterisation
  and effective stripe width. No change to the 40 mm model is justified here.
- Opus independently reproduced the 111, 179 and combined refits. Its proxy
  caveats are useful; describing acknowledged approximations as defects
  overstates the finding. Sol edited the report and performed a cold read.
  The combined report audit scored 88.9/100; dense figure panels remain a limit.
- The plot summary initially confused polarity-only and total centre-to-edge
  movement. Both values are now separate and the plotting run passes, exit 0.
- The WebUI net-versus-court brief and exact two-case packet were committed
  and pushed as `94821ee58a07b169095832295d2f43e2ed40c8a8`. Source code and images
  are available at the pinned revision in the brief. Optional local image
  copies and the ZIP are not required for GitHub access.
- [Automatic retention findings](../../evidence/webui_followup3_20260922/review_20260923/automatic_retention/README.md):
  all eight historically approved witnesses survive. All nine best automatic
  reference-agreement candidates and all 34 candidates within one working
  pixel of those best errors survive. Three existing score-winner roles drop.
- Reconstructing cached corners requires the producer's canonical float32
  court coordinates. A worker's float64 literals produced large near-horizon
  discrepancies; replacing those literals with `CORNER_COURT_M` gives exact
  agreement for every cached candidate. The temporary relative tolerance was
  removed. Both retention scripts and scoped Ruff pass, exit 0. Whole-project
  Pyrefly passes with 0 errors and 39 suppressed, exit 0.
- Actual matcher runtime has not been measured. The next comparison keeps
  the full 16-family and G0 comparators. No production fitting or search rule
  was changed. All sharing with delegated Codex agents is explicitly authorised.
