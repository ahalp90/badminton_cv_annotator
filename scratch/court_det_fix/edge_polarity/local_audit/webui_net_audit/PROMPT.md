# WebUI task: distinguish a court boundary from the net's bottom band

Please do a bounded, independent reasoning audit. The question is:

**Can the current court evidence distinguish Am1's false far baseline from a
legitimate partial court view, and what single additional check would make
that distinction defensible?**

This complements local work. Do not repeat the centre-label fitting experiment,
SVD family-retention screen, full candidate search or broader gallery evaluation.
Do not redesign the detector. A derivation, small counterexample and one precise
local test are more useful than a list of possible features.

## Available evidence

Fetch the current case packet from GitHub:
`https://raw.githubusercontent.com/ahalp90/badminton_cv_annotator/fix/court-det/scratch/court_det_fix/edge_polarity/local_audit/webui_net_audit/cases.json.gz`.
Use the two source images listed below at the pinned source revision.
The optional upload bundle contains the same packet and images as
`cases.json.gz`, `am1_frame54.png` and `gx_frame5.png`.
The JSON contains two exact frozen parents, original fitted points/weights,
finite marking intervals, raw fragment geometry and the completed fits.
Coordinates in the fitting constraints are working pixels. Native and working
image dimensions are supplied separately. The raw images are native-sized.

Source is public on GitHub:
`ahalp90/badminton_cv_annotator`, pinned revision
`13f02fdbf8954ead15dbc7241a696a314473f9c5`.
Fetch raw files using:
`https://raw.githubusercontent.com/ahalp90/badminton_cv_annotator/13f02fdbf8954ead15dbc7241a696a314473f9c5/<path>`.

Read only what is needed from:

- `experiments/annotator/independent_court/fixed_stripe_refit.py`
- `experiments/annotator/independent_court/stripe_observations.py`
- `experiments/annotator/independent_court/assignment.py`
- `experiments/annotator/independent_court/paint_geometry.py`
- `experiments/annotator/independent_court/detector.py`
- `scratch/court_det_fix/w5_holistic/verifier.py`
- `scratch/court_det_fix/w5_holistic/run_w5.py`
- Direct dependencies of the actual camera/full-court gate path, if necessary

Source packs and images are also available at the pinned revision:

- `scratch/court_det_fix/frozen_views/packs/marking_refit_inputs.json.gz`
- `scratch/court_det_fix/frozen_views/packs/gx_extension_inputs.json.gz`
- `scratch/court_det_fix/frozen_views/frames/amateur/am1/frame_00000054.png`
- `scratch/court_det_fix/frozen_views/frames/gx/images/gxBQ_window_00_frame_00000005.png`

The case packet avoids depending on large, uncommitted candidate caches.
If an essential dependency is unavailable, state that limit rather than invent
its behaviour.

## Observations and requirements

The intended output is the outside boundary of a badminton court, with a 40 mm
paint model. Some legitimate courts extend beyond the image. Cropping alone
must not be treated as evidence of a wrong court.

- On Am1 frame 54, the user identifies the net's bottom white band as the
  hallucinated far baseline. The saved selected court is already very bad.
  Original-label and polarity-only refits have invalid projections. The
  centre-to-edge arm converges to a very large court with distant off-image
  corners. The existing camera and full-court gates accept it.
- Am1 has 14 retained fragments. Twelve have strong greyscale polarity.
  The retained marking assignments include no right sideline or near baseline.
  These are observations, not a proven causal explanation.
- GX frame 5 is the previously approved partial-view control. Its farthest
  corner is outside the image. A gate that simply requires all corners inside
  the frame would discard it. The user prefers centre-to-edge across the
  initial four-case comparison; this does not make every boundary exact.
- Polarity identifies which side of a fragment is bright. It does not identify
  whether the fragment belongs to paint, net tape or another object.
- Keep the source parents and fitted point membership fixed in any small test.
  Reference or approved corners may enter post-test comparison, never the
  rule's decision. Do not use case IDs to decide acceptance.

## Requested reasoning

1. Trace the actual acceptance path. Identify which properties it establishes
   and which it leaves undecided. Explain how the Am1 geometry can pass using
   the recorded inputs; cite exact source lines. Check actual callers before
   claiming that a gate is missing or bypassed.
2. Analyse identifiability: which changes to court depth, scale or marking
   assignment can preserve the available visible evidence? Give a small
   mathematical construction or counterexample. Separate an ill-conditioned
   fit from a well-fitted but semantically wrong interpretation. Raw optimiser
   convergence or a condition-number threshold alone is not a conclusion.
3. Propose **one** bounded, testable check using existing observations or a
   clearly specified additional image measurement. State precisely what it
   observes, what it decides, and why the same test could retain the GX view.
   If the existing evidence cannot make that distinction, establish the limit
   and name the smallest additional evidence needed instead.
4. State the predicted outcome on Am1 and GX before running any small numerical
   demonstration. Preserve a contrary result. Two examples cannot calibrate a
   general acceptance threshold; identify what remains unvalidated.

Return a short standalone report: conclusion, derivation/counterexample, exact
code anchors, one proposed local test and its prediction, and checks actually
run. Include any small calculation script needed to reproduce the result.
Separate verified findings from leads. No production edit, new search or large
parameter sweep is requested.
