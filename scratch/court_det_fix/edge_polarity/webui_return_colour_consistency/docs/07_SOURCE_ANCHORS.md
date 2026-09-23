# Pinned source anchors

Repository: `ahalp90/badminton_cv_annotator`

Revision: `13f02fdbf8954ead15dbc7241a696a314473f9c5`

These are the relevant functions at the pinned revision. Function names are the
primary anchors; approximate line ranges are included where they were traced in
the audit.

## Fitting and observations

- `experiments/annotator/independent_court/fixed_stripe_refit.py`
  - `prepare`: freezes supported fragments, finite interval identities and
    sample membership.
  - `residual`: weighted point-to-assigned-finite-segment distance.
  - `refine`: least-squares fit, finite projection, Jacobian rank/condition.
- `experiments/annotator/independent_court/stripe_observations.py`
  - `interval_evidence`, `measure`, `resolve_fragments`, `score_model`.
  - Module docstring explicitly states that paired-edge evidence is diagnostic,
    not an acceptance rule.
- `experiments/annotator/independent_court/assignment.py`
  - `prepare_observations`, `measure_support`, `choose_assignment`.
  - Module docstring states that groups are detector responses, not verified
    physical stripes.
- `experiments/annotator/independent_court/paint_geometry.py`
  - 40 mm stripe model and centre/edge offsets.
- `experiments/annotator/independent_court/detector.py`
  - `_visible_samples`: clips finite projected markings to the image.
  - `_score`: geometry, family support and distinct-line counts; failing support
    sets score to `-1`.

## Gate and caller path

- `scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis/run_diagnosis.py`
  - `gate_evidence` (approximately lines 62–81): calls player fractions and
    `_score`; sets `geometry_valid` based on whether a projected candidate
    remains, and returns `floor_score` separately.
- `scratch/court_det_fix/frozen_helpers_20260914/legacy/zone_net.py`
  - `player_fractions`: tests player positions inside the complete projected
    court with a margin.
- `scratch/court_det_fix/frozen_helpers_20260914/legacy/camera_diagnostic.py`
  - `camera`: square-pixel, centred-principal-point camera plausibility.
  - `net_segments`: derives a projected physical net but the W5 gate retains
    only camera error.
- `scratch/court_det_fix/w5_holistic/verifier.py`
  - `hard_validity` and `historical_predicates` (approximately lines 277–308).
  - `physical_marking_evidence`, `measure_candidate`.
  - `camera_eligible`, `rank_candidates` (ranking approximately lines 762–812).
- `scratch/court_det_fix/w5_holistic/run_w5.py`
  - `make_parent_record` (parent validity/evidence, approximately 644–657).
  - `attempt_refit` (fixed assignment refit and child checks, approximately
    684–770).
  - `process_case` (parents/children enter pools and are ranked, approximately
    923–935).

## Frozen inputs

- `scratch/court_det_fix/edge_polarity/local_audit/webui_net_audit/cases.json.gz`
- `scratch/court_det_fix/frozen_views/packs/marking_refit_inputs.json.gz`
- `scratch/court_det_fix/frozen_views/packs/gx_extension_inputs.json.gz`
- `scratch/court_det_fix/frozen_views/frames/amateur/am1/frame_00000054.png`
- `scratch/court_det_fix/frozen_views/frames/gx/images/gxBQ_window_00_frame_00000005.png`
