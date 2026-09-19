# W5 source ledger — what the handover is grounded in

This file has one job: make it easy to get from a claim in the handover to the code or saved result that supports it.

Repository: `ahalp90/badminton_cv_annotator`  
Branch: `fix/court-det`  
Reviewed branch head: `82d3b871bfb784f91b283761148eaa0d6501c732`

Links below are pinned to that revision. The important reproducibility distinction is **which producer/path/stage was used**, especially where several scripts share a basename.

## R1 — independent 2D court proposals

- [`experiments/annotator/independent_court/detector.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/experiments/annotator/independent_court/detector.py)
- [`experiments/annotator/independent_court/README.md`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/experiments/annotator/independent_court/README.md)

What it establishes:

- CourtKeyNet-free image-line/template enumeration already exists.
- It constructs complete court quadrilaterals, keeps competing candidates and can use broader line families.
- Its acceptance/scoring is experimental; it is useful as a proposal source, not as an independent voter in W5.
- The historical evaluator sometimes reports at 1280×720. W5 works at 960×540 and must not mix those coordinate conventions.

## R2 — fragment ownership and stripe evidence

- [`experiments/annotator/independent_court/stripe_observations.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/experiments/annotator/independent_court/stripe_observations.py)
- [`experiments/annotator/independent_court/assignment.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/experiments/annotator/independent_court/assignment.py)

What it establishes:

- Eleven named markings are represented by twelve finite intervals; the split centre line is one marking.
- `measure()` exposes per-marking, per-position, per-sample, per-fragment forward evidence.
- `score_model()` assigns each raw fragment to one marking/centre-or-edge position by reverse support, preventing the same fragment from freely explaining incompatible markings.
- The existing aggregate score averages forward and reverse evidence. W5 reuses the assignments but keeps both sample-level fragment support and raw photometric evidence so the steering model can decide how to combine them.
- Inherited constants include 64 marking samples, 16 fragment samples, 2 px distance sigma and 5 px fitting support distance. These are implementation settings, not universal accuracy thresholds.

## R3 — physical paint geometry

- [`experiments/annotator/independent_court/paint_geometry.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/experiments/annotator/independent_court/paint_geometry.py)

What it establishes:

- 40 mm stripe width and the physical stripe-centre/edge convention measured from the outside court boundary.
- The physical stripe centres are not identical to the older nominal `detector.SEGMENTS_M` template.
- W5 B/C should pass `CENTRE_SEGMENTS_M` explicitly; legacy A should remain unchanged for faithful replay.

## R4 — existing fixed-identity whole-court refit

- [`experiments/annotator/independent_court/fixed_stripe_refit.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/experiments/annotator/independent_court/fixed_stripe_refit.py)
- [`scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis/run_diagnosis.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis/run_diagnosis.py)

What it establishes:

- The refitter moves eight normalised homography parameters while keeping raw-fragment, finite-interval and stripe-position identities fixed.
- It already reports objective change, solver status, Jacobian rank/condition and projected-corner validity.
- It is local. A smaller objective does not prove the court identity is correct.
- The existing solver budget is 100 evaluations.

## R5 — C2: candidate-population and representative-selection witnesses

- [`scratch/court_det_fix/next_steps_20260916/webui_seed/witnesses.json`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/next_steps_20260916/webui_seed/witnesses.json)

What it establishes:

- GX0's direction-pair-143 product and the all-pair pre-global shortlist are different populations; their nearest-court changes must not be conflated.
- The saved pre-global counts are 30,001 baseline candidates and 19,763 paint-filtered candidates.
- In the Am3 trace, the score-maximising representative for the same axis assignment can be considerably farther from the approved control than another representative.
- The witness records which older producers created the traces. Those paths matter when two `run_automatic.py` files are present; W5 does not need to rebuild every historical producer.

## R6 — L1 admission and the actual axis matcher

- [`scratch/court_det_fix/next_steps_20260916/L1_admission/comparison.csv`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/next_steps_20260916/L1_admission/comparison.csv)
- [`scratch/court_det_fix/next_steps_20260916/L1_admission/run_l1_admission.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/next_steps_20260916/L1_admission/run_l1_admission.py)
- [`scratch/court_det_fix/frozen_helpers_20260914/axis_matching/projective_seed.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/frozen_helpers_20260914/axis_matching/projective_seed.py)
- [`scratch/court_det_fix/direction_agreement/selection.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/direction_agreement/selection.py)

What it establishes:

- L1 tested a fixed-budget admission/diversity scheme rather than a new court scorer.
- `match_axis` sorts by score before deduplicating assignment tuples; the chosen representative is not an arbitrary enumeration artefact.
- Coverage/diversity operates on direction-support sets, not verified court marking identities.
- This evidence is why a later “good court missing” diagnosis should distinguish **retention** from **scoring**.

## R7 — L2 scoring, winner eligibility and saved visual controls

- [`scratch/court_det_fix/next_steps_20260916/L2_scoring/run_l2_scoring.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/next_steps_20260916/L2_scoring/run_l2_scoring.py)
- [`scratch/court_det_fix/next_steps_20260916/L2_scoring/comparison.csv`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/next_steps_20260916/L2_scoring/comparison.csv)
- [`scratch/court_det_fix/next_steps_20260916/L2_scoring/result.md`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/next_steps_20260916/L2_scoring/result.md)

What it establishes:

- L2's legacy winner predicate is `camera_error <= 0.1` plus a non-null legacy paint profile.
- Legacy line ranking maximises exclusive stripe score; paint-first ranking uses `(profile.score, exclusive stripe score)`.
- The coarse pixel profile uses 24 samples per finite interval, five centre shifts, side pixels 6 px away, contrast threshold 10 and a 40% interval vote. In W5 these are historical reference settings: the raw ridge contrast is retained so the old cut can be tested rather than silently inherited as a new gate.
- Saved visual controls:
  - Am2-150 `30:33` — approved/very good
  - SS03-19 `1:60` — usable
  - Am2-28019 `184:4123` — rejected false paint winner
  - SS03-19 `165:6702` — rejected false/hallucinated paint winner

## R8 — L3 temporal pilot

- [`scratch/court_det_fix/next_steps_20260916/L3_temporal/run_l3_temporal.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/next_steps_20260916/L3_temporal/run_l3_temporal.py)
- [`scratch/court_det_fix/next_steps_20260916/L3_temporal/score_matrix.csv`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/next_steps_20260916/L3_temporal/score_matrix.csv)
- [`scratch/court_det_fix/next_steps_20260916/L3_temporal/source_manifest.json`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/next_steps_20260916/L3_temporal/source_manifest.json)
- [`scratch/court_det_fix/next_steps_20260916/L3_temporal/result.md`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/next_steps_20260916/L3_temporal/result.md)

What it establishes:

- L3 used uncapped camera pools, not L2's bounded generation populations.
- Its 30-court panel over seven frames selected a median-line winner that the saved reference says is extremely poor, while other candidates had stronger paint/reverse evidence.
- L3 reference errors are manual-reference metrics and should not be mixed with C2/L2's GX0 approved-control distances.

## R9 — junction semantics

- [`experiments/annotator/independent_court/junction_observations.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/experiments/annotator/independent_court/junction_observations.py)
- [`tests/test_independent_court_junctions.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/tests/test_independent_court_junctions.py)
- [`tests/test_independent_court_stripes.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/tests/test_independent_court_stripes.py)

What it establishes:

- The helper measures six named centre-line sites on a supplied court; it is not a proposal generator.
- Missing/clipped/occluded arms are kept as unknown.
- Existing fragment thresholds are `PRESENT_SUPPORT = 0.55`, `ABSENT_SUPPORT = 0.20`; 16 arm samples are used and the historical classifier requires at least 8 usable samples.
- W5 keeps those historical classifications for comparison but records the raw arm support, sample availability and photometric continuation evidence. The initial W5 pass does not treat the old thresholds as a hard veto.

## R10 — player and camera assumptions

- [`experiments/annotator/independent_court/player_guided.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/experiments/annotator/independent_court/player_guided.py)
- [`experiments/annotator/independent_court/temporal.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/experiments/annotator/independent_court/temporal.py)
- [`experiments/annotator/independent_court/net_geometry.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/experiments/annotator/independent_court/net_geometry.py)
- [`scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis/summarise.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis/summarise.py)

What it establishes:

- Player-guided checks operate on complete court hypotheses; missing player detections remain missing.
- The stricter historical full-court eligibility is: valid geometry, one-player fraction 1.0, two-player fraction at least 0.5, and camera error at most 0.1. W5 preserves this as a diagnostic subset rather than assuming it should gate the first-pass holistic ranking.
- The camera model assumes square pixels, centred principal point and a searched focal length. Its error is useful but conditional.

## R11 — why temporal/scene consensus is gated

- [`src/courtkeynet/court_corners.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/src/courtkeynet/court_corners.py) — historical `consensus_repair`
- [`docs/courtkeynet/fallback_evaluation/README.md`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/docs/courtkeynet/fallback_evaluation/README.md)
- [`docs/courtkeynet/fallback_evaluation/scene_geometry_repair.md`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/docs/courtkeynet/fallback_evaluation/scene_geometry_repair.md)

What it establishes:

- Per-corner consensus assumes a trustworthy same-camera majority and can make a consistently wrong majority look very stable.
- Historical scene repair found whole-video replacement unsafe without donor agreement, target-frame paint and fresh player checks.
- W5 therefore delays temporal pooling until the single-view global judge can reject known aliases.

## R12 — frozen case loading and helper resolution

- [`scratch/court_det_fix/line_identity/shared.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/line_identity/shared.py)
- [`scratch/court_det_fix/next_steps_20260916/webui_seed/source/SOURCE_MAP.md`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/next_steps_20260916/webui_seed/source/SOURCE_MAP.md)
- [`scratch/court_det_fix/frozen_helpers_20260914/vp_pruning/run_population.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/82d3b871bfb784f91b283761148eaa0d6501c732/scratch/court_det_fix/frozen_helpers_20260914/vp_pruning/run_population.py)

What it establishes:

- `I/shared.py` is the authoritative nine-view identity/provenance/coordinate helper for this experiment family.
- Same-image person boxes are available for GX0, Am2-150, Am2-28019 and Am3-0. GX5's chosen boxes come from frame 6 rather than the frame-5 anchor, and the four broadcast cached images are median thumbnails while their boxes come from a source frame. Those five box sets are not valid photometric masks for the W5 image.
- Some historical scripts with the same basename are not the same producer. `SOURCE_MAP.md` and the existing import-path setup explain which snapshot L2 intended.
- `run_population.prepare` recreates the cached fragment families and working-image scaling without needing references.

## Access and interpretation limits

The first handover review did not run the repository detector or inspect native image pixels. It relied on saved visual rulings for the four diagnostic courts. The refreshed pack resolves implementation contracts from the source, but it still does not claim that W5 has been run.

The historical path `scratch/court_det_fix/worklog/checks/independent/player_guided/20260909/paint_geometry/` was not present at the reviewed revision. That only limits that old trace; it does not imply the user's local host lacks other archived material.

The numerical checks in this pack use a compact selection of saved real-data fields, not the full image dataset or every candidate row. Their role is to keep the important arithmetic distinctions honest, especially the C2 population difference and the selected L3 candidate medians.
