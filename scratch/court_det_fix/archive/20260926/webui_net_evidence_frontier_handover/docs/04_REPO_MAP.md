# Repository map for net-assisted court work

## Branch and revisions

```text
Branch: fix/court-det
Observed tip: 1353541e9e0f08633a12c746f9690785af543aa5
Frozen Am1/GX audit revision: 13f02fdbf8954ead15dbc7241a696a314473f9c5
Historical net-geometry commit: be94d28dbc78289aa21386bbefcf059d416f407e
```

The branch tip may advance independently of this pack. The paths below were
present at the observed tip.

## Net geometry and historical scorer

```text
experiments/annotator/independent_court/net_geometry.py
experiments/annotator/independent_court/player_guided.py
tests/test_independent_court_net_geometry.py
tests/test_independent_court_player_guided.py
scratch/court_det_fix/frozen_helpers_20260914/legacy/zone_net.py
scratch/court_det_fix/frozen_helpers_20260914/legacy/camera_diagnostic.py
```

## Historical replay and interpretation

```text
experiments/annotator/independent_court/recorded/player_guided/replay.zip
experiments/annotator/independent_court/recorded/player_guided/summary.json.gz
experiments/annotator/independent_court/recorded/player_guided/yellow_net.jpg
experiments/annotator/independent_court/recorded/player_guided/projective_patterns/evaluation/temporal_assessment.md
scratch/court_det_fix/evidence/independent_proposals/README.md
scratch/court_det_fix/DETECTOR_DECISIONS.md
```

The replay archive contains retained proposals, observations, references, code,
and complete saved results for the three historical clips. Its published
account notes that source images are omitted from the compact archive.

## Current W5 and gate path

```text
scratch/court_det_fix/w5_holistic/run_w5.py
scratch/court_det_fix/w5_holistic/verifier.py
scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis/run_diagnosis.py
experiments/annotator/independent_court/detector.py
experiments/annotator/independent_court/fixed_stripe_refit.py
experiments/annotator/independent_court/stripe_observations.py
experiments/annotator/independent_court/assignment.py
experiments/annotator/independent_court/paint_geometry.py
```

## Frozen image corpus

```text
scratch/court_det_fix/frozen_views/frames/
scratch/court_det_fix/frozen_views/packs/
```

Known cases include:

```text
scratch/court_det_fix/frozen_views/frames/amateur/am1/frame_00000054.png
scratch/court_det_fix/frozen_views/frames/gx/images/gxBQ_window_00_frame_00000005.png
```

The subtree also contains many galleries and overlays. Their usefulness varies:
raw source frames support image measurement, while rendered overlays often
contain labels or historical geometry that can leak into naive detectors.

## Candidate and ranking evidence

Likely sources of frozen candidate populations and score records include:

```text
scratch/court_det_fix/w5_holistic/
scratch/court_det_fix/archive/
scratch/court_det_fix/evidence/
scratch/court_det_fix/next_steps_20260916/
experiments/annotator/independent_court/recorded/player_guided/
experiments/annotator/independent_court/recorded/player_guided/projective_patterns/
```

Frequently useful filename patterns include:

```text
*results*.json.gz
*records*.json.gz
*ranking*.json.gz
*candidates*.json.gz
*population*.json.gz
*inputs*.json.gz
*summary*.json.gz
*.zip
```

The branch documents distinguish result snapshots from full candidate
populations. Some public packs preserve scores and winners without the entire
search population; some development directories retain much larger populations.

## Am1/GX local packet

The exact two-case packet included in this handover originated at:

```text
scratch/court_det_fix/edge_polarity/local_audit/webui_net_audit/cases.json.gz
```

It avoids reliance on large uncommitted caches and contains enough information
for fixed-membership local analyses, but not enough competing candidates for a
full reranking study.

## Broader image and annotation sources

Additional court imagery and labels also exist outside `scratch/court_det_fix`,
including:

```text
data/amateur_court_corners/2026-09-08/
experiments/annotator/independent_court/recorded/
```

Those sources may support detector prototyping or validation, although their
annotations, image kinds, and historical metrics differ from the W5 packet.
