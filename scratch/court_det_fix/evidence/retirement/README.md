# CourtKeyNet retirement evidence

CourtKeyNet and its fallback/scene-repair chain are retired as detector
contenders. The chain remains useful as a bounded broadcast baseline and as a
source of non-model lessons. It must not receive more recovery exceptions or
be presented as a solution for amateur footage.

This note replaces the earlier fallback narratives. Their paths are retained
as historical source citations for the tidy backup, not as required reading.

## What the chain measured

The wrapper threshold was selected on 400 court-view frames from 10 ShuttleSet
broadcasts and 63 hand-checked non-court frames. At peak floor `0.02`, 88% of
court frames passed, seven videos reached 100%, every non-court frame was
rejected, and the median corner error was 3.5 pixels in 1280x720 coordinates.
Per-video medians ranged from 2.2 to 5.5 pixels. The fallback requires at
least two confident neural corners, enough line evidence in a static scene,
and line/anchor reprojection gates. Zero or one confident corner returns no
court.

The video-wide consensus repair fixed a specific fixed-camera failure. In
video 3 it flagged 7 of 46 scenes: clean scenes were at most 18.1 pixels from
the scene median, while aliased scenes were at least 177.3 pixels away. Mean
error changed from 23.1724 to 5.3698 pixels and P90 from 107.2800 to 10.9483.
Video 21 flagged no scenes and remained at 4.7757 mean and 5.0847 P90. The
55-pixel threshold is tied to these 1280x720 fixed-camera controls.

The amateur check is the retirement boundary. Across 11 labelled frames in
eight scenes and four videos, the chain produced 0/11 scoreable courts. Raw
ungated median errors were 849, 453, 457 and 376 pixels. Consensus could not
help because no scene produced a quad to vote on. Relaxing the gates would
have accepted inaccurate geometry.

A later scene-repair integration improved two known ShuttleSet22 failures:
at a ±10-frame contact tolerance, video 17 F1 changed 0.877 to 0.881 and video 53 changed 0.342 to
0.961, with no previously correct rally section lost. These were development
checks on known failures. They did not measure amateur footage or the
512x288 proxy path. Downstream annotations had to be regenerated before
scoring.

## Non-model lessons worth keeping

The following findings remain useful to any replacement detector:

- A donor must agree with the target's confident anchors. The target's own
  painted lines must support a borrowed court.
- Inferred repairs must not become new donors. Otherwise one bad repair can
  spread through scene order.
- Finite painted-line coverage and a fresh person check reduce false courts,
  but neither is a sufficient detector. The pilot's 50% coverage floor was
  validated only on selected examples.
- Repeated-view grouping improves calibration consistency, not proven accuracy.
  Original video 21 gained seven grouped scenes (35 to 42) while shared-median
  error changed only 4.636 to 4.632 pixels. A pooled visible-line fit was
  worse than the median on both original controls: 4.62 to 6.73 pixels in
  video 3 and 4.62 to 4.94 in video 21.
- Camera novelty alone cannot veto a live view as replay. Colour priors are
  also unsafe: the video-specific lightness split did not generalise to venues
  or line paint.

## Evidence and rerun route

The compact deterministic check uses the current `consensus_repair()` code and
two compressed scene records:

- [consensus checker](../../../../docs/courtkeynet/fallback_evaluation/check_consensus_repair.py)
- [video 3 input](../../../../docs/courtkeynet/fallback_evaluation/recorded_inputs/fb5_3.csv.gz)
- [video 21 input](../../../../docs/courtkeynet/fallback_evaluation/recorded_inputs/fb5_21.csv.gz)

Run it from the repository root:

```bash
~/.venvs/badminton-cicd/bin/python \
  docs/courtkeynet/fallback_evaluation/check_consensus_repair.py
```

The five small figures under
[legacy/figures/independent/](legacy/figures/independent/) show the decisive
false accepts and the neural-line selection failure. The saved production
comparison and its manifest are linked from the [court-geometry evidence
bundle](../../../../experiments/annotator/court_geometry_repair/README.md).

The [issue-148 fixture checker](issue148/tools/check_issue148.py) preserves a
specific failure mechanism without loading the model. Video 53 scene 334's
saved fallback quad is non-convex and misordered despite passing the saved
residual thresholds. Its top-right corner moved 712.7927 pixels from the raw
neural corner. The checker also demonstrates scale sensitivity and an unsafe
camera-change replacement on synthetic examples; those examples are not
replays of video 17.

## Limits and status

These results cover professional broadcasts, two fixed-camera consensus
videos and known ShuttleSet22 failures. They do not establish performance on
amateur, moving-camera or arbitrary partial-court footage. The historical
source notes also contain revision-specific test counts; those counts must not
be mixed as if they were one run.

Historical sources for the backup are
`docs/courtkeynet/fallback_evaluation/README.md`,
`scene_geometry_repair.md`, `scene_geometry_repair_worklog.md` and
`scene_grouping.md`. The independent replacement work is consolidated in
[`independent_proposals`](../independent_proposals/README.md).
