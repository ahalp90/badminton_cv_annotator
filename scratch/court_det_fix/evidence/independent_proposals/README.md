# Independent court proposals

No independent proposal is production-ready. The settled finding is that line
extraction can supply useful evidence, but selecting the target playing court
and accepting it safely remain separate unsolved problems. The annotation
pipeline does not use these outputs. This note replaces the overlapping
OpenCV, neural-line and player-guided narratives while preserving the exact
recorded bundles and the pending direction-selection evidence.

## Settled comparisons

The first comparison used 20 cached broadcast medians, 11 labelled amateur
frames and 24 broadcast controls. It was development data: search settings
changed during the work, there was no holdout, and two broadcast medians had
unverified mixed views.

The OpenCV prototype accepted 2/20 broadcast medians, 3/11 amateur frames and
3/24 controls. All three accepted amateur fits were wrong; their worst-corner
errors were 891.9 to 925.7 pixels. One accepted control was a building. The
candidate pool contained an accurate candidate in 16/18 matching broadcast
views and 0/11 amateur views, but that label-guided pool inspection is not an
automatic selection rule.

Frozen neural line extractors did not solve the same bottleneck. All methods
produced zero accurate amateur top proposals and zero accurate amateur
accepts. Painted-stripe Hough was the strongest broadcast variant at 10/18
accurate accepts with no wrong accepts. DeepLSD MegaDepth default reached
12/18 accurate accepts but also accepted three wrong courts. Broader direction
groups recovered reference-line support in up to 10/11 amateur diagnostics,
yet MegaDepth default ranked a wrong court first in every amateur frame. A
separate footpoint check accepted one 126.8-pixel wrong amateur court and one
1,654.1-pixel wrong broadcast court. More lines and a person check are not a
sufficient admission rule.

Nine later example frames were manually annotated, three per new clip. The
published neural results have not been rescored against those annotations.
They are a future comparison set, not evidence of a corrected score.

## Temporal, paint and projective branches

The short temporal/player-guided replay shows why player and net evidence is
useful but incomplete. Across three clips, floor-only to multi-fragment worst
corner errors were Yellow 221.19 to 21.22 pixels, Letterboxed 10.06 to 14.75,
and Centre 4.04 to 3.20. Two clips meet the descriptive 15-pixel criterion
after refitting, but the replay does not define automatic acceptance. Selecting
the intended player pair is a separate problem, and missing detections remain
missing.

The newer projective matcher has a clearer bottleneck. Supplied directions
produced usable geometry in eight inspected views. Automatic direction
selection produced a usable winner on six of nine views only after a human
choice between line and paint rankings. GX0, GX5 and Amateur-2 frame 28019
remain poor automatic cases. Supplying discarded observed directions to the
unchanged matcher produced close or ideal courts; this proves matching
capacity, not automatic direction recovery.

The fixed-membership SVD diagnostic is cheap but partial. In 960x540 working
coordinates, best diagnostic error changed as follows:

| Case | Original | Fixed-group SVD |
| --- | ---: | ---: |
| GX frame 0 | 6.798 | 3.961 |
| GX frame 5 | 23.800 | 22.502 |
| Amateur-2 frame 28019 | 2.972 | 1.626 |

The 16 refits took 1.1–1.9 ms per case. The run evaluated fixed line-group
membership and did not run the matcher or a new visual check. Control-selected
groups are label-guided and are not a performance ceiling. The next useful
comparison is line-agreement measurement, then direction/group selection, then
ranking against saved wrong-court examples. kNN and a larger early pool remain
conditional ideas, not results.

## Other branches and why their records remain useful

These are distinct tests, not successive versions of one accuracy number.
The 20-frame studies used a 15-pixel worst-corner criterion at 1280×720.
That includes unseen corners and is not the same as visual usability.

| Tested change | Result and implication |
| --- | --- |
| Partial marking assignment | 5/20 accurate selections versus 7/20 existing and 8/20 bidirectional scores, on 682 eligible fixed courts. This limited the tested representation, not every possible court-graph matcher. |
| Fixed-assignment stripe refitting | Keeping starts beside refits changed 9/20 to 10/20 under the old cutoff. Refits can also worsen courts; a smaller fitting objective is not an identity check. |
| Marking refit acceptance | Accepting improvements among 19 current leaders helped four and harmed none, but the same rule across all 357 pairs chose 35 harmful refits. The leader-only result did not generalise even within the saved pools. |
| Physical paint geometry | All 1,364 requested fits converged, but winner and boundary effects were mixed. Post observations exposed missing compatible proposals on Amateur-3 frame 24515 and Amateur-4 frame 0. |
| Wider broadcast ranking | Stripe ranking reached 16/18 on matching views. Corrected junction-first ranking reached only 1/18 under both mask policies. The former 3/18 figure is invalid. |
| GX proposal trace | Seven frames generated 4,271,850 proposals; 521,473 passed player/geometry checks, one passed floor support, and none survived the camera check. Frame 0 had a useful rejected start; frame 5 also had a sampling-coverage problem. |
| Fixed-court directional evidence | On 63 fixed courts across 22 frames, direction-compatible fragments restored some useful support. One good amateur court regressed before angle filtering; this did not test generation or selection. |
| Raster-distance access comparison | Combining direction-compatible fragments with an all-fragment merged pool let the usable GX5 court pass and rejected two wrong courts, but also rejected a good Amateur-3 fit. Changing distance implementation did not explain away that regression. |
| Spread seed sampling | At the same 4,096-seed budget it improved nearest geometry on five of seven GX frames, accepted none, and regressed both controls. Keep the original sampler. |
| Covered-length-ranked seeds | Weight 1 improved nearest geometry on all seven GX frames and preserved both controls. It still accepted no GX court; the new frame-5 example was visually rejected despite lower error. Keep it diagnostic. |

The [recorded replay archives](../../../../experiments/annotator/independent_court/recorded/player_guided/)
retain complete comparison records and settings. The
[box-provenance account](../holistic_admission/box_provenance.md) identifies
unsafe historical masked arms separately from valid unmasked results.
The corrected frozen junction packet leaves all three scheme winners
unchanged across 20 cases; it does not repair every older masked experiment.

The [fixed-refit audit extracts](refit_audit/) retain the numerical review
inputs and three sample fits from the 9 September review. They are saved
extracts, not a fresh validation of the old masked arms. The 9/20 to 10/20
count used an off-screen-corner cutoff; it does not establish improved
visible alignment. Full replay records remain in the bundles below.

## Retained evidence and runnable routes

The reusable implementation and focused tests are in the
[independent-court package](../../../../experiments/annotator/independent_court/README.md).
The compact statistical evidence is retained once in these bundles:

- [`original.json.gz`](../../../../experiments/annotator/independent_court/recorded/original.json.gz),
  [`amateur.json.gz`](../../../../experiments/annotator/independent_court/recorded/amateur.json.gz),
  [`controls.json.gz`](../../../../experiments/annotator/independent_court/recorded/controls.json.gz)
- [`ridge.json.gz`](../../../../experiments/annotator/independent_court/recorded/ridge.json.gz)
  and [`neural.json.gz`](../../../../experiments/annotator/independent_court/recorded/neural.json.gz)
- the four frozen caches under
  [`recorded/neural_lines/`](../../../../experiments/annotator/independent_court/recorded/neural_lines/)

The latest projective pack is the [projective-pattern evidence
snapshot](../../../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/).
Its records include automatic arms, ranking examples, visual rulings, GX0
control measurements and SVD measurements. It is not a full candidate
population or a standalone replay package.

Keep the exact replay inputs needed by current scripts and tests:

- [`marking_refit_replay.zip`](../../../../experiments/annotator/independent_court/recorded/player_guided/marking_refit_replay.zip)
  and [`assignment_results.json.gz`](../../../../experiments/annotator/independent_court/recorded/player_guided/assignment_results.json.gz)
- [`stripe_diagnostics.zip`](../../../../experiments/annotator/independent_court/recorded/player_guided/stripe_diagnostics.zip),
  [`stripe_results.json.gz`](../../../../experiments/annotator/independent_court/recorded/player_guided/stripe_results.json.gz)
  and [`paint_results.json.gz`](../../../../experiments/annotator/independent_court/recorded/player_guided/paint_results.json.gz)
- the fixed/raster matching, temporal and extension replay archives in the
  same `recorded/player_guided/` directory.

Run the focused checks from the repository root:

```bash
pytest -q tests/test_independent_court.py \
  tests/test_independent_court_temporal.py \
  tests/test_independent_court_people_export.py \
  tests/test_independent_court_player_guided.py \
  tests/test_independent_court_assignment.py \
  tests/test_independent_court_stripes.py
```

For a cached neural comparison, use the evaluator and one retained line cache:

```bash
python -m experiments.annotator.independent_court.evaluate \
  --manifest scratch/court_det_fix/evidence/independent_proposals/development/inputs/original/manifest.json.gz \
  --line-cache experiments/annotator/independent_court/recorded/neural_lines/deeplsd_md_default.json.gz \
  --output /tmp/independent-court-neural
```

## Preservation boundary

Retain [the full 20260914 populations](development/player_guided/20260914/).
It contains full direction/selection candidate populations behind the pending
comparisons. The public projective pack is a result snapshot, not a substitute
for those populations. Current direction tests still import the old worklog helper
path through a compatibility link. The data and helpers now live together in
`development/`; producer behaviour is unchanged.

The canonical [input frames and manifests](development/inputs/) also survive:
public JSON records preserve metadata and hashes, not the image pixels. Keep
the complete `examples_updated/` directory, including its MP4s, PNGs and
corner CSVs. `gxBQ_HwdgN4.mp4` is a 419 MB original reusable video, not a
publication intermediate. Retain original video and frame inputs in the
[source collection](development/).
Incomplete `.mp4.part` downloads are separate discarded transfers, not usable
source videos.
The 2026-09-08 annotated example dataset remains unrescored. Do not invalidate
geometry results solely because a separate player-box provenance issue exists;
the experiments distinguish geometry, player evidence and production
acceptance.

The supplied-direction scene-16 candidates remain unjudged. Keep the exact
[visual rulings](../../../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/axis_matching_visual_judgements.md)
beside their images. Older observed-bank controls, the remaining GX frames and
the cause of apparent bowing are unfinished historical questions, not an
automatic queue of new experiments. Unknown visibility of the far GX centre
line must not be silently relabelled as occlusion.

Historical source paths for the backup are
`docs/courtkeynet/fallback_evaluation/independent_detector.md`,
`neural_lines.md`, and the former reports under
`experiments/annotator/independent_court/recorded/player_guided/`.
The earlier literature survey is recoverable as
`local_scratch/court_corner_detector_research/REPORT.md` and `CURRENT_STATE.md`.
Its CourtKeyNet recommendation is superseded; its source bibliography remains
historical research, not a current shortlist.
