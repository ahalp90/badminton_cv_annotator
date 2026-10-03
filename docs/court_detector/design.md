# Court detector design

The detector combines straight-line evidence, the known badminton court layout
and player positions to find the four outer court corners. It uses pretrained
line and pose models, but does not need a court model trained for each venue.

The [overview](README.md) describes the main stages; the [usage guide](usage.md)
covers setup and commands. [Sampling and output modes](sampling_and_reuse.md)
explains how video frames contribute to a result. This page preserves the
reasons behind the design; [evaluation](../../experiments/court_detector/comparisons/development_evaluation.md) holds the measurements
and their limits.

## What the geometry represents

Four court corners define a **homography**, a mapping between the flat court
floor and image pixels. The annotator uses it to express player and shuttle
positions in court coordinates. The mapping assumes a flat floor and straight
projected lines. It cannot correct lens distortion or follow a moving camera
with one fixed set of corners.

A **marking** is a named court line, such as the right doubles sideline. Its
painted stripe has a width. The detector must decide whether an observed line
represents the stripe's centre or an edge, as well as which marking it belongs
to. Confusing those two decisions can produce a plausible but misplaced court.

The internal score combines three kinds of evidence:

- **Paint support:** how well projected markings land on bright, paint-like stripes.
- **Geometry support:** how well detected line fragments support the court layout.
- **Net reward:** evidence near the two expected net posts.

The default ranking is `0.9 × paint + 0.1 × geometry + 0.04 × net reward`.
The net reward ranges from zero to one. This is a way to choose candidates,
not an accuracy measurement against labelled courts.

## Why search from lines

The earlier CourtKeyNet repair chain produced no scoreable courts on its
11 labelled amateur frames. Relaxing the checks would have accepted bad
geometry. The replacement therefore searches court layouts supported by the
current image, with player positions helping to identify the court in use.

| Choice | Reason |
| --- | --- |
| Search all line fragments and paint-like fragments separately | Either search can recover useful courts that the other misses. Pale-paint filtering alone loses some difficult courts. |
| Check where players stand | Many line arrangements look like a court but do not contain the players. The detector uses box bottoms as foot-position estimates, excluding mostly seated people. |
| Hide people from paint measurements | Players obscure stripes and introduce unrelated edges. Boxes may mask an image only when they came from that image. |
| Give paint most of the score | General line support can favour walls and boards. A small geometry share resolved one tested far-end slip. |
| Cap the net-post reward | Net evidence recovered useful courts, but an unrestricted preference displaced other good choices. Missing net evidence stays neutral. |
| Relabel stripe centres and edges only on clear evidence | Treating a stripe centre as its outside edge can pull the fitted court inward. Ambiguous evidence should not force a relabelling. |
| Reject sideways and upside-down camera geometry early | These candidates are unsuitable for the intended footage and waste expensive scoring work. |
| Keep all 16 direction groups and 256-court shortlists | Smaller searches saved work but removed useful alternatives or worsened complete-detector results. |

Required-player searches prune impossible axis hypotheses before scoring or capping
them. Combined courts need a player on the court in every sample and players in both
halves in at least half the samples. Equal scores keep their original order.
No-player detection uses line templates by default; `--full` adds direction-pair
searches without the player gates, using player support only to break exact score ties.

These choices came from development cases. They are useful defaults, with no
claim that their thresholds are optimal on unfamiliar footage.

## Why combine observations across frames

A player may cover a marking in one frame and reveal it in another. A fresh
scene search therefore combines the strongest observations of each marking
from its first, middle and last sampled frames. This **composite** is a fitted
court assembled from several observations, rather than an averaged image.

The middle and endpoint searches run independently. The accepted individual courts
and the composite compete on the same middle-frame evidence after their checks.
The best score wins; equal scores keep an individual court ahead of the composite.
The [earlier visual comparison](../../experiments/court_detector/comparisons/development_evaluation.md#composition-visual-review-and-consistency)
records examples and its small sample size.

Chronological reuse saves searches when an earlier court can be aligned,
refitted and validated in a returning view. A reused court cannot become a
new reuse reference. This prevents a sequence of small adjustments from
accumulating through the video.

## Why video pooling needs a whole-scene alternative

Video-robust mode combines observations from accepted individual frames, including
scenes whose composite failed or lost. A group needs at least two independent donor
scenes. Reused courts can receive the result but cannot donate evidence.
The final fit minimises distances between court lines and the donated samples;
it does not maximise the paint/geometry/net score.

That distinction matters when scenes disagree about a marking's identity.
In video 003, two scenes gave the same physical stripe different names. The
pooled fit compromised between the incompatible constraints and put lines on
blank floor. Aligning the images correctly did not resolve the naming conflict.

The detector compares the pool with the completed scene courts and retained individual
courts carried across the group. Each alternative must pass its source frame's checks
and have a source score. It ranks by its mean score over the members that can score it.
In video-robust mode, equal scores retain the pool. Fast-robust mode favours a scene
court on an equal score. Both modes can choose an eligible existing court when the
pool fit fails. Each member keeps its own court if it rejects the chosen alternative.
The [cached comparison](../../experiments/court_detector/comparisons/development_evaluation.md#cached-pooling-checks) records both a
whole-scene winner and a pooled winner.

Pooling is an offline operation: earlier scenes can use evidence from later
ones. It runs after the chronological pass and never feeds pooled courts back
into reuse. A prepared scene without a court can receive a matching group's chosen
court after alignment, geometry and camera checks. Receivers never donate or change
the group's choice.

## Useful optimisations

The retained changes reduce repeated work while keeping candidate order and
numerical behaviour stable:

- Worker processes stay alive across stages and scenes. Independent direction
  pairs and candidate scores run in parallel, then return to their original
  order before ranking. Equal-score ties depend on that order.
- Bulk arrays use float32; fitting and sensitive geometry retain float64.
  Template scores sum integer counts before division to preserve ties.
- A cached Numba kernel compiles the direction-pair support calculation.
  It runs serially inside each worker to avoid competing layers of threads.
- CuPy evaluates line templates on the GPU using the same calculation as the
  NumPy CPU path. Neural inference and template scoring have separate device
  settings.
- Required-player checks skip a search when the available people cannot
  possibly satisfy its occupancy rule.

[Recorded timings](../../experiments/court_detector/comparisons/development_evaluation.md#controlled-timing) cover the retained worker
and GPU implementation before three-frame composition. They do not establish
the speed of the final video-robust mode. The target of roughly 30 seconds per
five minutes of footage, with 90 seconds as an upper target, remains unmet.

## Approaches tried and dropped

| Approach | Why it was dropped or left optional |
| --- | --- |
| CourtKeyNet fallback and scene repair | Failed the amateur check; relaxing acceptance checks admitted inaccurate geometry. |
| Keep 12 of 16 direction groups | Saved matcher time but clearly worsened two views when checked in the full detector. |
| Colour-based court rejection | The tested rules did not reliably reject wrong courts and failed a pale same-hue control. |
| Unrestricted net-post preference | Lost good choices; replaced by the small capped reward. |
| Shortlists of 128 instead of 256 | Preserved the saved winners but removed close alternatives on six hard views. |
| Stricter paint test bounded by the next stripe | Fixed neither targeted far-end slip, worsened another view and took longer. |
| Refit only the top 15 courts before choosing | Recovered one view but worsened two others. |
| Whole-line paint contrast | Distance, lighting and stripe width prevented reliable comparison between lines. |
| Reject implausible player widths | Still let a slipped court win, with no clear speed gain. |
| Different lengthwise/crosswise score blends | A version that repaired one view displaced a whole court end in two others. |
| Cheap-first scoring (`full_score_limit`) | Retained as an optional trial. A small-sample score shortlists courts for full scoring, but has not been shown to preserve choices on new footage. |
| Torch scoring backend | Its measured gain did not justify adoption beside CuPy. |
| CUDA-MLIR scoring backend | Promising component timings and three final-court checks provided less complete whole-path evidence than CuPy. Further backend work was set aside. |
| Shared memory for worker inputs | A laptop probe transferred roughly 6 MB in milliseconds. The expected saving did not justify shared-memory lifecycle code; server transfer time was not measured. |
| A 1-pixel camera-movement veto for pooling | Rejected a real returning view even though alignment already accounted for movement. Chronological reuse keeps its own checks. |
| Unconditional pooling | Pulled the good video 003 court onto blank floor. |
| Reassign samples to each scene composite before pooling | Repaired video 003 but reduced the video 040 comparison score. The whole-scene alternative preserved the better cached results. |

The [earlier approach comparisons](../../experiments/court_detector/comparisons/earlier_approaches.md) explain the main
rejected alternatives. The defaults above describe the maintained detector.

## Limits a maintainer should expect

- **False courts remain possible.** Crowd shots, referee chairs and advertising
  boards have produced accepted courts. Geometry and player checks reduce this
  risk but do not establish correctness.
- **Net tape can look like a court boundary.** A finite, camera-plausible fit
  can still assign net tape to a floor marking. Geometry, greyscale contrast
  and the tested net-mesh cue did not reliably distinguish them. Same-paint
  colour helped one of two frozen cases; there is no calibrated colour gate.
- **Far-end and partial-view errors.** Small image offsets can move a boundary
  far across the floor. Close-ups, heavy occlusion and lens distortion remain
  difficult even when a few strong lines are visible.
- **One projection per scene.** Missed cuts, pans and zooms can make the middle
  frame's projection unsuitable elsewhere in the scene.
- **Short scenes are often skipped.** Required-player mode needs the scheduled
  three-second window. Optional-player mode can analyse shorter scenes, with
  weaker evidence for accepting a court.
- **Alignment can be wrong despite a passing score.** Repeated patterns or heavy
  occlusion can align the wrong structures. Pooling adds no new labelled
  assurance that the images show the same view.
- **Reuse thresholds remain heuristic.** Its paint and movement checks have
  not been tuned on a labelled set of matching camera views.
- **Pooling scores and costs need care.** A pooled diagnostic mean may include
  a member that rejects that court. Checking every donor court in every member
  also grows with the product of those counts; line maps are rebuilt for each
  check. In the completed video 040 run, final group comparison took 346 s
  (4.25% of wall time), with 71 donors in the one eligible group.
- **Pose extraction can omit people.** The full pose pre-run keeps ten people
  per frame. A crowded image can lose a player and change the court decision.

Maintenance depends on preserving coordinate units, scene intervals, candidate
order and the rule that inferred results do not become new donors. The
[source README](../../src/court_detector/README.md#code-map) links the modules
and tests. Numerical comparisons need the same environment and inspection of
the chosen courts; a higher score can accompany a worse fit.
