# Video sampling, reuse and output modes

The video runner gives **at most one court per scene**. A court here means four
corner points in the image, which together fix where the whole court floor sits.
This page is the contract for how each scene is sampled and how scenes can share
evidence. The [usage guide](usage.md) has the commands, setup and top-level output
fields. The [design page](design.md) explains why, and
the [evaluation page](../../experiments/court_detector/comparisons/development_evaluation.md) holds the evidence.

These options decide how scenes relate to each other:

| Behaviour | Option | When it acts | What a scene can end up with |
| --- | --- | --- | --- |
| Scene-robust | `--court-mode scene-robust` | Per scene | Its best accepted individual court or combined refit |
| Chronological reuse | `--reuse-courts` | Per scene, in time order, before the search | An earlier scene's court, refitted and checked in this scene |
| Video-robust (default) | `--court-mode video-robust` | Once, after every scene has finished | One court shared by scenes of the same camera view, or its own court |
| Fast-robust | `--court-mode fast-robust` | Fits each middle frame, then pools after every scene has finished | A pooled court, the best complete middle-frame court, or its own middle-frame court |

Reuse and video-robust mode are independent and can run together. Both are
described in full below.
Fast-robust requires independent fresh fits and rejects `--reuse-courts`.

Separately, `--fast` and `--full` choose search breadth within each frame when
players are optional. Omitted flags or `--fast` use line templates; `--full` adds
all-lines and painted-lines searches. They are mutually exclusive and leave
player-required search unchanged. A still image remains one input frame.

## Scenes and frame ranges

A scene runs from one camera cut to the next.

| Run configuration | Where the cuts come from |
| --- | --- |
| `--pyscenedetect` | PySceneDetect's ContentDetector, threshold 27, with the annotation pipeline's minimum scene length (about half a second) |
| `--scenes FILE` | Saved ranges |
| Neither | The whole video is one scene |
| Dataset builder | PySceneDetect |
| End-to-end annotator | Its fixture's saved, shared scene ranges |

The minimum scene length only controls cut detection. It does not decide
whether a scene is long enough for the player checks.

Frame numbers start at zero. Scene ranges are **half-open**, like Python's
`range`: `[start_frame, end_frame)` holds `start_frame` up to `end_frame - 1`.

- A whole video is `[0, frame_count)`
- A cut frame is one scene's `end_frame` and the next scene's `start_frame`, so
  it belongs only to the later scene
- Saved ranges must cover `[0, frame_count)` in order with no gaps or overlaps.
  An old file of inclusive ranges leaves gaps, so the runner rejects it

Older results used inclusive ends and stored `first_frame` and `last_frame`. An
old `[first, last]` is `[first, last + 1)` now. Results with no `schema` field
analysed the lower middle frame of an even-length scene, one frame earlier than
now.

## What gets sampled

Each scene has one **anchor**: its middle frame, `(start_frame + end_frame) // 2`.
For an even number of frames this is the later of the two middle frames.

| Purpose | Frames used |
| --- | --- |
| Find lines, search and fit the court | The anchor; also the scheduled endpoints in scene-robust and video-robust modes |
| Check players' positions | 31 frames at about 10 per second, three seconds around the anchor |
| Compare individual courts and a combined refit (fresh searches) | The anchor plus the first and last frames of that player window |
| Compare views for `--reuse-courts` | A median image of the window's first frame, the anchor and the window's last frame |
| Group views in video-robust mode | The anchor alone, with people left out of the alignment |

The player window slides as a block to stay inside the scene. Frame numbers are
rounded to the video's frame rate. For example, a 30 fps video with its anchor
at frame 300 samples frames 255, 258, …, 345.

A shot check then compares small greyscale thumbnails of each sample with the
anchor. It keeps the unbroken run of samples around the anchor whose average
difference is at most eight grey levels. This drops frames from across a missed
cut. Moving players and camera motion can also trip it, so a dropped sample does
not prove a missed cut.

The detector keeps standing people in those samples. It filters out seated
people and restores brief crouches on tracks that mostly stand. Their feet are
checked against every candidate court. These samples are player evidence; they
are not 31 separate court detections. Tracks can swap identities when players
cross, which can affect the standing/crouching decision.

**Short scenes.** With people required, a scene that cannot hold the window gets
`scene_too_short_for_feet`. That means unanalysed, not "no court". With
`--no-require-people`, a short scene is still analysed. It uses only the anchor's
people, if any, and is never composed. For reuse, its median image uses the
scene's first frame, anchor and last frame instead.

Saved poses and live sparse RTMLib give the same sample frames. A full pose
prerun extracts every frame for later pipeline stages; it does not add samples.

## Composing a court within a scene

Scene-robust and video-robust search the middle frame and both scheduled window
endpoints independently. A rejected or failed middle search does not prevent
endpoint searches. Composition needs a scene long enough for the window.
Successful chronological reuse and fast-robust skip endpoint searches.
Required-player mode also skips all three searches when the shared feet samples
contain too few people to satisfy the occupancy rule. It keeps the prepared middle
frame so the scene can receive a shared court. Diagnostic artefact runs still search.

1. **Search each frame.** Endpoints use their own lines and person boxes with the
   middle frame's feet. Each fresh court must pass its source checks. A failed
   endpoint is logged and left out.
2. **Pick a reference and align.** The highest-scoring accepted frame supplies the
   reference. Exact ties go middle, first, last. Missing score terms do not veto
   the scored frames. Image alignment needs correlation of at least 0.8, with
   person boxes left out. The middle image can align using the reference court as
   an initial mask even when its own search found no court.
3. **Refit the clearest markings.** With at least two aligned accepted frames,
   each marking contributes the stripe samples from its strongest paint donor.
   One court is fitted to those samples.
4. **Compare in the middle image.** Every carried individual and a valid refit
   receive the same paint, line and net score there. Transfer checks require
   valid geometry, a plausible camera and uprightness when enabled. They leave
   out the middle frame's player requirement; the source courts already passed
   their own player checks. The middle frame's own candidate retains its checks.
5. **Keep the best passing court.** Exact ties prefer an individual in middle,
   first, last order. A failed or worse refit preserves the best individual.
   One accepted endpoint can supply the scene's court without a combined fit.

The row's `composition.court` is `middle`, `first`, `last`, `composite`, or `null`
when no court survives. `fallback_reason` explains a missing or losing refit.
Accepted aligned frames retain their stripe evidence and complete court candidates
for video sharing, even when composition fails or an individual wins.

The endpoints remain the scheduled window ends, even when the shot check drops
one. Alignment and transfer checks decide whether their courts can contribute.

## Chronological reuse (`--reuse-courts`)

Reuse tries to carry an earlier court into a scene that returns to the same
camera view. It works one scene at a time, in time order, and changes only the
current scene. It is off in the standalone runner and on in both supplied
dataset-builder configs.

For each new scene the runner:

1. **Builds the scene's evidence.** It reads the anchor, finds its lines and
   player samples, and makes the median image. The median keeps the static court
   and removes most moving players.
2. **Picks earlier courts to try.** It keeps a store of up to **eight** recent
   courts and tries at most **three**. With PySceneDetect, scenes are tried in
   order of luminance-histogram similarity. Saved scene files carry no histograms,
   so the most recent court comes first. Histograms only set the order; they do
   not decide a match.
3. **Aligns and refits.** It aligns the earlier median image with the new one
   inside the earlier court. It carries the court's corners across, then refits
   them to the new anchor's painted stripes.
4. **Checks the result.** An attempt is accepted only if:
   - the images correlate at 0.8 or more, and every corner shifts by at most
     one pixel at the 1280 × 720 reference size; the alignment images themselves
     are 960 × 540
   - the carried court and the refitted court pass the court-shape checks
   - the camera is plausible and upright, and players fit when required
   - the refit moves no point of the court floor by more than about 0.23 m, half
     the spacing between the singles and doubles sidelines
   - the new paint support is at least 80% of the earlier court's
5. **Accepts the first passing court, or searches afresh.** A reused scene gets
   `chosen_key` `reuse` and names its source in `reused_from`. If every attempt
   fails, the scene gets a full search on the same prepared inputs, and then
   composition.

**Which courts enter the store.** Only a court from a fresh search with positive
paint support, whether an individual frame's court or a combined refit. A reused court
never enters, so small refit shifts cannot pile up along a chain of reuses.
Unanalysed short scenes, no-court rows and failed scenes never enter. A short
scene analysed with `--no-require-people` can enter if its fresh search returns a
court with positive paint support. The store starts empty for each video; the
newest court pushes out the oldest past eight.

`reuse.py` marks the 80% paint ratio and the 0.23 m shift limit as unvalidated
placeholders.

**Example.** Scene 1 shows the whole court, scene 2 is a close-up and scene 3
returns to the wide view. Scene 3 gets fresh lines and player samples. If
scene 1's court passes every check, scene 3 gets an adjusted copy of it.
Otherwise scene 3 gets a full search. Scene 1's own result never changes, and
scene 1 may have left the eight-court store or missed the top three.

## Video-robust pooling (`--court-mode video-robust`)

Video-robust mode lets scenes of one camera view share evidence and, when it
passes the selection and validity checks below, one court. Grouping happens
as scenes finish. Fitting and comparison run after the last scene, and final
rows print only after that pass.
[view_pool.py](../../src/court_detector/view_pool.py) owns these steps.

Scenes with `status` `court` join as members. That includes reused courts and
courts found in a single frame. A `no_court` scene can join later as a receiver,
using the group's reference court to mask its alignment. Receivers receive the
chosen court only after geometry, camera and score checks; they never donate or
influence candidate ranking. Unanalysed short scenes and failed scenes stay as
recorded.

### 1. Group scenes by camera view

As each scene finishes, it joins the first existing group that shows the same
view. Otherwise it starts a new group and becomes that group's fixed
**reference**.

- A perceptual hash of the anchor shortlists groups. It must differ from the
  reference's hash in at most 30% of its bits
- The anchor must then align with the reference's anchor, with people left out,
  at a correlation of 0.8 or more
- Camera movement is allowed. The alignment carries evidence into the
  reference's pixels and the fitted court back into each scene. The recorded
  `same_camera` flag and corner shift are diagnostics only
- Every member aligns directly with the reference, never with another member

A scene that cannot join because of an error keeps its court. Its row records the
error under `view_pool`.

### 2. Donors: who contributes evidence

Fresh accepted individual frames donate their line samples and stripe labels.
They remain eligible when the combined refit fails or an individual scores better.
A freshly searched anchor-only court can also donate, including a short scene
analysed without required players. Reused courts can receive but never donate.

Donor evidence moves into the reference's pixels and turns to match its court's
orientation. Each scene counts once toward the minimum number of independent
donors, however many accepted frames it contributes.

For each marking, the group keeps the donor with the strongest paint across all
donors. An exact tie keeps the earlier scene.

### 3. Fit one pooled court

A group needs **at least two donor scenes**. One donor may still win every
marking. Otherwise the group's `reason` is `too_few_donor_scenes` and every
member keeps its court.

The pooled court is one stripe fit to the kept samples, in the reference's
pixels, starting from the reference's court. An invalid fit records
`fit_<reason>` and leaves the accepted existing courts eligible for selection.

### 4. Score in every member

In every member's anchor, three courts get the same combined score the detector
uses to choose courts:

- `pooled`: the pooled court, carried into this member and checked there
- `scene`: the member's own finished court
- `middle`: the anchor's own court before composition, absent after endpoint-only recovery

A missing score term leaves no combined score; it is never treated as zero. A
diagnostic group mean exists only when every member has that court's score.
Candidate ranking below also records a mean over the members that could score it.

### 5. Choose one shared court

Two donors can call the same painted stripe by different marking names. The one
fit can then land between them on blank floor. Each donor scene's finished court
and its retained accepted individuals are also candidates for the whole group,
with no further refit.

A scene candidate already passed its source scene's checks. The pooled fit must
pass geometry and camera checks in the reference frame where it was fitted.
A candidate without a combined score at its source is excluded.

Each candidate is carried into the other members and scored. Rank it by its
mean combined score across members with a measurement. A failed transfer check
does not exclude the candidate for the group. Its score still counts when it
can be measured. Missing scores are omitted, and `measured_members` records how
many remain. Different counts mean candidates were compared on different sets
of frames; this can favour a candidate whose difficult transfers were unmeasurable.

The best scene court replaces the pool when its mean is higher, or when the pool
is invalid or failed its source check. An exact tie keeps the earlier scene
candidate. A tie between that candidate and the pool keeps the pool in
video-robust mode.

### 6. Apply the winner per scene

A member gets the winner only if its transferred geometry and camera pass the
checks and it has a combined score. Player presence is not required during
sharing: a matching view can show a break between points.

- A passing member takes the selected scene court (`video_pool_scene`) or pooled
  court (`video_pool`), in its own pixels. The final runner output then applies
  the exported far-baseline-first corner convention.
- A failed transfer leaves only that member's original court in place and
  records its rejection. Other members still receive the same winner.
- The source scene also changes when one of its retained individual courts wins
  over its finished scene court.
- A handled fit or scoring error sets `pooling_failed` and leaves the rows alone.

Rows change only after every court in the group is measured.

### Pooling and reuse together

Reuse still uses scene courts, never pooled ones. A reused scene runs no fresh
search, so it cannot donate. Turning reuse on therefore leaves fewer donors and
fewer groups able to pool.

## Fast-robust pooling (`--court-mode fast-robust`)

Each scene gets one fresh middle-frame search, including stripe refitting and
scoring. It skips the endpoint searches and three-frame composition. The full
temporal foot window still runs when people are required.

The middle frame's observed markings go directly into the same view grouping
and pooling machinery described above. Image alignment confirms the view;
histogram similarity alone does not establish a match. A group needs at least
three independent scene fits. Smaller groups keep their individual courts.

The pooled fit and complete scene courts are compared on the same member images.
The pool must score strictly higher than the best valid complete court to win;
a tie keeps the complete court. An invalid pool can also fall back to that
complete court. A failed transfer preserves that member's individual court. If no candidate
is eligible at its source, all individual courts remain. Final rows print after pooling.

This saves endpoint work but gives each scene fewer observations. A line hidden
in its middle frame may be visible in another scene; a unique view gets no such
help. Fast-robust therefore trades some scene-level evidence for less work.

## Output fields by mode

Final exported `corners_native_px` puts the far baseline first. The runner rolls
corners by two only when the first baseline's mean image y exceeds the second's.
It preserves the geometry and never quarter-turns or refits the court. Internal
`CourtResult`, `SceneCourts` and `scene_courts` rows retain their fitting order;
original scene corners and diagnostic candidate arrays do too.

Every analysed row carries these fields in all modes:

| Field | Meaning |
| --- | --- |
| `chosen_key` | Where the court came from: a search key, `first`, `last`, `reuse`, `composite`, `video_pool` or `video_pool_scene` |
| `reused_from` | The earlier `view_id` for a reused court, else `null` |
| `composition` | `null` unless endpoints were searched. Otherwise `court` (`middle`, `first`, `last`, `composite` or `null`), `fallback_reason`, `reference`, `used_frames` (roles), each endpoint's outcome, any `errors` and `middle_chosen_key` |
| `stage_seconds` | Detector seconds per step for this scene. Composition adds `endpoint_inputs`, `first_frame_search`, `last_frame_search` and `composition` |
| `seconds` | This scene's wall time up to its row |

Video-robust and fast-robust modes add:

- **On a row whose court changed:** `scene_corners_native_px`,
  `scene_chosen_key` and `scene_reused_from` keep the scene's own court.
  `reused_from` becomes `null`
- **On each member of a successfully scored group:** a `view_pool` record with
  `reference_view_id`, `alignment` (`null` for the reference), `court`
  (`pooled`, `scene` or `group_scene`), `pooled_corners_native_px`,
  `pooled_rejection` and `scores` (`pooled`, `scene`, `middle`, and
  `group_scene` when a scene wins). A winning scene also adds
  `group_scene_view_id`, `group_scene_corners_native_px` and
  `group_scene_rejection`. Nonfinite transferred corners are stored as `null`
- **On a `no_court` row that took its group's court:** `status` becomes `court`
  and `no_court_reason` `null`. `scene_status` and `scene_no_court_reason` keep
  its own outcome, and `scene_corners_native_px` is `null`
- **On each receiver of a successfully scored group:** a `view_pool` record with
  `reference_view_id`, `alignment`, `receiver` (`true`), `court` (the chosen
  court, or `null` when it stays without one), `rejection`, `corners_native_px`
  and `score`. A receiver is a `no_court` scene placed in the group. It never
  donates or scores a candidate
- **At the top of the result:** `view_groups`, one summary per group. It lists
  `reference_view_id`, `member_view_ids`, `donor_view_ids`, `pooled_view_ids`,
  `reason`, `chosen_court`, `chosen_view_id`, `receiver_view_ids` and
  `received_view_ids` (the receivers that took the court). As processing proceeds, it adds
  `markings` (each marking's donor), `fit`, `mean_combined_scores` and
  `scene_candidates` and `pooled_candidate`. `chosen_frame_role` identifies a
  retained individual winner when present. Each candidate records its mean,
  `measured_members`, source `rejection` and `transfer_rejections`. A group that
  stops early lacks the later fields

If a scene cannot join a group because of a handled error, its row instead has
`view_pool.error` and keeps its own court.

`chosen_court` is `pooled` when no eligible scene candidate displaced the pool.
Some or all
members may still have rejected it; `pooled_view_ids` lists the members that
actually took the pooled court.

**Timing.** `stage_seconds` and `seconds` stop before a scene joins a view group,
so they leave out all grouping and pooling time. `processing_seconds` covers
them. In either pooling mode, `processing_seconds` minus the sum of scene `seconds`
roughly shows the grouping and pooling cost.

**Progress.** Scene-by-scene progress and pooling start, per-group and end
messages go to stderr. In scene-robust mode each row also prints to stdout as its
scene finishes. In either pooling mode rows print after pooling. Stdout can also
hold other diagnostics, so read results from the saved file.

## What the modes establish, and their limits

- **Every mode:** one fixed court per scene. The runner does not follow a pan or
  zoom within a scene. The shot check can trim player samples but never adds a
  scene boundary. Missed cuts, camera movement and poor visibility can make a
  scene's court wrong for part of it.
- **Scene-robust** retains the result of the per-scene pass, including a
  validated reuse when that option is on. Scenes of the same view can disagree.
- **Reuse** makes a returning, barely moved view agree with an earlier
  searched court, and skips that scene's search. It sees only a few recent
  references, and its thresholds are placeholders.
- **Video-robust** lets returning views share the best-painted evidence across
  the video, with each member's own checks as a safeguard. A hash and an image
  alignment cannot prove two scenes show the same court. Extra scoring grows
  with donors times members.
- **Scores are internal.** The combined score ranks courts; it is not accuracy
  against hand-labelled courts.

The annotation pipeline applies each scene's court across its whole frame range.
It then applies its own player vote with the full pose arrays, separate from the
detector's 31-frame window. It requires exactly two people inside the court's
margin in at least half the scene's frames. See
[court_evidence.py](../../src/annotator/court_evidence.py).

The [design page](design.md) records why composition
and pooling work this way. The [evaluation page](../../experiments/court_detector/comparisons/development_evaluation.md)
links the measured runs and their remaining limitations.

## Where the behaviour lives

- [scene_sources.py](../../src/court_detector/scene_sources.py): PySceneDetect cuts, histograms and saved ranges
- [feet.py](../../src/court_detector/feet.py): player window, shot check and feet
- [run_video.py](../../src/court_detector/run_video.py): anchors, median images and the reuse store
- [composition.py](../../src/court_detector/composition.py): composing one court within a scene
- [reuse.py](../../src/court_detector/reuse.py): reuse alignment, refit and checks
- [view_pool.py](../../src/court_detector/view_pool.py): video-robust grouping, donors, pooling and fallback
- [court_views.py](../../src/annotator/court_views.py): perceptual hashes and image alignment, shared by all three
