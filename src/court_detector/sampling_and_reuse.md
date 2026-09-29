# Video sampling, reuse and output modes

The video runner gives **at most one court per scene**. A court here means four
corner points in the image, which together fix where the whole court floor sits.
This page is the contract for how each scene is sampled and how scenes can share
evidence. The [README](README.md) has the commands, setup and top-level output
fields. The [design page](../../docs/court_detector/design.md) explains why, and
the [evaluation page](../../docs/court_detector/evaluation.md) holds the evidence.

These options decide how scenes relate to each other:

| Behaviour | Option | When it acts | What a scene can end up with |
| --- | --- | --- | --- |
| Scene-robust | `--court-mode scene-robust` | Per scene | Its own searched or composed court |
| Chronological reuse | `--reuse-courts` | Per scene, in time order, before the search | An earlier scene's court, refitted and checked in this scene |
| Video-robust (default) | `--court-mode video-robust` | Once, after every scene has finished | One court shared by scenes of the same camera view, or its own court |
| Fast-robust | `--court-mode fast-robust` | Fits each middle frame, then pools after every scene has finished | A pooled court, the best complete middle-frame court, or its own middle-frame court |

Reuse and video-robust mode are independent and can run together. Both are
described in full below.
Fast-robust requires independent fresh fits and rejects `--reuse-courts`.

## Scenes and frame ranges

A scene runs from one camera cut to the next.

| How you run it | Where the cuts come from |
| --- | --- |
| `--pyscenedetect` | PySceneDetect's ContentDetector, threshold 27, with the annotation pipeline's minimum scene length (about half a second) |
| `--scenes FILE` | Your saved ranges |
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
| Find lines, search and fit the court | The anchor |
| Check players' positions | 31 frames at about 10 per second, three seconds around the anchor |
| Compose a court (fresh search only) | The anchor plus the first and last frames of that player window |
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
are not 31 separate court detections.

**Short scenes.** With people required, a scene that cannot hold the window gets
`scene_too_short_for_feet`. That means unanalysed, not "no court". With
`--no-require-people`, a short scene is still analysed. It uses only the anchor's
people, if any, and is never composed. For reuse, its median image uses the
scene's first frame, anchor and last frame instead.

Saved poses and live sparse RTMLib give the same sample frames. A full pose
prerun extracts every frame for later pipeline stages; it does not add samples.

## Composing a court within a scene

Players hide different markings in different frames. Composition fits one court
from the frames that show each marking best.

It runs only when the scene holds the player window and a **fresh search** of
the anchor found a court. It is skipped for a reused court, a short scene and an
anchor with no court.

1. **Search the endpoints.** The window's first and last frames are searched like
   the anchor, with their own lines and person boxes but the anchor's feet. An
   endpoint whose search fails or finds nothing is left out, and the failure is
   logged.
2. **Pick a reference.** The frame with the highest combined score becomes the
   reference. Exact ties go middle, then first, then last.
3. **Align.** Each other frame is aligned to the reference inside its own court,
   with person boxes left out. The alignment must reach a correlation of 0.8.
   Camera movement is allowed. At least two frames, including the anchor, must
   align.
4. **Choose donors.** Each court marking takes its stripe samples from the frame
   with the strongest paint on that marking.
5. **Fit and check.** One court is fitted to those samples, then carried into the
   anchor. It must pass the anchor's geometry and camera checks, the player check
   when people are required, and the upright-camera check.

A passing composite replaces the anchor's court, with `chosen_key` set to
`composite`. Otherwise the anchor's own court stands. The row's `composition`
record says which happened. Its `fallback_reason` is one of
`too_few_accepted_frames`, `score_evidence_missing`, `too_few_aligned_frames`,
`middle_not_aligned`, `composition_failed`, `fit_<reason>` or `middle_<reason>`.

The endpoints are always the scheduled window ends, even when the shot check
dropped one. The alignment and the anchor checks must reject a bad contribution.

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
paint support, whether the anchor's own court or a composite. A reused court
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
[view_pool.py](view_pool.py) owns these steps.

Only scenes with `status` `court` take part. That includes reused courts and
anchor-only courts. No-court rows, unanalysed short scenes and failed scenes are
left exactly as they are.

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

A scene donates only if its fresh search ended in an **accepted composite**. In
code, this is a scene whose `SceneCourts.used_frames` is non-empty. It offers the
frames that won its markings, with their line samples and stripe labels. These
are carried into the reference's pixels and turned to match the reference's
court if needed.

These scenes never donate, but can still receive a shared court:

- a reused court
- an anchor-only court, where composition fell back
- a scene that was not composed, such as a short scene

For each marking, the group keeps the donor with the strongest paint across all
donors. An exact tie keeps the earlier scene.

### 3. Fit one pooled court

A group needs **at least two donor scenes**. One donor may still win every
marking. Otherwise the group's `reason` is `too_few_donor_scenes` and every
member keeps its court.

The pooled court is one stripe fit to the kept samples, in the reference's
pixels, starting from the reference's court. If the fit is invalid, `reason` is
`fit_<reason>` and every member keeps its court.

### 4. Score in every member

In every member's anchor, three courts get the same combined score the detector
uses to choose courts:

- `pooled`: the pooled court, carried into this member and checked there
- `scene`: the member's own finished court
- `middle`: the anchor's own court before composition

A missing score term leaves no combined score; it is never treated as zero. A
group mean exists only when every member has that court's score.

### 5. Whole-scene fallback

Two donors can call the same painted stripe by different marking names. The one
fit can then land between them on blank floor. So each donor's finished court is
also a candidate for the whole group, with no refit.

- The candidate is carried into every member and checked and scored there
- It is out if it fails any member's check or lacks any member's score
- Among the remaining candidates, the highest mean wins. An exact tie keeps the
  earlier scene
- The winner replaces the pool only if its mean beats the pool's mean, or the
  pool has no mean. **An exact tie keeps the pool**

The pool's mean counts every member, including members whose pooled court fails
its check.

### 6. Apply

- **A scene candidate wins:** every member gets that court in its own pixels and
  corner order. `chosen_key` becomes `video_pool_scene`. The winning scene's row
  is unchanged
- **The pool wins:** each member whose pooled court passed its own checks gets
  it, with `chosen_key` `video_pool`. A member that failed keeps its own court
- **The fit or scoring raised an error:** `reason` is `pooling_failed` and no row
  changes

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
complete court. If no shared court passes the relevant checks, individual
scene courts remain. Final rows print after pooling.

This saves endpoint work but gives each scene fewer observations. A line hidden
in its middle frame may be visible in another scene; a unique view gets no such
help. Fast-robust therefore trades some scene-level evidence for less work.

## Output fields by mode

Every analysed row carries these fields in all modes:

| Field | Meaning |
| --- | --- |
| `chosen_key` | Where the court came from: a search key, `reuse`, `composite`, `video_pool` or `video_pool_scene` |
| `reused_from` | The earlier `view_id` for a reused court, else `null` |
| `composition` | `null` unless endpoints were searched. Otherwise `court` (`middle` or `composite`), `fallback_reason`, `reference`, `used_frames` (roles), each endpoint's outcome, any `errors` and `middle_chosen_key` |
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
  `group_scene_view_id` and `group_scene_corners_native_px`
- **At the top of the result:** `view_groups`, one summary per group. It lists
  `reference_view_id`, `member_view_ids`, `donor_view_ids`, `pooled_view_ids`,
  `reason`, `chosen_court` and `chosen_view_id`. As processing proceeds, it adds
  `markings` (each marking's donor), `fit`, `mean_combined_scores` and
  `scene_candidates` (each candidate's mean and first rejection). A group that
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
[court_evidence.py](../annotator/court_evidence.py).

The [design page](../../docs/court_detector/design.md) records why composition
and pooling work this way. The [evaluation page](../../docs/court_detector/evaluation.md)
holds the scene checks, the cached pooling checks and the pending full GPU run.

## Where the behaviour lives

- [scene_sources.py](scene_sources.py): PySceneDetect cuts, histograms and saved ranges
- [feet.py](feet.py): player window, shot check and feet
- [run_video.py](run_video.py): anchors, median images and the reuse store
- [composition.py](composition.py): composing one court within a scene
- [reuse.py](reuse.py): reuse alignment, refit and checks
- [view_pool.py](view_pool.py): video-robust grouping, donors, pooling and fallback
- [court_views.py](../annotator/court_views.py): perceptual hashes and image alignment, shared by all three
