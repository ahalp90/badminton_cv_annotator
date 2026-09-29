# Run the court detector

`CourtDetector` finds a badminton court in a prepared image. It runs the
search and scoring steps in memory and returns four court corners, or a reason that no
court was found. This page owns its inputs, behaviour, settings and commands.

Read [pickup.md](../../scratch/court_det_fix/pickup.md) for current work,
[the decisions](../../scratch/court_det_fix/DETECTOR_DECISIONS.md#d19) for measured results, and
[PERFORMANCE.md](../../scratch/court_det_fix/court_detector/PERFORMANCE.md) for the speed-up design.

## Scenes, sampling and returning views

For video, the runner returns **at most one court per scene**, with a reason
when none is returned. The projection is fixed within that scene; it does not
follow camera movement frame by frame.

- **Scene boundaries:** use `--pyscenedetect` to find cuts, or `--scenes` for
  saved boundaries. Without either, the whole video is treated as one scene.
  The dataset builder requests PySceneDetect unless supplied with boundaries.
- **Samples:** court lines and fitting use the middle frame. Player checks use
  up to 31 samples across a three-second window around it. With players required
  (the default), scenes unable to hold that window are marked unanalysed.
- **Composition:** when a fresh search finds a court, the runner also searches
  the first and last frames of that window. A court fitted to the best markings
  from all three frames can replace the middle frame's court.
- **Returning views:** `--reuse-courts` checks earlier courts against fresh
  evidence from each new scene, including non-consecutive scenes. It keeps eight
  recent fully searched courts and tries up to three. Reuse can adjust the
  projection; it does not assign one identical projection to a video-wide group.
- **Video-robust courts:** `--court-mode video-robust` pools composites across
  scenes of the same camera view after the last scene, then refits one court per
  view. See [Video-robust courts](#video-robust-courts). The default,
  `scene-robust`, keeps each scene's own court.
- **Defaults:** standalone reuse is off. The supplied `shuttleset_fixed.toml`
  and `trial.toml` builder configurations enable it.

PySceneDetect is optional. A short clip can run as one scene; use
`--no-require-people` if it cannot hold the player window. A single image file
can use the [image runner](#run-from-an-image), which runs DeepLSD itself.

See [Sampling and scene reuse](sampling_and_reuse.md) for the sample schedule,
reuse checks, an example and the limits of this approach.

## Inputs and result

`CourtDetector.detect(view, people, frames)` takes three inputs from `inputs.py`:

- `view` (`ViewInputs`): an image, its detected line fragments as float32
  native pixels, person boxes, represented video frame, scene boundaries, and
  the source frames for the image and boxes. The detector uses that source
  information to avoid hiding people in the wrong image
- `people` (`PeopleSource`): boxes and 17 COCO-layout body joints for people
  in nearby video frames
- `frames` (`FrameReader`): access to those nearby video frames

A caller supplies these inputs directly, or uses the video runner below to
build them from video frames, line detections and people detections.

### Frame ranges

Frame indices are zero-based. Scene ranges are half-open, like Python's
`range`: `[start_frame, end_frame)` holds `start_frame` up to `end_frame - 1`.
This applies to `SceneInfo`, `ViewInputs.scene_frames`,
`feet.window_frames(anchor, fps, start_frame, end_frame)`, `--scenes` files
and the runner's output rows.

- A whole video is `[0, frame_count)`
- A cut frame is the `end_frame` of one scene and the `start_frame` of the next,
  so it belongs only to the following scene
- The analysed frame is `SceneInfo.middle_frame`, `(start_frame + end_frame) // 2`.
  An even-length scene uses the upper of its two middle frames
- Explicit frame lists, such as foot samples and `FrameReader.read` requests,
  are plain indices

Older versions used inclusive scene ends. Their runner rows store
`first_frame` and `last_frame`. An old scene `[first, last]` is
`[first, last + 1)` now, and it samples the same frames.

Runner results without a `schema` field analysed the lower middle frame,
`(start_frame + end_frame - 1) // 2`. The two choices differ only for
even-length scenes, by one frame.

The returned `CourtResult` contains:

| Field | Meaning |
| --- | --- |
| `corners_native_px` | Four float64 corners in the original image's pixels, or `None`. Corners can lie outside the image |
| `no_court_reason` | Why no court was returned. `no_gated_court` means none passed the court checks; `rank_deficient` means the final fit lacked enough independent information. `refit_camera_implausible` and `refit_players_not_on_court` mean the final correction failed those checks |
| `chosen_key` | The saved identifier of the chosen court; `reuse` for a reused court, `composite` for a court composed across frames |
| `stage_seconds` | Time per step when timing is enabled. Composing adds `endpoint_inputs`, `first_frame_search`, `last_frame_search` and `composition` |
| `paint_score` | Paint support after the final stripe fit, for checking later reuse. A composite's is measured in the middle frame |
| `reused_from` | Earlier view used for checked reuse, or `None` for a full search |
| `composition` | `None` unless the endpoint frames were searched. Then `court` is `middle` or `composite`, with `fallback_reason`, the `reference` frame, `used_frames`, each endpoint's outcome, any `errors` and `middle_chosen_key` |
| `scene` | With a court, the middle frame's context and image, its own court and any composite's frames, for [video-robust courts](#video-robust-courts). It is not saved |

## How it finds the court

Most steps shrink the image to a longest side of at most 960 pixels.
A 16:9 image becomes 960×540.

1. **Find players' feet.** Sample 31 frames at 10 per second. Shift the window
   to stay within the scene; a scene shorter than the window raises
   `ValueError`. Keep the continuous run of frames from the same shot around
   the target. Use the bottom centre of each box, excluding people whose
   poses indicate they are seated
2. **Prepare the lines and image.** Shrink line fragments and group them by
   direction. Hide person boxes from paint measurements only when the boxes
   belong to this image
3. **Search for courts.** Match pairs of line directions to court markings.
   Run once with all fragments (`all_lines`), and once with only paint-like
   fragments (`painted_lines`). Older saved records and notes call these G0 and
   G1. The second search uses pale lines that are brighter than their
   surroundings and not strongly coloured. Players' feet must fit the court.
   Reject sideways and upside-down camera geometry. Keep up to 256 distinct
   courts per pair and 256 per complete search
4. **Build courts from crossing lines.** Use rectangles and extra starting
   points from the three longest lengthwise lines. Require at least four
   lengthwise and three cross-court lines, and a plausible camera
5. **Score and refit the possible courts.** Merge duplicates, check shapes,
   measure paint support and try refitting each court to its stripes. Rank
   the courts and successful refits together. A court can win only if its
   geometry and camera are valid and the players fit. At least one player
   must fit in every sampled frame, and one in each half in at least half
   the frames. The code calls this stage scoring; older records and notes
   call it W5
6. **Choose a court.** Combine 90% paint score and 10% line-support score.
   Add 0.02 for each supported net post, up to 0.04. Missing net support is
   neutral. If no court passed the checks, return no court
7. **Adjust the stripe fit.** Check whether each fitted line lies on the
   centre or an edge of its painted stripe. Change its label only when the
   image evidence is clear, then refit the corners. The result must pass the
   camera check and, when required, the player check again. A failed fit or
   check returns no court
8. **Compose across frames (video only).** When the runner supplies a scene's
   endpoint frames and step 7 returns a court, repeat steps 2 to 7 on the first
   and last scheduled frames of the player window. They use the middle frame's
   feet. Rank the frames that found a court by the step 6 blend; the best is the
   reference. Align the others to it with people hidden, and drop weak
   alignments. For each marking, take the stripes from the frame with the
   strongest paint on it. Fit one court to them in the reference frame, then carry
   it into the middle frame. There it must pass the geometry, camera and required
   player checks, plus the upright-camera check when enabled. A passing composite
   replaces the court, and its paint support is measured in the middle
   frame. Otherwise, or when the middle frame does not align, the middle frame's
   own court stands. A failed endpoint search is logged and left out

The court checks allow feet up to 15% beyond the court. The shot check keeps
frames whose small grey thumbnail differs from the target by no more than
eight grey levels on average. These are existing rules, not extra input
preparation supplied by the caller.

## Run it

Run from the repository root with `PYTHONPATH=.:src`. Follow
[run_views.py](../../scratch/court_det_fix/court_detector/run_views.py) for single-threaded numerical
libraries and OpenCV decoding. It sets these before loading NumPy.

```python
from court_detector.detect import CourtDetector, Switches

detector = CourtDetector(Switches())
result = detector.detect(view, people, frames)
```

People are required by default. For a plausible fallback without boxes or
keypoints, use `Switches(require_people=False)` and pass `people=None` with an
empty `(0, 4)` box array in the view. The detector then uses line-template
proposals, court geometry, camera checks and the usual stripe fit. Player
measurements remain unavailable rather than being treated as passing checks.
This fallback has not been tuned for the same precision as detection with people.

Required mode skips the candidate search when the sampled people cannot meet
the existing occupancy checks. Runs with full artefacts still build candidate
records so saved-run comparisons can inspect them.

When people are supplied in optional mode, their boxes still mask occlusions
and support the existing proposal search. Missing or off-court people do not
veto the final choice. Short scenes use their anchor's people when available.

The saved-view runner stays with the research records in
`scratch/court_det_fix/court_detector` and imports the detector from this package.
It takes IDs from the
[28-view list](../../scratch/court_det_fix/court_detector_optimisation_handover/claude_evidence/fresh_feet/views.json).
Its image, line and box inputs come from saved research records. `PEOPLE_DIR`
holds one record per view with people, poses and the source video's path.

```bash
PYTHONPATH=.:src python -m scratch.court_det_fix.court_detector.run_views \
  --people PEOPLE_DIR \
  --output OUT \
  VIEW [VIEW ...]
```

Add `--baseline ARM_DIR --feet FEET_FILE --artefacts` to compare against a saved
research run. The baseline arm uses the older names: G0 and G1 for the two
searches and W5 in some field names. The runner reads them in the current names
and leaves the saved files unchanged. To reproduce the original 25 September
comparison, also use `--any-camera-roll --geometry-weight 0`. The newer defaults
intentionally change results. The [original check](../../scratch/court_det_fix/court_detector/check_20260925/README.md)
states exactly what was compared; the runner docstring supplies the full command
details.

The people records and videos used on Carmack are not in git. See
[the data map](../../scratch/court_det_fix/FP_INDEX.md#data-that-is-not-in-git) before a remote rerun or
new checkout.

Use `--workers 8` to search direction pairs and score candidates in eight processes.
These stages run in sequence. Results are collected in their original order before choosing courts. Keep the
total CPU allocation at eight cores on Carmack: run one eight-worker view at a
time, or several serial views whose combined allocation stays within that limit.
The runner reports peak memory for the parent and largest worker separately;
those figures do not measure the combined peak of every process.

Both runners keep one set of worker processes for the whole run, so each worker
starts and imports the detector once. The workers close when the run ends,
including when it fails. A script that calls `CourtDetector.detect` itself
shares workers the same way inside `with CourtDetector(switches) as detector:`.
Without the `with` block, each search and scoring step starts its own workers.
A worker holds the last view it scored until it scores another view or closes.
Parallel candidate scoring writes one temporary input pickle per view. Each
worker loads it once into memory; results return through the process pool.
Workers never write to the file, which is removed after the tasks finish.
The saved-view batch runner replaces a crashed pool before trying the next view.

Direction-pair support scoring uses a serial Numba kernel. Numba is a project
runtime dependency. The first call compiles and caches the kernel; later
processes load that cache. It adds no threads beyond the selected worker count.
Bulk scoring arrays and responses use float32. Fitting, sensitive geometry and
the compiled scorer's small projection/clipping calculations use float64.

Both runners accept `--template-device cuda` as an explicit option. It scores the
line-template hypotheses on a GPU with CuPy. Rectangle setup, admission and the
court checks stay on the CPU, and other stages keep their own settings.
CuPy is not a project dependency, so install it where a GPU is available. The
detector stops at setup when CuPy or a CUDA GPU is missing; it never falls back
to the CPU. GPU rounding can change the last bits of results, such as each line
template's `camera_error_before_candidate_gates`. It can also occasionally move a
line sample to a neighbouring pixel. Saved-run comparisons report any such
differences.

## Run from video

The video caller can run DeepLSD for lines and read frame-aligned RTMLib pose
extracts. With the default `--require-people`, omit `--people` to run the project's
RTMLib extractor on the requested frames. Both live models load once.
Use `--no-require-people` without `--people` to skip pose inference and use the
line-only fallback. This mode also analyses scenes shorter than the usual
three-second people window. Supplying `--people` still uses those saved inputs,
but leaves the player checks optional. The saved-view runner supports the same
gate toggle while retaining its required fixture directory.

Live pose extraction uses `shared.rtmlib_pose`, the same extractor as BST-X
preparation and the dataset builder. RTMLib is imported only when the live-pose
extractor is built. `RtmlibPeople` serves one video because it caches by frame
number. Give each video's `RtmlibPeople` the same `RtmlibPoseExtractor` so a
batch loads the models once.

DeepLSD uses gradient validation by default. This matches the saved amateur
example's line extract; the less selective hard variant remains available through
`DeepLSDLines(..., grad_nfa=False)`.

```bash
PYTHONPATH=.:src python -m court_detector.run_video \
  --video VIDEO.mp4 \
  --deeplsd-source DEEPLSD_CHECKOUT \
  --deeplsd-weights DEEPLSD_WEIGHTS.tar \
  --people POSE_DIR \
  --pyscenedetect --workers 8 \
  --output courts.json.gz
```

`POSE_DIR` contains `pose_bboxes.npy.xz`, `pose_kps.npy.xz` and
`pose_ndet.npy.xz`. They must use source-video frame indices and native image
pixels. The reader drops padded detection slots using `ndet`.

For saved-line trials, replace the DeepLSD arguments with `--saved-lines FILE`.
That file is a gzipped JSON object mapping source frame numbers to `(N, 4)`
native-pixel line arrays, read as float32. Missing frame entries raise an error.
It needs each scene's middle frame. A scene that holds the player window also
needs the window's first and last frames, which are read only when composing.

PySceneDetect is optional. Its adapter uses the existing ContentDetector cut
settings and can supply a normalised luminance histogram for each scene.
Histograms can prioritise camera comparisons; they do not establish that courts
match. The `SceneSource` and `LineSource` protocols allow other input providers.
The prepared-image detector needs neither PySceneDetect nor DeepLSD installed.

Use `--scenes FILE` for a gzipped list of `[start_frame, end_frame]` pairs, with
exclusive ends, instead of detecting cuts. The scenes must cover
`[0, frame_count)` in order without gaps or overlaps. An old file of inclusive
ranges leaves gaps, so the runner rejects it. With neither scene option, the
caller analyses one middle frame from the whole video. It analyses one middle
frame per supplied scene and keeps foot samples inside that scene. Each output
row repeats its scene's `start_frame` and `end_frame`, with the analysed frame
as `frame_index`. Source codecs need a seek-versus-sequential frame check before
benchmarking a new dataset.

Use `--pose-prerun DIR` to extract poses for every frame before court detection.
The pose stage uses the dataset builder's settings: eight shards, ten people per
frame and the selected `--device`. `--pose-python PYTHON` selects its interpreter;
the default is the current one. Poses are saved under `DIR/VIDEO_ID`, which must
not already exist. Reuse completed poses later with `--people`.

In a batch, manifest entries with `people` reuse those files; other entries get
the full pose pass. Sparse live extraction remains the cheaper default for
court-only work. Full extraction is useful when later stages also need poses.
The full pass keeps the ten highest-scoring person detections per frame; sparse
live extraction keeps all detections above the person threshold. Courts can
differ when the ten-person limit drops a player in a crowded frame.

### Scene statuses

Each output row has one `status`:

| Status | Meaning |
| --- | --- |
| `court` | `corners_native_px` holds the court |
| `no_court` | The detector found no court; `no_court_reason` says why |
| `scene_too_short_for_feet` | People are required and the scene cannot hold the foot window. The scene is unanalysed; this is not evidence that no court is present |
| `detection_failed` | The court search or fit raised an error on this scene. `error` and `traceback` record it, and the next scene runs |

A scene fails alone for a `CourtFitError` from geometry search, scoring or refitting
after input validation. Other errors stop the video, such as a broken worker
pool, a GPU error or a people source that returns the wrong frames. A fit error
in an endpoint frame or the composition keeps the middle frame's court and is
recorded under `composition.errors`. Only `court` rows from a full search or a
composite can become reuse templates.

### Video result

The result is a gzipped JSON object:

| Field | Meaning |
| --- | --- |
| `schema` | `court-detector-video/1` |
| `video_id` | The start of each row's `view_id`: the video's file stem, or its manifest `id` |
| `video`, `fps`, `frame_count`, `native_size` | The source video's file name and properties |
| `saved_people`, `saved_lines` | Whether people and lines came from saved extracts, so those timings exclude live inference |
| `require_people`, `reuse_courts`, `court_mode`, `template_device` | The detector settings used |
| `view_groups` | Video-robust mode only: one summary per camera-view group; see [Video-robust courts](#video-robust-courts) |
| `tools_seconds` | Loading lines, live pose and the detector, once per run. Every result in a batch repeats the same figure |
| `setup_seconds`, `scene_seconds`, `processing_seconds` | This video's input setup, scene detection and per-scene work |
| `pose_prerun_seconds` | Full pose extraction, when requested; included in `setup_seconds` and `total_seconds` |
| `total_seconds` | This video's time, excluding `tools_seconds` |
| `scenes` | One row per scene, in order |

With live people, `processing_seconds` includes sparse RTMLib inference. It is
not a detector-only timing. Saved inputs omit their earlier extraction cost;
`pose_prerun_seconds` records that cost when this invocation produced the poses.

### Video-robust courts

`--court-mode video-robust` can give scenes of one camera view a shared court.
It waits for every scene's scene-robust court, so rows print only after the
last scene. [view_pool.py](view_pool.py) owns the steps:

1. **Groups.** Each scene with a court joins the first group whose fixed
   reference scene shows the same camera view. A perceptual hash only shortlists
   groups. The people-free alignment used by composition must then pass the reuse
   correlation and the one-reference-pixel movement limit.
2. **Donors.** Only a scene whose fresh search ended in an accepted composite
   donates. It offers the observed line samples that won its markings. Reused,
   middle-frame and fallback courts can receive a pooled court but never donate.
   Each marking keeps its strongest donor across the group.
3. **One fit.** A group with at least two independently computed composites gets
   one stripe fit to those samples, in the reference scene's pixels. One scene
   may win every marking; the pool does not force a mixture.
4. **Output.** A valid pooled court replaces each member's court that passes
   that member's own checks, including its players' feet. A member that fails
   keeps its scene court, as does every member when the fit fails.
5. **Scores.** In every member's middle frame, the pooled court, the scene's
   court and the middle frame's court before composition get the net choice's
   combined score. These scores are reported for comparison; they do not choose
   the output. A missing score term stays missing, never zero.

A changed row keeps its earlier court as `scene_corners_native_px`,
`scene_chosen_key` and `scene_reused_from`. A pooled court's `chosen_key` is
`video_pool`. Each member of a group with a valid fit gets a `view_pool` record: the
group's `reference_view_id`, its `alignment`, the `court` the row now holds, the
pooled corners and any `pooled_rejection`, and all three courts' `scores`.
`view_groups` lists each group's members, donors, `pooled_view_ids`, per-marking
donors, fit and mean scores. Its `reason` says why a group kept its scene
courts. A mean is `None` unless every member has that court's score.

`--reuse-courts` works as before and uses scene courts, not pooled ones. A
scene that reuses a court runs no fresh search, so reuse leaves fewer donors.

### Run a batch

A batch loads DeepLSD, RTMLib and the detector once and shares one worker pool
across its videos. Each video still gets its own frame reader, live-pose cache
and known courts, so no cached frame or court crosses between videos. A Python
caller does the same with one `load_court_tools()` call, then one
`detect_video()` call per video inside a single `with tools.detector:` block.

```bash
PYTHONPATH=.:src python -m court_detector.run_video \
  --manifest videos.json.gz \
  --deeplsd-source DEEPLSD_CHECKOUT \
  --deeplsd-weights DEEPLSD_WEIGHTS.tar \
  --pyscenedetect --workers 8 \
  --output-dir OUT_DIR
```

The manifest is a gzipped JSON list with one object per video:

```json
[
  {"id": "match_01", "video": "videos/match_01.mp4", "people": "poses/match_01"},
  {"id": "match_02", "video": "videos/match_02.mp4", "scenes": "scenes/match_02.json.gz"}
]
```

- `id` and `video` are required. Each `id` must be unique and a plain file name
- `people` is that video's `POSE_DIR`. Without it, the batch runs live RTMLib,
  or uses no people with `--no-require-people`
- `scenes` is that video's `--scenes` file. It cannot combine with
  `--pyscenedetect`. Without either, the whole video is one scene
- Relative paths resolve from the working directory. Other keys raise an error

`--people`, `--scenes` and `--saved-lines` describe one video, so a batch
rejects them. Batches therefore use DeepLSD. `OUT_DIR` must not exist yet.
The batch writes:

- `OUT_DIR/videos/<id>.json.gz`: the video result above, for each complete video
- `OUT_DIR/summary.json.gz`: `"schema": "court-detector-batch/1"`, rewritten
  after every video. Each `videos` entry has `status` `complete`, `failed` or
  `not_run`. A failed entry keeps its `error` and `traceback`; a complete one
  counts its scene statuses

Unreadable, malformed or truncated video inputs fail that video alone. These come
from its own files, such as an unreadable video or scenes that miss frames. A
broken worker pool is replaced before the next video. Any other failure stops
the batch, because the shared models, GPU or workers may be in an unknown
state. `stopped_after` then names that video, `finished` is false and the later
videos stay `not_run`. The exit status is 1 unless every video is complete.

### Reuse from prepared images

The video runner's reuse behaviour is described in
[Sampling and scene reuse](sampling_and_reuse.md). Rejected reuse attempts fall
through to a full search using the same prepared inputs.

Prepared-image callers can pass `known_courts=` to `detect()`. Build each entry
with `reuse.make_known_court()` from a fully searched result and its input frame.
These callers use the raw image unless they supply `ViewInputs.alignment_image`
and the matching `alignment_image=` when building a reference. Prepared alignment
images are greyscale uint8 at `court_views.VIEW_RESOLUTION`.

## Run from an image

The image runner finds the court in one JPG, PNG or other file OpenCV can read.
It runs DeepLSD on the image and reports corners in the image's own pixels.
It needs no video, pose file or scene boundaries, and never loads PySceneDetect.
An unreadable image fails before any model loads.

```bash
PYTHONPATH=.:src python -m court_detector.run_image \
  --image HALL.jpg \
  --deeplsd-source DEEPLSD_CHECKOUT \
  --deeplsd-weights DEEPLSD_WEIGHTS.tar \
  --output hall_court.json.gz
```

By default RTMLib is not loaded, and the court comes from lines and geometry.
Add `--with-people` to run RTMLib once on the image. Those people then mask
occlusions and support the proposal search, as with optional people in video.
One image cannot supply the three-second player window. The video mode's player
requirement therefore does not apply: missing or off-court people do not veto a court.

In Python, load the models once and reuse them for several images:

```python
from court_detector.detect import Switches
from court_detector.run_image import detect_image, load_image_tools, read_image

tools = load_image_tools(Switches(require_people=False, workers=8, timing=True),
                         DEEPLSD_CHECKOUT, DEEPLSD_WEIGHTS, with_people=False)
with tools.detector:
    results = [detect_image(read_image(path), tools, image_id=path.stem, source=path.name)
               for path in image_paths]
```

The result is a gzipped JSON object. `no_court` is a normal result; a search or
fit error stops the run instead.

| Field | Meaning |
| --- | --- |
| `schema` | `court-detector-image/1` |
| `image_id`, `image` | The image's ID (the file stem on the command line) and file name; `image` is `null` when a Python caller gives none |
| `native_size` | `[width, height]` of the decoded image |
| `status` | `court` or `no_court` |
| `corners_native_px` | Four corners in the image's pixels, or `null`. Corners can lie outside the image |
| `no_court_reason` | Why no court was returned, or `null` |
| `with_people` | Whether RTMLib ran on the image |
| `tools_seconds` | Loading DeepLSD, the detector and, when requested, RTMLib |
| `line_seconds`, `people_seconds`, `detection_seconds` | This image's DeepLSD, RTMLib (`null` without people) and detector time |
| `stage_seconds` | The detector's seconds per step, or `null` when `timing` is off |

## Settings

| `Switches` field | Default | Behaviour |
| --- | --- | --- |
| `self_checks` | On | Check intermediate results, including replaying the scoring-stage fit; this is separate from comparison with a saved run |
| `enforce_scene_consistency` | On | Restrict the foot samples to the target's shot |
| `upright_camera` | On | Reject courts requiring a sideways or upside-down camera; the runner's `--any-camera-roll` disables it |
| `geometry_weight` | 0.1 | Share of line support in the final score; the rest is paint support. The runner accepts `--geometry-weight` |
| `workers` | 1 | Processes for direction pairs and candidate scoring; the runner accepts `--workers` |
| `full_score_limit` | `None` | Optional trial: score each pair's courts with 16 samples, then fully score only the best given number with 64 samples. Both runners accept `--full-score-limit`; omitting it keeps exhaustive scoring |
| `template_device` | `"cpu"` | Line-template scoring device; `"cuda"` uses CuPy on a GPU and fails at setup without one. Both runners accept `--template-device` |
| `timing` | Off | Report seconds per step |
| `artefacts_dir` | None | Optionally write intermediate results |

The camera filter skips a direction pair when its horizon tilts more than
45 degrees. It also skips individual courts that imply an upside-down camera.
A horizon more than ten image diagonals away passes both tests.

Seeded templates, the paint-line search, the bounded net reward and stripe
refitting have no individual switches. [STRIPPED.md](../../scratch/court_det_fix/court_detector/STRIPPED.md) records the
research features left out and where they could be restored.

## Code map

| File | Responsibility |
| --- | --- |
| [detect.py](detect.py) | `CourtDetector`, `Switches`, `CourtResult`, and the order of the steps |
| [inputs.py](inputs.py) | Input types and image/box source information |
| [feet.py](feet.py) | Player feet and shot checks |
| [search.py](search.py) | Search settings, paint-like fragments and extra starting points |
| [net_choice.py](net_choice.py) | Final choice and net-post reward |
| [stripe_refit.py](stripe_refit.py) | Adjust stripe labels and refit |
| [composition.py](composition.py) | Compose one court from a scene's middle and endpoint frames |
| [view_pool.py](view_pool.py) | Pool composites across a video's returning camera views and refit once per view |
| [run_video.py](run_video.py), [video_inputs.py](video_inputs.py) | Run one video or a batch, and build each scene's inputs |
| [run_image.py](run_image.py) | Run one image file with live DeepLSD and optional RTMLib |
| [line_sources.py](line_sources.py), [scene_sources.py](scene_sources.py) | DeepLSD or saved lines; PySceneDetect or saved scenes |

The search, scoring and geometry code now lives in this package. Imports use
normal package paths and leave `sys.path` unchanged. The package imports nothing
from `experiments` or `scratch`. The court's size and painted lines come from
`shared.court_model`.

| Files | Responsibility |
| --- | --- |
| [generation.py](generation.py), [candidate_pool.py](candidate_pool.py), [proposals.py](proposals.py) | Search direction pairs and keep candidate courts |
| [directions.py](directions.py), [line_matching.py](line_matching.py), [prepare_lines.py](prepare_lines.py) | Estimate directions, match markings and prepare fragments |
| [line_templates.py](line_templates.py), [template_arrays.py](template_arrays.py) | Build candidate courts from crossing lines; score them with NumPy or CuPy |
| [scoring.py](scoring.py), [measurements.py](measurements.py), [sampling.py](sampling.py) | Measure and rank courts, refit candidates and share image samples |
| [geometry.py](geometry.py), [candidate_geometry.py](candidate_geometry.py), [camera.py](camera.py) | Court coordinates, transforms and camera checks |
| [court_checks.py](court_checks.py), [players.py](players.py) | Check court shape and whether players fit |
| [line_observations.py](line_observations.py), [stripe_measurements.py](stripe_measurements.py), [stripe_fitting.py](stripe_fitting.py) | Group fragments, measure stripes and fit corners |
| [paint_geometry.py](paint_geometry.py), [net_geometry.py](net_geometry.py), [junctions.py](junctions.py) | Painted markings, projected net and line crossings |
| [image_sources.py](image_sources.py), [search_records.py](search_records.py) | Image/box source types and search-record checks |

The research harness in `scratch/court_det_fix/court_detector` holds the code
that reads saved research records. Its inputs remain in the older data folders
listed in [FP_INDEX.md](../../scratch/court_det_fix/FP_INDEX.md#code-and-input-paths-to-keep-stable).

| File | Responsibility |
| --- | --- |
| [run_views.py](../../scratch/court_det_fix/court_detector/run_views.py) | Run the prepared views and compare with saved results |
| [frozen_cases.py](../../scratch/court_det_fix/court_detector/frozen_cases.py) | Load the frozen views' packs, provenance and frames |
| [paint_profiles.py](../../scratch/court_det_fix/court_detector/paint_profiles.py) | Paint-profile diagnostics for research callers |

The [27 September archive map](../../scratch/court_det_fix/archive/20260927_code/README.md) records the
former code locations. Historical scripts there keep their original paths;
they need a matching checkout or path repair before a rerun.
