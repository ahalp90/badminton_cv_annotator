# Run the court detector

The court detector finds a badminton court in an image or video. For each image,
or each scene of a video, it returns four court corners in the image's own
pixels, or a reason that it found no court. It needs no trained court model. It
works from detected straight lines, the known court layout and, for video, where
the players stand.

This page covers setup, choosing a runner, running it and reading its result.
Three other pages hold the rest:

- [Sampling and scene reuse](sampling_and_reuse.md): which frames each scene
  uses, within-scene composition, `--reuse-courts`, `--court-mode video-robust`
  and the exact output contract for each mode
- [Design](../../docs/court_detector/design.md): why the detector works this way
  and the settled design decisions
- [Evaluation](../../docs/court_detector/evaluation.md): the evidence behind it,
  and its remaining limits

**Status.** A full `--court-mode video-robust` GPU run completed with live player
checks. Its main recurring view fitted well in three inspected scenes, but
false courts remain in standalone views. Processing took about twice the
video's duration. See the evaluation page for results and remaining limits.

## How it finds a court

A "projection" below means the four court corners in the image. Together they
define a homography: a mapping between the flat court floor and the image.

1. **Lines.** DeepLSD, a neural line detector, finds straight line fragments in
   the image. Most later steps work on a copy shrunk to at most 960 pixels on its
   longest side.
2. **Players' feet (video).** RTMLib runs models that find people and their body
   joints. The detector samples 31 frames at 10 per second around the analysed
   frame and takes the bottom centre of each standing person's box as their feet.
3. **Search.** It matches pairs of line directions to court markings, and also
   builds courts from crossing lines. Players' feet must land on the court, and
   the implied camera must be plausible and upright.
4. **Score and choose.** Each candidate court gets a combined score: 90% paint
   support plus 10% line support, plus up to 0.04 for visible net posts. "Paint
   support" measures how well the projected markings land on bright painted
   stripes. The best court that passed every check wins.
5. **Refit.** The chosen court's lines are refitted to the centre or edge of
   their painted stripes. The refit must pass the same checks again.
6. **Compose (video).** After a fresh search, the first and last frames of the
   player window are searched too. One court is fitted from whichever frame shows
   each marking best. It replaces the middle frame's court when it passes that
   frame's checks.

A video gets **at most one court per scene**. The court is fixed for the whole
scene; it does not follow a pan or zoom. [Sampling and scene reuse](sampling_and_reuse.md)
describes steps 2 and 6 and the optional cross-scene steps in full.

## Set up

Create the base Python environment using the root README's
[setup instructions](../../README.md#running-it). Run every command below from
the repository root with that environment active and `PYTHONPATH=.:src`.
Replace uppercase paths such as `VIDEO.mp4` with your own inputs. The video CLI
uses Linux CPU-affinity controls.

**Checked-in dependencies.** The root `pyproject.toml` supplies NumPy, SciPy,
Numba, OpenCV and PyTorch. Numba compiles one scoring kernel on first use and
caches it for later runs. The detector also imports `src/shared` and
`src/annotator` from this repository.

**External inputs you supply.**

| Need | Used for | Where it comes from |
| --- | --- | --- |
| DeepLSD source checkout and weights (`deeplsd_md.tar`) | Live line detection in both runners | [DeepLSD](https://github.com/cvg/DeepLSD#usage). The dataset builder expects them at `runtime/checkpoints/deeplsd/DeepLSD`; see [runtime/README.md](../../runtime/README.md). They are gitignored |
| RTMLib and ONNX Runtime | Live pose (`--with-people`, or video without `--people`) and `--pose-prerun` | Pinned in [src/bst_x/preparing_data/requirements.txt](../bst_x/preparing_data/requirements.txt), not in `pyproject.toml` |
| PySceneDetect (`scenedetect`) | `--pyscenedetect`, and the view grouping in `--court-mode video-robust` | Not pinned in `pyproject.toml`; install it in the court environment |
| CuPy | `--template-device cuda` only | Not a project dependency; install it on the GPU machine |

DeepLSD loads from its checkout, not from an installed package. The loader puts
the checkout first on `sys.path` and fails if the folder is missing. Follow its
upstream installation steps to build the required line-detection dependencies
in the same environment.

RTMLib downloads the configured RTMDet person and RTMPose pose models on first
use. Populate its model cache before an offline run. Live CUDA pose extraction
needs `onnxruntime-gpu` and compatible CUDA/cuDNN libraries. Follow the GPU notes
in the linked pose requirements file, including exporting `LD_LIBRARY_PATH`
before Python starts. The adapter raises an error if ONNX Runtime silently
falls back to CPU. The court environment is not fully specified by
`uv sync --extra dev`: PySceneDetect, GPU template dependencies and the external
model setup above remain additional steps.

**Devices.**

- `--device` sets the device for DeepLSD and RTMLib. It defaults to `cuda` and
  fails if CUDA is unavailable. Use `--device cpu` on a CPU-only machine.
- `--template-device` sets where line templates are scored. It defaults to
  `cpu`. `cuda` needs CuPy and a GPU, and never falls back to the CPU. GPU
  rounding can shift results slightly, such as a line sample moving by a pixel.
- `--workers` (1 to 8, default 8) sets the number of CPU worker processes. The
  video runner also pins itself to the first eight CPUs it is allowed to use.

## Choose a runner

| You have | Use |
| --- | --- |
| One still image | `python -m court_detector.run_image` |
| One video | `python -m court_detector.run_video --video` |
| Many videos | `python -m court_detector.run_video --manifest` |
| Your own lines, boxes and frames in Python | `CourtDetector.detect`; see [Python API](#python-api) |

Both runners write a gzipped JSON result file. That file is the authoritative
result. They also print to stdout, but stdout can hold progress and diagnostic
lines as well as result rows, so do not parse it as JSON Lines.

### Corner coordinates

`corners_native_px` is a `(4, 2)` array of floating-point `(x, y)` positions in
the decoded image, whose size is `[width, height]`. Coordinates start at the
top left; x increases rightward and y downward. Corners are ordered **top left,
top right, bottom right, bottom left**, clockwise in the image. For the usual
view from behind a baseline, the far baseline comes first.

The corners describe the outer doubles court, 6.10 × 13.40 metres, even for
singles play. They can lie outside the image. Preserve their order when making
a homography; do not sort them separately by x or y. The canonical model is
[shared/court_model.py](../shared/court_model.py).

## Run from an image

```bash
PYTHONPATH=.:src python -m court_detector.run_image \
  --image HALL.jpg \
  --deeplsd-source DEEPLSD_CHECKOUT \
  --deeplsd-weights DEEPLSD_WEIGHTS.tar \
  --output hall_court.json.gz
```

The image runner reads any file OpenCV can open, runs DeepLSD on it and never
loads PySceneDetect. An unreadable image fails before any model loads. By
default the court comes from lines and geometry alone. Add `--with-people` to run
RTMLib once on the image. Those people then hide themselves from the paint
measurements and support the search. One image has no three-second player
window, so missing or off-court people never veto a court here.

The result holds:

| Field | Meaning |
| --- | --- |
| `schema` | `court-detector-image/1` |
| `image_id`, `image` | The file stem and file name (`image` is `null` when a Python caller gives none) |
| `native_size` | `[width, height]` of the decoded image |
| `status` | `court` or `no_court` |
| `corners_native_px` | Four corners in the image's pixels, or `null`. Corners can lie outside the image |
| `no_court_reason` | Why no court was returned, or `null` |
| `with_people` | Whether RTMLib ran |
| `tools_seconds` | Model loading |
| `line_seconds`, `people_seconds`, `detection_seconds` | DeepLSD, RTMLib (`null` without people) and detector time for this image |
| `stage_seconds` | Detector seconds per step |

`no_court` is a normal result. A search or fit error stops the run before a new
output file is written.

To reuse loaded models for many images in Python:

```python
from pathlib import Path

from court_detector.detect import Switches
from court_detector.run_image import detect_image, load_image_tools, read_image

deeplsd = Path("runtime/checkpoints/deeplsd/DeepLSD")
image_paths = [Path("hall_1.jpg"), Path("hall_2.jpg")]
tools = load_image_tools(
    Switches(require_people=False, workers=8, timing=True),
    deeplsd, deeplsd / "weights/deeplsd_md.tar", with_people=False,
)
with tools.detector:
    results = [detect_image(read_image(path), tools, image_id=path.stem, source=path.name)
               for path in image_paths]
```

## Run from a video

The smallest useful command uses live DeepLSD, live RTMLib and PySceneDetect:

```bash
PYTHONPATH=.:src python -m court_detector.run_video \
  --video VIDEO.mp4 \
  --deeplsd-source DEEPLSD_CHECKOUT \
  --deeplsd-weights DEEPLSD_WEIGHTS.tar \
  --pyscenedetect \
  --output courts.json.gz
```

Change the inputs with these options:

| Input | Live inference (default) | Saved input |
| --- | --- | --- |
| Lines | `--deeplsd-source` and `--deeplsd-weights` | `--saved-lines FILE` instead of both DeepLSD options |
| People | RTMLib on only the frames the detector asks for | `--people POSE_DIR`, or `--pose-prerun DIR` to extract every frame first |
| Scenes | `--pyscenedetect` | `--scenes FILE`. With neither, the whole video is one scene |

- `POSE_DIR` holds `pose_bboxes.npy.xz`, `pose_kps.npy.xz` and
  `pose_ndet.npy.xz`, indexed by source-video frame in native pixels. This is
  the layout the dataset builder's pose stage writes. Saved arrays must cover
  at least the video's frame count; extra trailing frames are allowed.
- `--saved-lines FILE` is a gzipped JSON object mapping frame numbers to
  `(N, 4)` arrays of `x1, y1, x2, y2` in native pixels. It needs every frame the
  detector asks for, or the run stops with an error. That is each scene's middle
  frame, plus the player window's first and last frames when a scene is composed.
- `--scenes FILE` is a gzipped JSON list of `[start_frame, end_frame]` pairs with
  exclusive ends. The scenes must cover the whole video in order, without gaps or
  overlaps. [Sampling and scene reuse](sampling_and_reuse.md#scenes-and-frame-ranges)
  explains the frame ranges.
- `--pose-prerun DIR` saves full-video poses to `DIR/VIDEO_ID`, which must not
  exist yet. It uses the dataset builder's settings: eight shards and at most ten
  people per frame. `--pose-python` picks its interpreter. Reuse the poses later
  with `--people`. It cannot combine with `--people` or `--no-require-people`.
  Live sparse RTMLib keeps every person above its threshold instead, so courts can
  differ when the ten-person limit drops a player in a crowded frame.

**People are required by default.** A scene too short for the three-second
player window is then marked unanalysed. `--no-require-people` lets the detector
use lines and geometry alone. Without `--people` it then skips pose inference
entirely. With `--people`, the saved people still help, but missing or off-court
people no longer veto a court. This line-only fallback is less tuned than
detection with people.

A short clip, with no scene detection and no people:

```bash
PYTHONPATH=.:src python -m court_detector.run_video \
  --video CLIP.mp4 \
  --deeplsd-source DEEPLSD_CHECKOUT \
  --deeplsd-weights DEEPLSD_WEIGHTS.tar \
  --no-require-people \
  --output clip_court.json.gz
```

### Output modes

Two options control how scenes relate to each other. `--reuse-courts` defaults
off, and `--court-mode` defaults to `video-robust`.

| Option | What it does |
| --- | --- |
| `--court-mode scene-robust` | Each scene keeps its own court |
| `--court-mode video-robust` (default) | After the last scene, groups scenes that show the same camera view and may give the group one shared court. Rows print only after that step |
| `--court-mode fast-robust` | Searches only each scene's middle frame, then pools matching views with at least three independent scene fits. Keeps the best complete court unless the pooled fit scores higher |
| `--reuse-courts` | While scenes run in order, tries to carry an earlier court into a returning view, checked against the new scene. Otherwise it searches afresh |

Reuse and video-robust mode can run together, but a reused scene cannot donate
to the shared court. [Sampling and scene reuse](sampling_and_reuse.md) is the
full contract for both. Fast-robust requires fresh fits and cannot be combined
with `--reuse-courts`. It retains the usual temporal player checks even though
it fits the court in only one frame per scene.

### Dataset-builder integration

The dataset builder runs this video runner as a child program with saved poses
and `--pyscenedetect`. The separate end-to-end annotator passes its fixture's
shared scene cuts with `--scenes`. The builder's `court_reuse_courts`
config key controls `--reuse-courts`; both supplied configs,
[shuttleset_fixed.toml](../../configs/dataset_builder/shuttleset_fixed.toml)
and [trial.toml](../../configs/dataset_builder/trial.toml), turn it on. The
optional `vision.court_mode` key sets `--court-mode` and defaults to
`video-robust`. To use `fast-robust`, also set `court_reuse_courts = false`.

Both supplied configurations read the court interpreter from
`BADMINTON_COURT_PYTHON`; set it to the absolute path of the court environment's
Python executable. They request CUDA for both neural inference and template
scoring, so that environment needs the GPU dependencies above. The builder sets
the child working directory to the repository root and `PYTHONPATH=src:src/bst_x`.

The result reader checks the schema, video dimensions, frame count and complete
scene coverage. A detector crash fails the stage with its stderr. Even a
successful detector run can yield no court accepted by the annotator's separate
player vote; that raises `NoAcceptedCourtError` and writes `court_failure.json.gz`.
See [vision.py](../dataset_builder/vision.py) for the process boundary and
[court_evidence.py](../annotator/court_evidence.py) for result validation.

### Video result

The `--output` file is a gzipped JSON object:

| Field | Meaning |
| --- | --- |
| `schema` | `court-detector-video/1` |
| `video_id`, `video` | The file stem (or manifest `id`) and the file name |
| `fps`, `frame_count`, `native_size` | Source video properties |
| `saved_people`, `saved_lines` | Whether people and lines came from saved files |
| `require_people`, `reuse_courts`, `court_mode`, `template_device` | The settings used |
| `scenes` | One row per scene, in order |
| `view_groups` | Video-robust only: one summary per camera-view group |
| `tools_seconds` | Model loading, once per run. Every video in a batch repeats it |
| `setup_seconds`, `scene_seconds`, `processing_seconds` | Input setup (including any pose pre-run), scene detection, and scene work including grouping and pooling |
| `pose_prerun_seconds` | Full pose extraction time; zero without a pre-run. Already included in `setup_seconds` |
| `total_seconds` | This video's time, excluding `tools_seconds` |

With live people, `processing_seconds` includes RTMLib inference, so it is not a
detector-only timing.

Each scene row has `view_id`, `start_frame`, `end_frame` (exclusive),
`frame_index` (the analysed middle frame), `status`, `corners_native_px` and
`seconds`. The `status` is one of:

| Status | Meaning |
| --- | --- |
| `court` | `corners_native_px` holds four corners in native pixels. They can lie outside the image |
| `no_court` | No court passed the checks. `no_court_reason` says why |
| `scene_too_short_for_feet` | People are required and the scene cannot hold the player window. This means unanalysed, not "no court" |
| `detection_failed` | The search or fit raised an error on this scene. `error` and `traceback` record it, and the next scene still runs |

Common `no_court_reason` values: `no_gated_court` (no candidate passed the
checks), `rank_deficient` (the final fit lacked enough independent lines), and
`refit_camera_implausible` or `refit_players_not_on_court` (the refitted court
failed that check).

Analysed rows also carry `chosen_key` (where the court came from; `reuse`,
`composite`, `video_pool` and `video_pool_scene` mark courts that did not come
straight from the middle frame's search),
`reused_from`, `composition` and `stage_seconds`. [Sampling and scene reuse](sampling_and_reuse.md#output-fields-by-mode)
defines those fields and the extra video-robust fields.

Only a `CourtFitError` fails a single scene. Other errors stop the video, such
as a broken worker pool, a GPU error or a people source that returns the wrong
frames. The single-video CLI then exits non-zero without writing a new
`--output` file. Scene-robust stdout may contain earlier scene rows mixed with
diagnostics. Video-robust mode buffers rows until pooling, so a failure before
then leaves no partial court rows. Keep stderr and use a new output path for
each run to avoid mistaking an older file for a completed result.

### Run a batch

A batch loads DeepLSD, RTMLib and the detector once and shares one worker pool.
Each video still gets its own frames, pose cache and reuse store.

```bash
PYTHONPATH=.:src python -m court_detector.run_video \
  --manifest videos.json.gz \
  --deeplsd-source DEEPLSD_CHECKOUT \
  --deeplsd-weights DEEPLSD_WEIGHTS.tar \
  --pyscenedetect \
  --output-dir OUT_DIR
```

The manifest is a gzipped JSON list, one object per video:

```json
[
  {"id": "match_01", "video": "videos/match_01.mp4", "people": "poses/match_01"},
  {"id": "match_02", "video": "videos/match_02.mp4"}
]
```

- `id` and `video` are required. Each `id` must be unique and a plain file name.
  Other keys are rejected
- `people` is that video's `POSE_DIR`. Without it, the video gets live RTMLib,
  a full pose pass with `--pose-prerun`, or no people with `--no-require-people`
- `scenes` is that video's scenes file. It cannot combine with `--pyscenedetect`
- Relative paths resolve from the working directory

A batch rejects `--people`, `--scenes` and `--saved-lines`, because each
describes one video. `OUT_DIR` must not exist yet. The batch writes:

- `OUT_DIR/videos/<id>.json.gz`: the video result, for each completed video
- `OUT_DIR/summary.json.gz` (`court-detector-batch/1`), rewritten after every
  video. Each entry's `status` is `complete`, `failed` or `not_run`

A video whose own files are bad fails alone, such as an unreadable video. A
broken worker pool is replaced before the next video. Any other failure stops
the batch, because the shared models may be in an unknown state; `stopped_after`
names that video. The exit status is 1 unless every video completes.

## Python API

`CourtDetector.detect(view, people, frames)` takes three inputs from
[inputs.py](inputs.py):

- `view` (`ViewInputs`): the image, its line fragments (float32 native pixels),
  person boxes, frame index, scene range and where the image and boxes came from
- `people` (`PeopleSource`): boxes and 17 COCO-layout body joints in nearby frames
- `frames` (`FrameReader`): access to those nearby frames

```python
from court_detector.detect import CourtDetector, Switches

with CourtDetector(Switches(workers=8)) as detector:
    result = detector.detect(view, people, frames)
```

The `with` block keeps one set of worker processes for every call. Without it,
each step starts its own workers. For a line-only court, use
`Switches(require_people=False)`, pass `people=None` and an empty `(0, 4)` box
array. The result is a `CourtResult`: `corners_native_px` or `None`,
`no_court_reason`, `chosen_key`, `paint_score`, `reused_from`, `composition`
and, when timing is on, `stage_seconds`.

To try earlier courts first, pass `known_courts=`, built with
`reuse.make_known_court()` from fully searched results.
`run_video.load_court_tools()` and `run_video.detect_video()` give Python callers
the video runner's behaviour. Open `tools.detector` as a context manager around
the calls, as `run_batch` does.

### Settings

| `Switches` field | Default | Behaviour |
| --- | --- | --- |
| `require_people` | On | Require players' feet on the court. `--no-require-people` turns it off |
| `workers` | 1 (runners: 8) | Worker processes. `--workers` |
| `template_device` | `"cpu"` | Line-template scoring device. `--template-device` |
| `full_score_limit` | `None` | Optional trial: fully score only this many cheaply ranked courts per pair. `run_video --full-score-limit`; `run_image` has no such option |
| `geometry_weight` | 0.1 | Line support's share of the score; paint gets the rest. Python only |
| `upright_camera` | On | Reject courts implying a sideways or upside-down camera. Python only |
| `enforce_scene_consistency` | On | Keep only player samples that look like the analysed frame's shot |
| `self_checks` | On | Check intermediate results |
| `timing` | Off (runners: on) | Report `stage_seconds` |
| `artefacts_dir` | `None` | Write each view's intermediate results here, for debugging |

## Limitations

- **One fixed court per scene.** A pan, zoom or missed cut within a scene can make
  the court wrong for part of it.
- **Scores are not accuracy.** The combined score is the detector's own ranking
  value. It is not measured against hand-labelled courts.
- **Short scenes.** With people required, scenes shorter than about three seconds
  are unanalysed.
- **Line-only mode** has not been tuned to the same precision as detection with
  people.
- **Reuse and pooling thresholds** are heuristic. Difficult views can still give
  false courts.
- **Video-robust mode is under evaluation.** See the
  [evaluation page](../../docs/court_detector/evaluation.md) for its status.

The [design page](../../docs/court_detector/design.md) lists the known limits and
the reasons behind them.

### Reporting a failure

Include:

1. The full command, interpreter and devices used
2. The saved `.json.gz` result, not stdout
3. The stderr log. It records each finished scene and, in video-robust mode, the
   pooling start, each group and the end
4. For a bad scene: its `view_id`, `status`, `chosen_key`, `no_court_reason` or
   `error`, and its `composition` and `view_pool` records when present
5. For a visually wrong court: the frame index and a screenshot with the corners
   drawn

To see intermediate results for one image, run the Python API with
`Switches(artefacts_dir=...)`.

## Code map

Start with [detect.py](detect.py) for the order of the steps, then
[run_video.py](run_video.py) for how scenes are fed in.

| File | Responsibility |
| --- | --- |
| [detect.py](detect.py) | `CourtDetector`, `Switches`, `CourtResult` and the order of the steps |
| [run_video.py](run_video.py), [video_inputs.py](video_inputs.py) | Video and batch runner; frame reading and people sources |
| [run_image.py](run_image.py) | Still-image runner |
| [inputs.py](inputs.py), [image_sources.py](image_sources.py) | Input types and image/box provenance |
| [line_sources.py](line_sources.py), [scene_sources.py](scene_sources.py) | DeepLSD or saved lines; PySceneDetect or saved scenes |
| [feet.py](feet.py) | Player sample window, shot check and feet |
| [composition.py](composition.py) | Compose one court from a scene's middle and endpoint frames |
| [reuse.py](reuse.py) | Chronological reuse checks |
| [view_pool.py](view_pool.py) | Video-robust grouping, pooling and fallback |
| [generation.py](generation.py), [candidate_pool.py](candidate_pool.py), [proposals.py](proposals.py), [search.py](search.py) | Search line-direction pairs and keep candidate courts |
| [line_templates.py](line_templates.py), [template_arrays.py](template_arrays.py) | Build courts from crossing lines; score them on CPU or GPU |
| [scoring.py](scoring.py), [measurements.py](measurements.py), [sampling.py](sampling.py) | Measure, rank and refit candidates |
| [net_choice.py](net_choice.py), [stripe_refit.py](stripe_refit.py) | Final choice with the net reward; stripe refit |
| [geometry.py](geometry.py), [candidate_geometry.py](candidate_geometry.py), [camera.py](camera.py), [court_checks.py](court_checks.py), [players.py](players.py) | Court coordinates, camera, shape and player checks |
| [directions.py](directions.py), [line_matching.py](line_matching.py), [prepare_lines.py](prepare_lines.py), [line_observations.py](line_observations.py) | Line directions, marking matches and fragment groups |
| [stripe_measurements.py](stripe_measurements.py), [stripe_fitting.py](stripe_fitting.py), [paint_geometry.py](paint_geometry.py), [net_geometry.py](net_geometry.py), [junctions.py](junctions.py), [search_records.py](search_records.py) | Stripe fitting, painted markings, net and record checks |

The court's size and markings come from `shared.court_model`. The package imports
nothing from `experiments` or `scratch`. Older notes call the two line searches
G0 and G1 (now `all_lines` and `painted_lines`) and the scoring stage W5.

### Checks after a change

The runner help commands are a quick import and option check. For changes to
behaviour, start with the tests at the relevant boundary:

- [Image runner](../../tests/test_court_detector_run_image.py) and
  [video caller](../../tests/test_court_detector_video_caller.py): inputs, scene
  outcomes and failure handling
- [Composition](../../tests/test_court_detector_composition.py),
  [reuse](../../tests/test_court_detector_reuse.py) and
  [video pooling](../../tests/test_court_view_pool.py): sharing and fallback rules
- [Court evidence](../../tests/test_court_evidence.py): downstream interpretation

Use the project's wider checks for shared-interface or pipeline changes.
Hardware-specific tests and live-model runs require the corresponding GPU
environment. The [evaluation record](../../docs/court_detector/evaluation.md)
provides saved numerical evidence and outlines for interpreting a change.
