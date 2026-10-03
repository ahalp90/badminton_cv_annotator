# Using the court detector

The image and video runners find four outer court corners and write compressed
JSON results. This guide covers the required inputs, setup, commands, every CLI
option and the output formats.

The [overview](README.md) explains the detector's role. [Sampling and reuse](sampling_and_reuse.md)
describes the video modes in detail. The [Python API](../../src/court_detector/README.md#python-api)
is documented beside the implementation.

Sections: [setup](#setup) · [input requirements](#input-requirements) ·
[image example](#run-from-an-image) · [video examples](#run-from-a-video) ·
[batch manifest](#run-a-batch) · [all CLI options](#complete-cli-reference).

## Setup

The base Python environment is described in the repository's
[setup instructions](../../README.md#running-it). Commands here run from the
repository root with that environment active and `PYTHONPATH=.:src`. Uppercase
paths such as `VIDEO.mp4` are placeholders for local files. The video CLI uses
Linux CPU-affinity controls.

**Checked-in dependencies.** The root `pyproject.toml` supplies NumPy, SciPy,
Numba, OpenCV and PyTorch. Numba compiles one scoring kernel on first use and
caches it for later runs. The detector also imports `src/shared` and
`src/annotator` from this repository.

**Additional dependencies and model files.**

| Need | Used for | Where it comes from |
| --- | --- | --- |
| DeepLSD source checkout and weights (`deeplsd_md.tar`) | Live line detection in both runners | [DeepLSD](https://github.com/cvg/DeepLSD#usage). The dataset builder expects them at `runtime/checkpoints/deeplsd/DeepLSD`; see [runtime/README.md](../../runtime/README.md). They are gitignored |
| RTMLib and ONNX Runtime | Live pose (`--with-people`, or player-required video without `--people`) and `--pose-prerun` | Pinned in [src/bst_x/preparing_data/requirements.txt](../../src/bst_x/preparing_data/requirements.txt), not in `pyproject.toml` |
| PySceneDetect (`scenedetect`) | `--pyscenedetect`, and view grouping in `video-robust` and `fast-robust` modes | Required in the court environment; not pinned in `pyproject.toml` |
| CuPy | `--template-device cuda` only | Required in the GPU environment; not a project dependency |

DeepLSD loads from its checkout, not from an installed package. The loader puts
the checkout first on `sys.path` and fails if the folder is missing. Its upstream installation instructions cover
the line-detection dependencies needed in the same environment.

RTMLib downloads the configured RTMDet person and RTMPose pose models on first
use. An offline run therefore needs those weights cached beforehand. Live CUDA
pose extraction needs `onnxruntime-gpu` and compatible CUDA/cuDNN libraries. The
linked pose requirements describe the GPU setup, including the `LD_LIBRARY_PATH`
required before Python starts. The adapter raises an error if ONNX Runtime silently
falls back to CPU. The court environment is not fully specified by
`uv sync --extra dev`: PySceneDetect, GPU template dependencies and the external
model setup above remain additional steps.

**Devices.**

- `--device` sets the device for DeepLSD and RTMLib. It defaults to `cuda` and
  fails if CUDA is unavailable. `--device cpu` selects CPU inference.
- `--template-device` sets where line templates are scored. It defaults to
  `cpu`. `cuda` needs CuPy and a GPU, and never falls back to the CPU. GPU
  rounding can shift results slightly, such as a line sample moving by a pixel.
- `--workers` (1 to 8, default 8) sets the number of CPU worker processes. The
  video runner also pins itself to the first eight CPUs it is allowed to use.

## Input requirements

| Run | Required input | Optional input |
| --- | --- | --- |
| Image | Image, DeepLSD source checkout and weights | Live person detection with `--with-people` |
| Single video | Video; saved lines or DeepLSD source and weights | Saved poses, a full pose pre-run, saved scene ranges or scene detection |
| Batch | Gzipped manifest with one ID and video path per entry; DeepLSD source and weights | Per-entry saved poses and scene ranges; live pose or a pose pre-run for entries without poses |

Player evidence is required by default for video. Live RTMLib supplies it when
saved poses or a pre-run are absent. `--no-require-people` enables line-only
video detection. Scene detection is optional; without it or saved ranges, the
whole video is treated as one scene.

The video examples below specify the saved-array shapes, line-map format and
scene ranges. The [batch section](#run-a-batch) defines the manifest.

## Runners

| Input | Entry point |
| --- | --- |
| One still image | `python -m court_detector.run_image` |
| One video | `python -m court_detector.run_video --video` |
| Many videos | `python -m court_detector.run_video --manifest` |
| Prepared lines, boxes and frames in Python | `CourtDetector.detect`; see [Python API](../../src/court_detector/README.md#python-api) |

Both runners write a gzipped JSON result file. That file is the authoritative
result. They also print to stdout, but stdout can hold progress and diagnostic
lines as well as result rows. The saved JSON file contains the complete result.

### Corner coordinates

`corners_native_px` is a `(4, 2)` array of floating-point `(x, y)` positions in
the decoded image, whose size is `[width, height]`. Coordinates start at the
top left; x increases rightward and y downward. Corners are ordered **top left,
top right, bottom right, bottom left**, clockwise in the image. Final image and
video outputs put the far baseline first. If the first baseline's mean image y is
larger than the second's, the runner rolls the four corners by two. It preserves
every coordinate and never applies a quarter-turn or another fit.

This convention applies to exported `corners_native_px`. Private detector results,
original scene courts and diagnostic candidate arrays keep their fitting order.

The corners describe the outer doubles court, 6.10 × 13.40 metres, even for
singles play. They can lie outside the image. Homography construction depends on the supplied
corner order; sorting by x or y changes the court correspondence. The canonical model is
[shared/court_model.py](../../src/shared/court_model.py).

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
default the court comes from lines and geometry alone. `--with-people` runs RTMLib once on the image. Their boxes mask occluded paint during measurement.
With `--full`, player support also breaks exact direction-search score ties.
One image has no three-second player window, so missing or off-court people
never veto a court here.

### Search breadth without required players

| Option | Search within each image or sampled frame |
| --- | --- |
| Omitted, or `--fast` | Line templates alone |
| `--full` | All-lines, painted-lines and line-template searches |

`--fast` and `--full` are mutually exclusive. They apply to image mode and video
mode with `--no-require-people`, including every frame of a three-frame scene
search. They leave the player-required search policy unchanged. A still image
remains a single-frame input. `--court-mode` separately controls video sampling
and sharing; `--fast` does not select `--court-mode fast-robust`.

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

Loaded models can be reused across images in Python:

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

Video inputs can come from live inference or saved files:

| Input | Live inference (default) | Saved input |
| --- | --- | --- |
| Lines | `--deeplsd-source` and `--deeplsd-weights` | `--saved-lines FILE` instead of both DeepLSD options |
| People | RTMLib on only the frames the detector asks for | `--people POSE_DIR`, or `--pose-prerun DIR` to extract every frame first |
| Scenes | `--pyscenedetect` | `--scenes FILE`. With neither, the whole video is one scene |

- `POSE_DIR` holds `pose_bboxes.npy.xz`, `pose_kps.npy.xz` and
  `pose_ndet.npy.xz`, indexed by source-video frame in native pixels. This is
  the layout the dataset builder's pose stage writes. Saved arrays must cover
  at least the video's frame count; extra trailing frames are allowed.
  Their shapes are `(frames, people_slots, 4)` for boxes,
  `(frames, people_slots, 17, 2)` for COCO-17 keypoints, and `(frames,)` for
  integer detection counts. Each count is between zero and `people_slots`.
  Boxes use `(x1, y1, x2, y2)`; keypoints use `(x, y)` in the same source image.
- `--saved-lines FILE` is a gzipped JSON object mapping frame numbers to
  `(N, 4)` arrays of finite `x1, y1, x2, y2` values in native pixels.
  Empty line arrays are allowed. The map must contain every requested frame;
  a missing frame stops the run. That is each scene's middle
  frame, plus the player window's first and last frames for a three-frame search.
  `fast-robust` uses only middle-frame lines.
- `--scenes FILE` is a gzipped JSON list of `[start_frame, end_frame]` pairs with
  exclusive ends. The scenes must cover the whole video in order, without gaps or
  overlaps. [Sampling and scene reuse](sampling_and_reuse.md#scenes-and-frame-ranges)
  explains the frame ranges.
- `--pose-prerun DIR` saves full-video poses to `DIR/VIDEO_ID`, which must not
  exist yet. It uses the dataset builder's settings: eight shards and at most ten
  people per frame. `--pose-python` picks its interpreter. Later runs can load
  the saved poses through `--people`. The CLI rejects combining it with `--people` or
  `--no-require-people`. In a batch, manifest entries with saved poses use those
  arrays; the pre-run supplies entries without poses.
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
`video-robust`. `fast-robust` requires `court_reuse_courts = false`.

Both supplied configurations read the court interpreter from
`BADMINTON_COURT_PYTHON`, whose value is the absolute path of the court
environment's Python executable. They request CUDA for both neural inference and template
scoring, so that environment needs the GPU dependencies above. The builder sets
the child working directory to the repository root and `PYTHONPATH=src:src/bst_x`.

The result reader checks the schema, video dimensions, frame count and complete
scene coverage. A detector crash fails the stage with its stderr. Even a
successful detector run can yield no court accepted by the annotator's separate
player vote; that raises `NoAcceptedCourtError` and writes `court_failure.json.gz`.
[vision.py](../../src/dataset_builder/vision.py) defines the process boundary;
[court_evidence.py](../../src/annotator/court_evidence.py) validates results.

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
| `view_groups` | In `video-robust` and `fast-robust`: one summary per camera-view group |
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
`first`, `last`, `composite`, `video_pool` and `video_pool_scene` mark courts
that did not come straight from the middle frame's search),
`reused_from`, `composition` and `stage_seconds`. [Sampling and scene reuse](sampling_and_reuse.md#output-fields-by-mode)
defines those fields and the extra video-robust fields.

Only a `CourtFitError` fails a single scene. Other errors stop the video, such
as a broken worker pool, a GPU error or a people source that returns the wrong
frames. The single-video CLI then exits non-zero without writing a new
`--output` file. Scene-robust stdout may contain earlier scene rows mixed with
diagnostics. Video-robust mode buffers rows until pooling, so a failure before
then leaves no partial court rows. Stderr records the failure. An existing
output file can remain from an earlier run when the new run fails.

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

- The accepted keys are `id`, `video`, `people` and `scenes`. `id` and `video`
  are required; each `id` must be non-empty, unique and a plain file name
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

## Complete CLI reference

The two entry points are `python -m court_detector.run_image` and
`python -m court_detector.run_video`. The tables below cover every option
accepted by their parsers, including defaults and required combinations.

### Options shared by both runners

| Option | Default | Behaviour |
| --- | --- | --- |
| `-h`, `--help` | — | Prints the runner's options and exits before model loading |
| `--deeplsd-source PATH` | Unset | DeepLSD source checkout; required for images and for video without `--saved-lines` |
| `--deeplsd-weights PATH` | Unset | DeepLSD checkpoint; required together with its source checkout for live line detection |
| `--device VALUE` | `cuda` | Neural inference device for DeepLSD and live people detection; `cpu` runs those models on the CPU |
| `--template-device {cpu,cuda}` | `cpu` | Line-template scoring device; `cuda` requires CuPy and a GPU |
| `--workers INTEGER` | `8` | Number of CPU worker processes, from 1 through 8 |
| `--fast` | Effective default when players are optional | Searches line templates alone; mutually exclusive with `--full` |
| `--full` | Off | Adds all-lines and painted-lines searches when players are optional; has no effect on the player-required search |

`--fast` and `--full` control the search within a frame. The video option
`--court-mode fast-robust` controls which frames are searched and how their
courts are shared.

### Image options

| Option | Default | Behaviour |
| --- | --- | --- |
| `--image PATH` | Required | Image file readable by OpenCV |
| `--output PATH` | Required | Destination for the gzipped JSON result |
| `--with-people` | Off | Runs RTMLib once; person boxes mask obscured paint and can break exact direction-search score ties with `--full` |

Image detection always treats people as optional. It has no temporal player
window and no `--require-people` switch.

### Video and batch options

| Option | Default | Behaviour |
| --- | --- | --- |
| `--video PATH` | Unset | One video; exactly one of `--video` and `--manifest` is required |
| `--manifest PATH` | Unset | Gzipped JSON batch manifest; mutually exclusive with `--video` |
| `--output PATH` | Unset | Required with `--video`; rejected with `--manifest` |
| `--output-dir DIR` | Unset | Required with `--manifest`; the directory must be new. Rejected with `--video` |
| `--people DIR` | Unset | Saved pose arrays for one video; mutually exclusive with `--pose-prerun` |
| `--pose-prerun DIR` | Off | Extracts poses into `DIR/VIDEO_ID` before court detection; that per-video directory must be new. Requires player checks to remain enabled |
| `--pose-python PATH` | Current Python interpreter | Interpreter used for the full pose pre-run |
| `--require-people` | On | Requires player-position evidence for fresh courts |
| `--no-require-people` | Off | Disables player rejection; skips live pose inference. Supplied saved poses still contribute |
| `--saved-lines PATH` | Unset | Saved line fragments for one video; takes precedence over the DeepLSD options when present |
| `--scenes PATH` | Unset | Saved scene ranges for one video; mutually exclusive with `--pyscenedetect` |
| `--pyscenedetect` | Off | Detects scene cuts and representative histograms; without either scene option, the whole video is one scene |
| `--full-score-limit INTEGER` | Unset | Experimental limit on courts reaching full scoring per direction pair; must be at least 1. Omitted means exhaustive scoring |
| `--reuse-courts` | Off | Tries a checked transfer from an earlier matching view before searching afresh |
| `--court-mode {scene-robust,video-robust,fast-robust}` | `video-robust` | Selects within-scene fitting and cross-scene pooling, described in [output modes](#output-modes) |

The remaining combination rules are:

- `--court-mode fast-robust` rejects `--reuse-courts` because it needs fresh scene fits.
- `--pose-prerun` rejects `--no-require-people`.
- Batch mode rejects the single-video options `--people`, `--scenes` and
  `--saved-lines`. Manifest entries can supply `people` and `scenes`; batch
  line detection uses DeepLSD.
- `--pyscenedetect` rejects a manifest containing any saved `scenes` entry.
- With `--pose-prerun` in batch mode, entries with saved `people` use those
  arrays. The pre-run supplies poses for entries without them.

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
- **Video-robust mode has bounded evaluation.** The
  [development evaluation](../../experiments/court_detector/comparisons/development_evaluation.md) records the tested footage and remaining errors.

The [design page](design.md) lists the known limits and
the reasons behind them.

### Evidence for investigating a failure

A useful failure record contains:

1. The full command, interpreter and devices used
2. The saved `.json.gz` result, not stdout
3. The stderr log. It records each finished scene and, in video-robust mode, the
   pooling start, each group and the end
4. For a bad scene: its `view_id`, `status`, `chosen_key`, `no_court_reason` or
   `error`, and its `composition` and `view_pool` records when present
5. For a visually wrong court: the frame index and a screenshot with the corners
   drawn

The Python API can save intermediate results for one image with
`Switches(artefacts_dir=...)`.
