# PySceneDetect 0.7.1 affordance investigation

Performance measured only on `sset_01_288p.mp4`

## 1. Installed API inventory

### Package and detector exports

The installed package top-level exports include `ContentDetector`, `AdaptiveDetector`, `ThresholdDetector`, `HistogramDetector`, `HashDetector`, `SceneManager`, `StatsManager`, `open_video`, `save_images`, the two video splitters, and CSV/HTML scene-list writers. (`scenedetect/__init__.py:59-82`.)

The public detector constructors are:

| Detector | Constructor parameters | Installed source |
|---|---|---|
| `ContentDetector` | `threshold=27.0`, `min_scene_len=15`, `weights=Components(delta_hue=1.0, delta_sat=1.0, delta_lum=1.0, delta_edges=0.0)`, `luma_only=False`, `kernel_size=None`, `filter_mode=FlashFilter.Mode.MERGE` | `scenedetect/detectors/content_detector.py:104-112` |
| `AdaptiveDetector` | `adaptive_threshold=3.0`, `min_scene_len=15`, `window_width=2`, `min_content_val=15.0`, `weights=ContentDetector.DEFAULT_COMPONENT_WEIGHTS`, `luma_only=False`, `kernel_size=None` | `scenedetect/detectors/adaptive_detector.py:37-46` |
| `HistogramDetector` | `threshold=0.05`, `bins=256`, `min_scene_len=15` | `scenedetect/detectors/histogram_detector.py:33-38` |
| `HashDetector` | `threshold=0.395`, `size=16`, `lowpass=2`, `min_scene_len=15` | `scenedetect/detectors/hash_detector.py:47-53` |
| `ThresholdDetector` | `threshold=12`, `min_scene_len=15`, `fade_bias=0.0`, `add_final_scene=False`, `method=Method.FLOOR`, `block_size=None` | `scenedetect/detectors/threshold_detector.py:48-56` |

`TransnetV2Detector` also exists in `scenedetect/detectors/transnet_v2.py:131-139`. Its constructor parameters are `model_path='tests/resources/transnetv2.onnx'`, `onnx_providers=None`, `threshold=0.5`, `min_scene_len=15`, and `filter_mode=FlashFilter.Mode.MERGE`. Construction creates a `Predictor`, whose constructor imports `onnxruntime`. (`scenedetect/detectors/transnet_v2.py:49-68`, `:150-155`.) Introspection imported it from its module, but importing it from `scenedetect.detectors` raised `ImportError`; the installed top-level package does not re-export it. (`scenedetect/detectors/__init__.py:38-42`; command output below.)

The base detector interface is `SceneDetector`; its `process_frame()` returns detected cut timecodes, and its `post_process()` may return additional cut timecodes. (`scenedetect/detector.py:37-73`.)

Captured detector-module inventory output:

```text
scenedetect.detectors: AdaptiveDetector, ContentDetector, HashDetector,
HistogramDetector, ThresholdDetector
TransnetV2Detector_imported (... threshold: float = 0.5 ... filter_mode ...)
TransnetV2Detector_from_detectors_error ImportError("cannot import name 'TransnetV2Detector'")
```

### SceneManager, StatsManager, and the convenience helper

`SceneManager` has constructor signature `SceneManager(stats_manager=None)`. Its public methods include `add_detector`, `detect_scenes`, `get_scene_list`, `get_cut_list`, `clear`, `clear_detectors`, `stop`, and `get_num_detectors`. (`scenedetect/scene_manager.py:218-227`, `:337-376`, `:442-455`, `:709-737`.)

`StatsManager` has constructor signature `StatsManager(base_timecode=None)`. Its public methods include `register_metrics`, `get_metrics`, `set_metrics`, `metrics_exist`, `is_save_required`, `save_to_csv`, and deprecated `load_from_csv`. (`scenedetect/stats_manager.py:85-105`, `:120-168`, `:219-241`.)

Importing `scenedetect.video_manager` in the installed environment raised `ModuleNotFoundError: No module named 'scenedetect.video_manager'`. Captured introspection output was:

```text
scenedetect.video_manager IMPORT ERROR ModuleNotFoundError: No module named 'scenedetect.video_manager'
```

The active video abstraction is `VideoStream` opened through `open_video()`. (`scenedetect/video_stream.py:79-203`.)

The package also exposes `scenedetect.detect(video_path, detector, stats_file_path=None, show_progress=False, start_time=None, end_time=None, start_in_scene=False, backend='opencv')`. The helper opens the video, creates a `SceneManager`, optionally binds a `StatsManager`, calls `detect_scenes(video=video)`, optionally saves stats, and returns `get_scene_list()`. (`scenedetect/__init__.py:154-213`.)

### Video backends

The installed `AVAILABLE_BACKENDS` mapping contains only OpenCV:

```text
AVAILABLE_BACKENDS {'opencv': 'scenedetect.backends.opencv.VideoStreamCv2'}
VideoStreamAv_export None
VideoStreamMoviePy_export None
ffmpeg_available True
mkvmerge_available False
scenedetect.backends.pyav import_error ModuleNotFoundError("No module named 'av'")
scenedetect.backends.moviepy import_error ModuleNotFoundError("No module named 'moviepy'")
```

The source constructs `AVAILABLE_BACKENDS` from `VideoStreamCv2`, `VideoStreamAv`, and `VideoStreamMoviePy`, filtering out unavailable imports. (`scenedetect/backends/__init__.py:94-122`.) `open_video()` defaults to `backend='opencv'`, accepts `frame_rate`, and falls back to `VideoStreamCv2` when the requested backend is unavailable or fails. (`scenedetect/__init__.py:88-151`.) A list or tuple of paths selects `VideoStreamConcat`. (`scenedetect/__init__.py:122-125`; `scenedetect/backends/concat.py:88-95`.)

The tested file opened as `scenedetect.backends.opencv.VideoStreamCv2`, with frame rate 25, frame size `(512, 288)`, and duration frame number `154393`. Command output:

```text
backend: scenedetect.backends.opencv.VideoStreamCv2
frame_rate: 25
frame_size: (512, 288)
duration_frame_num: 154393
duration_timecode: 01:42:55.720
```

### Output helpers

The installed Python output helpers and signatures are:

| Capability | Helper and signature | Source |
|---|---|---|
| Scene images | `save_images(scene_list, video, num_images=3, frame_margin=1, image_extension='jpg', encoder_param=95, image_name_template='$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER', output_dir=None, show_progress=False, scale=None, height=None, width=None, interpolation=Interpolation.CUBIC, threading=True)` | `scenedetect/output/image.py:352` |
| Video split | `split_video_ffmpeg(input_video_path, scene_list, output_dir=None, output_file_template='$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4', video_name=None, arg_override='-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset veryfast -crf 22 -c:a aac', show_progress=False, show_output=False, suppress_output=None, hide_progress=None, formatter=None)` | `scenedetect/output/video.py:255` |
| Video split | `split_video_mkvmerge(input_video_path, scene_list, output_dir=None, output_file_template='$VIDEO_NAME.mkv', video_name=None, show_output=False, suppress_output=None)` | `scenedetect/output/video.py:159` |
| CSV scene list | `write_scene_list(output_csv_file, scene_list, include_cut_list=True, cut_list=None, col_separator=',', row_separator='\n')` | `scenedetect/output/__init__.py:71-78` |
| HTML scene list | `write_scene_list_html(output_html_filename, scene_list, cut_list=None, css=None, css_class='mytable', image_filenames=None, image_width=None, image_height=None)` | `scenedetect/output/__init__.py:135` |
| CMX 3600 EDL | `write_scene_list_edl(output_path, scene_list, title='PySceneDetect', reel='AX', start_timecode=None)` | `scenedetect/output/__init__.py:295` |
| Final Cut Pro X XML | `write_scene_list_fcpx(output_path, scene_list, video_path, frame_rate, frame_size, video_name=None)` | `scenedetect/output/__init__.py:350` |
| Final Cut Pro 7 XML | `write_scene_list_fcp7(output_path, scene_list, video_path, frame_rate, frame_size, video_name=None, source_duration=None)` | `scenedetect/output/__init__.py:449` |
| OpenTimelineIO | `write_scene_list_otio(output_path, scene_list, video_path, frame_rate, name=None, audio=True)` | `scenedetect/output/__init__.py:569` |

The `scenedetect.output` package re-exports `save_images`, both splitters, availability checks, and the CSV/HTML writers. (`scenedetect/output/__init__.py:41-66`.)

### CLI subcommands

Command:

```text
~/.venvs/badminton-cicd/bin/python -m scenedetect --help
```

The captured help output listed these subcommands:

```text
about             detect-adaptive   detect-content    detect-hash
detect-hist       detect-threshold  help              list-scenes
load-scenes       save-edl          save-fcp          save-html
save-images       save-otio         save-qp           split-video
time              version
```

The CLI help described `detect-adaptive`, `detect-content`, and `detect-threshold` as detector commands and also listed `detect-hash` and `detect-hist`. It listed `save-images`, `split-video`, scene-list/export commands, `load-scenes`, and `time`. This list is captured CLI output; the command exited 0.

The `TransnetV2Detector` module docstring names a `detect-transnetv2` command, but `detect-transnetv2` was absent from the installed CLI help. (`scenedetect/detectors/transnet_v2.py:12-15`; captured help output above.)

### Config-file support

The CLI has global `-c/--config FILE`. Its help says the default lookup is `~/.config/PySceneDetect/scenedetect.cfg`. (`python -m scenedetect --help` captured output.)

The source uses `ConfigParser`, names the file `scenedetect.cfg`, derives the default directory with `user_config_dir('PySceneDetect', False)`, and combines them as `CONFIG_FILE_PATH`. (`scenedetect/_cli/config.py:22-23`, `:346-350`.) `ConfigRegistry(path=None, throw_exception=True)` loads an explicit path or the default path if it exists, parses it with `ConfigParser`, and validates the resulting sections and options. (`scenedetect/_cli/config.py:706-778`.)

The config map includes backend, detector, global, list-scenes, save-edl, save-html, save-images, save-otio, save-qp, save-fcp, and split-video sections. (`scenedetect/_cli/config.py:354-480`.) The supported detector config choices in the map are adaptive, content, threshold, hash, and histogram; the map has no TransNetV2 entry. (`scenedetect/_cli/config.py:485-503`.)

## 2. Back-compatibility and preferred 0.7 call pattern

### Exact current call pattern

The repository call pattern was run with:

```text
PYTHONPATH=src ~/.venvs/badminton-cicd/bin/python -W always::DeprecationWarning - <<'PY'
from scenedetect import ContentDetector, SceneManager, open_video
video = open_video(str(video_path))
manager = SceneManager()
manager.add_detector(ContentDetector(threshold=27, min_scene_len=15))
frames_read = manager.detect_scenes(video, show_progress=False)
scenes = manager.get_scene_list()
PY
```

Captured output was:

```text
META duration.frame_num=154393
RESULT frames_read=154393 scene_count=418 elapsed_seconds=88.798247
```

No `DeprecationWarning` was emitted by this exact call pattern under `-W always::DeprecationWarning`. `detect_scenes()` documents its return value as the number of frames read and processed, and its implementation returns `video.frame_number - start_frame_num`. (`scenedetect/scene_manager.py:456-486`, `:613-616`.) `get_scene_list()` is an active method, not marked deprecated. (`scenedetect/scene_manager.py:376-401`.) `video.duration.frame_num` is used by the active `SceneManager` implementation when calculating total frames. (`scenedetect/scene_manager.py:549-555`.)

### Deprecated or future-removal surfaces found in installed source

- `detect_scenes(frame_source=...)` is deprecated. The source says it is for compatibility, emits a warning, and has a `TODO(v0.8)` to remove it. (`scenedetect/scene_manager.py:446-495`.) A live one-frame probe captured the warning verbatim:

  ```text
  <stdin>:7: DeprecationWarning: The `frame_source` argument is deprecated, use `video` instead.
  frame_source_compat_frames_read= 1
  ```

- `SceneManager.get_cut_list()` is deprecated and emits `get_cut_list() is deprecated and will be removed in a future release.` (`scenedetect/scene_manager.py:706-737`.) The repository uses `get_scene_list()`, not `get_cut_list()`.

- `open_video(..., framerate=...)` is documented as a deprecated alias for `frame_rate`. The implementation currently maps it to `frame_rate`; a source TODO says to emit a `DeprecationWarning` after downstream migration. (`scenedetect/__init__.py:88-121`.) The repository does not pass `framerate=`.

- `StatsManager.load_from_csv()` is marked deprecated, is documented to become a no-op and then be removed, and currently logs `load_from_csv() is deprecated and will be removed in a future release.` (`scenedetect/stats_manager.py:219-245`.)

- The default `StatsManager(base_timecode=None)` is documented as being removed in a future release. (`scenedetect/stats_manager.py:99-105`.) This is not part of the repository call pattern because the repository does not construct a `StatsManager` there.

- `ThresholdDetector(block_size=...)` is deprecated and its source says it will be removed in v0.8. (`scenedetect/detectors/threshold_detector.py:48-78`.) The shoot-out did not pass `block_size`.

- The `scenedetect.frame_timecode` submodule is deprecated and warns to import from the base package instead. (`scenedetect/frame_timecode.py:12-20`.) The repository imports the public top-level API instead.

### Documented 0.7 call form

The installed `SceneManager` module documents this form:

```python
video = open_video(test_video_file)
scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector())
scene_manager.detect_scenes(video=video, callback=on_new_scene)
scene_list = scene_manager.get_scene_list()
```

(`scenedetect/scene_manager.py:47-50`; the stats variant is at `:65-73`.) The keyword `video=video` is the documented spelling. The repository's positional `detect_scenes(video, ...)` call is accepted by the active signature and emitted no deprecation warning in the full run.

## 3. Detector shoot-out

The runner used `time.perf_counter()`, passed `min_scene_len=15` to every constructor, and armed a 600-second per-detector alarm. Every run reported `status=ok` and `frames_read=154393`; no run exceeded the 10-minute limit.

The baseline cut list contained 417 cuts. For overlap, a baseline cut was matched when any candidate cut was within ±2 frames. `candidate_matched` counts candidate cuts in that relation; `extra` is the candidate count minus candidate matches; `missing` is the baseline count minus baseline matches.

| Detector and parameters used | Cuts | Wall time (s) | Baseline matched ±2 | Candidate matched ±2 | Extra | Missing |
|---|---:|---:|---:|---:|---:|---:|
| `ContentDetector(threshold=27, min_scene_len=15)` | 417 | 89.263667 | 417 | 417 | 0 | 0 |
| `AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=15, window_width=2, min_content_val=15.0, weights=default, luma_only=False, kernel_size=None)` | 386 | 87.784510 | 352 | 352 | 34 | 65 |
| `HistogramDetector(threshold=0.05, bins=256, min_scene_len=15)` | 1,398 | 84.531028 | 345 | 345 | 1,053 | 72 |
| `HashDetector(threshold=0.395, size=16, lowpass=2, min_scene_len=15)` | 220 | 100.529683 | 163 | 163 | 57 | 254 |
| `ThresholdDetector(threshold=12, min_scene_len=15, fade_bias=0.0, add_final_scene=False, method=FLOOR, block_size=None)` | 105 | 80.286218 | 1 | 1 | 104 | 416 |

Captured runner result lines:

```text
RESULT name=ContentDetector(threshold=27,min_scene_len=15) status=ok frames_read=154393 scene_count=418 cut_count=417 elapsed_seconds=89.263667
RESULT name=AdaptiveDetector(defaults,min_scene_len=15) status=ok frames_read=154393 scene_count=387 cut_count=386 elapsed_seconds=87.784510
OVERLAP ... baseline_matched_pm2=352 candidate_matched_pm2=352 extra_candidate_cuts=34 missing_baseline_cuts=65
RESULT name=HistogramDetector(defaults,min_scene_len=15) status=ok frames_read=154393 scene_count=1399 cut_count=1398 elapsed_seconds=84.531028
OVERLAP ... baseline_matched_pm2=345 candidate_matched_pm2=345 extra_candidate_cuts=1053 missing_baseline_cuts=72
RESULT name=HashDetector(defaults,min_scene_len=15) status=ok frames_read=154393 scene_count=221 cut_count=220 elapsed_seconds=100.529683
OVERLAP ... baseline_matched_pm2=163 candidate_matched_pm2=163 extra_candidate_cuts=57 missing_baseline_cuts=254
RESULT name=ThresholdDetector(defaults,min_scene_len=15) status=ok frames_read=154393 scene_count=106 cut_count=105 elapsed_seconds=80.286218
OVERLAP ... baseline_matched_pm2=1 candidate_matched_pm2=1 extra_candidate_cuts=104 missing_baseline_cuts=416
```

The separate exact-call baseline run measured 417 cuts as 418 scenes in 88.798247 seconds. (`Task 2` command output above.)

## 4. StatsManager affordance

The stats CSV is [content_stats.csv](content_stats.csv). It was written by `StatsManager.save_to_csv()` under SCRATCH.

The first run used `ContentDetector(threshold=27, min_scene_len=15)` with `SceneManager(stats_manager=StatsManager())`:

```text
FIRST threshold=27 frames_read=154393 read_calls=154394 cuts=417 elapsed_seconds=250.999622 stats_save_seconds=9.123397 metric_keys=['content_val', 'delta_edges', 'delta_hue', 'delta_lum', 'delta_sat']
```

The CSV header and first three data rows were:

```text
['Frame Number', 'Timecode', 'content_val', 'delta_edges', 'delta_hue', 'delta_lum', 'delta_sat']
['2', '00:00:00.040', '0.6987395109953703', '7.048746744791667', '0.5224066840277778', '0.1242947048611111', '1.4495171440972223']
['3', '00:00:00.080', '0.5085087528935185', '6.239420572916667', '0.3339029940277778', '0.10530598958333333', '1.0863172743055556']
['4', '00:00:00.120', '0.9021357783564815', '6.10107421875', '0.7566189236111112', '0.1767578125', '1.7730305980972223']
CSV_DATA_ROWS 154392
```

The CSV contains the five detector metric columns `content_val`, `delta_edges`, `delta_hue`, `delta_lum`, and `delta_sat`. `ContentDetector.METRIC_KEYS` defines the same five names. (`scenedetect/detectors/content_detector.py:85-89`.) The first frame is used to initialise the previous frame and returns `0.0` without recording a metric; subsequent frames calculate and store the metrics. (`scenedetect/detectors/content_detector.py:154-186`.)

For the second run, the same in-memory `StatsManager` was reused with a fresh video stream and `ContentDetector(threshold=20, min_scene_len=15)`:

```text
SECOND threshold=20 frames_read=154393 read_calls=154394 cuts=589 elapsed_seconds=269.303276 stats_is_save_required=True metric_keys=['content_val', 'delta_edges', 'delta_hue', 'delta_lum', 'delta_sat']
```

The second run therefore did not avoid video decoding: its wrapped `video.read()` call count was again 154,394. The measured detection time was also 269.303276 seconds. The installed `ContentDetector` source computes the HSV components and frame score, then calls `set_metrics()`; it has no `metrics_exist()` branch. (`scenedetect/detectors/content_detector.py:154-190`; `rg` search for `metrics_exist` returned no match in that file.) `StatsManager.metrics_exist()` exists as an API, but detector reuse of cached metrics is detector-specific. (`scenedetect/stats_manager.py:147-153`.)

## 5. `min_scene_len` and flash-filter semantics

`FlashFilter` is defined in `scenedetect/detector.py:106-124`. Its two modes are:

```python
MERGE = 0
"""Merge consecutive cuts shorter than filter length."""
SUPPRESS = 1
"""Suppress consecutive cuts until the filter length has passed."""
```

The source implements the modes as follows:

- `SUPPRESS` emits a cut only when the current frame is above threshold and the minimum interval since the previous above-threshold frame has elapsed. (`scenedetect/detector.py:171-187`.)
- `MERGE` advances the last above-threshold frame, holds consecutive short events after the first cut, and emits the last above-threshold frame once the below-threshold interval is long enough. (`scenedetect/detector.py:189-224`.)

`ContentDetector` configures the filter in its constructor with `self._flash_filter = FlashFilter(mode=filter_mode, length=min_scene_len)`. Its default is `filter_mode=FlashFilter.Mode.MERGE`. (`scenedetect/detectors/content_detector.py:104-142`.)

`AdaptiveDetector` has no `filter_mode` constructor argument. It passes `min_scene_len=0` to its `ContentDetector` base constructor and checks its own minimum interval after the adaptive ratio test. (`scenedetect/detectors/adaptive_detector.py:37-83`, `:134-142`.) `HistogramDetector`, `HashDetector`, and `ThresholdDetector` implement their own minimum-scene-length checks rather than accepting a `filter_mode` argument. (`scenedetect/detectors/histogram_detector.py:49-57`; `scenedetect/detectors/hash_detector.py:54-63`; `scenedetect/detectors/threshold_detector.py:80-95`.)

## 6. ContentDetector score components

The installed constructor signature is:

```python
def __init__(
    self,
    threshold: float = 27.0,
    min_scene_len: TimecodeLike = 15,
    weights: "ContentDetector.Components" = DEFAULT_COMPONENT_WEIGHTS,
    luma_only: bool = False,
    kernel_size: int | None = None,
    filter_mode: FlashFilter.Mode = FlashFilter.Mode.MERGE,
):
```

(`scenedetect/detectors/content_detector.py:104-112`.)

The `Components` named tuple contains `delta_hue`, `delta_sat`, `delta_lum`, and `delta_edges`; their defaults are `1.0`, `1.0`, `1.0`, and `0.0`. (`scenedetect/detectors/content_detector.py:58-71`.) The detector converts each BGR frame to HSV and splits it into `hue`, `sat`, and `lum`. (`scenedetect/detectors/content_detector.py:154-155`.) It computes mean pixel distances for those three arrays and, when enabled, for an edge map. (`scenedetect/detectors/content_detector.py:157-175`.)

Custom component weights are supported through the `weights` parameter. The frame score is the weighted sum of the four component values divided by the sum of the absolute weights. (`scenedetect/detectors/content_detector.py:166-180`.) `luma_only=True` replaces the supplied weights with `LUMA_ONLY_WEIGHTS`, which sets hue and saturation to zero, luma to `1.0`, and edges to zero. (`scenedetect/detectors/content_detector.py:77-83`, `:128-134`.) `kernel_size` controls the edge-detection kernel and must be an odd integer at least 3 when supplied. (`scenedetect/detectors/content_detector.py:124-138`.)

The stats metric names are `content_val`, `delta_hue`, `delta_sat`, `delta_lum`, and `delta_edges`. (`scenedetect/detectors/content_detector.py:85-89`.)

## 7. Scene classification and motion affordances

The installed-source search command was:

```text
rg -n -i 'close.?up|slow.?motion|replay|motion|optical.?flow|classification|classif|label|content label|scene content|transnet' \
  ~/.venvs/badminton-cicd/lib/python3.11/site-packages/scenedetect \
  --glob '*.py' --glob '!_thirdparty/**'
```

The matches were:

- AdaptiveDetector prose about fast camera motion. (`scenedetect/detectors/adaptive_detector.py:12-16`.)
- ContentDetector comments saying motion estimation and optical flow are TODO ideas, not implemented detector options. (`scenedetect/detectors/content_detector.py:147-152`.)
- A commented-out `MotionDetector` sketch in `detectors/__init__.py`, not a class definition. (`scenedetect/detectors/__init__.py:44-71`.)
- The `TransnetV2Detector` module. (`scenedetect/detectors/transnet_v2.py:12-15`, `:131-139`.)

No installed-source match implemented close-up, slow-motion, replay, or scene-content labelling. The implemented detector interface returns cut timecodes, and `TransnetV2Detector` also returns filtered cut timecodes from `process_frame()` rather than content labels. (`scenedetect/detector.py:48-60`; `scenedetect/detectors/transnet_v2.py:165-187`.) The installed public detector module exports only the five classical boundary detectors listed in Task 1. (`scenedetect/detectors/__init__.py:14-42`.)
