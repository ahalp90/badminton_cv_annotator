# Court detector implementation

The [detector overview](../../docs/court_detector/README.md) describes its role
and main stages. The [usage guide](../../docs/court_detector/usage.md) covers
setup, input files, commands and the complete CLI reference. [Sampling and
reuse](../../docs/court_detector/sampling_and_reuse.md) defines the video modes.

This page contains the Python API and code map for development.

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
each step starts its own workers. A line-only court uses `Switches(require_people=False)`, `people=None`
and an empty `(0, 4)` box array. The result is a `CourtResult`: `corners_native_px` or `None`,
`no_court_reason`, `chosen_key`, `paint_score`, `reused_from`, `composition`
and, when timing is on, `stage_seconds`. `CourtResult` keeps internal corner
order. `geometry.normalise_output_corners` converts accepted corners to export
order; `detect_image` and `detect_video` apply it at their output boundaries. `Switches(require_people=False, full_no_people_search=True)` enables the
broader no-player search.

The optional `known_courts=` argument tries earlier courts first.
`reuse.make_known_court()` builds those entries from fully searched results.
`run_video.load_court_tools()` and `run_video.detect_video()` give Python callers
the video runner's behaviour. Their worker pool lives within the
`tools.detector` context manager, as shown in `run_batch`.

### Settings

| `Switches` field | Default | Behaviour |
| --- | --- | --- |
| `require_people` | On | Require players' feet for freshly searched courts. `--no-require-people` turns it off; transferred courts use geometry and camera checks |
| `full_no_people_search` | Off | With `require_people=False`, add all-lines and painted-lines searches to templates. `--full` turns it on; omitted flags or `--fast` leave it off |
| `workers` | 1 (runners: 8) | Worker processes. `--workers` |
| `template_device` | `"cpu"` | Line-template scoring device. `--template-device` |
| `full_score_limit` | `None` | Optional trial: fully score only this many cheaply ranked courts per pair. `run_video --full-score-limit`; `run_image` has no such option |
| `geometry_weight` | 0.1 | Line support's share of the score; paint gets the rest. Python only |
| `upright_camera` | On | Reject courts implying a sideways or upside-down camera. Python only |
| `enforce_scene_consistency` | On | Keep only player samples that look like the analysed frame's shot |
| `self_checks` | On | Check intermediate results |
| `timing` | Off (runners: on) | Report `stage_seconds` |
| `artefacts_dir` | `None` | Write each view's intermediate results here, for debugging |

## Code map

[detect.py](detect.py) defines the order of the steps.
[run_video.py](run_video.py) supplies the scenes.

| File | Responsibility |
| --- | --- |
| [detect.py](detect.py) | `CourtDetector`, `Switches`, `CourtResult` and the order of the steps |
| [run_video.py](run_video.py), [video_inputs.py](video_inputs.py) | Video and batch runner; frame reading and people sources |
| [run_image.py](run_image.py) | Still-image runner |
| [inputs.py](inputs.py), [image_sources.py](image_sources.py) | Input types and image/box provenance |
| [line_sources.py](line_sources.py), [scene_sources.py](scene_sources.py) | DeepLSD or saved lines; PySceneDetect or saved scenes |
| [feet.py](feet.py) | Player sample window, shot check and feet |
| [composition.py](composition.py) | Align scene frames; compare individual courts and the combined refit |
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
Saved-view runners and their frozen-case helpers live under
[experiments/court_detector/saved_views](../../experiments/court_detector/saved_views/).

### Checks after a change

The runner help commands are a quick import and option check. The following tests cover the relevant behaviour:

- [Image runner](../../tests/test_court_detector_run_image.py) and
  [video caller](../../tests/test_court_detector_video_caller.py): inputs, scene
  outcomes and failure handling
- [Composition](../../tests/test_court_detector_composition.py),
  [reuse](../../tests/test_court_detector_reuse.py) and
  [video pooling](../../tests/test_court_view_pool.py): sharing and fallback rules
- [Court evidence](../../tests/test_court_evidence.py): downstream interpretation

Shared-interface or pipeline changes require the project's wider checks.
Hardware-specific tests and live-model runs require the corresponding GPU
environment. The [evaluation record](../../experiments/court_detector/comparisons/development_evaluation.md)
provides saved numerical evidence and outlines for interpreting a change.
