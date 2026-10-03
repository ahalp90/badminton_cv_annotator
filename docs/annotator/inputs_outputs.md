# Inputs and outputs

The standalone annotator reads one video's saved court, pose and shuttle data
in the dataset builder's formats and writes four annotation files. This page
specifies those files, their array shapes and the fields in the saved result.

**Contents**

| Inputs | Model and outputs |
| --- | --- |
| [Directory layout](#input-directory-layout) | [Model directory](#model-directory) |
| [Video metadata](#video-metadata) | [Output files](#output-files) |
| [Shuttle files](#shuttle-files) | [AnnotatorResult fields](#annotatorresult-fields) |
| [Pose files](#pose-files) | [Direct Python call](#direct-python-call) |
| [Court files](#court-files) |  |

## Input directory layout

For `--run-dir data/dataset-run --video-id match-name`, the files are:

```text
data/dataset-run/
└── stages/
    ├── metadata/
    │   └── match-name/
    │       └── video_metadata.json.gz
    ├── shuttle/
    │   └── match-name/
    │       ├── shuttle_track.npy.xz
    │       ├── shuttle_guard_codes.npy.xz
    │       └── *_inpaint_mask.json.gz
    ├── pose/
    │   └── match-name/
    │       ├── pose_kps.npy.xz
    │       ├── pose_bboxes.npy.xz
    │       ├── pose_scores.npy.xz
    │       ├── pose_kp_scores.npy.xz
    │       └── pose_ndet.npy.xz
    └── court/
        └── match-name/
            ├── court_evidence.json.gz
            ├── court_keep_vote.npy.xz
            └── court_present.npy.xz
```

The shuttle directory must contain exactly one `*_inpaint_mask.json.gz` file.

## Video metadata

`video_metadata.json.gz` contains:

- the original absolute source path;
- exact frame rate;
- frame count;
- width and height;
- sample aspect ratio.

The annotation command uses the numeric metadata but does not reopen the source video.

All other arrays use the same zero-based frame timeline and must have the recorded frame count.

## Shuttle files

### `shuttle_track.npy.xz`

Shape:

```text
(t, 3)
```

Columns:

```text
[x_norm, y_norm, visibility]
```

`x_norm` and `y_norm` are image coordinates normalised to `[0, 1]` on visible frames. `visibility == 1` means the shuttle is visible.

### `shuttle_guard_codes.npy.xz`

Shape:

```text
(t,)
```

These codes mark shuttle positions considered unreliable because of repeated or fabricated-looking track patterns. The standalone command recomputes the codes from `shuttle_track.npy.xz` and requires an exact match with the saved file.

The default annotator settings flag codes `1`, `2` and `3` as unreliable. Flagged positions are left out of the slow-motion speed estimate and are not used for landings. They are removed from contact candidates only when the model directory turns on the optional rule described in [Fixed heuristics](heuristics.md#optional-rule-for-guarded-candidates).

### `*_inpaint_mask.json.gz`

This sidecar records which shuttle positions came from inpainting, along with the extraction information needed to rebuild the boolean fill mask used by annotation.

## Pose files

`load_pose_arrays()` reads five frame-aligned arrays:

| File | Shape | Meaning |
| --- | --- | --- |
| `pose_kps.npy.xz` | `(t, n_max, 17, 2)` | COCO-style keypoint coordinates |
| `pose_bboxes.npy.xz` | `(t, n_max, 4)` | detection boxes in `xyxy` form |
| `pose_scores.npy.xz` | `(t, n_max)` | pose detection scores |
| `pose_kp_scores.npy.xz` | `(t, n_max, 17)` | per-keypoint confidence saved by the pose stage |
| `pose_ndet.npy.xz` | `(t,)` | number of stored detections on each frame |

Contact and side assignment mainly use boxes, keypoint coordinates, detection scores and detection counts. The keypoint-score array remains part of the saved pose stage even though the main annotator path does not directly depend on it.

## Court files

### `court_evidence.json.gz`

Current schema:

```text
court-evidence/0.2
```

It contains:

- camera-cut intervals;
- court geometry used by the annotator;
- per-scene records.

This geometry includes scene homographies, net/court measurements and the information used to assign player poses to court halves and estimate landings.

### `court_keep_vote.npy.xz`

A frame-aligned boolean array used by court/composition logic.

### `court_present.npy.xz`

A frame-aligned boolean array marking frames where usable court geometry is present.

The court loader checks that scene intervals cover the full video timeline and that both arrays match the metadata frame count.

## Model directory

The `--models` argument points to a directory:

```text
models/annotator/
├── models.joblib
└── metadata.json
```

Loading checks:

- schema `annotator-models/1`;
- exact scikit-learn version;
- exact contact feature order;
- that `models.joblib` contains an `AnnotatorModels` object;
- that the side-geometry mode in the object and metadata agree.

`models.joblib` is created with `joblib.dump()`. It is a Joblib-serialised `AnnotatorModels` object containing the fitted classifiers and their stored settings. It also stores the rough-rally/mask settings and the contact-selection settings used with the fitted trees: the score cutoff and the optional rule for guarded candidates.

## Output files

A full annotation run writes:

```text
<output-dir>/
├── annotator_result.json.gz
├── raw_replay_mask.npy.xz
├── definitive_exclusion_mask.npy.xz
└── shuttle_quality.json.gz
```

### `annotator_result.json.gz`

Current schema:

```text
annotator-result/0.2
```

Top-level shape:

```json
{
  "schema": "annotator-result/0.2",
  "video_id": "match-name",
  "result": {
    "...": "every AnnotatorResult field"
  }
}
```

The serialiser writes every field from `AnnotatorResult`, so the saved JSON and the Python result type stay in step when fields are added.

### `raw_replay_mask.npy.xz`

The first replay/off-rally mask built for the run. In the dataset-builder path it combines sustained court absence with slow-motion-like shuttle movement. A `True` frame is withheld from normal rally/contact evidence. Short mask runs have not yet been removed at this point.

### `definitive_exclusion_mask.npy.xz`

The final boolean mask used by annotation. It starts from the replay/off-rally mask, removes very short flagged runs, then adds frames outside usable court geometry. The full reason for each mask signal is described in [Fixed heuristics](heuristics.md).

### `shuttle_quality.json.gz`

Counts and summary information for shuttle visibility, inpainting and guard codes. Guard codes describe the reliability of the shuttle track itself rather than masking the whole video frame. The summary also records which codes the model settings flag as unreliable; the default is `1`, `2` and `3`. [Fixed heuristics](heuristics.md) explains how those codes are produced, how each stage uses flagged positions, and why they are kept separate from the frame mask.

## `AnnotatorResult` fields

| Field | Meaning | Common use |
| --- | --- | --- |
| `spans` | Final rally intervals as `(start, end)` | Rally timing |
| `contacts` | Final contact rows | Compatibility with existing callers |
| `filtered_contacts` | Final accepted contact rows | Contact-level processing |
| `filtered_by_rally` | `rally_id -> [contact_frame, ...]` | Final contact membership for each rally |
| `striker_halves` | Court half of the final contact in each rally | Side/outcome logic |
| `n_strokes_list` | Number of final contacts in each rally | Downstream features |
| `next_servers` | Predicted next-server court half for each rally | Winner logic |
| `fitted_first_all` | Predicted first-contact/server half for each rally | Serve information |
| `verdict_rows` | Combined point-outcome records for rallies with a resolved striker | Outcome export/inspection |
| `landings` | Landing estimate or `None` for each eligible rally | Outcome analysis |
| `geometric_verdict_rows` | Geometry-only point-outcome diagnostics | Debugging winner logic |
| `hit_height_by_frame` | Contact frame -> hit-height code | Downstream export/features |
| `hit_height_failures` | Explicit hit-height failures with rally/stroke/frame/error | Diagnostics |
| `contact_events` | Full final contact stream, including events outside rally spans | Model diagnostics |
| `rally_confidence` | One review-ordering score per rally | Review prioritisation |

### Contact membership

`filtered_by_rally` contains the final contacts assigned to each rally.
`contact_events` contains the complete final stream, including hits outside
all rally spans. Filtering that stream by the clip boundaries can therefore
give a different result from the saved membership.

### Historical contact field names

In the fitted-model path, `contacts` and `filtered_contacts` contain the same
chosen rows. The dataset builder also saves those final contacts under the
older name `raw_candidates`. Despite the name, that field contains the selected
hits. This explains why older benchmark reports show identical raw and final
contact metrics for these runs; use the final metrics.

### `Top` and `Bot`

These labels refer to physical court halves in the image:

- `Top` — far half of the court;
- `Bot` — near half of the court.

They are not permanent player identities. When players change ends, the labels remain attached to the court halves.

### Missing outcome values

Some outcome fields can be unresolved even when the contact sequence is usable. For example, unreliable shuttle data at the last hit leaves the landing empty without invalidating the contact frame itself. A shuttle that is not visible on the contact frame leaves that hit height unresolved and adds a row to `hit_height_failures`.

## Direct Python call

For callers that already hold the arrays in memory, the main function is `run_video()`:

```python
from pathlib import Path

from annotator.models import load_models
from annotator.run_video import run_video

models = load_models(Path("models/annotator"))
result = run_video(
    track,
    bboxes,
    scores,
    kps,
    ndet,
    fps=fps,
    models=models,
    **court_and_mask_inputs,
)
```

The five positional arrays are the frame-aligned shuttle track and pose detections. The keyword mapping supplies the court geometry, the replay/court masks and the landing inputs. The `run_video()` docstring lists the full in-memory argument shapes. `dataset_builder.vision.run_full_annotation_stage()` shows how the saved stage files are assembled into those arguments.

A few behaviours are worth knowing before calling it directly:

- When `models` is omitted, full annotation loads `models/annotator` relative to the working directory. Passing the loaded models makes the location explicit.
- Annotation uses the preprocessing settings stored in the model directory. Passing a different `base` config raises an error rather than quietly changing the model inputs.
- `heuristic_only=True` runs the rules without the fitted trees. It exists for feature preparation and comparisons. Normal annotation uses the fitted models.
