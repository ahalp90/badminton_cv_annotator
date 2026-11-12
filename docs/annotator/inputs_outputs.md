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
            └── court_present.npy.xz
```

The annotation command needs exactly one `*_inpaint_mask.json.gz` file in the
shuttle directory. Training uses the other saved inputs and does not need this
fill-mask sidecar.

## Video metadata

`video_metadata.json.gz` contains:

- the original absolute source path;
- exact frame rate;
- frame count;
- width and height;
- sample aspect ratio.

The annotation command uses the numeric metadata but does not reopen the source video.

All other arrays use the same zero-based frame timeline and must have the
recorded frame count. In the shapes below, `t` is the number of video frames.

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

The default annotator settings flag codes `1`, `2`, `3` and `4` as unreliable. Flagged positions are left out of the slow-motion speed estimate and are not used for landings. Player-availability features remain available to the contact tree when it scores candidate frames.

### `*_inpaint_mask.json.gz`

This file records which shuttle positions were filled in by the inpainting
stage. It also stores the extraction information needed to rebuild the boolean
mask that identifies those positions during annotation.

## Pose files

`load_pose_arrays()` reads five arrays, with one entry per video frame.
`n_max` is the maximum number of detection slots stored per frame:

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
court-evidence/0.6
```

It contains:

- camera-cut intervals;
- court geometry used by the annotator;
- per-scene records.

Save the detected scenes’ court corners and video resolution. Annotation uses
these to calculate each scene’s image-to-court transform, net position and
landing error band.

### `court_present.npy.xz`

A frame-aligned boolean array marking frames where usable court geometry is present.

The court loader checks that scene intervals cover the full video timeline and that the mask matches the metadata frame count.

## Model directory

The `--models` argument points to a directory:

```text
runtime/deployed/annotator/
├── models.joblib
└── metadata.json
```

Loading checks:

- schema `annotator-four-models/1`;
- exact contact, rally and review feature order;
- proposal budgets for both rally selectors and the fitted review model;
- that `models.joblib` contains an `AnnotatorModels` object.

`models.joblib` is created with `joblib.dump()`. It is a Joblib-serialised `AnnotatorModels` object containing the fitted classifiers and their stored settings. It also stores the rough-rally/mask settings and the contact probability cutoff used with the fitted trees.

Library versions are recorded in the metadata for reference. A version difference
alone does not prevent loading.

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
annotator-result/0.8
```

Top-level shape:

```json
{
  "schema": "annotator-result/0.8",
  "video_id": "match-name",
  "result": {
    "...": "every AnnotatorResult field"
  }
}
```

The serialiser writes every field from `AnnotatorResult`, so the saved JSON and the Python result type stay in step when fields are added.

Each rally outcome stores `winner`, `winner_source`, landing evidence and
landing-margin flags. `conservative_landing_winner` withholds image-border and
exact-line calls. `search_shortened_by_exclusion` is true when an excluded frame
stops the landing search before any other boundary would stop it.

In Python, `CourtHalf.TOP` and `CourtHalf.BOT` identify the camera-relative court
halves; `WinnerSource` identifies the observation or rule used to choose the
winner. Their saved values remain `Top`/`Bot` and `next_server`,
`landing_geometry` or `net_rule`. Rally-record exports give `won`/`lost` relative to the final striker.

### `raw_replay_mask.npy.xz`

The first replay/off-rally mask built for the run. In the dataset-builder path it combines sustained court absence with slow-motion-like shuttle movement. A `True` frame is withheld from normal rally/contact evidence. Short mask runs have not yet been removed at this point.

### `definitive_exclusion_mask.npy.xz`

The final boolean mask used by annotation. It starts from the replay/off-rally mask, removes very short flagged runs, then adds frames outside usable court geometry. The full reason for each mask signal is described in [Fixed heuristics](heuristics.md).

### `shuttle_quality.json.gz`

Counts and summary information for shuttle visibility, inpainting and guard codes. Guard codes describe the reliability of the shuttle track itself rather than masking the whole video frame. The summary also records which codes the model settings flag as unreliable; the default is `1`, `2` and `3`. [Fixed heuristics](heuristics.md) explains how those codes are produced, how each stage uses flagged positions, and why they are kept separate from the frame mask.

## `AnnotatorResult` fields

The result keeps the finished rally objects produced by contact selection. Rally IDs
are their positions in `rallies`; each object contains its bounds, selected contact
events and fitted final striker. The dataset builder derives export columns from
these objects.

| Field | Meaning | Common use |
| --- | --- | --- |
| `rallies` | Finished rallies, with bounds, contact events and fitted final striker | Rally timing, contact membership and server estimates |
| `outcomes` | Rally ID -> winner estimate, landing and geometric comparison | Outcome export and inspection |
| `hit_heights` | Contact frame -> ShuttleSet hit-height code | Contact export and features |
| `unavailable_hit_height_reasons` | Contact frame -> reason its height could not be measured | Missing-evidence diagnostics |
| `contact_events` | Full learned contact stream, including events outside rallies | Model diagnostics |
| `heuristic_candidates` | Raw heuristic candidates with wrist-distance and proximity-suppression flags; empty in learned mode | Heuristic rejection analysis |
| `rally_confidence` | One review-ordering score per rally; empty in heuristic mode | Review prioritisation |

### Contact membership

Use `result.rallies[rally_id].sequence.events` for the contacts assigned to a rally.
Each event contains its frame, model probability and court-half assignment. When
the alternating-side fit is tied, it uses summed wrist distance, then the first
known guess if the distances cannot separate the orders. When no guess is known,
all players stay unknown. Unknown probabilities are NaN in memory and JSON null when saved. `contact_events` also
includes model contacts outside all rallies, so filtering that stream by the final
rally bounds can give different membership.

The dataset builder derives stroke counts, fitted servers and next servers from
these rallies. It writes the existing rally-record columns, including accepted
contact sides. During learned inference the `raw_candidates` record field contains the
selected contact frames with unset heuristic flags. Reconstructing a result
from saved rally records retains their raw candidate rows for scoring; those
records do not identify which inference mode produced them. In heuristic mode it
retains all raw candidates, including those rejected by wrist distance, proximity suppression or excluded footage.

### `Top` and `Bot`

These labels refer to physical court halves in the image:

- `Top` — far half of the court;
- `Bot` — near half of the court.

They are not permanent player identities. When players change ends, the labels remain attached to the court halves.

### Missing outcome values

Some outcome fields can be unresolved even when the contact sequence is usable. For example, unreliable shuttle data at the last hit leaves the landing empty without invalidating the contact frame itself. An invisible shuttle or missing scene geometry at a contact frame leaves its hit height unresolved and adds its frame and reason to `unavailable_hit_height_reasons`.

## Direct Python call

For callers that already hold the arrays in memory, the main function is `annotate_video()`:

```python
from annotator.models import DEFAULT_MODEL_DIRECTORY, load_models
from annotator.annotate_video import annotate_video

models = load_models(DEFAULT_MODEL_DIRECTORY)
annotation = annotate_video(
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

`annotation.result` contains the finished annotations. Both exclusion masks
are returned alongside it. Pass `keep_working_data=True` to also retain prepared
inputs and model scores.

The five positional arrays are the frame-aligned shuttle track and pose detections. The keyword mapping supplies the court geometry, the replay/court masks and the landing inputs. The `annotate_video()` docstring lists the full in-memory argument shapes. `dataset_builder.vision.run_full_annotation_stage()` shows how the saved stage files are assembled into those arguments.

The direct call uses the model directory and settings as follows:

- When `models` is omitted, full annotation loads the tracked bundle named by `DEFAULT_MODEL_DIRECTORY`. Passing loaded models selects another fitted bundle.
- Annotation uses the preprocessing settings stored in the model directory. Passing a different `base` config raises an error rather than quietly changing the model inputs.
- `heuristic_only=True` runs the rules without the fitted trees. It supports full rule-based annotation and heuristic experiments. Pass `base` to choose its preprocessing settings.

`prepare_annotation_inputs()` in [prepare_annotation_inputs.py](../../src/annotator/inference/prepare_annotation_inputs.py)
accepts the same five arrays plus settings scaled to the video frame rate and
the court, mask and scene inputs. Training and rally-model experiments call it directly. Its
`PreparedAnnotationInputs` retains the original arrays, tracked players, wrist gaps, initial rallies,
all heuristic contact rows, exclusion masks and rejected-shuttle-frame mask.
It does not calculate final contact order, landing, winner or hit height.
