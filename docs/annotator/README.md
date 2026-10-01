# Auto-annotator

The auto-annotator turns shuttle, pose and court evidence into rally bounds,
contact times and player-side assignments. It combines fixed heuristics with
trained tree models. The dataset builder uses this same path before producing
rally records.

Start here to run it. [How it works](how_it_works.md) explains the model and
outputs; [retuning](retuning.md) covers fitting a fresh bundle from labelled
videos.

## Before running

Use Python 3.12 or newer and the dependencies resolved in `uv.lock`. The supported
annotator needs a fresh model bundle fitted in that environment. Old experiment
binaries are historical comparison material; copying them into the model
location does not make them supported models.

A bundle directory contains `models.joblib` and `metadata.json`. Together they
hold the contact tree, sequence refinement trees, review-ranking tree and the
preprocessing settings used during fitting. Loading checks the model schema,
contact feature order and exact scikit-learn version. Refit after changing that
version. Keep both bundle files together.

The next court-based retune is still required. Code cleanup alone does not
establish prediction quality on the new court evidence.

## Dataset builder

Set the model directory in your existing dataset-builder TOML configuration:

```toml
[models]
annotator = "models/annotator"
```

Add this key to the existing `[models]` table, alongside the vision models.
Relative paths resolve from the repository root. The default is
`models/annotator`.

From the repository root, with the project environment active:

```bash
PYTHONPATH=src python -m dataset_builder run \
  --config configs/dataset_builder/trial.toml \
  --run-dir data/dataset-run
```

The builder loads and checks the bundle before running annotation. Its resume
checks include both bundle files, so replacing a bundle invalidates affected
annotation results. It writes final contacts, sides, rally bounds and review
scores into the annotation artefacts used by later stages. The trial
configuration also performs acquisition and vision extraction; it is not a
command for fitting annotator trees.

## Python callers

Applications with prepared vision inputs call
[`run_video`](../../src/annotator/run_video.py). Load the bundle explicitly so the
model location is clear:

```python
from pathlib import Path

from annotator.models import load_models
from annotator.run_video import run_video

models = load_models(Path("models/annotator"))
result = run_video(
    track, bboxes, scores, kps, ndet,
    fps=fps, models=models,
    **court_and_mask_inputs,
)
```

Here the five arrays are frame-aligned shuttle and pose evidence. The keyword
mapping supplies validated court geometry, replay/court masks and landing
inputs; the `run_video` docstring gives their shapes and required fields. The
builder's [`run_full_annotation_stage`](../../src/dataset_builder/vision.py)
shows the complete call from its public artefact loaders. Normal full annotation
loads `models/annotator` from the working directory if `models` is omitted.

Use the bundle's preprocessing settings. Supplying a different `base` config
fails rather than silently changing model inputs. `heuristic_only=True` exists
for feature preparation and comparisons; supported final annotation uses the
mixed model.

## Reading the result

- `spans` are half-open rally bounds: the start frame is included, the end frame
  is excluded
- `filtered_by_rally` holds final contact frames for each rally; `contact_events`
  retains the full stream, including unassigned contacts, with scores and
  individual `Top`/`Bot` side assignments
- `rally_confidence` ranks rallies for review, in span order; it does not change
  contacts or grant automatic approval
- Server, winner, landing and hit-height fields are downstream estimates from
  the final contacts and available geometry. Missing or unresolved values need
  to remain visible to consumers

Use `filtered_by_rally` for contact membership. Do not reconstruct it by
filtering the full event stream against rally bounds.

`Top` is the far court half and `Bot` is the near court half. These are camera
positions, not stable player identities across a change of ends.

## Source map

Read [`run_video.py`](../../src/annotator/run_video.py) for the full call and
[`hybrid.py`](../../src/annotator/hybrid.py) for the learned contact path.

| Location | Responsibility |
| --- | --- |
| [`contacts/`](../../src/annotator/contacts/) | Fixed contact features and base-tree scoring |
| [`rally/`](../../src/annotator/rally/) | Heuristic rally spans and contact evidence |
| [`masks/`](../../src/annotator/masks/) | Exclusion of replay, invalid and off-rally frames |
| [`courts/`](../../src/annotator/courts/) | Scene geometry and court-view evidence |
| [`sequence/`](../../src/annotator/sequence/) | Finite contact-sequence repairs, sides and review ranking |
| [`outcomes/`](../../src/annotator/outcomes/) | Server/winner, landing and hit-height estimates |
| [`training/`](../../src/annotator/training/) | Labelled fitting and held-out evaluation |
| [`models.py`](../../src/annotator/models.py) | Complete bundle save/load contract |

Research experiments are separate from this reading path. For optional context,
[PR 149](https://github.com/ahalp90/badminton_cv_annotator/pull/149) describes the
retained mixed model and
[PR 150](https://github.com/ahalp90/badminton_cv_annotator/pull/150) its court-failure
analysis. Those results do not validate the new court inputs.
