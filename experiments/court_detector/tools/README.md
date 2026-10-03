# Reproducing the court comparisons

The scripts in this folder run the search trial and rebuild its comparisons.
`trial_run_video.py` runs paired detector alternatives; `compare_search.py`
summarises their saved outputs.

The commands below use saved court predictions. They can rebuild the reported
measurements without running court detection again. Only the image-rendering
step needs the source videos.

The commands run from the repository root in the project's Python environment
(with NumPy, pandas and OpenCV). The repository [setup instructions](../../../README.md#running-it)
cover installation. This setup writes new results under `/tmp`:

```bash
BASE=experiments/court_detector
DATASET=data/court_detections/sset_and_sset22/extractions_20261003
WORK_DIR=/tmp/court-evaluation
mkdir -p "$WORK_DIR"
```

## Compare the released courts with the original courts

The first command compares the released predictions with the official court
labels and writes a row for each scene, rally and video. The second summarises
the camera view used during each rally. The third compares those results with
the original detections, before the sharing fix.

```bash
PYTHONPATH=.:src python scripts/evaluate_courts_fast_robust.py analyse \
  --input-root "$DATASET" \
  --shuttleset-root data/shuttleset/set \
  --shuttleset22-root data/shuttleset22/set \
  --output "$WORK_DIR/released"

PYTHONPATH=.:src python scripts/summarise_court_rally_views.py \
  --input "$WORK_DIR/released" --output "$WORK_DIR/released/rally_views"

PYTHONPATH=.:src python scripts/compare_court_sharing.py \
  --original "$BASE/evidence/baseline" --patched "$WORK_DIR/released"
```

The before/after results are the three `comparison_*.csv.gz` files under
`$WORK_DIR/released`. The saved originals are in
[evidence/release/](../evidence/release/).

The official court labels are in `data/shuttleset/set/homography.csv` and
`data/shuttleset22/set/homography.csv.gz`. Each labels one camera view per match,
not every scene. The evaluator converts the labels to four corners and compares
both labels and predictions at 1280 × 720 resolution. It handles the two
datasets' different court-template coordinates.

The original 86-video predictions are also available under `evidence/inputs/`. Using
that directory as `--input-root` rebuilds the original evaluation tables.

## Compare the three search methods

This command rebuilds the eight-video trial tables from the predictions in
`evidence/inputs/player_tiebreak_results/`. It produces per-video and per-rally results,
method summaries and before/after differences.

```bash
PYTHONPATH=.:src python "$BASE/tools/compare_search.py" \
  --output "$WORK_DIR/search-trial"
```

The [trial report](../comparisons/search.md) explains what each method changed.

## Draw courts on the video frames

The [saved gallery](../comparisons/gallery.md) contains 94 examples:
one typical detected court from each of the 86 videos, plus eight scenes with
large differences from the official main-camera corners. The frame numbers
and corners for those images are in
[`render_requests.json.gz`](../evidence/release/render_requests.json.gz).

This command prepares those 94 images plus 21 stills covering three selected
rallies. It writes the list of video frames and court corners to draw.

```bash
PYTHONPATH=.:src python scripts/plan_court_review_stills.py \
  --input "$WORK_DIR/released" --dataset "$DATASET" \
  --output "$WORK_DIR/image-plan"

export REQUESTS="$WORK_DIR/image-plan/render_requests.json.gz"
```

The list initially contains video filenames. The following step adds the
location of the source videos. `VIDEO_DIR` is the directory containing those
files on the machine doing the rendering.

```bash
export VIDEO_DIR=/path/to/source/videos
python - <<'PY'
import gzip
import json
import os
from pathlib import Path

path = Path(os.environ["REQUESTS"])
with gzip.open(path, "rt") as handle:
    frames = json.load(handle)
for frame in frames:
    frame["source_video"] = str(Path(os.environ["VIDEO_DIR"]) / frame["source_video"])
with gzip.open(path, "wt") as handle:
    json.dump(frames, handle)
PY

PYTHONPATH=.:src python scripts/evaluate_courts_fast_robust.py fetch-frames \
  --requests "$REQUESTS" --output "$WORK_DIR/frames"

PYTHONPATH=.:src python scripts/evaluate_courts_fast_robust.py render \
  --requests "$REQUESTS" --frames "$WORK_DIR/frames" \
  --output "$WORK_DIR/images"
```

`$WORK_DIR/images` contains the annotated PNGs. The rally stills show one frame
from each scene crossed by the rally, including scenes with no detected court.
Those frames have no court outline. Frame numbers start at zero.

The outline follows the edges of the 40 mm court stripes, with thin dashed
lines. Published PNGs are compressed with pngquant, then oxipng.

## Using the predictions directly

The [dataset README](../../../data/court_detections/sset_and_sset22/extractions_20261003/README.md)
contains a loading example and field descriptions. Its final corners already
follow the viewer-facing order, with the upper baseline first.

## Search-trial images and clips

[`render_court_trial_rallies.py`](../../../scripts/render_court_trial_rallies.py)
prepares the frame list, extracts the source images and renders the courts.
Its `plan`, `fetch` and `render` commands show their arguments with `--help`.
The [manifest](../evidence/search/rally_review/manifest.json.gz) records which frames and courts were used.

A typical sequence, from the repository root, is:

```bash
PYTHONPATH=.:src python scripts/render_court_trial_rallies.py plan \
  --output /tmp/court-trial-plan.json.gz
```

The plan's `source_video` fields initially contain filenames. After those fields
contain the local paths to the source videos, the following commands extract
and annotate the frames:

```bash
PYTHONPATH=.:src python scripts/render_court_trial_rallies.py fetch \
  --plan /tmp/court-trial-plan.json.gz --output /tmp/court-trial-frames
PYTHONPATH=.:src python scripts/render_court_trial_rallies.py render \
  --plan /tmp/court-trial-plan.json.gz --fetched /tmp/court-trial-frames \
  --output /tmp/court-trial-images
```

The [general rendering section](#draw-courts-on-the-video-frames)
also shows how to add local video paths to a frame list. Published PNGs are
compressed with pngquant, then oxipng.
