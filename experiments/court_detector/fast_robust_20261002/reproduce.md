# Reproduce the evaluation

Run from the repository root with the NumPy, pandas and OpenCV environment.
The report scores detector commit `17a50b57418ade93c70e89b48117ad535133cf34`.
The production detector is unchanged by this evaluation.

## Inputs

On Carmack, `/scratch/ahalperi/court_det_fix/release_courts_fast_robust/` contains
`cohort.json.gz` and the 86 `videos/*.json.gz` results. The cohort records exact
source-video paths. Copy those inputs to `local_scratch/court_evaluation/` with
`hpcrsync`. Original videos stay on Carmack.

ShuttleSet annotations are tracked at `data/shuttleset/set/`. ShuttleSet22
annotations are on Carmack at
`/scratch/cmarti56/issue106-shuttleset22-data/annotations/set/`; copy that directory
to `local_scratch/court_evaluation/shuttleset22/set/`.

## Analyse

```bash
PYTHONPATH=src ~/.venvs/badminton-cicd/bin/python \
  scripts/evaluate_courts_fast_robust.py analyse \
  --input-root local_scratch/court_evaluation \
  --shuttleset-root data/shuttleset/set \
  --shuttleset22-root local_scratch/court_evaluation/shuttleset22/set \
  --output experiments/court_detector/fast_robust_20261002
```

The script checks that the cohort matches the eligible release manifests and
that scene spans cover each video without gaps. It excludes invalid rally
sequences and writes the metrics and 94 frame requests. The per-scene table
contains reference errors even for non-rally scenes; use `rally_frames`,
`majority_rallies` and `is_dominant_group` to select the report's populations.
Non-rally error values are not accuracy labels.

Reference corners use the inverse of `homography_matrix`. Template coordinates
are x=25..325 for ShuttleSet 1–20, x=27.4..327.6 for later ShuttleSet and all
ShuttleSet22, and y=150..810 throughout. Those are the annotation template's
units, not metres. Predictions and references are compared at 1280×720.

## Fetch frames on Carmack, then render locally

Copy the evaluation script and `render_requests.json.gz` to
`/scratch/ahalperi/court_det_fix/` on Carmack. The following command runs there;
the existing checkout supplies the small geometry modules imported by the script.

```bash
PYTHONPATH=/scratch/ahalperi/court_det_fix/release-courts-checkout/src \
  ~/.venvs/court_det/bin/python \
  /scratch/ahalperi/court_det_fix/evaluate_courts_fast_robust.py fetch-frames \
  --requests /scratch/ahalperi/court_det_fix/render_requests.json.gz \
  --output /scratch/ahalperi/court_det_fix/evaluation_frames_20261002
```

Copy that frame directory locally to `local_scratch/court_evaluation/frames/`.
Then run:

```bash
PYTHONPATH=src ~/.venvs/badminton-cicd/bin/python \
  scripts/evaluate_courts_fast_robust.py render \
  --requests experiments/court_detector/fast_robust_20261002/render_requests.json.gz \
  --frames local_scratch/court_evaluation/frames \
  --output experiments/court_detector/fast_robust_20261002

find experiments/court_detector/fast_robust_20261002 -name '*.png' -print0 \
  | xargs -0 -P 4 -I '{}' pngquant --force --ext .png --speed 3 --nofs 256 -- '{}'
oxipng -o 2 --strip safe -r experiments/court_detector/fast_robust_20261002
```

Frames use OpenCV's zero-based frame index. The extractor checks the next-frame
position after decoding. Overlays use the detector's 40 mm painted-stripe model,
projected from its saved outer court corners. Strokes are 1 pixel wide with no
antialiasing; the dash pattern is 7 pixels on, 5 off. PNG quantisation changes
colours slightly but preserves image size and overlay positions.
