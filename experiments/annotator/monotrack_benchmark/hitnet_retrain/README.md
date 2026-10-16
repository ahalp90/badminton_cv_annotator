# Retrain MonoTrack HitNet on ShuttleSet

This trains two HitNet variants using MonoTrack's own player selection: TrackNetV3
with inpainting retained (`kept`), and the same tracks with the inpainted frames
zeroed (`zeroed`). Each variant has six dropout/L2 settings. The lowest validation
loss selects its exported model. Test data do not participate in training or
checkpoint selection.

## Setup and run

Use Python 3.11 and install `requirements.txt`. TensorFlow 2.15 preserves the
released Keras architecture and HDF5 model format. Run from this directory:

```bash
python -m pip install -r requirements.txt
cp config.toml config.local.toml
```

Edit the three input paths in `config.local.toml` (ignored by Git):

- `stages`: saved ShuttleSet RTMPose detections and TrackNetV3 tracks, including
  each track's inpainting mask.
- `spans`: the frozen rally windows and contact labels. Windows include the
  agreed one-second padding and use half-open frame bounds.
- `courts`: MonoTrack's saved court-detector results, one result per rally.

Keep bulky output outside the checkout. The default runs all twelve trials
concurrently, with two TensorFlow compute threads per trial:

```bash
python -u run_training.py \
  --config config.local.toml \
  --output /path/to/fresh-training-output
```

The output directory must be new. Feature preparation runs first. Each trial
then writes its own log, training history, best checkpoint and summary. A failed
trial prevents winner export. Completed trials remain available: rerun a failed
setting with `train_hitnet.py` into a fresh trial directory, then use
`run_training.py --output /path/to/training-output --select-only` after all
twelve settings have completed.

`weights/` receives the two winning HDF5 models, their histories, the complete
grid summaries and training-data counts. These are the small results intended
for committing. Machine-specific configuration stays in the external run
directory.

## Training choices

- The committed split fixes 32 training and eight validation videos. The builder
  checks the supplied windows against that split.
- RTMPose provides detections. MonoTrack's released pose and court code selects
  the highest-confidence eligible player in each half, independently per frame.
  Missing poses are filled within each rally as in the release.
- Ground-truth striker labels retain their physical meaning: Bot=1, Top=2.
  Incorrect player or court-half selections remain in the training inputs.
- Court-detector failures are skipped during training and counted by split in
  `training_data.json.gz`. Missing or unfinished court results fail the build.
  Benchmark inference must retain failed rallies in its denominator.
- Shuttle interpolation is off during training. The benchmark will compare
  released inference interpolation with inference interpolation disabled.
- Features use the corrected, common training/inference court-corner order.
  Each 12-frame window contains shuttle coordinates, two 17-keypoint poses and
  four court corners per frame. Cached values are float32.
- The released two-layer bidirectional GRU, augmentation, class balancing,
  batch size 64, early stopping and learning-rate reduction are retained.
  Initial learning rate is 0.001. The released tenfold updates-per-epoch
  multiplier is removed; floor division still determines steps per epoch.
- Both arms use seed 0 and dropout `{0, 0.2}` × L2 `{0, 1e-6, 1e-4}`. The best
  validation checkpoint is reloaded before temperature calibration and export.

## Checks

Install `pytest` and `tqdm` alongside the runtime requirements, then run:

```bash
python -m pytest tests -q
```

The tests compare player selection and filling with the released file processor,
check feature and augmentation parity, preserve rally boundaries and GT labels,
count court failures, and require a complete grid before selecting winners.

For a TensorFlow smoke test, first build features with `build_training_set.py`,
then run a single setting with `train_hitnet.py --smoke` and the required
`--features`, `--arm`, `--dropout`, `--l2` and fresh `--output` arguments. This runs
two batches, validates, calibrates and exports a model. Smoke results cannot be
selected as winners.
