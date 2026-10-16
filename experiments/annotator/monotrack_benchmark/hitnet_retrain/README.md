# Retrain MonoTrack HitNet on ShuttleSet

This trains MonoTrack's 130,179-parameter HitNet with two shuttle inputs:

- **`kept`**: TrackNetV3 tracks with InpaintNet predictions retained.
- **`zeroed`**: the same tracks, with inpainted frames set to zero.

Each input has six training runs: dropout `{0, 0.2}` × L2 `{0, 1e-6, 1e-4}`. All twelve
runs use the same architecture and seed 0. The *kept* and *zeroed* hp configurations are
decided by lowest validation loss.

## Changes from the released setup

These choices adapt MonoTrack to the available data and extraction pipeline:

- **Shuttle tracks:** TrackNetV3 (trialled with and without InpaintNet interpolation)
  replace the MonoTrack author's custom TrackNetV2 build. The *kept*/*zeroed* comparison
  accounts for known issues with regular hallucinated patterns produced by InpaintNet
  which MonoTrack does not explicitly address (it couldn't--given it was published at the
  time of TrackNetV2).
- **Training/validation data:** We use 40 videos and their ground truths from ShuttleSet;
  32 for train, 8 for validation. These splits are deliberately stratified and contain a
  mixture of 25fps and 30fps videos. The original MonoTrack contact labels are
  unavailable.
- **Pose detections:** We use the RTMPose detections used to train our tree annotator
  rather than MMPose extraction, which would take several days of compute time. The
  RTMPose models are at least as accurate. MonoTrack's own player-selection
  and missing-pose filling code processes these detections unchanged.
- **Rally windows:** each ground-truth contact sequence has up to one second of padding
  at each end, clipped at the midpoint between neighbouring rallies to avoid overlap.
  The release operates on rally clips, but its external frame-offsetting process is
  unavailable. The same padded windows are used for training and benchmark runs with
  supplied rally boundaries.
- **Striker labels:** ShuttleSet ground truth replaces MonoTrack's striker estimator and
  manual corrections. Bot=1 and Top=2 keep their physical meaning, including when a poor
  court fit selects the wrong player or half.
- **Court corners:** training follows the released notebook's order: near left, near
  right, far left, far right. The benchmark inference adapter uses that same order to
  correct the train/inference mismatch. The vendored MonoTrack source is unchanged.
- **Optimisation:** the starting learning rate is `1e-3`, after `1e-2` showed poor
  convergence. Each epoch uses approximately one pass through the balanced training
  windows. This roughly tenfold larger dataset already produces far more updates per pass,
  so the release's tenfold update multiplier is removed.
- **Court failures:** rallies where MonoTrack's court detector fails are skipped during
  training and counted by split in `training_data.json.gz`: 50 training rallies and
  nine validation rallies. The saved results contain 52 failed fits, five detector
  crashes and two jobs that reached the existing one-hour court-detection limit.
  We reject a crashed detector's output even if it wrote a corner file before exiting;
  the released pipeline can still try to use that file. All five crash cases reproduce
  a memory error through the released executable with memory-error checking enabled.
  Benchmark scoring retains failed rallies so these errors affect the result.
- **Pose-filling failures:** MonoTrack's missing-pose interpolation crashes when a
  selected player has a detection in only one frame. We skip these rallies during
  training and count them separately from court failures: four training rallies
  containing four contacts, and no validation rallies. Benchmark inference still
  attempts every rally. If it crashes, that rally receives no predicted hits and
  all its ground-truth contacts remain in the score. The benchmark continues with the
  next rally and produces labelled review clips for these failures.
- **Feature storage:** cached features use float32. The resulting rounding differences are
  negligible.

HitNet retains the released two-layer bidirectional GRU, 12-frame inputs,
augmentation, class balancing, batch size 64, early stopping and learning-rate
reduction. The best validation checkpoint is reloaded before temperature
calibration and export.

MonoTrack's shuttle interpolation is off during training. The main benchmark
also disables it at inference and uses `hitnet_zeroed.h5`, with InpaintNet-filled
positions replaced by zeros. Both shuttle inputs were evaluated with inference
interpolation on and off; Appendix A of the reports compares all four settings.
Zeroed input with interpolation off scored highest on ShuttleSet22. MonoTrack's
interpolation is separate from InpaintNet's predictions.

## Setup and run

Use Python 3.11. From this directory, install the dependencies and create a local
configuration:

```bash
python -m pip install -r requirements.txt
cp config.toml config.local.toml
```

Set these paths in `config.local.toml`, which is ignored by Git:

- `stages`: saved ShuttleSet RTMPose detections and TrackNetV3 tracks, including
  the inpainting masks.
- `spans`: the frozen rally windows and contact labels.
- `courts`: MonoTrack's saved court-detector results, named
  `<fixture>_<span:04d>.json.gz`, one result per rally.

Choose a new output directory outside the checkout:

```bash
python -u run_training.py \
  --config config.local.toml \
  --output /path/to/training-output
```

Feature preparation runs first. By default, all twelve trials then run in
parallel, each with two TensorFlow compute threads. Logs, features, checkpoints
and trial summaries go into the output directory.

After all twelve trials complete, `weights/` receives `hitnet_kept.h5` and
`hitnet_zeroed.h5`, their training histories, the grid results and training-data
counts. These files are intended for committing.

If a trial fails, move its incomplete directory aside and rerun that setting
into its original directory. For example, to repeat `kept_d0_l20`:

```bash
run=/path/to/training-output
mv "$run/trials/kept_d0_l20" "$run/trials/kept_d0_l20.failed" &&
python train_hitnet.py \
  --features "$run/features" \
  --arm kept --dropout 0 --l2 0 \
  --output "$run/trials/kept_d0_l20"
```

Once all twelve trials have completed, export the winners without retraining:

```bash
python run_training.py --output /path/to/training-output --select-only
```

## Checks

Install `pytest` and `tqdm` alongside the runtime dependencies, then run:

```bash
python -m pytest tests -q
```

The tests compare player selection, pose filling, features and augmentation with
the released code. They also check rally boundaries, striker labels, court-failure
counts and model selection.

For a short TensorFlow check, build features with `build_training_set.py`, then
run `train_hitnet.py --smoke` with the required `--features`, `--arm`, `--dropout`,
`--l2` and fresh `--output` arguments. This trains two batches, validates,
calibrates and exports a model. Smoke-test models are excluded from selection.
