# Video-model contact benchmarks

This benchmark compares pretrained RGB video models with the badminton
annotator and MonoTrack. Models predict contact times and hitters inside
supplied rally windows, using the existing training, validation and test
splits.

## X3D-M experiment

The first experiment adapts Kinetics-400 X3D-M with a three-class classifier:

| Class | Meaning |
|---|---|
| 0 — NO_CONTACT | No contact near the target frame |
| 1 — BOT_CONTACT | Contact by the player at the bottom of the image |
| 2 — TOP_CONTACT | Contact by the player at the top of the image |

Each input contains 16 consecutive frames. Frames `t-8` through `t+7` predict
the target at frame `t`. The full image is resized proportionally and padded
to 256 × 256. Training augmentation varies between clips and remains
consistent across the frames within each clip.

Apply horizontal flip, random square crop and colour augmentation independently
with probability 0.5 each. The crop retains 80–100% of the padded image area
before resizing to 256 × 256. Brightness, contrast and saturation factors range
from 0.8 to 1.2. Apply these transformations on the GPU, consistently across
each clip. Omit MixUp, CutMix and perspective augmentation for the pilot.

Two training targets are compared:

| Scheme | Positive centres | Ignored centres |
|---|---|---|
| A | Within one frame of a contact | Two frames from a contact, unless another contact makes the centre positive |
| B | Within two frames of a contact | None |

Other centres inside the supplied rally windows are background. When both
players have contacts within the positive band, TOP takes precedence,
following HitNet's class-ID ordering. Both schemes use the original event
annotations for evaluation.

Fine-tune the full network from the pretrained weights, beginning with a short
linear learning-rate warm-up. Each epoch visits the eligible training windows
in shuffled order, followed by evaluation on the complete validation set.
Reduce the learning rate tenfold when validation loss plateaus and use early
stopping, retaining the best validation checkpoint. Use cross-entropy without
label smoothing and retain X3D's training-time classifier dropout of 0.5.

Start with batch size 16 and SGD with learning rate 0.02, momentum 0.9 and
weight decay `1e-5`. Warm up from learning rate 0.002; determine the duration
after measuring throughput. Use ReduceLROnPlateau patience 3 and early-stopping
patience 15, with a maximum of 1,000 epochs. Confirm these intervals against
measured epoch duration before training. The first pilot uses one seed and
no class weighting or background resampling; the final comparison uses five
seeds.

## Evaluation

Supplied windows extend one second before the first labelled contact and one
second after the last, using the stored benchmark bounds. Input clips can
read surrounding video frames for context.

At each centre frame, take argmax over the three classes and discard
NO_CONTACT predictions. Within each rally, greedily retain the strongest
player-class probabilities, suppressing candidates within six frames at 30 fps,
scaled to the source frame rate. Equal-score ties favour earlier frames.
Choose the alternating TOP/BOT sequence agreeing with more retained player
predictions; retain their original player labels if the two sequences tie.
These fixed rules follow the annotator's contact suppression and alternation.
No confidence-threshold or spacing search is used.

Select checkpoints by validation loss. Select experiment configurations by
fully correct validation rallies, with correct-player contact F1 breaking
ties, following the [MonoTrack benchmark](../monotrack_benchmark/README.md).
Report contact F1, correct-player F1 and fully correct rallies using the
existing scorer and its frame-rate-scaled timing tolerances.

## References

- [X3D paper](https://arxiv.org/abs/2004.04730) and
  [released X3D-M configuration](https://github.com/facebookresearch/SlowFast/blob/main/configs/Kinetics/X3D_M.yaml)
- [E2E-Spot](https://arxiv.org/abs/2207.10213), including its centre-frame
  video-classification baseline, VC-Spot

## Running the pilot

Install the repository dependencies plus `requirements.txt` in a CUDA-enabled
PyTorch environment, with matching torch and torchvision versions. FFmpeg must
be on `PATH`. Run these commands from the repository root with `PYTHONPATH=src:.`.

The 256×256 inputs follow the [PyTorch Hub inference example](https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/).
The released backbone and pooling remain unchanged. The classification head
returns three logits for cross-entropy; probabilities are computed for decoding.
Normalisation uses RGB mean 0.45 and standard deviation 0.225.

```bash
export PYTHONPATH=src:.
python -m experiments.annotator.video_models_benchmark.prepare \
  --spans /path/to/sset_spans.json.gz \
  --videos /path/to/videos \
  --cache /path/to/rgb256-cache
```

Preparation writes each context-inclusive frame once as padded uint8 RGB.
The raw `.rgb` files support memory mapping and share the operating system's
RAM cache between workers. Completed videos are reused on restart. The same
cache serves both target schemes and every seed. A separate cache made with
`--smoke-rallies 2` supports bounded timing; training rejects that partial cache.
The preparation command currently accepts the frozen development split only.

```bash
python -m experiments.annotator.video_models_benchmark.train time \
  --cache /path/to/rgb256-cache --output /path/to/timing --scheme A
```

Timing includes shuffled loading, host-to-device transfer, GPU augmentation,
forward/backward computation and SGD. Ten disposable batches warm the runtime;
forty measure training throughput. Model weights and batch-normalisation
statistics are restored afterwards. No trained checkpoint is written. A small
cache measures steady-state throughput; full-cache startup and event-scoring
time are additional to its epoch estimate.

`pilot.toml` specifies batch 16, SGD at 0.02, momentum 0.9, weight decay 1e-5,
bfloat16 mixed precision and one seed. Warm-up duration must be supplied
explicitly after measuring runtime. For an agreed number of warm-up updates:

```bash
python -m experiments.annotator.video_models_benchmark.train train \
  --cache /path/to/rgb256-cache --output /path/to/scheme-A-seed-0 \
  --scheme A --warmup-steps "$WARMUP_STEPS"
```

Training visits every eligible centre once per epoch. Validation predicts all
centres in one pass, excludes ignored targets from CE and frame diagnostics,
and reuses those probabilities for both event tolerances. Strict CE
improvements save `best.pt`, validation probabilities and decoded events.
After warm-up, the scheduler and stopping rule use an absolute CE improvement
of 0.0001. The scheduler drops LR after four bad checks; stopping follows 15
bad checks. The maximum is 1,000 epochs. Training restores the best checkpoint.
Choose between A and B using fully correct validation rallies and correct-player
F1, rather than comparing their different frame losses.

To evaluate a saved checkpoint without fitting again:

```bash
python -m experiments.annotator.video_models_benchmark.infer \
  --checkpoint /path/to/scheme-A-seed-0/best.pt \
  --cache /path/to/rgb256-cache --output /path/to/evaluation
```
