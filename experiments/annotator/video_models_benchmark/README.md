# Video-model contact benchmarks

This benchmark compares pretrained RGB video models with the badminton
annotator and MonoTrack. Models predict contact times and hitters inside
supplied rally windows, using the existing training, validation and test
splits.

[X3D-M](x3d_results.md) reaches 96.94% contact F1 and 96.39% correct-player
F1 across five seeds. [VideoMAE Base](videomae_results.md) reaches 95.49% and
94.93% across two seeds. These use the winning validation configurations,
raw hitter labels and ±10-frame tolerance. Sliding-window inference is
computationally expensive; all systems receive rally bounds.

Download the seven selected checkpoints from the [weights release](weights/README.md).
The [completed run records](results/stride3_20261105/) include checkpoint
summaries for all 23 runs and detailed predictions for the selected models.
See the [VideoMAE run guide](VIDEOMAE.md) for model settings and timing.

## X3D rerun with stride-three validation

Train all combinations of positive centres ±1/±2 frames, contact shares 25/50%,
and learning rates 0.005/0.02, each with seeds 0 and 1: 16 runs in total.
Validation and test evaluate every third centre at absolute frame offset zero.
Training clips still contain 16 consecutive frames.

Each run completes the initial 30,000 updates, then continues in the same
training process. Check validation every 5,000 further updates and stop after
three consecutive checks without a lower validation loss. Carry the existing
patience count across 30,000 updates. The continuation learning rate cycles
from zero to the original 20,000-update rate and back over 20,000 updates.

Retain the checkpoint with the lowest validation cross-entropy, taking the
earlier checkpoint on a tie. Rank the eight configurations by mean fully
correct validation rallies across their two seeds at the ±10-frame tolerance.
Mean correct-player F1 breaks ties; an exact tie retains grid order. Save the
ranking before evaluating both seeds of the winner on test. Report each seed
and its mean; the two models are evaluated separately.

For this rerun, player assignments always alternate. Choose the order agreeing
with more predicted hitters. If the agreement counts tie, choose the order
with the greater sum of assigned-player probabilities. On an exact probability
tie, preserve the first predicted hitter. Contact times are unchanged. Earlier
runs kept raw player predictions when agreement counts tied.

```bash
python -m experiments.annotator.video_models_benchmark.grid init \
  --output "$RUNS" --cache "$TRAIN_VAL_CACHE" --full-grid --test-cache "$TEST_CACHE"
python -m experiments.annotator.video_models_benchmark.grid run --output "$RUNS"
```

Use `run --resume` to recover an interrupted queue. The full-grid mode includes
continuation and both positive windows; no separate `extend` or `scheme-b`
command is needed. Those commands remain available for older queues.

Keep every validation check under `validation_checks/stepNNNNNN/`: checkpoint
weights, frame probabilities, frame metrics and event scores. Keep these large
artefacts outside Git. `best.pt` and the top-level validation outputs refer to
the lowest-loss check; `last.pt` retains the optimiser and random state for
recovery.

### Five-seed evaluation of the X3D winner

After the original X3D and VideoMAE queues finish, add X3D seeds 2, 3 and 4
using the saved X3D validation winner. Copy its training configuration unchanged
except for seed, train each model from pretrained weights, and test each model's
minimum-validation-loss checkpoint. Keep the original selection fixed.

```bash
python -m experiments.annotator.video_models_benchmark.x3d_seed_queue follow \
  --x3d-output "$X3D_RUNS" --after "$VIDEOMAE_RUNS" --output "$EXTRA_SEED_RUNS"
python -m experiments.annotator.video_models_benchmark.x3d_seed_queue status \
  --output "$EXTRA_SEED_RUNS"
```

Both dependency queues must use launchers that hold `runner.lock` until they
write `queue-exit-code.txt`. The follower waits for those locks, then requires
exit zero, a complete queue and complete selected tests before starting.
Use an outer launcher lock for the follower too. Failed or paused dependencies
stop the follower; fix the dependency and relaunch the same command to resume.

The new `grid.json.gz` freezes the winner and includes `test_runs` paths for
all five seeds: 0/1 from the original grid and 2/3/4 from this follow-up.
Completed training and tests are skipped on restart. Preserve an incomplete
test directory elsewhere before retrying. Use `pause` to stop at a validation
boundary, then `continue` to clear pause markers before relaunching `follow`.
The three added seeds measure the chosen configuration's spread; they do not
participate in another hyperparameter selection.

## Original X3D-M experiment

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

Fine-tune the full network from the pretrained weights for 30,000 updates,
with batch size 16. Warm up linearly from one tenth of the peak learning rate
for 2,000 updates, then use cosine decay. Retain the checkpoint with the lowest
cross-entropy on naturally distributed validation centres at the configured stride.
Use SGD with momentum 0.9 and weight decay `1e-5`, cross-entropy without label
smoothing, and X3D's classifier dropout of 0.5.

The initial grid compares learning rates 0.005 and 0.02 with contact shares of
25% and 50%, using target scheme A. Every training batch contains exactly four
or eight contact centres, respectively. Contact and background streams each
visit a seeded permutation of their centres before independently reshuffling.
TOP and BOT contact centres share one stream. Augmentations remain freshly
randomised on every visit. Each run sees 480,000 centre-labelled clips.

Run all four settings on seed 0, then repeat them on seed 1. This grid informs
the eventual target-scheme and five-seed comparison; it does not choose a winner
automatically. Validation occurs at updates 2,500, 5,000, 10,000, 15,000, 20,000,
25,000 and 30,000. A poor early event score does not automatically terminate a
run. Non-finite loss fails the run and stops the queue for inspection.

## Evaluation

New X3D and VideoMAE runs validate at **stride 3, offset 0**, evaluating absolute
video frames 0, 3, 6… across rally boundaries. Each input still contains 16
consecutive frames. Standalone validation and test inference use the same defaults;
pass `--stride 1` to evaluate every centre.

Training uses `validation_stride = 3` in the run configuration. Set it to 1 for
a new dense-validation run. Resuming or extending a run preserves its saved
validation stride; older runs without that setting retain dense validation.
The original X3D and VideoMAE pilot results used dense validation.

Validation loss and frame metrics use evaluated centres only. Event scores retain
all labelled contacts and rallies. Saved arrays retain one row per original
centre, with `(1, 0, 0)` placeholders for skipped frames. Validation histories and
event records include the stride and centre counts in `sampling`; inference frame
metrics include them too. Sparse arrays support only the evaluated offset.

Supplied windows extend one second before the first labelled contact and one
second after the last, using the stored benchmark bounds. Input clips can
read surrounding video frames for context.

At each evaluated centre frame, take argmax over the three classes and discard
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
For the frozen test split, replace `--spans` with `--test-inputs` pointing to
`experiments/annotator/monotrack_benchmark/benchmark_results/inputs`, use the
ShuttleSet22 video directory and a separate cache. Evaluate selected checkpoints
with `infer --role test`. The one contact with an unknown hitter remains in event
scoring under the shared scorer's rules; its nearby centres are ignored for
frame loss.

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

`pilot.toml` specifies the 30,000-update schedule, batch 16, SGD at 0.02,
50% contact sampling, bfloat16 mixed precision and seed 0:

```bash
python -m experiments.annotator.video_models_benchmark.train train \
  --cache /path/to/rgb256-cache --output /path/to/scheme-A-seed-0 \
  --scheme A
```

Validation predicts all centres in one pass, excludes ignored targets from CE
and frame diagnostics, and reuses those probabilities for both event tolerances.
Strict CE improvements save `best.pt`, validation probabilities and decoded
events. Training restores the best checkpoint after the fixed update budget.
Choose configurations using fully correct validation rallies and correct-player
F1. Use the same rule for the later A/B comparison; their frame losses have
different target definitions.

To evaluate a saved checkpoint without fitting again:

```bash
python -m experiments.annotator.video_models_benchmark.infer \
  --checkpoint /path/to/scheme-A-seed-0/best.pt \
  --cache /path/to/rgb256-cache --output /path/to/evaluation
```

## Running and reviewing the grid

Initialise the eight-run queue once, then launch it from the repository root:

```bash
python -m experiments.annotator.video_models_benchmark.grid init \
  --cache /path/to/rgb256-cache --output /path/to/grid
nohup python -u -m experiments.annotator.video_models_benchmark.grid run \
  --output /path/to/grid > /path/to/grid/queue.log 2>&1 </dev/null &
python -m experiments.annotator.video_models_benchmark.grid status \
  --output /path/to/grid --watch
```

The queue uses one GPU sequentially. Every run keeps its own configuration,
checkpoints, learning history, best validation predictions and event scores.
The status table shows completed runs and provisional results from the current
run. Metrics in that table belong to each run's lowest-CE checkpoint, not its
highest event score. Both event tolerances are in `history.json.gz`.

Measured on an L40, training took 0.209 seconds per batch and complete validation
about 10.4 minutes. This projects about 3 hours per run and 24 hours for the grid,
plus preparation, startup and event scoring. Full-cache loading may change the
observed throughput. The first validation result is expected about 19 minutes
after training starts.

To review a setting before spending the remaining budget, request a pause:

```bash
python -m experiments.annotator.video_models_benchmark.grid pause --output /path/to/grid
```

The active run pauses after saving its next validation checkpoint and the queue
starts no further runs. Check status for `paused` before restarting. To continue:

```bash
python -m experiments.annotator.video_models_benchmark.grid continue --output /path/to/grid
nohup python -u -m experiments.annotator.video_models_benchmark.grid run \
  --output /path/to/grid --resume >> /path/to/grid/queue.log 2>&1 </dev/null &
```

A failed run stops the queue. Inspect its log, fix the cause, then restart with
`run --resume`; successful runs are skipped. Preserve the original queue and
configs when changing scientific settings: use a new grid/run directory so
partial experiments remain interpretable.

## Continue the completed grid

To queue continuation behind the original grid, run this from the same checkout:

```bash
nohup python -u -m experiments.annotator.video_models_benchmark.grid extend \
  --output /path/to/grid >> /path/to/grid/continuation-queue.log 2>&1 </dev/null &
```

The runner waits on the original queue's lock. It starts only after all original
runs complete successfully, and stops if that queue fails or is paused. Results
go into `grid/continuation/`, starting from copies of the original recovery
checkpoints, best models and histories. Original results remain intact.

Each run resumes at 30,000 updates with model, optimiser momentum and RNG state.
Its LR rises smoothly from zero to its original step-20,000 rate at step 40,000,
returns to zero at 50,000, then repeats. Validate every 5,000 updates and stop
after three consecutive checks without a strictly lower validation CE. Carry
the original non-improvement count forward: a best at 25k followed by worse
checks at 30k, 35k and 40k stops at 40k. Improvements reset patience. There is
no requirement to complete a cycle and no fixed end while CE keeps improving.
The best checkpoint and event reports cover both training phases.

The existing `grid status --output /path/to/grid [--watch]` command shows the
waiting runner and automatically switches to continuation results afterwards.
`grid pause` pauses both phases. To restart after a pause, clear markers with
`grid continue`; restart the original queue with `grid run --resume` if it is
unfinished, then run `grid extend` again. It resumes unfinished continuations
and skips completed ones. Operational failures stop the continuation queue.

## Queue scheme B from the completed scheme-A results

After starting the scheme-A continuation runner, queue the wider-label comparison:

```bash
nohup python -u -m experiments.annotator.video_models_benchmark.grid scheme-b \
  --output /path/to/grid >> /path/to/grid/scheme-b-queue.log 2>&1 </dev/null &
```

This waits for all eight scheme-A continuations to finish successfully. For each
run it takes the global lowest-validation-CE checkpoint, including the original
phase, and reads the event scores at ±10 frames (scaled by fps). It ranks the four
configurations by mean fully correct rally percentage across seeds 0 and 1,
breaking ties by mean correct-player F1, then original grid order for exact ties.
The full ranking and selected configurations are saved in `scheme-b/grid.json.gz`.
Selection stays fixed when resuming.
Compare schemes using event scores: A and B validation CE use different target
labels, so their CE values should not rank the label schemes.

The top two configurations each run at seeds 0 and 1: four fresh K400 starts with
scheme B (±2 positive labels), copied hyperparameters and unchanged uniform
sampling across positive clip centres. The positive/background batch fraction
is unchanged. These runs complete their original 30k phase, then automatically
use the same cosine continuation and three-check historical patience. Results
are under `scheme-b/` and `scheme-b/continuation/`; scheme-A results are retained.

The root `status`, `status --watch`, `pause` and `continue` commands follow all
stages. A paused or unsuccessful dependency prevents B training. After a pause
or failure, inspect logs, clear pause markers with `grid continue`, and restart
unfinished A stages as above. Then rerun `grid scheme-b --output /path/to/grid`
with the same detached command. It resumes incomplete B runs and skips completed
ones. Its own lock prevents duplicate selection runners. Start it only after
the A continuation runner has acquired its lock, or after A has completed.

## Progress and recovery

`status.json.gz` updates about every 30 seconds and at phase changes. It shows
update count, training/validation progress, current learning rate and losses.
For one run, use `python -m experiments.annotator.video_models_benchmark.status
--output /path/to/run --watch`.

`training_progress.json.gz` retains the training-loss readings from those
roughly 30-second updates, including during warm-up. Each record stores the
step, learning rate, timestamp, session start and training interval start.
`train_loss` is the mean since that interval began; the interval restarts after
each validation. To recover the mean between two readings in the same interval
and session, difference their cumulative losses (`train_loss` multiplied by
`step - interval_start_step`) and divide by the number of intervening updates.
The file is replaced atomically and retained across resumes. Replayed updates
remain recorded under their new session, so use the latest attempt when plotting
a resumed run. Logging reuses the existing CPU loss readings and adds no model
passes or GPU synchronisation. The first reading follows training; there is no
step-zero loss evaluation. Older runs started before this logger was added have
only the validation-interval averages in `history.json.gz`.

`last.pt` saves the training state at validation boundaries, including optimiser
momentum, update count and random-number state. The sampler regenerates its
seeded index sequence and skips completed updates. An interrupted interval is
replayed from its preceding checkpoint. `best.pt` separately retains the
lowest-validation-loss model. Checkpoints and progress files are written
atomically. Training options must match the saved run when resuming.

The standalone launcher records its PID and exit code, and prevents duplicate
launches in one run directory:

```bash
nohup bash experiments/annotator/video_models_benchmark/launch.sh \
  /path/to/python /path/to/rgb256-cache /path/to/run A 2000 \
  > /path/to/launcher.log 2>&1 </dev/null &
```
