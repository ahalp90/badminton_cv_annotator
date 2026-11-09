# VideoMAE contact benchmark

Fine-tune VideoMAE V1 Base from supervised Kinetics-400 weights using the shared
RGB cache, contact sampler and event scorer. The pilot compares peak learning
rates `3.125e-5` and `1e-4`, each at seeds 0 and 1. Each run starts afresh.

## Follow the X3D validation winner

Run four fresh models: learning rates `3.125e-5` and `1e-4`, each at seeds 0 and 1.
Inherit both the contact/background sampling ratio and the positive-centre
window from the X3D validation winner. A 25% contact share means 75% background;
a 50% contact share means 50% background. The positive window is ±1 or ±2 frames.

```bash
python -u -m experiments.annotator.video_models_benchmark.videomae_queue follow-x3d \
  --x3d-output "$X3D_RUNS" --output "$VIDEOMAE_RUNS"
```

The follow-up waits for X3D training and its selected test runs to finish.
It reads the saved validation ranking, copies the winning sampling and label
settings into all four VideoMAE configurations, and uses the same caches.
An interrupted or failed X3D queue prevents VideoMAE from starting. Repeating
`follow-x3d` resumes the VideoMAE queue with its original inherited settings.

Each run includes the initial 30,000 updates and continuation to three
consecutive validation checks without a lower loss. There is no new training
cap. Validation and test use native stride 3, offset 0, with forced alternation:
choose the order matching more predicted hitters, then break count ties by
summed player probability and exact probability ties by the first predicted
hitter. Keep every validation check's weights, probabilities and metrics.

Select each checkpoint by minimum validation cross-entropy, taking the earlier
check on a tie. Rank learning rates by mean fully correct validation rallies
across both seeds at the ±10-frame tolerance, then mean correct-player F1;
an exact tie chooses the lower learning rate. Save this ranking before testing
both seeds of the winner. Test scores do not affect either selection.

## Original pilot settings

Training labels are positive within **±1 frame of contact**. Centres two frames
from contact are ignored unless another contact makes them positive. Half of
sampled clips come from positive centres. The existing CLI calls these labels
`--scheme A`; event scoring always uses the original annotations.

The model receives 16 consecutive frames, centred at `t-8` through `t+7`.
The existing letterboxed 256px RGB cache supplies the clips. Shared augmentation
runs before resizing to 224px and normalising each RGB channel with the
checkpoint's mean `(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`. Validation uses deterministic preprocessing.

AdamW uses weight decay 0.05, excluding biases and normalisation parameters.
The head receives the configured learning rate, the last transformer block
receives 0.75 times that rate, and each earlier block receives another factor
of 0.75. Embeddings receive `0.75^13` times the head rate. Stochastic depth
increases from zero in the first block to 0.2 in the last; evaluation uses
all branches. The new head uses trunc-normal weights with standard deviation
0.02, scaled by 0.001, following the
[authors' fine-tuning recipe](https://github.com/MCG-NJU/VideoMAE/blob/main/run_class_finetuning.py).
There is no gradient clipping or additional head dropout.

The pinned [pretrained checkpoint](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics/tree/488eb9a0565f257b32866000305c8178965eb9f6)
uses older query/value bias names. The loader translates them, retains the
original absence of key biases, and requires a complete weight load before
replacing the 400-class head. Recovery and inference reconstruct the model
offline from saved settings and weights. Transformers is pinned in
`requirements.txt` because the stochastic-depth placement depends on its
residual-branch structure.

## Batch size and timing

The pilot uses batch 16. On an NVIDIA L40 with BF16 and 12 loading workers,
batch 16 processed more clips per second than batch 32:

| Batch | GPU-only clips/s | Clips/s including loading | Peak training memory |
|---|---:|---:|---:|
| 16 | 62.15 | 59.06 | 10.41 GiB |
| 32 | 57.04 | 56.27 | 19.66 GiB |

The [saved measurements](results/videomae_timing/) include the settings and timings.

At batch 16, full validation over 180,179 clip centres took 16.6 minutes,
with another 5 seconds for event scoring. The initial 30,000 updates and seven
validation checks take about 4.2 hours per run, before continuation.
These measurements used dense validation. New runs use stride 3, offset 0,
which evaluates about one third of the validation centres.

Increasing batch size while retaining 30,000 updates increases the number of sampled
clips. When changing batch size, choose the total training exposure, learning
rate and schedule together.

Run commands from the repository root with `PYTHONPATH=src:.`. Use the full
train/validation cache to measure complete validation and estimate pilot time.
A bounded smoke cache is also accepted for timing, but yields no full-run
estimate. Timing performs temporary updates and restores model weights.

```bash
python -m experiments.annotator.video_models_benchmark.train time \
  --config experiments/annotator/video_models_benchmark/videomae.toml \
  --cache /path/to/rgb256-cache --output /path/to/timing-batch16 \
  --batch-size 16 --workers 12
```

Compare batches 16, 32 and 64 where memory permits. Use a separate output
directory for each measurement. Timing reports GPU-only updates with a batch
already on the device, updates including loading and transfer, peak allocated
GPU memory, complete validation and event-scoring time. Each mode measures 200
updates after 10 warm-up updates. A large gap between the
two update times points to data feeding; compare worker counts to test that.
The duration estimate covers the initial training budget and validation.
Continuation, startup and checkpoint writes add time.
Compare clips per second when choosing batch size. The hour estimate uses the
configured update count, so changing only `--batch-size` changes clip exposure.

If a desired effective batch does not fit, `--microbatch-size` splits each
sampled batch across several forward/backward passes. Loss is averaged over
all clips and the optimiser updates once. Validation also uses that smaller
batch size. Accumulation retains the full update's 50% positive sampling;
individual microbatches need not have equal class proportions.

## Run the four configurations

The pilot configuration uses batch 16, 30,000 updates, 2,000 warm-up updates and
validation at 2.5k, 5k, then every 5k through 30k. The learning rate warms up
from one tenth of its peak, then decays to zero. Continued training uses the
shared 20k-update cosine cycle and stops after three validation checks without
lower cross-entropy, carrying forward earlier patience. There is no fixed cap
while cross-entropy keeps improving. `continuation_half_cycle_steps` and
`continuation_validation_interval` are explicit in the configuration. If the
training schedule is rescaled to keep clip exposure fixed at a different batch,
rescale these continuation counts along with warm-up and initial validation.
The continuation peak then retains approximately the same fraction of the trial LR;
the discrete schedule introduces a one-update rounding difference.

```bash
python -m experiments.annotator.video_models_benchmark.videomae_queue init \
  --config /path/to/chosen-videomae.toml \
  --cache /path/to/rgb256-cache --test-cache /path/to/test-rgb256-cache \
  --output /path/to/videomae-runs

nohup python -u -m experiments.annotator.video_models_benchmark.videomae_queue run \
  --output /path/to/videomae-runs \
  > /path/to/videomae-runs/queue.log 2>&1 </dev/null &
```

The queue runs low LR / seed 0, high LR / seed 0, then the same pair at seed 1.
Each run finishes its continuation before the next starts. Settings, training
history, event scores, `last.pt` recovery state and `best.pt` remain in that
run's directory. Failures stop the queue for inspection.

Use the same module with `status` or `pause` and `--output`. A pause saves at
the next validation boundary. To restart, use `continue` to clear pause markers,
then `run --resume`. Finished runs are skipped; unfinished continuations resume
with their optimiser state, random state and accumulated validation history.

`rank --output /path/to/videomae-runs` writes `selection.json.gz` after all four
runs finish. Each checkpoint is the earliest minimum-validation-CE checkpoint.
Learning rates are ranked by mean fully correct validation rallies, then mean
correct-player F1; an exact tie chooses the lower LR.

With `--test-cache` set at initialisation, the queue writes the selection and
evaluates both seeds of the winning learning rate after all four runs finish.
Each seed uses its minimum-validation-CE checkpoint. Test outputs go to
`test/seed0/` and `test/seed1/`; test scores do not affect selection. A resumed
queue skips completed test seeds. Preserve an incomplete test directory elsewhere
before resuming so inference can start that seed again.

Test inference evaluates centres at **stride 3, offset 0**: original video frames
0, 3, 6… across rally boundaries. Each centre still uses 16 consecutive frames.
This setting was chosen from the validation stride comparison before test scoring.
The original pilot retains dense validation. New runs use `validation_stride = 3`
with offset 0 for both checkpoint selection and validation event scores. Training
clip sampling is unchanged. Resuming or extending an existing run preserves its
saved validation stride, with missing settings interpreted as dense validation.
Set `validation_stride = 1` to start a new run with dense validation.
For a bounded continuation, set `max_total_steps` in the run configuration.
The initial schedule is unchanged; continuation stops at that total update count
or after three non-improving validation checks, whichever comes first.

The queue passes `--stride 3 --offset 0`; these are also the standalone inference
defaults for both VideoMAE and X3D, for validation and test. Explicit `--stride`
and `--offset` arguments override them.

Validation and test frame loss and confusion matrices cover evaluated centres only. Event
scoring still covers all labelled contacts and rallies. Saved probability arrays
retain one row per original centre; skipped rows contain `(1, 0, 0)` so they emit
no contacts. Validation histories and `validation_events.json.gz`, plus test
`events.json.gz` and `frame_metrics.json.gz`, record `sampling` with stride, offset
and centre counts. These arrays support the measured offset
only; dense prediction and the other offsets require additional model evaluation.
