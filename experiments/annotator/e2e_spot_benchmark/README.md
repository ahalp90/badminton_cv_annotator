# E2E-Spot badminton benchmark

Train RegNet-Y 200MF or 800MF with gated shift modules (GSM), a one-layer
bidirectional GRU and a dense contact classifier. All model weights are trained,
starting from ImageNet backbone weights. This package reuses the canonical
32-video training / 8-video validation split and the frozen 46-video test set.

The model follows [E2E-Spot](https://github.com/jhong93/spot), revision
`edec4201471beed631bed374bd0b95fcdc8a2f4f`. GSM wraps every RegNet bottleneck's
first convolution. GRU hidden dimensions are 368/768 for 200MF/800MF, with 0.5
dropout before the classifier. See [LICENSE.spot](LICENSE.spot) and
[LICENSE.gsm](LICENSE.gsm). GSM padding uses the input tensor's device and dtype.
The tested backbone dependency is `timm==1.0.31`, using the explicit
`regnety_002.pycls_in1k` and `regnety_008.pycls_in1k` ImageNet weight tags.

## Frames and labels

Prepare fresh source frames at **224 pixels of actual picture height**, keeping
the source aspect ratio. Store JPEG quality 95 with 4:2:0 colour subsampling in
one ZIP per video. During training, select one random horizontal 224-square crop
for the whole clip. Validation and test use the centre crop. Crop before batching,
so videos with different widths work together. There is no letterboxing,
additional area crop, displacement or resize after cropping.

Training retains clip-consistent horizontal flipping and brightness, contrast
and saturation changes, followed by ImageNet normalisation. Blur is off. MixUp
is off initially; the staged search compares it at the selected learning rate.
MixUp uses Beta(0.2, 0.2), with each component's ignored labels excluded separately.

Supplied mode uses the existing one-second-padded rally windows and classes
NO_CONTACT=0, BOT_CONTACT=1 and TOP_CONTACT=2. Discovery adds NOT_RALLY=3 outside
those windows, and samples contiguous clips throughout the complete videos.
Choose videos in proportion to duration. The measured class counts are reported;
they do not alter sampling.

Known contacts label frames ±1; exactly ±2 are ignored. Unknown hitter labels
retain their ignored bands. Positive labels override ignored bands, and TOP wins
colliding positives, matching the shared RGB label helper.

For supplied training, choose each rally interval in proportion to its duration,
then choose a centre uniformly inside it. A 100-frame clip uses only frames from
that one interval: first contact minus one second through last contact plus one
second. Positions outside it are artificial padding, even if the cache contains
neighbouring footage. Keep rally scopes separate when their stored intervals
merge. Discovery continues to use full-video scopes.

A frame-validity mask is separate from the ignored loss labels. Real pixels at
±2 contact frames, unknown-hitter frames and positions excluded by a partial
training budget still provide input context. Exclude artificial padding from
contrast statistics, zero it after augmentation and ImageNet normalisation, then
apply MixUp. Paired real frames therefore remain available. Padding is excluded
from loss and prediction assembly; zero input does not instruct the GRU to skip
a time step. Scheduled padding still counts towards the initial 500,000-position
cycle budget; eligible counts and proportions show how much carries a target.
Validation uses deterministic 50% overlap, centring a short interval in a padded
clip without borrowing footage from another rally. A clip never joins frames
across a missing-frame gap.


This follows the X3D/VideoMAE treatment of uncertain contact timing: the immediate
neighbours count as positive, while the next frames are too ambiguous to label
confidently as background. The ignored band affects training and validation loss
only. Predictions there still count in contact and rally metrics, against the
original contact timestamps and the chosen evaluation tolerance.

## Prepare and inspect

Use the project's PyTorch/torchvision environment plus [requirements.txt](requirements.txt).
FFmpeg and FFprobe must be installed for preparation. Run from the repository root.
Expose the repository's source packages before training or scoring:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

All runtime locations are command-line arguments; the examples use shell variables.

```bash
python -m experiments.annotator.e2e_spot_benchmark.prepare \
  --spans "$DEVELOPMENT_SPANS" --videos "$VIDEO_ROOT" \
  --cache "$SUPPLIED_CACHE" --mode supplied --workers 2

python -m experiments.annotator.e2e_spot_benchmark.prepare \
  --spans "$DEVELOPMENT_SPANS" --videos "$VIDEO_ROOT" \
  --cache "$DISCOVERY_CACHE" --mode discovery --workers 2
```

You can use `--index "$EXISTING_INDEX"` to reuse canonical video/split metadata.
Frame pixels must still be extracted from the source videos. Supplied preparation
stores only the existing one-second-padded target bounds, merging storage
intervals where they overlap. There is no extra-context expansion. Discovery
stores every source frame. A full-video JPEG cache can also serve supplied mode;
the dataset still limits each clip to its selected rally interval.

Preparation resumes at video boundaries. Completed ZIPs are reused only when
metadata and member counts agree. An interrupted video is rebuilt through a
`.partial` file. Check available space before extraction; compressed size depends
on source content. Each persistent data worker keeps its ZIP readers open.

## Train and resume

The [pilot configuration](configs/pilot.toml) uses supplied windows, 200MF,
100-frame clips, learning rate 0.001 and seed 0. Batch size 8 is a provisional
profiling value, not the selected training batch. Profile 8/16 (and possibly 32
for supplied200MF100); measure longer discovery clips separately. Validation
batch size is independent through `validation_batch_size` or
`--validation-batch-size`. Choosing a batch never automatically scales learning rate. AdamW weight decay
is fixed at 0.01. Training weights are [1,5,5] or [1,5,5,1].

Train 50 cycles of exactly 500,000 sampled frame positions each, with three
linear warmup cycles followed by cosine decay over the remaining 47. The final
batch may be smaller, and the final clip's unused positions are masked. Record
both scheduled positions and eligible class counts/proportions, plus ignored positions. No automatic early stopping
is applied.

```bash
python -m experiments.annotator.e2e_spot_benchmark train \
  --config experiments/annotator/e2e_spot_benchmark/configs/pilot.toml \
  --cache "$SUPPLIED_CACHE" --output "$RUN" --device "$DEVICE" --workers 4

python -m experiments.annotator.e2e_spot_benchmark train \
  --config experiments/annotator/e2e_spot_benchmark/configs/pilot.toml \
  --cache "$SUPPLIED_CACHE" --output "$RUN" --device "$DEVICE" --workers 4 \
  --resume
```

`recovery.pt` atomically saves model, optimiser, random states, frame counts and
position within the cycle. Sample and crop choices depend on the seed and
absolute sample index, so worker prefetch does not change resumed sampling.
Progress checkpoints are written about every 30 seconds and at cycle boundaries.
If startup or the first step fails before a checkpoint exists, rerun without
`--resume` using unchanged settings and input metadata. Once `recovery.pt` exists,
use `--resume`; a fresh start in that directory is rejected.
`--max-steps N` stops after N additional training steps with a recovery checkpoint.

Validation covers every relevant original frame using deterministic 50% overlap,
including short windows and tails. Average probabilities across overlapping
clips, then compute unweighted cross-entropy once per eligible original frame.
Report per-class and combined contact losses. Select minimum validation loss,
keeping the earlier cycle on a tie. Validation is unaugmented and uses no MixUp.

`best.pt` points to the retained `validation-cycleNNNN/` directory containing
compressed probability arrays and decoded streams from that same validation pass.
Array row N corresponds to original frame N. Supplied-mode rows outside target
windows are zero. Older best-validation arrays are removed after the replacement
checkpoint is recoverable. `history.json.gz` retains every cycle's diagnostics.

## Event scoring and explicit test inference

Use the shared contact decoder: argmax, TOP/BOT candidates, six-frame suppression
at 30 fps, then forced alternating hitters. Resolve first-hitter ties by agreement
count, probability and the first predicted hitter. These rules affect event
metrics only. Discovery first groups contiguous frames with rally probability
at least 0.5, without ground-truth boundaries or a gap/minimum-duration heuristic.
Empty predicted rallies are omitted. Fully-correct rally counts match each GT and
prediction at most once: an additional correct prediction of the same GT is a
false positive. Report rally precision, recall and F1. Shared test exclusions and
their ten-frame keep margin apply through the common scorer. Discovery scoring
uses the first and last accepted contacts after exclusions, as in the existing
discovery benchmark; saved streams retain the original probability-mask bounds.

```bash
python -m experiments.annotator.e2e_spot_benchmark infer \
  --checkpoint "$RUN/best.pt" --cache "$TEST_CACHE" \
  --role test --output "$TEST_RESULTS" --device "$DEVICE" --workers 4
```

Prepare test frames explicitly with `prepare --test-inputs "$TEST_INPUTS" ...`.
Training and search do not read test data. Inference also accepts `--role validation`.

## Staged search and timing

Plan one stage at a time. Ranking requires completed validation results from
seeds 0 and 1 under unchanged settings. Rank mean fully-correct rally recall,
then correct-player contact F1, at the shared ten-frame tolerance. Freeze the
winner and complete seeds 0–4; completed matching seeds are reused.

Stages are `supplied200-lr`, `supplied200-mixup`, `supplied800-lr`,
`discovery200-length-lr`, `discovery800-lr`, and `complete`. Each later stage takes
an explicit preceding selection. The exact predecessor is checked: the MixUp
stage follows supplied200 learning-rate selection; supplied800 and discovery200
follow supplied200 MixUp selection; discovery800 follows discovery200 selection.
For a new candidate stage, `--config` explicitly supplies training/validation
batch sizes, cycle counts and the remaining runtime protocol. Only searched
model/mode/clip-length/learning-rate/MixUp choices are inherited. Supply the
profiled stage-specific configuration; changing batch size does not scale LR.
The `complete` stage preserves the winner's full protocol and rejects conflicting
config fields, so matching screening seeds remain part of the final five.
Discovery 200MF compares clip lengths 250/500
and learning rates 0.001/0.0003; 800MF inherits the selected length. Backbones
inherit the preceding label and augmentation choices.

For the discovery200 length grid, pass the separately profiled training batches:

```bash
python -m experiments.annotator.e2e_spot_benchmark.search plan \
  --stage discovery200-length-lr --config "$DISCOVERY_CONFIG" \
  --selection "$SUPPLIED200_SELECTION" --runs "$SEARCH_ROOT" \
  --batch-size-for-length 250=8 --batch-size-for-length 500=4
```

Use the same overrides for `plan`, `run` and `rank`. Unspecified lengths use the
config's training batch; validation batch remains common to both lengths.
These overrides apply only to this length grid. For `complete`, omit them and
use a config matching the selected winner's full protocol, including its batch.

```bash
python -m experiments.annotator.e2e_spot_benchmark.search plan \
  --stage supplied200-lr \
  --config experiments/annotator/e2e_spot_benchmark/configs/pilot.toml \
  --runs "$SEARCH_ROOT"
```

Use `run` with `--cache`, `--device` and `--workers` to execute a planned stage.
The default `--limit 1` launches at most one new run. Use `rank` to write that
stage's selection, then pass `--selection` when planning its successor. No search
command automatically evaluates candidates on test. Use `summarise --stage complete`
with the frozen selection to report five-seed validation means and standard deviations.

`smoke` and `timing` use real clips with randomly initialised architecture weights,
so they download nothing. Both perform forward/backward/optimiser steps. Timing reports both end-to-end throughput from the prefetched loader and throughput
from one already-loaded device batch, plus peak GPU memory. Both use identical
augmentation and training arithmetic. Exclude explicitly requested warmup steps.
A large throughput gap points to loading/transfer costs worth investigating
before choosing the training batch. `--full-validation` also times the complete
deterministic validation pass using its independently configured batch size. They save no trained checkpoint.

```bash
python -m experiments.annotator.e2e_spot_benchmark smoke \
  --config "$CONFIG" --cache "$SUPPLIED_CACHE" --device "$DEVICE" --steps 1

python -m experiments.annotator.e2e_spot_benchmark timing \
  --config "$CONFIG" --cache "$SUPPLIED_CACHE" --device "$DEVICE" \
  --workers 4 --batch-size 8 --validation-batch-size 16 \
  --warmup 3 --steps 10 --full-validation --output "$TIMING_RESULT"
```

Inspect every interface with `--help`. Measure the first full validation pass
before choosing a search budget. Exact-frame labels, supplied250 and the previous
RGB spatial augmentation are separate follow-ups, not initial search axes.
