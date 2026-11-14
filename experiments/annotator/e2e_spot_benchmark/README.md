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

Both recipes end in **224-square RGB input with ImageNet normalisation**. Prepare
fresh source frames as JPEG quality 95 with 4:2:0 colour subsampling, in one ZIP
per video. Each recipe has its own source geometry and cache:

- **X3D/VideoMAE recipe** (`augmentation = "user"`): resize/letterbox to 256-square, then
  reuse the X3D/VideoMAE colour, crop and flip implementation. Colour changes
  apply together with probability 0.5; brightness/contrast/saturation factors
  range from 0.8 to 1.2. Independently, a square crop covers 80–100% of the area
  with probability 0.5, and horizontal flip applies with probability 0.5. The
  VideoMAE preprocessing branch then resizes to 224 and ImageNet-normalises.
  Validation/test deterministically resize the whole letterboxed image to 224.
  This matches the original augmentation; JPEG storage replaces the original
  uncompressed RGB cache. There is no hue jitter, blur or MixUp in this recipe.
- **E2E recipe** (`augmentation = "e2e"`, `mixup = true`): preserve aspect ratio
  at height 224, then use one horizontal 224-square crop for the clip (random in
  training, centre in validation/test). Flip applies with probability 0.5. Hue
  (±0.2), saturation, brightness and contrast (factors 0.7–1.2) each apply
  independently with probability 0.25, followed by Gaussian blur (probability 0.25,
  kernel 5, sigma 0.1–2.0). Sample an independent partner clip and Beta(0.2, 0.2)
  mixing weight for each training clip. Augment both clips before mixing; each
  component keeps its own ignored-label mask.

Random spatial/colour transforms are consistent across each clip. The legacy
`hybrid` setting retains E2E cropping with the previous colour changes and
optional within-batch MixUp for existing checkpoints. Hybrid runs remain
separate from the new recipe comparisons.

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
  --cache "$USER_CACHE" --mode discovery --geometry user --workers 2

python -m experiments.annotator.e2e_spot_benchmark.prepare \
  --spans "$DEVELOPMENT_SPANS" --videos "$VIDEO_ROOT" \
  --cache "$E2E_CACHE" --mode discovery --geometry e2e --workers 2
```

You can use `--index "$EXISTING_INDEX"` to reuse canonical video/split metadata.
Frame pixels must still be extracted from the source videos. Supplied preparation
stores only the existing one-second-padded target bounds, merging storage
intervals where they overlap. There is no extra-context expansion. Discovery
stores every source frame. A full-video JPEG cache can also serve supplied mode;
the dataset still limits each clip to its selected rally interval. Geometry
must match the recipe; prepare user frames from source rather than resizing
already-compressed E2E frames.

Preparation resumes at video boundaries. Completed ZIPs are reused only when
metadata and member counts agree. An interrupted video is rebuilt through a
`.partial` file. Check available space before extraction; compressed size depends
on source content. Each persistent data worker keeps its ZIP readers open.

## Train and resume

Use [supplied200-user.toml](configs/supplied200-user.toml) for the first new
comparison. [discovery200-user.toml](configs/discovery200-user.toml) starts a
separate 250-frame discovery comparison. Starting training batches are 16/8/4 for
100/250/500 frames. Keep each batch fixed across LR/seed/recipe comparisons.
For 800MF, confirm the intended batch fits and halve it if necessary before
starting its comparison; its configuration files are starting values.
Validation batch size is independent through `validation_batch_size` or
`--validation-batch-size`. Choosing a batch does not automatically scale LR.
AdamW weight decay stays 0.01; training class weights are [1,5,5] or [1,5,5,1].

Train at most 50 cycles of exactly 500,000 scheduled frame positions each, with
three linear warmup cycles then cosine decay over the original 50-cycle horizon.
Stop after 10 completed validation cycles without a strictly lower validation
loss. An equal loss does not reset patience. Early-stopped runs retain the
best-loss checkpoint and count as completed candidates. Resume preserves the
stopping decision. Set `early_stopping_patience = 0` to disable stopping, as in
the historical [pilot configuration](configs/pilot.toml).

The final batch can be smaller; unused positions in the final clip are masked.
Only the primary scheduled positions count towards the frame budget when MixUp
samples partner clips. Record eligible class counts/proportions and ignored
positions alongside the scheduled budget. New configs enable
`cudnn_benchmark = true`; use the same setting for fit checks and training.

```bash
python -m experiments.annotator.e2e_spot_benchmark train \
  --config experiments/annotator/e2e_spot_benchmark/configs/supplied200-user.toml \
  --cache "$USER_CACHE" --output "$RUN" --device "$DEVICE" --workers 8

python -m experiments.annotator.e2e_spot_benchmark train \
  --config experiments/annotator/e2e_spot_benchmark/configs/supplied200-user.toml \
  --cache "$USER_CACHE" --output "$RUN" --device "$DEVICE" --workers 8 \
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

## Staged search

Rank each configuration using completed validation results from seeds 0 and 1.
Select mean fully correct rally percentage at ±10 frames, then correct-player
contact F1. Each run contributes its minimum-validation-loss checkpoint.

| Stage | Comparison | New runs |
|---|---|---:|
| `supplied200-lr` | User recipe, 100 frames, LR 0.001/0.0003 | 4 |
| `supplied200-recipe` | Full E2E recipe at selected supplied LR; reuse user pair | 2 |
| `supplied800` | Inherit selected supplied LR and recipe, 100 frames | 2 |
| `discovery200-lr` | Fresh user-recipe LR 0.001/0.0003 comparison, 250 frames | 4 |
| `discovery200-recipe` | Full E2E recipe at selected discovery LR; reuse user pair | 2 |
| `discovery200-length` | Try 500 frames; reuse selected 250-frame pair | 2 |
| `discovery800-length` | Inherit discovery LR and recipe; test 250 and 500 frames | 4 |

There are 20 new screening runs. Discovery starts independently of the supplied
selection. The 800MF stages inherit LR and recipe rather than repeating the LR
grid. Recipe comparisons change the complete augmentation recipe, including
spatial preprocessing and MixUp; they are not MixUp-only comparisons.

`--config` sets the runtime protocol for a stage. Recipe comparisons require it
to match the preceding winner so the reference pair can be reused. The
200MF length comparison similarly preserves the selected 250-frame protocol.
Length stages scale training batch inversely with clip length using the supplied
config as the reference. Override this explicitly with
`--batch-size-for-length 250=8 --batch-size-for-length 500=4` where needed.
Validation batch remains the independent value from the config.

```bash
CONFIG=experiments/annotator/e2e_spot_benchmark/configs/supplied200-user.toml
python -m experiments.annotator.e2e_spot_benchmark.search plan \
  --stage supplied200-lr --config "$CONFIG" --runs "$SEARCH_ROOT"

python -m experiments.annotator.e2e_spot_benchmark.search run \
  --stage supplied200-lr --config "$CONFIG" --runs "$SEARCH_ROOT" \
  --user-cache "$USER_CACHE" --e2e-cache "$E2E_CACHE" \
  --device "$DEVICE" --workers 8 --limit 4

python -m experiments.annotator.e2e_spot_benchmark.search rank \
  --stage supplied200-lr --config "$CONFIG" --runs "$SEARCH_ROOT"
```

Pass the generated selection to the next stage with `--selection`. Start
`discovery200-lr` without a preceding selection. Both recipe caches can be passed
to every run command; training uses the cache matching each candidate. `--cache`
remains a single-recipe fallback. Run names distinguish user/E2E/hybrid recipes;
completed runs are reused only when their full settings match.

The default `--limit 1` starts at most one new run. No command automatically
launches the whole search or evaluates candidates on test. The optional
`complete`/`summarise` interfaces can later complete a frozen winner to five
seeds; that is separate from this 20-run screening plan and preserves the full
winning protocol.

## Fit checks

Use a short forward/backward smoke to confirm each new model/clip combination
fits, with its actual augmentation and autotuning settings. No batch-size sweep
or throughput profiling is required. Smoke uses random model weights, so it
needs no pretrained-weight download and saves no trained checkpoint.

```bash
python -m experiments.annotator.e2e_spot_benchmark smoke \
  --config "$CONFIG" --cache "$CACHE" --device "$DEVICE" \
  --workers 8 --warmup 1 --steps 2
```

`--full-validation` additionally checks a complete deterministic validation pass.
The existing `timing` command remains available for explicit diagnostics; unlike
smoke, it compares end-to-end throughput with a device-resident batch.
Inspect each interface with `--help`.
