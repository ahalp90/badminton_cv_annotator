# Dataset-builder video requirements and throughput research

Status: evidence adopted by the approved Issue 15 Batch 5 extension

Prepared: 2026-08-10

Evidence snapshot: `issue-15-dataset-builder` at `51fa592`

## Technical summary

The combined TrackNet and Inpaint shuttle stage is the first external trial's
clear bottleneck. The first completed 85-minute video required 12 hours and 45
minutes for that stage. At this rate, one GPU-day processes about 2.7 hours of
source video.

The approved near-term change is TrackNet stride 8 with a lossless, FFmpeg-
bicubic 512x288 stage input. A prior Bourbaki experiment measured about an
eight-fold reduction in runtime at 512x288, with no measured improvement from
stride 1 on that test video. The current stride-1 path can resize one decoded
1080p frame once for each overlapping window.

The original 1080p video should remain the master. TrackNet and scene detection
can use stage-specific lower-resolution inputs while preserving frame count,
frame rate, and indexing. Pose and future RGB crops still need enough player
detail, so a permanent global reduction to 288p would create avoidable risk.

The current builder also drops Inpaint provenance before annotation. This is a
quality blocker for scaling because 57.0% of frames in the first completed
trial video were selected for filling.

## Scope and evidence basis

This note covers the Issue 15 dataset-builder implementation and the bounded
external trial running on Bourbaki. It uses:

- a read-only inspection of the Bourbaki trial on 2026-08-10;
- the tracked trial configuration and dataset-builder implementation;
- prior TrackNet stride and Inpaint investigations in this repository;
- the existing RTMLib sharding proof of concept; and
- the documented input contracts for metadata, TrackNet, pose, scene, court,
  and annotation stages.

The live process was not interrupted or changed during inspection. Trial
figures below are a dated snapshot. They are not final trial acceptance
results.

## Post-snapshot decision and stopped-run outcome

After the evidence capture, the user approved stopping the first attempt and
adopted its throughput findings. The second stride-1 shuttle process was
terminated at batch 7,804 of about 10,322, after 10 hours and 23 minutes of
inference. No second-video CSV or shuttle array had been published. The first
video's completed output and every partial run artefact remain preserved.

The implementation target is now a TrackNet-only lossless FFV1 AVI proxy made
with `scale=512:288:flags=bicubic,setsar=1/1`, followed by stride-8 inference.
The 1080p download remains canonical. A direct format probe rejected FFV1 Matroska for
this boundary because it did not expose `nb_frames`; FFV1 AVI passed the
existing exact header/count/timestamp metadata contract. Two independent
synthetic AVI generations were byte-identical and decoded to identical ordered
frames.

Issue 37 RTMLib sharding is included as a separate second batch before the next
full E2E run. It must pass source-specific seek and exact array-parity gates so
the shuttle and pose numerical changes remain independently attributable.

Batch 5A is now committed locally as `5337163`. Batch 5B integrates the tested
Issue 37 core behind a positive `pose_shards` setting: one shard preserves the
original sequential child, while multiple shards require OpenCV's plan count
to match canonical metadata before independent direct-seek workers start. The
stitcher validates range, run, source, extractor, decode mode, shape, and dtype;
the dataset-builder boundary then validates canonical frame count, detection
counts, finite active values, and NaN padding before publishing the same five
compressed pose arrays. Temporary shard compression streams through atomic XZ
files. The two real-source seek and A100 numerical/performance gates remain to
be rerun on the moved production boundary.

## The first video spent 12 hours and 45 minutes in shuttle extraction

The first completed shuttle stage had these properties:

| Measure | Observed value |
| --- | ---: |
| Video ID | `9WVwZSzixh0` |
| Resolution | 1920x1080 |
| Frame rate | 30 fps, constant |
| Frame count | 153,600 |
| Source duration | 85 minutes 20 seconds |
| Shuttle-stage elapsed time | 45,897 seconds |
| Effective processing rate | 3.35 source fps |
| Runtime relative to source duration | 8.96 times real time |
| Source-video capacity | about 2.7 hours per GPU-day |

The second selected video was 91 minutes and 45 seconds long, with 165,150
frames at 1080p30. It was still in shuttle extraction when inspected.

Other completed stages were much smaller:

| Stage | Observed elapsed time |
| --- | ---: |
| Search | 162 seconds |
| Transcript retrieval | 140 seconds |
| Selection | less than 1 second |
| Two video downloads | 1,448 seconds total |
| Metadata, first video | 548 seconds |
| Metadata, second video | 547 seconds |
| Shuttle, first video | 45,897 seconds |

The stopped `449d8b1` trial configuration used one TrackNet worker, batch size
16, stride 1, and large-video mode. The current trial configuration belongs to
the replacement run and now uses stride 8.

The shuttle stage runs TrackNet and Inpaint together, so the live manifest
does not separate their wall times. An exact tracker versus Inpaint split is
therefore unavailable from this trial snapshot.

## Each stage has a different useful video floor

The pipeline has strict timing and alignment requirements, but it has no
formal minimum source resolution. A technically accepted video can still be
too poor for useful computer-vision output. The practical requirement must be
set per stage.

| Stage | Hard input requirement | Practical video requirement | Safe efficiency direction |
| --- | --- | --- | --- |
| Search and triage | Search metadata, title, duration, and selection evidence | No decoded pixels | Run before download where possible |
| Transcript and commentary | Captions or an audio stream | Pixel resolution is irrelevant | Keep audio separate from visual proxies |
| Download and canonical metadata | Positive dimensions and FPS, exact frame count, constant frame rate, zero start time, no rotation metadata, positive sample aspect ratio | A stable, decodable H.264 source | Validate once and reuse metadata |
| TrackNet | Exact source frame order and timing | Model input is fixed at 512x288 | Use exact cached preprocessing or a validated 512x288 proxy |
| Inpaint | Frame-aligned TrackNet predictions | Same temporal coverage as TrackNet | Preserve fill codes and mask downstream |
| RTMLib pose | Every retained source frame | Players and wrists must remain clear after person cropping | Keep 1080p or validate a 720p pose proxy |
| Scene detection | Same frame count and frame rate as the master | The implementation expects a scale-only 288p input | Use the intended 288p proxy |
| Court detection | Selected frames plus frame-aligned pose | Court lines must survive model resize | Sample sparse scene evidence |
| Annotation | Frame-aligned shuttle, pose, court, and mask arrays | No source pixels are read | Avoid video decoding entirely |
| Future RGB features | Frame-indexed player or action crops | Enough detail for hands, rackets, and body motion | Retain the original master until validated |

The canonical metadata checks are in
[video_metadata.py](../../src/annotator/video_metadata.py). They reject
variable frame rate, conflicting counted and declared frame totals, non-zero
start time, and rotation metadata.

TrackNet's fixed dimensions are declared in
[general.py](../../src/shared/tracknetv3/utils/general.py). RTMLib uses a
640x640 detector input and 192x256 per-person pose input in
[rtmlib_pose.py](../../src/bst_x/preparing_data/rtmlib_pose.py). Scene detection
explicitly calls for a frame-count-preserving 288p downsample in
[composition_mask.py](../../src/annotator/composition_mask.py).

## Stride and repeated resizing are the strongest known shuttle costs

Stride 1 evaluates overlapping TrackNet windows. Stride 8 uses non-overlapping
windows. The mode selection and sliding steps are implemented in
[predict.py](../../src/shared/tracknetv3/predict.py).

The large-video iterable decodes frames sequentially, but it resizes images
while processing each window. At stride 1, a frame can appear in up to eight
TrackNet windows and be resized repeatedly. See
[dataset.py](../../src/shared/tracknetv3/dataset.py).

A prior experiment used about 154,000 frames at native 512x288 on a Bourbaki
A100. It measured roughly 2 hours 40 minutes for stride 1 and about 22 minutes
for stride 8. This was about an eight-fold overall difference.

That experiment also measured an invented-coordinate share of 34.1% for
stride 8 and 35.8% for stride 1. Stride 1 changed the failure pattern, but it
did not reduce the measured share. See the
[Inpaint investigation](../tracknet/evidence/inpaint_fabrications_20260722/inpaint_fabrications_investigation.md).

Those results come from one video and do not establish general accuracy.
They are strong enough to justify a fixed comparison on the new trial footage.

## Inpaint provenance currently stops before annotation

TrackNet writes an Inpaint mask sidecar. The dataset-builder shuttle stage
records the TrackNet CSV and converted shuttle array, but the conversion keeps
only frame, position, and visibility data. See
[the shuttle plan](../../src/dataset_builder/_vision_plans.py) and
[the conversion stage](../../src/dataset_builder/vision.py).

The full annotation entry point supports either frame-level Inpaint codes or a
boolean shuttle-hallucination mask. When neither is supplied, it creates an
all-false mask. See [run_video.py](../../src/annotator/run_video.py).

The first completed trial video's sidecar reported:

| Inpaint measure | Observed value |
| --- | ---: |
| Total frames | 153,600 |
| Frames selected for filling | 87,538 |
| Selected share | 57.0% |
| Fill spans | 1,833 |
| Evaluation mode | `weight` |
| Stride | 1 |

The selected share measures frames sent through filling. It does not prove
that every filled coordinate is false. It does show that annotation currently
receives substantial model-filled evidence without its available provenance.

## Pose sharding has a measured 2.38 times speedup

The current builder launches one pose subprocess per video. An existing
Bourbaki proof of concept tested exact frame sharding on a 14,401-frame 1080p
section.

| Pose workers | Wall time | Throughput |
| ---: | ---: | ---: |
| 1 | 711.5 seconds | 20.2 fps |
| 8 | 299.3 seconds | 48.1 fps |

Eight workers were 2.38 times faster. The test included worker startup, model
loading, decoding, inference, compressed writes, and stitching. The test also
reported byte-exact CPU and CUDA output on its validated source and stack.

The sharded path needs a seek and parity check for each new codec or source
class before production use. Full details are in
[the RTMLib sharding handoff](../../src/shared/video_sharding/HANDOFF.md).

## Ranked opportunities

### 1. Change future TrackNet runs to stride 8 after a fixed clip comparison

This has the largest measured speed difference and no measured stride-1
quality advantage in the existing experiment. It should be tested on reviewed
rally-heavy and replay-heavy trial clips before becoming the production
default.

### 2. Resize each decoded TrackNet frame once

A streaming ring buffer can retain the current PIL resize result while reusing
it across overlapping windows. Exact output parity is practical because this
change need not alter model weights, source frames, interpolation, or window
assembly.

A compressed 512x288 proxy is another option. It introduces another lossy
encode, so exact in-memory caching is the lower-risk first experiment.

### 3. Increase TrackNet batch size on the A100

The trial uses batch 16. Local TrackNet guidance permits batch 64 with one
worker. Test 32 and 64 on the same clips while recording peak VRAM and sustained
GPU utilisation. One live snapshot showed about 9.7 GiB in use on the 40 GiB
A100, but a single snapshot cannot establish peak capacity.

### 4. Pass the Inpaint mask into annotation

This is primarily a correctness and provenance change. It should happen before
large-scale production. A separate TrackNet-only comparison can determine
whether Inpaint remains useful once its evidence is graded correctly.

### 5. Integrate the existing RTMLib sharding path

The current proof of concept already shows a measured gain. Start with four
workers and extend to eight after source-specific seek and parity checks.

### 6. Use the intended 288p scene-detection input

The court composition code already specifies a 288p scale-only source with
unchanged frame count. Supplying the full 1080p master adds decode and resize
work without adding intended scene-detector detail.

### 7. Gate expensive vision to likely main-court ranges

A recall-first court gate could find broadcast cuts and likely full-court
scenes before running TrackNet and pose. Conservative buffers would protect
rally boundaries. This has high potential, but a missed live-play scene could
silently remove training data. It should remain a follow-up architecture issue
until recall is measured.

### 8. Reuse vision outputs across runs

Current stage reuse is scoped to one run directory and its fingerprints. A
shared cache keyed by source checksum, model checksum, and effective
configuration would prevent repeated GPU work across experiments. Cache
integrity and invalidation need their own design.

### 9. Consider concurrent videos only after single-process tuning

Increasing `tracknet_workers` alone does not parallelise a single-video stage.
The runtime passes one video to each shuttle plan. Two independent video stages
may fit in A100 memory, but CPU decode, storage, and GPU contention need a
measured comparison.

## Proposed benchmark and decision rule

Use two fixed 5 to 10-minute clips from the first completed trial video:

1. a rally-heavy section with reviewed shuttle and event evidence; and
2. a replay or close-up-heavy section that stresses false detections and
   Inpaint.

Measure these shuttle variants:

| Variant | Stride | Batch | Source processing | Inpaint |
| --- | ---: | ---: | --- | --- |
| A, current baseline | 1 | 16 | Current 1080p path | Enabled |
| B | 8 | 16 | Current 1080p path | Enabled |
| C | 8 | 32 | Current 1080p path | Enabled |
| D | 8 | 64 | Current 1080p path | Enabled |
| E | 8 | Best safe batch | Resize-once cache | Enabled |
| F | 8 | Best safe batch | Resize-once cache | Disabled |

Record:

- wall time and source frames per second;
- sustained GPU utilisation and peak VRAM;
- exact output frame count and frame-index coverage;
- direct detection coverage and Inpaint-selected share;
- reviewed rally recall and precision;
- contact F1 or reviewed contact timing error;
- landing and player coverage; and
- every lost or newly invented reviewed rally.

Benchmark pose separately with one, four, and eight shards. Require exact
frame coverage, stitching checks, and a source-specific seek parity test.

Select the fastest configuration that preserves all reviewed rallies and has
no unexplained material loss in contact or landing quality. Report a
conservative capacity estimate from observed end-to-end throughput rather than
from model inference alone.

## Limits and open questions

- The live timing covers one completed video and one partial video.
- TrackNet and Inpaint wall times are combined in the current manifest.
- Prior stride-quality results come from one annotated source video.
- The 57.0% fill share measures selection for filling, not confirmed false
  coordinates.
- The A100 utilisation and memory figures were snapshots rather than sustained
  measurements.
- A 720p pose proxy may be sufficient, but current evidence does not establish
  wrist, racket, or future RGB-crop parity.
- Court-first gating may provide the largest architectural saving, but its
  live-play recall has not been measured.
- Metadata validation currently reads every frame's timing information. A
  controlled proxy could make that cheaper, but metadata is small beside the
  current shuttle cost.

## Recommended issue boundary

The first issue should own the fixed benchmark, stride and batch decision,
resize-once experiment, pose-sharding comparison, Inpaint provenance gate, and
production capacity estimate.

Create follow-up issues for court-first gating, cross-run caching, reduced-FPS
experiments, mixed precision, or permanent source-resolution changes. These
options require separate accuracy or architecture decisions and would make the
initial throughput issue harder to review.
