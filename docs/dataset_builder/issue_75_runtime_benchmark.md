# Issue 75 TrackNet runtime benchmark

## Decision

PR #101 changes no runtime configuration. Retain the TrackNet preset that was
already active when the Issue 75 benchmark began:

- non-overlapping stride 8;
- batch size 16;
- one TrackNet worker with large-video mode enabled.

Keep the persisted 512x288 FFV1 input until the exact streaming candidate is
implemented behind the dataset-builder contract. The candidate was 26.0%
faster across the fixed clips and exactly reproduced proxy-decoded pixels and
TrackNet CSVs. This benchmark records that result but does not put scratch
prototype code into production.

The benchmark compared TrackNet stride, batch size, and input paths. It held
`tracknet_workers = 1` constant. Pose and court settings were outside its
measurement scope.

## Configuration chronology and work-start baseline

"Pre-Issue-75" is ambiguous because the issue was opened before its benchmark
worktree was created. These are separate states.

Issue #75 was created at `2026-08-10T04:01:20Z`. The dataset builder had not
yet reached `main`; the active Issue 15 configuration introduced by `449d8b1`
and present at `51fa592` was:

- TrackNet worker 1, batch 16, stride 1, and large-video mode;
- InpaintNet enabled by its configured checkpoint;
- direct canonical-source input, with no persisted TrackNet proxy stage;
- one sequential CUDA pose process with `n_max = 10` and no `pose_shards`
  field;
- CUDA court inference with padded resize.

This is the older state described in the Issue #75 body. No accepted run
manifest was checked for it, so these are code and configuration facts, not a
claim that every setting completed an external run.

Before the Issue 75 worktree started, Issue 15 commit `5337163` changed stride
1 to stride 8 and added the persisted 512x288 FFV1 proxy. Commit `32dcefa`
then added eight pose shards. The Issue 75 worktree was created later at branch
point `462d5b868e5308a497794955c486f3312fc876a6`.

Those were the only tracked trial-TOML changes between the issue-creation
configuration and the Issue 75 work-start preset:

| Setting | Issue-creation configuration | Issue 75 work-start preset |
| --- | --- | --- |
| `tracknet_stride` | `1` | `8` |
| `pose_shards` | absent; sequential pose producer | `8` |

TrackNet workers remained 1, batch size remained 16, and large-video mode
remained enabled. Pose stayed on CUDA with `n_max = 10`; court stayed on CUDA
with padded resize. Model paths, the two-video limit, and the single download
worker also did not change. The proxy was an implementation-stage addition,
not another TOML option: it changed TrackNet input from the canonical source
to a derived 512x288 FFV1 video. Adding `pose_shards = 8` changed pose execution
from one sequential process to eight seek-based shard processes.

The original stride-1 choice in `449d8b1` has no recorded rationale. Earlier
evidence committed in `d68550d` had already measured about 2 hours 40 minutes
for stride 1 versus 22 minutes for stride 8, with detected-artefact shares of
35.8% and 34.1% respectively. It described stride 1 as a regression in
detectability, not an improvement. The later throughput note made the intended
next step explicit: compare fixed clips, then move future runs to stride 8 if
quality held.

The complete `[vision]` section at that branch point is the authoritative
Issue 75 work-start preset. `load_builder_config` requires every field, and
the stage plans pass each value explicitly to its producer.

| Setting | Tracked preset at `462d5b8` | Accepted-run evidence |
| --- | --- | --- |
| TrackNet worker processes | `tracknet_workers = 1` | both shuttle stages recorded `workers: 1` |
| TrackNet batch size | `tracknet_batch_size = 16` | both shuttle stages recorded `batch_size: 16` |
| TrackNet stride | `tracknet_stride = 8` | both shuttle stages recorded `stride: 8`; the wrapper maps this to `nonoverlap` |
| TrackNet decode path | `tracknet_large_video = true` | both shuttle stages recorded `large_video: true` |
| InpaintNet | configured checkpoint, therefore enabled | both shuttle stages recorded `inpainting: true` and bound the checkpoint |
| TrackNet input | derived stage, not a TOML option | persisted 512x288 bicubic FFV1 level 3, GOP 1, YUV420P, square-pixel video |
| Pose device | `pose_device = "cuda"` | both pose stages recorded `device: "cuda"` |
| Pose capacity | `pose_n_max = 10` | both pose stages recorded `n_max: 10` |
| Pose shards | `pose_shards = 8` | both pose stages recorded `shards: 8` and `decode_mode: "seek"` |
| Court device | `court_device = "cuda"` | both court stages recorded `device: "cuda"` |
| Court resize | `court_resize_mode = "pad"` | both court stages recorded `resize_mode: "pad"` |

The accepted two-video run manifest is
`/scratch/cmarti/issue84_6359790/external/trial-run/run_manifest.json.gz`. Its
stage fingerprints name source commit
`6359790063216b0270aadc89a4c9f5a006eccf85`, and its compressed SHA-256 is
`e98f12c7aa28a30619f0fa2c513b13f2356afff37a7c1cdd4becf0f6e173b5e3`.
The tracked trial configuration is byte-identical at that run commit, the
Issue 75 branch point, and audited `origin/main` commit `5576410`; its SHA-256
is `884d01cf52622e399c8337ff8df5aa4b23534cc6296c5836b6f5b47bbc2a8fd5`.

PR #101 changes no configuration, source, or test file. It records these
states and the benchmark results only.

The lower-level `extract_all_shuttles` helper still declares generic callable
defaults of two workers, batch 32, stride 1, and non-large-video mode. They are
not dataset-builder defaults: the builder call site overrides all four with
the required tracked preset. This distinction matters when reconstructing an
executed run.

## Evidence

The bounded benchmark used repository commit
`462d5b868e5308a497794955c486f3312fc876a6` and an A100 40 GB GPU. TrackNet and
InpaintNet checkpoint SHA-256 values were:

- `df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a`;
- `5749b66b8002f3ad9e0af841604004706fc796df30599e6bf01952696009688c`.

Two immutable clips were mapped exactly to accepted source video
`9WVwZSzixh0`:

| Clip | Source frames | Frames | Content |
| --- | ---: | ---: | --- |
| rally-heavy | `[57900, 66899)` | 8,999 | nine accepted rallies |
| replay-heavy | `[105300, 114300)` | 9,000 | 83.2% replay or non-court |

Stride-8 cases ran twice after one warm-up. The stride-1 historical control ran
once. All 18 measured commands exited successfully. Timings below sum the
per-clip medians.

| Input | Stride | Batch | Inference | Frames/s | Peak VRAM | Peak host RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| proxy | 8 | 16 | 247.48 s | 72.73 | 9,711 MiB | 3.52 GiB |
| proxy | 8 | 32 | 252.63 s | 71.25 | 18,701 MiB | 4.23 GiB |
| proxy | 8 | 64 | 252.49 s | 71.29 | 36,683 MiB | 7.20 GiB |
| direct source | 8 | 64 | 1,667.91 s | 10.79 | 36,683 MiB | 40.04 GiB |
| proxy | 1 | 16 | 1,012.52 s | 17.78 | 9,711 MiB | 3.52 GiB |

Batch 32 was 2.1% slower than batch 16 on the bounded clips. Batch 64 was 2.0%
slower and used 89.6% of the GPU's memory. Larger batches raised peak GPU use
without improving wall time because decoding, resizing, median generation, and
post-processing remain important costs.

The current direct-source implementation was 6.7 times slower than proxy
inference. It was 2.8 times slower than the complete batch-16 proxy lane after
adding 347.43 seconds of proxy creation. Its high-resolution median path also
raised host RSS to 40 GiB. The current direct path is not a safe replacement
for the proxy.

The two proxy files total 1,094,137,396 bytes. This remains a material storage
and preprocessing cost.

## Exact streaming resize experiment

The parity candidate applies the existing FFmpeg bicubic scale and YUV420
conversion, then sends lossless FFV1 through an in-memory NUT pipe. A second
FFmpeg process decodes that stream to the BGR frames TrackNet already receives.
It performs one sampled median pass and one inference pass without publishing
an intermediate video.

A simpler FFmpeg-to-BGR pipe was rejected before inference. It changed channel
values in every frame because it did not preserve the established encode and
decode colour path. The FFV1 pipe exactly matched every decoded proxy pixel:

| Clip | Frames compared | Changed frames | Changed channel values | Largest channel change |
| --- | ---: | ---: | ---: | ---: |
| rally-heavy | 8,999 | 0 | 0 | 0 |
| replay-heavy | 9,000 | 0 | 0 | 0 |

One full TrackNet and Inpaint run then passed on each clip:

| Clip | Proxy creation + inference | Exact stream | Reduction | CSV parity | Fill-span parity |
| --- | ---: | ---: | ---: | --- | --- |
| rally-heavy | 301.95 s | 225.62 s | 25.3% | byte-identical | exact |
| replay-heavy | 292.97 s | 214.63 s | 26.7% | byte-identical | exact |
| **Total** | **594.91 s** | **440.25 s** | **26.0%** | **byte-identical** | **exact** |

End-to-end TrackNet-lane throughput rose from 30.25 to 40.89 source frames per
second. Peak host RSS was 3.50 GiB on the streamed rally clip and 2.95 GiB on
the replay clip. The candidate also avoids the 1.09 GB of bounded proxy files.

The sidecars differed only in `input_video` (`.mp4` source versus `.avi` proxy)
and `extracted_utc`; all fill spans and model fields matched. A production
implementation must define that logical input identity, propagate FFmpeg
failures, reject short streams, preserve cancellation, and retain an explicit
fallback to the persisted proxy. Issue #100 owns that production implementation
and its whole-video and resume gates.

## Output validation

Every stride-8 result had exactly one contiguous row per source frame, finite
coordinates, valid visibility values, and a valid `inpaint_fill_mask/1`
sidecar. Repeated TrackNet CSVs were byte-identical. Repeated sidecars had the
same fields and spans after excluding `extracted_utc`; their compressed bytes
differed because each extraction recorded its own timestamp.

Batch 32 and batch 64 produced byte-identical CSVs and semantically identical
sidecars. They did not exactly match batch 16:

| Clip | Changed rows | Visibility changes | Largest coordinate change | Fill-mask changes |
| --- | ---: | ---: | ---: | ---: |
| rally-heavy | 1 of 8,999 | 0 | 1 proxy pixel | 0 |
| replay-heavy | 3 of 9,000 | 0 | 1 proxy pixel | 0 |

The larger batches therefore fail the predeclared exact-output gate. The
changes are small, but neither larger batch improved speed.

A second comparison used both complete accepted videos. Batch 32 took 2,430.47
seconds versus 2,330.84 seconds for batch 16, so it was 4.3% slower. It changed
37 of 318,750 coordinates by one pixel. Visibility and semantic Inpaint
sidecars were unchanged. This confirms the bounded decision at production
scale.

That complete-video comparison reused the fixed-input Issue 94 experiment at
commit `ce9405b2c1cb9aec948e510f9f1e6e3af410aabf`. Its TrackNetV3 inference
tree was identical to the bounded benchmark. Its shuttle wrapper differed only
in post-inference validation and conversion code, outside the timed model
path.

Direct input changed 655 visibility values and 1,132 fill-mask positions across
the bounded clips. On mutually visible frames its 95th-percentile coordinate
difference was 28.42 proxy pixels on the rally-heavy clip and 132.77 proxy
pixels on the replay-heavy clip. It fails both the runtime and equivalence
gates. Downstream annotation was not run because the decision rule required a
direct candidate to reduce total lane time before quality review.

## Production capacity

The accepted two-video builder run processed 318,750 frames at 30 fps. That is
2.951 source-video hours in 5 hours 42 minutes 23 seconds of wall time. A GPU
running continuously at that measured rate would process 86.9 source-video
hours per week.

For planning, reserve 15% to 20% of the week for queue gaps, retries, transfers,
and operational checks. The resulting conservative range is 70 to 74 hours of
source video per GPU-week. At the accepted pair's average duration of 1.476
hours, that is about 47 to 50 videos per week. This is a throughput estimate,
not a delivery guarantee; it excludes manual review and assumes one comparable
A100 is available for the remaining time.

## Stride-1 control defect

The rally-heavy stride-1 control produced 9,000 rows from an 8,999-frame proxy.
The extra final row was an invisible zero coordinate, and the sidecar reported
the same incorrect row count. The streaming overlap path pads a final window
and its disabled prediction-length assertion does not reject the extra row.

The production stride-8 path did not reproduce this defect. Both bounded clips
and both complete production videos had exact frame counts. The stride-1 result
is retained only as timing evidence and is not structurally valid.

## Reproduction

The immutable remote evidence is under
`/scratch/cmarti/issue75_462d5b8`. The external Issue 75 planning directory
contains a compact evidence copy, `analyse-benchmark.py`, and generated
`benchmark-validation.json`. Its `benchmark-evidence/streaming` directory holds
the accepted streamed CSVs, sidecars, wall-time records, and all-frame pixel
hashes. `compare-stream-pixels.py` and `benchmark-stream-tracknet.py` are the
scratch prototypes that generated those checks; neither is production code.

Run the saved validation with:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  issue-75-dataset-builder-runtime/analyse-benchmark.py \
  issue-75-dataset-builder-runtime/benchmark-evidence \
  --output issue-75-dataset-builder-runtime/benchmark-validation.json
```

For the original 18-case matrix, the GPU sampler recorded utilisation and
memory once per second. The runner checked for other CUDA processes before
every case. Peak VRAM was stable for each batch size across clips and repeats.
That sampler did not record process identities, so it cannot independently
prove that no process appeared mid-case. No timing, utilisation, or memory
trace showed evidence of overlap.

The two exact-streaming runs had an idle GPU check immediately before launch
but did not save GPU or process samples. Their 26.0% timing result therefore
cannot independently exclude a workload that started after either launch.
