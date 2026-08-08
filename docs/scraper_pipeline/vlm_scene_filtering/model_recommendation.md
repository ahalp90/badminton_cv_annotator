# Local VLM recommendation for badminton scene filtering

*Research cut-off: 6 August 2026. Sources checked 8 August 2026.*

## Result

Trial `yanziang/InternVideo3-8B-Instruct` first on each complete video shard.
Keep `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` as the single fallback if
InternVideo3 cannot process the complete request or performs poorly on the
existing annotated broadcasts.

Use two calls for different jobs:

1. A complete 20-30 minute shard establishes broadcast phase, camera view,
   playback speed and continuity.
2. A short, densely sampled boundary clip refines a candidate transition to
   source-frame precision.

The model should use existing cuts, court evidence and suspected rally spans as
context. It does not replace player, shuttle or court tracking. This design is
a project recommendation that still needs a short run on the intended GPU and
an accuracy measurement. It is not a published guarantee from either model
author.

GitHub issue [#38](https://github.com/ahalp90/badminton_cv_annotator/issues/38)
tracks implementation and evaluation.

## Why InternVideo3 is first

InternVideo3 is designed for long-video context and temporal grounding. Its
[model card](https://huggingface.co/yanziang/InternVideo3-8B-Instruct) records:

- BF16 precision;
- a maximum context of 262,144 tokens;
- training that included 2,048 frames at 4 fps;
- direct video input through its custom Transformers processor; and
- M²LA attention, which stores a compact latent attention state while retaining
  the multimodal token stream.

The accessible checkpoint is
[`yanziang/InternVideo3-8B-Instruct`](https://huggingface.co/yanziang/InternVideo3-8B-Instruct).
The model-card example also names an `OpenGVLab/...` path, but the public file
tree and the author model zoo use `yanziang/...`. Pin and review the accessible
repository revision before enabling `trust_remote_code=True`.

The [InternVideo3 paper](https://arxiv.org/abs/2606.12195) reports strong
long-video and temporal-grounding results relative to several similarly sized
models. Those benchmarks justify a first trial; they do not predict badminton
scene accuracy.

The 2,048-frame and 4-fps figures describe training, not an unconditional
runtime cap or default. The model card shows a separate 1-fps example. The
first project run must therefore log the actual sampled frame IDs, token count,
resolution and coverage rather than assuming the processor consumed the
requested grid.

## Qwen fallback

Use
[`Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)
only if InternVideo3 fails the runtime or label test. Qwen's official model card
describes the checkpoint as block-128 FP8 quantisation of the BF16 30B-A3B
Instruct model. It documents vLLM and SGLang deployment and says direct
Transformers loading is not supported for those weights at the research cut-off.

Qwen's card records native 256K context and an optional extension to 1M. Treat
the larger value as an explicit extension configuration, not the default.

The fallback is attractive because vLLM provides a mature video path and can
constrain output structure. It is less attractive for this first trial because
its larger checkpoint leaves less spare GPU memory for a long video request.
No official source establishes that the proposed complete-shard request
fits the project's 45 GB L40. Hardware fit remains a project estimate to test,
not a property of the checkpoint name or download size.

Do not assume that the attention cache uses FP8 because the model weights do.
Confirm the selected cache data type in the runtime log and record it with each
measurement.

## First pass: review the whole shard

Send each 20-30 minute shard as one video item. Sample uniformly across the
whole shard and ask the model to classify contiguous scene intervals.

Start the InternVideo3 trial at 1 fps and 512x288 because that matches the
project's current low-resolution analysis and the model card's example
sampling rate. These are trial settings, not model defaults. Keep one request
at a time until memory and elapsed time have been measured.

Using the complete shard preserves the sequence of live play, replays,
close-ups and side views. Do not start with a one-to-two-hour match. Its frame
and token coverage are harder to verify, and it adds no value before shard
behaviour is known.

Pass these existing signals as compact text or structured metadata alongside
the video:

- source frame IDs and timestamps for sampled frames;
- PySceneDetect cut IDs;
- full-court confidence and detector failures;
- suspected rally start and stop intervals; and
- neighbouring scene IDs and any coarse labels already accepted by the
  whole-shard pass.

The VLM still receives the complete shard. The deterministic signals provide
extra context. They do not choose frames in advance or remove frames from the
model's available context.

## Second pass: refine each transition

After the first pass proposes a transition, send a short raw clip around that
boundary. Start with two to five seconds on each side and 6-10 fps sampling.

Ask for the source-frame boundary rather than a free-form timestamp. Convert
the returned frame ID to time in the application code. The first measurement
target is an error within 5-10 frames at 30 fps, or 10-20 frames at 60 fps.

Keep PySceneDetect hard cuts where they already provide the boundary. Use the
dense VLM pass mainly for within-shot changes such as slow-motion onset or a
camera view that changes without a hard cut.

This whole-shard and short-clip schedule is a project design. The model
authors document long-video reasoning, temporal grounding and adjustable video
sampling, but do not prescribe this badminton-specific two-pass policy.

## Structured output

Return one short JSON object per segment and no separate reasoning prose. The
following schema is provisional until the first labelled run:

```json
{
  "start_frame": 184220,
  "end_frame": 184278,
  "scene_label": "live-non-standard",
  "broadcast_phase": "live_rally",
  "view": "partial_court",
  "playback": "real_time",
  "continuity_from_previous": "same_rally",
  "data_use": "usable_alternate_view",
  "confidence": 0.86,
  "evidence_frames": ["F03", "F07"],
  "reason": "Live players and shuttle remain visible, but the full court is outside the view."
}
```

Use the existing five-way dataset label for `scene_label`: `live`,
`live-non-standard`, `replay`, `cutaway` or `other`. Keep view, playback,
continuity and data use as separate fields because a side view can be either
live or replayed.

InternVideo3's custom Transformers path cannot be assumed to enforce a JSON
schema while generating. Validate the object in the application code. Return
the validation error for one retry, then record the segment as failed if the
retry is invalid.

## Runtime limits and what the sources establish

- Run locally on project-controlled GPU hardware. Pin the model revision,
  runtime image and Python dependencies after the first successful short run.
- Keep the whole-shard input at the analysis resolution. Use 1080p only for
  later, short targeted clips if the low-resolution result misses necessary
  detail.
- Log sampled frame IDs, source duration, processor frame grid, visual tokens,
  total tokens, peak VRAM, cache dtype, CPU offload and elapsed time.
- Reject a run that silently truncates the shard or offloads enough work to make
  dataset processing impractical.
- Treat vLLM's video settings as backend-specific. Its dynamic video backend
  documents a 300-second `max_duration` default, while the generic backend
  ignores that parameter. The value is not a universal video timeout.
- Treat M²LA's roughly 50% KV-cache result as the paper's batch-1 BF16 H200
  experiment on a converted Qwen3-VL-8B backbone. It is not a fixed reduction
  for this checkpoint, GPU or request.
- Convert Docker/OCI images to Apptainer SIFs only after a short Docker test.
  Apptainer's `--nv` binds host NVIDIA devices and libraries; it does not make
  a container independent of host driver compatibility.

## First test and when to stop

Run one complete 30-minute annotated shard through InternVideo3 at the proposed
1-fps, 512x288 setting.

Before assessing labels, confirm:

- frames are sampled uniformly across the full duration;
- the token count stays inside the model's context;
- the processor kept the intended resolution;
- the model remained on GPU without CPU offload; and
- peak memory and elapsed time are plausible for the dataset volume.

Then align predictions to the existing scene annotations by source-frame
overlap. Record:

- the five-class confusion table and macro-F1;
- `live` versus `live-non-standard` confusion;
- valid JSON rate;
- boundary error in source frames;
- the proportion of boundaries inside the 5-10-frame target;
- sampled frames and tokens; and
- peak VRAM and elapsed time.

Repeat on the other two annotated videos after the first run passes. Try the
Qwen fallback once only if InternVideo3 cannot cover the complete shard or its
scene labels are not useful. Stop rather than removing the complete-shard
context or quietly changing attention precision. If both models fail, record
that the current hardware and design do not meet the requirement.

## Official and author sources

- [InternVideo3 model card and file tree](https://huggingface.co/yanziang/InternVideo3-8B-Instruct)
- [InternVideo3 author repository](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo3)
- [InternVideo3 paper](https://arxiv.org/abs/2606.12195)
- [Qwen3-VL-30B-A3B-Instruct-FP8 model card](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)
- [Qwen3-VL author repository](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631)
- [vLLM multimodal input documentation](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)
- [vLLM video backend reference](https://docs.vllm.ai/en/stable/api/vllm/multimodal/video/)
- [vLLM cache configuration](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)
- [Apptainer Docker and OCI support](https://apptainer.org/docs/user/latest/docker_and_oci.html)
- [Apptainer NVIDIA GPU support](https://apptainer.org/docs/user/latest/gpu.html)
