# Local VLM recommendation for badminton scene filtering

*Research cut-off: 6 August 2026. Local results updated 14 August 2026.*

## Measured local result

Do not integrate either tested model. The revised InternVideo3 long pass and
the Qwen short boundary probe both completed on Sutherland, but both failed
the label-quality test. See the [benchmark report](benchmark_20260810.md) for
the retained records, raw responses, logs, and deterministic scores.

- InternVideo3 covered all 1,200 requested frames at 1 FPS and 512x288 with no
  CPU offload. It took 824.05 seconds and peaked at 41,079 MiB on the L40.
- The fixed-width response contained 1,316 complete codes. The parser accepted
  the required first 1,200 and ignored only the post-coverage continuation.
  Every accepted label was `live`. Accuracy was 25.12%, macro-F1 was 0.0803,
  and it found none of 93 truth boundaries.
- Qwen covered the complete 10-second boundary clip at 5 FPS. It took 225.39
  seconds and peaked at 40,831 MiB with BF16 KV cache, no CPU offload, and no
  swap. Every label was `other`. Accuracy and macro-F1 were zero, and it found
  none of the one truth boundary.
- Qwen's original whole-shard request still cannot fit on the L40. The short
  probe used the planned 16,384-token boundary configuration and does not
  establish whole-shard support.

These results satisfy the planned stop condition. Do not repeat the same
settings on the other two annotated videos. Resume only for a materially
different model, prompt, or study design. No 4-bit substitute was used.

## Tested design

The pre-test recommendation was to trial
`yanziang/InternVideo3-8B-Instruct` on a complete shard, then use
`Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` for one short boundary probe if the long
model's labels were poor. Both parts of that design are now measured.

The tested design used two calls for different jobs:

1. A complete 20-30 minute shard establishes broadcast phase, camera view,
   playback speed and continuity.
2. A short, densely sampled boundary clip refines a candidate transition to
   source-frame precision.

The model would use existing cuts, court evidence and suspected rally spans as
context. It would not replace player, shuttle or court tracking. The local
measurements show that the current prompt and checkpoints are not accurate
enough for that role. The two-pass design remains a project experiment, not a
published guarantee from either model author.

GitHub issue [#38](https://github.com/ahalp90/badminton_cv_annotator/issues/38)
tracks implementation and evaluation.

## Why InternVideo3 was first

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
project run therefore logged the actual sampled frame IDs, token count,
resolution, and coverage instead of assuming the processor consumed the
requested grid.

## Qwen fallback rationale

The fallback candidate was
[`Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)
because InternVideo3 failed the label test. Qwen's official model card
describes the checkpoint as block-128 FP8 quantisation of the BF16 30B-A3B
Instruct model. It documents vLLM and SGLang deployment and says direct
Transformers loading is not supported for those weights at the research cut-off.

Qwen's card records native 256K context and an optional extension to 1M. Treat
the larger value as an explicit extension configuration, not the default.

The fallback was attractive because vLLM provides a mature video path and can
constrain output structure. It was less attractive for this trial because
its larger checkpoint leaves less spare GPU memory for a long video request.
No official source establishes that the proposed complete-shard request
fits the project's 45 GB L40. Hardware fit remains a project estimate to test,
not a property of the checkpoint name or download size.

Do not assume that the attention cache uses FP8 because the model weights do.
Confirm the selected cache data type in the runtime log and record it with each
measurement.

## Long-pass test design

The long pass sent the 20-minute shard as one video item. It sampled uniformly
across the whole shard and asked the model to classify contiguous intervals.

The InternVideo3 trial used 1 fps and 512x288 because that matches the
project's current low-resolution analysis and the model card's example
sampling rate. These are trial settings, not model defaults. Keep one request
at a time until memory and elapsed time have been measured.

Using the complete shard preserves the sequence of live play, replays,
close-ups and side views. Do not start with a one-to-two-hour match. Its frame
and token coverage are harder to verify, and it adds no value before shard
behaviour is known.

A future revision could pass these existing signals as compact text or
structured metadata alongside the video:

- source frame IDs and timestamps for sampled frames;
- PySceneDetect cut IDs;
- full-court confidence and detector failures;
- suspected rally start and stop intervals; and
- neighbouring scene IDs and any coarse labels already accepted by the
  whole-shard pass.

The VLM still receives the complete shard. The deterministic signals provide
extra context. They do not choose frames in advance or remove frames from the
model's available context.

## Boundary-probe test design

The boundary probe sent a short raw clip around a reviewed transition. The
general design uses two to five seconds on each side and denser sampling.

It asked for the source-frame boundary rather than a free-form timestamp. Convert
the returned frame ID to time in the application code. The first measurement
target is an error within 5-10 frames at 30 fps, or 10-20 frames at 60 fps.

The design retains PySceneDetect hard cuts where they already provide the
boundary. The dense VLM pass targets within-shot changes such as slow-motion
onset or a camera view that changes without a hard cut.

This whole-shard and short-clip schedule is a project design. The model
authors document long-video reasoning, temporal grounding and adjustable video
sampling, but do not prescribe this badminton-specific two-pass policy.

## Structured output

The benchmark requested one fixed-width code for every sampled frame. The
response has one `frames` array and no other key:

```json
{
  "frames": ["LBRFRS9B", "LLRFRS9R"]
}
```

Each eight-character code records scene, phase, playback, view, continuity,
data use, confidence, and visible reason in that order. The scene character
uses the existing five-way labels: `L` for `live`, `N` for
`live-non-standard`, `R` for `replay`, `C` for `cutaway`, and `O` for `other`.
The prompt defines the allowed character for every other position.

The array must match the ordered sampled-frame grid exactly. The parser maps
each code back to its absolute source-frame interval and merges adjacent equal
states into the retained segment contract. The model must not return frame
numbers or segment objects.

InternVideo3's custom Transformers path cannot be assumed to enforce a JSON
schema while generating. The runner validates the response, includes the
validation error in one correction request, and records the run as failed if
the correction is invalid. A complete fixed-code prefix is accepted only when
it covers the exact requested frame grid.

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

## Completed test and stop condition

The complete 20-minute InternVideo3 shard and the Qwen boundary probe are now
complete. Both runs confirmed:

- frames are sampled uniformly across the full duration;
- the token count stays inside the model's context;
- the processor kept the intended resolution;
- the model remained on GPU without CPU offload; and
- peak memory and elapsed time are plausible for the dataset volume.

Their predictions were aligned to the existing scene annotations by
source-frame overlap. The retained scores record:

- the five-class confusion table and macro-F1;
- `live` versus `live-non-standard` confusion;
- valid JSON rate;
- boundary error in source frames;
- the proportion of boundaries inside the 5-10-frame target;
- sampled frames and tokens; and
- peak VRAM and elapsed time.

The first long run did not pass the label test, and the one planned Qwen probe
also failed. The experiment therefore stops before the other two annotated
videos. The current checkpoints, prompt, hardware, and two-pass design do not
meet the scene-filtering requirement.

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
