# Issue 38 local VLM benchmark

*Run date: 10 August 2026. Report completed 11 August 2026.*

## Decision

Neither candidate passes the deployment gate for the fixed 30-minute test on
the project's current GPUs. Do not integrate either model into the annotator
yet.

The 10-frame `yanziang/InternVideo3-8B-Instruct` smoke run fits on Carmack's
L40 and processes its requested smoke frame grid. It returned the same
Markdown-fenced JSON after the one allowed correction retry. The strict
response contract therefore rejected the run.

`Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` loads correctly with official vLLM 0.11.0.
It cannot reserve enough BF16 KV cache for the complete-shard context on the
L40. The other documented project GPUs have less memory.

No accuracy or boundary result is reported. Both models failed before the
accuracy gate, so a confusion table or F1 value would be misleading.

## Fixed test

| Item | Value |
| --- | --- |
| Source | `yu9oyMXRGHY.mp4` |
| Source SHA-256 | `cbad108386055835bcd6e479adc297e18eb2d0df7ae2310857589f523bb3785f` |
| Full source range | `[18419, 63419)` at 25 FPS |
| Full source duration | 1,800 seconds |
| Model input | 1,800 frames at 1 FPS and 512x288 |
| Model-input SHA-256 | `70c71ff8b45339a2829e248ad1a87b056996cb06e36908b358bda537d05628ae` |
| Smoke source range | `[18419, 18669)` |
| Smoke model input | 10 frames at 1 FPS and 512x288 |
| GPU | NVIDIA L40, 46,068 MiB total |
| CPU offload | Prohibited |
| Human labels | Excluded from inference |

The source, reference video, and model video matched at the first, middle, and
last sampled positions. All staged artifact hashes matched on Carmack. The
full Qwen input requires 129,600 visual tokens before prompt text.

## Results

| Candidate | Exact runtime | Last completed stage | Peak VRAM | Elapsed | Result |
| --- | --- | --- | ---: | ---: | --- |
| InternVideo3 8B | Transformers 4.57.3 | Smoke generation | 22,804 MiB | 208.46 s | Failed strict JSON after retry |
| Qwen3-VL 30B-A3B FP8 | vLLM 0.11.0 | Model load and cache sizing | 40,758 MiB | 133.49 s | Full context cannot fit |

### InternVideo3

- Model revision:
  `c4602918b65225650d152db2850fe34e01d21fcd`.
- Runtime SIF SHA-256:
  `5861127b58769a2ad413b3ab817d61121f74566c50e8a0edc39226282be283f1`.
- The processor consumed all 10 requested frames on a uniform grid.
- Observed resolution was 512x288.
- The request used 720 visual tokens and 2,804 total input tokens.
- BF16 cache was used with no CPU offload.
- Both responses had SHA-256
  `22b49e89d60c21bda69301736aaea6d934c6f2d00bcbcd1d60999a121a8cdf2a`.

The two responses were byte-identical. Each contained a complete JSON object
inside a `json` Markdown fence. The parser rejected the first byte as invalid
JSON, and the correction retry did not remove the fence. The full run was not
started because the frozen smoke gate failed.

The generated content looked usable enough to justify one narrow follow-up.
That follow-up must first approve deterministic Markdown-fence removal as a
contract change. It must retain the original response before normalization.

### Qwen3-VL

- Model revision:
  `d9748a51ae66354c4dad665aab2c71f26cf2c8cd`.
- Official runtime image: `vllm/vllm-openai:v0.11.0`.
- Runtime SIF SHA-256:
  `1ee3797ccb230f937b5235b812265ba8d7e9400c48d30c49168e37515a39f03f`.
- Runtime code revision:
  `0ad9bd98e21c23b0a1c5788f0586905eb6779df7`.
- Model loading used 30.3924 GiB and took 31.98 seconds.
- vLLM loaded all four checkpoint shards without changing the model.
- The engine selected the BF16 model dtype for KV cache through its documented
  `auto` setting.
- CPU offload and CPU swap were both explicitly zero.

After model loading and compilation, vLLM reported 6.30 GiB available for KV
cache. It requires 24.00 GiB for the pinned 262,144-token context. It estimated
a maximum supported length of 68,800 tokens on this L40.

The complete video alone requires 129,600 visual tokens. It therefore exceeds
the measured limit before prompt text or generated output is counted. Starting
the full command would repeat the same deterministic capacity failure.

The documented alternatives are Bourbaki's 40-GB A100 and Engelbart's 16-GB
V100. Neither has more memory than Carmack's L40.

## Runtime choice

The canonical Qwen result uses the official stable version named by the Qwen
deployment guidance. vLLM 0.11.0 loaded the exact model successfully. Released
vLLM 0.25.1 and a pinned nightly were diagnostic attempts, and both rejected
the checkpoint's fused-MoE tensors during loading.

The model is still exactly
`Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`. The long hexadecimal value beside its
name is its pinned Hugging Face revision. It is not another model version.

## Evidence

The compressed records are exact copies of the validated Carmack result files:

- [InternVideo3 smoke record](data/benchmark_20260810/internvideo3-smoke-run.json.gz),
  uncompressed SHA-256
  `67a746f50c6ad4850c64025a27773555d1b7358464e1c1af9abb488b5da1f533`.
- [Qwen stable capacity record](data/benchmark_20260810/qwen3-vl-stable-capacity-run.json.gz),
  uncompressed SHA-256
  `e2b18019cc308164a4c14b91579f75dc8fb67ec91711f7abfdd8b680ecec96a3`.
- [InternVideo3 first response](data/benchmark_20260810/internvideo3-attempt-1.txt.gz)
  and [correction response](data/benchmark_20260810/internvideo3-attempt-2.txt.gz),
  each with uncompressed SHA-256
  `22b49e89d60c21bda69301736aaea6d934c6f2d00bcbcd1d60999a121a8cdf2a`.
- [Qwen stable capacity log](data/benchmark_20260810/qwen3-vl-stable-capacity.log.gz),
  uncompressed SHA-256
  `d15f096acdf77fc83bba550c5b51b6b4361de0afe9adb4acee25d7ff1b395072`.

The linked Qwen log records the exact snapshot path, BF16 engine dtype, all
four loaded shards, model-memory measurement, and cache-sizing failure. The
linked InternVideo3 files retain both fenced responses byte for byte. Human
truth was never copied into the inference directory.

## Next bounded experiment

There are two honest next options:

1. Approve a parser change that unwraps one plain Markdown JSON fence. Then run
   the complete shard through InternVideo3 and score it.
2. Keep the response contract unchanged and rerun Qwen on an 80-GB-class GPU,
   or a supported multi-GPU host. This is a conservative hardware estimate
   based on the measured cache shortfall.

Lower sampling, FP8 KV cache, and CPU offload would define different
experiments. They should not be reported as this benchmark passing.
