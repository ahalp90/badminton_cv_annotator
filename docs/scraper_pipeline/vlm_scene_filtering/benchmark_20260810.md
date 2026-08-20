# Issue 38 local VLM benchmark

*Run dates: 10-14 August 2026. Report updated 14 August 2026.*

## Decision

Do not integrate either tested VLM into the scene-filtering pipeline. Both
models now have measured GPU results, but neither produced useful scene labels.

InternVideo3 completed the full 20-minute shard on Sutherland. It covered all
1,200 requested model frames at 1 FPS, used BF16 cache with no CPU offload, and
peaked at 41,079 MiB. It predicted `live` for every one of the 30,000 source
frames. Accuracy was 25.12%, macro-F1 was 0.0803, and it found none of the 93
truth boundaries. The truth majority class, `cutaway`, covers 54.46% of this
shard, so the model was also worse than that simple baseline.

Qwen3-VL completed the planned 10-second boundary probe on the same L40. It
covered all 50 requested model frames at 5 FPS, used BF16 cache with no CPU
offload or swap, and peaked at 40,831 MiB. It predicted `other` for all 250
source frames. The clip contains 125 `live-non-standard` frames followed by
125 `cutaway` frames. Accuracy and macro-F1 were both zero, and it found none
of the one truth boundary.

The result is enough to stop this experiment. Repeating these settings on the
other labelled videos would spend GPU time without evidence that either model
can separate the five scene classes. A new run needs a materially different
model, prompt, or study design.

## Fixed inputs

| Item | InternVideo3 long pass | Qwen3-VL boundary probe |
| --- | --- | --- |
| Source | `yu9oyMXRGHY.mp4` | `yu9oyMXRGHY.mp4` |
| Source SHA-256 | `cbad108386055835bcd6e479adc297e18eb2d0df7ae2310857589f523bb3785f` | same |
| Source frames | `[18419, 48419)` | `[20695, 20945)` |
| Source duration | 1,200 seconds | 10 seconds |
| Model sampling | 1 FPS, 1,200 frames | 5 FPS, 50 frames |
| Resolution | 512x288 | 512x288 |
| Prepared-input SHA-256 | `f5b93940aae493bff88fbde4b04b15e86356d0eee618125519bf79e0cc4560fc` | `bec483acfdbb98938b2200aefaa8ffb4275e5b23de15d14287e55cb44a3f8fb0` |
| Human truth in inference directory | No | No |

The source, reference video, and prepared model video matched at the first,
middle, and last sampled positions. The truth file was read only by the local
scoring step after inference.

## Results

| Candidate | Runtime | Coverage | Elapsed | Peak VRAM | Accuracy | Macro-F1 | Boundaries |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| InternVideo3 8B | Transformers 4.57.3 | 1,200/1,200 | 824.05 s | 41,079 MiB | 25.12% | 0.0803 | 0/93 |
| Qwen3-VL 30B-A3B FP8 | vLLM 0.11.0 | 50/50 | 225.39 s | 40,831 MiB | 0% | 0 | 0/1 |

Both runs used one generation attempt. Both frame grids were complete and
uniform. InternVideo3 used 86,400 visual tokens and 101,349 total input tokens.
Qwen used 3,600 visual tokens and 4,798 total input tokens.

### InternVideo3

- Model revision:
  `c4602918b65225650d152db2850fe34e01d21fcd`.
- Runtime SIF SHA-256:
  `fd1c42ea24386dde021f12c0fe9458f0d4f5f43ea97af2ad19c2b3ea9925c76a`.
- The fixed-width response contained 1,316 complete codes before ending
  mid-code at the generation limit.
- The parser used the first 1,200 complete codes and ignored only the 116
  complete codes after full input coverage.
- Every accepted code was `LBRFRS9B`. This decodes to scene label `live` and
  broadcast phase `between_rallies`.

Earlier runs exposed two independent runtime problems. A retry could retain
CUDA tensors from the first generation, and a complete fixed-width prefix
could be enclosed in truncated JSON. The runner now releases generation
tensors before retrying and accepts a bounded complete prefix only when it
covers the exact requested frame grid.

The retained InternVideo3 record uses the older schema. Its
`first_attempt_valid_json` field says `true`, although the authenticated raw
response ends in truncated JSON. The current provenance gate detects this
legacy metadata mismatch. The raw-response digest still matches the record,
the bounded prefix reconstructs the retained segment, and two score replays
were byte-identical. The semantic score is usable with that caveat. The legacy
record was not rewritten.

### Qwen3-VL

- Model revision:
  `d9748a51ae66354c4dad665aab2c71f26cf2c8cd`.
- Official runtime image: `vllm/vllm-openai:v0.11.0`.
- Runtime SIF SHA-256:
  `1cf06bf5a8a7bd5a2b2c469f0e72ac150f0781c126b593c7fcd9d7df4eb34d37`.
- The exact FP8 checkpoint ran with BF16 KV cache. CPU offload and vLLM swap
  were both zero.
- The response passed the strengthened provenance and deployment gate on its
  first attempt.
- Every accepted code was `OBRFRS9G`. This decodes to scene label `other` and
  broadcast phase `between_rallies`.

The original whole-shard Qwen request still cannot fit on a 48-GB L40. The
retained capacity test measured 6.30 GiB available for KV cache and 24.00 GiB
required for the 262,144-token configuration. Its estimated maximum was
68,800 tokens, below the video's 129,600 visual tokens before prompt text.
The completed boundary probe used a separate 16,384-token configuration. It
tests the recommended short second pass without claiming whole-shard support.

## Evidence

The following files are deterministic gzip copies of the retained run data.
Each digest is for the uncompressed content.

| Evidence | Uncompressed SHA-256 |
| --- | --- |
| [InternVideo3 run record](data/benchmark_20260810/internvideo3-long20-run.json.gz) | `11875041edc5b9f2db11bbec0258dfdd96fbfdcd2ed99cf8ad35638f1cfcf639` |
| [InternVideo3 raw response](data/benchmark_20260810/internvideo3-long20-attempt-1.txt.gz) | `3ded3e69ed49a6f4ddbf10e76f91ed339e8e007c537c6574a5eba27d52e134ac` |
| [InternVideo3 log](data/benchmark_20260810/internvideo3-long20.log.gz) | `36a1831365e4da394b58f975add14e0db3260ad08b70eab61aa407e29e15653c` |
| [InternVideo3 score](data/benchmark_20260810/internvideo3-long20-score.json.gz) | `de05a1ce11638e62ea42373091d2b2083e443176777ebebdb79781ee6a679766` |
| [Qwen run record](data/benchmark_20260810/qwen3-vl-fine-run.json.gz) | `fa65367b2b40d73c3a6a24d1b8a9672a9e6bfbbd2a81e601a18efb68c07bae1b` |
| [Qwen raw response](data/benchmark_20260810/qwen3-vl-fine-attempt-1.txt.gz) | `2f46d1c733da7aa6e9bd8e58a4b0f4bd906d76bb354b970bd7812b6a35a6c5c8` |
| [Qwen log](data/benchmark_20260810/qwen3-vl-fine.log.gz) | `79960a777f7d805533c01e7a81a2dbab48ab515bb4a43410abf807ad821ffc18` |
| [Qwen score](data/benchmark_20260810/qwen3-vl-fine-score.json.gz) | `9a521c2db2cbb2615ecc9bb925b50ec74ef36b0c523ef2c8549d641a4df6f779` |

The earlier Carmack smoke and Qwen capacity evidence remains linked in this
directory. The full retained Sutherland run directories are outside Git at:

- `issue-38-vlm-benchmark/runs/issue38-intern-framecodes-prefix-v6-20260813T2256`;
- `issue-38-vlm-benchmark/runs/issue38-4e051e6fa1cce60f`.

## Integration consequence

No model output is wired into `raw_exclusion_mask`. The existing pipeline
behaviour remains unchanged. A later experiment can reuse the benchmark
contracts and the existing injection point without silently enabling these
predictions.
