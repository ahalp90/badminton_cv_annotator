# Qwen3.8 comparison

**Result:** Qwen3.8 works on the retained experiments and is worth keeping as
an experimental alternative. It is not a safe production replacement for the
scene or rally-start decisions.

The comparison used `Qwen/Qwen3.8-27B-FP8` at revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` with vLLM 0.17.0. Thinking was
disabled so the model used the same direct JSON interface as the retained
models. Every Qwen3.8 reply parsed, and every case used the complete 120-frame
input grid.

The project timing tolerance is the retained ±5-frame tolerance at 30 frames
per second, scaled to each clip's frame rate.

## Rally starts

The same 32 reviewed rally starts produced:

| Measure | Old Qwen | InternVideo3 | Qwen3.8 |
|---|---:|---:|---:|
| Server correct | 14/32 | **23/32** | 20/32 |
| Serve-state label correct | **19/32** | **19/32** | 17/32 |
| Visible contact within project tolerance | 1/19 | 1/19 | **11/19** |
| Exact frame claimed when contact was not visible | 13/13 | 13/13 | **11/13** |

Qwen3.8 sits between the retained models on server attribution. Its contact
timing is materially better when the reviewed truth says contact is visible.
The model cannot provide the visibility gate itself: it labelled serve state
correctly in only 17 cases and still supplied a frame in 11 of the 13 cases
where contact was not visible.

The useful new lead is therefore timing behind an independent visibility
check. It is not evidence for using Qwen3.8 as an end-to-end serve judge.

## Short scene clips

The strongest retained scene arm used the same 463 clips and the same prompt.
The main comparison excludes 116 very short boundary targets, leaving 347
material cases.

| Intended decision | Old Qwen | InternVideo3 | Qwen3.8 |
|---|---:|---:|---:|
| Keep ordinary full-court live clips | **288/290** | 270/290 | 278/290 |
| Keep unusual-view live clips | 0/10 | **6/10** | 0/10 |
| Send clips containing non-live footage for checking | 15/47 | 21/47 | **22/47** |
| Send pure replay clips for checking | 1/25 | **5/25** | 4/25 |
| Correct material route | **303/347** | 297/347 | 300/347 |

Qwen3.8 gives the most balanced result of the three on ordinary live retention
and non-live recall. The one-case non-live lead over Intern is small. It still
rejects every unusual-view live target and accepts 21 of 25 pure replays as
live. Short local clips remain insufficient for a safe final scene decision.

## Decision

Keep Qwen3.8 as a reproducible experiment backend. Do not connect it to the
production annotator from this evidence.

If serve timing becomes important, the next bounded test should supply an
independent visibility gate and matched controls that move or remove any
candidate marker. That would test whether the 11/19 timing result follows the
physical contact rather than a stable cue in the clip.

The compact machine-readable record, including model, runtime, prompt, input
and truth hashes, is
[`experiments/results/qwen3_8_comparison.json`](experiments/results/qwen3_8_comparison.json).
