# Issue 38 VLM benchmark quickstart

The benchmark is complete. The retained results do not support integrating
either model. This guide records the reproducible runner for future experiments
with a materially changed model, prompt, or study design.

The runner pins these exact candidates:

- `yanziang/InternVideo3-8B-Instruct` at
  `c4602918b65225650d152db2850fe34e01d21fcd`;
- `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` at
  `d9748a51ae66354c4dad665aab2c71f26cf2c8cd`.

InternVideo3 uses source frames `[18419, 48419)` at 1 FPS. Qwen uses the
10-second boundary clip `[20695, 20945)` at 5 FPS. Both inputs are 512x288.
The Qwen clip is centred on frame 20820, but human truth is never copied into
the inference directory.

## Run on Sutherland

Sutherland is reached through Turing and has a separate home-directory
mapping. The wrapper stages its own restricted repository snapshot, inputs,
environments, images, and caches under `/scratch/cmarti56/issue38-vlm`.

An interactive login uses two commands:

```bash
ssh turing
ssh sutherland
```

The benchmark also needs `rsync` to follow the same nested route. Create an
SSH-compatible wrapper outside the repository:

```bash
set -euo pipefail

SSH_WRAPPER=/tmp/issue38-sutherland-rsh
cat >"$SSH_WRAPPER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
readonly target_host=${1:?rsync did not provide a remote host}
shift
test "$target_host" = sutherland
printf -v inner_command '%q ' "$@"
exec ssh -F /home/curtis/.ssh/config -o BatchMode=yes turing \
  "exec ssh -o BatchMode=yes sutherland $inner_command"
EOF
chmod 700 "$SSH_WRAPPER"
```

Inspect every resolved path and pin without changing local or remote state:

```bash
scripts/vlm_scene_benchmark/run_carmack.sh all \
  --plan \
  --remote-host sutherland \
  --remote-root /scratch/cmarti56/issue38-vlm \
  --ssh-command /tmp/issue38-sutherland-rsh
```

Run the full workflow only while the GPU is idle:

```bash
scripts/vlm_scene_benchmark/run_carmack.sh all \
  --remote-host sutherland \
  --remote-root /scratch/cmarti56/issue38-vlm \
  --ssh-command /tmp/issue38-sutherland-rsh
```

The wrapper validates the frozen source, prepares or reuses the exact inputs,
stages only runtime code and video artifacts, verifies the runtime images,
creates pinned environments, and starts each GPU task in `tmux`. It runs one
model at a time, collects retained evidence, applies the provenance and
deployment gate, then scores successful records locally against human truth.

A failed model does not block the other model. A disconnected local SSH
session does not stop an active remote `tmux` task. Re-running the same command
reconnects to active work or reuses complete immutable evidence. If raw model
evidence exists without a complete record, the wrapper preserves it and prints
a retry command with a new run tag.

## Resume and inspect

The first positional argument selects a bounded stage:

| Command | Purpose |
| --- | --- |
| `check` | Check the source, tools, remote host, scratch space, and GPU state. |
| `prepare` | Create and validate the exact smoke, long, and boundary inputs. |
| `stage` | Copy the inference-only snapshot and prepared videos. |
| `setup` | Verify images and create or reuse all pinned environments. |
| `smoke` | Run both smoke tests, collect evidence, and gate each model. |
| `full` | Run each candidate whose own smoke record passes. |
| `qwen-fine` | Verify setup, run or reuse Qwen smoke, run the boundary probe, and score it. |
| `collect` | Download logs, records, statuses, and provenance. |
| `score` | Gate and score retained candidate records locally. |
| `status` | Show retained statuses and active remote sessions. |

Keep the same path options and run tag when resuming. For example:

```bash
scripts/vlm_scene_benchmark/run_carmack.sh status \
  --remote-host sutherland \
  --remote-root /scratch/cmarti56/issue38-vlm \
  --ssh-command /tmp/issue38-sutherland-rsh \
  --run-tag issue38-4e051e6fa1cce60f
```

## Fixed identities

The source must match both recorded digests:

```text
MD5     2827bca5d829cde15591dc110f5b2904
SHA-256 cbad108386055835bcd6e479adc297e18eb2d0df7ae2310857589f523bb3785f
```

The verified Sutherland SIF digests are:

```text
InternVideo3 fd1c42ea24386dde021f12c0fe9458f0d4f5f43ea97af2ad19c2b3ea9925c76a
Qwen3-VL    1cf06bf5a8a7bd5a2b2c469f0e72ac150f0781c126b593c7fcd9d7df4eb34d37
```

The runner refuses a changed source, manifest, model revision, image digest,
backend version, required package, cache dtype, CPU-offload setting, frame
grid, resolution, or retained response digest.

## Runtime rules

- Keep one GPU task active at a time.
- Use BF16 cache for both backends.
- Keep CPU offload and vLLM swap at zero.
- Do not copy the truth CSV into the staged inference snapshot.
- Do not edit or replace a retained raw response or result record.
- Use a new run tag when code or prepared inputs change.
- Keep the complete logs and raw responses for failed runs.

InternVideo3 may use the complete fixed-width frame-code prefix when it covers
the exact requested grid. Any incomplete grid still fails. A retry releases
per-generation CUDA tensors before rebuilding inputs.

Qwen's original whole-shard request cannot fit on the L40. Its boundary probe
uses `--max-model-len 16384`. That short configuration must not be reported as
whole-shard support.

## Retained outcome

The completed InternVideo3 run covered 1,200 model frames in 824.05 seconds
and peaked at 41,079 MiB. It scored 25.12% accuracy and 0.0803 macro-F1, with
zero of 93 truth boundaries found.

The completed Qwen boundary probe covered 50 model frames in 225.39 seconds
and peaked at 40,831 MiB. It scored zero accuracy and macro-F1, with zero of
one truth boundary found.

See the [benchmark report](benchmark_20260810.md) for exact records, raw
responses, logs, scores, digests, and the integration decision.
