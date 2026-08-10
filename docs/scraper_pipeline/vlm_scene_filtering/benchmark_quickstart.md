# Issue 38 VLM benchmark quickstart

This benchmark compares these exact candidates on the same prepared video:

- `yanziang/InternVideo3-8B-Instruct` at
  `c4602918b65225650d152db2850fe34e01d21fcd`;
- `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` at
  `d9748a51ae66354c4dad665aab2c71f26cf2c8cd`.

The measured Carmack run selected neither model. The
[benchmark report](benchmark_20260810.md) records the strict-JSON and GPU-memory
failures. The commands below remain the reproduction path, and the smoke gate
must stop the full commands when it fails.

The first deployment requirement is one complete 30-minute `sset_15` shard,
source frames `[18419, 63419)`, sampled at 1 FPS and 512x288. Human timeline
labels are used only by the separate scoring command.

## 1. Prepare the videos locally

Run preparation in a disposable environment with the pinned OpenCV and
PySceneDetect versions. It writes a 25 FPS reference video, a 1 FPS model
video, and a strict manifest that maps each supplied video frame back to an
absolute source frame.

First require the frozen source identity. The Python check also hashes the
file, so a replacement with matching container metadata is rejected.

```bash
set -euo pipefail

REPO=/srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/issue-38-vlm-benchmark
SOURCE=/srv/mergerfs/main_pool/Scratch_Backup/Uni/cosc595/issue-31-shuttle-hallucination-audit-assets/downloads/yu9oyMXRGHY.mp4
ARTIFACTS=/srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/issue-38-vlm-benchmark/artifacts

SOURCE_MD5=$(md5sum "$SOURCE" | awk '{print $1}')
test "$SOURCE_MD5" = 2827bca5d829cde15591dc110f5b2904

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" \
  uv run --no-project \
  --with-requirements "$REPO/scripts/vlm_scene_benchmark/requirements-prepare.txt" \
  python -c 'import sys; from pathlib import Path; from annotator.vlm_scene_benchmark.prepare import probe_video; video = probe_video(Path(sys.argv[1])); expected = ("cbad108386055835bcd6e479adc297e18eb2d0df7ae2310857589f523bb3785f", 25.0, 149487, 640, 360); actual = (video.sha256, video.fps, video.frame_count, video.width, video.height); print(actual); assert actual == expected' \
  "$SOURCE"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" \
  uv run --no-project \
  --with-requirements "$REPO/scripts/vlm_scene_benchmark/requirements-prepare.txt" \
  python \
  -m annotator.vlm_scene_benchmark.prepare \
  --source "$SOURCE" \
  --output-dir "$ARTIFACTS/smoke" \
  --video-id sset_15 \
  --start-frame 18419 \
  --end-frame 18669

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" \
  uv run --no-project \
  --with-requirements "$REPO/scripts/vlm_scene_benchmark/requirements-prepare.txt" \
  python \
  -m annotator.vlm_scene_benchmark.prepare \
  --source "$SOURCE" \
  --output-dir "$ARTIFACTS/full" \
  --video-id sset_15 \
  --start-frame 18419 \
  --end-frame 63419
```

Preparation refuses to overwrite existing evidence. Move an old output aside
before intentionally repeating a run.

Extract the same three mapped frames from the original source, the full
reference video, and the full model video. View all three PNG files and require
the left, middle, and right images to show the same content before staging.

```bash
set -euo pipefail

PREVIEW=/srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/issue-38-vlm-benchmark/visual-preflight
REFERENCE="$ARTIFACTS/full/sset_15_f18419_f63419_512x288_reference_25fps.mp4"
MODEL_INPUT="$ARTIFACTS/full/sset_15_f18419_f63419_512x288_model_1fps.mp4"
mkdir -p "$PREVIEW"

ffmpeg -v error -nostdin -y -i "$SOURCE" \
  -vf "select='eq(n\,18419)+eq(n\,40919)+eq(n\,63394)',scale=512:288:flags=lanczos,tile=3x1" \
  -frames:v 1 "$PREVIEW/source.png"

ffmpeg -v error -nostdin -y -i "$REFERENCE" \
  -vf "select='eq(n\,0)+eq(n\,22500)+eq(n\,44975)',tile=3x1" \
  -frames:v 1 "$PREVIEW/reference.png"

ffmpeg -v error -nostdin -y -i "$MODEL_INPUT" \
  -vf "select='eq(n\,0)+eq(n\,900)+eq(n\,1799)',tile=3x1" \
  -frames:v 1 "$PREVIEW/model-input.png"
```

The accepted source is a different encode from the human review copy. The
frozen local alignment check compared every canonical boundary in the full
shard: 141 of 143 matched a detected cut at the exact frame, one was 21 frames
away, and one semantic split was 107 frames away. Human labels and the truth
CSV must not be copied into the inference environment.

## 2. Stage one isolated Carmack directory

Carmack's system Python is 3.9 and the account cannot access its Docker socket.
Use Apptainer images in `/scratch` instead. Copy only the runtime package,
requirement files, and both manifest directories under one run root. This
allowlist keeps all Git metadata, documentation, and human truth outside the
inference snapshot.

```bash
set -euo pipefail

REMOTE_ROOT=/scratch/cmarti56/issue38-vlm

ssh carmack "mkdir -p \
  '$REMOTE_ROOT/repo' '$REMOTE_ROOT/artifacts' '$REMOTE_ROOT/images' \
  '$REMOTE_ROOT/envs' '$REMOTE_ROOT/cache/apptainer' \
  '$REMOTE_ROOT/cache/huggingface' '$REMOTE_ROOT/cache/pip' \
  '$REMOTE_ROOT/cache/torch' '$REMOTE_ROOT/cache/triton' \
  '$REMOTE_ROOT/cache/flashinfer' \
  '$REMOTE_ROOT/cache/vllm' '$REMOTE_ROOT/cache/vllm-config' \
  '$REMOTE_ROOT/cache/xdg' '$REMOTE_ROOT/tmp' \
  '$REMOTE_ROOT/results' '$REMOTE_ROOT/logs'"

rsync -a --delete --delete-excluded \
  --include='/src/' \
  --include='/src/annotator/***' \
  --include='/scripts/' \
  --include='/scripts/vlm_scene_benchmark/***' \
  --exclude='*' \
  "$REPO/" "carmack:$REMOTE_ROOT/repo/"
rsync -a "$ARTIFACTS/" "carmack:$REMOTE_ROOT/artifacts/"
```

`--delete` and `--delete-excluded` are limited to the dedicated remote
runtime snapshot. They must not be used against the wider `/scratch`
directory. `--delete-excluded` removes every file outside the runtime
allowlist, including stale Git pointers, documentation, or truth copied by an
earlier version of this command.

## 3. Pull and identify the two runtime images

Run these commands on Carmack. Keep the SIF hashes with the results because OCI
tags can move.

```bash
set -euo pipefail

REMOTE_ROOT=/scratch/cmarti56/issue38-vlm
export APPTAINER_CACHEDIR="$REMOTE_ROOT/cache/apptainer"

apptainer pull "$REMOTE_ROOT/images/internvideo3-pytorch-2.8.0-cu129.sif" \
  docker://pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

apptainer pull "$REMOTE_ROOT/images/qwen3-vl-vllm-v0.11.0.sif" \
  docker://vllm/vllm-openai:v0.11.0

sha256sum "$REMOTE_ROOT/images/"*.sif
printf '%s  %s\n' \
  5861127b58769a2ad413b3ab817d61121f74566c50e8a0edc39226282be283f1 \
  "$REMOTE_ROOT/images/internvideo3-pytorch-2.8.0-cu129.sif" \
  1ee3797ccb230f937b5235b812265ba8d7e9400c48d30c49168e37515a39f03f \
  "$REMOTE_ROOT/images/qwen3-vl-vllm-v0.11.0.sif" | sha256sum --check -
```

The Qwen image pins official vLLM 0.11.0, the first stable release documented
for Qwen3-VL. The retained SIF loaded all four shards of this exact FP8
checkpoint on the L40. Keep its SIF hash with the run and refuse a different
image.

The Qwen runner downloads the exact model revision into `HF_HOME` before
starting vLLM. It then gives both the processor and the spawned engine that
local snapshot path. This prevents a child process from replacing the pinned
revision with the model repository's moving `main` branch.

Create a small writable environment over each immutable image:

```bash
set -euo pipefail

apptainer exec \
  --no-mount home,cwd \
  --bind "$REMOTE_ROOT:$REMOTE_ROOT" \
  --env PYTHONNOUSERSITE=1 \
  --env PIP_CACHE_DIR="$REMOTE_ROOT/cache/pip" \
  "$REMOTE_ROOT/images/internvideo3-pytorch-2.8.0-cu129.sif" \
  python -m venv --system-site-packages "$REMOTE_ROOT/envs/internvideo3"

apptainer exec \
  --no-mount home,cwd \
  --bind "$REMOTE_ROOT:$REMOTE_ROOT" \
  --env PYTHONNOUSERSITE=1 \
  --env PIP_CACHE_DIR="$REMOTE_ROOT/cache/pip" \
  "$REMOTE_ROOT/images/internvideo3-pytorch-2.8.0-cu129.sif" \
  "$REMOTE_ROOT/envs/internvideo3/bin/python" -m pip install \
  -r "$REMOTE_ROOT/repo/scripts/vlm_scene_benchmark/requirements-internvideo3.txt"

apptainer exec \
  --no-mount home,cwd \
  --bind "$REMOTE_ROOT:$REMOTE_ROOT" \
  --env PYTHONNOUSERSITE=1 \
  --env PIP_CACHE_DIR="$REMOTE_ROOT/cache/pip" \
  "$REMOTE_ROOT/images/qwen3-vl-vllm-v0.11.0.sif" \
  /usr/bin/python3 -m venv --system-site-packages \
  "$REMOTE_ROOT/envs/qwen3-vl-v0.11.0"

apptainer exec \
  --no-mount home,cwd \
  --bind "$REMOTE_ROOT:$REMOTE_ROOT" \
  --env PYTHONNOUSERSITE=1 \
  --env PIP_CACHE_DIR="$REMOTE_ROOT/cache/pip" \
  "$REMOTE_ROOT/images/qwen3-vl-vllm-v0.11.0.sif" \
  "$REMOTE_ROOT/envs/qwen3-vl-v0.11.0/bin/python" -m pip install \
  -r "$REMOTE_ROOT/repo/scripts/vlm_scene_benchmark/requirements-qwen3-vl.txt"
```

## 4. Run both smoke tests under tmux

Use one model at a time. The commands below keep work alive across SSH drops
and write unbuffered logs.

InternVideo3:

```bash
set -euo pipefail

REMOTE_ROOT=/scratch/cmarti56/issue38-vlm
MANIFEST="$REMOTE_ROOT/artifacts/smoke/sset_15_f18419_f18669_512x288_manifest.json"
test -f "$MANIFEST"

tmux new-session -d -s issue38-intern-smoke \
  "apptainer exec --nv --no-mount home,cwd --bind '$REMOTE_ROOT:$REMOTE_ROOT' \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH='$REMOTE_ROOT/repo/src' \
  --env HF_HOME='$REMOTE_ROOT/cache/huggingface' \
  --env TORCH_HOME='$REMOTE_ROOT/cache/torch' \
  --env TRITON_CACHE_DIR='$REMOTE_ROOT/cache/triton' \
  --env XDG_CACHE_HOME='$REMOTE_ROOT/cache/xdg' \
  --env TMPDIR='$REMOTE_ROOT/tmp' \
  '$REMOTE_ROOT/images/internvideo3-pytorch-2.8.0-cu129.sif' \
  '$REMOTE_ROOT/envs/internvideo3/bin/python' -u \
  -m annotator.vlm_scene_benchmark.run_cli \
  --backend internvideo3 --manifest '$MANIFEST' \
  --run-id internvideo3-sset15-smoke \
  --max-new-tokens 2048 \
  --out '$REMOTE_ROOT/results/internvideo3-smoke.json' \
  > '$REMOTE_ROOT/logs/internvideo3-smoke.log' 2>&1"
```

Qwen3-VL:

```bash
set -euo pipefail

REMOTE_ROOT=/scratch/cmarti56/issue38-vlm
MANIFEST="$REMOTE_ROOT/artifacts/smoke/sset_15_f18419_f18669_512x288_manifest.json"
test -f "$MANIFEST"

tmux new-session -d -s issue38-qwen-smoke \
  "apptainer exec --nv --no-mount home,cwd --bind '$REMOTE_ROOT:$REMOTE_ROOT' \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH='$REMOTE_ROOT/repo/src' \
  --env HF_HOME='$REMOTE_ROOT/cache/huggingface' \
  --env TORCH_HOME='$REMOTE_ROOT/cache/torch' \
  --env TRITON_CACHE_DIR='$REMOTE_ROOT/cache/triton' \
  --env FLASHINFER_WORKSPACE_BASE='$REMOTE_ROOT/cache/flashinfer' \
  --env VLLM_CACHE_ROOT='$REMOTE_ROOT/cache/vllm' \
  --env VLLM_CONFIG_ROOT='$REMOTE_ROOT/cache/vllm-config' \
  --env VLLM_NO_USAGE_STATS=1 \
  --env XDG_CACHE_HOME='$REMOTE_ROOT/cache/xdg' \
  --env TMPDIR='$REMOTE_ROOT/tmp' \
  '$REMOTE_ROOT/images/qwen3-vl-vllm-v0.11.0.sif' \
  '$REMOTE_ROOT/envs/qwen3-vl-v0.11.0/bin/python' -u \
  -m annotator.vlm_scene_benchmark.run_cli \
  --backend qwen3-vl --manifest '$MANIFEST' \
  --run-id qwen3-vl-sset15-smoke \
  --max-new-tokens 2048 \
  --out '$REMOTE_ROOT/results/qwen3-vl-smoke.json' \
  > '$REMOTE_ROOT/logs/qwen3-vl-smoke.log' 2>&1"
```

Check progress with `tmux ls` and `tail -f` on the matching log. Run Qwen only
after the InternVideo3 smoke process has exited and released the GPU.

Do not start the full shard until this exact gate exits successfully. It
reloads both strict records, applies the deployment gate, verifies the retained
raw-response digest, and prints the processor and runtime telemetry. It does
not read human truth.

```bash
set -euo pipefail

REMOTE_ROOT=/scratch/cmarti56/issue38-vlm

apptainer exec \
  --no-mount home,cwd \
  --bind "$REMOTE_ROOT:$REMOTE_ROOT" \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH="$REMOTE_ROOT/repo/src" \
  "$REMOTE_ROOT/images/internvideo3-pytorch-2.8.0-cu129.sif" \
  "$REMOTE_ROOT/envs/internvideo3/bin/python" - <<'PY'
from pathlib import Path

from annotator.vlm_scene_benchmark.contracts import read_run_record
from annotator.vlm_scene_benchmark.runtime import sha256_file
from annotator.vlm_scene_benchmark.scoring import deployment_failures

root = Path("/scratch/cmarti56/issue38-vlm/results")
for name in ("internvideo3-smoke.json", "qwen3-vl-smoke.json"):
    path = root / name
    record = read_run_record(path)
    failures = deployment_failures(record)
    if record.attempt_count < 1 or record.raw_response_sha256 is None:
        failures.append("no retained model response")
    else:
        raw_path = path.with_name(f"{path.stem}.attempt-{record.attempt_count}.txt")
        if record.raw_response_sha256 != sha256_file(raw_path):
            failures.append("retained raw-response digest differs")
    print(name, record.observed_sampling, record.runtime, failures)
    if failures:
        raise SystemExit(f"smoke deployment gate failed for {name}")
PY
```

## 5. Run the full shard

Keep the models sequential so each receives the same free GPU memory.

InternVideo3:

```bash
set -euo pipefail

REMOTE_ROOT=/scratch/cmarti56/issue38-vlm
MANIFEST="$REMOTE_ROOT/artifacts/full/sset_15_f18419_f63419_512x288_manifest.json"
test -f "$MANIFEST"

tmux new-session -d -s issue38-intern-full \
  "apptainer exec --nv --no-mount home,cwd --bind '$REMOTE_ROOT:$REMOTE_ROOT' \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH='$REMOTE_ROOT/repo/src' \
  --env HF_HOME='$REMOTE_ROOT/cache/huggingface' \
  --env TORCH_HOME='$REMOTE_ROOT/cache/torch' \
  --env TRITON_CACHE_DIR='$REMOTE_ROOT/cache/triton' \
  --env XDG_CACHE_HOME='$REMOTE_ROOT/cache/xdg' \
  --env TMPDIR='$REMOTE_ROOT/tmp' \
  '$REMOTE_ROOT/images/internvideo3-pytorch-2.8.0-cu129.sif' \
  '$REMOTE_ROOT/envs/internvideo3/bin/python' -u \
  -m annotator.vlm_scene_benchmark.run_cli \
  --backend internvideo3 --manifest '$MANIFEST' \
  --run-id internvideo3-sset15-full \
  --max-new-tokens 32768 \
  --out '$REMOTE_ROOT/results/internvideo3-full.json' \
  > '$REMOTE_ROOT/logs/internvideo3-full.log' 2>&1"
```

Qwen3-VL, after the InternVideo3 process exits:

```bash
set -euo pipefail

REMOTE_ROOT=/scratch/cmarti56/issue38-vlm
MANIFEST="$REMOTE_ROOT/artifacts/full/sset_15_f18419_f63419_512x288_manifest.json"
test -f "$MANIFEST"

tmux new-session -d -s issue38-qwen-full \
  "apptainer exec --nv --no-mount home,cwd --bind '$REMOTE_ROOT:$REMOTE_ROOT' \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH='$REMOTE_ROOT/repo/src' \
  --env HF_HOME='$REMOTE_ROOT/cache/huggingface' \
  --env TORCH_HOME='$REMOTE_ROOT/cache/torch' \
  --env TRITON_CACHE_DIR='$REMOTE_ROOT/cache/triton' \
  --env FLASHINFER_WORKSPACE_BASE='$REMOTE_ROOT/cache/flashinfer' \
  --env VLLM_CACHE_ROOT='$REMOTE_ROOT/cache/vllm' \
  --env VLLM_CONFIG_ROOT='$REMOTE_ROOT/cache/vllm-config' \
  --env VLLM_NO_USAGE_STATS=1 \
  --env XDG_CACHE_HOME='$REMOTE_ROOT/cache/xdg' \
  --env TMPDIR='$REMOTE_ROOT/tmp' \
  '$REMOTE_ROOT/images/qwen3-vl-vllm-v0.11.0.sif' \
  '$REMOTE_ROOT/envs/qwen3-vl-v0.11.0/bin/python' -u \
  -m annotator.vlm_scene_benchmark.run_cli \
  --backend qwen3-vl --manifest '$MANIFEST' \
  --run-id qwen3-vl-sset15-full \
  --max-new-tokens 32768 \
  --out '$REMOTE_ROOT/results/qwen3-vl-full.json' \
  > '$REMOTE_ROOT/logs/qwen3-vl-full.log' 2>&1"
```

## 6. Score retained records

Run scoring outside the model environments. This is the first command that may
read human labels.

```bash
set -euo pipefail

REPO=/srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/issue-38-vlm-benchmark
TRUTH="$REPO/docs/scraper_pipeline/broadcast_nonstandard_camera_id/data/sset_15_broadcast_timeline_labels.csv.gz"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" \
  /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/badminton_cv_annotator/.venv/bin/python \
  -m annotator.vlm_scene_benchmark.score_cli \
  path/to/run.json "$TRUTH" --out path/to/score.json
```

The score CLI returns status 3 when deployment evidence fails. Accuracy remains
unset for truncated input, missing telemetry, or prohibited CPU offload.
