#!/usr/bin/env bash
set -u -o pipefail

readonly TASK=${1:?usage: remote_task.sh TASK REMOTE_ROOT RUN_ROOT [BACKEND] [STAGE]}
readonly REMOTE_ROOT=${2:?usage: remote_task.sh TASK REMOTE_ROOT RUN_ROOT [BACKEND] [STAGE]}
readonly RUN_ROOT=${3:?usage: remote_task.sh TASK REMOTE_ROOT RUN_ROOT [BACKEND] [STAGE]}
readonly BACKEND=${4:-}
readonly STAGE=${5:-}

atomic_status() {
  local path=$1
  local value=$2
  printf '%s\n' "$value" >"$path.tmp"
  mv -- "$path.tmp" "$path"
}

runtime_environment_ready() (
  local runtime_environment="$RUN_ROOT/control/runtime.env"
  [[ -f "$runtime_environment" ]] || return 1
  source "$runtime_environment" || return 1
  [[ -x ${INTERN_ENVIRONMENT:-}/bin/python || -L ${INTERN_ENVIRONMENT:-}/bin/python ]] || return 1
  [[ -f ${INTERN_ENVIRONMENT:-}/.issue38-ready ]] || return 1
  [[ -d ${INTERN_FFMPEG_ENVIRONMENT:-}/lib ]] || return 1
  [[ -f ${INTERN_FFMPEG_ENVIRONMENT:-}/.issue38-ready ]] || return 1
  [[ -x ${QWEN_ENVIRONMENT:-}/bin/python ]] || return 1
  [[ -f ${QWEN_ENVIRONMENT:-}/.issue38-ready ]] || return 1
)

verify_snapshot() {
  cd "$RUN_ROOT" || return
  sha256sum --check --quiet control/staged.sha256
}

run_setup() {
  local log="$RUN_ROOT/logs/setup.log"
  local ready="$RUN_ROOT/status/setup.ok"
  local last_exit="$RUN_ROOT/status/setup.last-exit"
  if [[ -f "$ready" && $(<"$ready") == 0 ]] && runtime_environment_ready; then
    echo "Setup already passed."
    return 0
  fi
  if [[ -f "$ready" ]]; then
    mv -- "$ready" "$ready.incomplete" || return 71
  fi
  if [[ -f "$last_exit" ]]; then
    mv -- "$last_exit" "$last_exit.previous" || return 71
  fi
  {
    printf '\n[%s] setup start\n' "$(date --iso-8601=seconds)"
    verify_snapshot && \
      "$RUN_ROOT/repo/scripts/vlm_scene_benchmark/setup_remote.sh" "$REMOTE_ROOT" "$RUN_ROOT"
  } >>"$log" 2>&1
  local status=$?
  if ((status == 0)); then
    atomic_status "$ready" 0 || return 71
  else
    atomic_status "$last_exit" "$status" || return 71
  fi
  return "$status"
}

benchmark_paths() {
  case "$STAGE" in
    smoke)
      MANIFEST="$RUN_ROOT/artifacts/smoke/sset_15_f18419_f18669_512x288_manifest.json"
      MAX_NEW_TOKENS=2048
      QWEN_MAX_MODEL_LEN=8192
      ;;
    long20)
      [[ $BACKEND == internvideo3 ]] || {
        echo "Only InternVideo3 may run the 20-minute stage." >&2
        return 2
      }
      MANIFEST="$RUN_ROOT/artifacts/long20/sset_15_f18419_f48419_512x288_manifest.json"
      MAX_NEW_TOKENS=9216
      ;;
    fine)
      [[ $BACKEND == qwen3-vl ]] || {
        echo "Only Qwen3-VL may run the fine stage." >&2
        return 2
      }
      MANIFEST="$RUN_ROOT/artifacts/fine/sset_15_f20695_f20945_512x288_manifest.json"
      MAX_NEW_TOKENS=4096
      QWEN_MAX_MODEL_LEN=16384
      ;;
    *)
      echo "Unknown benchmark stage: $STAGE" >&2
      return 2
      ;;
  esac
  KEY="$BACKEND-$STAGE"
  RESULT="$RUN_ROOT/results/$KEY.json"
  LOG="$RUN_ROOT/logs/$KEY.log"
  STATUS_PATH="$RUN_ROOT/status/$KEY.status"
}

common_apptainer_arguments() {
  APPTAINER_ARGUMENTS=(
    exec --nv --no-mount home,cwd
    --bind "$RUN_ROOT:$RUN_ROOT"
    --bind "$REMOTE_ROOT/cache:$REMOTE_ROOT/cache"
    --bind "$REMOTE_ROOT/envs:$REMOTE_ROOT/envs"
    --bind "$RUN_ROOT/tmp:$RUN_ROOT/tmp"
    --env PYTHONNOUSERSITE=1
    --env PYTHONDONTWRITEBYTECODE=1
    --env PYTHONPATH="$RUN_ROOT/repo/src"
    --env HF_HOME="$REMOTE_ROOT/cache/huggingface"
    --env TORCH_HOME="$REMOTE_ROOT/cache/torch"
    --env TRITON_CACHE_DIR="$REMOTE_ROOT/cache/triton"
    --env XDG_CACHE_HOME="$REMOTE_ROOT/cache/xdg"
    --env TMPDIR="$RUN_ROOT/tmp"
  )
}

intern_command() {
  APPTAINER_ARGUMENTS+=(
    --env LD_LIBRARY_PATH="$INTERN_FFMPEG_ENVIRONMENT/lib"
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  )
  COMMAND=(
    apptainer "${APPTAINER_ARGUMENTS[@]}"
    "$INTERN_IMAGE" "$INTERN_ENVIRONMENT/bin/python" -u
    -m annotator.vlm_scene_benchmark.run_cli
    --backend internvideo3 --manifest "$MANIFEST"
    --run-id "internvideo3-sset15-$STAGE-$RUN_ID_SUFFIX" --max-new-tokens "$MAX_NEW_TOKENS" --out "$RESULT"
  )
}

qwen_command() {
  APPTAINER_ARGUMENTS+=(
    --env FLASHINFER_WORKSPACE_BASE="$REMOTE_ROOT/cache/flashinfer"
    --env VLLM_CACHE_ROOT="$REMOTE_ROOT/cache/vllm"
    --env VLLM_CONFIG_ROOT="$REMOTE_ROOT/cache/vllm-config"
    --env VLLM_NO_USAGE_STATS=1
  )
  COMMAND=(
    apptainer "${APPTAINER_ARGUMENTS[@]}"
    "$QWEN_IMAGE" "$QWEN_ENVIRONMENT/bin/python" -u
    -m annotator.vlm_scene_benchmark.run_cli
    --backend qwen3-vl --manifest "$MANIFEST"
    --run-id "qwen3-vl-sset15-$STAGE-$RUN_ID_SUFFIX"
    --max-new-tokens "$MAX_NEW_TOKENS" --max-model-len "$QWEN_MAX_MODEL_LEN" --out "$RESULT"
  )
}

refuse_partial_benchmark() {
  local partial
  shopt -s nullglob
  for partial in "$RESULT" "$RUN_ROOT/results/$KEY".attempt-*.txt; do
    [[ -e "$partial" ]] || continue
    echo "Refusing to overwrite retained benchmark evidence: $partial" >&2
    return 1
  done
  return 0
}

run_benchmark() {
  benchmark_paths || return
  RUN_ID_SUFFIX=${RUN_ROOT##*/}
  if [[ -f "$STATUS_PATH" ]]; then
    return "$(<"$STATUS_PATH")"
  fi
  (
    set -euo pipefail
    printf '\n[%s] %s start\n' "$(date --iso-8601=seconds)" "$KEY"
    refuse_partial_benchmark
    verify_snapshot
    source "$RUN_ROOT/control/runtime.env"
    printf '%s  %s\n%s  %s\n' \
      "$INTERN_IMAGE_SHA256" "$INTERN_IMAGE" \
      "$QWEN_IMAGE_SHA256" "$QWEN_IMAGE" | sha256sum --check --quiet -
    assert_gpu_idle
    common_apptainer_arguments
    case "$BACKEND" in
      internvideo3) intern_command ;;
      qwen3-vl) qwen_command ;;
      *) echo "Unknown backend: $BACKEND" >&2; exit 2 ;;
    esac
    "${COMMAND[@]}"
  ) >>"$LOG" 2>&1
  local status=$?
  if ((status == 0 || status == 4)); then
    atomic_status "$STATUS_PATH" "$status" || return 71
  fi
  return "$status"
}

assert_gpu_idle() {
  local active
  active=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
  if [[ -n "$active" ]]; then
    if [[ ${ALLOW_SHARED_GPU:-0} == 1 && $BACKEND == internvideo3 ]]; then
      echo "Allowing InternVideo3 shared-GPU run; active process: $active" >&2
      return 0
    fi
    echo "Refusing to start $KEY while another GPU process is active: $active" >&2
    return 75
  fi
}

run_gate() {
  benchmark_paths || return
  source "$RUN_ROOT/control/runtime.env" || return
  apptainer exec \
    --no-mount home,cwd \
    --bind "$RUN_ROOT:$RUN_ROOT" \
    --bind "$REMOTE_ROOT/envs:$REMOTE_ROOT/envs" \
    --env PYTHONNOUSERSITE=1 \
    --env PYTHONPATH="$RUN_ROOT/repo/src" \
    "$INTERN_IMAGE" \
    "$INTERN_ENVIRONMENT/bin/python" \
    -m annotator.vlm_scene_benchmark.gate_cli \
    "$BACKEND=$RESULT"
}

main() {
  mkdir -p "$RUN_ROOT/results" "$RUN_ROOT/logs" "$RUN_ROOT/status" "$RUN_ROOT/tmp"
  case "$TASK" in
    setup) run_setup ;;
    benchmark) run_benchmark ;;
    gate) run_gate ;;
    *) echo "Unknown remote task: $TASK" >&2; return 2 ;;
  esac
}

main
