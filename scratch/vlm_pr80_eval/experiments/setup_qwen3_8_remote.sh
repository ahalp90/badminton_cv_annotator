#!/usr/bin/env bash
set -euo pipefail

plan=false
if [[ ${1:-} == "--plan" ]]; then
  plan=true
  shift
fi
RUNTIME_ROOT=$(realpath -m -- "${1:?usage: setup_qwen3_8_remote.sh [--plan] RUNTIME_ROOT}")
SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly RUNTIME_ROOT SCRIPT_ROOT
readonly REQUIREMENTS="$SCRIPT_ROOT/requirements-qwen3-8.txt"
readonly IMAGE="$RUNTIME_ROOT/images/qwen3.8-vllm-v0.17.0.sif"
readonly IMAGE_URI=docker://vllm/vllm-openai@sha256:14ea8b431aaaf75eb873c46c8ebfbad2b4b0790d30c66126d789d8cb9bd0aab9
readonly IMAGE_SHA256=72afb9dc802cd6a5d2808eeba68dac660d9faf75a4ebbea49b4c9f05029e9f1f
readonly MODEL_ID=Qwen/Qwen3.8-27B-FP8
readonly MODEL_REVISION=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a

requirements_sha256=$(sha256sum "$REQUIREMENTS" | awk '{print $1}')
readonly requirements_sha256
readonly ENVIRONMENT="$RUNTIME_ROOT/envs/qwen3.8-v0.17.0-${IMAGE_SHA256:0:12}-${requirements_sha256:0:12}"
readonly SENTINEL="$ENVIRONMENT/.qwen3.8-ready"

print_plan() {
  cat <<EOF
Qwen3.8 trial runtime plan
runtime root: $RUNTIME_ROOT
container source: $IMAGE_URI
converted image: $IMAGE
converted image sha256: $IMAGE_SHA256
Python environment: $ENVIRONMENT
requirements sha256: $requirements_sha256
model: $MODEL_ID@$MODEL_REVISION
EOF
}

pull_image() {
  local partial="${IMAGE%.sif}.partial.sif"

  if [[ ! -f $IMAGE ]]; then
    rm -f -- "$partial"
    apptainer pull "$partial" "$IMAGE_URI"
    printf '%s  %s\n' "$IMAGE_SHA256" "$partial" | sha256sum --check -
    mv -- "$partial" "$IMAGE"
  fi
  printf '%s  %s\n' "$IMAGE_SHA256" "$IMAGE" | sha256sum --check -
}

environment_ready() {
  [[ -x "$ENVIRONMENT/bin/python" ]] || return 1
  [[ -f $SENTINEL ]] || return 1
  [[ $(<"$SENTINEL") == "$IMAGE_SHA256 $requirements_sha256" ]]
}

create_environment() {
  if environment_ready; then
    echo "Reusing $ENVIRONMENT"
    return
  fi
  if [[ ! -x "$ENVIRONMENT/bin/python" ]]; then
    apptainer exec \
      --no-mount home,cwd \
      --bind "$RUNTIME_ROOT:$RUNTIME_ROOT" \
      --env PYTHONNOUSERSITE=1 \
      --env PIP_CACHE_DIR="$RUNTIME_ROOT/cache/pip" \
      "$IMAGE" \
      /usr/bin/python3 -m venv --system-site-packages "$ENVIRONMENT"
  fi
  apptainer exec \
    --no-mount home,cwd \
    --bind "$RUNTIME_ROOT:$RUNTIME_ROOT" \
    --bind "$SCRIPT_ROOT:$SCRIPT_ROOT:ro" \
    --env PYTHONNOUSERSITE=1 \
    --env PIP_CACHE_DIR="$RUNTIME_ROOT/cache/pip" \
    "$IMAGE" \
    "$ENVIRONMENT/bin/python" -m pip install -r "$REQUIREMENTS"
  apptainer exec \
    --no-mount home,cwd \
    --bind "$RUNTIME_ROOT:$RUNTIME_ROOT" \
    --env PYTHONNOUSERSITE=1 \
    "$IMAGE" \
    "$ENVIRONMENT/bin/python" -c \
    'import importlib.metadata as m; assert m.version("vllm") == "0.17.0"; import av, cv2, qwen_vl_utils'
  printf '%s %s\n' "$IMAGE_SHA256" "$requirements_sha256" >"$SENTINEL.tmp"
  mv -- "$SENTINEL.tmp" "$SENTINEL"
}

write_runtime_environment() {
  local output="$RUNTIME_ROOT/control/qwen3.8-runtime.env"

  {
    printf 'export VLM_QWEN38_IMAGE=%q\n' "$IMAGE"
    printf 'export VLM_QWEN38_IMAGE_SHA256=%q\n' "$IMAGE_SHA256"
    printf 'export VLM_QWEN38_IMAGE_URI=%q\n' "$IMAGE_URI"
    printf 'export VLM_QWEN38_PYTHON=%q\n' "$ENVIRONMENT/bin/python"
    printf 'export VLM_QWEN38_REQUIREMENTS_SHA256=%q\n' "$requirements_sha256"
    printf 'export VLM_QWEN38_MODEL_ID=%q\n' "$MODEL_ID"
    printf 'export VLM_QWEN38_MODEL_REVISION=%q\n' "$MODEL_REVISION"
  } >"$output.tmp"
  mv -- "$output.tmp" "$output"
}

main() {
  if [[ $plan == true ]]; then
    print_plan
    return
  fi
  mkdir -p \
    "$RUNTIME_ROOT/images" \
    "$RUNTIME_ROOT/envs" \
    "$RUNTIME_ROOT/cache/apptainer" \
    "$RUNTIME_ROOT/cache/huggingface" \
    "$RUNTIME_ROOT/cache/pip" \
    "$RUNTIME_ROOT/control" \
    "$RUNTIME_ROOT/tmp"
  exec 9>"$RUNTIME_ROOT/.qwen3.8-setup.lock"
  flock 9
  export APPTAINER_CACHEDIR="$RUNTIME_ROOT/cache/apptainer"
  export TMPDIR="$RUNTIME_ROOT/tmp"

  pull_image
  create_environment
  write_runtime_environment
  echo "Runtime environment: $RUNTIME_ROOT/control/qwen3.8-runtime.env"
}

main "$@"
