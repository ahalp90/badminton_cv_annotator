#!/usr/bin/env bash
set -euo pipefail

readonly REMOTE_ROOT=${1:?usage: setup_remote.sh REMOTE_ROOT RUN_ROOT}
readonly RUN_ROOT=${2:?usage: setup_remote.sh REMOTE_ROOT RUN_ROOT}
readonly SCRIPT_ROOT="$RUN_ROOT/repo/scripts/vlm_scene_benchmark"
readonly INTERN_IMAGE="$REMOTE_ROOT/images/internvideo3-pytorch-2.8.0-cu129.sif"
readonly QWEN_IMAGE="$REMOTE_ROOT/images/qwen3-vl-vllm-v0.11.0.sif"
readonly INTERN_IMAGE_SHA256=fd1c42ea24386dde021f12c0fe9458f0d4f5f43ea97af2ad19c2b3ea9925c76a
readonly QWEN_IMAGE_SHA256=1cf06bf5a8a7bd5a2b2c469f0e72ac150f0781c126b593c7fcd9d7df4eb34d37
readonly INTERN_IMAGE_URI=docker://pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
readonly QWEN_IMAGE_URI=docker://vllm/vllm-openai:v0.11.0

pull_image() {
  local destination=$1
  local source_uri=$2
  local expected_sha256=$3
  local partial="${destination%.sif}.partial.sif"

  if [[ ! -f "$destination" ]]; then
    rm -f -- "$partial"
    echo "Pulling $source_uri"
    apptainer pull "$partial" "$source_uri"
    printf '%s  %s\n' "$expected_sha256" "$partial" | sha256sum --check -
    mv -- "$partial" "$destination"
  fi
  printf '%s  %s\n' "$expected_sha256" "$destination" | sha256sum --check -
}

environment_path() {
  local name=$1
  local image_sha256=$2
  local requirements=$3
  local requirements_sha256
  requirements_sha256=$(sha256sum "$requirements" | awk '{print $1}')
  printf '%s/envs/%s-%s-%s\n' \
    "$REMOTE_ROOT" "$name" "${image_sha256:0:12}" "${requirements_sha256:0:12}"
}

environment_ready() {
  local environment=$1
  local image_sha256=$2
  local requirements=$3
  local sentinel="$environment/.issue38-ready"
  local requirements_sha256
  requirements_sha256=$(sha256sum "$requirements" | awk '{print $1}')
  [[ -f "$sentinel" ]] || return 1
  [[ $(<"$sentinel") == "$image_sha256 $requirements_sha256" ]]
}

mark_environment_ready() {
  local environment=$1
  local image_sha256=$2
  local requirements=$3
  local sentinel="$environment/.issue38-ready"
  local requirements_sha256
  requirements_sha256=$(sha256sum "$requirements" | awk '{print $1}')
  printf '%s %s\n' "$image_sha256" "$requirements_sha256" >"$sentinel.tmp"
  mv -- "$sentinel.tmp" "$sentinel"
}

create_intern_environment() {
  local environment=$1
  local requirements=$2
  if environment_ready "$environment" "$INTERN_IMAGE_SHA256" "$requirements"; then
    echo "Reusing $environment"
    return
  fi
  if [[ ! -x "$environment/bin/python" ]]; then
    apptainer exec \
      --no-mount home,cwd \
      --bind "$RUN_ROOT:$RUN_ROOT" \
      --bind "$REMOTE_ROOT/envs:$REMOTE_ROOT/envs" \
      --bind "$REMOTE_ROOT/cache:$REMOTE_ROOT/cache" \
      --env PYTHONNOUSERSITE=1 \
      --env PIP_CACHE_DIR="$REMOTE_ROOT/cache/pip" \
      "$INTERN_IMAGE" \
      python -m venv --system-site-packages "$environment"
  fi
  apptainer exec \
    --no-mount home,cwd \
    --bind "$RUN_ROOT:$RUN_ROOT" \
    --bind "$REMOTE_ROOT/envs:$REMOTE_ROOT/envs" \
    --bind "$REMOTE_ROOT/cache:$REMOTE_ROOT/cache" \
    --env PYTHONNOUSERSITE=1 \
    --env PIP_CACHE_DIR="$REMOTE_ROOT/cache/pip" \
    "$INTERN_IMAGE" \
    "$environment/bin/python" -m pip install -r "$requirements"
  mark_environment_ready "$environment" "$INTERN_IMAGE_SHA256" "$requirements"
}

create_intern_ffmpeg_environment() {
  local environment=$1
  local requirements=$2
  local intern_environment=$3
  if environment_ready "$environment" "$INTERN_IMAGE_SHA256" "$requirements"; then
    echo "Reusing $environment"
    return
  fi
  mkdir -p "$REMOTE_ROOT/cache/conda-home/.cache/conda/notices" "$REMOTE_ROOT/cache/conda-pkgs"
  apptainer exec --no-mount home,cwd --bind "$RUN_ROOT:$RUN_ROOT" --bind "$REMOTE_ROOT:$REMOTE_ROOT" --env CONDA_PKGS_DIRS="$REMOTE_ROOT/cache/conda-pkgs" --env XDG_CACHE_HOME="$REMOTE_ROOT/cache/conda-home/.cache" "$INTERN_IMAGE" /opt/conda/bin/conda create -y -p "$environment" --channel conda-forge --override-channels --file "$requirements"
  apptainer exec --no-mount home,cwd --bind "$REMOTE_ROOT:$REMOTE_ROOT" --env LD_LIBRARY_PATH="$environment/lib" "$INTERN_IMAGE" "$intern_environment/bin/python" -c "import torchcodec"
  apptainer exec --no-mount home,cwd --bind "$REMOTE_ROOT:$REMOTE_ROOT" "$INTERN_IMAGE" /opt/conda/bin/conda list -p "$environment" --explicit >"$RUN_ROOT/control/internvideo3-ffmpeg-explicit.txt.tmp"
  mv "$RUN_ROOT/control/internvideo3-ffmpeg-explicit.txt.tmp" "$RUN_ROOT/control/internvideo3-ffmpeg-explicit.txt"
  mark_environment_ready "$environment" "$INTERN_IMAGE_SHA256" "$requirements"
}

create_qwen_environment() {
  local environment=$1
  local requirements=$2
  if environment_ready "$environment" "$QWEN_IMAGE_SHA256" "$requirements"; then
    echo "Reusing $environment"
    return
  fi
  if [[ ! -x "$environment/bin/python" ]]; then
    apptainer exec \
      --no-mount home,cwd \
      --bind "$RUN_ROOT:$RUN_ROOT" \
      --bind "$REMOTE_ROOT/envs:$REMOTE_ROOT/envs" \
      --bind "$REMOTE_ROOT/cache:$REMOTE_ROOT/cache" \
      --env PYTHONNOUSERSITE=1 \
      --env PIP_CACHE_DIR="$REMOTE_ROOT/cache/pip" \
      "$QWEN_IMAGE" \
      /usr/bin/python3 -m venv --system-site-packages "$environment"
  fi
  apptainer exec \
    --no-mount home,cwd \
    --bind "$RUN_ROOT:$RUN_ROOT" \
    --bind "$REMOTE_ROOT/envs:$REMOTE_ROOT/envs" \
    --bind "$REMOTE_ROOT/cache:$REMOTE_ROOT/cache" \
    --env PYTHONNOUSERSITE=1 \
    --env PIP_CACHE_DIR="$REMOTE_ROOT/cache/pip" \
    "$QWEN_IMAGE" \
    "$environment/bin/python" -m pip install -r "$requirements"
  mark_environment_ready "$environment" "$QWEN_IMAGE_SHA256" "$requirements"
}

write_runtime_environment() {
  local intern_environment=$1
  local intern_ffmpeg_environment=$2
  local qwen_environment=$3
  local output="$RUN_ROOT/control/runtime.env"
  {
    printf 'INTERN_IMAGE=%q\n' "$INTERN_IMAGE"
    printf 'INTERN_IMAGE_SHA256=%q\n' "$INTERN_IMAGE_SHA256"
    printf 'INTERN_ENVIRONMENT=%q\n' "$intern_environment"
    printf 'INTERN_FFMPEG_ENVIRONMENT=%q\n' "$intern_ffmpeg_environment"
    printf 'QWEN_IMAGE=%q\n' "$QWEN_IMAGE"
    printf 'QWEN_IMAGE_SHA256=%q\n' "$QWEN_IMAGE_SHA256"
    printf 'QWEN_ENVIRONMENT=%q\n' "$qwen_environment"
  } >"$output.tmp"
  mv -- "$output.tmp" "$output"
}

main() {
  local intern_requirements="$SCRIPT_ROOT/requirements-internvideo3.txt"
  local intern_ffmpeg_requirements="$SCRIPT_ROOT/requirements-internvideo3-conda.txt"
  local qwen_requirements="$SCRIPT_ROOT/requirements-qwen3-vl.txt"
  local intern_environment
  local intern_ffmpeg_environment
  local qwen_environment

  mkdir -p "$REMOTE_ROOT"
  exec 9>"$REMOTE_ROOT/.setup.lock"
  echo "Waiting for the shared setup lock: $REMOTE_ROOT/.setup.lock"
  flock 9

  mkdir -p \
    "$REMOTE_ROOT/images" "$REMOTE_ROOT/envs" \
    "$REMOTE_ROOT/cache/apptainer" "$REMOTE_ROOT/cache/huggingface" \
    "$REMOTE_ROOT/cache/pip" "$REMOTE_ROOT/cache/torch" \
    "$REMOTE_ROOT/cache/triton" "$REMOTE_ROOT/cache/flashinfer" \
    "$REMOTE_ROOT/cache/vllm" "$REMOTE_ROOT/cache/vllm-config" \
    "$REMOTE_ROOT/cache/xdg" "$RUN_ROOT/tmp" "$RUN_ROOT/control"
  export APPTAINER_CACHEDIR="$REMOTE_ROOT/cache/apptainer"
  export TMPDIR="$RUN_ROOT/tmp"

  pull_image "$INTERN_IMAGE" "$INTERN_IMAGE_URI" "$INTERN_IMAGE_SHA256"
  pull_image "$QWEN_IMAGE" "$QWEN_IMAGE_URI" "$QWEN_IMAGE_SHA256"
  intern_environment=$(environment_path internvideo3 "$INTERN_IMAGE_SHA256" "$intern_requirements")
  intern_ffmpeg_environment=$(environment_path internvideo3-ffmpeg "$INTERN_IMAGE_SHA256" "$intern_ffmpeg_requirements")
  qwen_environment=$(environment_path qwen3-vl-v0.11.0 "$QWEN_IMAGE_SHA256" "$qwen_requirements")
  create_intern_environment "$intern_environment" "$intern_requirements"
  create_intern_ffmpeg_environment "$intern_ffmpeg_environment" "$intern_ffmpeg_requirements" "$intern_environment"
  create_qwen_environment "$qwen_environment" "$qwen_requirements"
  write_runtime_environment "$intern_environment" "$intern_ffmpeg_environment" "$qwen_environment"
}

main "$@"
