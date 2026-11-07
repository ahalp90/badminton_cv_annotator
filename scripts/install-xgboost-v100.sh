#!/usr/bin/env bash
set -euo pipefail

cuda_root=${1:?Usage: install-xgboost-v100.sh CUDA_TOOLKIT_ROOT [PYTHON]}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${2:-"$repo_root/.venv/bin/python"}
xgboost_requirement=$("$python_bin" - "$repo_root/pyproject.toml" <<'PY'
import sys
import tomllib

with open(sys.argv[1], 'rb') as handle:
    dependencies = tomllib.load(handle)['project']['dependencies']
print(next(requirement for requirement in dependencies if requirement.startswith('xgboost==')))
PY
)

# CUDA 12.9's default XGBoost build omits the V100 (compute capability 7.0).
uv pip install --python "$python_bin" --no-deps \
    --reinstall-package xgboost --no-binary xgboost \
    --config-settings cmake.define.USE_CUDA=ON \
    --config-settings cmake.define.USE_NCCL=OFF \
    --config-settings cmake.define.CMAKE_CUDA_ARCHITECTURES=70 \
    --config-settings "cmake.define.CMAKE_CUDA_COMPILER=$cuda_root/bin/nvcc" \
    "$xgboost_requirement"
