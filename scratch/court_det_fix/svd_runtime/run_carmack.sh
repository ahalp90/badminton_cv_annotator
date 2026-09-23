#!/usr/bin/env bash
set -euo pipefail

checkout=/scratch/ahalperi/court_det_fix/svd_runtime_checkout_20260923
run_root=/scratch/ahalperi/court_det_fix/svd_runtime_20260923
python=/home/ahalperi/.venvs/venv-rtmlib/bin/python

cd "$checkout"
mkdir -p "$run_root"
git rev-parse HEAD > "$run_root/revision.txt"
unset PYTHONOPTIMIZE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

(
    trap 'status=$?; printf "%s\n" "$status" > "$run_root/full.exit"' EXIT
    "$python" scratch/court_det_fix/svd_runtime/run_benchmark.py \
        --root "$checkout" \
        --output "$run_root/full" \
        --workers 6 \
        --repeats 1
) > "$run_root/full.log" 2>&1
