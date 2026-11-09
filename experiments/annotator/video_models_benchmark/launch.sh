#!/usr/bin/env bash
# Run inside nohup or a terminal; record termination independently of Python.
set -euo pipefail
benchmark_python=$1
benchmark_cache=$2
benchmark_output=$3
benchmark_scheme=$4
benchmark_warmup=$5
shift 5
mkdir -p "$benchmark_output"
exec 9> "$benchmark_output/run.lock"
flock -n 9 || { printf "A launcher already owns this run directory\n" >&2; exit 1; }
if [[ -e "$benchmark_output/exit-code.txt" ]]; then
  mv "$benchmark_output/exit-code.txt" "$benchmark_output/previous-exit-code.txt"
fi
if [[ -e "$benchmark_output/status.json.gz" ]]; then
  mv "$benchmark_output/status.json.gz" "$benchmark_output/previous-status.json.gz"
fi
printf '%s\n' "$$" > "$benchmark_output/launcher.pid"
trap 'code=$?; printf "%s\n" "$code" > "$benchmark_output/exit-code.txt"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
export PYTHONPATH="src:.${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
"$benchmark_python" -u -m experiments.annotator.video_models_benchmark.train train \
  --cache "$benchmark_cache" --output "$benchmark_output" \
  --scheme "$benchmark_scheme" --warmup-steps "$benchmark_warmup" "$@" \
  >> "$benchmark_output/train.log" 2>&1 &
benchmark_child=$!
trap 'kill -TERM "$benchmark_child" 2>/dev/null || true; exit 143' TERM
trap 'kill -TERM "$benchmark_child" 2>/dev/null || true; exit 130' INT
trap 'kill -TERM "$benchmark_child" 2>/dev/null || true; exit 129' HUP
wait "$benchmark_child"
