#!/usr/bin/env bash
# The paired score-first court trial on one host: one detached lane per CPU set, three workers each.
#   launch_trial.sh check    read-only preflight: inputs, checkout, models, disk
#   launch_trial.sh ready    refuse to start over live lanes or a changed commit
#   launch_trial.sh run      start or resume the lanes
#   launch_trial.sh status   progress from the run root's own files
# The run root is this script's directory. It holds cohort.json.gz, run_courts.py and
# trial_run_video.py. Relaunching resumes: complete videos are skipped, failed ones rerun.
# Stop a lane with `kill -TERM -- -PGID`, using the pgid from `ps -o pgid= -p PID`.
set -euo pipefail
root=${COURT_ROOT:-$(cd "$(dirname "$0")" && pwd)}
# Carmack defaults; the COURT_* variables override them on another host.
checkout=${COURT_CHECKOUT:-$root/checkout}
venv=${COURT_VENV:-$HOME/.venvs/court_det}
deeplsd=${COURT_DEEPLSD:-/scratch/ahalperi/court_det_fix/independent_detector/DeepLSD}
# One lane per CPU set. Eight lanes on one host: COURT_CPU_SETS="0-2 3-5 6-8 9-11 12-14 15-17 18-20 21-23".
read -r -a cpu_sets <<< "${COURT_CPU_SETS:-0-2 3-5 6-8 9-11}"
runner=$root/run_courts.py
export NUMBA_NUM_THREADS=1
export LD_LIBRARY_PATH="$venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
common=(--root "$root" --checkout "$checkout" --python "$venv/bin/python"
        --deeplsd "$deeplsd" --lanes "${#cpu_sets[@]}" --workers 3)

case "${1:-}" in
  check)
    exec "$venv/bin/python" "$runner" check "${common[@]}"
    ;;
  ready)
    exec "$venv/bin/python" "$runner" ready "${common[@]}"
    ;;
  run | run-wait)
    mkdir -p "$root/lanes"
    # Serialise launcher invocations; individual lanes also hold their own locks.
    exec 9>"$root/lanes/parallel-launch.lock"
    flock -n 9 || { echo "another launcher is running" >&2; exit 1; }
    "$venv/bin/python" "$runner" ready "${common[@]}"
    lane_pids=()
    for lane in "${!cpu_sets[@]}"; do
      cpus=${cpu_sets[$lane]}
      if [[ "$1" == run ]]; then
        setsid nohup taskset -c "$cpus" "$venv/bin/python" -u "$runner" run "${common[@]}" \
          --lane "$lane" >> "$root/lanes/lane-$lane.log" 2>&1 < /dev/null 9>&- &
      else
        taskset -c "$cpus" "$venv/bin/python" -u "$runner" run "${common[@]}" \
          --lane "$lane" >> "$root/lanes/lane-$lane.log" 2>&1 < /dev/null 9>&- &
      fi
      lane_pids+=("$!")
      echo "lane $lane of ${#cpu_sets[@]} started: pid $!, CPUs $cpus, workers 3"
    done
    if [[ "$1" == run-wait ]]; then
      failed=0
      for pid in "${lane_pids[@]}"; do
        wait "$pid" || failed=1
      done
      exit "$failed"
    fi
    ;;
  status)
    exec "$venv/bin/python" "$runner" status --root "$root"
    ;;
  *)
    echo "usage: $0 check | ready | run | run-wait | status" >&2
    exit 2
    ;;
esac
