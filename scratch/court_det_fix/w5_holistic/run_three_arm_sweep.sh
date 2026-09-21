#!/usr/bin/env bash
# Run the authorised W5 three-arm sweep in a fixed, fail-fast order.
set -euo pipefail

if (( $# != 1 )); then
  printf 'usage: %s RUN_PREFIX\n' "$0" >&2
  exit 2
fi

run_prefix=$1
if [[ ! "$run_prefix" =~ ^[[:alnum:]][[:alnum:]_-]{0,79}$ ]]; then
  printf 'RUN_PREFIX must start with a letter or number and contain only letters, numbers, underscores and hyphens\n' >&2
  exit 2
fi

script_dir="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
experiment_root="$(cd "$script_dir/.." && pwd)"
run_root="$experiment_root/w5_holistic/runs"
run_remote="$script_dir/run_remote.sh"

if ! repo_root=$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null); then
  printf 'W5 sweep requires a Git checkout\n' >&2
  exit 1
fi
if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=no)" ]]; then
  printf 'W5 sweep requires a clean tracked worktree\n' >&2
  exit 1
fi

cases=(
  gxBQ_window_00_frame_0
  gxBQ_window_00_frame_5
  am2_window_00_frame_150
  am2_window_01_frame_28019
  am3_window_00_frame_0
  shuttleset_03_scene_0017
  shuttleset_03_scene_0019
  shuttleset_03_scene_0016
  shuttleset_21_scene_0020
  gxBQ_window_00_frame_689
  gxBQ_window_01_frame_5111
  gxBQ_window_02_frame_5766
  gxBQ_window_03_frame_77876
  gxBQ_window_04_frame_86088
  yellow_short_frame_14
  letterboxed_short_frame_45
  centre_short_frame_36
  am1_window_00_frame_54
  am3_window_01_frame_10514
  am4_window_00_frame_0
  am4_window_01_frame_13782
  shuttleset_03_scene_0029
  shuttleset_03_scene_0034
  shuttleset_03_scene_0038
  shuttleset_21_scene_0000
  shuttleset_21_scene_0010
  shuttleset_21_scene_0039
)

arms=(
  "3 3"
  "4 3"
  "5 3"
)

for arm in "${arms[@]}"; do
  read -r lengthwise cross_court <<< "$arm"
  arm_id="${lengthwise}${cross_court}"
  run_name="${run_prefix}_${arm_id}"
  if [[ -e "$run_root/$run_name" ]]; then
    printf 'Refusing to reuse existing run directory for arm %s\n' "$arm" >&2
    exit 1
  fi
done
comparison_dir="$run_root/${run_prefix}_comparison"
if [[ -e "$comparison_dir" ]]; then
  printf 'Refusing to reuse existing comparison directory\n' >&2
  exit 1
fi

workers="${W5_WORKERS:-6}"
if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
  printf 'W5_WORKERS must be a positive integer\n' >&2
  exit 2
fi
export W5_WORKERS="$workers"

python="${REMOTE_PYTHON:-$HOME/.venvs/venv-pipeline/bin/python}"
home_prefix="\$HOME/"
if [[ "$python" == "$home_prefix"* ]]; then
  python="$HOME/${python#\$HOME/}"
fi

run_stage() {
  local run_name=$1
  local label=$2
  local script=$3
  shift 3
  if ! "$run_remote" "$run_name" "$label" "$script" "$@"; then
    printf 'W5 arm %s stage %s failed\n' "$run_name" "$label" >&2
    return 1
  fi
  local receipt="$run_root/$run_name/receipts/${label}_exit_code.txt"
  if [[ ! -f "$receipt" ]] || [[ "$(tr -d '[:space:]' < "$receipt")" != 0 ]]; then
    printf 'W5 arm %s stage %s did not leave a successful receipt\n' "$run_name" "$label" >&2
    return 1
  fi
}

check_pilot_packet() {
  local run_name=$1
  local run_dir="$run_root/$run_name"
  local record_dir="$run_dir/case_records"
  local manifest="$run_dir/manifest.json"
  [[ -f "$manifest" ]] || {
    printf 'W5 arm %s has no manifest\n' "$run_name" >&2
    return 1
  }
  [[ -d "$record_dir" ]] || {
    printf 'W5 arm %s has no case-record directory\n' "$run_name" >&2
    return 1
  }
  local record_count
  record_count=$(find "$record_dir" -maxdepth 1 -type f -name '*.json.gz' -printf '%f\n' | wc -l)
  if [[ "$record_count" -ne "${#cases[@]}" ]]; then
    printf 'W5 arm %s has %s case records; expected %s\n' "$run_name" "$record_count" "${#cases[@]}" >&2
    return 1
  fi
  local case_id
  for case_id in "${cases[@]}"; do
    [[ -f "$record_dir/$case_id.json.gz" ]] || {
      printf 'W5 arm %s is missing case record %s\n' "$run_name" "$case_id" >&2
      return 1
    }
  done
  "$python" -c 'import json, sys
manifest = json.load(open(sys.argv[1]))
expected = sys.argv[2:]
if manifest.get("requested_cases") != expected or manifest.get("stopped_views"):
    raise SystemExit(1)
' "$manifest" "${cases[@]}"
}

for arm in "${arms[@]}"; do
  read -r lengthwise cross_court <<< "$arm"
  arm_id="${lengthwise}${cross_court}"
  run_name="${run_prefix}_${arm_id}"
  printf 'Starting W5 arm (%s,%s)\n' "$lengthwise" "$cross_court"
  run_stage "$run_name" preflight run_w5 \
    --stage preflight \
    --cases "${cases[@]}" \
    --min-visible-lengthwise "$lengthwise" \
    --min-visible-cross-court "$cross_court"
  run_stage "$run_name" pilot run_w5 \
    --stage pilot \
    --workers "$workers" \
    --cases "${cases[@]}" \
    --min-visible-lengthwise "$lengthwise" \
    --min-visible-cross-court "$cross_court"
  check_pilot_packet "$run_name"
  run_stage "$run_name" gallery render_gallery --cases "${cases[@]}"
  [[ -f "$run_root/$run_name/gallery/index.md" ]] || {
    printf 'W5 arm %s has no gallery index\n' "$run_name" >&2
    exit 1
  }
  printf 'Completed W5 arm (%s,%s)\n' "$lengthwise" "$cross_court"
done

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$repo_root:$repo_root/src:$experiment_root/w5_holistic:$experiment_root/next_steps_20260916/webui_seed/source:$experiment_root/frozen_helpers_20260914/marking_diagnosis:$experiment_root/frozen_helpers_20260914/vp_pruning:$experiment_root/frozen_helpers_20260914/axis_matching:$experiment_root/frozen_helpers_20260914/legacy:$experiment_root/src:$experiment_root"
cd "$experiment_root"
"$python" "$script_dir/compare_directional_runs.py" \
  --run-33 "$run_root/${run_prefix}_33" \
  --run-43 "$run_root/${run_prefix}_43" \
  --run-53 "$run_root/${run_prefix}_53" \
  --output-dir "$comparison_dir"
printf 'W5 three-arm comparison written\n'
