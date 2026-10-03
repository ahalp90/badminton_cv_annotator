#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_VIDEOS=86
readonly PYTHON=/home/ahalperi/.venvs/court_det/bin/python
readonly ORIGINAL_ROOT=/scratch/ahalperi/court_det_fix/release_courts_fast_robust
worker_exitfile=

usage() {
    printf 'usage: %s RUN_ROOT run|status|run-one [VIDEO_ID]\n' "${0##*/}" >&2
    exit 2
}

[[ $# -ge 2 ]] || usage
run_root=$(cd -- "$1" 2>/dev/null && pwd -P) || {
    printf 'run root does not exist: %s\n' "$1" >&2
    exit 2
}
action=$2
script_path=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")

die() {
    printf '%s\n' "$*" >&2
    exit 2
}

load_cohort_ids() {
    [[ -f "$run_root/cohort.json.gz" ]] || die "missing cohort: $run_root/cohort.json.gz"
    "$PYTHON" - "$run_root/cohort.json.gz" <<'PY'
import gzip
import json
import re
import sys

with gzip.open(sys.argv[1], "rt", encoding="utf-8") as handle:
    rows = json.load(handle)
if not isinstance(rows, list):
    raise SystemExit("cohort must be a JSON array")
video_ids = [row["id"] for row in rows]
if len(video_ids) != 86 or len(set(video_ids)) != 86:
    raise SystemExit(f"expected 86 unique videos, got {len(video_ids)}")
if any(not isinstance(video_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", video_id) is None
       for video_id in video_ids):
    raise SystemExit("cohort contains an invalid video id")
print("\n".join(video_ids))
PY
}

load_video_ids() {
    local ids_text
    ids_text=$(load_cohort_ids) || return
    mapfile -t video_ids <<< "$ids_text"
}

output_path() { printf '%s/videos/%s.json.gz' "$run_root" "$1"; }
log_path() { printf '%s/logs/%s.log' "$run_root" "$1"; }
pid_path() { printf '%s/pids/%s.pid' "$run_root" "$1"; }
exit_path() { printf '%s/exitcodes/%s.exit' "$run_root" "$1"; }

record_unexpected_worker_exit() {
    local status=$?
    if [[ -n $worker_exitfile && ! -f $worker_exitfile ]]; then
        printf '%s\n' "$status" > "$worker_exitfile"
    fi
}

is_complete() {
    local video_id=$1 result_code
    local result_file exit_file
    result_file=$(output_path "$video_id")
    exit_file=$(exit_path "$video_id")
    [[ -s "$result_file" && -f "$exit_file" ]] || return 1
    result_code=$(<"$exit_file")
    [[ $result_code == 0 ]]
}

pid_is_live_with_args() {
    local pid=$1 expected_arg expected_count argument_index
    shift
    local -a process_args=()
    [[ $pid =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    mapfile -d '' -t process_args < "/proc/$pid/cmdline" || return 1
    expected_count=$#
    argument_index=$((${#process_args[@]} - expected_count))
    ((argument_index >= 0)) || return 1
    for expected_arg in "$@"; do
        [[ ${process_args[$argument_index]} == "$expected_arg" ]] || return 1
        ((argument_index += 1))
    done
}

run_one() {
    [[ $# == 1 ]] || usage
    local video_id=$1 result_file partial_file logfile pidfile input_file slot replay_status
    [[ $video_id =~ ^[A-Za-z0-9_-]+$ ]] || die "invalid video id: $video_id"
    result_file=$(output_path "$video_id")
    partial_file=$result_file.partial
    logfile=$(log_path "$video_id")
    pidfile=$(pid_path "$video_id")
    worker_exitfile=$(exit_path "$video_id")
    input_file=$ORIGINAL_ROOT/videos/$video_id.json.gz
    slot=${COURT_SLOT:-0}
    if [[ ! $slot =~ ^[0-9]+$ ]] || ((slot > 20)); then
        die "invalid CPU slot: $slot"
    fi

    mkdir -p "$run_root/videos" "$run_root/logs" "$run_root/pids" "$run_root/exitcodes"
    if is_complete "$video_id"; then
        printf '[%s] already complete; skipping\n' "$(date --iso-8601=seconds)" >> "$logfile"
        return 0
    fi

    printf '%s\n' "$$" > "$pidfile"
    rm -f -- "$result_file" "$partial_file" "$worker_exitfile"
    trap record_unexpected_worker_exit EXIT

    if [[ ! -f "$input_file" ]]; then
        printf '[%s] missing input: %s\n' "$(date --iso-8601=seconds)" "$input_file" >> "$logfile"
        printf '2\n' > "$worker_exitfile"
        exit 255
    fi
    [[ -f "$run_root/scripts/replay_court_sharing.py" ]] || {
        printf 'missing replay utility: %s/scripts/replay_court_sharing.py\n' "$run_root" | tee -a "$logfile" >&2
        printf '2\n' > "$worker_exitfile"
        exit 255
    }
    [[ -d "$run_root/checkout/src" ]] || {
        printf 'missing frozen checkout: %s/checkout\n' "$run_root" | tee -a "$logfile" >&2
        printf '2\n' > "$worker_exitfile"
        exit 255
    }

    cd -- "$run_root/checkout"
    export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export LD_LIBRARY_PATH="/home/ahalperi/.venvs/court_det/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/ahalperi/.venvs/court_det/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH=.:src
    {
        printf '[%s] start video=%s cpu=%s\n' "$(date --iso-8601=seconds)" "$video_id" "$slot"
        printf 'input=%s\n' "$input_file"
        printf 'output=%s\n' "$result_file"
    } >> "$logfile"

    set +e
    taskset -c "$slot" "$PYTHON" -u "$run_root/scripts/replay_court_sharing.py" \
        --input "$input_file" --config "$ORIGINAL_ROOT/run_config.json.gz" --output "$result_file" \
        >> "$logfile" 2>&1
    replay_status=$?
    set -e

    if ((replay_status == 0)) && [[ -s "$result_file" ]]; then
        printf '0\n' > "$worker_exitfile"
        printf '[%s] complete\n' "$(date --iso-8601=seconds)" >> "$logfile"
        return 0
    fi
    if ((replay_status == 0)); then
        replay_status=1
        printf '[%s] replay exited successfully without writing its output\n' \
            "$(date --iso-8601=seconds)" >> "$logfile"
    else
        printf '[%s] replay failed with exit %s\n' "$(date --iso-8601=seconds)" "$replay_status" >> "$logfile"
    fi
    rm -f -- "$result_file" "$partial_file"
    printf '%s\n' "$replay_status" > "$worker_exitfile"
    exit 255
}

show_status() {
    load_video_ids
    local done_count=0 failure_count=0 pending_count=0 id result_code pid logfile progress active_text
    local -a active_ids=() failed_ids=() stale_ids=()
    for id in "${video_ids[@]}"; do
        if is_complete "$id"; then
            ((done_count += 1))
        elif [[ -f "$(exit_path "$id")" ]]; then
            result_code=$(<"$(exit_path "$id")")
            ((failure_count += 1))
            failed_ids+=("$id:$result_code")
        else
            pid=
            if [[ -f "$(pid_path "$id")" ]]; then
                pid=$(<"$(pid_path "$id")")
            fi
            if [[ -n $pid ]] && pid_is_live_with_args "$pid" "$script_path" "$run_root" run-one "$id"; then
                active_ids+=("$id")
            else
                ((pending_count += 1))
                [[ -n $pid ]] && stale_ids+=("$id")
            fi
        fi
    done

    printf 'Videos: done %d/%d; failures %d; pending %d\n' \
        "$done_count" "$EXPECTED_VIDEOS" "$failure_count" "$pending_count"
    active_text=${active_ids[*]}
    [[ -n $active_text ]] || active_text='(none)'
    printf 'Active IDs: %s\n' "$active_text"
    for id in "${active_ids[@]}"; do
        logfile=$(log_path "$id")
        progress=
        if [[ -f "$logfile" ]]; then
            progress=$(awk 'tolower($0) ~ /(prepared|pool|rebuilt court scene|sharing courts)/ { line = $0 } END { print line }' "$logfile")
        fi
        if [[ -n $progress ]]; then
            printf '  %s: %s\n' "$id" "${progress:0:240}"
        else
            printf '  %s: waiting for prepared/pooling log output\n' "$id"
        fi
    done
    [[ ${#failed_ids[@]} == 0 ]] || printf 'Failed IDs: %s\n' "${failed_ids[*]}"
    [[ ${#stale_ids[@]} == 0 ]] || printf 'Stale PID files ignored: %s\n' "${stale_ids[*]}"

    if [[ -f "$run_root/run.exit" ]]; then
        printf 'Parent runner: exited %s\n' "$(<"$run_root/run.exit")"
    elif [[ -f "$run_root/run.pid" ]] && pid_is_live_with_args "$(<"$run_root/run.pid")" \
        "$script_path" "$run_root" run; then
        printf 'Parent runner: running (PID %s)\n' "$(<"$run_root/run.pid")"
    elif [[ -f "$run_root/run.pid" ]]; then
        printf 'Parent runner: not running (stale PID file ignored)\n'
    else
        printf 'Parent runner: not started\n'
    fi
    if [[ -f "$run_root/run.started" ]]; then
        local started now
        started=$(<"$run_root/run.started")
        now=$(date +%s)
        if [[ $started =~ ^[0-9]+$ ]] && ((now >= started)); then
            printf 'Elapsed: %d min\n' "$(((now - started) / 60))"
        fi
    fi
}

run_all() {
    [[ -e "$run_root/checkout/src" ]] || die "missing frozen checkout: $run_root/checkout"
    [[ -f "$run_root/scripts/replay_court_sharing.py" ]] || die "missing replay utility under $run_root/scripts"
    [[ -f "$ORIGINAL_ROOT/run_config.json.gz" && -d "$ORIGINAL_ROOT/videos" ]] || \
        die "missing original inputs under $ORIGINAL_ROOT"
    load_video_ids
    mkdir -p "$run_root/videos" "$run_root/logs" "$run_root/pids" "$run_root/exitcodes"
    if [[ -f "$run_root/run.pid" ]] && pid_is_live_with_args "$(<"$run_root/run.pid")" \
        "$script_path" "$run_root" run; then
        die "runner is already active (PID $(<"$run_root/run.pid"))"
    fi

    local ids_file id
    ids_file=$(mktemp "$run_root/.cohort.XXXXXX")
    printf '%s\n' "${video_ids[@]}" > "$ids_file"
    rm -f -- "$run_root/run.exit"
    printf '%s\n' "$$" > "$run_root/run.pid"
    date +%s > "$run_root/run.started"
    trap 'printf "%s\\n" "$?" > "$run_root/run.exit"; rm -f -- "$ids_file"' EXIT

    set +e
    xargs -r -P 21 --process-slot-var=COURT_SLOT -I{} \
        bash "$script_path" "$run_root" run-one '{}' < "$ids_file"
    local queue_status=$?
    set -e
    exit "$queue_status"
}

case "$action" in
    run)
        [[ $# == 2 ]] || usage
        run_all
        ;;
    status)
        [[ $# == 2 ]] || usage
        show_status
        ;;
    run-one)
        [[ $# == 3 ]] || usage
        run_one "$3"
        ;;
    *) usage ;;
esac
