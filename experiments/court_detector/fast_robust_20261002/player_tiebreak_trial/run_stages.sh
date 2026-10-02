#!/usr/bin/env bash
# Run each host's four videos in stage A, then stage B; retain each stage's status.
set -euo pipefail
run_root=${1:?usage: run_stages.sh RUN_ROOT run|status}
record_exit() {
  printf '%s\n' "$?" > "$run_root/chain.exit"
}
case "${2:-}" in
  run)
    trap record_exit EXIT
    for stage in stage_a stage_b; do
      printf '%s\n' "$stage" > "$run_root/current_stage"
      "$run_root/$stage/launch_trial.sh" run-wait
      printf '0\n' > "$run_root/$stage/stage.exit"
    done
    printf 'complete\n' > "$run_root/current_stage"
    ;;
  status)
    if [[ -f "$run_root/current_stage" ]]; then
      printf 'Stage: %s\n' "$(cat "$run_root/current_stage")"
    fi
    if [[ -f "$run_root/chain.exit" ]]; then
      printf 'Chain exit: %s\n' "$(cat "$run_root/chain.exit")"
    fi
    for stage in stage_a stage_b; do
      printf '\n%s\n' "$stage"
      "$run_root/$stage/launch_trial.sh" status
    done
    ;;
  *) echo 'usage: run_stages.sh RUN_ROOT run|status' >&2; exit 2 ;;
esac
