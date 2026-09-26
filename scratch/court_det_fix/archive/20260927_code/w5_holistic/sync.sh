#!/usr/bin/env bash
# Local helper for the remote W5 run. Private paths come from paths.local.sh.
# Usage: sync.sh push | launch <label> <script> ... | status | tail <label> [lines] | pull | sh '<command>'
set -euo pipefail
if (( $# == 0 )); then
  printf 'usage: %s push|launch|status|tail|pull|sh ...\n' "$0" >&2
  exit 2
fi
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$here/paths.local.sh"
run="$(cat "$here/run_name.txt")"
remote_dir="$REMOTE_ROOT/w5_holistic"

case "$1" in
  push)
    printf '%s\n' 'sync.sh push is disabled: commit and push tracked code, then fetch it with Git on the compute host.' >&2
    exit 2
    ;;
  launch)
    if (( $# < 3 )); then
      printf 'usage: %s launch LABEL SCRIPT [ARGS...]\n' "$0" >&2
      exit 2
    fi
    label=$2
    script=$3
    shift 3
    remote_args=()
    for argument in "$run" "$label" "$script" "$@"; do
      printf -v quoted '%q' "$argument"
      remote_args+=("$quoted")
    done
    remote_python="${REMOTE_PYTHON:-\$HOME/.venvs/venv-pipeline/bin/python}"
    home_prefix="\$HOME/"
    if [[ "$remote_python" == "$home_prefix"* ]]; then
      remote_python_command="\"\$HOME/${remote_python#\$HOME/}\""
    else
      printf -v remote_python_command '%q' "$remote_python"
    fi
    remote_command="cd $(printf '%q' "$remote_dir/..") && REMOTE_PYTHON=$remote_python_command nohup setsid nice -n 10 bash w5_holistic/run_remote.sh ${remote_args[*]} > /dev/null 2>&1 < /dev/null & echo launched $(printf '%q' "$label")"
    "$HPCSSH" "$REMOTE_HOST" "$remote_command"
    ;;
  status)
    "$HPCSSH" "$REMOTE_HOST" "cd $remote_dir/runs/$run/receipts 2>/dev/null || exit 0; for p in *.pid; do [ -f \"\$p\" ] || continue; pid=\$(cat \"\$p\"); if kill -0 \"\$pid\" 2>/dev/null; then echo \"\$p \$pid alive\"; else echo \"\$p \$pid dead\"; fi; done; for r in *_exit_code.txt; do [ -f \"\$r\" ] && echo \"\$r=\$(cat \"\$r\")\"; done; true"
    ;;
  tail)
    "$HPCSSH" "$REMOTE_HOST" "tail -n ${3:-20} $remote_dir/runs/$run/logs/$2.log"
    ;;
  pull)
    mkdir -p "$here/runs/$run"
    "$HPCRSYNC" -a --exclude cache/ --exclude arrays/ --exclude case_records/ "$REMOTE_HOST:$remote_dir/runs/$run/" "$here/runs/$run/"
    ;;
  sh)
    "$HPCSSH" "$REMOTE_HOST" "cd $REMOTE_ROOT && $2"
    ;;
  *)
    echo "unknown command: $1" >&2
    exit 2
    ;;
esac
