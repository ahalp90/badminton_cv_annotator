#!/usr/bin/env bash
# Local helper for the remote run. Host, user and absolute paths come from the gitignored
# paths.local.sh, so this file carries none of them.
# Usage:
#   sync.sh push                         copy the experiment code to the remote root
#   sync.sh launch <label> <script> ...  start a detached stage via run_remote.sh
#   sync.sh status                       PIDs, liveness and exit receipts for the run
#   sync.sh tail <label> [lines]         tail a stage log
#   sync.sh pull                         copy the run directory back
#   sync.sh sh '<command>'               run a shell command in the remote experiment directory
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$here/paths.local.sh"
run="$(cat "$here/run_name.txt")"
remote_dir="$REMOTE_ROOT/line_identity"
case "$1" in
  push)
    "$HPCRSYNC" -a --delete --exclude runs/ --exclude prior_checks/ --exclude __pycache__/ --exclude .ruff_cache/ --exclude .pytest_cache/ \
      --exclude .pyrefly_cache/ --exclude paths.local.sh "$here/" "$REMOTE_HOST:$remote_dir/" ;;
  launch)
    label=$2; script=$3; shift 3
    # The braces keep '&' on the nohup command alone; 'cd && nohup ... &' would background the
    # whole list in a subshell that still holds the SSH channel until the job ends.
    "$HPCSSH" "$REMOTE_HOST" "cd $remote_dir || exit 1; { nohup setsid nice -n 10 bash run_remote.sh $run $label $script $* > /dev/null 2>&1 < /dev/null & }; echo launched $label" ;;
  status)
    "$HPCSSH" "$REMOTE_HOST" "cd $remote_dir/runs/$run/receipts 2>/dev/null || exit 0; for p in *.pid; do [ -f \"\$p\" ] || continue; pid=\$(cat \"\$p\"); if kill -0 \"\$pid\" 2>/dev/null; then echo \"\$p \$pid alive\"; else echo \"\$p \$pid dead\"; fi; done; for r in *_exit_code.txt; do [ -f \"\$r\" ] && echo \"\$r=\$(cat \"\$r\")\"; done; true" ;;
  tail)
    "$HPCSSH" "$REMOTE_HOST" "tail -n ${3:-20} $remote_dir/runs/$run/logs/$2.log" ;;
  pull)
    mkdir -p "$here/runs/$run"
    "$HPCRSYNC" -a --exclude cache/ "$REMOTE_HOST:$remote_dir/runs/$run/" "$here/runs/$run/" ;;
  sh)
    "$HPCSSH" "$REMOTE_HOST" "cd $remote_dir && $2" ;;
  *)
    echo "unknown command: $1" >&2; exit 2 ;;
esac
