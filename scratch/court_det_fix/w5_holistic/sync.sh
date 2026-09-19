#!/usr/bin/env bash
# Local helper for the remote W5 run. Private paths come from paths.local.sh.
# Usage: sync.sh push | launch <label> <script> ... | status | tail <label> [lines] | pull | sh '<command>'
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$here/paths.local.sh"
run="$(cat "$here/run_name.txt")"
remote_dir="$REMOTE_ROOT/w5_holistic"

case "$1" in
  push)
    "$HPCRSYNC" -ai --delete \
      --exclude runs/ --exclude __pycache__/ --exclude .ruff_cache/ --exclude .pytest_cache/ \
      --exclude .pyrefly_cache/ --exclude paths.local.sh "$here/" "$REMOTE_HOST:$remote_dir/"
    "$HPCRSYNC" -ai --delete "$LOCAL_EXPERIMENTS/" "$REMOTE_HOST:$REMOTE_ROOT/experiments/annotator/independent_court/"
    "$HPCSSH" "$REMOTE_HOST" "seed_dir=$REMOTE_ROOT/next_steps_20260916/webui_seed/source; if [ -L \"\$seed_dir\" ]; then mv \"\$seed_dir\" \"\$seed_dir.automatic_axes_link\"; fi; mkdir -p \"\$seed_dir\""
    "$HPCRSYNC" -ai --delete "$LOCAL_SEED/" "$REMOTE_HOST:$REMOTE_ROOT/next_steps_20260916/webui_seed/source/"
    for frozen_input in \
      "$LOCAL_G0|$REMOTE_ROOT/automatic_axes_20260914/all_camera|automatic_axes_20260914/all_camera" \
      "$LOCAL_BASELINE|$REMOTE_ROOT/frozen_views/baseline_generation|frozen_views/baseline_generation"; do
      IFS='|' read -r local_path remote_path label <<< "$frozen_input"
      if "$HPCSSH" "$REMOTE_HOST" "[ -d \"$remote_path\" ]"; then
        changes=$("$HPCRSYNC" -aic --delete --omit-dir-times --dry-run --itemize-changes \
          "$local_path/" "$REMOTE_HOST:$remote_path/" | awk '$1 != ".f" && $1 != ".d" {print}')
        if [ -n "$changes" ]; then
          printf 'Frozen input differs: %s\n%s\n' "$label" "$changes" >&2
          exit 1
        fi
        printf 'Frozen input matches: %s\n' "$label"
      else
        printf 'Frozen input was absent; copying: %s\n' "$label"
        "$HPCRSYNC" -ai "$local_path/" "$REMOTE_HOST:$remote_path/"
      fi
    done
    ;;
  launch)
    label=$2
    script=$3
    shift 3
    "$HPCSSH" "$REMOTE_HOST" "cd $remote_dir/.. || exit 1; { nohup setsid nice -n 10 bash w5_holistic/run_remote.sh $run $label $script $* > /dev/null 2>&1 < /dev/null & }; echo launched $label"
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
