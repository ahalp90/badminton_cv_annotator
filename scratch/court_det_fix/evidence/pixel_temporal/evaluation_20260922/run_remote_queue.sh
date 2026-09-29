#!/usr/bin/env bash
# Six independent target frames at a time; each child owns its score checkpoint.
set -u
if [ "$#" -lt 2 ]; then
    echo "usage: $0 <gx|am3> <target-case> [target-case ...]" >&2
    exit 2
fi
cohort=$1
shift
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/receipts"
workers=${TEMPORAL_WORKERS:-6}
batch=${TEMPORAL_BATCH:-$cohort}
printf '%s\n' "$@" | xargs -P "$workers" -n 1 bash "$here/run_remote_targets.sh" "$cohort"
status=$?
printf '%s\n' "$status" > "$here/receipts/$batch.queue.exit_code.txt"
exit "$status"
