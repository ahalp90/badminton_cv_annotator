#!/usr/bin/env bash
# Run one batch of merged-seat codex jobs (luna, effort max, workspace-write sandbox).
# Usage: run_codex_batch.sh <module_stem>...   (intended batch size ~5, all concurrent)
set -u
cd "$(dirname "$0")/../.."   # repo root
JOBS=scratch/swarm_review/codex_jobs
PACKETS=scratch/swarm_review/packets

for stem in "$@"; do
  (
    codex exec -m gpt-5.6-luna -c 'model_reasoning_effort="max"' --sandbox workspace-write - \
      < "${PACKETS}/${stem}__brief.md" \
      > "${JOBS}/${stem}.log" 2>&1
    echo $? > "${JOBS}/${stem}.exit"
  ) &
done
wait

echo "--- batch summary ---"
for stem in "$@"; do
  verdict=no
  [ -s "${PACKETS}/${stem}__verdict.json" ] && verdict=yes
  echo "${stem}: exit=$(cat "${JOBS}/${stem}.exit" 2>/dev/null || echo '?') verdict_file=${verdict}"
done
