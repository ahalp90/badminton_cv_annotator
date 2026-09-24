#!/usr/bin/env bash
set -euo pipefail

repo="$(git rev-parse --show-toplevel)"
cd "$repo"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
session="${COURT_OPT_SESSION:-scratch/court_det_fix/optimisation_${stamp}}"

if [[ -e "$session" ]]; then
  echo "Refusing to reuse existing session path: $session" >&2
  exit 2
fi

mkdir -p "$session"
{
  echo "# Resume"
  echo
  echo "Created: $stamp UTC"
  echo
  echo "## Repository state"
  echo
  echo '```text'
  echo "repo: $repo"
  echo "branch: $(git branch --show-current)"
  echo "head: $(git rev-parse HEAD)"
  git status --short
  echo '```'
  echo
  echo "No optimisation run has been launched."
} > "$session/worklog.md"

cat > "$session/evidence.md" <<'EOF'
# Evidence

Record source paths, profiler output, before/after record comparisons, and
contrary cases here. Existing frozen data only; no new annotation campaign.
EOF

cat > "$session/mechanisms.md" <<'EOF'
# Mechanisms

For every proposed patch, state exactly which operations are removed, moved, or
batched; the invariant that makes the change safe; and the observed counters.
EOF

cat > "$session/runs.md" <<'EOF'
# Runs

For every command, record revision, host, hardware, thread settings, cold/warm
status, input identity, output directory, exit code, stage wall/CPU time, and
peak RSS.
EOF

git rev-parse HEAD > "$session/revision.txt"
git status --short > "$session/status.txt"
python --version > "$session/python.txt" 2>&1
python -m pip freeze > "$session/pip-freeze.txt" 2>&1 || true
(lscpu || true) > "$session/lscpu.txt" 2>&1
(free -h || true) > "$session/memory.txt" 2>&1
(env | sort | grep -E '^(OPENBLAS|MKL|OMP|NUMEXPR|VECLIB|BLIS)_' || true) \
  > "$session/thread-env.txt"

critical=(
  "local_scratch/net_recovery/20260923/bounded/bounded_trial.json.gz"
  "local_scratch/net_recovery/20260923/selected_polarity/bounded_results.json.gz"
  "local_scratch/net_recovery/20260923/selected_polarity/bounded_requests.json.gz"
  "local_scratch/external_delegate/20260923-am1-recovery-trial/seeded_pool.json.gz"
  "local_scratch/external_delegate/20260923-net-bounded-audit/result.md"
)
{
  for path in "${critical[@]}"; do
    if [[ -f "$path" ]]; then
      bytes="$(wc -c < "$path")"
      sha="$(sha256sum "$path" | awk '{print $1}')"
      printf 'present\t%s\t%s\t%s\n' "$bytes" "$sha" "$path"
    else
      printf 'missing\t-\t-\t%s\n' "$path"
    fi
  done
} > "$session/local-input-inventory.tsv"

cat <<EOF
Created non-destructive optimisation session:
  $session

Next:
  1. Read the handover packet.
  2. Resolve active hot-path imports.
  3. Select one fast and one slow case from saved timings.
  4. Capture a one-worker baseline in fresh output directories.

No detector run was launched and no tracked/untracked data was removed.
EOF
