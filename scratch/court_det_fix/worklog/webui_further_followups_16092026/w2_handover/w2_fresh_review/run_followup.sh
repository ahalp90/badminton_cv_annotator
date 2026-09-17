#!/usr/bin/env bash
# Run inside the existing approved host/environment; this wrapper does not open SSH,
# install dependencies, reset a checkout, alter thread/scheduler policy, or submit jobs.
set -euo pipefail
PACK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
OUT="${OUT:-${TMPDIR:-/tmp}/w2_pair_atlas_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON="${PYTHON:-python}"
"$PYTHON" "$PACK/build_pair_atlas.py" --repo "$REPO" --output "$OUT"
printf '\nReturn this file: %s\n' "$OUT/return_pack.zip"
