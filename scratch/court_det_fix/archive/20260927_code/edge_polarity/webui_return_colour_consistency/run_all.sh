#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$ROOT/results/generated" "$ROOT/overlays/generated"

"$PYTHON_BIN" "$ROOT/scripts/net_boundary_bounded_audit.py" \
  "$ROOT/data/cases.json.gz" \
  --am1-image "$ROOT/data/images/am1_frame54_user_supplied.png" \
  --gx-image "$ROOT/data/images/gx_frame5_user_supplied.png" \
  --mesh-test \
  > "$ROOT/results/generated/net_boundary_bounded_audit_with_images.txt"

"$PYTHON_BIN" "$ROOT/scripts/boundary_paint_followup.py" \
  "$ROOT/data/cases.json.gz" \
  --am1-image "$ROOT/data/images/am1_frame54_user_supplied.png" \
  --gx-image "$ROOT/data/images/gx_frame5_user_supplied.png" \
  --output "$ROOT/results/generated/boundary_paint_followup_results.json" \
  --overlay-dir "$ROOT/overlays/generated" \
  > "$ROOT/results/generated/boundary_paint_followup_stdout.txt"

if cmp -s \
  "$ROOT/results/boundary_paint_followup_results.json" \
  "$ROOT/results/generated/boundary_paint_followup_results.json"; then
  echo "Boundary-paint JSON matches the canonical result byte-for-byte."
else
  echo "Boundary-paint JSON differs from the canonical result." >&2
  exit 1
fi

echo "Generated outputs are under results/generated and overlays/generated."
