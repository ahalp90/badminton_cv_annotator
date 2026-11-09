#!/usr/bin/env bash
# Native court annotation and related inspection tools.
# Usage: ./annotate.sh <video> [options] | check [csv] | list [csv] | key [out.png]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${COURT_ANNOTATOR_PYTHON:-$HOME/.venvs/badminton-cicd/bin/python}"

case "${1:-}" in
    check)
        exec "$PY" "$HERE/check_extrapolation.py" --corners-csv "${2:-hand_corners.csv}"
        ;;
    key)
        exec "$PY" "$HERE/make_landmark_key.py" --out "${2:-landmark_key.png}"
        ;;
    list)
        CSV="${2:-hand_corners.csv}"
        [ -f "$CSV" ] || { echo "no annotations at $CSV"; exit 1; }
        exec "$PY" - "$CSV" <<'PYEOF'
import csv, sys
from collections import defaultdict
from pathlib import Path

frames = defaultdict(list)
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        frames[(Path(row["video"]).name, int(row["frame"]))].append(row)
for (video, frame), group in sorted(frames.items()):
    group.sort(key=lambda row: int(row["corner_idx"]))
    marks = "  ".join(
        f"{row['corner_label']}:{'extrapolated' if row['source'] != 'click' else 'clicked'}" for row in group
    )
    note = "" if len(group) == 4 else f"  MALFORMED ({len(group)} rows)"
    print(f"{video}  frame {frame}   {marks}{note}")
print(f"{len(frames)} annotated frame(s) in {sys.argv[1]}")
PYEOF
        ;;
    "")
        echo "usage: ./annotate.sh <video> [tool args] | check [csv] | list [csv] | key [out.png]"
        exit 1
        ;;
    *)
        VIDEO="$1"
        shift
        # Defaults first, so any --out-csv/--orientation the caller passes wins.
        exec "$PY" "$HERE/annotate_court_corners_offframe.py" \
            --video "$VIDEO" --out-csv hand_corners.csv --orientation landscape "$@"
        ;;
esac
