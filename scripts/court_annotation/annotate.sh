#!/usr/bin/env bash
# Runner for the court-corner annotation tooling in this directory.
#
# First use creates a lightweight GUI venv (numpy, pandas, opencv-python; no
# torch) under ~/.cache/bst-annotate-venv, outside the repo, so annotation
# works on any laptop without the full project environment.
#
# Usage:
#   ./annotate.sh <video.mp4> [tool args]   annotate; h in the window shows keys
#   ./annotate.sh check [corners.csv]       extrapolation error report
#   ./annotate.sh list  [corners.csv]       every annotated frame in the CSV
#   ./annotate.sh key   [out.png]           render the landmark naming diagram
#
# The corners CSV defaults to hand_corners.csv in the CURRENT directory (its
# landmarks sidecar lands beside it); pass --out-csv to the annotate form to
# override. opencv is pinned to the repo's version: its Qt5 GUI behaves on
# Wayland desktops where opencv 5's Qt6 can black-screen.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

VENV="${XDG_CACHE_HOME:-$HOME/.cache}/bst-annotate-venv"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
    echo "First run: creating $VENV with numpy pandas opencv-python==4.13.0.92..."
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.12 "$VENV"
        uv pip install --python "$PY" numpy pandas "opencv-python==4.13.0.92"
    else
        python3 -m venv "$VENV"
        "$PY" -m pip install --quiet --upgrade pip
        "$PY" -m pip install --quiet numpy pandas "opencv-python==4.13.0.92"
    fi
fi
if [ "${XDG_SESSION_TYPE:-}" = "wayland" ] && [ -z "${QT_QPA_PLATFORM:-}" ]; then
    export QT_QPA_PLATFORM=xcb
fi

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
