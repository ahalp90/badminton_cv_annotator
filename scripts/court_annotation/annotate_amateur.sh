#!/usr/bin/env bash
# Open the amateur collection in one workspace. Pass a filename to open that video first.
set -euo pipefail
shopt -s nullglob
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
video_dir="$repo_dir/local_scratch/amateur_vids"
mkdir -p "$video_dir/court_annotations"
videos=()
for video in "$video_dir/"*.mp4 "$video_dir/"*.mkv; do
    case "${video##*/}" in
        # Excluded unless named explicitly.
        nTL8uUlb4fo.*|i1KGN5cDoVA.*) [[ "${video##*/}" == "${1:-}" ]] || continue ;;
    esac
    videos+=("$video")
done
args=(
    --video "${videos[@]}"
    --out-csv "$video_dir/court_annotations/corners.csv"
    --candidate-dir "$repo_dir/local_scratch/amateur_camera_scan"
)
if (( $# )); then
    args+=(--start-video "$1")
fi
if [[ "${INCLUDE_UNCERTAIN:-0}" == 1 ]]; then
    args+=(--include-uncertain)
fi
exec "$HOME/.venvs/badminton-cicd/bin/python" "$repo_dir/scripts/court_annotation/annotate_court_corners_offframe.py" "${args[@]}"
