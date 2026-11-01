#!/usr/bin/env bash
# Build the existing image adapter against MonoTrack's released detector stages.
set -euo pipefail
output=${1:?output binary path}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
source_dir="$repo_root/experiments/annotator/monotrack_benchmark/monotrack/badminton-court-detection"
flags=(-std=c++11 -O3 -DNDEBUG -I"$source_dir"
  -include fstream
  -include opencv2/imgproc/types_c.h
  -include opencv2/videoio/legacy/constants_c.h
  -include opencv2/highgui/highgui_c.h)
read -r -a opencv_flags <<< "$(pkg-config --cflags --libs opencv4)"
sources=()
for source_file in "$source_dir"/*.cpp; do
  if [[ $(basename "$source_file") != main.cpp ]]; then sources+=("$source_file"); fi
done
mkdir -p "$(dirname -- "$output")"
g++ "${flags[@]}" "$script_dir/monotrack_image.cpp" "${sources[@]}" \
  "${opencv_flags[@]}" -o "$output"
