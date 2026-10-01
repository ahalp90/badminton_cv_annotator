#!/bin/bash
# Run batch_feedback.py inside the InternVideo3 container.
#
# usage: ./run_batch.sh CLIPS_DIR OUT_JSONL [batch_feedback.py options]
#   e.g. ./run_batch.sh /scratch/$USER/clips /scratch/$USER/runs/version_a/predictions.jsonl --limit 3
#
# CLIPS_DIR is the copied data/feedback_eval/clips/ directory (the .mp4 files and clips.csv).
# The model cache goes to $HF_CACHE (default: hf-cache/ next to this script). The model is
# about 16 GB and the home quota is 40 GB, so point HF_CACHE at /scratch if home is tight.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CLIPS_DIR=$(realpath "${1:?usage: run_batch.sh CLIPS_DIR OUT_JSONL [options]}")
OUT=$(realpath -m "${2:?usage: run_batch.sh CLIPS_DIR OUT_JSONL [options]}")
shift 2
HF_CACHE=$(realpath -m "${HF_CACHE:-$HERE/hf-cache}")
mkdir -p "$(dirname "$OUT")" "$HF_CACHE"

apptainer exec --cleanenv --nv \
  --env HF_HOME=/hf-cache,PYTHONNOUSERSITE=1 \
  --bind "$HERE:/workspace" \
  --bind "$CLIPS_DIR:/clips:ro" \
  --bind "$(dirname "$OUT"):/out" \
  --bind "$HF_CACHE:/hf-cache" \
  "$HERE/container.sif" \
  python /workspace/batch_feedback.py \
    --clips /clips/clips.csv --out "/out/$(basename "$OUT")" "$@"
