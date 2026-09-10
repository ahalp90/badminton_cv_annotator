#!/bin/bash
CURRENT_DIR="$(pwd)"
# VLM_ROOT=/scratch/$USER/mllm
VLM_ROOT=$CURRENT_DIR

# apptainer exec --cleanenv --nv \
#   --env HF_HOME=/vlm/hf-cache \
#   --bind "$VLM_ROOT:/vlm" \
#   --bind "$PWD:/workspace" \
#   "$VLM_ROOT/images/internvideo3-20260806.sif" \
  # python /workspace/run_internvideo3.py
apptainer exec --cleanenv --nv \
  --env HF_HOME=$CURRENT_DIR/hf-cache,PYTHONNOUSERSITE=1 \
  --bind "$VLM_ROOT:/vlm" \
  --bind "$PWD:/workspace" \
  "$VLM_ROOT/container.sif" \
  python /workspace/run_internvideo3.py

