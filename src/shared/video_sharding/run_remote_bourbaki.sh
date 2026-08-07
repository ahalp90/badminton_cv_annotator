#!/bin/bash
# Bourbaki experiment ladder for the video-sharding PoC. Host-specific paths.
#
# Run inside tmux from the synced PoC tree root:
#   tmux new-session -d -s shard_poc 'bash src/shared/video_sharding/run_remote_bourbaki.sh'
#
# Each step logs to $OUT/<step>.log and writes its exit code to $OUT/<step>.exit;
# "ALL_STEPS_LAUNCHED" in $OUT/driver.done marks the ladder finishing.
set -u

VENVPY=~/.venvs/venv-rtmlib/bin/python
SP=$($VENVPY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
# Required for the CUDA provider: no system cuDNN on the node (see
# docs/architecture_notes/rtmlib_migration/extraction_saturation_runbook.md).
export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:/usr/local/cuda-13.3/lib64"
export PYTHONPATH=src:src/bst_x

V21=/scratch/comp320a/ahalperi/s31_fps_eval/sset_21_gloiZ_gTJaE.mp4
OUT=/scratch/comp320a/ahalperi/rtmlib_sharding_poc_out
mkdir -p "$OUT"

step() {
  local name=$1; shift
  echo "=== $name: $(date -Is)"
  "$@" > "$OUT/$name.log" 2>&1
  echo $? > "$OUT/$name.exit"
}

MOD=shared.video_sharding

# A. Frame-range identity on the full 1080p match (sequential MD5 ledger, then
#    seek-mode checks at default awkward boundaries plus extra unaligned cuts).
step identity_baseline_21 $VENVPY -m $MOD.gate_decode_identity baseline "$V21" "$OUT/ledger_21.txt"
step identity_check_21_seek $VENVPY -m $MOD.gate_decode_identity check "$V21" "$OUT/ledger_21.txt" --mode seek
step identity_check_21_more $VENVPY -m $MOD.gate_decode_identity check "$V21" "$OUT/ledger_21.txt" \
  --mode seek --ranges 29:83,999:1031,49999:50021,74123:74191,100000:100349

# D3. Real RTMLib CPU parity, bounded, threads pinned for determinism.
step cpu_selfvar env OMP_NUM_THREADS=2 $VENVPY -m $MOD.gate_parity \
  --video "$V21" --workdir "$OUT/cpu_selfvar" --extractor cpu --limit-frames 600 --self-variance
step cpu_parity env OMP_NUM_THREADS=2 $VENVPY -m $MOD.gate_parity \
  --video "$V21" --workdir "$OUT/cpu_parity" --extractor cpu --limit-frames 600 --n-shards 4

# D4. CUDA: self-variance control first, then sequential-vs-sharded.
step cuda_selfvar $VENVPY -m $MOD.gate_parity \
  --video "$V21" --workdir "$OUT/cuda_selfvar" --extractor cuda --limit-frames 3000 --self-variance
step cuda_parity $VENVPY -m $MOD.gate_parity \
  --video "$V21" --workdir "$OUT/cuda_parity" --extractor cuda --limit-frames 3000 --n-shards 4

# E. Worker scaling probe on a bounded span.
step scaling env OMP_NUM_THREADS=2 $VENVPY -m $MOD.bench_worker_scaling \
  --video "$V21" --workdir "$OUT/scaling" --extractor cuda --limit-frames 12000 --worker-counts 1,2,4,8

echo "ALL_STEPS_LAUNCHED $(date -Is)" > "$OUT/driver.done"
