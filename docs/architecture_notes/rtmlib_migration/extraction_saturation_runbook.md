# Extraction saturation runbook

How to run a full keypoint re-extract across the HPC nodes so each GPU is
actually saturated. The single-worker numbers below show why this matters: one
extraction process leaves the A100 at 37% and the L40 at 39%, because per-frame
time is dominated by Python, cv2 decode and preprocessing, not the model. The
lever is several worker processes per GPU, each on a disjoint shard of clips.

All numbers measured 2026-07-06 on the 50-stem probe list (2498 frames,
~50 frames/clip) through `preparing_data.raw_extract` on CUDA, rtmlib
RTMDet-M@640 + RTMPose-L@256 (the shipped config), venv-rtmlib
(onnxruntime-gpu 1.27.0). The single-worker bourbaki figure (83.7 ms/frame)
reproduces the 81 ms/frame A100 benchmark in the README, so these numbers sit
on the same scale as the config benchmarks there.

## Node specs (verified on-node 2026-07-06)

| node | GPU | driver / CUDA | vCPUs | RAM | /scratch free |
|---|---|---|---|---|---|
| bourbaki | A100-PCIE-40GB | 610.43.02 / 13.3 | 16 (Xeon 6342 2.8GHz) | 251 Gi | 1.8 T |
| carmack | L40 45GB | 610.43.02 / 13.3 | 32 (Xeon 6448H) | 375 Gi | 1.8 T |
| engelbart | V100-PCIE-16GB | 580.173.02 / 13.0 | 16 (Xeon 6148 2.4GHz) | 125 Gi | 913 G |

The clips live at `/scratch/comp320a/ShuttleSet/clips` on all three nodes,
33,481 mp4s each (verified identical counts). `/home` is shared, `/scratch` is
host-local. All vCPUs are presented virtualised (each core as its own socket).

## Measured throughput

Worker ladder over the probe list, disjoint shards, `OMP_NUM_THREADS` set to
cores/workers for multi-worker runs (single worker runs with the environment's
natural threading):

| node | workers | wall | ms/frame | GPU util (median) |
|---|---|---|---|---|
| bourbaki | 1 | 209 s | 83.7 | 37% |
| bourbaki | 2 | 137 s | 54.8 | 62% |
| bourbaki | 4 | 107 s | 42.8 | 86% |
| bourbaki | 8 | 95 s | 38.0 | 97% |
| carmack | 1 | 147 s | 58.8 | 39% |
| carmack | 4 | 85 s | 34.0 | 80% |
| carmack | 8 | 68 s | 27.2 | 96% |

What binds: GPU memory never (each worker holds ~1 GiB, so even 8 workers use
under a quarter of either card), and CPU cores only indirectly (carmack still
had 38% idle CPU at 8 workers). The GPU pins first, at 8 workers on both
cards, which is where the recommendation comes from. Past that point more
workers just queue on the GPU. cv2 decode-thread thrash did not show up at
these worker counts; `cv2.setNumThreads` / `OPENCV_NUM_THREADS` stay untouched.

**Recommended counts: 8 workers on bourbaki (`OMP_NUM_THREADS=2`), 8 workers
on carmack (`OMP_NUM_THREADS=4`).** Engelbart sits out (next section).

## Engelbart is a dead end for extraction

onnxruntime-gpu 1.27 is a CUDA-13 build, and CUDA 13 dropped Volta. The V100
fails at session creation: `CUBLAS failure 8: the function requires an
architectural feature absent from the device`. No environment recipe fixes a
dropped architecture; the options would be a separate CUDA-12 onnxruntime venv
or the CPU provider.

CPU was measured for the record: 334.7 ms/frame single-worker, 12x slower than
carmack saturated. Sharding makes it worse, not better: two workers at
`OMP_NUM_THREADS=8` took 2.5x the single-worker wall (842.7 ms/frame
aggregate). The virtualised one-core-per-socket topology may be why
onnxruntime's threading degrades so badly here; not chased further, the node
is not worth the time for this workload.

## Full re-extract time estimates

33,481 clips at ~50 frames/clip is ~1.67M frames:

| plan | time |
|---|---|
| carmack alone, 8 workers | ~12.6 h |
| bourbaki alone, 8 workers | ~17.7 h |
| both, stems split 58:42 carmack:bourbaki | ~7.4 h |

## Command blocks

Both entrypoints resume per clip (a finished clip is skipped on rerun), abort
loudly if failures exceed 0.3 of the clips slated in that invocation, and
append failures to `failed_clips.log` under the save dir. With all shards
writing one shared save dir the log is shared too; the abort threshold stays
per-shard, so a mass failure on one shard reads as that shard's 0.3, not the
batch's.

Shards must be disjoint. CUDA extraction is treated as nondeterministic, so a
clip extracted twice would leave its provenance ambiguous. (On this exact
stack, onnxruntime-gpu 1.27 on the A100, the G7 gate measured zero
self-variance across repeat runs, 50 stems, 2026-07-06. The mmpose-era noise
floor did not reproduce. Disjoint shards stay mandatory anyway; zero measured
variance on one build and card is not a guarantee.)

Common setup on any node (the LD_LIBRARY_PATH recipe is required: no system
cuDNN exists on any node, and without it onnxruntime silently lacks the CUDA
provider):

```
VENVPY=~/.venvs/venv-rtmlib/bin/python
SP=$($VENVPY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:/usr/local/cuda-13.3/lib64"
cd ~/badminton_stroke_classification
export PYTHONPATH=src/bst_x:src/bst_x/stroke_classification
```

### raw_extract (takes a stems list natively)

Split the stems N ways and launch one tmux session per shard, exit codes
captured to files:

```
W=8; OMP=2                      # bourbaki; carmack: W=8; OMP=4
SHARD_DIR=/scratch/comp320a/ahalperi/extract_shards
mkdir -p "$SHARD_DIR"
split -d -n "l/$W" stems.txt "$SHARD_DIR/shard_"

for SH in "$SHARD_DIR"/shard_*; do
  tmux new-session -d -s "extract_$(basename "$SH")" \
    "OMP_NUM_THREADS=$OMP $VENVPY -m preparing_data.raw_extract \
       --clips-dir /scratch/comp320a/ShuttleSet/clips \
       --clip-stems-file $SH \
       --save-dir <SAVE_DIR> \
       --device cuda > $SH.log 2>&1; echo \$? > $SH.exit"
done
```

Watch with `nvidia-smi dmon` (expect ~97% sm once all workers are past model
load) and `tail -f` on a shard log. Done when every `shard_*.exit` exists and
reads 0; nonzero means read that shard's log, and the resume-skip makes a
relaunch of just that shard safe.

### prepare_train_on_shuttleset (consumes a clips folder)

The pose step has no stems-list argument (adding one is a code change, flagged
as an option, not assumed). Shard with per-worker symlink dirs into the shared
save root; the per-clip resume-skip makes overlap harmless but keep the shards
disjoint anyway per the provenance rule:

```
W=8; OMP=2
SHARD_DIR=/scratch/comp320a/ahalperi/prepare_shards
mkdir -p "$SHARD_DIR"
find /scratch/comp320a/ShuttleSet/clips -name '*.mp4' | sort > "$SHARD_DIR/all_clips.txt"
split -d -n "l/$W" "$SHARD_DIR/all_clips.txt" "$SHARD_DIR/list_"

for L in "$SHARD_DIR"/list_*; do
  D="$SHARD_DIR/clips_$(basename "$L")"
  mkdir -p "$D"
  xargs -a "$L" -I{} ln -s {} "$D/"
  tmux new-session -d -s "prepare_$(basename "$L")" \
    "OMP_NUM_THREADS=$OMP $VENVPY -m preparing_data.prepare_train_on_shuttleset \
       --skip-collate --clips-dir $D <other flags per the walkthrough> \
       > $L.log 2>&1; echo \$? > $L.exit"
done
```

Collation stays a single-process step afterwards; only the pose step shards.

Clean up shard lists, symlink dirs and per-shard logs once the extract is
verified (count the output npys against the stems list).
