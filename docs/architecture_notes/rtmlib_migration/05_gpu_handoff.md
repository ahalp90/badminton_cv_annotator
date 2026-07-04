# GPU handoff: Phase-A parity gates (Bourbaki)

> STATUS: complete. Phase-A GO at 0.15 (see `06_phase_a_decision.md`); retained as the run record.
> **Superseded (2026-07-04):** that run used rtmdet-nano@320; the detector is now
> RTMDet-M@640 at 0.3 and this loop re-runs with `_m`-named gate JSONs
> ([07_detector_restoration.md](07_detector_restoration.md)). Env recipe below stays valid.

> The CPU gates (G1-G6 + raw-schema) are green on the dev box. G7/G8/G9 run the
> CUDA extract-and-compare that can only run where Phase B will run: the CUDA
> execution provider is nondeterministic and CPU ≠ CUDA bit-for-bit, so these are
> never self-certified on the dev box. This is a **HALT-AND-HANDOFF loop**: you
> run a phase, send the output back, I check it against the thresholds (and the
> G7 noise floor) and give go / no-go for the next phase.

## The handoff loop

For each phase below: run the command, then send me both the terminal output
and the JSON file it wrote. I verify against the documented thresholds and the
measured noise floor, then confirm the next phase (or flag a regression). Nothing
here mutates tracked files: the gates only read the committed raw/clean and write
throwaway `*_parity.json` / `*_selfvariance.json` (gitignored).

**Order matters:** G7 first (it measures the CUDA noise floor that G8/G9's
thresholds must sit above), then G8 (parity), then G9 (decision). Do the fast
smoke50 pass first to validate the pipeline end-to-end before the ~200-clip
authoritative run.

---

## Phase G-0: one-time setup

- [x] **Check out the branch** on Bourbaki (after I hand you the commit script and
      you push, see the commit plan):
      ```
      git fetch origin && git checkout migrate-mmpose-to-rtmlib
      ```
- [x] **Create the extraction venv.** `onnxruntime-gpu==1.27.0` needs CUDA 13.x +
      cuDNN 9.x. Its `[cuda,cudnn]` extras pin the CUDA-13 toolkit under NVIDIA's
      *old* `*-cu13`-suffixed wheel names, which aren't published (plain PyPI has only
      `0.0.x` stubs; NVIDIA's index 404s), so the extras cannot resolve. Instead take
      the toolkit from the `cuda/13.3` module and pip-install just cuDNN. That
      pulls the *renamed* `nvidia-cublas` + `nvidia-cuda-nvrtc` (CUDA-13, unsuffixed) as
      deps. rtmlib depends on the CPU `onnxruntime`, so it lands first and is swapped
      for the GPU build (both together clash on the `onnxruntime` import):
      ```
      module load cuda/13.3                      # toolkit: cudart/nvrtc/cublas/cufft/curand
      python3.11 -m venv ~/venv-rtmlib-gpu
      # system python3.11 may not bootstrap pip; if ~/venv-rtmlib-gpu/bin/pip is absent:
      #   ~/venv-rtmlib-gpu/bin/python -m ensurepip --upgrade
      ~/venv-rtmlib-gpu/bin/pip install --upgrade pip
      ~/venv-rtmlib-gpu/bin/pip install rtmlib==0.0.15 numpy==2.4.6 \
          opencv-python==4.13.0.92 pandas==3.0.3 scipy==1.17.1 tqdm==4.68.3 \
          parse==1.22.1
      ~/venv-rtmlib-gpu/bin/pip uninstall -y onnxruntime
      ~/venv-rtmlib-gpu/bin/pip install onnxruntime-gpu==1.27.0
      ~/venv-rtmlib-gpu/bin/pip install "nvidia-cudnn-cu13~=9.0" \
          --extra-index-url https://pypi.nvidia.com
      # CPU torch: the deployed-parity gates (G5-G8) import sticky_anchor, which
      # lazily imports prepare_train_on_shuttleset (torch at module top):
      ~/venv-rtmlib-gpu/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
      ```
      (Bourbaki: A100 + driver CUDA UMD 13.3. pip `nvidia-cublas` can be a newer CUDA-13
      minor than the module's `cudart`; if a gate later dies at CUDA session creation,
      pin `nvidia-cublas~=13.3.0` to match the module; otherwise leave it.)
- [x] **Verify the CUDA provider is present**: run it where the GPU is visible (on
      Bourbaki the A100 shows in `nvidia-smi` on the host), with `cuda/13.3` loaded. If
      it lists only CPU, the gates silently run on CPU; stop and fix first:
      ```
      ~/venv-rtmlib-gpu/bin/python -c \
        "import onnxruntime as ort; ort.preload_dlls(); print(ort.get_available_providers())"
      # want CUDAExecutionProvider in the list (TensorrtExecutionProvider may appear too)
      ~/venv-rtmlib-gpu/bin/pip list | grep -iE 'onnxruntime|nvidia'
      ```
- [x] **Fill in the env file** for Bourbaki's data paths (copy the template at the
      bottom of this doc to `gpu_env.sh`, edit, then `source gpu_env.sh` in each
      shell). The gates default to the dev-pool paths; set them to Bourbaki's
      actual clip / raw / clean / provenance locations.
- [x] **Verify + warm the models** (SHA-asserts the two ONNX against the pinned
      constants and vendors them; also populates the cache the gates load from):
      ```
      source gpu_env.sh
      XDG_CACHE_HOME="$XDG_CACHE_HOME" PYTHONPATH=src/bst_x \
        ~/venv-rtmlib-gpu/bin/python \
        src/bst_x/validation_scripts/rtmlib_migration/download_and_verify_models.py
      ```
      (If Bourbaki compute nodes have no internet: copy the vendored `.onnx` from
      the dev-pool `RTMLIB_MODEL_VENDOR_DIR` into `$XDG_CACHE_HOME/rtmlib/hub/
      checkpoints/` first; this step then finds them cached, skips the download,
      and just SHA-verifies.)

**Send me:** the provider list + the model-verify output. I confirm `PASS: model
SHA verification` and that CUDA is present before you spend GPU time.

---

## Phase G-1: G7 CUDA self-variance floor (run FIRST)

Two CUDA runs of smoke50; measures the run-to-run noise floor (`eps_kp`, `eps_fail`)
that every G8/G9 threshold must sit above.

- [x] Run it:
      ```
      source gpu_env.sh
      PYTHONUNBUFFERED=1 PYTHONPATH=src/bst_x:src RTMLIB_GATE_DEVICE=cuda \
        RTMLIB_GATE_G7_JSON=g7_selfvariance.json \
        ~/venv-rtmlib-gpu/bin/python \
        src/bst_x/validation_scripts/rtmlib_migration/gate_cuda_selfvariance.py \
        2>&1 | tee g7.out
      ```

**Send me:** `g7.out` + `g7_selfvariance.json`. I check `eps_kp` median ≤ 3 px,
`eps_fail` ≤ 0.02, and `PASS`. These floors get carried into G9; if they're
implausibly large the extract isn't reproducible and we stop here.

---

## Phase G-2: G8 extraction parity, smoke50 (pipeline validation)

Runs the shipped adapter on smoke50 (CUDA) and compares each clip to the committed
mmpose baseline on both axes (keypoint value + deployed `sticky_anchor` output),
plus the directional failed-frame split.

- [x] Run it (smoke50 is the default stemfile):
      ```
      source gpu_env.sh
      PYTHONUNBUFFERED=1 PYTHONPATH=src/bst_x:src RTMLIB_GATE_DEVICE=cuda \
        RTMLIB_GATE_G8_JSON=g8_parity_smoke50.json \
        ~/venv-rtmlib-gpu/bin/python \
        src/bst_x/validation_scripts/rtmlib_migration/gate_gpu_parity.py \
        2>&1 | tee g8_smoke50.out
      ```
      Result (2026-07-03, then-shipped `DET_SCORE_THR=0.3`): 45/50 PASS. The 5 per-clip
      fails are all `jntMed > 0.03`, the report-only body7 pose drift, not a regression
      (see `06_phase_a_decision.md`). smoke50 showed no frame-loss bias (easy court); the
      G-4 authoritative run did, and the fix moved the shipped threshold to 0.15, so
      re-running this smoke50 gate now uses 0.15.

**Send me:** `g8_smoke50.out` + `g8_parity_smoke50.json`. I check `dF == 0` on
every clip, `fmatch`, the keypoint `kp_med`/`kp_p90` (read against the G7 floor),
the directional `rtLoss`/`mmLoss`, no dropped players, and `PASS`.

---

## Phase G-3: G9 Phase-A decision, smoke50 (provisional)

- [x] Run it against the smoke50 G8 JSON, carrying the G7 floors:
      ```
      source gpu_env.sh
      PYTHONPATH=src/bst_x:src RTMLIB_GATE_G7_JSON=g7_selfvariance.json \
        ~/venv-rtmlib-gpu/bin/python \
        src/bst_x/validation_scripts/rtmlib_migration/phase_a_decision.py \
        g8_parity_smoke50.json 2>&1 | tee g9_smoke50.out
      ```
      Result: every hard criterion passes EXCEPT `per-clip G8 verdicts all PASS`,
      which fails only because `g8_ok` re-imports G8's per-clip `jntMed` hard-gate,
      the metric G9's own policy (this file's docstring / `03_verification.md`)
      designates report-only. Read past that gate inconsistency, the smoke50 verdict
      is **GO (provisional)** (coverage 1/44). Rationale + the exception:
      `06_phase_a_decision.md`.

**Send me:** `g9_smoke50.out`. Expect **GO (provisional)**: smoke50 is one match
(video 11), so coverage is 1/44 and the gate correctly withholds an authoritative
GO. I confirm every hard criterion passed and the floors are valid. If this is a
NO-GO, we diagnose before the big run.

---

## Phase G-4: authoritative coverage (all video-ids)

Only after G-3 is a clean GO(provisional). Builds a stratified ~200-clip sample
(5 per video-id, ~12k frames, all 40 video-ids), re-runs G8 on it, then G9 for the
authoritative decision.

- [x] Build the stratified sample:
      ```
      source gpu_env.sh
      PYTHONPATH=src/bst_x:src ~/venv-rtmlib-gpu/bin/python \
        src/bst_x/validation_scripts/rtmlib_migration/make_phase_a_sample.py \
        --per-video 5 --out phase_a_stems.txt
      # prints the video-id coverage; expect ~200 stems across ~40 video-ids
      ```
- [x] Run G8 on the stratified sample (this is the long one, ~200 clips on GPU):
      ```
      PYTHONUNBUFFERED=1 PYTHONPATH=src/bst_x:src RTMLIB_GATE_DEVICE=cuda \
        RTMLIB_GATE_STEMFILE=phase_a_stems.txt \
        RTMLIB_GATE_G8_JSON=g8_parity_full.json \
        ~/venv-rtmlib-gpu/bin/python \
        src/bst_x/validation_scripts/rtmlib_migration/gate_gpu_parity.py \
        2>&1 | tee g8_full.out
      ```
- [x] Run G9 for the authoritative decision:
      ```
      PYTHONPATH=src/bst_x:src RTMLIB_GATE_G7_JSON=g7_selfvariance.json \
        ~/venv-rtmlib-gpu/bin/python \
        src/bst_x/validation_scripts/rtmlib_migration/phase_a_decision.py \
        g8_parity_full.json 2>&1 | tee g9_full.out
      ```
      Result (200 clips, 40/40 extractable courts): NO-GO at 0.3 (frame-loss bias, 50:7
      directional); diagnosed as under-scored players, not recall misses; re-run at 0.15
      gives GO, rtmlib matching or beating mmpose on the parity metrics on 199/200,
      shipping 0.15 with one documented hard-clip residual (`2_1_10_2`). Full account:
      `06_phase_a_decision.md`.

**Send me:** `g8_full.out` + `g8_parity_full.json` + `g9_full.out`. This is the
real Phase-A call. I check full video-id coverage, the aggregate + per-clip
keypoint bounds, the directional loss across all courts, and the GO / NO-GO.
Decision recorded: Phase-A GO at 0.15; proceed to Phase B (full 30,487-stem
re-extract + retrain). The 640 detector was considered and rejected (different
ONNX).

---

## Env template: copy to `gpu_env.sh`, edit, `source` it

```bash
# CUDA 13 toolkit for onnxruntime-gpu (cudart/nvrtc/cublas/cufft/curand). REQUIRED
# at gate runtime, not just install time; onnxruntime loads these off LD_LIBRARY_PATH.
module load cuda/13.3

# Put the pip cuDNN (+ CUDA-13) libs on the loader path so onnxruntime finds
# libcudnn.so.9 without an explicit preload_dlls() (the gates don't call it).
# Appended AFTER the module so its consistent CUDA 13.3 toolkit wins; pip fills cuDNN.
NV_LIB=$(~/venv-rtmlib-gpu/bin/python -c "import nvidia,os,glob; print(':'.join(sorted({d for b in nvidia.__path__ for d in glob.glob(os.path.join(b,'*','lib'))})))")
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$NV_LIB"

# Bourbaki data paths (data root: /scratch/comp320a, confirmed 2026-07-03). The
# gates default to the dev-pool copies, so these overrides are required here.
# NOTE: use the plain dirs, NOT the *_unknown variants sitting alongside them.
export RTMLIB_GATE_CLIPS=/scratch/comp320a/ShuttleSet/clips
export RTMLIB_GATE_RAW=/scratch/comp320a/ShuttleSet_keypoints_raw
export RTMLIB_GATE_CLEAN=/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor
export RTMLIB_GATE_STEMFILE=/scratch/comp320a/ShuttleSet_keypoints_raw_provenance/_smoke50.txt
export RTMLIB_GATE_EXTRACT_LIST=/scratch/comp320a/ShuttleSet_keypoints_raw_provenance/stems_to_extract.txt
# CLIPS is globbed recursively (**/<stem>.mp4); if the .mp4s aren't under
# ShuttleSet/clips, point CLIPS at whatever dir holds them. Sanity-check first:
#   ls /scratch/comp320a/ShuttleSet/clips | head
#   ls /scratch/comp320a/ShuttleSet_keypoints_raw_provenance/{_smoke50,stems_to_extract}.txt

# Model cache + vendor dir (keep XDG_CACHE_HOME consistent across verify + gates).
export XDG_CACHE_HOME="$HOME/.cache"
export RTMLIB_MODEL_VENDOR_DIR="$HOME/rtmlib_models_vendored"

# Pin CPU threads only for a determinism check; the GPU gates ignore it.
# export OMP_NUM_THREADS=1
```

## What each gate proves / what I check

| Phase | Gate | Proves | I verify |
|-------|------|--------|----------|
| G-1 | G7 self-variance | the CUDA extract is reproducible run-to-run | `eps_kp` med ≤ 3px, `eps_fail` ≤ 0.02, PASS |
| G-2 | G8 parity (smoke50) | rtmlib≈mmpose on video-11 (value + deployed) | `dF=0`, fmatch, kp_med/p90 above floor, directional |
| G-3 | G9 decision (smoke50) | pipeline end-to-end; provisional | all hard criteria pass, coverage 1/44 gives GO(prov) |
| G-4 | G8+G9 (stratified) | parity across every court; authoritative | full coverage, per-clip bounds, final GO/NO-GO |

## If a gate fails

Send me the output regardless; a NO-GO is a result, not a dead end. The likely
one is the documented one-directional frame-loss bias (rtmlib's 320-input
detector misses a salient player on hard/blur contact frames): if G9's directional
line shows a large `rtmlib-only-fail` excess, that is the model behaviour the 0.15
keep-threshold already addresses (post-inference filter, model unchanged); the 640
detector was considered and rejected as a different ONNX. Re-diagnose with
`diag_g4_fails.py`.
