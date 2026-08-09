# Issue 15 Batch 5 preflight handoff

Status: ready except for the protected commentary credential file on Bourbaki;
the external trial has not started.

This document freezes the state immediately before Batch 5 of
`issue_15_implementation_plan.md`. It records what is implemented, what the
Bourbaki preflight established, and the exact conditions for starting and
accepting the bounded external trial.

## Git and implementation state

The implementation source is the local `issue-15-dataset-builder` worktree at
commit `449d8b1935b1c084df4fbf58ba984ef417d3f30e`.

The completed implementation commits are:

1. `ee1d5f0 Add dataset-builder run contracts and video metadata`
2. `d4fd8f0 Wire full-video extraction and annotation`
3. `94c50f6 Assemble provisional rally records`
4. `449d8b1 Add the end-to-end dataset-builder command`

The post-Batch 4 local acceptance gate passed:

- 1,508 tests passed, 29 skipped, with the 31 known warnings unchanged.
- Repository-wide Ruff passed.
- Configured whole-project Pyrefly passed with 0 errors and 12 suppressions.
- The largest production module is 999 lines and the largest changed function
  is 98 lines.
- The final fresh adversarial review reported zero findings and preserved the
  exact Git state.

The implementation plan is tracked. `main` has not been modified, rebased, or
merged into this worktree.

## Git publication boundary

Bourbaki is an execution host only. Do not commit, push, open a pull request, or
modify `main` from the Bourbaki clone.

All source commits, branch pushes, and pull-request updates must originate from
the local `issue-15-dataset-builder` worktree. After the trial, copy only the
small report evidence needed by the approved plan back into this worktree. Keep
videos, arrays, caches, model files, logs, credentials, and operational
manifests outside Git.

The Bourbaki repository was created from a verified Git bundle rather than a
push. It is a clean execution copy of `449d8b1` with only ignored model
checkpoints added.

## Bourbaki layout

The writable allocation is `/scratch/cmarti`, not
`/scratch/comp320a/cmarti56`.

| Path | Purpose | Git status |
| --- | --- | --- |
| `/scratch/cmarti/issue15_449d8b1/repo` | Exact source clone at `449d8b1` | Clean tracked state |
| `/scratch/cmarti/issue15_449d8b1/external/trial-run` | First run and unchanged resume | Absent before launch |
| `/scratch/cmarti/issue15_449d8b1/overlay` | Isolated Python, Deno, and interpreter wrappers | Outside Git |
| `/scratch/cmarti/issue15_449d8b1/external/config` | Isolated yt-dlp configuration | Outside Git; yt-dlp config mode 600 |
| `/scratch/cmarti/issue15_449d8b1/external/cache` | Deno, Hugging Face, Torch, and XDG caches | Outside Git |
| `/scratch/cmarti/issue15_449d8b1/logs` | Setup and trial logs | Outside Git |
| `/scratch/cmarti/issue15_449d8b1/credentials.env` | Gemini key environment file | Outside Git; currently absent |

The trial run directory is deliberately absent. No search, transcript, model,
download, annotation, assembly, or resume stage has run against external data.

## Transfer and model integrity

The first source-bundle copy was interrupted. Its SHA-256 differed from the
local bundle and `git clone` rejected it with `early EOF`. The copy was resumed
with verification before cloning; no truncated source was used.

The settled source bundle SHA-256 is:

`6ba6a3f0ca0026b7bc3e75a4e0df9e069c65bc7f31101afc6d5c1b07245a2895`

The required model files are present and match these MD5 values:

| Model | Repository-relative path | MD5 |
| --- | --- | --- |
| TrackNet | `src/shared/tracknetv3/ckpts/TrackNet_best.pt` | `6540c256b1237cacdea3d05c16de8353` |
| InpaintNet | `src/shared/tracknetv3/ckpts/InpaintNet_best.pt` | `25aecc665050480a9bfb2fe2df275d14` |
| CourtKeyNet | `src/courtkeynet/weights/courtkeynet_finetuned.safetensors` | `94fb21c26a12f0e9aa20df5a443d8bb2` |

The TrackNet and InpaintNet files live in the repository's ignored checkpoint
directory. CourtKeyNet remains the tracked source file from `449d8b1`.

## Runtime findings

Bourbaki exposes one NVIDIA A100-PCIE-40GB with 40,960 MiB VRAM. Network DNS,
`ffmpeg`, and `ffprobe` are available.

The existing GPU environments are usable but require regular-file interpreter
wrappers:

- TrackNet source interpreter:
  `/home/cmarti56/venv-cleanup-dedup-gpu/bin/python`
- TrackNet wrapper:
  `/scratch/cmarti/issue15_449d8b1/overlay/bin/issue15-tracknet-python`
- RTMLib source interpreter:
  `/home/cmarti56/venv-rtmlib-gpu/bin/python`
- RTMLib wrapper:
  `/scratch/cmarti/issue15_449d8b1/overlay/bin/issue15-pose-python`

The wrappers are required for two reasons:

1. `resolve_interpreter` canonicalises symlinks. Pointing it directly at these
   venv launchers resolves to an underlying system or Conda executable and can
   lose the intended venv context when the child stage runs.
2. RTMLib needs its NVIDIA library directories for ONNX CUDA, but exporting
   those directories in the coordinator process caused the coordinator's
   PySceneDetect/OpenCV import to exceed a five-minute ceiling. The pose wrapper
   adds the libraries only for the pose child process.

With the wrappers in place:

- TrackNet imports OpenCV, NumPy, and Torch and reports the A100 as CUDA
  available.
- RTMLib imports successfully and reports TensorRT, CUDA, and CPU ONNX
  providers.
- The coordinator imports PySceneDetect normally.
- The concrete `DefaultPipelineRuntime.preflight()` passes at source commit
  `449d8b1` and records both wrapper paths as interpreter provenance.

A direct diagnostic initially omitted `src/bst_x` from `PYTHONPATH` and could
not import `pipeline`. This was a diagnostic setup error: the real coordinator
adds `src/bst_x` before importing its runtime. The corrected concrete preflight
passes with the same path that the launch environment records below.

## Isolated dependencies

The coordinator overlay contains:

| Package | Version |
| --- | --- |
| `google-genai` | 2.17.0 |
| `bert-score` | 0.3.13 |
| `yt-dlp` | 2026.03.17 |
| `scenedetect` | 0.7.1 |
| `yt-dlp-ejs` | 0.8.0 |
| `bgutil-ytdlp-pot-provider` | 1.3.1 |

YouTube extraction uses checksum-verified Deno 2.8.1. The official archive
SHA-256 is
`2d7bb6195226ac832e0bf7109a115f0af65ee69ac797a4bbde5b27a06cc242d9`.
The bgutil provider checkout is exactly tag 1.3.1 at
`7608dd51ee813b48cf9a6d68c6e42cb197ce10e0`, with no tracked changes.

This setup follows the upstream [yt-dlp EJS guide][ejs], the pinned
[yt-dlp-ejs release][ejs-release], and the [bgutil provider instructions][pot].

[ejs]: https://github.com/yt-dlp/yt-dlp/wiki/EJS
[ejs-release]: https://github.com/yt-dlp/ejs/releases/tag/0.8.0
[pot]: https://github.com/Brainicism/bgutil-ytdlp-pot-provider/tree/1.3.1

The isolated yt-dlp configuration at
`external/config/yt-dlp/config` contains:

```text
--js-runtimes deno:/scratch/cmarti/issue15_449d8b1/overlay/deno/bin/deno
--extractor-args youtubepot-bgutilscript:server_home=/scratch/cmarti/issue15_449d8b1/external/bgutil-ytdlp-pot-provider/server
```

## Remaining launch gate

The only missing runtime prerequisite is a protected Gemini credential file on
Bourbaki. No key is exported on the execution host, and no standard dotenv or
shell-profile file there contains an assignment. Commentary must not be
disabled to bypass this gate because the approved Batch 5 plan requires
commentary credentials.

Do not paste the key into a command argument, log, issue, pull request, or this
repository. Create the protected environment file interactively on Bourbaki:

```bash
umask 077
read -rsp 'Gemini API key: ' GEMINI_API_KEY; echo
printf 'export GEMINI_API_KEY=%q\n' "$GEMINI_API_KEY" \
  > /scratch/cmarti/issue15_449d8b1/credentials.env
chmod 600 /scratch/cmarti/issue15_449d8b1/credentials.env
unset GEMINI_API_KEY
```

Before launch, verify only that the file is a regular, non-symlink file owned by
the trial account, has mode 600, and defines a non-empty `GEMINI_API_KEY`. Never
print or persist its value.

## Exact launch environment

Run from Bourbaki after the credential gate passes:

```bash
TRIAL_ROOT=/scratch/cmarti/issue15_449d8b1
REPO="$TRIAL_ROOT/repo"
RUN_DIR="$TRIAL_ROOT/external/trial-run"
BASE_PYTHON=/home/cmarti56/venv-cleanup-dedup-gpu/bin/python

set -a
source "$TRIAL_ROOT/credentials.env"
set +a

unset LD_LIBRARY_PATH
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src:$REPO/src/bst_x:$TRIAL_ROOT/overlay/lib/python3.12/site-packages"
export PATH="$TRIAL_ROOT/overlay/bin:$TRIAL_ROOT/overlay/deno/bin:$PATH"
export BADMINTON_TRACKNET_PYTHON="$TRIAL_ROOT/overlay/bin/issue15-tracknet-python"
export BADMINTON_POSE_PYTHON="$TRIAL_ROOT/overlay/bin/issue15-pose-python"
export XDG_CONFIG_HOME="$TRIAL_ROOT/external/config"
export XDG_CACHE_HOME="$TRIAL_ROOT/external/cache/xdg"
export DENO_DIR="$TRIAL_ROOT/external/cache/deno"
export HF_HOME="$TRIAL_ROOT/external/cache/huggingface"
export TORCH_HOME="$TRIAL_ROOT/external/cache/torch"
export TMPDIR="$TRIAL_ROOT/external/tmp"

mkdir -p "$XDG_CACHE_HOME" "$DENO_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
cd "$REPO"
"$BASE_PYTHON" -m dataset_builder run \
  --config configs/dataset_builder/trial.toml \
  --run-dir "$RUN_DIR"
```

The tracked trial configuration enforces one professional-singles search term,
five discovery results, one download worker, and at most two selected videos.
Do not loosen those bounds for the acceptance run.

## First-run and resume gates

Before the first command:

- Confirm the repository HEAD is exactly `449d8b1` and tracked state is clean.
- Recheck all three model MD5 values.
- Confirm the run directory is absent.
- Confirm CUDA, RTMLib providers, `ffprobe`, yt-dlp, Deno, EJS, and the bgutil
  provider are available through the isolated paths above.
- Confirm the credential file passes its ownership, type, mode, and non-empty
  assignment checks without printing the value.

After a successful first run:

1. Record the MD5 values of `run_manifest.json.gz`,
   `rally_records.json.gz`, `dataset_builder_report.json.gz`, and
   `selected_videos.csv.gz`.
2. Run the exact same command once more with no source, model, input,
   configuration, interpreter, credential-name, or path changes.
3. Require all four publications to remain byte-identical.
4. Require every existing stage record to remain unchanged. That demonstrates
   that every reusable stage took the reuse path rather than being rewritten.
5. Scan the decompressed manifest, records, report, selection, commentary JSON,
   TOML, and trial log for the exact secret value. The scan must find none and
   must never print the value.

## Evidence required for the Batch 5 report

The final tracked report must contain the evidence required by the approved
plan:

- Source commit and exact first/resume command.
- Every stage fingerprint, outcome, count, reason, and resume result.
- Ordered search candidates and selection decisions.
- Selected video IDs, source basenames, finite CFR FPS values, dimensions, and
  positive frame counts.
- Rally and accepted-contact counts per video and in aggregate.
- Commentary coverage, transcript/cleaning methods, missing reasons, and proof
  that commentary unavailability did not remove a selected visual record.
- Exclusions and failures.
- External artifact paths, sizes, and MD5 values.
- Raw replay-mask and definitive-exclusion-mask references, shapes, boolean
  dtypes, frame alignment, and integrity values.
- Proof that ordered TrackNet indices and every frame-aligned array match the
  canonical frame count.
- Byte-identical unchanged-resume publications and the reuse result for every
  stage.
- Final acceptance result against every gate in the implementation plan.

Only after that evidence passes should the report be copied into this local
worktree and committed with the approved Batch 5 message:

`Record the issue 15 end-to-end trial`

Until then, the correct state is “Batch 5 ready, credential pending, trial not
started.”
