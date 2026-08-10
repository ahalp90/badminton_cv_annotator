# Issue 15 Batch 5 preflight handoff

Status: first external attempt stopped; Batch 5A is committed and Batch 5B is
implemented locally, with the replacement Bourbaki gates still pending.

This document began as the state freeze immediately before Batch 5 of
`issue_15_implementation_plan.md`. It now also records the stopped first
attempt and the exact conditions for starting and accepting its replacement.
The original pre-launch evidence remains below as a historical record.

## Current Batch 5 state

The guarded first attempt ran on Bourbaki from
2026-08-09T16:06:26+10:00 to 2026-08-10T16:10:05+10:00. It used the clean
execution clone at exact commit `449d8b1`. The run directory is preserved at
`/scratch/cmarti/issue15_449d8b1/external/trial-run` and occupies about 2.7
GiB. Bourbaki remains execution-only. Nothing was committed or pushed there.

The supplied protected credential passed file ownership, type, mode-600, and
non-empty checks. Google rejected it with HTTP 401
`ACCESS_TOKEN_TYPE_UNSUPPORTED`. Transcript acquisition retained two of five
candidate transcripts, relevance triage recorded `unavailable`, and the
reviewed visual fallback still selected and downloaded two videos. This proves
the unavailable-commentary visual lane, but it does not satisfy the required
live-commentary acceptance gate. A valid Gemini credential is still required.

The selected canonical sources were:

| Video ID | Resolution | CFR FPS | Frames | Duration |
| --- | ---: | ---: | ---: | ---: |
| `9WVwZSzixh0` | 1920x1080 | 30 | 153,600 | 85m20s |
| `P3OcTzwmqeY` | 1920x1080 | 30 | 165,150 | 91m45s |

The first stride-1 shuttle stage completed in 45,897.246 seconds. Its
persisted Inpaint mask, TrackNet CSV, and compressed shuttle-array MD5 values
are respectively:

- `804a5e101f620bc78570d23140a0df22`
- `84f0d1da6ac70d962987b983eadcf6d6`
- `c5973aef1f7473b473d30291a0184a49`

The second stride-1 shuttle process reached batch 7,804 of about 10,322 after
10:23:42 of inference. It had not published a CSV or array. At the user's
request, the exact coordinator PID received SIGTERM. Its launcher recorded
exit 143. The third-party TrackNet subprocess became an orphan and was then
terminated by exact PID. The supervisor recorded
`resume_not_started=first_run_failed` and exited 1. No resume was launched.
No dataset-builder, TrackNet, launcher, or supervisor process remained, and
the A100 was clear after shutdown.

The final pre-stop log snapshot was 505,423 bytes with SHA-256
`caccf26b1f4301c7359d29101508fbc46448e846d0732c3f89544f02073ac669`.
All completed and partial run artefacts remain untouched under the old scratch
root.

Local commit `5337163` implements the independently reviewed Batch 5A path:
an integrity-checked, lossless 512x288 FFV1 AVI made with FFmpeg bicubic
scaling and explicit square pixels, followed by TrackNet stride 8. The Batch
5B working tree integrates Issue 37 direct-seek pose sharding while retaining
the exact sequential producer for `pose_shards = 1`. Its deterministic local
test and static-analysis gate has passed; the source-specific seek, A100 RTMLib
parity, and worker-scaling gates remain external prerequisites rather than
claimed local results.

The replacement run must use a new clean execution clone and a new run root
from the final performance-extension commit. Do not reuse or mutate the
stopped `449d8b1` run. Before restarting the bounded two-video trial:

1. complete and locally gate the FFmpeg bicubic 512x288 TrackNet-input stage;
2. use TrackNet stride 8 while retaining the canonical 1080p masters;
3. pass the fixed-source frame and numerical checks on Bourbaki;
4. enable the multi-shard trial path only after its exact RTMLib parity gate;
5. provide a valid protected Gemini credential; and
6. rerun the unchanged-resume and secret-persistence gates from this handoff.

## Git and implementation state

The Batch 5 performance extension starts from local commit `5337163`; the
Batch 5B pose-sharding changes follow as their own reviewed commit.

The completed implementation commits are:

1. `ee1d5f0 Add dataset-builder run contracts and video metadata`
2. `d4fd8f0 Wire full-video extraction and annotation`
3. `94c50f6 Assemble provisional rally records`
4. `449d8b1 Add the end-to-end dataset-builder command`
5. `51fa592 Document the issue 15 Batch 5 preflight`
6. `7f7bf3c Document dataset-builder throughput research`
7. `5337163 Speed up full-video shuttle extraction`
8. `Shard full-video pose extraction` (this Batch 5B commit)

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

The preserved first-attempt Bourbaki repository was created from a verified
Git bundle rather than a push. It is a clean execution copy of `449d8b1` with
only ignored model checkpoints added.

## Preserved first-attempt Bourbaki layout

The writable allocation is `/scratch/cmarti`, not
`/scratch/comp320a/cmarti56`.

| Path | Purpose | Git status |
| --- | --- | --- |
| `/scratch/cmarti/issue15_449d8b1/repo` | Exact source clone at `449d8b1` | Clean tracked state |
| `/scratch/cmarti/issue15_449d8b1/external/trial-run` | Preserved stopped first attempt | Outside Git; about 2.7 GiB |
| `/scratch/cmarti/issue15_449d8b1/overlay` | Isolated Python, Deno, and interpreter wrappers | Outside Git |
| `/scratch/cmarti/issue15_449d8b1/external/config` | Isolated yt-dlp configuration | Outside Git; yt-dlp config mode 600 |
| `/scratch/cmarti/issue15_449d8b1/external/cache` | Deno, Hugging Face, Torch, and XDG caches | Outside Git |
| `/scratch/cmarti/issue15_449d8b1/logs` | Setup and trial logs | Outside Git |
| `/scratch/cmarti/issue15_449d8b1/credentials.env` | Rejected Gemini credential file | Outside Git; protected mode 600 |

Before the first attempt, the trial directory was deliberately absent. The
current preserved contents and terminal state are recorded in the current
Batch 5 section above.

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

## Replacement-root template

The live replacement command cannot be made exact inside the Batch 5B commit
because that commit cannot contain its own Git hash. After the local commit,
record its full hash and create a new root named for that commit, for example
`/scratch/cmarti/issue15_<short-Batch5B-hash>`. The replacement must use:

- a new clean clone at the recorded full Batch 5B commit;
- a new absent `external/trial-run` path beneath the new root;
- newly verified wrappers, overlays, caches, configuration, and protected
  credential paths beneath the new root; and
- a launcher-owned coordinator process group. Cancellation sends `SIGTERM` to
  that recorded group and waits for it; during sharded pose, the coordinator
  forwards cancellation into its separately owned pose-worker group and reaps
  the direct child; and
- the source-specific seek, A100 parity, and worker-scaling gates before the
  two-video E2E command.

Do not substitute `issue15_449d8b1` for the placeholder. That root and its
2.7 GiB run are preserved evidence. The exact replacement paths and commands
must be generated after the Batch 5B hash exists and recorded in the external
worklog before transfer.

## Historical first-attempt credential gate (do not rerun)

At the pre-launch freeze, the only missing runtime prerequisite was a
protected Gemini credential file on Bourbaki. The first supplied credential
was later rejected as described above. Commentary must not be disabled to
bypass this gate because the approved Batch 5 plan requires live commentary.

The credential was entered with the following protected first-attempt command.
It is retained as historical evidence, not as a replacement command. Never
paste a key into a command argument, log, issue, pull request, or this
repository.

```bash
umask 077
read -rsp 'Gemini API key: ' GEMINI_API_KEY; echo
printf 'export GEMINI_API_KEY=%q\n' "$GEMINI_API_KEY" \
  > /scratch/cmarti/issue15_449d8b1/credentials.env
chmod 600 /scratch/cmarti/issue15_449d8b1/credentials.env
unset GEMINI_API_KEY
```

The replacement credential needs the same regular-file, ownership, mode-600,
and non-empty-variable checks at its new path. Never print or persist its
value.

## Historical first-attempt launch environment (do not rerun)

The following block records the stopped `449d8b1` invocation. It targets the
preserved old run and must not be executed again:

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

## Replacement first-run and resume gates

Before the replacement command:

- Confirm repository HEAD is the recorded full Batch 5B commit and tracked
  state is clean.
- Recheck all three model MD5 values.
- Confirm the new run directory is absent and is not beneath
  `/scratch/cmarti/issue15_449d8b1`.
- Confirm CUDA, RTMLib providers, `ffprobe`, yt-dlp, Deno, EJS, and the bgutil
  provider are available through the isolated paths above.
- Confirm the credential file passes its ownership, type, mode, and non-empty
  assignment checks without printing the value.
- Record the launcher, coordinator PID, and coordinator process-group ID. The
  coordinator must be the leader of its group before the trial starts.

For a controlled stop, send `SIGTERM` to the exact recorded coordinator process
group, wait for the launcher to finish, and then verify that no coordinator,
external child, pose orchestrator, or shard-worker process remains and that the
A100 is clear. Do not repeat the first attempt's exact-PID-only cancellation.

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

Until the performance batches and replacement external gates pass, the correct
state is “Batch 5 first attempt stopped; final acceptance pending.”
