#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
readonly DEFAULT_SOURCE=/srv/mergerfs/main_pool/Scratch_Backup/Uni/cosc595/issue-31-shuttle-hallucination-audit-assets/downloads/yu9oyMXRGHY.mp4
readonly DEFAULT_ARTIFACTS=/srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/issue-38-vlm-benchmark/artifacts
readonly DEFAULT_LOCAL_RESULTS=/srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/issue-38-vlm-benchmark/runs
readonly DEFAULT_REMOTE_HOST=carmack
readonly DEFAULT_REMOTE_ROOT=/scratch/cmarti56/issue38-vlm
readonly SOURCE_MD5=2827bca5d829cde15591dc110f5b2904
readonly SOURCE_SHA256=cbad108386055835bcd6e479adc297e18eb2d0df7ae2310857589f523bb3785f
readonly INTERN_MODEL=yanziang/InternVideo3-8B-Instruct
readonly INTERN_REVISION=c4602918b65225650d152db2850fe34e01d21fcd
readonly QWEN_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
readonly QWEN_REVISION=d9748a51ae66354c4dad665aab2c71f26cf2c8cd
readonly INTERN_IMAGE_SHA256=fd1c42ea24386dde021f12c0fe9458f0d4f5f43ea97af2ad19c2b3ea9925c76a
readonly QWEN_IMAGE_SHA256=1cf06bf5a8a7bd5a2b2c469f0e72ac150f0781c126b593c7fcd9d7df4eb34d37
readonly SMOKE_MANIFEST_REL=smoke/sset_15_f18419_f18669_512x288_manifest.json
readonly LONG20_MANIFEST_REL=long20/sset_15_f18419_f48419_512x288_manifest.json
readonly FINE_MANIFEST_REL=fine/sset_15_f20695_f20945_512x288_manifest.json
readonly TRUTH_REL=docs/scraper_pipeline/broadcast_nonstandard_camera_id/data/sset_15_broadcast_timeline_labels.csv.gz

COMMAND=all
SOURCE=$DEFAULT_SOURCE
ARTIFACTS=$DEFAULT_ARTIFACTS
LOCAL_RESULTS=$DEFAULT_LOCAL_RESULTS
REMOTE_HOST=$DEFAULT_REMOTE_HOST
REMOTE_ROOT=$DEFAULT_REMOTE_ROOT
RUN_TAG=
RUN_FINGERPRINT=
RUN_ROOT=
LOCAL_RUN_ROOT=
POLL_SECONDS=15
PLAN_ONLY=0
SSH_COMMAND=${ISSUE38_SSH_COMMAND:-}
PASSED_CANDIDATES=()

usage() {
  cat <<'EOF'
Usage: run_carmack.sh [COMMAND] [OPTIONS]

Commands:
  all       Prepare, stage, set up, smoke-test, gate, run full, collect, score.
  check     Run read-only local and Carmack prerequisite checks.
  prepare   Prepare and validate the exact smoke and full video inputs.
  stage     Copy an inference-only snapshot and exact inputs to Carmack.
  setup     Pull verified images and create pinned environments under tmux.
  smoke     Run both smoke tests sequentially, collect, and apply the gate.
  full      Run the 20-minute Intern and fine-grained Qwen candidate passes.
  qwen-fine Run only the Qwen fine pass after its smoke gate, then score it.
  collect   Download retained logs, records, status, and provenance.
  score     Gate and score successful candidate records locally against human truth.
  status    Show retained status and active tmux sessions for this run.

Options:
  --source PATH         Frozen sset_15 source video.
  --artifacts PATH      Local prepared artifact directory.
  --local-results PATH  Local root for collected run evidence.
  --remote-host HOST    SSH host alias. Default: carmack.
  --remote-root PATH    Dedicated Carmack root. Default: /scratch/cmarti56/issue38-vlm.
  --run-tag TAG         Override the content-derived run tag.
  --poll-seconds N      Seconds between tmux status checks. Default: 15.
  --ssh-command PATH    Executable SSH-compatible wrapper for a nested hop.
  --plan                Print the resolved workflow without checking or changing anything.
  -h, --help            Show this help.

The default command is `all`.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

parse_arguments() {
  if (($# > 0)) && [[ $1 != -* ]]; then
    COMMAND=$1
    shift
  fi
  while (($# > 0)); do
    case "$1" in
      --source) SOURCE=${2:?--source requires a path}; shift 2 ;;
      --artifacts) ARTIFACTS=${2:?--artifacts requires a path}; shift 2 ;;
      --local-results) LOCAL_RESULTS=${2:?--local-results requires a path}; shift 2 ;;
      --remote-host) REMOTE_HOST=${2:?--remote-host requires a host}; shift 2 ;;
      --remote-root) REMOTE_ROOT=${2:?--remote-root requires a path}; shift 2 ;;
      --run-tag) RUN_TAG=${2:?--run-tag requires a value}; shift 2 ;;
      --poll-seconds) POLL_SECONDS=${2:?--poll-seconds requires a number}; shift 2 ;;
      --ssh-command) SSH_COMMAND=${2:?--ssh-command requires a path}; shift 2 ;;
      --plan) PLAN_ONLY=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown option: $1" ;;
    esac
  done
  case "$COMMAND" in
    all|check|prepare|stage|setup|smoke|full|qwen-fine|collect|score|status) ;;
    *) die "unknown command: $COMMAND" ;;
  esac
  [[ $POLL_SECONDS =~ ^[1-9][0-9]*$ ]] || die "--poll-seconds must be a positive integer"
  [[ $REMOTE_HOST =~ ^[A-Za-z0-9_.@-]+$ ]] || die "remote host contains unsupported characters"
  [[ $REMOTE_ROOT =~ ^/scratch/[A-Za-z0-9_./-]+$ ]] || die "remote root must be a safe path under /scratch"
  [[ $REMOTE_ROOT != *..* ]] || die "remote root must not contain '..'"
  if [[ -n $RUN_TAG ]]; then
    [[ $RUN_TAG =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "run tag contains unsupported characters"
  fi
}

print_plan() {
  local execution
  if [[ $COMMAND == qwen-fine ]]; then
    execution="verify/reuse staged setup -> Qwen smoke gate -> Qwen fine pass(tmux) -> collect -> score"
  else
    execution=$'prepare -> stage -> setup(tmux) -> intern smoke(tmux) -> qwen smoke(tmux)\n           -> per-model truth-free gates -> Intern 20-minute / Qwen fine passes(tmux) -> collect -> score'
  fi
  cat <<EOF
Issue 38 VLM benchmark plan
command: $COMMAND
repo: $REPO
source: $SOURCE
artifacts: $ARTIFACTS
local results: $LOCAL_RESULTS
remote: $REMOTE_HOST:$REMOTE_ROOT
SSH command: ${SSH_COMMAND:-ssh}
run tag: ${RUN_TAG:-<content-derived after preparation>}

models:
  $INTERN_MODEL@$INTERN_REVISION
  $QWEN_MODEL@$QWEN_REVISION
runtime images:
  pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime sha256:$INTERN_IMAGE_SHA256
  vllm/vllm-openai:v0.11.0 sha256:$QWEN_IMAGE_SHA256
sampling: Intern [18419, 48419) at 1 FPS; Qwen [20695, 20945) at 5 FPS; 512x288
execution: $execution
isolation: remote staging includes src/annotator, benchmark scripts, and prepared videos only
truth: $TRUTH_REL stays local and is read only by score
resume: completed records are immutable; active tmux sessions are rejoined; caches and verified images are reused
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

ssh() {
  if [[ -n $SSH_COMMAND ]]; then
    "$SSH_COMMAND" "$@"
  else
    command ssh "$@"
  fi
}

configure_remote_shell() {
  if [[ -z $SSH_COMMAND ]]; then
    require_command ssh
    return
  fi
  [[ $SSH_COMMAND == /* ]] || die "--ssh-command must be an absolute path"
  [[ $SSH_COMMAND != *[[:space:]]* ]] || die "--ssh-command path cannot contain whitespace"
  [[ -x $SSH_COMMAND ]] || die "SSH command is not executable: $SSH_COMMAND"
  export RSYNC_RSH=$SSH_COMMAND
}

check_source_identity() {
  [[ -f $SOURCE ]] || die "source video is missing: $SOURCE"
  local actual_md5
  local actual_sha256
  actual_md5=$(md5sum "$SOURCE" | awk '{print $1}')
  actual_sha256=$(sha256sum "$SOURCE" | awk '{print $1}')
  [[ $actual_md5 == "$SOURCE_MD5" ]] || die "source MD5 differs: $actual_md5"
  [[ $actual_sha256 == "$SOURCE_SHA256" ]] || die "source SHA-256 differs: $actual_sha256"
}

check_local_prerequisites() {
  local command
  for command in ffmpeg git md5sum rsync sha256sum uv; do
    require_command "$command"
  done
  check_source_identity
}

check_remote_prerequisites() {
  ssh "$REMOTE_HOST" bash -s -- "$REMOTE_ROOT" "$INTERN_IMAGE_SHA256" "$QWEN_IMAGE_SHA256" <<'REMOTE'
set -euo pipefail
remote_root=$1
intern_sha256=$2
qwen_sha256=$3
for command in apptainer bash flock nvidia-smi sha256sum tmux; do
  command -v "$command" >/dev/null
done
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
df -Pk /scratch
intern_image="$remote_root/images/internvideo3-pytorch-2.8.0-cu129.sif"
qwen_image="$remote_root/images/qwen3-vl-vllm-v0.11.0.sif"
if [[ -f "$intern_image" ]]; then
  printf '%s  %s\n' "$intern_sha256" "$intern_image" | sha256sum --check -
fi
if [[ -f "$qwen_image" ]]; then
  printf '%s  %s\n' "$qwen_sha256" "$qwen_image" | sha256sum --check -
fi
REMOTE
}

prepare_stage() {
  local stage=$1
  local start_frame=$2
  local end_frame=$3
  local manifest_name=$4
  local sample_fps=$5
  local output_dir="$ARTIFACTS/$stage"
  local manifest="$output_dir/$manifest_name"
  if [[ -f $manifest ]]; then
    echo "Reusing prepared $stage manifest: $manifest"
    return
  fi
  if [[ -d $output_dir ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    die "incomplete $stage preparation exists at $output_dir; move it aside before retrying"
  fi
  mkdir -p "$output_dir"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" \
    uv run --no-project \
    --with-requirements "$SCRIPT_DIR/requirements-prepare.txt" \
    python -m annotator.vlm_scene_benchmark.prepare \
    --source "$SOURCE" --output-dir "$output_dir" --video-id sset_15 \
    --start-frame "$start_frame" --end-frame "$end_frame" --sample-fps "$sample_fps"
}

validate_prepared_inputs() {
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" \
    uv run --no-project \
    --with-requirements "$SCRIPT_DIR/requirements-prepare.txt" \
    python - "$SOURCE" "$ARTIFACTS" <<'PY'
from pathlib import Path
import sys

from annotator.vlm_scene_benchmark.prepare import probe_video, read_manifest, resolve_model_video

source_path = Path(sys.argv[1]).resolve()
artifacts = Path(sys.argv[2]).resolve()
source = probe_video(source_path)
expected_source = (
    "cbad108386055835bcd6e479adc297e18eb2d0df7ae2310857589f523bb3785f",
    25.0,
    149_487,
    640,
    360,
)
actual_source = (source.sha256, source.fps, source.frame_count, source.width, source.height)
if actual_source != expected_source:
    raise SystemExit(f"source media differs: {actual_source!r}")

specs = (
    ("smoke/sset_15_f18419_f18669_512x288_manifest.json", 18_419, 18_669, 1.0, 10),
    ("long20/sset_15_f18419_f48419_512x288_manifest.json", 18_419, 48_419, 1.0, 1_200),
    ("fine/sset_15_f20695_f20945_512x288_manifest.json", 20_695, 20_945, 5.0, 50),
)
for relative, start_frame, end_frame, sample_fps, model_frames in specs:
    path = artifacts / relative
    manifest = read_manifest(path)
    if manifest.original_source != source:
        raise SystemExit(f"manifest source differs: {path}")
    shard = manifest.shard
    expected_shard = ("sset_15", 25.0, 149_487, start_frame, end_frame)
    actual_shard = (shard.video_id, shard.fps, shard.frame_count, shard.start_frame, shard.end_frame)
    if actual_shard != expected_shard:
        raise SystemExit(f"manifest shard differs: {path}: {actual_shard!r}")
    expected_model = (sample_fps, model_frames, 512, 288)
    actual_model = (
        manifest.model_video.fps,
        manifest.model_video.frame_count,
        manifest.model_video.width,
        manifest.model_video.height,
    )
    if actual_model != expected_model:
        raise SystemExit(f"model input differs: {path}: {actual_model!r}")
    stride = round(source.fps / sample_fps)
    if manifest.sampled_source_frames != tuple(range(start_frame, end_frame, stride)):
        raise SystemExit(f"sampled frame grid differs: {path}")
    reference_path = path.parent / manifest.reference_video.file_name
    if probe_video(reference_path) != manifest.reference_video:
        raise SystemExit(f"reference video differs from manifest: {reference_path}")
    resolve_model_video(path, manifest)
    print(f"validated {path}")
PY
}

prepare_inputs() {
  check_source_identity
  prepare_stage smoke 18419 18669 "${SMOKE_MANIFEST_REL#smoke/}" 1
  prepare_stage long20 18419 48419 "${LONG20_MANIFEST_REL#long20/}" 1
  prepare_stage fine 20695 20945 "${FINE_MANIFEST_REL#fine/}" 5
  validate_prepared_inputs
}

runtime_file_list() {
  git -C "$REPO" ls-files --cached --others --exclude-standard -- \
    src/annotator scripts/vlm_scene_benchmark | LC_ALL=C sort
}

runtime_fingerprint() {
  {
    while IFS= read -r path; do
      printf '%s  %s\n' "$(sha256sum "$REPO/$path" | awk '{print $1}')" "$path"
    done < <(runtime_file_list)
    printf '%s  %s\n' "$(sha256sum "$ARTIFACTS/$SMOKE_MANIFEST_REL" | awk '{print $1}')" "$SMOKE_MANIFEST_REL"
    printf '%s  %s\n' "$(sha256sum "$ARTIFACTS/$LONG20_MANIFEST_REL" | awk '{print $1}')" "$LONG20_MANIFEST_REL"
    printf '%s  %s\n' "$(sha256sum "$ARTIFACTS/$FINE_MANIFEST_REL" | awk '{print $1}')" "$FINE_MANIFEST_REL"
  } | sha256sum | awk '{print $1}'
}

resolve_run_identity() {
  [[ -f $ARTIFACTS/$SMOKE_MANIFEST_REL ]] || die "smoke manifest is missing; run prepare first"
  [[ -f $ARTIFACTS/$LONG20_MANIFEST_REL ]] || die "20-minute manifest is missing; run prepare first"
  [[ -f $ARTIFACTS/$FINE_MANIFEST_REL ]] || die "fine manifest is missing; run prepare first"
  RUN_FINGERPRINT=$(runtime_fingerprint)
  if [[ -z $RUN_TAG ]]; then
    RUN_TAG="issue38-${RUN_FINGERPRINT:0:16}"
  fi
  [[ $RUN_TAG =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "run tag contains unsupported characters"
  RUN_ROOT="$REMOTE_ROOT/runs/$RUN_TAG"
  LOCAL_RUN_ROOT="$LOCAL_RESULTS/$RUN_TAG"
}

write_run_config() {
  local output=$1
  {
    printf 'schema=1\n'
    printf 'run_tag=%s\n' "$RUN_TAG"
    printf 'fingerprint=%s\n' "$RUN_FINGERPRINT"
    printf 'source_sha256=%s\n' "$SOURCE_SHA256"
    printf 'intern_model=%s@%s\n' "$INTERN_MODEL" "$INTERN_REVISION"
    printf 'qwen_model=%s@%s\n' "$QWEN_MODEL" "$QWEN_REVISION"
    printf 'intern_image_sha256=%s\n' "$INTERN_IMAGE_SHA256"
    printf 'qwen_image_sha256=%s\n' "$QWEN_IMAGE_SHA256"
    printf 'intern_candidate=20_minutes_1fps\n'
    printf 'qwen_candidate=10_seconds_5fps_max_model_len_16384\n'
  } >"$output"
}

stage_run_config() {
  local temporary
  local local_sha256
  local remote_sha256
  temporary=$(mktemp)
  write_run_config "$temporary"
  local_sha256=$(sha256sum "$temporary" | awk '{print $1}')
  ssh "$REMOTE_HOST" "mkdir -p '$RUN_ROOT/control' '$RUN_ROOT/repo' '$RUN_ROOT/artifacts' '$RUN_ROOT/results' '$RUN_ROOT/logs' '$RUN_ROOT/status' '$RUN_ROOT/tmp'"
  rsync -a --ignore-existing "$temporary" "$REMOTE_HOST:$RUN_ROOT/control/run.env"
  remote_sha256=$(ssh "$REMOTE_HOST" "sha256sum '$RUN_ROOT/control/run.env'" | awk '{print $1}')
  rm -f -- "$temporary"
  [[ $remote_sha256 == "$local_sha256" ]] || die "run tag $RUN_TAG already names a different configuration"
}

stage_runtime_snapshot() {
  local filter_file
  stage_run_config
  filter_file=$(mktemp)
  runtime_file_list | awk -F/ '
    {
      prefix = ""
      for (field = 1; field < NF; field++) {
        prefix = prefix $field "/"
        if (!seen[prefix]++) print "+ /" prefix
      }
      print "+ /" $0
    }
    END { print "- *" }
  ' >"$filter_file"
  rsync -a --delete --delete-excluded \
    --filter="merge $filter_file" \
    "$REPO/" "$REMOTE_HOST:$RUN_ROOT/repo/" || {
      rm -f -- "$filter_file"
      return 1
    }
  rm -f -- "$filter_file"
  rsync -a --delete --delete-excluded \
    --include='/smoke/' \
    --include='/smoke/sset_15_f18419_f18669_512x288_manifest.json' \
    --include='/smoke/sset_15_f18419_f18669_512x288_model_1fps.mp4' \
    --include='/smoke/sset_15_f18419_f18669_512x288_reference_25fps.mp4' \
    --include='/long20/' \
    --include='/long20/sset_15_f18419_f48419_512x288_manifest.json' \
    --include='/long20/sset_15_f18419_f48419_512x288_model_1fps.mp4' \
    --include='/long20/sset_15_f18419_f48419_512x288_reference_25fps.mp4' \
    --include='/fine/' \
    --include='/fine/sset_15_f20695_f20945_512x288_manifest.json' \
    --include='/fine/sset_15_f20695_f20945_512x288_model_5fps.mp4' \
    --include='/fine/sset_15_f20695_f20945_512x288_reference_25fps.mp4' \
    --exclude='*' \
    "$ARTIFACTS/" "$REMOTE_HOST:$RUN_ROOT/artifacts/"
  ssh "$REMOTE_HOST" bash -s -- "$RUN_ROOT" <<'REMOTE'
set -euo pipefail
run_root=$1
cd "$run_root"
find repo/src/annotator repo/scripts/vlm_scene_benchmark artifacts -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum >control/staged.sha256.tmp
mv -- control/staged.sha256.tmp control/staged.sha256
sha256sum --check --quiet control/staged.sha256
REMOTE
  echo "Staged inference-only snapshot: $REMOTE_HOST:$RUN_ROOT"
}

session_name() {
  local key=$1
  local digest
  digest=$(printf '%s' "$REMOTE_HOST:$RUN_ROOT:$key" | sha256sum | awk '{print $1}')
  printf 'i38-%s-%s\n' "${digest:0:16}" "$key"
}

remote_task_command() {
  local task=$1
  local backend=${2:-_}
  local stage=${3:-_}
  printf '%q ' \
    "$RUN_ROOT/repo/scripts/vlm_scene_benchmark/remote_task.sh" \
    "$task" "$REMOTE_ROOT" "$RUN_ROOT" "$backend" "$stage"
}

remote_status_value() {
  local path=$1
  ssh "$REMOTE_HOST" "test -f '$path' && cat '$path' || true"
}

setup_is_reusable() {
  local status=$1
  [[ $status == 0 ]] || return
  ssh "$REMOTE_HOST" bash -s -- "$RUN_ROOT/control/runtime.env" <<'REMOTE'
set -euo pipefail
runtime_environment=$1
source "$runtime_environment"
[[ -x ${INTERN_ENVIRONMENT:-}/bin/python || -L ${INTERN_ENVIRONMENT:-}/bin/python ]]
[[ -f ${INTERN_ENVIRONMENT:-}/.issue38-ready ]]
[[ -d ${INTERN_FFMPEG_ENVIRONMENT:-}/lib ]]
[[ -f ${INTERN_FFMPEG_ENVIRONMENT:-}/.issue38-ready ]]
[[ -x ${QWEN_ENVIRONMENT:-}/bin/python ]]
[[ -f ${QWEN_ENVIRONMENT:-}/.issue38-ready ]]
REMOTE
}

remote_result_exists() {
  local key=$1
  ssh "$REMOTE_HOST" "test -f '$RUN_ROOT/results/$key.json'"
}

remote_has_attempt_evidence() {
  local key=$1
  ssh "$REMOTE_HOST" bash -s -- "$RUN_ROOT" "$key" <<'REMOTE'
set -euo pipefail
run_root=$1
key=$2
for path in "$run_root/results/$key".attempt-*.txt; do
  if [[ -e "$path" ]]; then
    exit 0
  fi
done
exit 1
REMOTE
}

recover_completed_benchmark_status() {
  local backend=$1
  local stage=$2
  local status_path=$3
  local recovered_status
  if remote_gate "$backend" "$stage"; then
    recovered_status=0
  else
    recovered_status=$?
  fi
  ((recovered_status == 0 || recovered_status == 4)) || return "$recovered_status"
  ssh "$REMOTE_HOST" bash -s -- "$status_path" "$recovered_status" <<'REMOTE' || return 71
set -euo pipefail
status_path=$1
status=$2
printf '%s\n' "$status" >"$status_path.tmp"
mv -- "$status_path.tmp" "$status_path"
REMOTE
  return "$recovered_status"
}

print_new_tag_retry() {
  local retry_tag="${RUN_TAG}-retry-$(date -u +%Y%m%dT%H%M%SZ)"
  local -a retry_command=(
    "$SCRIPT_DIR/run_carmack.sh" all
    --source "$SOURCE"
    --artifacts "$ARTIFACTS"
    --local-results "$LOCAL_RESULTS"
    --remote-host "$REMOTE_HOST"
    --remote-root "$REMOTE_ROOT"
    --poll-seconds "$POLL_SECONDS"
    --run-tag "$retry_tag"
  )
  if [[ -n $SSH_COMMAND ]]; then
    retry_command+=(--ssh-command "$SSH_COMMAND")
  fi
  echo "Partial model evidence was retained and will not be overwritten." >&2
  echo "Retry the complete isolated run with this new tag:" >&2
  printf '  ' >&2
  printf '%q ' "${retry_command[@]}" >&2
  printf '\n' >&2
}

wait_for_task() {
  local task=$1
  local key=$2
  local backend=${3:-_}
  local stage=${4:-_}
  local session
  local status_path
  local failure_status_path
  local log_path="$RUN_ROOT/logs/$key.log"
  local task_command
  local launch_command
  local status
  local complete_result=0
  local last_log_line=
  local current_log_line
  session=$(session_name "$key")
  if [[ $task == setup ]]; then
    status_path="$RUN_ROOT/status/setup.ok"
    failure_status_path="$RUN_ROOT/status/setup.last-exit"
  else
    status_path="$RUN_ROOT/status/$key.status"
    failure_status_path=$status_path
  fi
  status=$(remote_status_value "$status_path")
  if [[ -n $status ]]; then
    if [[ $task != setup ]] || setup_is_reusable "$status"; then
      echo "Reusing completed $key task with status $status."
      return "$status"
    fi
    echo "Setup status exists without a complete runtime environment. Running setup again."
  fi
  if [[ $task == benchmark ]] && remote_result_exists "$key"; then
    complete_result=1
    if recover_completed_benchmark_status "$backend" "$stage" "$status_path"; then
      status=0
    else
      status=$?
    fi
    if ((status == 0 || status == 4)); then
      echo "Recovered completed $key task with status $status."
      return "$status"
    fi
  fi
  if ssh "$REMOTE_HOST" "tmux has-session -t '$session' 2>/dev/null"; then
    echo "Rejoining active tmux session $session."
  else
    task_command=$(remote_task_command "$task" "$backend" "$stage")
    printf -v launch_command 'tmux new-session -d -s %q %q' "$session" "$task_command"
    ssh "$REMOTE_HOST" "$launch_command"
    echo "Started tmux session $session."
  fi
  while ssh "$REMOTE_HOST" "tmux has-session -t '$session' 2>/dev/null"; do
    current_log_line=$(ssh "$REMOTE_HOST" "test -f '$log_path' && tail -n 1 '$log_path' || true")
    if [[ -n $current_log_line && $current_log_line != "$last_log_line" ]]; then
      printf '%s\n' "$current_log_line"
      last_log_line=$current_log_line
    fi
    sleep "$POLL_SECONDS"
  done
  status=$(remote_status_value "$status_path")
  if [[ -z $status ]]; then
    status=$(remote_status_value "$failure_status_path")
  fi
  if [[ -z $status ]]; then
    if [[ $task == benchmark ]] && remote_result_exists "$key"; then
      complete_result=1
      if recover_completed_benchmark_status "$backend" "$stage" "$status_path"; then
        status=0
      else
        status=$?
      fi
      if ((status == 0 || status == 4)); then
        echo "Recovered completed $key task with status $status."
        return "$status"
      fi
    fi
    ssh "$REMOTE_HOST" "test -f '$log_path' && tail -n 30 '$log_path' || true" >&2 || true
    if [[ $task == benchmark ]] && ((complete_result == 0)) && remote_has_attempt_evidence "$key"; then
      print_new_tag_retry
    else
      echo "Task $key ended without an atomic status file. Rerun the same command to retry." >&2
    fi
    return 70
  fi
  echo "Task $key finished with status $status."
  return "$status"
}

run_setup_task() {
  wait_for_task setup setup
}

run_model_pair() {
  local stage=$1
  local backend
  local status

  for backend in internvideo3 qwen3-vl; do
    if run_model_task "$backend" "$stage"; then
      continue
    else
      status=$?
    fi
    if ((status != 4)); then
      echo "$backend $stage task could not run; status $status." >&2
      return "$status"
    fi
  done
}

run_model_task() {
  local backend=$1
  local stage=$2
  local status
  if wait_for_task benchmark "$backend-$stage" "$backend" "$stage"; then
    status=0
  else
    status=$?
  fi
  echo "$backend $stage process status: $status"
  return "$status"
}

remote_gate() {
  local backend=$1
  local stage=$2
  local command
  command=$(remote_task_command gate "$backend" "$stage")
  ssh "$REMOTE_HOST" "$command"
}

gate_model_pair() {
  local stage=$1
  local backend
  local status
  local outcome=0
  for backend in internvideo3 qwen3-vl; do
    if remote_gate "$backend" "$stage"; then
      continue
    else
      status=$?
    fi
    if ((status == 4)); then
      outcome=4
    else
      echo "$backend $stage gate could not run; status $status." >&2
      return "$status"
    fi
  done
  return "$outcome"
}

run_candidate_models() {
  local backend
  local stage
  local candidate
  local status
  local -a candidates=()
  local -a completed_candidates=()
  PASSED_CANDIDATES=()
  if remote_gate internvideo3 smoke; then
    candidates+=("internvideo3:long20")
  else
    status=$?
    if ((status == 4)); then
      echo "InternVideo3 failed its smoke gate. Its 20-minute run will not start." >&2
    else
      echo "InternVideo3 smoke gate could not run; status $status." >&2
      return "$status"
    fi
  fi
  if remote_gate qwen3-vl smoke; then
    candidates+=("qwen3-vl:fine")
  else
    status=$?
    if ((status == 4)); then
      echo "Qwen3-VL failed its smoke gate. Its fine-grained run will not start." >&2
    else
      echo "Qwen3-VL smoke gate could not run; status $status." >&2
      return "$status"
    fi
  fi
  if ((${#candidates[@]} == 0)); then
    echo "No model passed its smoke gate." >&2
    return 4
  fi
  for candidate in "${candidates[@]}"; do
    backend=${candidate%%:*}
    stage=${candidate#*:}
    if run_model_task "$backend" "$stage"; then
      completed_candidates+=("$candidate")
      continue
    else
      status=$?
    fi
    if ((status == 4)); then
      echo "$backend failed its $stage task and will not be gated or scored." >&2
    else
      echo "$backend $stage task could not run; status $status." >&2
      return "$status"
    fi
  done
  collect_evidence
  for candidate in "${completed_candidates[@]}"; do
    backend=${candidate%%:*}
    stage=${candidate#*:}
    if remote_gate "$backend" "$stage"; then
      PASSED_CANDIDATES+=("$candidate")
    else
      status=$?
      if ((status == 4)); then
        echo "$backend failed its $stage deployment gate and will not be scored." >&2
      else
        echo "$backend $stage gate could not run; status $status." >&2
        return "$status"
      fi
    fi
  done
  if ((${#PASSED_CANDIDATES[@]} == 0)); then
    echo "No candidate passed its deployment gate." >&2
    return 4
  fi
}

run_qwen_fine_only() {
  local status
  if run_setup_task; then
    :
  else
    status=$?
    collect_evidence
    return "$status"
  fi
  if run_model_task qwen3-vl smoke; then
    :
  else
    status=$?
    if ((status == 4)); then
      echo "Qwen3-VL failed its smoke task. Its fine-grained run will not start." >&2
    else
      echo "Qwen3-VL smoke task could not run; status $status." >&2
    fi
    collect_evidence
    return "$status"
  fi
  if remote_gate qwen3-vl smoke; then
    :
  else
    status=$?
    if ((status == 4)); then
      echo "Qwen3-VL failed its smoke gate. Its fine-grained run will not start." >&2
    else
      echo "Qwen3-VL smoke gate could not run; status $status." >&2
    fi
    return "$status"
  fi
  if run_model_task qwen3-vl fine; then
    status=0
  else
    status=$?
  fi
  collect_evidence
  ((status == 0)) || return "$status"
  remote_gate qwen3-vl fine || return
  score_candidate_records qwen3-vl:fine
}

collect_evidence() {
  mkdir -p \
    "$LOCAL_RUN_ROOT/results" "$LOCAL_RUN_ROOT/logs" \
    "$LOCAL_RUN_ROOT/status" "$LOCAL_RUN_ROOT/control"
  rsync -a "$REMOTE_HOST:$RUN_ROOT/results/" "$LOCAL_RUN_ROOT/results/"
  rsync -a "$REMOTE_HOST:$RUN_ROOT/logs/" "$LOCAL_RUN_ROOT/logs/"
  rsync -a "$REMOTE_HOST:$RUN_ROOT/status/" "$LOCAL_RUN_ROOT/status/"
  rsync -a "$REMOTE_HOST:$RUN_ROOT/control/" "$LOCAL_RUN_ROOT/control/"
  echo "Collected evidence: $LOCAL_RUN_ROOT"
}

score_candidate_records() {
  local truth="$REPO/$TRUTH_REL"
  local backend
  local stage
  local candidate
  local record
  local output
  local status
  local outcome=0
  local -a candidates
  if (($# == 0)); then
    candidates=(internvideo3:long20 qwen3-vl:fine)
  else
    candidates=("$@")
  fi
  [[ -f $truth ]] || die "human truth is missing: $truth"
  mkdir -p "$LOCAL_RUN_ROOT/scores"
  for candidate in "${candidates[@]}"; do
    backend=${candidate%%:*}
    stage=${candidate#*:}
    record="$LOCAL_RUN_ROOT/results/$backend-$stage.json"
    output="$LOCAL_RUN_ROOT/scores/$backend-$stage-score.json"
    [[ -f $record ]] || die "collected candidate record is missing: $record"
    if PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" \
      uv run --no-project --with numpy==2.4.4 \
      python -m annotator.vlm_scene_benchmark.gate_cli \
      "$backend=$record"; then
      status=0
    else
      status=$?
    fi
    if ((status == 4)); then
      echo "Candidate record failed its local provenance gate: $record" >&2
      outcome=4
      continue
    elif ((status != 0)); then
      echo "Candidate record provenance gate could not run: $record; status $status." >&2
      return "$status"
    fi
    if [[ -f $output ]]; then
      echo "Reusing score: $output"
      continue
    fi
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" \
      uv run --no-project --with numpy==2.4.4 \
      python -m annotator.vlm_scene_benchmark.score_cli \
      "$record" "$truth" --out "$output" || return
  done
  return "$outcome"
}

show_status() {
  local key
  local -a sessions=()
  for key in setup internvideo3-smoke qwen3-vl-smoke internvideo3-long20 qwen3-vl-fine; do
    sessions+=("$(session_name "$key")")
  done
  echo "run tag: $RUN_TAG"
  echo "remote root: $REMOTE_HOST:$RUN_ROOT"
  echo "local evidence: $LOCAL_RUN_ROOT"
  ssh "$REMOTE_HOST" bash -s -- "$RUN_ROOT" "${sessions[@]}" <<'REMOTE'
set -euo pipefail
run_root=$1
shift
if [[ -d "$run_root/status" ]]; then
  find "$run_root/status" -maxdepth 1 -type f -printf '%f: ' -exec cat {} \;
else
  echo "No remote status directory."
fi
for session in "$@"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "$session"
  fi
done
REMOTE
}

run_all() {
  check_local_prerequisites
  check_remote_prerequisites
  prepare_inputs
  resolve_run_identity
  stage_runtime_snapshot
  run_setup_task
  run_model_pair smoke
  collect_evidence
  run_candidate_models
  score_candidate_records "${PASSED_CANDIDATES[@]}"
}

main() {
  parse_arguments "$@"
  if ((PLAN_ONLY)); then
    print_plan
    return
  fi
  configure_remote_shell
  case "$COMMAND" in
    check)
      check_local_prerequisites
      check_remote_prerequisites
      ;;
    prepare)
      require_command uv
      require_command md5sum
      require_command sha256sum
      prepare_inputs
      ;;
    all)
      run_all
      ;;
    stage)
      check_local_prerequisites
      check_remote_prerequisites
      validate_prepared_inputs
      resolve_run_identity
      stage_runtime_snapshot
      ;;
    setup)
      resolve_run_identity
      run_setup_task
      ;;
    smoke)
      resolve_run_identity
      run_model_pair smoke
      collect_evidence
      gate_model_pair smoke
      ;;
    full)
      resolve_run_identity
      run_candidate_models
      ;;
    qwen-fine)
      require_command uv
      resolve_run_identity
      run_qwen_fine_only
      ;;
    collect)
      resolve_run_identity
      collect_evidence
      ;;
    score)
      require_command uv
      resolve_run_identity
      score_candidate_records
      ;;
    status)
      resolve_run_identity
      show_status
      ;;
  esac
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
