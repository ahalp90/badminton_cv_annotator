"""Contract tests for the one-command Carmack benchmark wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "vlm_scene_benchmark"
ORCHESTRATOR = SCRIPTS / "run_carmack.sh"
REMOTE_TASK = SCRIPTS / "remote_task.sh"
SETUP_REMOTE = SCRIPTS / "setup_remote.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ORCHESTRATOR), *arguments],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


def test_plan_is_local_and_names_the_exact_models(tmp_path: Path) -> None:
    result = _run(
        "all",
        "--plan",
        "--source",
        str(tmp_path / "missing-source.mp4"),
        "--artifacts",
        str(tmp_path / "missing-artifacts"),
        "--local-results",
        str(tmp_path / "results"),
    )

    assert result.returncode == 0
    assert "yanziang/InternVideo3-8B-Instruct@c4602918b65225650d152db2850fe34e01d21fcd" in result.stdout
    assert "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8@d9748a51ae66354c4dad665aab2c71f26cf2c8cd" in result.stdout
    assert "truth-free gate" in result.stdout
    assert not (tmp_path / "results").exists()


def test_qwen_fine_plan_is_a_supported_local_command(tmp_path: Path) -> None:
    result = _run(
        "qwen-fine",
        "--plan",
        "--source",
        str(tmp_path / "missing-source.mp4"),
        "--ssh-command",
        str(tmp_path / "missing-ssh-wrapper"),
    )

    assert result.returncode == 0
    assert "command: qwen-fine" in result.stdout
    assert f"SSH command: {tmp_path / 'missing-ssh-wrapper'}" in result.stdout


def test_invalid_command_stops_before_any_external_work() -> None:
    result = _run("not-a-command", "--plan")

    assert result.returncode == 1
    assert "unknown command" in result.stderr


def test_remote_snapshot_and_tasks_keep_truth_out_of_inference() -> None:
    orchestrator = ORCHESTRATOR.read_text(encoding="utf-8")
    remote_task = (SCRIPTS / "remote_task.sh").read_text(encoding="utf-8")

    assert "runtime_file_list | awk" in orchestrator
    assert '--filter="merge $filter_file"' in orchestrator
    assert "git -C \"$REPO\" ls-files --cached --others --exclude-standard" in orchestrator
    assert "--include='/docs/'" not in orchestrator
    assert "--include='*.csv'" not in orchestrator
    assert "--delete --delete-excluded" in orchestrator
    assert "remote_gate internvideo3 smoke" in orchestrator
    assert "remote_gate qwen3-vl smoke" in orchestrator
    assert 'candidates+=("internvideo3:long20")' in orchestrator
    assert 'candidates+=("qwen3-vl:fine")' in orchestrator
    assert 'run_model_task "$backend" "$stage"' in orchestrator
    assert 'score_candidate_records "${PASSED_CANDIDATES[@]}"' in orchestrator
    assert "assert_gpu_idle" in remote_task
    assert "swap_space=0" not in remote_task
    assert "internvideo3-sset15-$STAGE-$RUN_ID_SUFFIX" in remote_task
    assert "qwen3-vl-sset15-$STAGE-$RUN_ID_SUFFIX" in remote_task
    assert "sset_15_f18419_f48419_512x288_manifest.json" in remote_task
    assert "sset_15_f20695_f20945_512x288_manifest.json" in remote_task
    assert "QWEN_MAX_MODEL_LEN=8192" in remote_task
    assert "QWEN_MAX_MODEL_LEN=16384" in remote_task
    assert '--max-model-len "$QWEN_MAX_MODEL_LEN"' in remote_task


def test_internvideo3_runtime_installs_and_loads_pinned_ffmpeg() -> None:
    setup = SETUP_REMOTE.read_text(encoding="utf-8")
    remote_task = REMOTE_TASK.read_text(encoding="utf-8")
    conda_requirements = (
        SCRIPTS / "requirements-internvideo3-conda.txt"
    ).read_text(encoding="utf-8")

    assert "ffmpeg=7.1.1=gpl_hbbdf940_911" in conda_requirements
    assert "create_intern_ffmpeg_environment" in setup
    assert "import torchcodec" in setup
    assert "internvideo3-ffmpeg-explicit.txt" in setup
    assert "INTERN_FFMPEG_ENVIRONMENT" in setup
    assert '--env LD_LIBRARY_PATH="$INTERN_FFMPEG_ENVIRONMENT/lib"' in remote_task
    assert "--env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in remote_task


def test_runtime_images_use_the_verified_sutherland_hashes() -> None:
    orchestrator = ORCHESTRATOR.read_text(encoding="utf-8")
    setup = SETUP_REMOTE.read_text(encoding="utf-8")
    hashes = (
        "fd1c42ea24386dde021f12c0fe9458f0d4f5f43ea97af2ad19c2b3ea9925c76a",
        "1cf06bf5a8a7bd5a2b2c469f0e72ac150f0781c126b593c7fcd9d7df4eb34d37",
    )

    for digest in hashes:
        assert digest in orchestrator
        assert digest in setup


def test_custom_ssh_command_drives_ssh_and_rsync(tmp_path: Path) -> None:
    wrapper = tmp_path / "nested-ssh"
    capture = tmp_path / "capture.txt"
    _write_executable(
        wrapper,
        "#!/bin/sh\nprintf '%s\\n' \"$*\" > \"$SSH_CAPTURE\"\n",
    )
    command = """
source "$1"
SSH_COMMAND=$2
configure_remote_shell
ssh sutherland hostname
printf 'rsync-rsh:%s\n' "$RSYNC_RSH"
"""

    result = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR), str(wrapper)],
        cwd=ROOT,
        env=os.environ | {"SSH_CAPTURE": str(capture)},
        capture_output=True,
        check=True,
        text=True,
    )

    assert capture.read_text(encoding="utf-8") == "sutherland hostname\n"
    assert f"rsync-rsh:{wrapper}" in result.stdout


def test_session_names_use_the_full_host_run_root_and_key() -> None:
    command = """
source "$1"
REMOTE_HOST=$2
REMOTE_ROOT=$3
RUN_TAG=$4
RUN_ROOT="$REMOTE_ROOT/runs/$RUN_TAG"
session_name internvideo3-smoke
"""

    def name(tag: str, remote_root: str = "/scratch/test-one") -> str:
        result = subprocess.run(
            ["bash", "-c", command, "bash", str(ORCHESTRATOR), "carmack", remote_root, tag],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()

    assert name("issue38-abcdefghijklmnop-one") != name("issue38-abcdefghijklmnop-two")
    assert name("issue38-same", "/scratch/test-one") != name(
        "issue38-same", "/scratch/test-two"
    )


def test_partial_evidence_retry_command_preserves_all_path_overrides(tmp_path: Path) -> None:
    command = """
source "$1"
SOURCE=$2
ARTIFACTS=$3
LOCAL_RESULTS=$4
REMOTE_HOST=$5
REMOTE_ROOT=$6
POLL_SECONDS=7
RUN_TAG=issue38-existing
SSH_COMMAND=$7
print_new_tag_retry
"""
    source = tmp_path / "source video.mp4"
    artifacts = tmp_path / "prepared artifacts"
    local_results = tmp_path / "local results"
    ssh_command = tmp_path / "nested-ssh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(ORCHESTRATOR),
            str(source),
            str(artifacts),
            str(local_results),
            "carmack",
            "/scratch/test-root",
            str(ssh_command),
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert f"{ORCHESTRATOR} all" in result.stderr
    assert "--source" in result.stderr
    assert "source\\ video.mp4" in result.stderr
    assert "--artifacts" in result.stderr
    assert "--local-results" in result.stderr
    assert "--remote-host carmack" in result.stderr
    assert "--remote-root /scratch/test-root" in result.stderr
    assert "--poll-seconds 7" in result.stderr
    assert "--run-tag issue38-existing-retry-" in result.stderr
    assert "--ssh-command" in result.stderr
    assert "nested-ssh" in result.stderr


def test_complete_record_recovers_missing_success_or_failure_status() -> None:
    command = """
source "$1"
REMOTE_HOST=carmack
RUN_ROOT=/scratch/test/run
remote_gate() { return "$RECOVER_GATE_STATUS"; }
ssh() { printf 'write:%s\n' "$*"; cat >/dev/null; }
if recover_completed_benchmark_status internvideo3 smoke "$RUN_ROOT/status/recovered"; then
  recovered=0
else
  recovered=$?
fi
printf 'status:%s\n' "$recovered"
"""
    for gate_status in (0, 4):
        result = subprocess.run(
            ["bash", "-c", command, "bash", str(ORCHESTRATOR)],
            cwd=ROOT,
            env=os.environ | {"RECOVER_GATE_STATUS": str(gate_status)},
            capture_output=True,
            check=False,
            text=True,
        )

        assert result.returncode == 0
        assert f"status:{gate_status}" in result.stdout
        assert f"/status/recovered {gate_status}" in result.stdout

    orchestrator = ORCHESTRATOR.read_text(encoding="utf-8")
    attempt_function = orchestrator.split("remote_has_attempt_evidence()", 1)[1].split(
        "recover_completed_benchmark_status()", 1
    )[0]
    assert ".attempt-*.txt" in attempt_function
    assert "$key.json" not in attempt_function
    wait_function = orchestrator.split("wait_for_task()", 1)[1].split("run_setup_task()", 1)[0]
    assert wait_function.index('remote_result_exists "$key"') < wait_function.index(
        'tmux has-session -t \'$session\''
    )


def test_complete_record_with_infrastructure_error_does_not_print_partial_retry() -> None:
    command = """
source "$1"
REMOTE_HOST=carmack
REMOTE_ROOT=/scratch/test
RUN_TAG=issue38-test
RUN_ROOT="$REMOTE_ROOT/runs/$RUN_TAG"
POLL_SECONDS=1
remote_status_value() { return 0; }
remote_result_exists() { return 0; }
recover_completed_benchmark_status() { return 70; }
remote_has_attempt_evidence() { return 0; }
print_new_tag_retry() { printf 'NEW_TAG_RETRY\n'; }
ssh() { case "$*" in *'tmux has-session'*) return 1 ;; *) return 0 ;; esac; }
if wait_for_task benchmark internvideo3-smoke internvideo3 smoke; then
  status=0
else
  status=$?
fi
printf 'status:%s\n' "$status"
"""
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR)],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "status:70" in result.stdout
    assert "NEW_TAG_RETRY" not in result.stdout


def test_setup_status_requires_runtime_environment_before_reuse() -> None:
    command = """
source "$1"
REMOTE_HOST=carmack
RUN_ROOT=/scratch/test/run
ssh() { return "$RUNTIME_ENV_EXISTS"; }
for status in 0 4; do
  if setup_is_reusable "$status"; then result=0; else result=$?; fi
  printf '%s:%s\n' "$status" "$result"
done
"""
    missing = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR)],
        cwd=ROOT,
        env=os.environ | {"RUNTIME_ENV_EXISTS": "1"},
        capture_output=True,
        check=True,
        text=True,
    )
    present = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR)],
        cwd=ROOT,
        env=os.environ | {"RUNTIME_ENV_EXISTS": "0"},
        capture_output=True,
        check=True,
        text=True,
    )

    assert "0:1" in missing.stdout
    assert "4:1" in missing.stdout
    assert "0:0" in present.stdout
    assert "4:1" in present.stdout


def test_setup_reuse_accepts_container_python_symlink(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    runtime = tmp_path / "control" / "runtime.env"
    intern = remote_root / "intern"
    ffmpeg = remote_root / "ffmpeg"
    qwen = remote_root / "qwen"
    (intern / "bin").mkdir(parents=True)
    (intern / "bin" / "python").symlink_to("/opt/conda/bin/python")
    (intern / ".issue38-ready").write_text("ready\n", encoding="utf-8")
    (ffmpeg / "lib").mkdir(parents=True)
    (ffmpeg / ".issue38-ready").write_text("ready\n", encoding="utf-8")
    (qwen / "bin").mkdir(parents=True)
    _write_executable(qwen / "bin" / "python", "#!/bin/sh\nexit 0\n")
    (qwen / ".issue38-ready").write_text("ready\n", encoding="utf-8")
    runtime.parent.mkdir()
    runtime.write_text(
        f"INTERN_ENVIRONMENT={intern}\n"
        f"INTERN_FFMPEG_ENVIRONMENT={ffmpeg}\n"
        f"QWEN_ENVIRONMENT={qwen}\n",
        encoding="utf-8",
    )
    command = """
source "$1"
REMOTE_HOST=sutherland
RUN_ROOT=$2
ssh() { shift; "$@"; }
setup_is_reusable 0
"""

    result = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR), str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_setup_ready_with_incomplete_qwen_runtime_is_not_reused(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    run_root = remote_root / "runs" / "test"
    setup_script = run_root / "repo" / "scripts" / "vlm_scene_benchmark" / "setup_remote.sh"
    marker = tmp_path / "setup-ran"
    setup_script.parent.mkdir(parents=True)
    _write_executable(setup_script, f"#!/bin/sh\n: > {marker}\n")
    (run_root / "control").mkdir()
    qwen_environment = remote_root / "envs" / "qwen"
    (qwen_environment / "bin").mkdir(parents=True)
    _write_executable(qwen_environment / "bin" / "python", "#!/bin/sh\nexit 0\n")
    (qwen_environment / ".issue38-ready").write_text("ready\n", encoding="utf-8")
    (run_root / "control" / "staged.sha256").write_text("valid\n", encoding="utf-8")
    (run_root / "control" / "runtime.env").write_text(
        "INTERN_ENVIRONMENT=/missing/intern\n"
        "INTERN_FFMPEG_ENVIRONMENT=/missing/ffmpeg\n"
        f"QWEN_ENVIRONMENT={qwen_environment}\n",
        encoding="utf-8",
    )
    (run_root / "status").mkdir()
    (run_root / "status" / "setup.ok").write_text("0\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "sha256sum", "#!/bin/sh\nexit 0\n")

    result = subprocess.run(
        [str(REMOTE_TASK), "setup", str(remote_root), str(run_root)],
        cwd=ROOT,
        env=os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"},
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert marker.exists()
    assert (run_root / "status" / "setup.ok.incomplete").read_text(encoding="utf-8") == "0\n"


def test_failed_qwen_smoke_does_not_block_internvideo3_long_run() -> None:
    command = """
source "$1"
remote_gate() {
  case "$1-$2" in
    internvideo3-smoke|internvideo3-long20) return 0 ;;
    *) return 4 ;;
  esac
}
run_model_task() { printf 'run:%s:%s\n' "$1" "$2"; }
collect_evidence() { printf 'collect\n'; }
run_candidate_models
printf 'passed:%s\n' "${PASSED_CANDIDATES[*]}"
"""

    result = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR)],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "run:internvideo3:long20" in result.stdout
    assert "run:qwen3-vl:fine" not in result.stdout
    assert "passed:internvideo3:long20" in result.stdout
    assert "Qwen3-VL failed its smoke gate" in result.stderr


def test_passing_smokes_route_models_to_distinct_candidate_stages() -> None:
    command = """
source "$1"
remote_gate() { return 0; }
run_model_task() { printf 'run:%s:%s\n' "$1" "$2"; }
collect_evidence() { printf 'collect\n'; }
run_candidate_models
printf 'passed:%s\n' "${PASSED_CANDIDATES[*]}"
"""
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR)],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "run:internvideo3:long20" in result.stdout
    assert "run:qwen3-vl:fine" in result.stdout
    assert "passed:internvideo3:long20 qwen3-vl:fine" in result.stdout


def test_qwen_fine_only_does_not_repeat_internvideo3_long_run() -> None:
    command = """
source "$1"
run_setup_task() { printf 'setup\n'; }
remote_gate() { printf 'gate:%s:%s\n' "$1" "$2"; }
run_model_task() { printf 'run:%s:%s\n' "$1" "$2"; }
collect_evidence() { printf 'collect\n'; }
score_candidate_records() { printf 'score:%s\n' "$*"; }
run_qwen_fine_only
"""

    result = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR)],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.splitlines()[0] == "setup"
    assert "run:qwen3-vl:smoke" in result.stdout
    assert "gate:qwen3-vl:smoke" in result.stdout
    assert "run:qwen3-vl:fine" in result.stdout
    assert "gate:qwen3-vl:fine" in result.stdout
    assert "score:qwen3-vl:fine" in result.stdout
    assert "internvideo3" not in result.stdout


def test_candidate_routing_propagates_smoke_gate_infrastructure_failure() -> None:
    command = """
source "$1"
remote_gate() { return 255; }
run_model_task() { printf 'unexpected-run\n'; }
collect_evidence() { printf 'unexpected-collect\n'; }
if run_candidate_models; then status=0; else status=$?; fi
printf 'status:%s\n' "$status"
"""
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR)],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "status:255" in result.stdout
    assert "unexpected-" not in result.stdout
    assert "smoke gate could not run; status 255" in result.stderr
    assert "failed its smoke gate" not in result.stderr


def test_model_task_returns_wait_status() -> None:
    command = """
source "$1"
wait_for_task() { return "$TASK_STATUS"; }
if run_model_task internvideo3 smoke; then status=0; else status=$?; fi
printf 'returned:%s\n' "$status"
"""
    for task_status in (4, 70, 75):
        result = subprocess.run(
            ["bash", "-c", command, "bash", str(ORCHESTRATOR)],
            cwd=ROOT,
            env=os.environ | {"TASK_STATUS": str(task_status)},
            capture_output=True,
            check=True,
            text=True,
        )

        assert f"process status: {task_status}" in result.stdout
        assert f"returned:{task_status}" in result.stdout


def test_model_pair_allows_model_failure_but_propagates_infrastructure_failure() -> None:
    command = """
source "$1"
run_model_task() { printf 'run:%s:%s\n' "$1" "$2"; return "$TASK_STATUS"; }
if run_model_pair smoke; then status=0; else status=$?; fi
printf 'returned:%s\n' "$status"
"""
    expected = {4: 0, 70: 70, 75: 75}
    for task_status, expected_status in expected.items():
        result = subprocess.run(
            ["bash", "-c", command, "bash", str(ORCHESTRATOR)],
            cwd=ROOT,
            env=os.environ | {"TASK_STATUS": str(task_status)},
            capture_output=True,
            check=True,
            text=True,
        )

        assert f"returned:{expected_status}" in result.stdout
        expected_runs = 2 if task_status == 4 else 1
        assert result.stdout.count("run:") == expected_runs


def test_candidate_routing_propagates_final_gate_infrastructure_failure() -> None:
    command = """
source "$1"
remote_gate() {
  case "$1-$2" in
    internvideo3-long20) return 70 ;;
    *) return 0 ;;
  esac
}
run_model_task() { printf 'run:%s:%s\n' "$1" "$2"; }
collect_evidence() { printf 'collect\n'; }
if run_candidate_models; then status=0; else status=$?; fi
printf 'status:%s\n' "$status"
"""
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR)],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "status:70" in result.stdout
    assert "internvideo3 long20 gate could not run; status 70" in result.stderr
    assert "failed its long20 deployment gate" not in result.stderr


def test_score_command_locally_gates_record_before_reusing_score(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "results").mkdir(parents=True)
    (run_root / "scores").mkdir()
    (run_root / "results" / "internvideo3-long20.json").write_text(
        "untrusted\n", encoding="utf-8"
    )
    (run_root / "scores" / "internvideo3-long20-score.json").write_text(
        "stale\n", encoding="utf-8"
    )
    command = """
source "$1"
LOCAL_RUN_ROOT=$2
uv() {
  printf 'uv:%s\n' "$*"
  case "$*" in
    *vlm_scene_benchmark.gate_cli*) return 4 ;;
    *vlm_scene_benchmark.score_cli*) printf 'unexpected-score\n'; return 0 ;;
  esac
}
if score_candidate_records internvideo3:long20; then status=0; else status=$?; fi
printf 'status:%s\n' "$status"
"""
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR), str(run_root)],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "status:4" in result.stdout
    assert "vlm_scene_benchmark.gate_cli" in result.stdout
    assert "unexpected-score" not in result.stdout
    assert "Reusing score" not in result.stdout
    assert "failed its local provenance gate" in result.stderr


def test_score_command_continues_after_one_semantic_gate_failure(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "results").mkdir(parents=True)
    for name in ("internvideo3-long20.json", "qwen3-vl-fine.json"):
        (run_root / "results" / name).write_text("record\n", encoding="utf-8")
    command = """
source "$1"
LOCAL_RUN_ROOT=$2
uv() {
  printf 'uv:%s\n' "$*"
  case "$*" in
    *gate_cli*internvideo3-long20.json*) return 4 ;;
    *gate_cli*qwen3-vl-fine.json*) return 0 ;;
    *score_cli*qwen3-vl-fine.json*) printf 'qwen-scored\n'; return 0 ;;
    *) return 70 ;;
  esac
}
if score_candidate_records internvideo3:long20 qwen3-vl:fine; then
  status=0
else
  status=$?
fi
printf 'status:%s\n' "$status"
"""
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR), str(run_root)],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "qwen-scored" in result.stdout
    assert "status:4" in result.stdout


def test_score_command_propagates_gate_infrastructure_failure(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "results").mkdir(parents=True)
    (run_root / "results" / "internvideo3-long20.json").write_text(
        "record\n", encoding="utf-8"
    )
    command = """
source "$1"
LOCAL_RUN_ROOT=$2
uv() { return 70; }
if score_candidate_records internvideo3:long20; then status=0; else status=$?; fi
printf 'status:%s\n' "$status"
"""
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR), str(run_root)],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "status:70" in result.stdout
    assert "provenance gate could not run" in result.stderr


def test_benchmark_stops_before_apptainer_when_snapshot_check_fails(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    run_root = remote_root / "runs" / "test"
    (run_root / "control").mkdir(parents=True)
    (run_root / "control" / "staged.sha256").write_text("invalid\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "apptainer-called"
    _write_executable(fake_bin / "sha256sum", "#!/bin/sh\nexit 1\n")
    _write_executable(
        fake_bin / "apptainer",
        "#!/bin/sh\n: > \"$APPTAINER_MARKER\"\nexit 0\n",
    )
    environment = os.environ | {
        "APPTAINER_MARKER": str(marker),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    result = subprocess.run(
        [str(REMOTE_TASK), "benchmark", str(remote_root), str(run_root), "internvideo3", "smoke"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert not marker.exists()
    assert not (run_root / "status" / "internvideo3-smoke.status").exists()


def test_setup_ready_write_failure_cannot_publish_success(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    run_root = remote_root / "runs" / "test"
    setup_script = run_root / "repo" / "scripts" / "vlm_scene_benchmark" / "setup_remote.sh"
    setup_script.parent.mkdir(parents=True)
    _write_executable(setup_script, "#!/bin/sh\nexit 0\n")
    (run_root / "control").mkdir()
    (run_root / "control" / "staged.sha256").write_text("valid\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "sha256sum", "#!/bin/sh\nexit 0\n")
    _write_executable(
        fake_bin / "mv",
        "#!/bin/sh\n"
        "case \"$*\" in */status/setup.ok) exit 1 ;; esac\n"
        "exec /bin/mv \"$@\"\n",
    )
    environment = os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"}

    result = subprocess.run(
        [str(REMOTE_TASK), "setup", str(remote_root), str(run_root)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 71
    assert not (run_root / "status" / "setup.ok").exists()
    assert not (run_root / "status" / "setup.last-exit").exists()


def test_nonzero_setup_ready_status_is_replaced_after_success(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    run_root = remote_root / "runs" / "test"
    setup_script = run_root / "repo" / "scripts" / "vlm_scene_benchmark" / "setup_remote.sh"
    setup_script.parent.mkdir(parents=True)
    _write_executable(setup_script, "#!/bin/sh\nexit 0\n")
    (run_root / "control").mkdir()
    (run_root / "control" / "staged.sha256").write_text("valid\n", encoding="utf-8")
    (run_root / "control" / "runtime.env").write_text("existing\n", encoding="utf-8")
    (run_root / "status").mkdir()
    (run_root / "status" / "setup.ok").write_text("4\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "sha256sum", "#!/bin/sh\nexit 0\n")
    environment = os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"}

    result = subprocess.run(
        [str(REMOTE_TASK), "setup", str(remote_root), str(run_root)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert (run_root / "status" / "setup.ok").read_text(encoding="utf-8") == "0\n"
    assert (run_root / "status" / "setup.ok.incomplete").read_text(encoding="utf-8") == "4\n"


def test_setup_serializes_shared_image_and_environment_mutations() -> None:
    setup = (SCRIPTS / "setup_remote.sh").read_text(encoding="utf-8")
    orchestrator = ORCHESTRATOR.read_text(encoding="utf-8")

    assert 'exec 9>"$REMOTE_ROOT/.setup.lock"' in setup
    assert "flock 9" in setup
    assert "apptainer bash flock nvidia-smi" in orchestrator


def test_status_queries_only_the_current_runs_exact_session_names() -> None:
    command = """
source "$1"
REMOTE_HOST=carmack
REMOTE_ROOT=/scratch/test-root
RUN_TAG=issue38-current
RUN_ROOT="$REMOTE_ROOT/runs/$RUN_TAG"
LOCAL_RUN_ROOT=/tmp/local-run
ssh() { printf 'ssh-arg:%s\n' "$@"; cat >/dev/null; }
show_status
"""
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR)],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    session_arguments = [
        line.removeprefix("ssh-arg:")
        for line in result.stdout.splitlines()
        if line.startswith("ssh-arg:i38-")
    ]

    assert len(session_arguments) == 5
    assert len(set(session_arguments)) == 5
    assert "grep '^i38-'" not in ORCHESTRATOR.read_text(encoding="utf-8")


def test_shell_scripts_parse() -> None:
    result = subprocess.run(
        ["bash", "-n", *(str(path) for path in SCRIPTS.glob("*.sh"))],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
