"""Production-boundary tests for dataset-builder RTMLib pose sharding."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import cast

import numpy as np
import pytest

from annotator.video_metadata import VideoMetadata
from dataset_builder import _pose_process as pose_process_module
from dataset_builder import _vision_plans, cli, pose_sharding, vision
from dataset_builder._pose_process import POSE_CHILD_STEM, run_isolated_pose_process
from dataset_builder._runtime_support import RuntimeSupport
from dataset_builder.models import InterpreterIdentity, StageOutcome


def _metadata(tmp_path: Path, *, frame_count: int = 4) -> VideoMetadata:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture source")
    return VideoMetadata(source.resolve(), Fraction(30), frame_count, 1920, 1080)


def _pose_arrays(frame_count: int, n_slots: int = 2) -> vision.PoseArrays:
    kps = np.full((frame_count, n_slots, 17, 2), np.nan, dtype=np.float32)
    bboxes = np.full((frame_count, n_slots, 4), np.nan, dtype=np.float32)
    scores = np.full((frame_count, n_slots), np.nan, dtype=np.float32)
    kp_scores = np.full((frame_count, n_slots, 17), np.nan, dtype=np.float32)
    ndet = np.ones(frame_count, dtype=np.int8)
    kps[:, 0] = 1.0
    bboxes[:, 0] = 2.0
    scores[:, 0] = 0.9
    kp_scores[:, 0] = 0.8
    return vision.PoseArrays(kps, bboxes, scores, kp_scores, ndet)


def _write_raw_pose(output_dir: Path, arrays: vision.PoseArrays) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for field_name in ("kps", "bboxes", "scores", "kp_scores", "ndet"):
        np.save(
            output_dir / f"{POSE_CHILD_STEM}_raw_{field_name}.npy",
            getattr(arrays, field_name),
            allow_pickle=False,
        )


def _interpreter(tmp_path: Path) -> Path:
    path = tmp_path / "rtmlib-python"
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_blocking_pose_child(tmp_path: Path) -> Path:
    child_script = tmp_path / "blocking_pose_child.py"
    child_script.write_text(
        """from pathlib import Path
import os
import subprocess
import sys
import time

descendant = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
Path(sys.argv[1]).write_text(f"{os.getpid()} {descendant.pid}", encoding="utf-8")
time.sleep(300)
""",
        encoding="utf-8",
    )
    return child_script


def test_pose_shards_configuration_must_be_positive(tmp_path: Path) -> None:
    tracked = cli.REPO_ROOT / "configs/dataset_builder/trial.toml"
    malformed = tracked.read_text(encoding="utf-8").replace("pose_shards = 8", "pose_shards = 0")
    path = tmp_path / "trial.toml"
    path.write_text(malformed, encoding="utf-8")

    with pytest.raises(ValueError, match="vision.pose_shards"):
        cli.load_builder_config(path)


def test_pose_child_module_does_not_require_coordinator_models(tmp_path: Path) -> None:
    blocker = tmp_path / "frozendict.py"
    blocker.write_text(
        'raise RuntimeError("coordinator-only frozendict was imported")\n',
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        os.fspath(tmp_path),
        os.fspath(cli.REPO_ROOT / "src"),
        os.fspath(cli.REPO_ROOT / "src/bst_x"),
    ))

    completed = subprocess.run(
        [sys.executable, "-c", "import dataset_builder.pose_sharding"],
        cwd=cli.REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_sharded_pose_uses_configured_interpreter_and_canonical_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata(tmp_path)
    expected = _pose_arrays(metadata.frame_count)
    interpreter = _interpreter(tmp_path)
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        raw_root = Path(command[command.index("--output-root") + 1])
        run_id = command[command.index("--run-id") + 1]
        _write_raw_pose(raw_root / f"publish_{run_id}", expected)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(pose_sharding, "run_isolated_pose_process", fake_run)
    output_dir = tmp_path / "pose"
    extraction = pose_sharding.extract_sharded_rtmlib_pose_stage(
        metadata=metadata,
        output_dir=output_dir,
        interpreter=interpreter,
        shards=4,
        device="cpu",
        n_max=2,
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[0] == str(interpreter.resolve())
    assert command[1:4] == [
        "-m", "dataset_builder.pose_sharding", "_extract-sharded-rtmlib-pose",
    ]
    assert command[command.index("--shards") + 1] == "4"
    assert command[command.index("--expected-frame-count") + 1] == "4"
    assert len(command[command.index("--run-id") + 1]) == 32
    assert command[command.index("--decode-mode") + 1] == "seek"
    assert extraction.command == tuple(command)
    loaded = vision.load_pose_arrays(output_dir, metadata.frame_count)
    for field_name in ("kps", "bboxes", "scores", "kp_scores", "ndet"):
        np.testing.assert_array_equal(
            getattr(loaded, field_name),
            getattr(expected, field_name),
        )
    assert not list(output_dir.glob(".rtmlib-sharded-*"))


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [("subprocess", RuntimeError), ("padding", ValueError)],
)
def test_sharded_pose_failure_publishes_no_final_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    error_type: type[Exception],
) -> None:
    metadata = _metadata(tmp_path)
    interpreter = _interpreter(tmp_path)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if failure == "subprocess":
            return subprocess.CompletedProcess(command, 17, stdout="", stderr="worker failed")
        arrays = _pose_arrays(metadata.frame_count)
        arrays.kps[:, 1] = 0.0
        raw_root = Path(command[command.index("--output-root") + 1])
        run_id = command[command.index("--run-id") + 1]
        _write_raw_pose(raw_root / f"publish_{run_id}", arrays)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(pose_sharding, "run_isolated_pose_process", fake_run)
    output_dir = tmp_path / "pose"
    with pytest.raises(error_type):
        pose_sharding.extract_sharded_rtmlib_pose_stage(
            metadata=metadata,
            output_dir=output_dir,
            interpreter=interpreter,
            shards=4,
            n_max=2,
        )

    assert not any(path.exists() for path in vision.pose_artifact_paths(output_dir).as_mapping().values())
    assert not list(output_dir.glob(".rtmlib-sharded-*"))


def test_pose_child_passes_canonical_contract_to_sharding_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shared.video_sharding import run_sharded

    metadata = _metadata(tmp_path, frame_count=11)
    output_root = tmp_path / "raw"
    run_id = "fixture_run"
    observed: dict[str, object] = {}

    def fake_extract_sharded(**kwargs: object) -> Path:
        observed.update(kwargs)
        published = output_root / f"publish_{run_id}"
        published.mkdir(parents=True)
        return published

    monkeypatch.setattr(run_sharded, "extract_sharded", fake_extract_sharded)

    result = pose_sharding._extract_pose_child(
        video_path=metadata.source_path,
        output_root=output_root,
        device="cuda",
        n_max=10,
        shards=8,
        expected_frame_count=metadata.frame_count,
        run_id=run_id,
        decode_mode="seek",
    )

    assert result == 0
    assert observed["video_path"] == metadata.source_path
    assert observed["n_shards"] == 8
    assert observed["n_max"] == 10
    assert observed["extractor_spec"] == "cuda"
    assert observed["expected_frame_count"] == 11
    assert observed["run_id"] == run_id


def test_sigterm_cleans_pose_process_group_and_temporary_tree(tmp_path: Path) -> None:
    pose_root = tmp_path / "pose"
    pose_root.mkdir()
    pid_file = tmp_path / "pose_pids.txt"
    child_script = _write_blocking_pose_child(tmp_path)
    marker_ready = threading.Event()
    coordinator_pid = os.getpid()
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate_after_child_starts() -> None:
        deadline = time.monotonic() + 10.0
        while not pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        if pid_file.is_file():
            marker_ready.set()
        os.kill(coordinator_pid, signal.SIGTERM)

    terminator = threading.Thread(target=terminate_after_child_starts, daemon=True)
    terminator.start()
    try:
        with pytest.raises(SystemExit) as termination:
            with tempfile.TemporaryDirectory(prefix=".rtmlib-sharded-", dir=pose_root):
                run_isolated_pose_process(
                    [sys.executable, os.fspath(child_script), os.fspath(pid_file)],
                    cwd=tmp_path,
                    env=os.environ.copy(),
                )
    finally:
        terminator.join(timeout=12.0)

    assert marker_ready.is_set()
    assert termination.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    assert not list(pose_root.glob(".rtmlib-sharded-*"))
    child_pid, descendant_pid = (int(value) for value in pid_file.read_text().split())
    exited = _wait_for_processes_to_exit((child_pid, descendant_pid))
    if not exited:
        try:
            os.killpg(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    assert exited


def test_sigterm_between_popen_return_and_assignment_cleans_pose_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pose_root = tmp_path / "pose"
    pose_root.mkdir()
    pid_file = tmp_path / "race_pose_pids.txt"
    child_script = _write_blocking_pose_child(tmp_path)
    real_popen = subprocess.Popen
    spawned: list[subprocess.Popen[str]] = []
    marker_ready = threading.Event()
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interposed_popen(
        command: list[str],
        *,
        stdout: int,
        stderr: int,
        text: bool,
        cwd: Path,
        env: dict[str, str],
        start_new_session: bool,
    ) -> subprocess.Popen[str]:
        child = cast(
            "subprocess.Popen[str]",
            real_popen(
                command,
                stdout=stdout,
                stderr=stderr,
                text=text,
                cwd=cwd,
                env=env,
                start_new_session=start_new_session,
            ),
        )
        spawned.append(child)
        deadline = time.monotonic() + 10.0
        while not pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        if pid_file.is_file():
            marker_ready.set()
        os.kill(os.getpid(), signal.SIGTERM)
        return child

    monkeypatch.setattr(pose_process_module.subprocess, "Popen", interposed_popen)
    with pytest.raises(SystemExit) as termination:
        with tempfile.TemporaryDirectory(prefix=".rtmlib-sharded-", dir=pose_root):
            run_isolated_pose_process(
                [sys.executable, os.fspath(child_script), os.fspath(pid_file)],
                cwd=tmp_path,
                env=os.environ.copy(),
            )

    assert marker_ready.is_set()
    assert termination.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    assert not list(pose_root.glob(".rtmlib-sharded-*"))
    assert len(spawned) == 1
    child_pid, descendant_pid = (int(value) for value in pid_file.read_text().split())
    assert spawned[0].pid == child_pid
    exited = _wait_for_processes_to_exit((child_pid, descendant_pid))
    if not exited:
        try:
            os.killpg(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    assert exited


@pytest.mark.parametrize(
    ("shards", "expected_boundary", "expected_module", "expected_decode_mode"),
    [
        (1, "sequential", "dataset_builder.vision", "sequential"),
        (4, "sharded", "dataset_builder.pose_sharding", "seek"),
    ],
)
def test_pose_plan_keeps_one_shard_sequential_and_selects_multiple_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shards: int,
    expected_boundary: str,
    expected_module: str,
    expected_decode_mode: str,
) -> None:
    metadata = _metadata(tmp_path)
    arrays = _pose_arrays(metadata.frame_count)
    config = replace(
        cli.load_builder_config(cli.REPO_ROOT / "configs/dataset_builder/trial.toml"),
        pose_shards=shards,
        pose_device="cpu",
        pose_n_max=2,
    )
    runtime = RuntimeSupport(config, tmp_path / "run")
    runtime.state.metadata["video"] = metadata
    runtime.pose_interpreter = InterpreterIdentity("/fixture/python", "Python 3.12")
    observed: list[str] = []

    def extraction(boundary: str, **kwargs: object) -> vision.PoseExtraction:
        observed.append(boundary)
        output_dir = Path(kwargs["output_dir"])
        artifacts = vision.save_pose_arrays(output_dir, arrays, metadata.frame_count)
        return vision.PoseExtraction(arrays, artifacts, (boundary,))

    monkeypatch.setattr(
        _vision_plans,
        "extract_rtmlib_pose_stage",
        lambda **kwargs: extraction("sequential", **kwargs),
    )
    monkeypatch.setattr(
        _vision_plans,
        "extract_sharded_rtmlib_pose_stage",
        lambda **kwargs: extraction("sharded", **kwargs),
    )

    plan = _vision_plans._pose_plan(runtime, "video")
    execution = plan.execute()

    assert execution.outcome is StageOutcome.PROCESSED
    assert observed == [expected_boundary]
    assert plan.command[2] == expected_module
    assert plan.configuration["shards"] == shards
    assert plan.configuration["decode_mode"] == expected_decode_mode


def _wait_for_processes_to_exit(process_ids: tuple[int, ...]) -> bool:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not any(_process_is_running(process_id) for process_id in process_ids):
            return True
        time.sleep(0.02)
    return not any(_process_is_running(process_id) for process_id in process_ids)


def _process_is_running(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    state_path = Path(f"/proc/{process_id}/stat")
    if not state_path.is_file():
        return True
    state = state_path.read_text(encoding="utf-8").rsplit(")", maxsplit=1)[-1].split()[0]
    return state != "Z"
