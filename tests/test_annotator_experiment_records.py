"""Focused tests for annotator experiment records and safe cleaning."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tarfile

import pytest

import annotator.experiment_records as records
from annotator.artifact_io import read_json_object, write_json_object


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runs = tmp_path / "experiments" / "annotator" / "runs"
    run = runs / "20260730-183124"
    run.mkdir(parents=True)
    monkeypatch.setattr(records, "RUNS_DIRECTORY", runs)
    monkeypatch.setattr(records, "BACKUPS_DIRECTORY", tmp_path / "backups")
    return run


def _leaf(command: list[str]) -> dict[str, object]:
    return {
        "configuration_id": "static_shuttleset_homography/sset_01/tracknet-stride-8",
        "status": "succeeded", "case_id": "sset_01/tracknet-stride-8",
        "court_parent": "static_shuttleset_homography", "tracknet_stride": 8,
        "tracknet_producer_mode": "nonoverlap", "command": command,
    }


def _root(run: Path, leaf_path: Path, command: list[str]) -> dict[str, object]:
    data = leaf_path.read_bytes()
    return {
        "run_id": "20260730-183124", "status": "succeeded", "source_commit": "a" * 40,
        "started_at_utc": "2026-07-30T00:00:00Z", "finished_at_utc": "2026-07-30T00:01:00Z",
        "elapsed_seconds": 60.0, "command": command,
        "input_manifest_source": "/home/student-user/badminton_cv_annotator/inputs.json",
        "environment": {"requested_device": "cpu", "resolved_device": "cpu", "packages": {}},
        "configurations": [{"manifest": {"path": leaf_path.relative_to(run).as_posix(), "bytes": len(data),
                                        "md5": hashlib.md5(data).hexdigest()}}],
    }


def _install_scanners(monkeypatch: pytest.MonkeyPatch, findings: list[str] | None = None) -> None:
    monkeypatch.setattr(records.shutil, "which", lambda _name: "/mock/tool")

    def fake_run(command, **_kwargs):
        if command[0] == "betterleaks":
            report = [{"File": path} for path in findings or []]
            return subprocess.CompletedProcess(command, int(bool(report)), json.dumps(report or None), "")
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(records.subprocess, "run", fake_run)


def test_utc_directory_naming_and_collision_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setattr(records, "RUNS_DIRECTORY", runs)
    path = records.utc_run_directory(datetime(2026, 7, 30, 18, 31, 24, tzinfo=timezone.utc))
    assert path.name == "20260730-183124"
    path.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        import annotator.e2e_court_annotator as runner
        monkeypatch.setattr(runner, "utc_run_directory", lambda: path)
        runner._run_cli_measurement(tmp_path / "missing.json", "cpu", ("runner",))


@pytest.mark.parametrize("repository_name", ["badminton_cv_annotator", "badminton_stroke_classification"])
def test_path_sanitiser_accepts_current_and_historical_repository_names(repository_name: str) -> None:
    assert records._sanitise_path(f"/home/student-user/{repository_name}/inputs.json") == "<repo>/inputs.json"


def test_cli_writes_records_before_cleaning_and_skips_them_after_measurement_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import annotator.e2e_court_annotator as runner

    run = tmp_path / "experiments" / "annotator" / "runs" / "20260730-183124"
    run.parent.mkdir(parents=True)
    events: list[str] = []
    monkeypatch.setattr(runner, "utc_run_directory", lambda: run)
    monkeypatch.setattr(runner, "run_annotator_measurement", lambda *_args, **_kwargs: 0)

    def write_records(_path: Path) -> tuple[Path, Path, dict[str, int]]:
        events.append("report")
        return run / "summary.json.gz", run / "report.md", {"file_count": 2, "total_bytes": 4}

    monkeypatch.setattr(runner, "write_summary_and_report", write_records)
    monkeypatch.setattr(runner, "clean_run", lambda _path: (events.append("clean") or None))
    assert runner._run_cli_measurement(tmp_path / "input.json", "cpu", ("runner",)) == 0
    assert events == ["report", "clean"]
    monkeypatch.setattr(runner, "run_annotator_measurement", lambda *_args, **_kwargs: 3)
    assert runner._run_cli_measurement(tmp_path / "input.json", "cpu", ("runner",)) == 3
    assert events == ["report", "clean"]
    monkeypatch.setattr(runner, "run_annotator_measurement", lambda *_args, **_kwargs: 0)

    def fail_summary(_path: Path) -> None:
        raise KeyError("status")

    monkeypatch.setattr(runner, "write_summary_and_report", fail_summary)
    assert runner._run_cli_measurement(tmp_path / "input.json", "cpu", ("runner",)) == 1
    assert "measurement completed" in capsys.readouterr().err


def test_summary_report_and_actual_npy_totals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run(tmp_path, monkeypatch)
    leaf = run / "static_shuttleset_homography" / "sset_01" / "tracknet-stride-8" / "manifest.json"
    _write_json(leaf, _leaf(["python", "/home/student-user/badminton_cv_annotator/run.py"]))
    _write_json(leaf.parent / "metrics.json", {
        "existing_calibration": {
            "covered_fraction": 1.0,
            "contact_precision": 0.5,
            "contact_recall": 0.6,
            "contact_f1": 0.545,
        },
        "strict_contacts": {"base30_5": {"precision": 0.1, "recall": 0.2, "f1": 0.13},
                            "base30_10": {"precision": 0.3, "recall": 0.4, "f1": 0.34}},
        "court_valid_fraction": 0.8,
    })
    for index in range(7):
        clone = run / f"parent{index}" / "case" / "manifest.json"
        _write_json(clone, _leaf(["python"]))
    root = _root(run, leaf, ["python", "/scratch/allocation/measurement-staging/run"])
    root["configurations"] = [
        {"manifest": {"path": path.relative_to(run).as_posix(), "bytes": len(path.read_bytes()),
                       "md5": hashlib.md5(path.read_bytes()).hexdigest()}}
        for path in [leaf, *(run / f"parent{index}" / "case" / "manifest.json" for index in range(7))]
    ]
    for path in [*(run / f"parent{index}" / "case" / "manifest.json" for index in range(7))]:
        _write_json(path.parent / "metrics.json", {"existing_calibration": {}, "strict_contacts": {}, "court_valid_fraction": 1.0})
    _write_json(run / "manifest.json", root)
    (run / "array.npy.xz").write_bytes(b"1234")
    summary_path, report_path, compressed = records.write_summary_and_report(run)
    assert compressed == {"file_count": 1, "total_bytes": 4}
    assert summary_path.name == "summary.json.gz"
    assert len(read_json_object(summary_path)["configurations"]) == 8
    report = report_path.read_text()
    assert "| 1.000 |" in report
    assert "CourtKeyNet/OpenCV" in report
    assert "Git will preserve them" in report


def test_cleaner_sanitises_only_manifests_updates_md5_and_preserves_npy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run(tmp_path, monkeypatch)
    leaf = run / "parent" / "case" / "manifest.json"
    scratch_path = "/scratch/allocation-id/student-user/sset_measure_deadbeef/control/input_manifest.json"
    _write_json(leaf, _leaf(["python", scratch_path]))
    root = _root(
        run,
        leaf,
        ["/home/student-user/badminton_cv_annotator/tool", scratch_path],
    )
    _write_json(run / "manifest.json", root)
    untouched = run / "other.json"
    _write_json(untouched, {"path": "/home/other-user/keep"})
    npy = run / "array.npy"
    npy.write_bytes(b"array")
    _install_scanners(monkeypatch)
    archive = records.clean_run(run)
    assert archive is not None and archive.is_file()
    cleaned_root = json.loads((run / "manifest.json").read_text())
    assert "<scratch>/<measurement-staging>" in cleaned_root["command"][1]
    assert cleaned_root["command"][0] == "<repo>/tool"
    assert cleaned_root["input_manifest_source"] == "<repo>/inputs.json"
    assert json.loads(leaf.read_text())["command"][1].startswith("<scratch>")
    assert cleaned_root["configurations"][0]["manifest"]["md5"] == hashlib.md5(leaf.read_bytes()).hexdigest()
    assert npy.read_bytes() == b"array"
    assert json.loads(untouched.read_text())["path"] == "/home/other-user/keep"
    with tarfile.open(archive) as handle:
        assert "array.npy" not in handle.getnames()
        assert "manifest.json" in handle.getnames()
    archives = list(records.BACKUPS_DIRECTORY.iterdir())
    assert records.clean_run(run) is None
    assert list(records.BACKUPS_DIRECTORY.iterdir()) == archives


def test_cleaner_sanitises_compressed_manifests_and_scans_decompressed_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run(tmp_path, monkeypatch)
    leaf = run / "parent" / "case" / "manifest.json.gz"
    scratch_path = "/scratch/allocation-id/student-user/sset_measure_deadbeef/input_manifest.json"
    write_json_object(leaf, _leaf(["python", scratch_path]))
    write_json_object(
        run / "manifest.json.gz",
        _root(run, leaf, ["/home/student-user/badminton_cv_annotator/tool"]),
    )
    unsafe = run / "unsafe.json.gz"
    write_json_object(unsafe, {"secret": "token"})
    array = run / "array.npy.xz"
    array.write_bytes(b"compressed-array")
    _install_scanners(monkeypatch, ["unsafe.json"])

    archive = records.clean_run(run)

    assert archive is not None and archive.is_file()
    cleaned_root = read_json_object(run / "manifest.json.gz")
    assert cleaned_root["command"][0] == "<repo>/tool"
    assert read_json_object(leaf)["command"][1].startswith("<scratch>")
    assert cleaned_root["configurations"][0]["manifest"]["md5"] == hashlib.md5(leaf.read_bytes()).hexdigest()
    assert not unsafe.exists()
    assert array.read_bytes() == b"compressed-array"
    with tarfile.open(archive) as handle:
        assert "manifest.json.gz" in handle.getnames()
        assert "array.npy.xz" not in handle.getnames()


def test_cleaner_rejects_outside_path_and_deletes_only_positive_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    run = _run(tmp_path, monkeypatch)
    _write_json(run / "manifest.json", {"configurations": [], "command": [], "input_manifest_source": "safe"})
    unsafe = run / "unsafe.txt"
    unsafe.write_text("secret", encoding="utf-8")
    npy = run / "keep.npy"
    npy.write_bytes(b"keep")
    _install_scanners(monkeypatch, ["unsafe.txt"])
    assert records.clean_run(run) is not None
    assert not unsafe.exists() and npy.exists()
    assert "Deleted unsafe file: unsafe.txt" in capsys.readouterr().out
    with pytest.raises(ValueError):
        records.clean_run(tmp_path)


def test_cleaner_reports_deletion_failure_with_run_and_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run(tmp_path, monkeypatch)
    _write_json(run / "manifest.json", {"configurations": [], "command": [], "input_manifest_source": "safe"})
    unsafe = run / "unsafe.txt"
    unsafe.write_text("secret", encoding="utf-8")
    _install_scanners(monkeypatch, ["unsafe.txt"])
    original_unlink = Path.unlink

    def fail_unsafe_unlink(path: Path, *args, **kwargs) -> None:
        if path == unsafe:
            raise OSError("delete failed")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_unsafe_unlink)

    with pytest.raises(RuntimeError, match=r"measurement remains at .*Pre-clean backup:"):
        records.clean_run(run)

    assert unsafe.exists()


def test_cleaner_rejects_scanner_paths_outside_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run(tmp_path, monkeypatch)
    _write_json(run / "manifest.json", {"configurations": [], "command": [], "input_manifest_source": "safe"})
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    _install_scanners(monkeypatch, [str(outside)])

    with pytest.raises(RuntimeError, match="outside the scan candidates"):
        records.clean_run(run)

    assert outside.read_text(encoding="utf-8") == "keep"


def test_cleaner_rejects_manifest_paths_outside_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run(tmp_path, monkeypatch)
    outside = tmp_path / "outside.json"
    _write_json(outside, _leaf(["/home/student-user/badminton_cv_annotator/run.py"]))
    _write_json(
        run / "manifest.json",
        {
            "configurations": [{"manifest": {"path": "../../../../outside.json"}}],
            "command": [],
            "input_manifest_source": "safe",
        },
    )

    with pytest.raises(ValueError, match="invalid file"):
        records.clean_run(run)

    assert "<repo>" not in outside.read_text(encoding="utf-8")


def test_candidate_files_exclude_npy_and_symlinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run(tmp_path, monkeypatch)
    regular = run / "record.json"
    regular.write_text("{}\n", encoding="utf-8")
    (run / "array.npy").write_bytes(b"array")
    (run / "array.npy.xz").write_bytes(b"compressed-array")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (run / "linked.txt").symlink_to(outside)

    assert records._candidate_files(run) == [regular]


def test_scanner_failure_keeps_run_and_reports_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run(tmp_path, monkeypatch)
    leaf = run / "parent" / "case" / "manifest.json"
    _write_json(leaf, _leaf(["/scratch/allocation-id/student-user/sset_measure_deadbeef/x"]))
    _write_json(run / "manifest.json", _root(run, leaf, ["python"]))
    monkeypatch.setattr(records.shutil, "which", lambda _name: "/mock/tool")
    monkeypatch.setattr(records.subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 2, "", "bad"))
    with pytest.raises(RuntimeError, match="Pre-clean backup"):
        records.clean_run(run)
    assert run.exists()


def test_ripgrep_exit_code_and_output_must_agree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run(tmp_path, monkeypatch)
    manifest = run / "manifest.json"
    _write_json(manifest, {"configurations": [], "command": [], "input_manifest_source": "safe"})
    monkeypatch.setattr(records.shutil, "which", lambda _name: "/mock/tool")

    def fake_run(command, **_kwargs):
        if command[0] == "betterleaks":
            return subprocess.CompletedProcess(command, 0, "null", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(records.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="ripgrep exit code and output disagree"):
        records.clean_run(run)


def test_rewrite_failure_reports_run_and_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run(tmp_path, monkeypatch)
    leaf = run / "parent" / "case" / "manifest.json"
    _write_json(leaf, _leaf(["/scratch/allocation-id/student-user/sset_measure_deadbeef/x"]))
    _write_json(run / "manifest.json", _root(run, leaf, ["python"]))

    def fail_write(*_args) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(records, "_write_json", fail_write)

    with pytest.raises(RuntimeError, match=r"measurement remains at .*Pre-clean backup:"):
        records.clean_run(run)
