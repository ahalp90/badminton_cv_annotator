"""Small records and cleaning helpers for fixed annotator measurements."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Sequence

from annotator.artifact_io import (
    encode_json_object,
    open_text_artifact,
    read_json_object,
    resolve_artifact_path,
    write_json_object,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIRECTORY = REPO_ROOT / "experiments" / "annotator" / "runs"
BACKUPS_DIRECTORY = REPO_ROOT / "local_scratch" / "annotator_experiment_backups"
RUN_ID_PATTERN = re.compile(r"\d{8}-\d{6}\Z")
HOME_REPOSITORY_PATTERN = re.compile(
    r"/home/(?P<username>[^/\s]+)/(?:[^/\s]+/)*"
    r"badminton_(?:cv_annotator|stroke_classification)"
)
SCRATCH_STAGING_PATTERN = re.compile(
    r"/scratch/(?P<allocation>[^/\s]+)/(?P<username>[^/\s]+)/[^/\s]+"
)


def utc_run_directory(now: datetime | None = None) -> Path:
    """Return a new UTC-named directory below the annotator runs directory."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("run timestamp must include a timezone")
    return RUNS_DIRECTORY / current.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")


def count_compressed_npy(run_root: Path) -> dict[str, int]:
    """Count compressed arrays that the run actually wrote."""
    count = 0
    total_bytes = 0
    for path in run_root.rglob("*.npy.xz"):
        if path.is_file() and not path.is_symlink():
            count += 1
            total_bytes += path.stat().st_size
    return {"file_count": count, "total_bytes": total_bytes}


def human_bytes(total_bytes: int) -> str:
    """Format a byte total for terminal output and the short report."""
    value = float(total_bytes)
    for unit in ("bytes", "KiB", "MiB"):
        if value < 1024:
            return f"{int(value)} {unit}" if unit == "bytes" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def _read_json(path: Path) -> dict[str, Any]:
    return read_json_object(path)


def _regular_run_file(run_root: Path, relative_path: str) -> Path:
    path = resolve_artifact_path(run_root / relative_path)
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file() or run_root not in resolved.parents:
        raise ValueError(f"run manifest references an invalid file: {relative_path}")
    return resolved


def build_summary(run_root: Path) -> dict[str, Any]:
    """Collect the closed measurement manifests and existing metrics files."""
    root_manifest = _read_json(_regular_run_file(run_root, "manifest.json.gz"))
    configurations: list[dict[str, Any]] = []
    for record in root_manifest.get("configurations", []):
        if not isinstance(record, dict) or not isinstance(record.get("manifest"), dict):
            raise ValueError("successful run manifest has an invalid configuration record")
        manifest_path = _regular_run_file(run_root, str(record["manifest"]["path"]))
        leaf = _read_json(manifest_path)
        metrics_relative = (manifest_path.parent / "metrics.json.gz").relative_to(run_root).as_posix()
        metrics = _read_json(_regular_run_file(run_root, metrics_relative))
        configurations.append({
            "configuration_id": leaf["configuration_id"],
            "status": leaf["status"],
            "case_id": leaf["case_id"],
            "court_parent": leaf["court_parent"],
            "tracknet_stride": leaf["tracknet_stride"],
            "tracknet_producer_mode": leaf["tracknet_producer_mode"],
            "metrics": metrics,
        })
    if len(configurations) != 8:
        raise ValueError(f"successful annotator run must have eight configurations, got {len(configurations)}")
    return {
        "schema_version": 1,
        "run_id": root_manifest["run_id"],
        "status": root_manifest["status"],
        "source_commit": root_manifest["source_commit"],
        "started_at_utc": root_manifest["started_at_utc"],
        "finished_at_utc": root_manifest["finished_at_utc"],
        "elapsed_seconds": root_manifest["elapsed_seconds"],
        "device": {
            "requested": root_manifest["environment"]["requested_device"],
            "resolved": root_manifest["environment"]["resolved_device"],
        },
        "environment": root_manifest["environment"],
        "configurations": configurations,
        "compressed_npy": count_compressed_npy(run_root),
    }


def _metric(metrics: dict[str, Any], group: str, key: str) -> Any:
    value = metrics.get(group, {})
    return value.get(key, "-") if isinstance(value, dict) else "-"


def _number(value: float | int | str | None) -> str:
    return "-" if value is None or value == "-" else f"{float(value):.3f}"


def format_report(summary: dict[str, Any]) -> str:
    """Render the short human-readable report from the summary only."""
    rows = [
        "| configuration | rally coverage | contact P/R/F1 | +/-5 P/R/F1 | +/-10 P/R/F1 | court valid |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in summary["configurations"]:
        metrics = item["metrics"]
        calibration = metrics.get("existing_calibration", {})
        strict = metrics.get("strict_contacts", {})
        coverage = calibration.get("covered_fraction", "-")
        contact = "/".join(_number(calibration.get(key)) for key in ("contact_precision", "contact_recall", "contact_f1"))
        five = "/".join(_number(_metric(strict, "base30_5", key)) for key in ("precision", "recall", "f1"))
        ten = "/".join(_number(_metric(strict, "base30_10", key)) for key in ("precision", "recall", "f1"))
        rows.append(f"| {item['configuration_id']} | {_number(coverage)} | {contact} | {five} | {ten} | "
                    f"{_number(metrics.get('court_valid_fraction'))} |")
    compressed = summary["compressed_npy"]
    return "\n".join([
        f"# Annotator run {summary['run_id']}",
        "",
        f"Outcome: {summary['status']}; elapsed: {float(summary['elapsed_seconds']):.1f} seconds.",
        f"Device: requested {summary['device']['requested']}, resolved {summary['device']['resolved']}.  "
        f"Source commit: {summary['source_commit']}.",
        "",
        *rows,
        "",
        "Live CourtKeyNet/OpenCV detection is the operational default. Static homography is the controlled "
        "reference and manual fixed-camera fallback.",
        f"Compressed masks and arrays: {compressed['file_count']} NPY.XZ files "
        f"({human_bytes(compressed['total_bytes'])}). Git will preserve them with the run.",
    ])


def write_summary_and_report(run_root: Path) -> tuple[Path, Path, dict[str, int]]:
    """Write the two small run records after a successful measurement."""
    summary = build_summary(run_root)
    summary_path = run_root / "summary.json.gz"
    write_json_object(summary_path, summary)
    report_path = run_root / "report.md"
    report_path.write_text(format_report(summary) + "\n", encoding="utf-8")
    return summary_path, report_path, summary["compressed_npy"]


def _validate_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if path.is_symlink() or not resolved.is_dir() or resolved.parent != RUNS_DIRECTORY.resolve():
        raise ValueError("cleaner requires one non-symlink direct child of experiments/annotator/runs")
    if not RUN_ID_PATTERN.fullmatch(resolved.name):
        raise ValueError("run directory name must be YYYYMMDD-HHMMSS")
    return resolved


def _candidate_files(run_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and not path.name.endswith((".npy", ".npy.xz"))
    ]


def _private_tokens(manifest: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for field in ("command", "input_manifest_source"):
        text = manifest.get(field)
        values = text if isinstance(text, list) else [text]
        for value in values:
            if not isinstance(value, str):
                continue
            for match in HOME_REPOSITORY_PATTERN.finditer(value):
                tokens.add(match.group("username"))
            for match in SCRATCH_STAGING_PATTERN.finditer(value):
                tokens.add(match.group("allocation"))
                tokens.add(match.group("username"))
    return {token for token in tokens if token}


def _sanitise_path(value: str) -> str:
    value = HOME_REPOSITORY_PATTERN.sub("<repo>", value)
    return SCRATCH_STAGING_PATTERN.sub("<scratch>/<measurement-staging>", value)


def _planned_json_changes(run_root: Path) -> tuple[dict[Path, dict[str, Any]], set[str]]:
    root_path = _regular_run_file(run_root, "manifest.json.gz")
    root = _read_json(root_path)
    tokens = _private_tokens(root)
    changes: dict[Path, dict[str, Any]] = {}
    for key in ("command", "input_manifest_source"):
        value = root.get(key)
        if isinstance(value, str):
            cleaned = _sanitise_path(value)
            if cleaned != value:
                root[key] = cleaned
                changes[root_path] = root
        elif key == "command" and isinstance(value, list):
            cleaned_values = [_sanitise_path(item) if isinstance(item, str) else item for item in value]
            if cleaned_values != value:
                root[key] = cleaned_values
                changes[root_path] = root
    for record in root.get("configurations", []):
        if not isinstance(record, dict) or not isinstance(record.get("manifest"), dict):
            continue
        leaf_path = _regular_run_file(run_root, str(record["manifest"]["path"]))
        leaf = _read_json(leaf_path)
        command = leaf.get("command")
        if isinstance(command, list):
            cleaned = [_sanitise_path(item) if isinstance(item, str) else item for item in command]
            if cleaned != command:
                leaf["command"] = cleaned
                encoded = encode_json_object(leaf_path, leaf)
                record["manifest"]["bytes"] = len(encoded)
                record["manifest"]["md5"] = hashlib.md5(encoded).hexdigest()
                changes[leaf_path] = leaf
                changes[root_path] = root
    return changes, tokens


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_object(path, payload)


def _backup(run_root: Path) -> Path:
    BACKUPS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    clean_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    archive = BACKUPS_DIRECTORY / f"{run_root.name}_cleaned_{clean_id}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in _candidate_files(run_root):
            handle.add(path, arcname=path.relative_to(run_root).as_posix(), recursive=False)
    print(f"Pre-clean backup: {archive}")
    return archive


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"missing {name}; install with: uv sync --extra annotator-experiments")


def _build_scan_mirror(run_root: Path, files: list[Path], mirror_root: Path) -> dict[Path, Path]:
    """Copy text candidates into a plain temporary tree mapped to their sources."""
    sources_by_mirror: dict[Path, Path] = {}
    for source in files:
        relative = source.relative_to(run_root)
        compressed_text = source.name.endswith((".json.gz", ".csv.gz"))
        if compressed_text:
            relative = relative.with_name(relative.name[:-3])
        mirrored = mirror_root / relative
        if mirrored in sources_by_mirror:
            raise ValueError(f"compressed and plain scan candidates collide: {relative}")
        mirrored.parent.mkdir(parents=True, exist_ok=True)
        if compressed_text:
            with open_text_artifact(source, newline="") as input_handle:
                with mirrored.open("w", encoding="utf-8", newline="") as output_handle:
                    shutil.copyfileobj(input_handle, output_handle)
        else:
            shutil.copyfile(source, mirrored)
        sources_by_mirror[mirrored] = source
    return sources_by_mirror


def _scanner_findings(run_root: Path, files: list[Path], tokens: set[str]) -> set[Path]:
    _require_tool("betterleaks")
    _require_tool("rg")
    relative = [path.relative_to(run_root).as_posix() for path in files]
    leaks = subprocess.run(
        [
            "betterleaks",
            "dir",
            "--redact",
            "--report-format",
            "json",
            "--report-path",
            "-",
            "--no-banner",
            *relative,
        ],
        cwd=run_root,
        capture_output=True, text=True, check=False,
    )
    if leaks.returncode not in {0, 1}:
        raise RuntimeError(f"betterleaks failed with exit code {leaks.returncode}")
    try:
        leak_data = json.loads(leaks.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("betterleaks returned an unreadable result") from error
    if leak_data is None:
        leak_data = []
    if not isinstance(leak_data, list):
        raise RuntimeError("betterleaks returned an unreadable result")
    if (leaks.returncode == 0) != (not leak_data):
        raise RuntimeError("betterleaks exit code and report disagree")
    candidate_set = set(files)
    findings: set[Path] = set()
    for item in leak_data:
        if not isinstance(item, dict) or not isinstance(item.get("File"), str):
            raise RuntimeError("betterleaks returned an unreadable finding")
        candidate = (run_root / item["File"]).resolve()
        if candidate not in candidate_set:
            raise RuntimeError("betterleaks returned a path outside the scan candidates")
        findings.add(candidate)
    patterns = [r"/(?:home|scratch|srv)/"] + [re.escape(token) for token in sorted(tokens)]
    ripgrep_command = ["rg", "--files-with-matches", "--no-messages"]
    for pattern in patterns:
        ripgrep_command.extend(["-e", pattern])
    scan = subprocess.run(
        [*ripgrep_command, "--", *relative], cwd=run_root,
        capture_output=True, text=True, check=False,
    )
    if scan.returncode not in {0, 1}:
        raise RuntimeError(f"ripgrep failed with exit code {scan.returncode}")
    matched_paths = scan.stdout.splitlines()
    if (scan.returncode == 0) != bool(matched_paths):
        raise RuntimeError("ripgrep exit code and output disagree")
    if scan.returncode == 0:
        for line in matched_paths:
            candidate = run_root / line
            if candidate not in files:
                raise RuntimeError("ripgrep returned a path outside the scan candidates")
            findings.add(candidate)
    return findings


def _decompressed_scanner_findings(run_root: Path, files: list[Path], tokens: set[str]) -> set[Path]:
    """Scan plain temporary copies and map each finding to its run artifact."""
    with tempfile.TemporaryDirectory(prefix="annotator-run-scan-") as temporary_directory:
        mirror_root = Path(temporary_directory)
        sources_by_mirror = _build_scan_mirror(run_root, files, mirror_root)
        mirror_findings = _scanner_findings(mirror_root, list(sources_by_mirror), tokens)
        return {sources_by_mirror[path] for path in mirror_findings}


def clean_run(run_path: Path) -> Path | None:
    """Back up and clean one completed run. Return the archive, if one was needed."""
    run_root = _validate_run_root(run_path)
    changes, tokens = _planned_json_changes(run_root)
    archive: Path | None = None
    try:
        if changes:
            archive = _backup(run_root)
            for path, payload in changes.items():
                _write_json(path, payload)
        findings = _decompressed_scanner_findings(run_root, _candidate_files(run_root), tokens)
        if findings and archive is None:
            archive = _backup(run_root)
        for path in sorted(findings):
            relative = path.relative_to(run_root)
            path.unlink()
            print(f"Deleted unsafe file: {relative.as_posix()}")
    except Exception as error:
        suffix = f" Pre-clean backup: {archive}." if archive else ""
        raise RuntimeError(f"measurement remains at {run_root}.{suffix} {error}") from error
    if archive is None:
        print("Run is already clean.")
    return archive


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Back up and clean an annotator experiment run")
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args(argv)
    try:
        clean_run(args.run_directory)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
