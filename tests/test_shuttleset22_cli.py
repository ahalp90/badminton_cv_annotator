"""Tests for the resumable ShuttleSet22 command-line workflow."""

from __future__ import annotations

from fractions import Fraction
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from annotator.video_metadata import VideoMetadata
from dataset_builder.manifest import artifact_integrity
from dataset_builder.vision import load_json_gz
from shuttleset22 import cli
from shuttleset22.acquisition import write_source_receipt
from shuttleset22.extraction import ExtractionSettings
from shuttleset22.manifest import (
    AnnotationCorpus,
    AnnotationMatch,
    ResolvedSource,
    SourceContext,
    SourceEntry,
    SourceKind,
    SourceManifest,
)


def _context(tmp_path: Path) -> SourceContext:
    available = SourceEntry(
        match_id=8,
        video="India_Open_Final",
        kind=SourceKind.DOWNLOAD,
        url="https://www.youtube.com/watch?v=7_O5r9CLOVw",
        youtube_id="7_O5r9CLOVw",
    )
    unavailable = SourceEntry(
        match_id=45,
        video="Trimmed_Match",
        kind=SourceKind.UNRESOLVED,
        unresolved_reason="No frame-aligned source is public.",
    )
    annotations = tuple(
        AnnotationMatch(
            match_id=entry.match_id,
            video=entry.video,
            tournament="Tournament",
            round_name="Finals",
            winner="Winner",
            loser="Loser",
            stroke_count=1,
            max_annotated_frame=2,
            annotation_files=(tmp_path / f"{entry.match_id}.csv",),
        )
        for entry in (available, unavailable)
    )
    return SourceContext(
        annotations=AnnotationCorpus(tmp_path, annotations),
        manifest=SourceManifest(
            schema="shuttleset22-sources/1",
            upstream_commit="upstream",
            dataset_path="dataset",
            annotation_sha256="a" * 64,
            expected_fps=Fraction(30),
            official_channel_id="channel",
            videos=(available, unavailable),
        ),
    )


def _resolved(context: SourceContext, source_root: Path) -> ResolvedSource:
    entry = context.manifest.by_id()[8]
    video = source_root / entry.canonical_filename
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    return ResolvedSource(
        entry=entry,
        annotations=context.annotations.by_id()[8],
        metadata=VideoMetadata(video.resolve(), Fraction(30), 3, 100, 50),
        integrity=artifact_integrity("shuttleset22_source_08", video),
    )


def _settings(tmp_path: Path) -> ExtractionSettings:
    return ExtractionSettings(
        source_commit="a" * 40,
        tracknet_dir=tmp_path / "tracknet",
        tracknet_model=tmp_path / "tracknet.pt",
        inpaint_model=tmp_path / "inpaint.pt",
        tracknet_python=tmp_path / "tracknet-python",
        pose_interpreter=tmp_path / "pose-python",
    )


def test_acquisition_writes_receipts_report_and_resume_status(tmp_path: Path) -> None:
    context = _context(tmp_path)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    overlap_root.mkdir()

    def acquire(_context, _match_id, **_kwargs):
        return _resolved(context, source_root)

    first = cli.run_acquisition(
        context,
        source_root=source_root,
        overlap_root=overlap_root,
        video_ids=[8, 45],
        acquire=acquire,
    )

    assert first.counts() == {"processed": 1, "unavailable": 1}
    payload = load_json_gz(first.report)
    assert payload["selected_ids"] == [8, 45]
    assert payload["statuses"] == list(first.statuses)
    assert (source_root / str(first.statuses[0]["receipt"])).is_file()
    assert first.statuses[0]["source_metadata"]["source_path"] == "08 India_Open_Final.mp4"

    second = cli.run_acquisition(
        context,
        source_root=source_root,
        overlap_root=overlap_root,
        video_ids=[8],
        acquire=acquire,
    )

    assert second.counts() == {"reused": 1}


def test_acquisition_records_failure_and_continues(tmp_path: Path) -> None:
    context = _context(tmp_path)
    overlap_root = tmp_path / "overlap"
    overlap_root.mkdir()

    result = cli.run_acquisition(
        context,
        source_root=tmp_path / "sources",
        overlap_root=overlap_root,
        video_ids=[8, 45],
        acquire=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    assert result.counts() == {"failed": 1, "unavailable": 1}
    assert result.failures[0]["reason"] == "RuntimeError: offline"
    assert load_json_gz(result.report)["statuses"] == list(result.statuses)


def test_existing_download_without_receipt_is_rejected(tmp_path: Path) -> None:
    context = _context(tmp_path)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    overlap_root.mkdir()
    source = _resolved(context, source_root)

    result = cli.run_acquisition(
        context,
        source_root=source_root,
        overlap_root=overlap_root,
        video_ids=[8],
        acquire=lambda *_args, **_kwargs: source,
    )

    assert result.counts() == {"failed": 1}
    assert "source receipt is not a regular file" in str(result.failures[0]["reason"])


def test_extraction_requires_receipt_and_writes_incremental_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    overlap_root.mkdir()
    source = _resolved(context, source_root)
    write_source_receipt(context, source, receipt_root=source_root)
    monkeypatch.setattr(cli, "resolve_sources", lambda *_args, **_kwargs: (source,))
    output_root = tmp_path / "outputs"

    def extract(_source, *, output_root, settings):
        assert settings.source_commit == "a" * 40
        output_dir = output_root / "08 India_Open_Final"
        output_dir.mkdir(parents=True)
        artifact = output_dir / "pose.npy.xz"
        receipt = output_dir / "extraction_receipt.json.gz"
        artifact.write_bytes(b"artifact")
        receipt.write_bytes(b"receipt")
        return SimpleNamespace(
            reused=False,
            receipt=receipt,
            output_dir=output_dir,
            artifacts={"pose": artifact},
        )

    result = cli.run_extraction(
        context,
        source_root=source_root,
        overlap_root=overlap_root,
        output_root=output_root,
        settings=_settings(tmp_path),
        video_ids=[8, 45],
        extract=extract,
    )

    assert result.counts() == {"processed": 1, "unavailable": 1}
    payload = load_json_gz(result.report)
    assert payload["source_commit"] == "a" * 40
    assert payload["settings"]["pose_n_max"] == 16
    assert payload["statuses"] == list(result.statuses)


def test_duplicate_requested_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicates"):
        cli.run_acquisition(
            _context(tmp_path),
            source_root=tmp_path / "sources",
            overlap_root=tmp_path / "overlap",
            video_ids=[8, 8],
        )


def test_extraction_stops_after_three_identical_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = tuple(
        SourceEntry(
            match_id=match_id,
            video=f"Match_{match_id}",
            kind=SourceKind.DOWNLOAD,
            url=f"https://www.youtube.com/watch?v=source{match_id:05d}",
            youtube_id=f"source{match_id:05d}",
        )
        for match_id in (8, 9, 10, 11)
    )
    annotations = tuple(
        AnnotationMatch(
            match_id=entry.match_id,
            video=entry.video,
            tournament="Tournament",
            round_name="Finals",
            winner="Winner",
            loser="Loser",
            stroke_count=1,
            max_annotated_frame=2,
            annotation_files=(tmp_path / f"{entry.match_id}.csv",),
        )
        for entry in entries
    )
    context = SourceContext(
        annotations=AnnotationCorpus(tmp_path, annotations),
        manifest=SourceManifest(
            schema="shuttleset22-sources/1",
            upstream_commit="upstream",
            dataset_path="dataset",
            annotation_sha256="a" * 64,
            expected_fps=Fraction(30),
            official_channel_id="channel",
            videos=entries,
        ),
    )
    source_root = tmp_path / "sources"

    def resolve(_context, *, video_ids, **_kwargs):
        entry = context.manifest.by_id()[video_ids[0]]
        video = source_root / entry.canonical_filename
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{entry.match_id}".encode())
        return (
            ResolvedSource(
                entry=entry,
                annotations=context.annotations.by_id()[entry.match_id],
                metadata=VideoMetadata(video.resolve(), Fraction(30), 3, 100, 50),
                integrity=artifact_integrity(f"source-{entry.match_id}", video),
            ),
        )

    monkeypatch.setattr(cli, "resolve_sources", resolve)
    monkeypatch.setattr(cli, "validate_source_receipt", lambda *_args, **_kwargs: None)
    calls = 0

    def fail_extract(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("GPU unavailable")

    result = cli.run_extraction(
        context,
        source_root=source_root,
        overlap_root=tmp_path / "overlap",
        output_root=tmp_path / "outputs",
        settings=_settings(tmp_path),
        video_ids=[8, 9, 10, 11],
        extract=fail_extract,
    )

    assert calls == 3
    assert result.counts() == {"failed": 3, "pending": 1}
    assert "three consecutive" in str(result.statuses[3]["reason"])
    assert load_json_gz(result.report)["statuses"] == list(result.statuses)


def test_validate_sources_cli_reports_source_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_source_context", lambda *_args: _context(tmp_path))

    status = cli.main(
        [
            "validate-sources",
            "--dataset-root",
            str(tmp_path),
            "--shuttleset-match-csv",
            str(tmp_path / "match.csv"),
            "--ids",
            "8",
            "45",
        ]
    )

    assert status == 0
    assert capsys.readouterr().out.strip() == (
        "validated 2 source identities: download=1, unresolved=1"
    )


def test_non_extraction_cli_does_not_require_tracknet_import_path() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(cli.REPO_ROOT / "src")

    completed = subprocess.run(
        [sys.executable, "-m", "shuttleset22", "validate-sources", "--help"],
        cwd=cli.REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
