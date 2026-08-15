"""Tests for pinned ShuttleSet22 source acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from annotator.video_metadata import VideoMetadata
from dataset_builder.manifest import artifact_integrity
from shuttleset22.acquisition import (
    YOUTUBE_FORMAT,
    acquire_source,
    download_command,
    inspect_public_source,
    validate_source_receipt,
    write_source_receipt,
)
from shuttleset22.manifest import (
    AnnotationCorpus,
    AnnotationMatch,
    ResolvedSource,
    SourceContext,
    SourceEntry,
    SourceKind,
    SourceManifest,
)


@dataclass
class FakeRunner:
    metadata: dict[str, object]
    download_bytes: bytes = b"video"
    fail_download: bool = False

    def __post_init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command) -> subprocess.CompletedProcess[str]:
        command = tuple(command)
        self.commands.append(command)
        if "--dump-single-json" in command:
            return subprocess.CompletedProcess(command, 0, json.dumps(self.metadata), "")
        if self.fail_download:
            return subprocess.CompletedProcess(command, 1, "", "download failed")
        template = Path(command[command.index("--output") + 1])
        template.with_name("source.mp4").write_bytes(self.download_bytes)
        return subprocess.CompletedProcess(command, 0, "", "")


def _entry() -> SourceEntry:
    return SourceEntry(
        match_id=8,
        video="India_Open_Final",
        kind=SourceKind.DOWNLOAD,
        url="https://www.youtube.com/watch?v=7_O5r9CLOVw",
        youtube_id="7_O5r9CLOVw",
    )


def _context(tmp_path: Path) -> SourceContext:
    annotation = AnnotationMatch(
        match_id=8,
        video="India_Open_Final",
        tournament="India Open",
        round_name="Finals",
        winner="Lakshya Sen",
        loser="Loh Kean Yew",
        stroke_count=1,
        max_annotated_frame=300,
        annotation_files=(tmp_path / "set1.csv",),
    )
    manifest = SourceManifest(
        schema="shuttleset22-sources/1",
        upstream_commit="commit",
        dataset_path="dataset",
        annotation_sha256="a" * 64,
        expected_fps=Fraction(30),
        official_channel_id="bwf-channel",
        videos=(_entry(),),
    )
    return SourceContext(
        annotations=AnnotationCorpus(tmp_path, (annotation,)),
        manifest=manifest,
    )


def _public_metadata(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "7_O5r9CLOVw",
        "channel_id": "bwf-channel",
        "title": "India Open Final",
        "webpage_url": "https://www.youtube.com/watch?v=7_O5r9CLOVw",
        "duration": 20,
        "formats": [
            {"format_id": "137", "ext": "mp4", "vcodec": "avc1", "fps": 30, "height": 1080},
        ],
    }
    payload.update(changes)
    return payload


def _probe(path: Path, *, fps: Fraction = Fraction(30), frames: int = 601) -> VideoMetadata:
    return VideoMetadata(path, fps, frames, 1920, 1080)


def test_download_command_is_pinned_and_bounded(tmp_path: Path) -> None:
    command = download_command(_entry(), tmp_path / "source.%(ext)s", ("uvx", "yt-dlp"))

    assert command[:2] == ("uvx", "yt-dlp")
    assert command[command.index("--format") + 1] == YOUTUBE_FORMAT
    assert command[-1] == _entry().url


def test_inspect_public_source_requires_identity_and_30_fps_format() -> None:
    runner = FakeRunner(_public_metadata())

    result = inspect_public_source(
        _entry(),
        official_channel_id="bwf-channel",
        minimum_duration_seconds=Fraction(10),
        command_runner=runner,
    )

    assert result.youtube_id == "7_O5r9CLOVw"
    assert result.compatible_format_ids == ("137",)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"id": "wrong-id-01"}, "video ID"),
        ({"channel_id": "copy-channel"}, "official BWF TV channel"),
        ({"duration": 10}, "does not cover annotation time"),
        ({"formats": []}, "no MP4 video format"),
    ],
)
def test_inspect_public_source_rejects_unpinned_or_incompatible_result(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        inspect_public_source(
            _entry(),
            official_channel_id="bwf-channel",
            minimum_duration_seconds=Fraction(10),
            command_runner=FakeRunner(_public_metadata(**changes)),
        )


def test_acquire_source_publishes_only_validated_video(tmp_path: Path) -> None:
    context = _context(tmp_path)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    overlap_root.mkdir()
    runner = FakeRunner(_public_metadata())

    source = acquire_source(
        context,
        8,
        source_root=source_root,
        overlap_root=overlap_root,
        command_runner=runner,
        metadata_probe=_probe,
    )

    destination = source_root / "08 India_Open_Final.mp4"
    assert destination.read_bytes() == b"video"
    assert source.metadata.source_path == destination.resolve()
    assert source.integrity.size_bytes == 5
    assert len(runner.commands) == 2


def test_acquire_source_failure_leaves_no_published_video(tmp_path: Path) -> None:
    context = _context(tmp_path)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    overlap_root.mkdir()

    with pytest.raises(RuntimeError, match="download failed"):
        acquire_source(
            context,
            8,
            source_root=source_root,
            overlap_root=overlap_root,
            command_runner=FakeRunner(_public_metadata(), fail_download=True),
            metadata_probe=_probe,
        )

    assert not (source_root / "08 India_Open_Final.mp4").exists()


def test_acquire_source_rejects_wrong_media_before_publish(tmp_path: Path) -> None:
    context = _context(tmp_path)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    overlap_root.mkdir()

    with pytest.raises(ValueError, match="FPS 25"):
        acquire_source(
            context,
            8,
            source_root=source_root,
            overlap_root=overlap_root,
            command_runner=FakeRunner(_public_metadata()),
            metadata_probe=lambda path: _probe(path, fps=Fraction(25)),
        )

    assert not (source_root / "08 India_Open_Final.mp4").exists()


def test_acquire_source_rejects_metadata_for_another_file(tmp_path: Path) -> None:
    context = _context(tmp_path)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    overlap_root.mkdir()

    with pytest.raises(ValueError, match="download metadata path"):
        acquire_source(
            context,
            8,
            source_root=source_root,
            overlap_root=overlap_root,
            command_runner=FakeRunner(_public_metadata()),
            metadata_probe=lambda _path: _probe((tmp_path / "other.mp4").resolve()),
        )

    assert not (source_root / "08 India_Open_Final.mp4").exists()


def test_source_receipt_round_trips_and_detects_changed_source(tmp_path: Path) -> None:
    context = _context(tmp_path)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    overlap_root.mkdir()
    source = acquire_source(
        context,
        8,
        source_root=source_root,
        overlap_root=overlap_root,
        command_runner=FakeRunner(_public_metadata()),
        metadata_probe=_probe,
    )

    receipt = write_source_receipt(context, source, receipt_root=source_root)

    assert receipt.name.endswith(".source.json.gz")
    assert validate_source_receipt(context, source, receipt_root=source_root) == receipt
    source.metadata.source_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from the current source"):
        validate_source_receipt(context, source, receipt_root=source_root)


def test_source_receipt_is_portable_between_storage_roots(tmp_path: Path) -> None:
    context = _context(tmp_path)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    overlap_root.mkdir()
    source = acquire_source(
        context,
        8,
        source_root=source_root,
        overlap_root=overlap_root,
        command_runner=FakeRunner(_public_metadata()),
        metadata_probe=_probe,
    )
    receipt = write_source_receipt(context, source, receipt_root=source_root)
    portable_root = tmp_path / "portable"
    portable_video = portable_root / source.metadata.source_path.name
    portable_receipt = portable_root / "receipts" / receipt.name
    portable_video.parent.mkdir(parents=True)
    portable_receipt.parent.mkdir(parents=True)
    shutil.copy2(source.metadata.source_path, portable_video)
    shutil.copy2(receipt, portable_receipt)
    moved = ResolvedSource(
        entry=source.entry,
        annotations=source.annotations,
        metadata=_probe(portable_video.resolve()),
        integrity=artifact_integrity(source.integrity.name, portable_video),
    )

    assert validate_source_receipt(context, moved, receipt_root=portable_root) == portable_receipt
