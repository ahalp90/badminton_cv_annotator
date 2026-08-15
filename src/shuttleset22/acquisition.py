"""Pinned ShuttleSet22 source inspection and atomic acquisition."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from annotator.video_metadata import VideoMetadata, probe_video_metadata
from dataset_builder.manifest import artifact_integrity
from dataset_builder.vision import load_json_gz, save_json_gz
from shuttleset22.manifest import (
    ResolvedSource,
    SourceContext,
    SourceEntry,
    SourceKind,
    resolve_sources,
)


YOUTUBE_FORMAT = (
    "bv*[ext=mp4][fps=30][height<=1080]+ba[ext=m4a]/"
    "b[ext=mp4][fps=30][height<=1080]"
)
SOURCE_RECEIPT_SCHEMA = "shuttleset22-source/1"
SOURCE_RECEIPT_DIRECTORY = "receipts"

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class PublicSourceMetadata:
    """Reviewed public identity and available 30 FPS video formats."""

    youtube_id: str
    channel_id: str
    title: str
    duration_seconds: int
    compatible_format_ids: tuple[str, ...]


def source_receipt_path(receipt_root: Path, source: ResolvedSource) -> Path:
    """Return the stable compressed receipt path for one validated source."""
    filename = f"{source.entry.match_id:02d} {source.entry.video}.source.json.gz"
    return Path(receipt_root) / SOURCE_RECEIPT_DIRECTORY / filename


def write_source_receipt(
    context: SourceContext,
    source: ResolvedSource,
    *,
    receipt_root: Path,
) -> Path:
    """Atomically persist and reload the exact source identity."""
    receipt = source_receipt_path(receipt_root, source)
    payload = _source_receipt_payload(context, source)
    save_json_gz(receipt, payload)
    if load_json_gz(receipt) != payload:
        raise ValueError(f"source receipt differs after publication: {receipt}")
    return receipt


def validate_source_receipt(
    context: SourceContext,
    source: ResolvedSource,
    *,
    receipt_root: Path,
) -> Path:
    """Require an existing source receipt to match the current source exactly."""
    receipt = source_receipt_path(receipt_root, source)
    if receipt.is_symlink() or not receipt.is_file():
        raise FileNotFoundError(f"source receipt is not a regular file: {receipt}")
    source_path = source.metadata.source_path
    if source_path.is_symlink() or not source_path.is_file():
        raise FileNotFoundError(f"source is not a regular non-symlink file: {source_path}")
    current = replace(
        source,
        integrity=artifact_integrity(source.integrity.name, source_path),
    )
    if load_json_gz(receipt) != _source_receipt_payload(context, current):
        raise ValueError(f"source receipt differs from the current source: {receipt}")
    return receipt


def metadata_command(entry: SourceEntry, yt_dlp: Sequence[str] = ("yt-dlp",)) -> tuple[str, ...]:
    """Build a metadata-only command for one pinned URL."""
    url = _source_url(entry)
    return (
        *yt_dlp,
        "--no-playlist",
        "--dump-single-json",
        "--skip-download",
        url,
    )


def download_command(
    entry: SourceEntry,
    output_template: Path,
    yt_dlp: Sequence[str] = ("yt-dlp",),
) -> tuple[str, ...]:
    """Build the exact bounded acquisition command for one pinned URL."""
    url = _source_url(entry)
    return (
        *yt_dlp,
        "--no-playlist",
        "--no-overwrites",
        "--format",
        YOUTUBE_FORMAT,
        "--merge-output-format",
        "mp4",
        "--output",
        os.fspath(output_template),
        url,
    )


def inspect_public_source(
    entry: SourceEntry,
    *,
    official_channel_id: str,
    minimum_duration_seconds: Fraction,
    yt_dlp: Sequence[str] = ("yt-dlp",),
    command_runner: CommandRunner | None = None,
) -> PublicSourceMetadata:
    """Inspect a pinned URL and reject a different or incompatible source."""
    runner = command_runner or _run_command
    command = metadata_command(entry, yt_dlp)
    completed = runner(command)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"yt-dlp metadata command failed: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("yt-dlp metadata output is not JSON") from exc
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError("yt-dlp metadata must be an object")

    youtube_id = _string(payload.get("id"), "yt-dlp id")
    if youtube_id != entry.youtube_id:
        raise ValueError(f"yt-dlp video ID {youtube_id!r} != {entry.youtube_id!r}")
    channel_id = _string(payload.get("channel_id"), "yt-dlp channel_id")
    if channel_id != official_channel_id:
        raise ValueError(f"yt-dlp channel {channel_id!r} != official BWF TV channel")
    webpage_url = _string(payload.get("webpage_url"), "yt-dlp webpage_url")
    if webpage_url != entry.url:
        raise ValueError(f"yt-dlp webpage URL {webpage_url!r} != {entry.url!r}")
    duration = _positive_integer(payload.get("duration"), "yt-dlp duration")
    if Fraction(duration) <= minimum_duration_seconds:
        raise ValueError(
            f"yt-dlp duration {duration} seconds does not cover annotation time "
            f"{float(minimum_duration_seconds):.3f}"
        )
    formats = payload.get("formats")
    if not isinstance(formats, list):
        raise ValueError("yt-dlp formats must be a list")
    compatible = tuple(
        _string(format_payload.get("format_id"), "yt-dlp format_id")
        for format_payload in formats
        if _compatible_video_format(format_payload)
    )
    if not compatible:
        raise ValueError("pinned source has no MP4 video format at exact 30 FPS up to 1080p")
    return PublicSourceMetadata(
        youtube_id=youtube_id,
        channel_id=channel_id,
        title=_string(payload.get("title"), "yt-dlp title"),
        duration_seconds=duration,
        compatible_format_ids=compatible,
    )


def acquire_source(
    context: SourceContext,
    match_id: int,
    *,
    source_root: Path,
    overlap_root: Path,
    yt_dlp: Sequence[str] = ("yt-dlp",),
    command_runner: CommandRunner | None = None,
    metadata_probe: Callable[[Path], VideoMetadata] = probe_video_metadata,
) -> ResolvedSource:
    """Download, validate, and atomically publish one non-overlap source."""
    entries = context.manifest.by_id()
    if match_id not in entries:
        raise ValueError(f"unknown ShuttleSet22 match ID: {match_id}")
    entry = entries[match_id]
    if entry.kind is not SourceKind.DOWNLOAD:
        raise ValueError(f"source {match_id} is not downloadable: {entry.kind}")
    downloads = Path(source_root)
    downloads.mkdir(parents=True, exist_ok=True)
    destination = downloads / entry.canonical_filename
    if destination.exists() or destination.is_symlink():
        return resolve_sources(
            context,
            source_root=downloads,
            overlap_root=overlap_root,
            video_ids=[match_id],
            metadata_probe=metadata_probe,
        )[0]

    annotation = context.annotations.by_id()[match_id]
    inspect_public_source(
        entry,
        official_channel_id=context.manifest.official_channel_id,
        minimum_duration_seconds=Fraction(
            annotation.max_annotated_frame,
            context.manifest.expected_fps,
        ),
        yt_dlp=yt_dlp,
        command_runner=command_runner,
    )
    with tempfile.TemporaryDirectory(prefix=f".shuttleset22-{match_id:02d}-", dir=downloads) as text:
        temporary_root = Path(text)
        output_template = temporary_root / "source.%(ext)s"
        command = download_command(entry, output_template, yt_dlp)
        if command_runner is None:
            print(f"running: {shlex.join(command)}", flush=True)
        completed = (command_runner or _run_download_command)(command)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "no diagnostic output").strip()
            raise RuntimeError(f"yt-dlp download command failed: {detail}")
        candidates = [path for path in temporary_root.iterdir() if path.suffix == ".mp4"]
        if len(candidates) != 1 or candidates[0].is_symlink() or not candidates[0].is_file():
            raise ValueError(f"yt-dlp must produce one regular MP4 file, got {candidates}")
        temporary = candidates[0]
        temporary_resolved = temporary.resolve(strict=True)
        metadata = metadata_probe(temporary_resolved)
        if metadata.source_path != temporary_resolved:
            raise ValueError(
                f"download metadata path {metadata.source_path} != {temporary_resolved}"
            )
        _validate_media(context, match_id, metadata)
        os.replace(temporary, destination)

    published_metadata = replace(metadata, source_path=destination.resolve(strict=True))
    return resolve_sources(
        context,
        source_root=downloads,
        overlap_root=overlap_root,
        video_ids=[match_id],
        metadata_probe=lambda _path: published_metadata,
    )[0]


def _validate_media(context: SourceContext, match_id: int, metadata: VideoMetadata) -> None:
    if metadata.fps != context.manifest.expected_fps:
        raise ValueError(f"source {match_id} FPS {metadata.fps} != {context.manifest.expected_fps}")
    maximum = context.annotations.by_id()[match_id].max_annotated_frame
    if metadata.frame_count <= maximum:
        raise ValueError(
            f"source {match_id} has {metadata.frame_count} frames but annotation requires {maximum}"
        )


def _source_receipt_payload(
    context: SourceContext,
    source: ResolvedSource,
) -> dict[str, object]:
    entry = source.entry
    annotation = source.annotations
    source_metadata = source.metadata.to_dict()
    source_metadata["source_path"] = source.metadata.source_path.name
    source_integrity = source.integrity.to_dict()
    source_integrity["path"] = source.metadata.source_path.name
    return {
        "schema": SOURCE_RECEIPT_SCHEMA,
        "source_manifest": {
            "schema": context.manifest.schema,
            "upstream_commit": context.manifest.upstream_commit,
            "dataset_path": context.manifest.dataset_path,
            "annotation_sha256": context.manifest.annotation_sha256,
            "expected_fps": str(context.manifest.expected_fps),
            "official_channel_id": context.manifest.official_channel_id,
        },
        "entry": {
            "id": entry.match_id,
            "video": entry.video,
            "source_kind": entry.kind.value,
            "url": entry.url,
            "youtube_id": entry.youtube_id,
            "overlap_shuttleset_id": entry.overlap_shuttleset_id,
            "unresolved_reason": entry.unresolved_reason,
        },
        "annotations": {
            "stroke_count": annotation.stroke_count,
            "max_annotated_frame": annotation.max_annotated_frame,
        },
        "source_metadata": source_metadata,
        "source_integrity": source_integrity,
    }


def _source_url(entry: SourceEntry) -> str:
    if entry.kind not in {SourceKind.DOWNLOAD, SourceKind.SHUTTLESET_OVERLAP} or entry.url is None:
        raise ValueError(f"source {entry.match_id} has no pinned public URL")
    return entry.url


def _compatible_video_format(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    fps = payload.get("fps")
    height = payload.get("height")
    return (
        payload.get("ext") == "mp4"
        and payload.get("vcodec") not in {None, "none"}
        and not isinstance(fps, bool)
        and isinstance(fps, (int, float))
        and Fraction(str(fps)) == 30
        and not isinstance(height, bool)
        and isinstance(height, int)
        and 0 < height <= 1080
    )


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _run_download_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, check=False)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result
