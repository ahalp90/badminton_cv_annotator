"""Pinned ShuttleSet22 annotations, video sources, and overlap validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from fractions import Fraction
import hashlib
from pathlib import Path
import re
from statistics import median
import tomllib
from urllib.parse import parse_qs, urlparse

from annotator.video_metadata import VideoMetadata, probe_video_metadata
from dataset_builder.manifest import artifact_integrity
from dataset_builder.models import ArtifactIntegrity


SOURCE_MANIFEST_SCHEMA = "shuttleset22-sources/1"
EXPECTED_MATCH_IDS = tuple(range(1, 59))
EXPECTED_FPS = Fraction(30, 1)
EXPECTED_UPSTREAM_COMMIT = "45517f7d4cb936b03f3eabf939cc7959d39226fe"
OFFICIAL_DATASET_PATH = "CoachAI-Challenge-IJCAI2023/ShuttleSet22"
OFFICIAL_BWF_CHANNEL_ID = "UChh-akEbUM8_6ghGVnJd6cQ"
_FPS_RATIO_TOLERANCE = Fraction(1, 50)
_YOUTUBE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_TIME_PATTERN = re.compile(r"(?P<hours>\d+):(?P<minutes>\d{2}):(?P<seconds>\d{2})\Z")


class SourceKind(StrEnum):
    """How a source enters the ShuttleSet22 corpus."""

    DOWNLOAD = "download"
    SHUTTLESET_OVERLAP = "shuttleset_overlap"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class AnnotationMatch:
    """One official ShuttleSet22 match and its contact-frame bounds."""

    match_id: int
    video: str
    tournament: str
    round_name: str
    winner: str
    loser: str
    stroke_count: int
    max_annotated_frame: int
    annotation_files: tuple[Path, ...]


@dataclass(frozen=True)
class AnnotationCorpus:
    """The exact 58-match official annotation set."""

    root: Path
    matches: tuple[AnnotationMatch, ...]

    def by_id(self) -> dict[int, AnnotationMatch]:
        """Return a new match-ID lookup."""
        return {match.match_id: match for match in self.matches}


@dataclass(frozen=True)
class SourceEntry:
    """One pinned public source for an official match."""

    match_id: int
    video: str
    kind: SourceKind
    url: str | None = None
    youtube_id: str | None = None
    overlap_shuttleset_id: int | None = None
    unresolved_reason: str | None = None

    @property
    def canonical_filename(self) -> str:
        """Return the published ShuttleSet22 raw-video filename."""
        return f"{self.match_id:02d} {self.video}.mp4"


@dataclass(frozen=True)
class SourceManifest:
    """Versioned public-source contract for all 58 matches."""

    schema: str
    upstream_commit: str
    dataset_path: str
    annotation_sha256: str
    expected_fps: Fraction
    official_channel_id: str
    videos: tuple[SourceEntry, ...]

    def by_id(self) -> dict[int, SourceEntry]:
        """Return a new source-ID lookup."""
        return {entry.match_id: entry for entry in self.videos}


@dataclass(frozen=True)
class SourceContext:
    """Annotations and sources after cross-dataset validation."""

    annotations: AnnotationCorpus
    manifest: SourceManifest


@dataclass(frozen=True)
class ResolvedSource:
    """A source file that passed identity, timing, and integrity checks."""

    entry: SourceEntry
    annotations: AnnotationMatch
    metadata: VideoMetadata
    integrity: ArtifactIntegrity


def load_source_context(
    manifest_path: Path,
    dataset_root: Path,
    shuttleset_match_csv: Path,
) -> SourceContext:
    """Load and cross-check the pinned source and annotation contracts.

    :param manifest_path: Versioned ShuttleSet22 source TOML.
    :param dataset_root: Official ShuttleSet22 directory at the pinned commit.
    :param shuttleset_match_csv: Existing ShuttleSet `match.csv` for overlap identity.
    :return: Fully cross-validated immutable context.
    """
    manifest = load_source_manifest(manifest_path)
    actual_annotation_sha256 = annotation_corpus_sha256(dataset_root)
    if actual_annotation_sha256 != manifest.annotation_sha256:
        raise ValueError(
            f"ShuttleSet22 annotation SHA-256 {actual_annotation_sha256} "
            f"!= {manifest.annotation_sha256}"
        )
    annotations = load_annotation_corpus(dataset_root)
    _validate_manifest_matches(manifest, annotations)
    _validate_overlap_contract(manifest, annotations, shuttleset_match_csv)
    return SourceContext(annotations=annotations, manifest=manifest)


def load_annotation_corpus(dataset_root: Path) -> AnnotationCorpus:
    """Load the official 58-match annotation contract and infer its frame rate."""
    root = _existing_directory(dataset_root, "ShuttleSet22 dataset root")
    set_root = _existing_directory(root / "set", "ShuttleSet22 set directory")
    rows = _read_csv_rows(set_root / "match.csv", "ShuttleSet22 match.csv")
    match_ids = [_integer(row.get("id"), "match.id") for row in rows]
    _require_exact_match_ids(match_ids, "annotation match IDs")

    matches: list[AnnotationMatch] = []
    seen_videos: set[str] = set()
    for row in rows:
        match_id = _integer(row.get("id"), "match.id")
        video = _nonempty(row.get("video"), f"match[{match_id}].video")
        if video in seen_videos:
            raise ValueError(f"duplicate ShuttleSet22 annotation video: {video}")
        seen_videos.add(video)
        match_dir = _existing_directory(set_root / video, f"match {match_id} annotation directory")
        annotation_files = tuple(sorted(match_dir.glob("set*.csv")))
        if not annotation_files or any(not path.is_file() for path in annotation_files):
            raise ValueError(f"match {match_id} has no set annotation files: {match_dir}")
        stroke_count, max_frame, inferred_fps = _annotation_frame_contract(
            annotation_files,
            match_id,
        )
        if abs(inferred_fps - EXPECTED_FPS) > _FPS_RATIO_TOLERANCE:
            raise ValueError(
                f"match {match_id} annotations infer {float(inferred_fps):.6f} FPS, "
                f"expected {EXPECTED_FPS}"
            )
        matches.append(
            AnnotationMatch(
                match_id=match_id,
                video=video,
                tournament=_nonempty(row.get("tournament"), f"match[{match_id}].tournament"),
                round_name=_nonempty(row.get("round"), f"match[{match_id}].round"),
                winner=_nonempty(row.get("winner"), f"match[{match_id}].winner"),
                loser=_nonempty(row.get("loser"), f"match[{match_id}].loser"),
                stroke_count=stroke_count,
                max_annotated_frame=max_frame,
                annotation_files=annotation_files,
            )
        )
    return AnnotationCorpus(root=root, matches=tuple(sorted(matches, key=lambda match: match.match_id)))


def annotation_corpus_sha256(dataset_root: Path) -> str:
    """Hash the annotation paths and bytes consumed by the source contract."""
    root = _existing_directory(dataset_root, "ShuttleSet22 dataset root")
    set_root = _existing_directory(root / "set", "ShuttleSet22 set directory")
    paths = (set_root / "match.csv", *sorted(set_root.glob("*/set*.csv")))
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise FileNotFoundError("ShuttleSet22 annotations must be regular non-symlink files")

    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def load_source_manifest(path: Path) -> SourceManifest:
    """Load the strict versioned ShuttleSet22 source TOML."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source manifest is not a regular file: {manifest_path}")
    with manifest_path.open("rb") as handle:
        payload = tomllib.load(handle)
    _require_exact_keys(
        payload,
        {
            "schema",
            "upstream_commit",
            "dataset_path",
            "annotation_sha256",
            "expected_fps",
            "official_channel_id",
            "videos",
        },
        "source manifest",
    )
    videos_payload = payload["videos"]
    if not isinstance(videos_payload, list):
        raise ValueError("source manifest videos must be an array of tables")
    videos = tuple(_source_entry(value, index) for index, value in enumerate(videos_payload))
    _require_exact_match_ids([entry.match_id for entry in videos], "source manifest IDs")
    _require_unique([entry.video for entry in videos], "source manifest video names")
    _require_unique(
        [entry.url for entry in videos if entry.url is not None],
        "source manifest URLs",
    )
    _require_unique(
        [entry.youtube_id for entry in videos if entry.youtube_id is not None],
        "source manifest YouTube IDs",
    )

    manifest = SourceManifest(
        schema=_nonempty(payload["schema"], "source manifest schema"),
        upstream_commit=_nonempty(payload["upstream_commit"], "source manifest upstream_commit"),
        dataset_path=_nonempty(payload["dataset_path"], "source manifest dataset_path"),
        annotation_sha256=_sha256(
            payload["annotation_sha256"],
            "source manifest annotation_sha256",
        ),
        expected_fps=_fraction(payload["expected_fps"], "source manifest expected_fps"),
        official_channel_id=_nonempty(
            payload["official_channel_id"],
            "source manifest official_channel_id",
        ),
        videos=tuple(sorted(videos, key=lambda entry: entry.match_id)),
    )
    expected_header = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "dataset_path": OFFICIAL_DATASET_PATH,
        "expected_fps": EXPECTED_FPS,
        "official_channel_id": OFFICIAL_BWF_CHANNEL_ID,
    }
    actual_header = {
        "schema": manifest.schema,
        "upstream_commit": manifest.upstream_commit,
        "dataset_path": manifest.dataset_path,
        "expected_fps": manifest.expected_fps,
        "official_channel_id": manifest.official_channel_id,
    }
    if actual_header != expected_header:
        raise ValueError(f"source manifest header {actual_header!r} != {expected_header!r}")
    return manifest


def resolve_sources(
    context: SourceContext,
    *,
    source_root: Path,
    overlap_root: Path,
    video_ids: Sequence[int] | None = None,
    metadata_probe: Callable[[Path], VideoMetadata] = probe_video_metadata,
) -> tuple[ResolvedSource, ...]:
    """Resolve, validate, and hash selected source files before GPU work.

    :param context: Cross-validated source and annotation context.
    :param source_root: Directory containing downloaded ShuttleSet22 sources.
    :param overlap_root: Directory containing the existing ShuttleSet sources.
    :param video_ids: Optional unique subset in requested order.
    :param metadata_probe: Injectable strict video metadata reader.
    :return: Resolved sources in requested order.
    """
    downloads = _existing_directory(source_root, "ShuttleSet22 source root")
    overlaps = _existing_directory(overlap_root, "ShuttleSet overlap root")
    requested = list(EXPECTED_MATCH_IDS if video_ids is None else video_ids)
    if len(requested) != len(set(requested)):
        raise ValueError(f"requested ShuttleSet22 IDs contain duplicates: {requested}")
    unknown = sorted(set(requested).difference(EXPECTED_MATCH_IDS))
    if unknown:
        raise ValueError(f"requested ShuttleSet22 IDs are unknown: {unknown}")

    entries = context.manifest.by_id()
    annotations = context.annotations.by_id()
    resolved: list[ResolvedSource] = []
    digests: dict[str, int] = {}
    for match_id in requested:
        entry = entries[match_id]
        source_path = _source_path(entry, downloads, overlaps)
        if source_path.is_symlink() or not source_path.is_file():
            raise FileNotFoundError(f"source must be a regular non-symlink file: {source_path}")
        resolved_path = source_path.resolve(strict=True)
        metadata = metadata_probe(resolved_path)
        if metadata.source_path != resolved_path:
            raise ValueError(
                f"source {match_id} metadata path {metadata.source_path} "
                f"!= {resolved_path}"
            )
        if metadata.fps != context.manifest.expected_fps:
            raise ValueError(
                f"source {match_id} FPS {metadata.fps} != {context.manifest.expected_fps}"
            )
        match = annotations[match_id]
        if metadata.frame_count <= match.max_annotated_frame:
            raise ValueError(
                f"source {match_id} has {metadata.frame_count} frames but annotation "
                f"requires frame {match.max_annotated_frame}"
            )
        integrity = artifact_integrity(
            f"shuttleset22_source_{match_id:02d}",
            resolved_path,
        )
        duplicate_id = digests.get(integrity.md5)
        if duplicate_id is not None:
            raise ValueError(
                f"source content is duplicated between matches {duplicate_id} and {match_id}"
            )
        digests[integrity.md5] = match_id
        resolved.append(
            ResolvedSource(
                entry=entry,
                annotations=match,
                metadata=metadata,
                integrity=integrity,
            )
        )
    return tuple(resolved)


def _annotation_frame_contract(paths: tuple[Path, ...], match_id: int) -> tuple[int, int, Fraction]:
    frame_time_ratios: list[Fraction] = []
    stroke_count = 0
    maximum = -1
    for path in paths:
        rows = _read_csv_rows(path, f"match {match_id} annotations")
        for row_index, row in enumerate(rows, start=2):
            frame = _integer(row.get("frame_num"), f"{path.name}:{row_index}.frame_num")
            if frame < 0:
                raise ValueError(f"annotation frame must be non-negative: {path}:{row_index}")
            seconds = _time_seconds(row.get("time"), f"{path.name}:{row_index}.time")
            if seconds > 0:
                frame_time_ratios.append(Fraction(frame, seconds))
            maximum = max(maximum, frame)
            stroke_count += 1
    if not frame_time_ratios or maximum < 0:
        raise ValueError(f"match {match_id} has no usable annotation frame-time pairs")
    return stroke_count, maximum, median(frame_time_ratios)


def _source_entry(value: object, index: int) -> SourceEntry:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"source manifest videos[{index}] must be a table")
    base_keys = {"id", "video", "source_kind"}
    kind = SourceKind(_nonempty(value.get("source_kind"), f"videos[{index}].source_kind"))
    source_keys = {
        SourceKind.DOWNLOAD: {"url", "youtube_id"},
        SourceKind.SHUTTLESET_OVERLAP: {"url", "youtube_id", "overlap_shuttleset_id"},
        SourceKind.UNRESOLVED: {"unresolved_reason"},
    }
    expected_keys = base_keys | source_keys[kind]
    _require_exact_keys(value, expected_keys, f"source manifest videos[{index}]")
    match_id = _integer(value["id"], f"videos[{index}].id")
    video = _nonempty(value["video"], f"videos[{index}].video")
    if Path(video).name != video or video in {".", ".."}:
        raise ValueError(f"videos[{index}].video must be a basename: {video!r}")
    youtube_id = None
    url = None
    overlap_id = None
    unresolved_reason = None
    if kind in {SourceKind.DOWNLOAD, SourceKind.SHUTTLESET_OVERLAP}:
        youtube_id = _nonempty(value["youtube_id"], f"videos[{index}].youtube_id")
        if not _YOUTUBE_ID_PATTERN.fullmatch(youtube_id):
            raise ValueError(f"videos[{index}].youtube_id is invalid: {youtube_id!r}")
        url = _youtube_url(value["url"], youtube_id, f"videos[{index}].url")
    if kind is SourceKind.SHUTTLESET_OVERLAP:
        overlap_id = _integer(
            value["overlap_shuttleset_id"],
            f"videos[{index}].overlap_shuttleset_id",
        )
    elif kind is SourceKind.UNRESOLVED:
        unresolved_reason = _nonempty(
            value["unresolved_reason"],
            f"videos[{index}].unresolved_reason",
        )
    return SourceEntry(
        match_id=match_id,
        video=video,
        kind=kind,
        url=url,
        youtube_id=youtube_id,
        overlap_shuttleset_id=overlap_id,
        unresolved_reason=unresolved_reason,
    )


def _validate_manifest_matches(manifest: SourceManifest, corpus: AnnotationCorpus) -> None:
    annotations = corpus.by_id()
    for entry in manifest.videos:
        expected_video = annotations[entry.match_id].video
        if entry.video != expected_video:
            raise ValueError(
                f"source manifest video {entry.match_id} {entry.video!r} != {expected_video!r}"
            )


def _validate_overlap_contract(
    manifest: SourceManifest,
    corpus: AnnotationCorpus,
    shuttleset_match_csv: Path,
) -> None:
    rows = _read_csv_rows(shuttleset_match_csv, "ShuttleSet match.csv")
    old_by_video: dict[str, tuple[int, str]] = {}
    for row in rows:
        old_id = _integer(row.get("id"), "ShuttleSet match.id")
        video = _nonempty(row.get("video"), f"ShuttleSet match[{old_id}].video")
        url = _nonempty(row.get("url"), f"ShuttleSet match[{old_id}].url")
        if video in old_by_video:
            raise ValueError(f"duplicate ShuttleSet video name: {video}")
        old_by_video[video] = (old_id, url)

    annotations = corpus.by_id()
    for entry in manifest.videos:
        overlap = old_by_video.get(annotations[entry.match_id].video)
        if overlap is None:
            if entry.kind is SourceKind.SHUTTLESET_OVERLAP or entry.overlap_shuttleset_id is not None:
                raise ValueError(f"source {entry.match_id} is marked overlap but has no ShuttleSet match")
            continue
        old_id, old_url = overlap
        if entry.kind is not SourceKind.SHUTTLESET_OVERLAP:
            raise ValueError(f"source {entry.match_id} must be marked as a ShuttleSet overlap")
        if entry.overlap_shuttleset_id != old_id:
            raise ValueError(
                f"source {entry.match_id} overlap ID {entry.overlap_shuttleset_id} != {old_id}"
            )
        if entry.url != old_url:
            raise ValueError(f"source {entry.match_id} overlap URL differs from ShuttleSet")


def _source_path(entry: SourceEntry, source_root: Path, overlap_root: Path) -> Path:
    if entry.kind is SourceKind.DOWNLOAD:
        return source_root / entry.canonical_filename
    if entry.kind is SourceKind.UNRESOLVED:
        raise ValueError(
            f"source {entry.match_id} is unresolved: {entry.unresolved_reason}"
        )
    if entry.overlap_shuttleset_id is None:
        raise ValueError(f"overlap source {entry.match_id} has no ShuttleSet ID")
    return overlap_root / f"{entry.overlap_shuttleset_id} {entry.video}.mp4"


def _read_csv_rows(path: Path, name: str) -> list[dict[str, str]]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{name} is not a regular file: {path}")
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{name} has no data rows: {path}")
    return rows


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if not decimal.is_finite() or decimal != decimal.to_integral_value():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(decimal)


def _time_seconds(value: object, name: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be H+:MM:SS, got {value!r}")
    match = _TIME_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"{name} must be H+:MM:SS, got {value!r}")
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"{name} must be H+:MM:SS, got {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def _youtube_url(value: object, youtube_id: str, name: str) -> str:
    url = _nonempty(value, name)
    parsed = urlparse(url)
    query_id = parse_qs(parsed.query).get("v", [None])[0]
    if parsed.scheme != "https" or parsed.hostname not in {"www.youtube.com", "youtube.com"}:
        raise ValueError(f"{name} must be an HTTPS YouTube watch URL: {url!r}")
    if parsed.path != "/watch" or query_id != youtube_id:
        raise ValueError(f"{name} does not identify YouTube video {youtube_id}: {url!r}")
    return f"https://www.youtube.com/watch?v={youtube_id}"


def _fraction(value: object, name: str) -> Fraction:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a fraction string, got {value!r}")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} must be a fraction string, got {value!r}") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters, got {value!r}")
    return value


def _existing_directory(path: Path, name: str) -> Path:
    candidate = Path(path)
    if not candidate.is_dir():
        raise FileNotFoundError(f"{name} is not a directory: {candidate}")
    return candidate.resolve(strict=True)


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value.strip()


def _require_exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_exact_match_ids(values: Sequence[int], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contain duplicates: {values}")
    if tuple(sorted(values)) != EXPECTED_MATCH_IDS:
        missing = sorted(set(EXPECTED_MATCH_IDS).difference(values))
        extra = sorted(set(values).difference(EXPECTED_MATCH_IDS))
        raise ValueError(f"{name} differ: missing={missing}, extra={extra}")


def _require_unique(values: Sequence[str], name: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"{name} contain duplicates: {duplicates}")
