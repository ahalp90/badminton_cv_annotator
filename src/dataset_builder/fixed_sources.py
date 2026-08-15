"""Strict fixed-source contracts for known ShuttleSet videos."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
import re
import tomllib
from urllib.parse import urlparse

from annotator.video_metadata import VideoMetadata, probe_video_metadata
from dataset_builder.manifest import artifact_integrity
from dataset_builder.models import ArtifactIntegrity


FIXED_SOURCE_MANIFEST_SCHEMA = "dataset-builder-fixed-sources/1"
FIXED_SOURCE_DATASET = "ShuttleSet"
FIXED_ACQUISITION_SCHEMA = "dataset-builder-fixed-acquisition/1"
FIXED_ACQUISITION_FILENAME = "fixed_acquisition.json.gz"
SUPPORTED_SOURCE_SUFFIXES = frozenset({".mp4"})
_MD5_PATTERN = re.compile(r"[0-9a-f]{32}\Z")

MetadataProbe = Callable[[Path], VideoMetadata]


@dataclass(frozen=True)
class GroundTruthMapping:
    """Explicit mapping from one production video to ShuttleSet labels."""

    match_id: str
    annotation_directory: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.match_id, "ground truth match_id")
        _validate_relative_path(self.annotation_directory, "ground truth annotation_directory")


@dataclass(frozen=True)
class FixedSourceEntry:
    """Pinned source and eligibility contract for one exact video ID."""

    video_id: str
    source_id: str
    source_url: str
    source_basename: str
    source_available: bool
    source_md5: str | None
    fps: Fraction | None
    frame_count: int | None
    ground_truth: GroundTruthMapping
    eligible: bool
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_video_id(self.video_id)
        _require_nonempty_string(self.source_id, "fixed source source_id")
        _validate_source_url(self.source_url)
        _validate_source_basename(self.source_basename)
        if not isinstance(self.source_available, bool):
            raise ValueError("fixed source source_available must be boolean")
        if not isinstance(self.ground_truth, GroundTruthMapping):
            raise ValueError("fixed source ground_truth must be a GroundTruthMapping")
        if self.ground_truth.match_id != self.source_id:
            raise ValueError("fixed source ground truth match_id must equal source_id")
        if not isinstance(self.eligible, bool):
            raise ValueError("fixed source eligible must be boolean")
        self._validate_source_metadata()
        self._validate_eligibility()

    def _validate_source_metadata(self) -> None:
        values = (self.source_md5, self.fps, self.frame_count)
        if not self.source_available:
            if any(value is not None for value in values):
                raise ValueError("unavailable fixed source must omit source_md5, fps, and frame_count")
            return
        if self.source_md5 is None or not _MD5_PATTERN.fullmatch(self.source_md5):
            raise ValueError("available fixed source source_md5 must be 32 lowercase hexadecimal characters")
        if not isinstance(self.fps, Fraction) or self.fps <= 0:
            raise ValueError("available fixed source fps must be a positive Fraction")
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int) or self.frame_count <= 0:
            raise ValueError("available fixed source frame_count must be a positive integer")

    def _validate_eligibility(self) -> None:
        if self.eligible:
            if not self.source_available:
                raise ValueError("eligible fixed source must be available")
            if self.exclusion_reason is not None:
                raise ValueError("eligible fixed source must omit exclusion_reason")
            return
        _require_nonempty_string(self.exclusion_reason, "ineligible fixed source exclusion_reason")


@dataclass(frozen=True)
class FixedSourceManifest:
    """Loaded, versioned fixed-source manifest with its file identity."""

    path: Path
    md5: str
    size_bytes: int
    schema: str
    dataset: str
    videos: tuple[FixedSourceEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("fixed source manifest path must be absolute")
        if not _MD5_PATTERN.fullmatch(self.md5):
            raise ValueError("fixed source manifest md5 must be 32 lowercase hexadecimal characters")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ValueError("fixed source manifest size_bytes must be an integer")
        if self.size_bytes <= 0:
            raise ValueError("fixed source manifest size_bytes must be positive")
        if self.schema != FIXED_SOURCE_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported fixed source manifest schema: {self.schema!r}")
        if self.dataset != FIXED_SOURCE_DATASET:
            raise ValueError(f"unsupported fixed source dataset: {self.dataset!r}")
        if not self.videos:
            raise ValueError("fixed source manifest videos must not be empty")
        _validate_manifest_uniqueness(self.videos)

    def entries_by_video_id(self) -> dict[str, FixedSourceEntry]:
        """Return a fresh exact-string lookup for manifest entries."""
        return {entry.video_id: entry for entry in self.videos}


@dataclass(frozen=True)
class ResolvedFixedSource:
    """One source proved safe for the shared production vision stages."""

    entry: FixedSourceEntry
    source: ArtifactIntegrity
    metadata: VideoMetadata
    ground_truth_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.entry, FixedSourceEntry):
            raise ValueError("resolved fixed source entry must be a FixedSourceEntry")
        if not isinstance(self.source, ArtifactIntegrity):
            raise ValueError("resolved fixed source source must be ArtifactIntegrity")
        if not isinstance(self.metadata, VideoMetadata):
            raise ValueError("resolved fixed source metadata must be VideoMetadata")
        if not isinstance(self.ground_truth_directory, Path):
            raise ValueError("resolved fixed source ground_truth_directory must be a Path")
        if not self.ground_truth_directory.is_absolute():
            raise ValueError("resolved fixed source ground_truth_directory must be absolute")
        if not self.entry.eligible:
            raise ValueError("resolved fixed source entry must be eligible")
        if self.source.name != f"source.{self.entry.video_id}":
            raise ValueError("resolved fixed source artifact name differs from video_id")
        source_path = Path(self.source.path)
        if not source_path.is_absolute() or source_path.name != self.entry.source_basename:
            raise ValueError("resolved fixed source artifact path differs from source_basename")
        if self.source.md5 != self.entry.source_md5:
            raise ValueError("resolved fixed source artifact digest differs from the manifest")
        if self.metadata.source_path != source_path:
            raise ValueError("resolved fixed source metadata path differs from the source")
        if self.metadata.fps != self.entry.fps:
            raise ValueError("resolved fixed source metadata FPS differs from the manifest")
        if self.metadata.frame_count != self.entry.frame_count:
            raise ValueError("resolved fixed source metadata frame count differs from the manifest")


def load_fixed_source_manifest(path: Path) -> FixedSourceManifest:
    """Load a strict versioned TOML fixed-source manifest."""
    requested_path = Path(path)
    if requested_path.is_symlink():
        raise FileNotFoundError(f"fixed source manifest must not be a symlink: {requested_path}")
    integrity = artifact_integrity("fixed_source_manifest", requested_path)
    resolved_path = Path(integrity.path)
    with resolved_path.open("rb") as handle:
        payload = tomllib.load(handle)
    root = _object(payload, "fixed source manifest")
    _require_exact_keys(root, {"schema", "dataset", "videos"}, "fixed source manifest")
    videos_raw = root["videos"]
    if not isinstance(videos_raw, list):
        raise ValueError("fixed source manifest videos must be an array")
    videos = tuple(_parse_entry(value, index) for index, value in enumerate(videos_raw))
    return FixedSourceManifest(
        path=resolved_path,
        md5=integrity.md5,
        size_bytes=integrity.size_bytes,
        schema=_string(root["schema"], "fixed source manifest schema"),
        dataset=_string(root["dataset"], "fixed source manifest dataset"),
        videos=videos,
    )


def preflight_fixed_sources(
    manifest: FixedSourceManifest,
    *,
    source_root: Path,
    ground_truth_root: Path,
    requested_video_ids: Sequence[str],
    metadata_probe: MetadataProbe = probe_video_metadata,
) -> tuple[ResolvedFixedSource, ...]:
    """Validate a requested fixed subset completely before any GPU stage."""
    if not isinstance(manifest, FixedSourceManifest):
        raise TypeError("manifest must be FixedSourceManifest")
    selected = select_fixed_source_entries(manifest, requested_video_ids)

    resolved_source_root = _existing_directory(source_root, "fixed source root")
    resolved_ground_truth_root = _existing_directory(ground_truth_root, "fixed source ground truth root")
    sources = _resolve_source_artifacts(selected, resolved_source_root)
    ground_truth = _resolve_ground_truth(selected, resolved_ground_truth_root)

    resolved: list[ResolvedFixedSource] = []
    for entry in selected:
        source = sources[entry.video_id]
        metadata = metadata_probe(Path(source.path))
        _validate_probed_metadata(entry, source, metadata)
        resolved.append(
            ResolvedFixedSource(
                entry=entry,
                source=source,
                metadata=metadata,
                ground_truth_directory=ground_truth[entry.video_id],
            )
        )
    return tuple(resolved)


def select_fixed_source_entries(
    manifest: FixedSourceManifest,
    requested_video_ids: Sequence[str],
) -> tuple[FixedSourceEntry, ...]:
    """Resolve an explicit eligible subset without reading source files."""
    if not isinstance(manifest, FixedSourceManifest):
        raise TypeError("manifest must be FixedSourceManifest")
    video_ids = _requested_video_ids(requested_video_ids)
    entries = manifest.entries_by_video_id()
    selected: list[FixedSourceEntry] = []
    for video_id in video_ids:
        entry = entries.get(video_id)
        if entry is None:
            raise ValueError(f"requested fixed source video_id is unknown: {video_id!r}")
        if not entry.eligible:
            raise ValueError(f"requested fixed source video_id {video_id!r} is ineligible: {entry.exclusion_reason}")
        selected.append(entry)
    return tuple(selected)


def save_fixed_acquisition(
    path: Path,
    manifest: FixedSourceManifest,
    resolved_sources: Sequence[ResolvedFixedSource],
) -> Path:
    """Persist the fixed acquisition boundary for reload and resume."""
    from dataset_builder.vision import save_json_gz

    selected = tuple(resolved_sources)
    if not selected:
        raise ValueError("fixed acquisition must contain at least one source")
    video_ids = tuple(source.entry.video_id for source in selected)
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("fixed acquisition contains duplicate video IDs")
    videos: list[dict[str, object]] = []
    for source in selected:
        videos.append(
            {
                "video_id": source.entry.video_id,
                "source": source.source.to_dict(),
                "metadata": source.metadata.to_dict(),
                "ground_truth_directory": str(source.ground_truth_directory),
            }
        )
    return save_json_gz(
        path,
        {
            "schema": FIXED_ACQUISITION_SCHEMA,
            "manifest": {
                "path": str(manifest.path),
                "md5": manifest.md5,
                "size_bytes": manifest.size_bytes,
            },
            "videos": videos,
        },
    )


def load_fixed_acquisition(
    path: Path,
    manifest: FixedSourceManifest,
    *,
    source_root: Path,
    ground_truth_root: Path,
    requested_video_ids: Sequence[str],
    validate_source_integrity: bool,
) -> tuple[ResolvedFixedSource, ...]:
    """Reload a fixed acquisition and optionally hash its external sources."""
    from dataset_builder.vision import load_json_gz

    if not isinstance(validate_source_integrity, bool):
        raise TypeError("validate_source_integrity must be boolean")
    payload = _object(load_json_gz(path), "fixed acquisition")
    _require_exact_keys(payload, {"schema", "manifest", "videos"}, "fixed acquisition")
    if payload["schema"] != FIXED_ACQUISITION_SCHEMA:
        raise ValueError(f"unsupported fixed acquisition schema: {payload['schema']!r}")
    _validate_acquisition_manifest(payload["manifest"], manifest)
    videos = payload["videos"]
    if not isinstance(videos, list):
        raise ValueError("fixed acquisition videos must be a list")
    entries = select_fixed_source_entries(manifest, requested_video_ids)
    if len(videos) != len(entries):
        raise ValueError("fixed acquisition video count differs from the requested sources")
    resolved_source_root = _existing_directory(source_root, "fixed source root")
    resolved_ground_truth_root = _existing_directory(ground_truth_root, "fixed source ground truth root")
    resolved: list[ResolvedFixedSource] = []
    for raw_video, entry in zip(videos, entries, strict=True):
        resolved.append(
            _load_acquisition_source(
                raw_video,
                entry,
                resolved_source_root,
                resolved_ground_truth_root,
                validate_source_integrity=validate_source_integrity,
            )
        )
    return tuple(resolved)


def _validate_acquisition_manifest(
    payload: object,
    manifest: FixedSourceManifest,
) -> None:
    record = _object(payload, "fixed acquisition manifest")
    _require_exact_keys(
        record,
        {"path", "md5", "size_bytes"},
        "fixed acquisition manifest",
    )
    expected: dict[str, object] = {
        "path": str(manifest.path),
        "md5": manifest.md5,
        "size_bytes": manifest.size_bytes,
    }
    if dict(record) != expected:
        raise ValueError("fixed acquisition manifest identity differs")


def _load_acquisition_source(
    payload: object,
    entry: FixedSourceEntry,
    source_root: Path,
    ground_truth_root: Path,
    *,
    validate_source_integrity: bool,
) -> ResolvedFixedSource:
    name = f"fixed acquisition video {entry.video_id!r}"
    record = _object(payload, name)
    _require_exact_keys(
        record,
        {"video_id", "source", "metadata", "ground_truth_directory"},
        name,
    )
    if record["video_id"] != entry.video_id:
        raise ValueError(f"{name} identity differs")
    source = ArtifactIntegrity.from_dict(record["source"])
    expected_path = _fixed_source_path(entry, source_root)
    if source.name != f"source.{entry.video_id}" or source.path != str(expected_path):
        raise ValueError(f"{name} source identity differs")
    if source.md5 != entry.source_md5:
        raise ValueError(f"{name} source digest differs from the manifest")
    if validate_source_integrity:
        observed = artifact_integrity(source.name, expected_path)
        if observed != source:
            raise ValueError(f"{name} source integrity differs")
    metadata = VideoMetadata.from_dict(record["metadata"])
    ground_truth = _resolve_ground_truth((entry,), ground_truth_root)[entry.video_id]
    if record["ground_truth_directory"] != str(ground_truth):
        raise ValueError(f"{name} ground-truth identity differs")
    return ResolvedFixedSource(entry, source, metadata, ground_truth)


def _fixed_source_path(entry: FixedSourceEntry, source_root: Path) -> Path:
    candidate = source_root / entry.source_basename
    if candidate.is_symlink():
        raise FileNotFoundError(f"fixed source must not be a symlink: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"fixed source is not a regular file: {candidate}")
    source_path = candidate.resolve(strict=True)
    _require_within(source_path, source_root, "fixed source")
    return source_path


def _parse_entry(payload: object, index: int) -> FixedSourceEntry:
    name = f"fixed source manifest videos[{index}]"
    record = _object(payload, name)
    source_available = _boolean(record.get("source_available"), f"{name}.source_available")
    eligible = _boolean(record.get("eligible"), f"{name}.eligible")
    expected = {
        "video_id",
        "source_id",
        "source_url",
        "source_basename",
        "source_available",
        "ground_truth",
        "eligible",
    }
    if source_available:
        expected.update({"source_md5", "fps", "frame_count"})
    if not eligible:
        expected.add("exclusion_reason")
    _require_exact_keys(record, expected, name)
    source_md5 = _optional_string(record.get("source_md5"), f"{name}.source_md5")
    fps = _optional_fraction(record.get("fps"), f"{name}.fps")
    frame_count = _optional_positive_int(record.get("frame_count"), f"{name}.frame_count")
    reason = _optional_string(record.get("exclusion_reason"), f"{name}.exclusion_reason")
    return FixedSourceEntry(
        video_id=_string(record["video_id"], f"{name}.video_id"),
        source_id=_string(record["source_id"], f"{name}.source_id"),
        source_url=_string(record["source_url"], f"{name}.source_url"),
        source_basename=_string(record["source_basename"], f"{name}.source_basename"),
        source_available=source_available,
        source_md5=source_md5,
        fps=fps,
        frame_count=frame_count,
        ground_truth=_parse_ground_truth(record["ground_truth"], name),
        eligible=eligible,
        exclusion_reason=reason,
    )


def _parse_ground_truth(payload: object, parent_name: str) -> GroundTruthMapping:
    name = f"{parent_name}.ground_truth"
    record = _object(payload, name)
    _require_exact_keys(record, {"match_id", "annotation_directory"}, name)
    return GroundTruthMapping(
        match_id=_string(record["match_id"], f"{name}.match_id"),
        annotation_directory=_string(record["annotation_directory"], f"{name}.annotation_directory"),
    )


def _validate_manifest_uniqueness(videos: Sequence[FixedSourceEntry]) -> None:
    fields = {
        "video_id": tuple(entry.video_id for entry in videos),
        "source_id": tuple(entry.source_id for entry in videos),
        "source_url": tuple(entry.source_url for entry in videos),
        "source_basename": tuple(entry.source_basename for entry in videos),
        "ground_truth.annotation_directory": tuple(entry.ground_truth.annotation_directory for entry in videos),
        "source_md5": tuple(entry.source_md5 for entry in videos if entry.source_md5 is not None),
    }
    for name, values in fields.items():
        if len(values) != len(set(values)):
            raise ValueError(f"fixed source manifest contains duplicate {name} values")


def _requested_video_ids(requested: Sequence[str]) -> tuple[str, ...]:
    if isinstance(requested, (str, bytes)):
        raise ValueError("requested fixed source video IDs must be a sequence of strings")
    video_ids = tuple(requested)
    if not video_ids:
        raise ValueError("requested fixed source video IDs must not be empty")
    for video_id in video_ids:
        _require_nonempty_string(video_id, "requested fixed source video_id")
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("requested fixed source video IDs contain duplicates")
    return video_ids


def _resolve_source_artifacts(
    entries: Sequence[FixedSourceEntry],
    source_root: Path,
) -> dict[str, ArtifactIntegrity]:
    resolved: dict[str, ArtifactIntegrity] = {}
    paths: set[Path] = set()
    digests: set[str] = set()
    for entry in entries:
        source_path = _fixed_source_path(entry, source_root)
        source = artifact_integrity(f"source.{entry.video_id}", source_path)
        if source.md5 != entry.source_md5:
            raise ValueError(
                f"fixed source digest mismatch for {entry.video_id!r}: "
                f"observed={source.md5}, expected={entry.source_md5}"
            )
        if source_path in paths or source.md5 in digests:
            raise ValueError(f"fixed source input is duplicated for {entry.video_id!r}")
        paths.add(source_path)
        digests.add(source.md5)
        resolved[entry.video_id] = source
    return resolved


def _resolve_ground_truth(
    entries: Sequence[FixedSourceEntry],
    ground_truth_root: Path,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for entry in entries:
        relative = PurePosixPath(entry.ground_truth.annotation_directory)
        candidate = ground_truth_root.joinpath(*relative.parts)
        if candidate.is_symlink():
            raise FileNotFoundError(f"ground truth directory must not be a symlink: {candidate}")
        if not candidate.is_dir():
            raise FileNotFoundError(f"ground truth directory is missing: {candidate}")
        directory = candidate.resolve(strict=True)
        _require_within(directory, ground_truth_root, "ground truth directory")
        if not any(path.is_file() for path in directory.glob("*.csv")):
            raise FileNotFoundError(f"ground truth directory contains no CSV files: {directory}")
        resolved[entry.video_id] = directory
    return resolved


def _validate_probed_metadata(
    entry: FixedSourceEntry,
    source: ArtifactIntegrity,
    metadata: VideoMetadata,
) -> None:
    if not isinstance(metadata, VideoMetadata):
        raise TypeError(f"metadata probe returned unsupported value for {entry.video_id!r}")
    source_path = Path(source.path)
    if metadata.source_path != source_path:
        raise ValueError(
            f"fixed source identity mismatch for {entry.video_id!r}: "
            f"probe={metadata.source_path}, expected={source_path}"
        )
    if metadata.fps != entry.fps:
        raise ValueError(
            f"fixed source FPS mismatch for {entry.video_id!r}: observed={metadata.fps}, expected={entry.fps}"
        )
    if metadata.frame_count != entry.frame_count:
        raise ValueError(
            f"fixed source frame-count mismatch for {entry.video_id!r}: "
            f"observed={metadata.frame_count}, expected={entry.frame_count}"
        )


def _existing_directory(path: Path, name: str) -> Path:
    candidate = Path(path)
    if not candidate.is_dir():
        raise FileNotFoundError(f"{name} is not a directory: {candidate}")
    return candidate.resolve(strict=True)


def _require_within(path: Path, root: Path, name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} is outside its configured root: {path}") from exc


def _validate_source_basename(value: object) -> None:
    _require_nonempty_string(value, "fixed source source_basename")
    assert isinstance(value, str)
    path = Path(value)
    if path.name != value or value in {".", ".."}:
        raise ValueError(f"fixed source source_basename must be a basename: {value!r}")
    if path.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
        raise ValueError(f"fixed source format is unsupported: {value!r}")


def _validate_video_id(value: object) -> None:
    _require_nonempty_string(value, "fixed source video_id")
    assert isinstance(value, str)
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"fixed source video_id must be a path-safe basename: {value!r}")


def _validate_source_url(value: object) -> None:
    _require_nonempty_string(value, "fixed source source_url")
    assert isinstance(value, str)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"fixed source source_url is unsupported: {value!r}")


def _validate_relative_path(value: object, name: str) -> None:
    _require_nonempty_string(value, name)
    assert isinstance(value, str)
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise ValueError(f"{name} must be a relative POSIX path: {value!r}")
    if "\\" in value:
        raise ValueError(f"{name} must be a relative POSIX path: {value!r}")


def _object(payload: object, name: str) -> Mapping[str, object]:
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{name} must be a TOML table")
    return payload


def _require_exact_keys(record: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(record)
    if actual != expected:
        raise ValueError(f"{name} keys differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")


def _string(value: object, name: str) -> str:
    _require_nonempty_string(value, name)
    assert isinstance(value, str)
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _require_nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _optional_fraction(value: object, name: str) -> Fraction | None:
    if value is None:
        return None
    text = _string(value, name)
    try:
        fraction = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} must be a canonical positive fraction") from exc
    if fraction <= 0 or text != f"{fraction.numerator}/{fraction.denominator}":
        raise ValueError(f"{name} must be a canonical positive fraction")
    return fraction


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
