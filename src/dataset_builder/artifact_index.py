"""Per-video index over production dataset-builder stage artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from annotator.video_metadata import VideoMetadata
from dataset_builder.fixed_sources import FixedSourceManifest, ResolvedFixedSource
from dataset_builder.manifest import artifact_integrity, run_manifest_sha256
from dataset_builder.models import ArtifactIntegrity, RunManifest, StageOutcome, StageRecord
from dataset_builder.records import SourceReference
from dataset_builder.vision import load_json_gz, save_json_gz


VIDEO_ARTIFACT_INDEX_SCHEMA = "dataset-builder-video-artifact-index/1"
VIDEO_ARTIFACT_INDEX_DIRECTORY = "artifact_index"
VIDEO_ARTIFACT_INDEX_FILENAME = "video_artifact_index.json.gz"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_GLOBAL_STAGE_NAMES = ("download", "commentary_cleaning")
_VIDEO_STAGE_BASES = (
    "metadata",
    "tracknet_input",
    "shuttle",
    "pose",
    "court",
    "annotation",
    "commentary_pairing",
    "primitive_projection",
)
_REPLAY_REQUIRED_STAGES = ("metadata", "tracknet_input", "shuttle", "pose", "court")
_MODEL_BOUND_STAGE_BASES = frozenset({"shuttle", "court"})


@dataclass(frozen=True)
class VideoArtifactIndex:
    """Reloadable source and stage records for one exact video."""

    run_id: str
    input_manifest_sha256: str
    source_dataset: str
    video_id: str
    source_manifest: ArtifactIntegrity
    source: ArtifactIntegrity
    source_reference: SourceReference
    metadata: VideoMetadata
    stages: tuple[StageRecord, ...]

    def __post_init__(self) -> None:
        for name in ("run_id", "source_dataset", "video_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"video artifact index {name} must be a non-empty string")
        if not _SHA256_PATTERN.fullmatch(self.input_manifest_sha256):
            raise ValueError("video artifact index manifest digest must be lowercase SHA-256")
        if self.source_manifest.name != "fixed_source_manifest":
            raise ValueError("video artifact index source manifest name differs")
        if self.source.name != f"source.{self.video_id}":
            raise ValueError("video artifact index source name differs from video_id")
        if self.source_reference.video_id != self.video_id:
            raise ValueError("video artifact index source reference differs from video_id")
        source_path = Path(self.source.path)
        if self.metadata.source_path != source_path:
            raise ValueError("video artifact index metadata differs from source identity")
        if self.source_reference.basename != source_path.name:
            raise ValueError("video artifact index source basename differs")
        self._validate_stages()

    def _validate_stages(self) -> None:
        names = tuple(stage.name for stage in self.stages)
        if len(names) != len(set(names)):
            raise ValueError("video artifact index contains duplicate stage records")
        allowed = set(_GLOBAL_STAGE_NAMES)
        allowed.update(f"{base}:{self.video_id}" for base in _VIDEO_STAGE_BASES)
        unknown = set(names) - allowed
        if unknown:
            raise ValueError(f"video artifact index contains unrelated stages: {sorted(unknown)}")
        if "download" not in names:
            raise ValueError("video artifact index must contain the download stage")
        expected_order = _artifact_stage_names(self.video_id)
        observed_order = tuple(name for name in expected_order if name in names)
        if names != observed_order:
            raise ValueError("video artifact index stage records are out of order")
        for stage in self.stages:
            counts = dict(stage.counts)
            if "frames" in counts and counts["frames"] != self.metadata.frame_count:
                raise ValueError(
                    f"video artifact index stage {stage.name!r} frame count differs from canonical metadata"
                )

    def stage(self, base_name: str) -> StageRecord | None:
        """Return one indexed global or video stage by base name."""
        name = base_name if base_name in _GLOBAL_STAGE_NAMES else f"{base_name}:{self.video_id}"
        return next((stage for stage in self.stages if stage.name == name), None)

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON-compatible index payload."""
        return {
            "schema": VIDEO_ARTIFACT_INDEX_SCHEMA,
            "run_id": self.run_id,
            "input_manifest_sha256": self.input_manifest_sha256,
            "source_dataset": self.source_dataset,
            "video_id": self.video_id,
            "source_manifest": self.source_manifest.to_dict(),
            "source": self.source.to_dict(),
            "source_reference": self.source_reference.to_dict(),
            "metadata": self.metadata.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
        }


def artifact_index_path(run_dir: Path, video_id: str) -> Path:
    """Return the standard per-video index path."""
    _validate_video_id(video_id)
    return (
        Path(run_dir)
        / "stages"
        / VIDEO_ARTIFACT_INDEX_DIRECTORY
        / video_id
        / VIDEO_ARTIFACT_INDEX_FILENAME
    )


def artifact_index_input_manifest(manifest: RunManifest) -> RunManifest:
    """Remove index and final-publication records from the bound snapshot."""
    stages = tuple(
        stage
        for stage in manifest.stages
        if not stage.name.startswith("artifact_index:")
        and stage.name not in {"assembly", "report"}
    )
    return manifest if stages == manifest.stages else RunManifest(
        run_id=manifest.run_id,
        created_at_utc=manifest.created_at_utc,
        stages=stages,
        schema=manifest.schema,
    )


def artifact_index_stage_records(
    manifest: RunManifest,
    video_id: str,
) -> tuple[StageRecord, ...]:
    """Select source-ordered production records relevant to one video."""
    _validate_video_id(video_id)
    by_name = {stage.name: stage for stage in artifact_index_input_manifest(manifest).stages}
    return tuple(
        by_name[name]
        for name in _artifact_stage_names(video_id)
        if name in by_name
    )


def write_video_artifact_index(
    path: Path,
    *,
    manifest: RunManifest,
    source_dataset: str,
    fixed_manifest: FixedSourceManifest,
    resolved_source: ResolvedFixedSource,
    source_reference: SourceReference,
) -> VideoArtifactIndex:
    """Project current production records into one versioned video index."""
    input_manifest = artifact_index_input_manifest(manifest)
    index = VideoArtifactIndex(
        run_id=input_manifest.run_id,
        input_manifest_sha256=run_manifest_sha256(input_manifest),
        source_dataset=source_dataset,
        video_id=resolved_source.entry.video_id,
        source_manifest=_source_manifest_integrity(fixed_manifest),
        source=resolved_source.source,
        source_reference=source_reference,
        metadata=resolved_source.metadata,
        stages=artifact_index_stage_records(input_manifest, resolved_source.entry.video_id),
    )
    save_json_gz(path, index.to_dict())
    return index


def load_video_artifact_index(
    path: Path,
    *,
    run_dir: Path,
    manifest: RunManifest,
    source_dataset: str,
    fixed_manifest: FixedSourceManifest,
    resolved_source: ResolvedFixedSource,
    source_reference: SourceReference,
    artifact_scope: str = "all",
    validate_models: bool = False,
) -> VideoArtifactIndex:
    """Load an index and prove its current run, source, and artifact binding."""
    payload = load_json_gz(path)
    index = _index_from_payload(payload)
    input_manifest = artifact_index_input_manifest(manifest)
    expected_stages = artifact_index_stage_records(input_manifest, index.video_id)
    if index.run_id != input_manifest.run_id:
        raise ValueError("video artifact index run identity differs")
    if index.input_manifest_sha256 != run_manifest_sha256(input_manifest):
        raise ValueError("video artifact index manifest identity differs")
    if index.source_dataset != source_dataset:
        raise ValueError("video artifact index source dataset differs")
    if index.video_id != resolved_source.entry.video_id:
        raise ValueError("video artifact index video identity differs")
    if index.source_manifest != _source_manifest_integrity(fixed_manifest):
        raise ValueError("video artifact index fixed-source manifest identity differs")
    if index.source != resolved_source.source:
        raise ValueError("video artifact index source identity differs")
    if index.source_reference != source_reference:
        raise ValueError("video artifact index source reference differs")
    if index.metadata != resolved_source.metadata:
        raise ValueError("video artifact index canonical metadata differs")
    if index.stages != expected_stages:
        raise ValueError("video artifact index stage records differ from the run manifest")
    if artifact_scope not in {"all", "replay", "none"}:
        raise ValueError(f"unsupported video artifact validation scope: {artifact_scope!r}")
    if artifact_scope != "none":
        stage_bases = _REPLAY_REQUIRED_STAGES if artifact_scope == "replay" else None
        _validate_stage_artifacts(index, Path(run_dir), stage_bases=stage_bases)
    if validate_models:
        _validate_model_identities(index)
    return index


def require_replayable_vision(index: VideoArtifactIndex) -> None:
    """Reject partial indexes that lack a complete expensive vision boundary."""
    for base_name in _REPLAY_REQUIRED_STAGES:
        stage = index.stage(base_name)
        if stage is None:
            raise ValueError(f"video artifact index is missing replay stage {base_name!r}")
        if stage.outcome is not StageOutcome.PROCESSED:
            raise ValueError(
                f"video artifact index replay stage {base_name!r} has outcome {stage.outcome.value!r}"
            )


def _index_from_payload(payload: object) -> VideoArtifactIndex:
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError("video artifact index must be a JSON object")
    expected = {
        "schema",
        "run_id",
        "input_manifest_sha256",
        "source_dataset",
        "video_id",
        "source_manifest",
        "source",
        "source_reference",
        "metadata",
        "stages",
    }
    if set(payload) != expected:
        raise ValueError("video artifact index fields differ")
    if payload["schema"] != VIDEO_ARTIFACT_INDEX_SCHEMA:
        raise ValueError(f"unsupported video artifact index schema: {payload['schema']!r}")
    stages = payload["stages"]
    if not isinstance(stages, list):
        raise ValueError("video artifact index stages must be a list")
    return VideoArtifactIndex(
        run_id=_string(payload["run_id"], "run_id"),
        input_manifest_sha256=_string(payload["input_manifest_sha256"], "input_manifest_sha256"),
        source_dataset=_string(payload["source_dataset"], "source_dataset"),
        video_id=_string(payload["video_id"], "video_id"),
        source_manifest=ArtifactIntegrity.from_dict(payload["source_manifest"]),
        source=ArtifactIntegrity.from_dict(payload["source"]),
        source_reference=_source_reference(payload["source_reference"]),
        metadata=VideoMetadata.from_dict(payload["metadata"]),
        stages=tuple(StageRecord.from_dict(stage) for stage in stages),
    )


def _validate_stage_artifacts(
    index: VideoArtifactIndex,
    run_dir: Path,
    *,
    stage_bases: tuple[str, ...] | None,
) -> None:
    root = Path(run_dir).resolve(strict=True)
    for stage in index.stages:
        if stage_bases is not None and stage.name.partition(":")[0] not in stage_bases:
            continue
        for expected in stage.outputs:
            relative = Path(expected.path)
            if relative.is_absolute():
                raise ValueError(f"indexed output {expected.name!r} has an absolute path")
            observed = artifact_integrity(expected.name, root / relative, relative_to=root)
            if observed != expected:
                raise ValueError(f"indexed output {expected.name!r} integrity differs")


def _validate_model_identities(index: VideoArtifactIndex) -> None:
    for stage in index.stages:
        base_name = stage.name.partition(":")[0]
        if base_name not in _MODEL_BOUND_STAGE_BASES:
            continue
        for expected in stage.fingerprint.model_weights:
            observed = artifact_integrity(expected.name, Path(expected.path))
            if observed != expected:
                raise ValueError(f"indexed model {expected.name!r} integrity differs")


def _source_manifest_integrity(manifest: FixedSourceManifest) -> ArtifactIntegrity:
    return ArtifactIntegrity(
        name="fixed_source_manifest",
        path=str(manifest.path),
        md5=manifest.md5,
        size_bytes=manifest.size_bytes,
    )


def _source_reference(payload: object) -> SourceReference:
    if not isinstance(payload, dict) or set(payload) != {
        "video_id",
        "basename",
        "title",
        "url",
        "commentary_eligible",
    }:
        raise ValueError("video artifact index source reference fields differ")
    commentary_eligible = payload["commentary_eligible"]
    if not isinstance(commentary_eligible, bool):
        raise ValueError("video artifact index commentary_eligible must be boolean")
    return SourceReference(
        video_id=_string(payload["video_id"], "source_reference.video_id"),
        basename=_string(payload["basename"], "source_reference.basename"),
        title=_string(payload["title"], "source_reference.title"),
        url=_string(payload["url"], "source_reference.url"),
        commentary_eligible=commentary_eligible,
    )


def _artifact_stage_names(video_id: str) -> tuple[str, ...]:
    return (
        "download",
        f"metadata:{video_id}",
        "commentary_cleaning",
        *(f"{base}:{video_id}" for base in _VIDEO_STAGE_BASES[1:]),
    )


def _validate_video_id(video_id: object) -> None:
    if (
        not isinstance(video_id, str)
        or not video_id
        or video_id in {".", ".."}
        or "/" in video_id
        or "\\" in video_id
    ):
        raise ValueError("video artifact index video_id must be a path-safe basename")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"video artifact index {name} must be a non-empty string")
    return value
