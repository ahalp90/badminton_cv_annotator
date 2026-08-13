"""Strict, reload-checked records for VLM scene benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, TypeVar

from annotator.broadcast_timeline_labels import SceneTruth


RUN_SCHEMA_VERSION = 2
SUPPORTED_RUN_SCHEMA_VERSIONS = (1, RUN_SCHEMA_VERSION)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
EnumType = TypeVar("EnumType", bound=StrEnum)


class RunOutcome(StrEnum):
    """Whether a backend produced a complete prediction record."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class BroadcastPhase(StrEnum):
    """Coarse broadcast state kept separate from the five-class scene label."""

    LIVE_RALLY = "live_rally"
    BETWEEN_RALLIES = "between_rallies"
    REPLAY = "replay"
    CUTAWAY = "cutaway"
    OTHER = "other"
    UNKNOWN = "unknown"


class CameraView(StrEnum):
    """Camera composition reported by the VLM."""

    FULL_COURT = "full_court"
    PARTIAL_COURT = "partial_court"
    SIDE_ON = "side_on"
    CLOSE_UP = "close_up"
    CROWD = "crowd"
    GRAPHIC = "graphic"
    OTHER = "other"
    UNKNOWN = "unknown"


class Playback(StrEnum):
    """Playback mode reported independently from the scene label."""

    REAL_TIME = "real_time"
    SLOW_MOTION = "slow_motion"
    FREEZE_FRAME = "freeze_frame"
    UNKNOWN = "unknown"


class Continuity(StrEnum):
    """Relationship to the preceding scene."""

    SAME_RALLY = "same_rally"
    NEW_RALLY = "new_rally"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class DataUse(StrEnum):
    """Suggested downstream treatment, retained as benchmark evidence only."""

    USABLE_STANDARD = "usable_standard"
    USABLE_ALTERNATE_VIEW = "usable_alternate_view"
    EXCLUDE = "exclude"
    REVIEW = "review"


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    found = set(value)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(f"{context} keys differ; missing={missing}, extra={extra}")


def _string(value: Any, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        suffix = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{context} must be {suffix}")
    return value


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return value


def _number(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return result


def _optional_number(value: Any, context: str, *, minimum: float = 0.0) -> float | None:
    return None if value is None else _number(value, context, minimum=minimum)


def _optional_integer(value: Any, context: str, *, minimum: int = 0) -> int | None:
    return None if value is None else _integer(value, context, minimum=minimum)


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be Boolean")
    return value


def _enum(value: Any, enum_type: type[EnumType], context: str) -> EnumType:
    raw = _string(value, context)
    try:
        return enum_type(raw)
    except ValueError as error:
        raise ValueError(f"{context} has unknown value {raw!r}") from error


def _sha256(value: Any, context: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    raw = _string(value, context)
    if not SHA256_PATTERN.fullmatch(raw):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return raw


@dataclass(frozen=True)
class ModelIdentity:
    """Pinned model and backend identity."""

    model_id: str
    model_revision: str
    backend: str
    backend_version: str

    @classmethod
    def from_json(cls, value: Any) -> ModelIdentity:
        data = _object(value, "model")
        _exact_keys(data, {"model_id", "model_revision", "backend", "backend_version"}, "model")
        return cls(
            model_id=_string(data["model_id"], "model.model_id"),
            model_revision=_string(data["model_revision"], "model.model_revision"),
            backend=_string(data["backend"], "model.backend"),
            backend_version=_string(data["backend_version"], "model.backend_version"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "backend": self.backend,
            "backend_version": self.backend_version,
        }


@dataclass(frozen=True)
class ShardSpec:
    """Original source, prepared input, and one absolute-frame range."""

    video_id: str
    source_file: str
    source_sha256: str
    prepared_input_file: str
    prepared_input_sha256: str
    fps: float
    frame_count: int
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        _string(self.video_id, "shard.video_id")
        _string(self.source_file, "shard.source_file")
        _sha256(self.source_sha256, "shard.source_sha256")
        _string(self.prepared_input_file, "shard.prepared_input_file")
        _sha256(self.prepared_input_sha256, "shard.prepared_input_sha256")
        _number(self.fps, "shard.fps", minimum=0.0)
        _integer(self.frame_count, "shard.frame_count", minimum=1)
        _integer(self.start_frame, "shard.start_frame", minimum=0)
        _integer(self.end_frame, "shard.end_frame", minimum=1)
        if not self.fps > 0:
            raise ValueError("shard.fps must be positive")
        if not 0 <= self.start_frame < self.end_frame <= self.frame_count:
            raise ValueError(
                f"shard range [{self.start_frame}, {self.end_frame}) is outside [0, {self.frame_count})"
            )
        if Path(self.source_file).name != self.source_file:
            raise ValueError("shard.source_file must be a file name without a directory")
        if Path(self.prepared_input_file).name != self.prepared_input_file:
            raise ValueError("shard.prepared_input_file must be a file name without a directory")

    @classmethod
    def from_json(cls, value: Any) -> ShardSpec:
        data = _object(value, "shard")
        _exact_keys(
            data,
            {
                "video_id",
                "source_file",
                "source_sha256",
                "prepared_input_file",
                "prepared_input_sha256",
                "fps",
                "frame_count",
                "start_frame",
                "end_frame",
            },
            "shard",
        )
        return cls(
            video_id=_string(data["video_id"], "shard.video_id"),
            source_file=_string(data["source_file"], "shard.source_file"),
            source_sha256=str(_sha256(data["source_sha256"], "shard.source_sha256")),
            prepared_input_file=_string(
                data["prepared_input_file"], "shard.prepared_input_file"
            ),
            prepared_input_sha256=str(
                _sha256(data["prepared_input_sha256"], "shard.prepared_input_sha256")
            ),
            fps=_number(data["fps"], "shard.fps", minimum=0.0),
            frame_count=_integer(data["frame_count"], "shard.frame_count", minimum=1),
            start_frame=_integer(data["start_frame"], "shard.start_frame", minimum=0),
            end_frame=_integer(data["end_frame"], "shard.end_frame", minimum=1),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "prepared_input_file": self.prepared_input_file,
            "prepared_input_sha256": self.prepared_input_sha256,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


@dataclass(frozen=True)
class SamplingRequest:
    """Sampling requested from either backend."""

    fps: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if _number(self.fps, "requested_sampling.fps", minimum=0.0) <= 0:
            raise ValueError("requested_sampling.fps must be positive")
        _integer(self.width, "requested_sampling.width", minimum=1)
        _integer(self.height, "requested_sampling.height", minimum=1)

    @classmethod
    def from_json(cls, value: Any) -> SamplingRequest:
        data = _object(value, "requested_sampling")
        _exact_keys(data, {"fps", "width", "height"}, "requested_sampling")
        return cls(
            fps=_number(data["fps"], "requested_sampling.fps", minimum=0.0),
            width=_integer(data["width"], "requested_sampling.width", minimum=1),
            height=_integer(data["height"], "requested_sampling.height", minimum=1),
        )

    def to_json(self) -> dict[str, Any]:
        return {"fps": self.fps, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class SamplingObservation:
    """Input grid and token evidence observed after backend processing."""

    sampled_source_frames: tuple[int, ...]
    width: int
    height: int
    visual_tokens: int | None
    total_input_tokens: int | None
    complete_source_coverage: bool
    uniform_frame_grid: bool

    def __post_init__(self) -> None:
        if not self.sampled_source_frames:
            raise ValueError("observed_sampling.sampled_source_frames must not be empty")
        previous: int | None = None
        for index, frame in enumerate(self.sampled_source_frames):
            _integer(frame, f"observed_sampling.sampled_source_frames[{index}]", minimum=0)
            if previous is not None and frame <= previous:
                raise ValueError("observed sampled source frames must be strictly increasing")
            previous = frame
        _integer(self.width, "observed_sampling.width", minimum=1)
        _integer(self.height, "observed_sampling.height", minimum=1)
        _optional_integer(self.visual_tokens, "observed_sampling.visual_tokens", minimum=1)
        _optional_integer(self.total_input_tokens, "observed_sampling.total_input_tokens", minimum=1)
        _boolean(self.complete_source_coverage, "observed_sampling.complete_source_coverage")
        _boolean(self.uniform_frame_grid, "observed_sampling.uniform_frame_grid")

    @classmethod
    def from_json(cls, value: Any) -> SamplingObservation:
        data = _object(value, "observed_sampling")
        _exact_keys(
            data,
            {
                "sampled_source_frames",
                "width",
                "height",
                "visual_tokens",
                "total_input_tokens",
                "complete_source_coverage",
                "uniform_frame_grid",
            },
            "observed_sampling",
        )
        frames = data["sampled_source_frames"]
        if not isinstance(frames, list):
            raise ValueError("observed_sampling.sampled_source_frames must be a JSON array")
        return cls(
            sampled_source_frames=tuple(
                _integer(frame, f"observed_sampling.sampled_source_frames[{index}]", minimum=0)
                for index, frame in enumerate(frames)
            ),
            width=_integer(data["width"], "observed_sampling.width", minimum=1),
            height=_integer(data["height"], "observed_sampling.height", minimum=1),
            visual_tokens=_optional_integer(data["visual_tokens"], "observed_sampling.visual_tokens", minimum=1),
            total_input_tokens=_optional_integer(
                data["total_input_tokens"], "observed_sampling.total_input_tokens", minimum=1
            ),
            complete_source_coverage=_boolean(
                data["complete_source_coverage"], "observed_sampling.complete_source_coverage"
            ),
            uniform_frame_grid=_boolean(data["uniform_frame_grid"], "observed_sampling.uniform_frame_grid"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "sampled_source_frames": list(self.sampled_source_frames),
            "width": self.width,
            "height": self.height,
            "visual_tokens": self.visual_tokens,
            "total_input_tokens": self.total_input_tokens,
            "complete_source_coverage": self.complete_source_coverage,
            "uniform_frame_grid": self.uniform_frame_grid,
        }


@dataclass(frozen=True)
class RuntimeTelemetry:
    """Hardware and elapsed-time evidence for one backend attempt."""

    hostname: str
    device_name: str
    peak_vram_mib: float | None
    elapsed_seconds: float
    cpu_offload: bool
    cache_dtype: str | None
    package_versions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _string(self.hostname, "runtime.hostname")
        _string(self.device_name, "runtime.device_name")
        _optional_number(self.peak_vram_mib, "runtime.peak_vram_mib")
        _number(self.elapsed_seconds, "runtime.elapsed_seconds", minimum=0.0)
        _boolean(self.cpu_offload, "runtime.cpu_offload")
        if self.cache_dtype is not None:
            _string(self.cache_dtype, "runtime.cache_dtype")
        names: set[str] = set()
        for name, version in self.package_versions:
            _string(name, "runtime.package_versions name")
            _string(version, f"runtime.package_versions[{name!r}]")
            if name in names:
                raise ValueError(f"runtime.package_versions repeats {name!r}")
            names.add(name)

    @classmethod
    def from_json(cls, value: Any) -> RuntimeTelemetry:
        data = _object(value, "runtime")
        _exact_keys(
            data,
            {
                "hostname",
                "device_name",
                "peak_vram_mib",
                "elapsed_seconds",
                "cpu_offload",
                "cache_dtype",
                "package_versions",
            },
            "runtime",
        )
        versions = _object(data["package_versions"], "runtime.package_versions")
        return cls(
            hostname=_string(data["hostname"], "runtime.hostname"),
            device_name=_string(data["device_name"], "runtime.device_name"),
            peak_vram_mib=_optional_number(data["peak_vram_mib"], "runtime.peak_vram_mib"),
            elapsed_seconds=_number(data["elapsed_seconds"], "runtime.elapsed_seconds", minimum=0.0),
            cpu_offload=_boolean(data["cpu_offload"], "runtime.cpu_offload"),
            cache_dtype=(None if data["cache_dtype"] is None else _string(data["cache_dtype"], "runtime.cache_dtype")),
            package_versions=tuple(
                sorted(
                    (_string(name, "runtime.package_versions name"), _string(version, f"version for {name!r}"))
                    for name, version in versions.items()
                )
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "device_name": self.device_name,
            "peak_vram_mib": self.peak_vram_mib,
            "elapsed_seconds": self.elapsed_seconds,
            "cpu_offload": self.cpu_offload,
            "cache_dtype": self.cache_dtype,
            "package_versions": dict(self.package_versions),
        }


@dataclass(frozen=True)
class PredictionSegment:
    """One validated scene prediction on absolute source frames."""

    start_frame: int
    end_frame: int
    scene_label: SceneTruth
    broadcast_phase: BroadcastPhase
    view: CameraView
    playback: Playback
    continuity_from_previous: Continuity
    data_use: DataUse
    confidence: float
    evidence_frames: tuple[int, ...]
    reason: str

    def __post_init__(self) -> None:
        _integer(self.start_frame, "segment.start_frame", minimum=0)
        _integer(self.end_frame, "segment.end_frame", minimum=1)
        if self.start_frame >= self.end_frame:
            raise ValueError("prediction segment must have start_frame < end_frame")
        if not isinstance(self.scene_label, SceneTruth):
            raise ValueError("segment.scene_label must be a SceneTruth")
        for name, value, enum_type in (
            ("broadcast_phase", self.broadcast_phase, BroadcastPhase),
            ("view", self.view, CameraView),
            ("playback", self.playback, Playback),
            ("continuity_from_previous", self.continuity_from_previous, Continuity),
            ("data_use", self.data_use, DataUse),
        ):
            if not isinstance(value, enum_type):
                raise ValueError(f"segment.{name} has the wrong enum type")
        confidence = _number(self.confidence, "segment.confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("segment.confidence must be in [0, 1]")
        for index, frame in enumerate(self.evidence_frames):
            _integer(frame, f"segment.evidence_frames[{index}]", minimum=0)
            if not self.start_frame <= frame < self.end_frame:
                raise ValueError(f"evidence frame {frame} is outside the prediction segment")
        _string(self.reason, "segment.reason")

    @classmethod
    def from_json(cls, value: Any, index: int) -> PredictionSegment:
        context = f"segments[{index}]"
        data = _object(value, context)
        _exact_keys(
            data,
            {
                "start_frame",
                "end_frame",
                "scene_label",
                "broadcast_phase",
                "view",
                "playback",
                "continuity_from_previous",
                "data_use",
                "confidence",
                "evidence_frames",
                "reason",
            },
            context,
        )
        evidence = data["evidence_frames"]
        if not isinstance(evidence, list):
            raise ValueError(f"{context}.evidence_frames must be a JSON array")
        return cls(
            start_frame=_integer(data["start_frame"], f"{context}.start_frame", minimum=0),
            end_frame=_integer(data["end_frame"], f"{context}.end_frame", minimum=1),
            scene_label=_enum(data["scene_label"], SceneTruth, f"{context}.scene_label"),
            broadcast_phase=_enum(data["broadcast_phase"], BroadcastPhase, f"{context}.broadcast_phase"),
            view=_enum(data["view"], CameraView, f"{context}.view"),
            playback=_enum(data["playback"], Playback, f"{context}.playback"),
            continuity_from_previous=_enum(
                data["continuity_from_previous"], Continuity, f"{context}.continuity_from_previous"
            ),
            data_use=_enum(data["data_use"], DataUse, f"{context}.data_use"),
            confidence=_number(data["confidence"], f"{context}.confidence"),
            evidence_frames=tuple(
                _integer(frame, f"{context}.evidence_frames[{evidence_index}]", minimum=0)
                for evidence_index, frame in enumerate(evidence)
            ),
            reason=_string(data["reason"], f"{context}.reason"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "scene_label": self.scene_label.value,
            "broadcast_phase": self.broadcast_phase.value,
            "view": self.view.value,
            "playback": self.playback.value,
            "continuity_from_previous": self.continuity_from_previous.value,
            "data_use": self.data_use.value,
            "confidence": self.confidence,
            "evidence_frames": list(self.evidence_frames),
            "reason": self.reason,
        }


def validate_prediction_partition(segments: tuple[PredictionSegment, ...], shard: ShardSpec) -> None:
    """Require a complete, gap-free partition of the benchmark shard."""
    if not segments:
        raise ValueError("a successful run must contain prediction segments")
    if segments[0].start_frame != shard.start_frame:
        raise ValueError(
            f"prediction partition starts at {segments[0].start_frame}, expected {shard.start_frame}"
        )
    for previous, current in zip(segments, segments[1:]):
        if current.start_frame != previous.end_frame:
            raise ValueError(
                f"prediction partition has a gap or overlap at [{previous.end_frame}, {current.start_frame})"
            )
    if segments[-1].end_frame != shard.end_frame:
        raise ValueError(f"prediction partition ends at {segments[-1].end_frame}, expected {shard.end_frame}")
    for index, segment in enumerate(segments):
        if not shard.start_frame <= segment.start_frame < segment.end_frame <= shard.end_frame:
            raise ValueError(f"segments[{index}] is outside the benchmark shard")


@dataclass(frozen=True)
class BenchmarkRunRecord:
    """One backend outcome with enough evidence to judge deployment fitness."""

    run_id: str
    outcome: RunOutcome
    model: ModelIdentity
    shard: ShardSpec
    requested_sampling: SamplingRequest
    observed_sampling: SamplingObservation | None
    runtime: RuntimeTelemetry
    attempt_count: int
    first_attempt_valid_json: bool
    first_attempt_valid_prediction: bool
    raw_response_sha256: str | None
    failure_reason: str | None
    segments: tuple[PredictionSegment, ...]
    attempt_response_sha256s: tuple[str, ...] | None = None
    schema_version: int = RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_RUN_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported run schema version {self.schema_version}")
        if not RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError("run_id must contain only letters, digits, dots, underscores, and hyphens")
        if not isinstance(self.outcome, RunOutcome):
            raise ValueError("outcome must be a RunOutcome")
        _integer(self.attempt_count, "attempt_count", minimum=0)
        if self.attempt_count > 2:
            raise ValueError("attempt_count cannot exceed the initial request and one correction retry")
        _boolean(self.first_attempt_valid_json, "first_attempt_valid_json")
        _boolean(self.first_attempt_valid_prediction, "first_attempt_valid_prediction")
        _sha256(self.raw_response_sha256, "raw_response_sha256", optional=True)
        if self.first_attempt_valid_json and self.attempt_count < 1:
            raise ValueError("first_attempt_valid_json requires at least one attempt")
        if self.attempt_count > 0 and self.raw_response_sha256 is None:
            raise ValueError("an attempted generation requires raw_response_sha256")
        if self.schema_version == 1:
            if self.attempt_response_sha256s is not None:
                raise ValueError("schema 1 cannot contain per-attempt response digests")
        else:
            if not isinstance(self.attempt_response_sha256s, tuple):
                raise ValueError("schema 2 requires per-attempt response digests")
            if len(self.attempt_response_sha256s) != self.attempt_count:
                raise ValueError("per-attempt response digest count differs from attempt_count")
            for index, digest in enumerate(self.attempt_response_sha256s):
                _sha256(digest, f"attempt_response_sha256s[{index}]")
            if self.attempt_response_sha256s:
                if self.raw_response_sha256 != self.attempt_response_sha256s[-1]:
                    raise ValueError("final response digest differs from the last attempt digest")
            elif self.raw_response_sha256 is not None:
                raise ValueError("a run without attempts cannot contain a response digest")

        if self.outcome is RunOutcome.SUCCEEDED:
            if self.attempt_count not in {1, 2}:
                raise ValueError("a successful run requires one or two attempts")
            if self.first_attempt_valid_prediction and self.attempt_count != 1:
                raise ValueError("a successful run with a valid first prediction requires exactly one attempt")
            if not self.first_attempt_valid_prediction and self.attempt_count != 2:
                raise ValueError("a successful run with an invalid first prediction requires the correction retry")
            if self.failure_reason is not None:
                raise ValueError("a successful run cannot have failure_reason")
            if self.observed_sampling is None:
                raise ValueError("a successful run requires observed_sampling")
            validate_prediction_partition(self.segments, self.shard)
        else:
            if not isinstance(self.failure_reason, str) or not self.failure_reason:
                raise ValueError("a failed run requires failure_reason")
            if self.segments:
                raise ValueError("a failed run cannot contain prediction segments")

        if self.observed_sampling is not None:
            for frame in self.observed_sampling.sampled_source_frames:
                if not self.shard.start_frame <= frame < self.shard.end_frame:
                    raise ValueError(f"observed sampled frame {frame} is outside the benchmark shard")

    @classmethod
    def from_json(cls, value: Any) -> BenchmarkRunRecord:
        data = _object(value, "run record")
        schema_version = _integer(data.get("schema_version"), "schema_version", minimum=1)
        keys = {
            "schema_version",
            "run_id",
            "outcome",
            "model",
            "shard",
            "requested_sampling",
            "observed_sampling",
            "runtime",
            "attempt_count",
            "first_attempt_valid_json",
            "raw_response_sha256",
            "failure_reason",
            "segments",
        }
        if schema_version == RUN_SCHEMA_VERSION:
            keys.update({"first_attempt_valid_prediction", "attempt_response_sha256s"})
        _exact_keys(
            data,
            keys,
            "run record",
        )
        segments = data["segments"]
        if not isinstance(segments, list):
            raise ValueError("segments must be a JSON array")
        attempt_digests = data.get("attempt_response_sha256s")
        if schema_version == RUN_SCHEMA_VERSION and not isinstance(attempt_digests, list):
            raise ValueError("attempt_response_sha256s must be a JSON array")
        parsed_attempt_digests: tuple[str, ...] | None = None
        if isinstance(attempt_digests, list):
            parsed_digests: list[str] = []
            for index, digest in enumerate(attempt_digests):
                parsed = _sha256(digest, f"attempt_response_sha256s[{index}]")
                assert parsed is not None
                parsed_digests.append(parsed)
            parsed_attempt_digests = tuple(parsed_digests)
        first_valid_json = _boolean(data["first_attempt_valid_json"], "first_attempt_valid_json")
        return cls(
            schema_version=schema_version,
            run_id=_string(data["run_id"], "run_id"),
            outcome=_enum(data["outcome"], RunOutcome, "outcome"),
            model=ModelIdentity.from_json(data["model"]),
            shard=ShardSpec.from_json(data["shard"]),
            requested_sampling=SamplingRequest.from_json(data["requested_sampling"]),
            observed_sampling=(
                None if data["observed_sampling"] is None else SamplingObservation.from_json(data["observed_sampling"])
            ),
            runtime=RuntimeTelemetry.from_json(data["runtime"]),
            attempt_count=_integer(data["attempt_count"], "attempt_count", minimum=0),
            first_attempt_valid_json=first_valid_json,
            first_attempt_valid_prediction=(
                first_valid_json
                if schema_version == 1
                else _boolean(
                    data["first_attempt_valid_prediction"],
                    "first_attempt_valid_prediction",
                )
            ),
            raw_response_sha256=_sha256(data["raw_response_sha256"], "raw_response_sha256", optional=True),
            failure_reason=(
                None if data["failure_reason"] is None else _string(data["failure_reason"], "failure_reason")
            ),
            segments=tuple(PredictionSegment.from_json(segment, index) for index, segment in enumerate(segments)),
            attempt_response_sha256s=parsed_attempt_digests,
        )

    def to_json(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "outcome": self.outcome.value,
            "model": self.model.to_json(),
            "shard": self.shard.to_json(),
            "requested_sampling": self.requested_sampling.to_json(),
            "observed_sampling": None if self.observed_sampling is None else self.observed_sampling.to_json(),
            "runtime": self.runtime.to_json(),
            "attempt_count": self.attempt_count,
            "first_attempt_valid_json": self.first_attempt_valid_json,
            "raw_response_sha256": self.raw_response_sha256,
            "failure_reason": self.failure_reason,
            "segments": [segment.to_json() for segment in self.segments],
        }
        if self.schema_version == RUN_SCHEMA_VERSION:
            value["first_attempt_valid_prediction"] = self.first_attempt_valid_prediction
            value["attempt_response_sha256s"] = list(self.attempt_response_sha256s or ())
        return value


def read_run_record(path: Path) -> BenchmarkRunRecord:
    """Read one UTF-8 JSON run record and validate every live field."""
    path = Path(path)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}") from error
    return BenchmarkRunRecord.from_json(value)


def write_run_record(path: Path, record: BenchmarkRunRecord) -> None:
    """Atomically write a deterministic record and verify its round trip."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(record.to_json(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if read_run_record(temporary) != record:
            raise RuntimeError(f"run record round trip changed values: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
