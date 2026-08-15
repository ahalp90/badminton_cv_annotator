"""Resumable whole-video TrackNet and RTMLib extraction for ShuttleSet22."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import cast
from uuid import uuid4

import numpy as np

from annotator.video_metadata import VideoMetadata
from bst_x.pipeline.shuttle_extractor import extract_all_shuttles, whole_video_csv_to_shuttle
from dataset_builder._pose_process import pose_subprocess_environment
from dataset_builder.manifest import artifact_integrity, resolve_interpreter
from dataset_builder.models import ArtifactIntegrity, InterpreterIdentity
from dataset_builder.pose_sharding import extract_sharded_rtmlib_pose_stage
from dataset_builder.shuttle_evidence import (
    ShuttleEvidenceArtifacts,
    load_shuttle_evidence,
    persist_shuttle_evidence,
    shuttle_evidence_artifacts,
)
from dataset_builder.vision import (
    PoseArrays,
    PoseExtraction,
    load_json_gz,
    load_npy_xz,
    pose_artifact_paths,
    save_json_gz,
    save_npy_xz,
    validate_pose_arrays,
)
from shuttleset22.manifest import ResolvedSource


EXTRACTION_RECEIPT_SCHEMA = "shuttleset22-extraction/1"
EXTRACTION_RECEIPT_FILENAME = "extraction_receipt.json.gz"
_POSE_IDENTITY_MARKER = "SHUTTLESET22_POSE_IDENTITY="
_POSE_MODEL_SHA256 = {
    "rtmdet": "4f4d7e07350b1753299111d1ae500fd64447a5b0e38e4bacbefab6573c742d30",
    "rtmpose": "cff059fd58a2c0d5fabaddcd66a96abcfb327563bcb0149ea59c9de4a8990fe2",
}
_TRACKNET_PACKAGES = ("torch", "torchvision", "numpy", "pandas", "Pillow", "opencv-python")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")

TracknetProducer = Callable[..., None]
PoseProducer = Callable[..., PoseExtraction]
PoseIdentityResolver = Callable[["ExtractionSettings"], "PoseRuntimeIdentity"]


@dataclass(frozen=True)
class PoseRuntimeIdentity:
    """Verified RTMLib model files and the package/provider environment."""

    models: tuple[ArtifactIntegrity, ...]
    runtime: Mapping[str, object]


@dataclass(frozen=True)
class ExtractionSettings:
    """Complete producer identity and bounded whole-video settings."""

    source_commit: str
    tracknet_dir: Path
    tracknet_model: Path
    inpaint_model: Path
    tracknet_python: Path
    pose_interpreter: Path
    tracknet_stride: int = 8
    tracknet_workers: int = 1
    tracknet_batch_size: int = 32
    tracknet_large_video: bool = True
    pose_shards: int = 8
    pose_n_max: int = 16
    pose_device: str = "cuda"
    pose_decode_mode: str = "seek"

    def __post_init__(self) -> None:
        if not _GIT_COMMIT_PATTERN.fullmatch(self.source_commit):
            raise ValueError("source_commit must be a 40-character lowercase Git object ID")
        for name, value in (
            ("tracknet_workers", self.tracknet_workers),
            ("tracknet_batch_size", self.tracknet_batch_size),
            ("pose_shards", self.pose_shards),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if isinstance(self.tracknet_stride, bool) or self.tracknet_stride not in {1, 8}:
            raise ValueError(f"tracknet_stride must be 1 or 8, got {self.tracknet_stride!r}")
        if not isinstance(self.tracknet_large_video, bool):
            raise ValueError(
                f"tracknet_large_video must be bool, got {self.tracknet_large_video!r}"
            )
        if (
            isinstance(self.pose_n_max, bool)
            or not isinstance(self.pose_n_max, int)
            or not 0 < self.pose_n_max <= 127
        ):
            raise ValueError(f"pose_n_max must be an integer in [1, 127], got {self.pose_n_max!r}")
        if self.pose_shards <= 1:
            raise ValueError(f"pose_shards must be greater than one, got {self.pose_shards!r}")
        if self.pose_device not in {"cpu", "cuda"}:
            raise ValueError(f"pose_device must be 'cpu' or 'cuda', got {self.pose_device!r}")
        if self.pose_decode_mode not in {"seek", "scan"}:
            raise ValueError(
                f"pose_decode_mode must be 'seek' or 'scan', got {self.pose_decode_mode!r}"
            )

    def configuration(self) -> dict[str, object]:
        """Return deterministic settings without host-specific paths."""
        return {
            "tracknet_stride": self.tracknet_stride,
            "tracknet_workers": self.tracknet_workers,
            "tracknet_batch_size": self.tracknet_batch_size,
            "tracknet_large_video": self.tracknet_large_video,
            "inpainting": True,
            "pose_shards": self.pose_shards,
            "pose_n_max": self.pose_n_max,
            "pose_device": self.pose_device,
            "pose_decode_mode": self.pose_decode_mode,
        }


@dataclass(frozen=True)
class ExtractionResult:
    """Validated published artifacts for one source."""

    output_dir: Path
    receipt: Path
    artifacts: Mapping[str, Path]
    reused: bool


@dataclass(frozen=True)
class ExtractionIdentity:
    """Inputs that must remain equal for receipt reuse."""

    source_commit: str
    source: ArtifactIntegrity
    source_metadata: VideoMetadata
    configuration: Mapping[str, object]
    models: tuple[ArtifactIntegrity, ...]
    producer_files: tuple[ArtifactIntegrity, ...]
    tracknet_interpreter: InterpreterIdentity
    pose_interpreter: InterpreterIdentity
    tracknet_runtime: Mapping[str, object]
    pose_runtime: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        """Return the receipt fields that precede output artifacts."""
        source = self.source.to_dict()
        source["path"] = Path(self.source.path).name
        source_metadata = self.source_metadata.to_dict()
        source_metadata["source_path"] = self.source_metadata.source_path.name
        return {
            "schema": EXTRACTION_RECEIPT_SCHEMA,
            "source_commit": self.source_commit,
            "source": source,
            "source_metadata": source_metadata,
            "configuration": dict(self.configuration),
            "models": [_portable_artifact(model) for model in self.models],
            "producer_files": [
                _portable_artifact(producer) for producer in self.producer_files
            ],
            "interpreters": {
                "tracknet": _portable_interpreter(self.tracknet_interpreter),
                "pose": _portable_interpreter(self.pose_interpreter),
            },
            "runtimes": {
                "tracknet": dict(self.tracknet_runtime),
                "pose": dict(self.pose_runtime),
            },
        }


def extraction_output_dir(output_root: Path, source: ResolvedSource) -> Path:
    """Return the stable per-match output directory."""
    return Path(output_root) / f"{source.entry.match_id:02d} {source.entry.video}"


def compress_csv(source: Path, destination: Path) -> Path:
    """Atomically gzip a CSV and verify its decompressed bytes before publication."""
    csv_path = Path(source)
    target = Path(destination)
    if not csv_path.is_file() or csv_path.is_symlink():
        raise FileNotFoundError(f"CSV source must be a regular non-symlink file: {csv_path}")
    if not target.name.endswith(".csv.gz"):
        raise ValueError(f"compressed CSV path must end in .csv.gz: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    source_digest = _file_digest(csv_path)
    try:
        with csv_path.open("rb") as input_handle, temporary.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_output,
                mtime=0,
            ) as output_handle:
                shutil.copyfileobj(input_handle, output_handle)
        with gzip.open(temporary, "rb") as handle:
            restored_digest = _stream_digest(handle)
        if restored_digest != source_digest:
            raise ValueError("compressed CSV round-trip differs from its source")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    csv_path.unlink()
    return target


def extract_source(
    source: ResolvedSource,
    *,
    output_root: Path,
    settings: ExtractionSettings,
    tracknet_producer: TracknetProducer = extract_all_shuttles,
    pose_producer: PoseProducer = extract_sharded_rtmlib_pose_stage,
    pose_identity_resolver: PoseIdentityResolver | None = None,
) -> ExtractionResult:
    """Run both producers and write a receipt only after full validation."""
    output_dir = extraction_output_dir(output_root, source)
    if Path(output_root).is_symlink() or output_dir.is_symlink():
        raise ValueError("extraction output directories must not be symlinks")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = source.metadata.source_path
    if source_path.is_symlink() or not source_path.is_file():
        raise FileNotFoundError(f"source is not a regular non-symlink file: {source_path}")
    current_source = artifact_integrity(source.integrity.name, source_path)
    if current_source != source.integrity:
        raise ValueError("source integrity differs from the resolved source contract")
    identity_resolver = pose_identity_resolver or resolve_pose_identity
    identity = _extraction_identity(source, settings, identity_resolver)
    receipt = output_dir / EXTRACTION_RECEIPT_FILENAME
    if receipt.is_file() and not receipt.is_symlink():
        try:
            artifacts = validate_receipt(source, output_dir=output_dir, identity=identity)
        except (EOFError, lzma.LZMAError, OSError, TypeError, ValueError):
            receipt.unlink()
        else:
            return ExtractionResult(output_dir, receipt, artifacts, reused=True)

    with tempfile.TemporaryDirectory(prefix=".extract-", dir=output_dir) as text:
        temporary = Path(text)
        shuttle_paths = shuttle_evidence_artifacts(
            temporary,
            input_video=source.metadata.source_path,
            stride=settings.tracknet_stride,
        )
        tracknet_producer(
            tracknet_dir=settings.tracknet_dir,
            clips_dir=source.metadata.source_path.parent,
            video_paths=[source.metadata.source_path],
            output_csv_dir=temporary,
            model_path=settings.tracknet_model,
            inpaintnet_path=settings.inpaint_model,
            tracknet_python=settings.tracknet_python,
            max_workers=settings.tracknet_workers,
            batch_size=settings.tracknet_batch_size,
            tracknet_stride=settings.tracknet_stride,
            large_video=settings.tracknet_large_video,
            enable_inpainting=True,
        )
        shuttle = whole_video_csv_to_shuttle(
            shuttle_paths.tracknet_csv,
            video_id=str(source.entry.match_id),
            frame_count=source.metadata.frame_count,
            width=source.metadata.width,
            height=source.metadata.height,
        )
        save_npy_xz(shuttle_paths.shuttle_track, shuttle.track)
        evidence = persist_shuttle_evidence(
            track=shuttle.track,
            artifacts=shuttle_paths,
            input_video=source.metadata.source_path,
            input_height=source.metadata.height,
            frame_count=source.metadata.frame_count,
            stride=settings.tracknet_stride,
            tracknet_model=settings.tracknet_model,
            inpaint_model=settings.inpaint_model,
        )
        csv_gz = compress_csv(
            shuttle_paths.tracknet_csv,
            shuttle_paths.tracknet_csv.with_suffix(".csv.gz"),
        )
        pose = pose_producer(
            metadata=source.metadata,
            output_dir=temporary,
            interpreter=settings.pose_interpreter,
            shards=settings.pose_shards,
            device=settings.pose_device,
            n_max=settings.pose_n_max,
            decode_mode=settings.pose_decode_mode,
        )
        validate_pose_arrays(pose.arrays, source.metadata.frame_count)
        _validate_pose_slot_count(pose.arrays, settings.pose_n_max)
        temporary_artifacts = {
            "tracknet_csv": csv_gz,
            "inpaint_sidecar": evidence.artifacts.inpaint_sidecar,
            "shuttle_track": evidence.artifacts.shuttle_track,
            "shuttle_guard_codes": evidence.artifacts.guard_codes,
            "shuttle_guard_diagnostics": evidence.artifacts.guard_diagnostics,
            **pose.artifacts.as_mapping(),
        }
        published = _publish_artifacts(temporary_artifacts, output_dir)

    current_identity = _extraction_identity(source, settings, identity_resolver)
    current_source = artifact_integrity(source.integrity.name, source.metadata.source_path)
    if current_identity != identity or current_source != source.integrity:
        raise ValueError("extraction inputs changed while producers were running")
    _write_receipt(receipt, identity, published, output_dir)
    validated = validate_receipt(source, output_dir=output_dir, identity=identity)
    return ExtractionResult(output_dir, receipt, validated, reused=False)


def validate_receipt(
    source: ResolvedSource,
    *,
    output_dir: Path,
    identity: ExtractionIdentity,
) -> dict[str, Path]:
    """Load a receipt, verify every hash, and semantically reload all arrays."""
    root = Path(output_dir).resolve(strict=True)
    receipt = root / EXTRACTION_RECEIPT_FILENAME
    payload = load_json_gz(receipt)
    expected_identity = identity.payload()
    expected_keys = {*expected_identity, "artifacts"}
    if set(payload) != expected_keys:
        raise ValueError("extraction receipt fields differ from the schema")
    for key, expected in expected_identity.items():
        if payload[key] != expected:
            raise ValueError(f"extraction receipt {key} differs from current inputs")
    artifact_payload = payload["artifacts"]
    if not isinstance(artifact_payload, list):
        raise ValueError("extraction receipt artifacts must be a list")
    records = tuple(ArtifactIntegrity.from_dict(value) for value in artifact_payload)
    stride = cast(int, identity.configuration["tracknet_stride"])
    expected_names = _expected_artifact_filenames(source, stride)
    if {record.name for record in records} != set(expected_names):
        raise ValueError("extraction receipt artifact names differ")

    artifacts: dict[str, Path] = {}
    for record in records:
        expected_filename = expected_names[record.name]
        if record.path != expected_filename:
            raise ValueError(f"artifact {record.name} path {record.path!r} != {expected_filename!r}")
        path = root / record.path
        actual = artifact_integrity(record.name, path, relative_to=root)
        if actual != record:
            raise ValueError(f"artifact {record.name} integrity differs from its receipt")
        artifacts[record.name] = path
    _semantic_reload(source, identity, artifacts)
    return artifacts


def _extraction_identity(
    source: ResolvedSource,
    settings: ExtractionSettings,
    pose_identity_resolver: PoseIdentityResolver,
) -> ExtractionIdentity:
    pose_identity = pose_identity_resolver(settings)
    models = (
        artifact_integrity("tracknet", settings.tracknet_model),
        artifact_integrity("inpaintnet", settings.inpaint_model),
        *pose_identity.models,
    )
    return ExtractionIdentity(
        source_commit=settings.source_commit,
        source=source.integrity,
        source_metadata=source.metadata,
        configuration=settings.configuration(),
        models=models,
        producer_files=(
            artifact_integrity(
                "tracknet_batch_predict",
                settings.tracknet_dir / "batch_predict.py",
            ),
        ),
        tracknet_interpreter=resolve_interpreter(settings.tracknet_python),
        pose_interpreter=resolve_interpreter(settings.pose_interpreter),
        tracknet_runtime={
            "packages": _python_package_versions(
                settings.tracknet_python,
                _TRACKNET_PACKAGES,
            ),
        },
        pose_runtime=pose_identity.runtime,
    )


def resolve_pose_identity(settings: ExtractionSettings) -> PoseRuntimeIdentity:
    """Resolve, SHA-verify, and describe the models used by RTMLib."""
    resolve_interpreter(settings.pose_interpreter)
    script = """
import importlib.metadata
import json
import onnxruntime
from preparing_data.rtmlib_pose import DET_URL, POSE_URL
from rtmlib.tools.file import download_checkpoint

packages = {}
for name in ("rtmlib", "onnxruntime", "onnxruntime-gpu", "numpy", "opencv-python"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
payload = {
    "models": {
        "rtmdet": str(download_checkpoint(DET_URL, progress=False)),
        "rtmpose": str(download_checkpoint(POSE_URL, progress=False)),
    },
    "packages": packages,
    "onnxruntime_providers": onnxruntime.get_available_providers(),
}
print("SHUTTLESET22_POSE_IDENTITY=" + json.dumps(payload, sort_keys=True))
"""
    completed = subprocess.run(
        [os.fspath(settings.pose_interpreter), "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=pose_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"could not inspect RTMLib runtime: {detail}")
    line = next(
        (row for row in reversed(completed.stdout.splitlines()) if row.startswith(_POSE_IDENTITY_MARKER)),
        None,
    )
    if line is None:
        raise ValueError("RTMLib identity command returned no structured identity")
    try:
        payload = json.loads(line.removeprefix(_POSE_IDENTITY_MARKER))
    except json.JSONDecodeError as exc:
        raise ValueError("RTMLib identity command returned malformed JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "models",
        "packages",
        "onnxruntime_providers",
    }:
        raise ValueError("RTMLib identity fields differ")
    model_payload = payload["models"]
    if not isinstance(model_payload, dict) or set(model_payload) != set(_POSE_MODEL_SHA256):
        raise ValueError("RTMLib model identity fields differ")
    models: list[ArtifactIntegrity] = []
    for name, expected_sha256 in _POSE_MODEL_SHA256.items():
        path_value = model_payload[name]
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"RTMLib model path {name} is invalid")
        path = Path(path_value)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"RTMLib model is not a regular non-symlink file: {path}")
        actual_sha256 = _file_digest(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"RTMLib model {name} SHA-256 {actual_sha256} != {expected_sha256}"
            )
        models.append(artifact_integrity(name, path))
    packages = payload["packages"]
    providers = payload["onnxruntime_providers"]
    if not isinstance(packages, dict) or any(
        not isinstance(name, str) or (version is not None and not isinstance(version, str))
        for name, version in packages.items()
    ):
        raise ValueError("RTMLib package identity is invalid")
    if packages.get("rtmlib") is None or (
        packages.get("onnxruntime") is None and packages.get("onnxruntime-gpu") is None
    ):
        raise ValueError("RTMLib runtime packages are missing")
    if not isinstance(providers, list) or any(not isinstance(value, str) for value in providers):
        raise ValueError("RTMLib provider identity is invalid")
    if settings.pose_device == "cuda" and "CUDAExecutionProvider" not in providers:
        raise RuntimeError("RTMLib runtime does not expose CUDAExecutionProvider")
    return PoseRuntimeIdentity(
        models=tuple(models),
        runtime={
            "packages": packages,
            "onnxruntime_providers": providers,
            "model_sha256": dict(_POSE_MODEL_SHA256),
        },
    )


def _publish_artifacts(paths: Mapping[str, Path], output_dir: Path) -> dict[str, Path]:
    published: dict[str, Path] = {}
    names: set[str] = set()
    for name, source in sorted(paths.items()):
        if source.name in names:
            raise ValueError(f"duplicate extraction artifact filename: {source.name}")
        names.add(source.name)
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"extraction artifact is not a regular file: {source}")
        destination = output_dir / source.name
        os.replace(source, destination)
        published[name] = destination
    return published


def _write_receipt(
    receipt: Path,
    identity: ExtractionIdentity,
    artifacts: Mapping[str, Path],
    output_dir: Path,
) -> None:
    records = tuple(
        artifact_integrity(name, path, relative_to=output_dir)
        for name, path in sorted(artifacts.items())
    )
    payload = identity.payload()
    payload["artifacts"] = [record.to_dict() for record in records]
    save_json_gz(receipt, payload)


def _expected_artifact_filenames(source: ResolvedSource, stride: int) -> dict[str, str]:
    stem = source.metadata.source_path.stem
    pose = pose_artifact_paths(Path("."))
    return {
        "tracknet_csv": f"{stem}_ball.csv.gz",
        "inpaint_sidecar": f"{stem}_stride{stride}_inpaint_mask.json.gz",
        "shuttle_track": "shuttle_track.npy.xz",
        "shuttle_guard_codes": "shuttle_guard_codes.npy.xz",
        "shuttle_guard_diagnostics": "shuttle_guard_diagnostics.json.gz",
        **{name: path.name for name, path in pose.as_mapping().items()},
    }


def _semantic_reload(
    source: ResolvedSource,
    identity: ExtractionIdentity,
    artifacts: Mapping[str, Path],
) -> None:
    settings = identity.configuration
    stride = cast(int, settings["tracknet_stride"])
    shuttle_artifacts = ShuttleEvidenceArtifacts(
        tracknet_csv=artifacts["tracknet_csv"],
        inpaint_sidecar=artifacts["inpaint_sidecar"],
        shuttle_track=artifacts["shuttle_track"],
        guard_codes=artifacts["shuttle_guard_codes"],
        guard_diagnostics=artifacts["shuttle_guard_diagnostics"],
    )
    evidence = load_shuttle_evidence(
        artifacts=shuttle_artifacts,
        input_video=source.metadata.source_path,
        input_height=source.metadata.height,
        frame_count=source.metadata.frame_count,
        stride=stride,
        tracknet_model=Path(identity.models[0].path),
        inpaint_model=Path(identity.models[1].path),
    )
    csv_shuttle = whole_video_csv_to_shuttle(
        artifacts["tracknet_csv"],
        video_id=str(source.entry.match_id),
        frame_count=source.metadata.frame_count,
        width=source.metadata.width,
        height=source.metadata.height,
    )
    if not np.array_equal(csv_shuttle.track, evidence.track):
        raise ValueError("compressed TrackNet CSV differs from the shuttle array")
    pose = PoseArrays(
        kps=load_npy_xz(artifacts["pose_kps"]),
        bboxes=load_npy_xz(artifacts["pose_bboxes"]),
        scores=load_npy_xz(artifacts["pose_scores"]),
        kp_scores=load_npy_xz(artifacts["pose_kp_scores"]),
        ndet=load_npy_xz(artifacts["pose_ndet"]),
    )
    validate_pose_arrays(pose, source.metadata.frame_count)
    _validate_pose_slot_count(pose, cast(int, settings["pose_n_max"]))


def _validate_pose_slot_count(arrays: PoseArrays, expected_slots: int) -> None:
    actual_slots = int(np.asarray(arrays.bboxes).shape[1])
    if actual_slots != expected_slots:
        raise ValueError(f"pose slot count {actual_slots} != configured n_max {expected_slots}")


def _python_package_versions(
    executable: Path,
    package_names: tuple[str, ...],
) -> dict[str, str]:
    resolve_interpreter(executable)
    script = """
import importlib.metadata
import json
import sys

names = json.loads(sys.argv[1])
versions = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps(versions, sort_keys=True))
"""
    completed = subprocess.run(
        [os.fspath(executable), "-c", script, json.dumps(package_names)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"could not inspect Python packages: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("Python package identity command returned malformed JSON") from exc
    if not isinstance(payload, dict) or set(payload) != set(package_names):
        raise ValueError("Python package identity fields differ")
    missing = sorted(name for name, version in payload.items() if version is None)
    if missing:
        raise ValueError(f"Python runtime is missing required packages: {missing}")
    if any(not isinstance(name, str) or not isinstance(version, str) for name, version in payload.items()):
        raise ValueError("Python package identity values are invalid")
    return {name: payload[name] for name in package_names}


def _portable_artifact(artifact: ArtifactIntegrity) -> dict[str, object]:
    payload = artifact.to_dict()
    payload["path"] = Path(artifact.path).name
    return payload


def _portable_interpreter(interpreter: InterpreterIdentity) -> dict[str, object]:
    payload = interpreter.to_dict()
    payload["path"] = Path(interpreter.path).name
    return payload


def _file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return _stream_digest(handle)


def _stream_digest(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()
