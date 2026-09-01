"""Freeze the fixed ShuttleSet22 contact predictions without reading labels."""

# ruff: noqa: E402, RUF100

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import lzma
import os
import platform
import re
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scratch.contact_det.scripts.freeze_contact_evidence import (
    FixtureSpec,
    _load_inputs,
    read_annotation,
)
from scratch.contact_det.scripts.freeze_tree_contact_features import (
    REGION_FIELDS,
    _fixture_rows,
)
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    SplitRole,
    VideoSpec,
)
from scratch.contact_det_full_ds_fit.scripts.inpaint_shuttleset22_tracks import (
    RECEIPT_FILENAME,
    VIDEO_IDS,
    VideoInput,
    load_npy_xz,
    output_paths,
    read_json_gz,
    sha256,
    validate_completed,
)
from scratch.contact_det_full_ds_fit.scripts.inpaint_shuttleset22_tracks import (
    RUN_STATE_FILENAME as INPAINT_RUN_STATE_FILENAME,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    SCORE_DTYPE,
    _feature_matrix,
    predictions_for_settings,
)

PREDICTION_SCHEMA = "shuttleset22-contact-predictions/1"
VIDEO_RESULT_SCHEMA = "shuttleset22-contact-prediction-result/1"
COMBINED_SCHEMA = "shuttleset22-contact-predictions-combined/1"
RUN_STATE_SCHEMA = "shuttleset22-contact-prediction-run/1"
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")

SOURCE_MANIFEST_SHA256 = (
    "746225f6b9bb1b257052224648c39e813792a75a7eb8711443688ca93fad7463"
)
ARTIFACT_IDENTITY_SHA256 = (
    "dffe2cc2afc75f78eb89b30236477eb732f92a824b22ee3a01a4f893a673864e"
)
INPAINT_RUN_STATE_SHA256 = (
    "ee5c55ec1ab0833e4bf0525dcabcf5b9eab5fde7c01dc08c47ab362ca447b160"
)
INPAINT_RUNNER_SHA256 = (
    "2e4fe812168ef2a7abadbd8594cc3ba9bf92f0b2f4677edfbc80466fa018e1b1"
)
MODEL_SHA256 = "ef7b66042ce2ed594572424ddd2c13f23092afcc8b259bccc8758af8cc11a8dc"
MODEL_RESULT_SHA256 = "5428bb69be41aea034fe56f5b812594404d1ac458392681f853b74a26600b4ed"
SETTING_RESULT_SHA256 = (
    "9c21575c457742bf71dea6a9105ba91234f1b0038ee70bc3f9d885c56ce8ac83"
)
COMBINED_DEVELOPMENT_SCORES_SHA256 = (
    "d464d396af9ff451878f40ead57d46d2dbde3a61ebfbe70adee14519334707d9"
)
FINAL_DEVELOPMENT_CONTACTS_SHA256 = (
    "947b87f3341edbb2a8a5f60bfacfd023f9a0ef45df507d38dbad6820b4f3471e"
)

EXPECTED_OVERLAPS = {1: 23, 2: 38, 3: 39, 4: 41, 5: 42, 6: 43, 7: 44, 58: 24}
EXPECTED_UNRESOLVED_IDS = {14, 45, 56}
EXPECTED_LIBRARY_VERSIONS = {
    "python": "3.11.13",
    "numpy": "2.2.6",
    "scikit-learn": "1.6.1",
    "joblib": "1.5.3",
}
EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080
EXPECTED_FPS = 30.0
SCORE_CUTOFF = 0.9
NEARBY_DISTANCE_AT_30_FPS = 6

POSE_FILENAMES = (
    "pose_kps.npy.xz",
    "pose_bboxes.npy.xz",
    "pose_scores.npy.xz",
    "pose_kp_scores.npy.xz",
    "pose_ndet.npy.xz",
)
COURT_FILENAMES = (
    "court_evidence.json.gz",
    "court_keep_vote.npy.xz",
    "court_present.npy.xz",
)
ANNOTATION_FILENAMES = (
    "annotator_result.json.gz",
    "raw_replay_mask.npy.xz",
    "definitive_exclusion_mask.npy.xz",
    "shuttle_quality.json.gz",
)
PREDICTION_OUTPUT_FILENAMES = (
    "contact_features.npy.xz",
    "candidate_scores.npy.xz",
    "predictions.json.gz",
) + tuple(f"annotation/{name}" for name in ANNOTATION_FILENAMES)


@dataclass(frozen=True)
class SourceSpec:
    """One fixed downloadable ShuttleSet22 source."""

    video_id: int
    video_name: str


@dataclass(frozen=True)
class ArtifactMetadata:
    """Validated capture metadata and prepared-artifact identities."""

    frame_count: int
    width: int
    height: int
    receipt_sha256: str
    shuttle_sha256: str
    court_code_id: str
    court_model_md5: str


@dataclass(frozen=True)
class CheckedVideo:
    """One source and its checked prepared/inpaint input directory."""

    source: SourceSpec
    source_video: Path
    directory: Path
    metadata: ArtifactMetadata


@dataclass(frozen=True)
class ModelBundle:
    """The checked final model and its fixed input fields."""

    model: Any
    input_fields: tuple[str, ...]


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encoded_json(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _gzip_json_bytes(payload: Mapping[str, object]) -> bytes:
    return gzip.compress(_encoded_json(payload), compresslevel=9, mtime=0)


def _write_bytes(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def _write_verified_bytes(path: Path, payload: bytes) -> None:
    """Write bytes atomically and require the saved file to reload exactly."""
    _write_bytes(path, payload)
    if Path(path).read_bytes() != payload:
        raise ValueError(f"saved file differs after reload: {Path(path).name}")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_bytes(path, _encoded_json(payload))


def _write_json_gz(path: Path, payload: Mapping[str, object]) -> None:
    _write_bytes(path, _gzip_json_bytes(payload))


def _write_npy_xz(path: Path, values: np.ndarray) -> None:
    buffer = io.BytesIO()
    with lzma.open(buffer, "wb", format=lzma.FORMAT_XZ, preset=9) as destination:
        np.save(destination, values, allow_pickle=False)
    _write_bytes(path, buffer.getvalue())


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _mapping(payload, name)


def source_specs_from_payload(payload: object) -> tuple[SourceSpec, ...]:
    """Validate the fixed source roles and return the 47 downloadable videos."""
    root = _mapping(payload, "source manifest")
    if root.get("schema") != "shuttleset22-sources/1":
        raise ValueError("source manifest schema differs")
    rows = _sequence(root.get("videos"), "source manifest videos")
    if len(rows) != 58:
        raise ValueError(f"source manifest must contain 58 videos, found {len(rows)}")

    ids: list[int] = []
    overlaps: dict[int, int] = {}
    unresolved: set[int] = set()
    downloads: list[SourceSpec] = []
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"source manifest videos[{index}]")
        video_id = _integer(row.get("id"), f"source manifest videos[{index}].id")
        video_name = _string(row.get("video"), f"source manifest videos[{index}].video")
        source_kind = _string(
            row.get("source_kind"), f"source manifest videos[{index}].source_kind"
        )
        ids.append(video_id)
        if source_kind == "download":
            downloads.append(SourceSpec(video_id, video_name))
        elif source_kind == "shuttleset_overlap":
            overlaps[video_id] = _integer(
                row.get("overlap_shuttleset_id"),
                f"source manifest videos[{index}].overlap_shuttleset_id",
            )
        elif source_kind == "unresolved":
            unresolved.add(video_id)
        else:
            raise ValueError(
                f"source manifest video {video_id} has an unknown source kind"
            )

    if ids != list(range(1, 59)):
        raise ValueError("source manifest video IDs must be ordered from 1 through 58")
    if overlaps != EXPECTED_OVERLAPS:
        raise ValueError("source manifest overlap mapping differs")
    if unresolved != EXPECTED_UNRESOLVED_IDS:
        raise ValueError("source manifest unresolved IDs differ")
    if tuple(source.video_id for source in downloads) != VIDEO_IDS:
        raise ValueError(
            "source manifest downloadable IDs differ from the fixed test set"
        )
    if len({source.video_name for source in downloads}) != len(downloads):
        raise ValueError("source manifest downloadable video names are repeated")
    return tuple(downloads)


def load_source_specs(path: Path) -> tuple[SourceSpec, ...]:
    """Load the exact historical source manifest without using annotations."""
    if sha256(path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("source manifest SHA-256 differs")
    with Path(path).open("rb") as source:
        return source_specs_from_payload(tomllib.load(source))


def _validate_identity(path: Path, raw_identity: object) -> None:
    identity = _mapping(raw_identity, f"{path.name} identity")
    if set(identity) != {"name", "path", "size_bytes", "md5"}:
        raise ValueError(f"artifact identity fields differ for {path.name}")
    if identity["path"] != path.name:
        raise ValueError(f"artifact receipt path differs for {path.name}")
    if identity["size_bytes"] != path.stat().st_size:
        raise ValueError(f"artifact size differs for {path.name}")
    if identity["md5"] != _md5(path):
        raise ValueError(f"artifact MD5 differs for {path.name}")


def validate_artifact_directory(
    directory: Path, source: SourceSpec
) -> ArtifactMetadata:
    """Validate the prepared pose/court receipt without opening annotations."""
    receipt_path = Path(directory) / "court_receipt.json.gz"
    receipt = read_json_gz(receipt_path)
    required = {
        "schema",
        "match_id",
        "video",
        "code_id",
        "configuration",
        "metadata",
        "model",
        "inputs",
        "completed",
        "scene_count",
        "outputs",
    }
    if set(receipt) != required or receipt["schema"] != "shuttleset22-court/0.1":
        raise ValueError(f"{source.video_id:02d}: unsupported court receipt")
    if receipt["match_id"] != source.video_id or receipt["video"] != source.video_name:
        raise ValueError(f"{source.video_id:02d}: court receipt identity differs")
    if receipt["completed"] is not True:
        raise ValueError(f"{source.video_id:02d}: court receipt is incomplete")

    metadata = _mapping(receipt["metadata"], "court metadata")
    if (metadata.get("fps_numerator"), metadata.get("fps_denominator")) != (30, 1):
        raise ValueError(f"{source.video_id:02d}: court receipt is not exact 30 fps")
    frame_count = _integer(metadata.get("frame_count"), "court metadata frame_count")
    width = _integer(metadata.get("width"), "court metadata width")
    height = _integer(metadata.get("height"), "court metadata height")
    if frame_count <= 0 or (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise ValueError(f"{source.video_id:02d}: court metadata dimensions differ")

    input_rows = {
        _string(
            _mapping(row, "court receipt input").get("name"), "court input name"
        ): row
        for row in _sequence(receipt["inputs"], "court receipt inputs")
    }
    for filename in POSE_FILENAMES:
        name = filename.removesuffix(".npy.xz")
        if name not in input_rows:
            raise ValueError(f"{source.video_id:02d}: {name} identity is missing")
        _validate_identity(Path(directory) / filename, input_rows[name])

    output_rows = {
        _string(
            _mapping(row, "court receipt output").get("name"), "court output name"
        ): row
        for row in _sequence(receipt["outputs"], "court receipt outputs")
    }
    output_roles = ("court_evidence", "court_keep_vote", "court_present")
    if set(output_rows) != set(output_roles):
        raise ValueError(f"{source.video_id:02d}: court output identities differ")
    for role, filename in zip(output_roles, COURT_FILENAMES, strict=True):
        _validate_identity(Path(directory) / filename, output_rows[role])

    model = _mapping(receipt["model"], "court model")
    return ArtifactMetadata(
        frame_count=frame_count,
        width=width,
        height=height,
        receipt_sha256=sha256(receipt_path),
        shuttle_sha256=sha256(Path(directory) / "shuttle_track.npy.xz"),
        court_code_id=_string(receipt["code_id"], "court code ID"),
        court_model_md5=_string(model.get("md5"), "court model MD5"),
    )


def _artifact_identity(checked: Sequence[CheckedVideo]) -> str:
    payload = {
        f"{video.source.video_id:02d}": {
            "court_receipt_sha256": video.metadata.receipt_sha256,
            "shuttle_track_sha256": video.metadata.shuttle_sha256,
            "court_code_id": video.metadata.court_code_id,
            "court_model_md5": video.metadata.court_model_md5,
        }
        for video in checked
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def check_input_set(
    source_manifest: Path,
    inpaint_root: Path,
    source_root: Path,
) -> tuple[CheckedVideo, ...]:
    """Check the fixed 47-video input identity without reading annotations."""
    if (
        sha256(Path(inpaint_root) / INPAINT_RUN_STATE_FILENAME)
        != INPAINT_RUN_STATE_SHA256
    ):
        raise ValueError("completed inpaint run-state SHA-256 differs")

    from scratch.contact_det_full_ds_fit.scripts import inpaint_shuttleset22_tracks

    if sha256(Path(inpaint_shuttleset22_tracks.__file__)) != INPAINT_RUNNER_SHA256:
        raise ValueError("deployed inpaint runner SHA-256 differs")

    sources = load_source_specs(source_manifest)
    expected_names = {
        f"{source.video_id:02d} {source.video_name}" for source in sources
    }
    actual_names = {path.name for path in Path(inpaint_root).iterdir() if path.is_dir()}
    working_names = {name for name in actual_names if name.endswith(".working")}
    if working_names:
        raise ValueError(
            f"unfinished inpaint directories remain: {sorted(working_names)}"
        )
    if actual_names != expected_names:
        raise ValueError(
            "completed inpaint directory set differs from the fixed source set"
        )

    checked: list[CheckedVideo] = []
    for source in sources:
        directory = Path(inpaint_root) / f"{source.video_id:02d} {source.video_name}"
        source_video = (
            Path(source_root) / f"{source.video_id:02d} {source.video_name}.mp4"
        )
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        metadata = validate_artifact_directory(directory, source)
        checked.append(CheckedVideo(source, source_video, directory, metadata))
    if _artifact_identity(checked) != ARTIFACT_IDENTITY_SHA256:
        raise ValueError("prepared artifact identity SHA-256 differs")
    return tuple(checked)


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": version("scikit-learn"),
        "joblib": version("joblib"),
    }


def load_model_bundle(
    model_path: Path,
    model_result_path: Path,
    setting_result_path: Path,
) -> ModelBundle:
    """Load the exact final model and verify the recorded setting identities."""
    if sha256(model_path) != MODEL_SHA256:
        raise ValueError("final contact model SHA-256 differs")
    if sha256(model_result_path) != MODEL_RESULT_SHA256:
        raise ValueError("final model result SHA-256 differs")
    if sha256(setting_result_path) != SETTING_RESULT_SHA256:
        raise ValueError("final setting result SHA-256 differs")

    model_result = _read_json(model_result_path, "final model result")
    setting_result = _read_json(setting_result_path, "final setting result")
    expected_result = {
        "schema": "final-contact-model-fit/1",
        "status": "complete",
        "labels_read": True,
        "model_sha256": MODEL_SHA256,
        "setting_result_sha256": SETTING_RESULT_SHA256,
        "combined_raw_score_sha256": COMBINED_DEVELOPMENT_SCORES_SHA256,
        "selected_score_cutoff": SCORE_CUTOFF,
        "selected_duplicate_distance_at_30_fps": NEARBY_DISTANCE_AT_30_FPS,
        "library_versions": EXPECTED_LIBRARY_VERSIONS,
    }
    if any(model_result.get(name) != value for name, value in expected_result.items()):
        raise ValueError("final model result identity or fixed setting differs")
    if setting_result.get("final_score_sha256") != FINAL_DEVELOPMENT_CONTACTS_SHA256:
        raise ValueError("final development contact SHA-256 differs")
    if _runtime_versions() != EXPECTED_LIBRARY_VERSIONS:
        raise ValueError(
            f"model runtime versions differ: expected {EXPECTED_LIBRARY_VERSIONS}, "
            f"found {_runtime_versions()}"
        )

    raw_fields = _sequence(model_result.get("model_input_fields"), "model input fields")
    input_fields = tuple(_string(field, "model input field") for field in raw_fields)
    if len(input_fields) != 85 or len(set(input_fields)) != len(input_fields):
        raise ValueError("final model input fields differ")

    import joblib

    model = joblib.load(model_path)
    if not np.array_equal(np.asarray(model.classes_), np.asarray([0, 1])):
        raise ValueError("final contact model classes differ")
    return ModelBundle(model, input_fields)


def fill_mask_from_sidecar(sidecar: Mapping[str, Any], frame_count: int) -> np.ndarray:
    """Restore the saved half-open InpaintNet fill spans as a boolean mask."""
    if (
        sidecar.get("schema") != "inpaint_fill_mask/1"
        or sidecar.get("n_rows") != frame_count
    ):
        raise ValueError("inpaint fill sidecar identity differs")
    mask = np.zeros(frame_count, dtype=bool)
    previous_stop = 0
    for index, raw_span in enumerate(
        _sequence(sidecar.get("inpaint_selected"), "inpaint fill spans")
    ):
        span = _sequence(raw_span, f"inpaint fill spans[{index}]")
        if len(span) != 2:
            raise ValueError("each inpaint fill span must contain start and stop")
        start = _integer(span[0], f"inpaint fill spans[{index}].start")
        stop = _integer(span[1], f"inpaint fill spans[{index}].stop")
        if not 0 <= previous_stop <= start < stop <= frame_count:
            raise ValueError(
                "inpaint fill spans must be ordered, disjoint and in range"
            )
        mask[start:stop] = True
        previous_stop = stop
    return mask


def _link_stage_inputs(
    stage_root: Path, video: CheckedVideo, fixture: FixtureSpec
) -> None:
    stages = Path(stage_root) / "stages"
    shuttle_dir = stages / "shuttle" / fixture.name
    pose_dir = stages / "pose" / fixture.name
    court_dir = stages / "court" / fixture.name
    shuttle_dir.mkdir(parents=True)
    pose_dir.mkdir(parents=True)
    court_dir.mkdir(parents=True)
    (shuttle_dir / "shuttle_track.npy.xz").symlink_to(
        (video.directory / "shuttle_track_inpainted.npy.xz").resolve(strict=True)
    )
    for filename in POSE_FILENAMES:
        (pose_dir / filename).symlink_to(
            (video.directory / filename).resolve(strict=True)
        )
    for filename in COURT_FILENAMES:
        (court_dir / filename).symlink_to(
            (video.directory / filename).resolve(strict=True)
        )


def _score_candidates(
    candidates: np.ndarray,
    fixture: FixtureSpec,
    bundle: ModelBundle,
) -> tuple[np.ndarray, np.ndarray]:
    if candidates.dtype.names is None or any(
        field not in candidates.dtype.names for field in bundle.input_fields
    ):
        raise ValueError(f"{fixture.name}: feature rows lack a final model input field")
    probabilities = np.asarray(
        bundle.model.predict_proba(_feature_matrix(candidates, bundle.input_fields))[
            :, 1
        ],
        dtype=np.float64,
    )
    if len(probabilities) != len(candidates) or not np.isfinite(probabilities).all():
        raise ValueError(
            f"{fixture.name}: contact probabilities are incomplete or non-finite"
        )
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(
            f"{fixture.name}: contact probabilities lie outside zero and one"
        )

    scores = np.empty(len(candidates), dtype=SCORE_DTYPE)
    for field in ("fixture", "interval_id", "frame", "fps"):
        scores[field] = candidates[field]
    scores["contact_score"] = probabilities
    scores["kept"] = False
    video_spec = VideoSpec(
        fixture=fixture.name,
        video_id=fixture.video_id,
        fps=fixture.fps,
        width=int(fixture.width),
        height=int(fixture.height),
        role=SplitRole.VALIDATION,
        winner="unknown",
        loser="unknown",
        tournament="ShuttleSet22",
        tournament_round="unknown",
    )
    predictions, kept = predictions_for_settings(
        scores,
        (video_spec,),
        SCORE_CUTOFF,
        NEARBY_DISTANCE_AT_30_FPS,
    )
    scores["kept"] = kept
    return scores, predictions[fixture.name]


def _normalise_side(value: object) -> str | None:
    if value is None:
        return None
    side = getattr(value, "value", value)
    if side == "Top":
        return "Top"
    if side in {"Bot", "Bottom"}:
        return "Bot"
    raise ValueError(f"player-side answer differs: {side!r}")


def _span_id(frame: int, spans: Sequence[Mapping[str, int]]) -> int | None:
    matches = [
        int(span["span_id"])
        for span in spans
        if int(span["start_frame"]) <= frame < int(span["end_frame"])
    ]
    if len(matches) > 1:
        raise ValueError(f"frame {frame} belongs to overlapping annotation spans")
    return matches[0] if matches else None


def _input_records(video: CheckedVideo) -> list[dict[str, object]]:
    roles = {
        "inpaint_receipt": video.directory / RECEIPT_FILENAME,
        "inpainted_shuttle_track": video.directory / "shuttle_track_inpainted.npy.xz",
        "inpaint_fill_sidecar": output_paths(
            video.directory, video.directory.name
        ).sidecar_path,
        "shuttle_guard_codes": video.directory / "shuttle_guard_codes_inpainted.npy.xz",
        "court_receipt": video.directory / "court_receipt.json.gz",
    }
    roles.update(
        {
            filename.removesuffix(".npy.xz"): video.directory / filename
            for filename in POSE_FILENAMES
        }
    )
    roles.update(
        {
            filename.split(".", maxsplit=1)[0]: video.directory / filename
            for filename in COURT_FILENAMES
        }
    )
    return [
        {
            "role": role,
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for role, path in roles.items()
    ]


def _validate_input_records(video: CheckedVideo, raw_records: object) -> None:
    recorded = list(_sequence(raw_records, "prediction input files"))
    if recorded != _input_records(video):
        raise ValueError(f"video {video.source.video_id}: saved input files differ")


def _validate_output_hashes(root: Path, raw_hashes: object) -> None:
    output_hashes = _mapping(raw_hashes, "prediction output hashes")
    if set(output_hashes) != set(PREDICTION_OUTPUT_FILENAMES):
        raise ValueError("saved prediction output list differs")
    for raw_name, raw_digest in output_hashes.items():
        digest = _string(raw_digest, f"{raw_name} SHA-256")
        path = root / raw_name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"saved prediction output differs: {raw_name}")


def process_video(
    video: CheckedVideo,
    output_root: Path,
    bundle: ModelBundle,
    source_commit: str,
    repo_src: Path,
) -> Path:
    """Build and atomically publish one complete label-free prediction."""
    fixture = FixtureSpec(
        name=str(video.source.video_id),
        video_id=video.source.video_id,
        fps=EXPECTED_FPS,
        width=float(video.metadata.width),
        height=float(video.metadata.height),
    )
    final_directory = Path(output_root) / "videos" / f"ss22_{video.source.video_id:02d}"
    working_directory = final_directory.with_name(f"{final_directory.name}.working")
    if final_directory.exists() or working_directory.exists():
        raise FileExistsError(
            f"prediction output already exists for video {video.source.video_id}"
        )
    working_directory.mkdir(parents=True)
    result_path = working_directory / "result.json"
    _write_json(
        result_path,
        {
            "schema": VIDEO_RESULT_SCHEMA,
            "status": "running",
            "video_id": video.source.video_id,
            "source_commit": source_commit,
            "labels_read": False,
        },
    )

    inpaint_input = VideoInput(
        video.source.video_id,
        video.directory,
        video.source_video,
        video.directory / f"{video.directory.name}_ball.csv.gz",
        video.directory / "shuttle_track.npy.xz",
    )
    receipt = validate_completed(inpaint_input, video.directory.parent, repo_src)
    if receipt is None:
        raise ValueError(f"video {video.source.video_id}: inpaint output is incomplete")

    from annotator.video_metadata import VideoMetadata
    from dataset_builder.vision import (
        load_court_vision,
        load_pose_arrays,
        run_full_annotation_stage,
    )

    metadata = VideoMetadata(
        source_path=video.source_video.resolve(strict=True),
        fps=Fraction(30, 1),
        frame_count=video.metadata.frame_count,
        width=video.metadata.width,
        height=video.metadata.height,
    )
    track = load_npy_xz(video.directory / "shuttle_track_inpainted.npy.xz")
    sidecar_path = output_paths(video.directory, video.directory.name).sidecar_path
    sidecar = read_json_gz(sidecar_path)
    fill_mask = fill_mask_from_sidecar(sidecar, metadata.frame_count)
    if int(fill_mask.sum()) != receipt.get("selected_frame_count"):
        raise ValueError(f"video {video.source.video_id}: inpaint fill count differs")
    guard_codes = load_npy_xz(video.directory / "shuttle_guard_codes_inpainted.npy.xz")
    if guard_codes.shape != (metadata.frame_count,):
        raise ValueError(
            f"video {video.source.video_id}: shuttle guard timeline differs"
        )
    pose = load_pose_arrays(video.directory, metadata.frame_count)
    court = load_court_vision(
        video.directory,
        video_id=fixture.name,
        frame_count=metadata.frame_count,
        resolution=(float(metadata.width), float(metadata.height)),
    )

    annotation_directory = working_directory / "annotation"
    run_full_annotation_stage(
        video_id=fixture.name,
        metadata=metadata,
        track=track,
        inpaint_fill_mask=fill_mask,
        guard_codes=guard_codes,
        pose=pose,
        court=court,
        output_dir=annotation_directory,
    )

    with tempfile.TemporaryDirectory(
        prefix="stage-layout-", dir=working_directory
    ) as temporary:
        stage_root = Path(temporary)
        _link_stage_inputs(stage_root, video, fixture)
        annotation_parent = stage_root / "stages" / "annotation"
        annotation_parent.mkdir(parents=True)
        (annotation_parent / fixture.name).symlink_to(
            annotation_directory.resolve(strict=True)
        )
        features, feature_summary = _fixture_rows(
            stage_root,
            fixture,
            motion_mode="raw_per_frame",
        )
        if not len(features):
            raise ValueError(f"{fixture.name}: feature calculation returned no rows")
        if features.dtype.names is None or any(
            field not in features.dtype.names
            for field in REGION_FIELDS + bundle.input_fields
        ):
            raise ValueError(f"{fixture.name}: feature fields differ")
        names = np.char.decode(features["fixture"], "ascii")
        if not np.all(names == fixture.name) or not np.all(
            features["fps"] == EXPECTED_FPS
        ):
            raise ValueError(f"{fixture.name}: feature identity differs")
        identities = np.column_stack((features["interval_id"], features["frame"]))
        if len(np.unique(identities, axis=0)) != len(features):
            raise ValueError(f"{fixture.name}: feature frame identities are repeated")
        for interval_id in np.unique(features["interval_id"]):
            frames = features["frame"][features["interval_id"] == interval_id]
            if len(frames) > 1 and not np.all(np.diff(frames) > 0):
                raise ValueError(f"{fixture.name}: feature frames are not ordered")

        _write_npy_xz(working_directory / "contact_features.npy.xz", features)
        selected = np.zeros(len(features), dtype=bool)
        for field in REGION_FIELDS:
            selected |= features[field].astype(bool)
        if not selected.any():
            raise ValueError(
                f"{fixture.name}: no row is inside a contact search region"
            )
        candidates = features[selected]
        scores, predicted_frames = _score_candidates(candidates, fixture, bundle)
        _write_npy_xz(working_directory / "candidate_scores.npy.xz", scores)

        loaded_track, loaded_pose, loaded_court, _, sticky, annotation = _load_inputs(
            stage_root, fixture
        )
        court_inputs = loaded_court.evidence.inputs
        if court_inputs is None:
            raise ValueError(
                f"{fixture.name}: player-side court inputs are unavailable"
            )
        spans = [
            {"span_id": span_id, "start_frame": start, "end_frame": stop}
            for span_id, (start, stop) in enumerate(annotation.spans)
        ]
        selected_scores = scores[scores["kept"]]
        if not np.array_equal(selected_scores["frame"], predicted_frames):
            raise ValueError(f"{fixture.name}: kept score rows differ from predictions")
        contacts = []
        from annotator.point_winner import attribute_half

        for row in selected_scores:
            frame = int(row["frame"])
            side = _normalise_side(
                attribute_half(
                    frame,
                    loaded_track,
                    sticky,
                    loaded_pose.bboxes,
                    court_inputs.net_band,
                )
            )
            contacts.append(
                {
                    "frame": frame,
                    "contact_score": float(row["contact_score"]),
                    "predicted_side": side,
                    "span_id": _span_id(frame, spans),
                }
            )

    prediction_payload: dict[str, object] = {
        "schema": PREDICTION_SCHEMA,
        "video_id": video.source.video_id,
        "fixture": fixture.name,
        "fps": EXPECTED_FPS,
        "frame_count": metadata.frame_count,
        "spans": spans,
        "contacts": contacts,
    }
    prediction_path = working_directory / "predictions.json.gz"
    _write_json_gz(prediction_path, prediction_payload)

    output_paths_by_name = {
        "contact_features.npy.xz": working_directory / "contact_features.npy.xz",
        "candidate_scores.npy.xz": working_directory / "candidate_scores.npy.xz",
        "predictions.json.gz": prediction_path,
    }
    output_paths_by_name.update(
        {
            f"annotation/{name}": annotation_directory / name
            for name in ANNOTATION_FILENAMES
        }
    )
    output_hashes = {
        name: sha256(path) for name, path in sorted(output_paths_by_name.items())
    }
    result: dict[str, object] = {
        "schema": VIDEO_RESULT_SCHEMA,
        "status": "complete",
        "video_id": video.source.video_id,
        "fixture": fixture.name,
        "source_commit": source_commit,
        "labels_read": False,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "prepared_artifact_identity_sha256": ARTIFACT_IDENTITY_SHA256,
        "inpaint_run_state_sha256": INPAINT_RUN_STATE_SHA256,
        "model_sha256": MODEL_SHA256,
        "model_result_sha256": MODEL_RESULT_SHA256,
        "setting_result_sha256": SETTING_RESULT_SHA256,
        "model_input_fields": list(bundle.input_fields),
        "score_cutoff": SCORE_CUTOFF,
        "nearby_distance_at_30_fps": NEARBY_DISTANCE_AT_30_FPS,
        "frame_count": metadata.frame_count,
        "feature_summary": feature_summary,
        "candidate_row_count": len(scores),
        "kept_contact_count": len(contacts),
        "input_files": _input_records(video),
        "output_hashes": output_hashes,
    }
    _write_json(result_path, result)
    os.replace(working_directory, final_directory)
    return final_directory


def validate_prediction_directory(
    directory: Path,
    video: CheckedVideo,
    bundle: ModelBundle,
    source_commit: str,
) -> Mapping[str, Any]:
    """Reload one saved result and check its fixed decisions and file hashes."""
    root = Path(directory)
    result = _read_json(root / "result.json", "video prediction result")
    expected = {
        "schema": VIDEO_RESULT_SCHEMA,
        "status": "complete",
        "video_id": video.source.video_id,
        "fixture": str(video.source.video_id),
        "source_commit": source_commit,
        "labels_read": False,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "prepared_artifact_identity_sha256": ARTIFACT_IDENTITY_SHA256,
        "inpaint_run_state_sha256": INPAINT_RUN_STATE_SHA256,
        "model_sha256": MODEL_SHA256,
        "model_result_sha256": MODEL_RESULT_SHA256,
        "setting_result_sha256": SETTING_RESULT_SHA256,
        "model_input_fields": list(bundle.input_fields),
        "score_cutoff": SCORE_CUTOFF,
        "nearby_distance_at_30_fps": NEARBY_DISTANCE_AT_30_FPS,
        "frame_count": video.metadata.frame_count,
    }
    if any(result.get(name) != value for name, value in expected.items()):
        raise ValueError(
            f"video {video.source.video_id}: saved prediction identity differs"
        )
    _validate_input_records(video, result.get("input_files"))
    _validate_output_hashes(root, result.get("output_hashes"))

    fixture = FixtureSpec(
        str(video.source.video_id), video.source.video_id, EXPECTED_FPS
    )
    features = load_npy_xz(root / "contact_features.npy.xz")
    if len(features) == 0 or features.dtype.names is None:
        raise ValueError(f"video {video.source.video_id}: saved features are empty")
    if any(
        field not in features.dtype.names
        for field in REGION_FIELDS + bundle.input_fields
    ):
        raise ValueError(f"video {video.source.video_id}: saved feature fields differ")
    scores = load_npy_xz(root / "candidate_scores.npy.xz")
    if scores.dtype != SCORE_DTYPE or not np.isfinite(scores["contact_score"]).all():
        raise ValueError(
            f"video {video.source.video_id}: saved candidate scores differ"
        )
    if np.any((scores["contact_score"] < 0.0) | (scores["contact_score"] > 1.0)):
        raise ValueError(
            f"video {video.source.video_id}: saved candidate score range differs"
        )
    selected = np.zeros(len(features), dtype=bool)
    for field in REGION_FIELDS:
        selected |= features[field].astype(bool)
    candidate_features = features[selected]
    if len(scores) != len(candidate_features) or any(
        not np.array_equal(scores[field], candidate_features[field])
        for field in ("fixture", "interval_id", "frame", "fps")
    ):
        raise ValueError(
            f"video {video.source.video_id}: candidate score identities differ"
        )
    video_spec = VideoSpec(
        fixture=fixture.name,
        video_id=fixture.video_id,
        fps=fixture.fps,
        width=EXPECTED_WIDTH,
        height=EXPECTED_HEIGHT,
        role=SplitRole.VALIDATION,
        winner="unknown",
        loser="unknown",
        tournament="ShuttleSet22",
        tournament_round="unknown",
    )
    predictions, kept = predictions_for_settings(
        scores,
        (video_spec,),
        SCORE_CUTOFF,
        NEARBY_DISTANCE_AT_30_FPS,
    )
    if not np.array_equal(scores["kept"], kept):
        raise ValueError(f"video {video.source.video_id}: saved kept flags differ")

    prediction = read_json_gz(root / "predictions.json.gz")
    expected_prediction_identity = {
        "schema": PREDICTION_SCHEMA,
        "video_id": video.source.video_id,
        "fixture": fixture.name,
        "fps": EXPECTED_FPS,
        "frame_count": video.metadata.frame_count,
    }
    if any(
        prediction.get(name) != value
        for name, value in expected_prediction_identity.items()
    ):
        raise ValueError(
            f"video {video.source.video_id}: prediction payload identity differs"
        )
    contacts = _sequence(prediction.get("contacts"), "saved prediction contacts")
    contact_rows = [_mapping(row, "saved contact") for row in contacts]
    contact_frames = np.asarray(
        [_integer(row.get("frame"), "saved contact frame") for row in contact_rows],
        dtype=np.int32,
    )
    if not np.array_equal(contact_frames, predictions[fixture.name]):
        raise ValueError(f"video {video.source.video_id}: saved contact frames differ")
    kept_scores = scores[kept]
    saved_probabilities = np.asarray(
        [row.get("contact_score") for row in contact_rows],
        dtype=np.float64,
    )
    if not np.array_equal(saved_probabilities, kept_scores["contact_score"]):
        raise ValueError(f"video {video.source.video_id}: saved contact scores differ")
    annotation = read_annotation(
        root / "annotation" / "annotator_result.json.gz", fixture
    )
    expected_spans = [
        {"span_id": span_id, "start_frame": start, "end_frame": stop}
        for span_id, (start, stop) in enumerate(annotation.spans)
    ]
    if prediction.get("spans") != expected_spans:
        raise ValueError(
            f"video {video.source.video_id}: saved annotation spans differ"
        )
    for row in contact_rows:
        if row.get("predicted_side") not in {"Top", "Bot", None}:
            raise ValueError(
                f"video {video.source.video_id}: saved player side differs"
            )
        frame = _integer(row.get("frame"), "saved contact frame")
        if row.get("span_id") != _span_id(frame, expected_spans):
            raise ValueError(
                f"video {video.source.video_id}: saved contact span differs"
            )
    if result.get("candidate_row_count") != len(scores) or result.get(
        "kept_contact_count"
    ) != len(contacts):
        raise ValueError(f"video {video.source.video_id}: saved row counts differ")
    return result


def combined_prediction_bytes(
    checked: Sequence[CheckedVideo],
    output_root: Path,
    bundle: ModelBundle,
    source_commit: str,
) -> bytes:
    """Build deterministic combined bytes from checked per-video predictions."""
    videos: list[dict[str, object]] = []
    for video in checked:
        directory = Path(output_root) / "videos" / f"ss22_{video.source.video_id:02d}"
        validate_prediction_directory(directory, video, bundle, source_commit)
        videos.append(read_json_gz(directory / "predictions.json.gz"))
    payload: dict[str, object] = {
        "schema": COMBINED_SCHEMA,
        "status": "complete",
        "source_commit": source_commit,
        "labels_read": False,
        "video_ids": list(VIDEO_IDS),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "prepared_artifact_identity_sha256": ARTIFACT_IDENTITY_SHA256,
        "inpaint_run_state_sha256": INPAINT_RUN_STATE_SHA256,
        "model_sha256": MODEL_SHA256,
        "model_result_sha256": MODEL_RESULT_SHA256,
        "setting_result_sha256": SETTING_RESULT_SHA256,
        "score_cutoff": SCORE_CUTOFF,
        "nearby_distance_at_30_fps": NEARBY_DISTANCE_AT_30_FPS,
        "videos": videos,
    }
    return _gzip_json_bytes(payload)


def _write_run_state(
    path: Path,
    completed_ids: Sequence[int],
    combined_sha256: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema": RUN_STATE_SCHEMA,
        "status": (
            "complete"
            if tuple(completed_ids) == VIDEO_IDS and combined_sha256 is not None
            else "running"
        ),
        "expected_video_ids": list(VIDEO_IDS),
        "completed_video_ids": list(completed_ids),
        "completed_count": len(completed_ids),
        "combined_prediction_sha256": combined_sha256,
    }
    _write_json(path, payload)


def prepare_predictions(
    source_manifest: Path,
    inpaint_root: Path,
    source_root: Path,
    model_path: Path,
    model_result_path: Path,
    setting_result_path: Path,
    output_root: Path,
    source_commit: str,
    repo_src: Path,
    video_id: int | None = None,
) -> Path | None:
    """Resume the fixed label-free run and combine it when all videos exist."""
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a short or full Git commit")
    checked = check_input_set(source_manifest, inpaint_root, source_root)
    bundle = load_model_bundle(model_path, model_result_path, setting_result_path)
    Path(output_root, "videos").mkdir(parents=True, exist_ok=True)

    working = sorted(
        path.name for path in (Path(output_root) / "videos").glob("*.working")
    )
    if working:
        raise ValueError(f"unfinished prediction directories remain: {working}")
    selected = [
        video
        for video in checked
        if video_id is None or video.source.video_id == video_id
    ]
    if not selected:
        raise ValueError(f"video {video_id} is outside the fixed test set")
    for video in selected:
        directory = Path(output_root) / "videos" / f"ss22_{video.source.video_id:02d}"
        if directory.exists():
            validate_prediction_directory(directory, video, bundle, source_commit)
            print(f"checked existing video {video.source.video_id}", flush=True)
        else:
            process_video(video, output_root, bundle, source_commit, repo_src)
            validate_prediction_directory(directory, video, bundle, source_commit)
            print(f"completed video {video.source.video_id}", flush=True)
        completed_ids = [
            item.source.video_id
            for item in checked
            if (
                Path(output_root) / "videos" / f"ss22_{item.source.video_id:02d}"
            ).is_dir()
        ]
        _write_run_state(Path(output_root) / "run_state.json", completed_ids)

    completed_ids = [
        video.source.video_id
        for video in checked
        if (Path(output_root) / "videos" / f"ss22_{video.source.video_id:02d}").is_dir()
    ]
    if tuple(completed_ids) != VIDEO_IDS:
        return None
    first = combined_prediction_bytes(checked, output_root, bundle, source_commit)
    second = combined_prediction_bytes(checked, output_root, bundle, source_commit)
    if first != second:
        raise ValueError("repeated combined prediction bytes differ")
    combined_path = Path(output_root) / "combined_predictions.json.gz"
    _write_verified_bytes(combined_path, first)
    combined_hash = sha256(combined_path)
    _write_run_state(Path(output_root) / "run_state.json", completed_ids, combined_hash)
    return combined_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--inpaint-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-result", type=Path, required=True)
    parser.add_argument("--setting-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repo-src", type=Path, default=REPO_ROOT / "src")
    parser.add_argument("--video-id", type=int, choices=VIDEO_IDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    combined = prepare_predictions(
        arguments.source_manifest,
        arguments.inpaint_root,
        arguments.source_root,
        arguments.model,
        arguments.model_result,
        arguments.setting_result,
        arguments.output_root,
        arguments.source_commit,
        arguments.repo_src,
        arguments.video_id,
    )
    if combined is None:
        print("selected videos complete; combined prediction is not ready", flush=True)
    else:
        print(f"saved {combined}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
