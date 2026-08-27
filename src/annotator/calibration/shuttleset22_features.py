"""Evaluate issue #22 feature prototypes on completed ShuttleSet22 artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from annotator.calibration.shuttleset_features import (
    derive_player_feature_inputs,
    evaluate_rally_features,
    feature_population,
)
from annotator.shuttle_track import validate_shuttle_track
from dataset_builder.vision import (
    COURT_EVIDENCE_FILENAME,
    COURT_KEEP_VOTE_FILENAME,
    COURT_PRESENT_FILENAME,
    POSE_FILENAMES,
    load_court_vision,
    load_json_gz,
    load_npy_xz,
    load_pose_arrays,
    save_json_gz,
)
from shuttleset22 import Source, SourceKind, load_sources


RESULT_SCHEMA = "shuttleset22-trial-feature-comparison/2"
COURT_RECEIPT_SCHEMA = "shuttleset22-court/0.1"
ISSUE106_HANDOFF_COMMIT = "ba24a95c334300c78e30a8d1b7c2a6134b8b5fa9"
ISSUE120_COURT_COMMIT = "0c873762d85719f65d6898b22ea2fc6b6327066a"
ANNOTATION_UPSTREAM_COMMIT = "45517f7d4cb936b03f3eabf939cc7959d39226fe"
ANNOTATION_SHA256 = "2c0208d13d13a4b72a9005ec16e92c442bfe5f223e0f9c499ea5a36f4339052c"
SOURCE_MANIFEST_SHA256 = "746225f6b9bb1b257052224648c39e813792a75a7eb8711443688ca93fad7463"
ANNOTATION_TREE_SHA256 = "55f832221646229b8b65dea31e24e8d02e0876fd6d0799cb0f6eff12583e1485"
ARTIFACT_IDENTITY_SHA256 = "dffe2cc2afc75f78eb89b30236477eb732f92a824b22ee3a01a4f893a673864e"
EXPECTED_FPS = 30.0


@dataclass(frozen=True)
class ArtifactMetadata:
    """Validated metadata and identities for one consumed artifact directory."""

    frame_count: int
    width: int
    height: int
    receipt_sha256: str
    shuttle_sha256: str
    court_code_id: str
    court_model_md5: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    source = Path(root).resolve(strict=True)
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_digest(name: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} SHA-256 differs: expected {expected}, found {actual}")


def _validate_identity(path: Path, identity: Mapping[str, object]) -> None:
    """Validate one receipt identity without trusting its size or MD5."""
    expected = {"name", "path", "size_bytes", "md5"}
    if set(identity) != expected:
        raise ValueError(f"artifact identity fields differ for {path.name}")
    if identity["path"] != path.name:
        raise ValueError(f"artifact receipt path differs for {path.name}")
    if identity["size_bytes"] != path.stat().st_size:
        raise ValueError(f"artifact size differs for {path.name}")
    if identity["md5"] != _md5(path):
        raise ValueError(f"artifact MD5 differs for {path.name}")


def validate_artifact_directory(output: Path, source: Source) -> ArtifactMetadata:
    """Validate every artifact consumed by the feature comparison."""
    receipt_path = output / "court_receipt.json.gz"
    receipt = load_json_gz(receipt_path)
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
    if set(receipt) != required or receipt["schema"] != COURT_RECEIPT_SCHEMA:
        raise ValueError(f"{source.match_id:02d}: unsupported court receipt")
    if receipt["match_id"] != source.match_id or receipt["video"] != source.video:
        raise ValueError(f"{source.match_id:02d}: court receipt identity differs")
    if receipt["completed"] is not True:
        raise ValueError(f"{source.match_id:02d}: court receipt is incomplete")

    metadata = _mapping(receipt["metadata"], "court metadata")
    if (metadata["fps_numerator"], metadata["fps_denominator"]) != (30, 1):
        raise ValueError(f"{source.match_id:02d}: expected exact 30 FPS metadata")
    frame_count = _positive_integer(metadata["frame_count"], "frame_count")
    width = _positive_integer(metadata["width"], "width")
    height = _positive_integer(metadata["height"], "height")

    pose_filenames = {
        f"pose_{name}": filename for name, filename in POSE_FILENAMES.items()
    }
    input_rows = {
        _string(_mapping(row, "receipt input")["name"], "input name"):
        _mapping(row, "receipt input")
        for row in _sequence(receipt["inputs"], "receipt inputs")
    }
    if not set(pose_filenames).issubset(input_rows):
        raise ValueError(f"{source.match_id:02d}: pose identities are incomplete")
    for name, filename in pose_filenames.items():
        _validate_identity(output / filename, input_rows[name])

    output_filenames = {
        "court_evidence": COURT_EVIDENCE_FILENAME,
        "court_keep_vote": COURT_KEEP_VOTE_FILENAME,
        "court_present": COURT_PRESENT_FILENAME,
    }
    output_rows = {
        _string(_mapping(row, "receipt output")["name"], "output name"):
        _mapping(row, "receipt output")
        for row in _sequence(receipt["outputs"], "receipt outputs")
    }
    if set(output_rows) != set(output_filenames):
        raise ValueError(f"{source.match_id:02d}: court identities are incomplete")
    for name, filename in output_filenames.items():
        _validate_identity(output / filename, output_rows[name])

    model = _mapping(receipt["model"], "court model")
    return ArtifactMetadata(
        frame_count=frame_count,
        width=width,
        height=height,
        receipt_sha256=_sha256(receipt_path),
        shuttle_sha256=_sha256(output / "shuttle_track.npy.xz"),
        court_code_id=_string(receipt["code_id"], "court code_id"),
        court_model_md5=_string(model["md5"], "court model MD5"),
    )


def load_annotation_rallies(
    set_dir: Path, frame_count: int
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Build feature records from usable ShuttleSet22 human contact rows."""
    tables = []
    for path in sorted(Path(set_dir).glob("set*.csv")):
        table = pd.read_csv(path)
        table["set_id"] = path.stem
        tables.append(table)
    if not tables:
        raise ValueError(f"no ShuttleSet22 set tables under {set_dir}")
    contacts = pd.concat(tables, ignore_index=True)
    frames = pd.to_numeric(contacts["frame_num"], errors="coerce")
    invalid_frame = frames.isna() | (frames < 0) | (frames >= frame_count)
    flaw_marked = contacts["flaw"].notna()
    rows: list[dict[str, object]] = []
    incomplete = 0
    incomplete_rows = 0
    non_monotonic = 0
    non_monotonic_rows = 0
    for (set_id, rally), group in contacts.groupby(["set_id", "rally"], sort=True):
        group_invalid = invalid_frame.loc[group.index]
        group_flaw = flaw_marked.loc[group.index]
        if bool((group_invalid | group_flaw).any()):
            incomplete += 1
            incomplete_rows += len(group)
            continue
        group = group.copy()
        group["frame_num"] = frames.loc[group.index].astype(int)
        ordered = group.sort_values(["ball_round", "frame_num"])
        contact_frames = ordered["frame_num"].tolist()
        if not contact_frames or any(
            right <= left for left, right in zip(contact_frames, contact_frames[1:])
        ):
            non_monotonic += 1
            non_monotonic_rows += len(group)
            continue
        first = ordered.iloc[0]
        server_prediction = _player_slot(first)
        rows.append(
            {
                "rally": {
                    "rally_id": len(rows),
                    "start_frame": contact_frames[0],
                    "end_frame": contact_frames[-1] + 1,
                },
                "contacts": {
                    "accepted": [
                        {"contact_frame": frame} for frame in contact_frames
                    ],
                    "stroke_count": len(contact_frames),
                },
                "outcomes": {"server_prediction": server_prediction},
                "source": {
                    "set_id": str(set_id),
                    "rally": int(rally),
                    "ball_rounds": [int(value) for value in ordered["ball_round"]],
                    "contact_types": [
                        None if pd.isna(value) else str(value)
                        for value in ordered["type"]
                    ],
                },
            }
        )
    population = {
        "source_contact_rows": len(contacts),
        "usable_contact_rows": sum(
            int(_mapping(row["contacts"], "contacts")["stroke_count"])
            for row in rows
        ),
        "excluded_flaw_rows": int(flaw_marked.sum()),
        "excluded_invalid_frame_rows": int((invalid_frame & ~flaw_marked).sum()),
        "usable_rallies": len(rows),
        "excluded_incomplete_rallies": incomplete,
        "excluded_incomplete_rally_rows": incomplete_rows,
        "excluded_non_monotonic_rallies": non_monotonic,
        "excluded_non_monotonic_rally_rows": non_monotonic_rows,
    }
    return rows, population


def _player_slot(row: pd.Series) -> str | None:
    player_y = pd.to_numeric(pd.Series([row["player_location_y"]]), errors="coerce").iloc[0]
    opponent_y = pd.to_numeric(pd.Series([row["opponent_location_y"]]), errors="coerce").iloc[0]
    if not np.isfinite(player_y) or not np.isfinite(opponent_y) or player_y == opponent_y:
        return None
    return "Top" if player_y < opponent_y else "Bot"


def evaluate_source(data_root: Path, source: Source) -> dict[str, object]:
    """Validate and evaluate one non-overlap ShuttleSet22 source."""
    output = data_root / "extracted-simple" / f"{source.match_id:02d} {source.video}"
    metadata = validate_artifact_directory(output, source)
    pose = load_pose_arrays(output, metadata.frame_count)
    track = load_npy_xz(output / "shuttle_track.npy.xz")
    validate_shuttle_track(track, metadata.frame_count)
    court = load_court_vision(
        output,
        video_id=str(source.match_id),
        frame_count=metadata.frame_count,
        resolution=(float(metadata.width), float(metadata.height)),
    )
    if len(court.raw_cuts) != _positive_integer(
        load_json_gz(output / "court_receipt.json.gz")["scene_count"], "scene_count"
    ):
        raise ValueError(f"{source.match_id:02d}: court scene count differs")
    player_inputs = derive_player_feature_inputs(
        track, pose, court, str(source.match_id)
    )
    records, annotation_population = load_annotation_rallies(
        data_root / "annotations" / "set" / source.video,
        metadata.frame_count,
    )
    rallies = []
    for record in records:
        feature = evaluate_rally_features(
            record,
            player_inputs.posture,
            player_inputs.court_positions,
            player_inputs.posture_interpolation,
            player_inputs.position_interpolation,
            EXPECTED_FPS,
        )
        feature["source"] = record["source"]
        rallies.append(feature)
    return {
        "match_id": source.match_id,
        "video": source.video,
        "fps": EXPECTED_FPS,
        "frame_count": metadata.frame_count,
        "annotation_population": annotation_population,
        "primitive_population": {
            "shuttle_visible_frames": int(np.count_nonzero(track[:, 2] == 1)),
            "court_present_frames": int(np.count_nonzero(court.evidence.court_present)),
            "pose_frames_with_two_detections": int(np.count_nonzero(pose.ndet >= 2)),
        },
        "artifact_identity": {
            "court_receipt_sha256": metadata.receipt_sha256,
            "shuttle_track_sha256": metadata.shuttle_sha256,
            "court_code_id": metadata.court_code_id,
            "court_model_md5": metadata.court_model_md5,
        },
        "population": feature_population(rallies),
        "summary": feature_summary(rallies),
        "rallies": rallies,
    }


def feature_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return concise distributions for defined numeric feature outputs."""
    values: dict[str, list[float]] = {
        "shots_per_rally": [],
        "posture_mad": [],
        "recovery_distance": [],
        "movement_inefficiency": [],
    }
    for row in rows:
        values["shots_per_rally"].append(float(row["shots_per_rally"]))
        for posture in _sequence(row["posture"], "posture rows"):
            value = _mapping(posture, "posture row")["mad"]
            if value is not None:
                values["posture_mad"].append(float(value))
        recovery = _mapping(row["recovery"], "recovery")
        for observation in _sequence(recovery["observations"], "recovery observations"):
            value = _mapping(observation, "recovery observation")["mean_distance"]
            if value is not None:
                values["recovery_distance"].append(float(value))
        for interval in _sequence(row["movement_inefficiency"], "movement rows"):
            movement = _mapping(interval, "movement row")
            for slot in ("top", "bottom"):
                value = movement[slot]
                if value is not None:
                    values["movement_inefficiency"].append(float(value))
    return {name: _distribution(samples) for name, samples in values.items()}


def leave_one_video_out(
    per_video: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, int | float | None]]:
    """Report the median range after excluding each source once."""
    medians: dict[str, list[float]] = {
        "shots_per_rally": [],
        "posture_mad": [],
        "recovery_distance": [],
        "movement_inefficiency": [],
    }
    for excluded in per_video:
        retained = []
        for video_id, video in per_video.items():
            if video_id != excluded:
                retained.extend(_sequence(video["rallies"], "rallies"))
        summary = feature_summary(retained)
        for name, values in summary.items():
            median = _mapping(values, f"{name} summary")["median"]
            if median is not None:
                medians[name].append(float(median))
    return {
        name: {
            "runs": len(values),
            "min_median": None if not values else min(values),
            "max_median": None if not values else max(values),
        }
        for name, values in medians.items()
    }


def _distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    samples = np.asarray(values, dtype=float)
    if not len(samples):
        return {"eligible": 0, "median": None, "p10": None, "p90": None}
    return {
        "eligible": len(samples),
        "median": float(np.median(samples)),
        "p10": float(np.percentile(samples, 10)),
        "p90": float(np.percentile(samples, 90)),
    }


def evaluate_corpus(data_root: Path, source_manifest: Path) -> dict[str, object]:
    """Evaluate every available non-overlap ShuttleSet22 source once."""
    source_manifest_digest = _sha256(source_manifest)
    annotation_tree_digest = _tree_digest(data_root / "annotations")
    _require_digest("source manifest", source_manifest_digest, SOURCE_MANIFEST_SHA256)
    _require_digest("annotation tree", annotation_tree_digest, ANNOTATION_TREE_SHA256)
    sources = tuple(
        source for source in load_sources(source_manifest) if source.kind is SourceKind.DOWNLOAD
    )
    if len(sources) != 47:
        raise ValueError(f"expected 47 non-overlap ShuttleSet22 sources, found {len(sources)}")
    per_video: dict[str, dict[str, object]] = {}
    for source in sources:
        print(f"{source.match_id:02d}: validating and extracting features", flush=True)
        per_video[f"{source.match_id:02d}"] = evaluate_source(data_root, source)
    all_rallies = [
        rally
        for video in per_video.values()
        for rally in _sequence(video["rallies"], "rallies")
    ]
    identity_payload = {
        video_id: video["artifact_identity"] for video_id, video in per_video.items()
    }
    identity_json = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    artifact_identity_digest = hashlib.sha256(identity_json.encode()).hexdigest()
    _require_digest(
        "artifact identity",
        artifact_identity_digest,
        ARTIFACT_IDENTITY_SHA256,
    )
    annotation_population = _sum_population(per_video, "annotation_population")
    primitive_population = _sum_population(per_video, "primitive_population")
    return {
        "schema": RESULT_SCHEMA,
        "provenance": {
            "issue106_handoff_commit": ISSUE106_HANDOFF_COMMIT,
            "issue120_court_commit": ISSUE120_COURT_COMMIT,
            "annotation_upstream_commit": ANNOTATION_UPSTREAM_COMMIT,
            "annotation_manifest_sha256": ANNOTATION_SHA256,
            "annotation_tree_sha256": annotation_tree_digest,
            "source_manifest_sha256": source_manifest_digest,
            "artifact_identity_sha256": artifact_identity_digest,
            "evaluator_files_sha256": {
                "shuttleset22_features.py": _sha256(Path(__file__)),
                "shuttleset_features.py": _sha256(Path(__file__).with_name("shuttleset_features.py")),
            },
        },
        "policy": {
            "rally_frames": "human contacts from first frame through final frame inclusive",
            "contact_frames": "human ShuttleSet22 frame_num after flaw and range exclusions",
            "player_attribution": "first striker top/bottom slot from human player and opponent image y",
            "rally_duration_end_offset": "unresolved",
            "serve_endpoint_and_static_tolerance": "unresolved",
            "degradation_temperature": "unresolved",
            "backward_extrapolation": "unresolved",
            "commentary": "unavailable",
        },
        "population": {
            "videos": len(per_video),
            "frames": sum(int(video["frame_count"]) for video in per_video.values()),
            **feature_population(all_rallies),
        },
        "annotation_population": annotation_population,
        "primitive_population": primitive_population,
        "summary": feature_summary(all_rallies),
        "leave_one_video_out": leave_one_video_out(per_video),
        "per_video": per_video,
    }


def _sum_population(
    per_video: Mapping[str, Mapping[str, object]], name: str
) -> dict[str, int]:
    rows = [_mapping(video[name], name) for video in per_video.values()]
    keys = set(rows[0]) if rows else set()
    if any(set(row) != keys for row in rows):
        raise ValueError(f"{name} fields differ across videos")
    return {key: sum(int(row[key]) for row in rows) for key in sorted(keys)}


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the exact ShuttleSet22 feature comparison from the command line."""
    arguments = _parse_args()
    result = evaluate_corpus(
        arguments.data_root.resolve(strict=True),
        arguments.source_manifest.resolve(strict=True),
    )
    save_json_gz(arguments.output, result)
    print(f"saved {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
