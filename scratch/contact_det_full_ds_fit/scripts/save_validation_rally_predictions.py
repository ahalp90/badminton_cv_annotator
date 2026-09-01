"""Save player-side answers for the nine fixed validation predictions.

This step uses only saved model results and outputs from the video pipeline. It writes all
contact frames, scores, player-side answers and rally spans before any
ShuttleSet contact or player-side label row is read.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scratch.contact_det.scripts.freeze_contact_evidence import (
    FixtureSpec,
    _load_inputs,
    _stage_paths,
)
from scratch.contact_det.scripts.freeze_tree_contact_features import _player_signals
from scratch.contact_det_full_ds_fit.scripts.baseline_results import (
    VerifiedBaselineMenu,
    load_completed_baseline_menu,
)

PREDICTION_SCHEMA = "full-dataset-contact-rally-predictions/1"
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
CENTRE_FEATURE_FIELDS = {
    "shuttle_x": "shuttle_x",
    "shuttle_y": "shuttle_y",
    "standing_count": "standing_count",
    "wrist_gap_min_t+0": "wrist_gap_min",
    "wrist_gap_top_t+0": "wrist_gap_top",
    "wrist_gap_bot_t+0": "wrist_gap_bot",
    "nearest_wrist_dx_t+0": "nearest_wrist_dx",
    "nearest_wrist_dy_t+0": "nearest_wrist_dy",
    "ankle_x_top": "ankle_x_top",
    "ankle_y_top": "ankle_y_top",
    "ankle_x_bot": "ankle_x_bot",
    "ankle_y_bot": "ankle_y_bot",
    "bbox_height_top": "bbox_height_top",
    "bbox_height_bot": "bbox_height_bot",
    "shuttle_visible_t+0": "shuttle_visible",
    "pose_valid_top_t+0": "pose_valid_top",
    "pose_valid_bot_t+0": "pose_valid_bot",
    "wrist_valid_top_t+0": "wrist_valid_top",
    "wrist_valid_bot_t+0": "wrist_valid_bot",
}

MenuLoader = Callable[[Path, Path, Path, Path, Path, Path], VerifiedBaselineMenu]
InputLoader = Callable[[Path, FixtureSpec], tuple[np.ndarray, Any, Any, list[tuple[int, int]], Any, Any]]
SideAttributor = Callable[[int, np.ndarray, Any, np.ndarray, tuple[float, float]], object]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be an object with string keys")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encoded_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    encoded = _encoded_json(value)
    if destination.name.endswith(".gz"):
        with temporary.open("wb") as raw, gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=9,
            mtime=0,
        ) as zipped:
            zipped.write(encoded)
    else:
        temporary.write_bytes(encoded)
    os.replace(temporary, destination)


def _fixture(video: Any) -> FixtureSpec:
    return FixtureSpec(
        name=video.fixture,
        video_id=video.video_id,
        fps=video.fps,
        width=float(video.width),
        height=float(video.height),
    )


def _video_feature_record(verified: VerifiedBaselineMenu, video_name: str) -> Mapping[str, Any]:
    raw_videos = verified.raw_features.record.get("videos")
    if not isinstance(raw_videos, list):
        raise TypeError("raw feature video records must be a list")
    matches = [
        _mapping(record, f"{video_name}: raw feature record")
        for record in raw_videos
        if isinstance(record, Mapping)
        and isinstance(record.get("video"), Mapping)
        and record["video"].get("name") == video_name
    ]
    if len(matches) != 1:
        raise ValueError(f"{video_name}: raw feature record is missing or repeated")
    return matches[0]


def _checked_stage_files(
    data_root: Path,
    fixture: FixtureSpec,
    feature_record: Mapping[str, Any],
) -> list[dict[str, object]]:
    raw_files = feature_record.get("input_files")
    if not isinstance(raw_files, list):
        raise TypeError(f"{fixture.name}: feature input files must be a list")
    by_role: dict[str, Mapping[str, Any]] = {}
    for raw_file in raw_files:
        record = _mapping(raw_file, f"{fixture.name}: feature input file")
        role = record.get("role")
        if not isinstance(role, str) or role in by_role:
            raise ValueError(f"{fixture.name}: feature input file roles differ")
        by_role[role] = record

    checked: list[dict[str, object]] = []
    for role, stage_path in _stage_paths(data_root, fixture).items():
        record = by_role.get(role)
        if record is None:
            raise ValueError(f"{fixture.name}: {role} input record is missing")
        if record.get("filename") != stage_path.name:
            raise ValueError(f"{fixture.name}: {role} filename differs")
        size_bytes = record.get("size_bytes")
        if type(size_bytes) is not int or stage_path.stat().st_size != size_bytes:
            raise ValueError(f"{fixture.name}: {role} file size differs")
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or _sha256(stage_path) != expected_hash:
            raise ValueError(f"{fixture.name}: {role} file hash differs")
        checked.append(
            {
                "role": role,
                "filename": stage_path.name,
                "size_bytes": size_bytes,
                "sha256": expected_hash,
            }
        )
    return checked


def _predicted_frames(verified: VerifiedBaselineMenu, video_name: str) -> np.ndarray:
    chunks = [run.predictions[video_name] for run in verified.runs]
    if not chunks:
        return np.empty(0, dtype=np.int32)
    return np.unique(np.concatenate(chunks)).astype(np.int32, copy=False)


def _feature_rows_for_frames(
    verified: VerifiedBaselineMenu,
    video_name: str,
    frames: np.ndarray,
) -> np.ndarray:
    start, end = verified.raw_features.video_ranges[video_name]
    video_rows = verified.raw_features.rows[start:end]
    if len(np.unique(video_rows["frame"])) != len(video_rows):
        raise ValueError(f"{video_name}: raw feature frames are repeated")
    indices = np.searchsorted(video_rows["frame"], frames)
    if np.any(indices >= len(video_rows)) or not np.array_equal(video_rows["frame"][indices], frames):
        raise ValueError(f"{video_name}: a predicted frame has no raw feature row")
    return video_rows[indices]


def _same_float_values(actual: np.ndarray, expected: np.ndarray) -> bool:
    actual_values = np.asarray(actual, dtype=np.float32)
    expected_values = np.asarray(expected, dtype=np.float32)
    return bool(np.array_equal(np.isnan(actual_values), np.isnan(expected_values))) and bool(
        np.array_equal(actual_values[~np.isnan(actual_values)], expected_values[~np.isnan(expected_values)])
    )


def _check_centre_feature_values(
    video_name: str,
    rows: np.ndarray,
    frames: np.ndarray,
    track: np.ndarray,
    pose: Any,
    sticky: Any,
    resolution: tuple[float, float],
) -> None:
    if rows.dtype.names is None or any(field not in rows.dtype.names for field in CENTRE_FEATURE_FIELDS):
        raise ValueError(f"{video_name}: raw feature rows lack a checked centre-frame value")
    visible = track[:, 2] == 1
    expected: dict[str, np.ndarray] = {
        "shuttle_x": np.where(visible, track[:, 0], np.nan),
        "shuttle_y": np.where(visible, track[:, 1], np.nan),
        "shuttle_visible": visible.astype(np.float32),
        "standing_count": np.asarray(sticky.standing_count, dtype=np.float32),
    }
    expected.update(_player_signals(track, pose.kps, sticky, resolution))
    for feature_field, expected_name in CENTRE_FEATURE_FIELDS.items():
        if not _same_float_values(rows[feature_field], expected[expected_name][frames]):
            raise ValueError(f"{video_name}: {feature_field} differs from the replay input")


def _normalise_side(value: object, label: str) -> str | None:
    if value is None:
        return None
    side = getattr(value, "value", value)
    if side == "Top":
        return "Top"
    if side in {"Bot", "Bottom"}:
        return "Bot"
    raise ValueError(f"{label}: player-side answer differs")


def _spans(annotation: Any, video_name: str) -> list[dict[str, int]]:
    raw_spans = getattr(annotation, "spans", None)
    if not isinstance(raw_spans, tuple):
        raise TypeError(f"{video_name}: annotation spans are unavailable")
    return [
        {"span_id": span_id, "start_frame": start, "end_frame": end}
        for span_id, (start, end) in enumerate(raw_spans)
    ]


def _span_id(frame: int, spans: Sequence[Mapping[str, int]]) -> int | None:
    matches = [
        int(span["span_id"])
        for span in spans
        if int(span["start_frame"]) <= frame < int(span["end_frame"])
    ]
    if len(matches) > 1:
        raise ValueError(f"frame {frame} belongs to overlapping rally spans")
    return matches[0] if matches else None


def _video_predictions(
    verified: VerifiedBaselineMenu,
    video_name: str,
    sides: Mapping[int, str | None],
    spans: Sequence[Mapping[str, int]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    encoded_name = video_name.encode("ascii")
    for checked_run in verified.runs:
        selected = checked_run.score_rows[
            (checked_run.score_rows["fixture"] == encoded_name) & checked_run.kept
        ]
        frames = selected["frame"].astype(np.int32, copy=False)
        expected_frames = checked_run.predictions[video_name]
        if not np.array_equal(frames, expected_frames):
            raise ValueError(f"{checked_run.run.run_id}/{video_name}: kept score rows differ")
        contacts = [
            {
                "frame": int(row["frame"]),
                "timing_score": float(row["contact_score"]),
                "predicted_side": sides[int(row["frame"])],
                "span_id": _span_id(int(row["frame"]), spans),
            }
            for row in selected
        ]
        output.append({"run_id": checked_run.run.run_id, "contacts": contacts})
    return output


def build_validation_rally_predictions(
    verified: VerifiedBaselineMenu,
    data_root: Path,
    source_commit: str,
    *,
    input_loader: InputLoader = _load_inputs,
    side_attributor: SideAttributor | None = None,
) -> dict[str, object]:
    """Replay the player-side rule and return a path-free saved result."""
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a short or full Git commit")
    if side_attributor is None:
        from annotator.point_winner import attribute_half

        side_attributor = attribute_half

    saved_videos: list[dict[str, object]] = []
    contacts_by_run = {checked_run.run.run_id: [] for checked_run in verified.runs}
    for video in verified.split.validation_videos:
        fixture = _fixture(video)
        feature_record = _video_feature_record(verified, video.fixture)
        checked_files = _checked_stage_files(Path(data_root), fixture, feature_record)
        track, pose, court, tracker_intervals, sticky, annotation = input_loader(
            Path(data_root), fixture
        )
        court_inputs = getattr(getattr(court, "evidence", None), "inputs", None)
        if court_inputs is None:
            raise ValueError(f"{video.fixture}: court inputs are unavailable")
        net_band = tuple(float(value) for value in court_inputs.net_band)
        if len(net_band) != 2 or not np.all(np.isfinite(net_band)) or net_band[0] > net_band[1]:
            raise ValueError(f"{video.fixture}: net band is invalid")
        if len(track) != feature_record["feature_summary"].get("frame_count"):
            raise ValueError(f"{video.fixture}: replay frame count differs")

        frames = _predicted_frames(verified, video.fixture)
        rows = _feature_rows_for_frames(verified, video.fixture, frames)
        _check_centre_feature_values(
            video.fixture,
            rows,
            frames,
            track,
            pose,
            sticky,
            (float(video.width), float(video.height)),
        )
        sides = {
            int(frame): _normalise_side(
                side_attributor(int(frame), track, sticky, pose.bboxes, net_band),
                f"{video.fixture}/{frame}",
            )
            for frame in frames
        }
        video_spans = _spans(annotation, video.fixture)
        if len(video_spans) != feature_record["feature_summary"].get("rally_span_count"):
            raise ValueError(f"{video.fixture}: rally span count differs")
        for run_result in _video_predictions(verified, video.fixture, sides, video_spans):
            contacts_by_run[str(run_result["run_id"])].append(
                {"fixture": video.fixture, "contacts": run_result["contacts"]}
            )
        saved_videos.append(
            {
                "fixture": video.fixture,
                "video_id": video.video_id,
                "fps": video.fps,
                "frame_count": len(track),
                "spans": video_spans,
                "replayed_contact_count": len(frames),
                "input_files": checked_files,
            }
        )
        # The next video can have large pose arrays. Drop this video's arrays
        # before its loader starts.
        del track, pose, court, tracker_intervals, sticky, annotation, court_inputs, rows

    return {
        "schema": PREDICTION_SCHEMA,
        "status": "complete",
        "source_commit": source_commit,
        "labels_read": False,
        "menu_result_file": verified.menu_path.name,
        "menu_result_sha256": _sha256(verified.menu_path),
        "split_file": verified.menu["split_file"],
        "split_sha256": verified.menu["split_sha256"],
        "raw_feature_record_file": verified.raw_features.record_path.name,
        "raw_feature_record_sha256": _sha256(verified.raw_features.record_path),
        "contact_label_file": verified.menu["contact_label_file"],
        "contact_label_sha256": verified.menu["contact_label_sha256"],
        "validation_videos": [video.fixture for video in verified.split.validation_videos],
        "centre_feature_fields_checked": list(CENTRE_FEATURE_FIELDS),
        "videos": saved_videos,
        "runs": [
            {"run_id": checked_run.run.run_id, "videos": contacts_by_run[checked_run.run.run_id]}
            for checked_run in verified.runs
        ],
    }


def save_validation_rally_predictions(
    menu_result_path: Path,
    config_path: Path,
    split_path: Path,
    raw_feature_record_path: Path,
    common30_feature_record_path: Path,
    shots_master_path: Path,
    data_root: Path,
    output_path: Path,
    source_commit: str,
    *,
    menu_loader: MenuLoader = load_completed_baseline_menu,
    input_loader: InputLoader = _load_inputs,
    side_attributor: SideAttributor | None = None,
) -> Path:
    """Check the saved runs, replay player sides and write the complete result."""
    destination = Path(output_path)
    _write_json(
        destination,
        {
            "schema": PREDICTION_SCHEMA,
            "status": "running",
            "source_commit": source_commit,
            "labels_read": False,
        },
    )
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a short or full Git commit")
    verified = menu_loader(
        Path(menu_result_path),
        Path(config_path),
        Path(split_path),
        Path(raw_feature_record_path),
        Path(common30_feature_record_path),
        Path(shots_master_path),
    )
    result = build_validation_rally_predictions(
        verified,
        Path(data_root),
        source_commit,
        input_loader=input_loader,
        side_attributor=side_attributor,
    )
    _write_json(destination, result)
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--menu-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--raw-feature-record", type=Path, required=True)
    parser.add_argument("--common30-feature-record", type=Path, required=True)
    parser.add_argument("--shots-master", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    output = save_validation_rally_predictions(
        arguments.menu_result,
        arguments.config,
        arguments.split,
        arguments.raw_feature_record,
        arguments.common30_feature_record,
        arguments.shots_master,
        arguments.data_root,
        arguments.output,
        arguments.source_commit,
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
