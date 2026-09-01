"""Save the pilot contact features for the videos in an accepted split file.

This script reads predictions from the existing video pipeline. It does not
read ShuttleSet contact labels. Each video gets its own feature file and record
so a stopped run leaves clear, usable progress.
"""

# ruff: noqa: E402, RUF100

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PILOT_SCRIPT_ROOT = REPO_ROOT / "scratch" / "contact_det" / "scripts"
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(PILOT_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(PILOT_SCRIPT_ROOT))

from freeze_contact_evidence import FixtureSpec, _sha256, _stage_paths
from freeze_tree_contact_features import (
    FEATURE_SCHEMA,
    IDENTITY_FIELDS,
    MOTION_MODES,
    _feature_family_names,
    _fixture_rows,
    _motion_scale_factor,
    _write_npy_xz,
)

from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    DevelopmentSplit,
    VideoSpec,
    load_development_split,
    verify_accepted_development_split,
    verify_against_shuttleset_tables,
)

RUN_RECORD_SCHEMA = "full-dataset-contact-features/1"
VIDEO_RECORD_SCHEMA = "full-dataset-contact-features-video/1"
PILOT_CHECK_SCHEMA = "full-dataset-contact-features-pilot-check/1"
FEATURE_FILENAME = "contact_features.npy.xz"
VIDEO_RECORD_FILENAME = "contact_features_record.json"
RUN_RECORD_FILENAME = "contact_features_record.json"
PILOT_CHECK_FILENAME = "pilot_feature_check.json"
PILOT_VIDEO_NAMES = ("sset_01", "sset_15", "sset_21")

RowBuilder = Callable[[Path, FixtureSpec, str], tuple[np.ndarray, dict[str, Any]]]


def _selected_videos(
    split: DevelopmentSplit,
    requested_names: Sequence[str],
) -> tuple[VideoSpec, ...]:
    if len(set(requested_names)) != len(requested_names):
        raise ValueError("requested video names must be unique")
    if not requested_names:
        return split.videos

    requested = set(requested_names)
    unknown = requested - split.by_fixture.keys()
    if unknown:
        raise ValueError(f"requested videos are absent from the split: {sorted(unknown)}")
    return tuple(video for video in split.videos if video.fixture in requested)


def _as_fixture(video: VideoSpec) -> FixtureSpec:
    return FixtureSpec(
        name=video.fixture,
        video_id=video.video_id,
        fps=video.fps,
        width=float(video.width),
        height=float(video.height),
    )


def _input_files(data_root: Path, fixture: FixtureSpec) -> list[dict[str, object]]:
    paths = _stage_paths(data_root, fixture)
    paths["definitive_exclusion_mask"] = (
        Path(data_root)
        / "stages"
        / "annotation"
        / fixture.name
        / "definitive_exclusion_mask.npy.xz"
    )
    return [
        {
            "role": role,
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for role, path in paths.items()
    ]


def _search_intervals(summary: Mapping[str, object], video_name: str) -> tuple[tuple[int, int], ...]:
    raw_intervals = summary.get("search_intervals")
    if not isinstance(raw_intervals, list):
        raise TypeError(f"{video_name}: feature summary has no search intervals")

    intervals: list[tuple[int, int]] = []
    for raw_interval in raw_intervals:
        if (
            not isinstance(raw_interval, list)
            or len(raw_interval) != 2
            or type(raw_interval[0]) is not int
            or type(raw_interval[1]) is not int
        ):
            raise ValueError(f"{video_name}: feature summary has an invalid search range")
        start, end = raw_interval
        if start < 0 or end <= start:
            raise ValueError(f"{video_name}: feature summary has an invalid search range")
        intervals.append((start, end))
    return tuple(intervals)


def _validate_feature_rows(video: VideoSpec, rows: np.ndarray, summary: Mapping[str, object]) -> None:
    field_names = rows.dtype.names
    if field_names is None or any(name not in field_names for name in IDENTITY_FIELDS):
        raise ValueError(f"{video.fixture}: feature rows are missing identity fields")
    if not rows.size:
        raise ValueError(f"{video.fixture}: feature rows are empty")
    if summary.get("fixture") != video.fixture or summary.get("row_count") != len(rows):
        raise ValueError(f"{video.fixture}: feature summary does not match its rows")
    if not np.all(rows["fixture"] == video.fixture.encode("ascii")):
        raise ValueError(f"{video.fixture}: feature rows contain another video name")
    if not np.all(rows["fps"] == video.fps):
        raise ValueError(f"{video.fixture}: feature rows contain another frame rate")

    intervals = _search_intervals(summary, video.fixture)
    interval_ids = rows["interval_id"]
    if not np.array_equal(np.unique(interval_ids), np.arange(len(intervals))):
        raise ValueError(f"{video.fixture}: feature rows do not cover the saved search ranges")
    for interval_id, (start, end) in enumerate(intervals):
        actual_frames = rows["frame"][interval_ids == interval_id]
        expected_frames = np.arange(start, end, dtype=actual_frames.dtype)
        if not np.array_equal(actual_frames, expected_frames):
            raise ValueError(f"{video.fixture}: feature frames differ from a saved search range")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.partial")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def _write_feature_file(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.partial")
    _write_npy_xz(temporary_path, rows)
    os.replace(temporary_path, path)


def _video_record(
    video: VideoSpec,
    rows: np.ndarray,
    summary: dict[str, Any],
    input_files: list[dict[str, object]],
    feature_path: Path,
    output_root: Path,
    split_filename: str,
    split_sha256: str,
    source_commit: str,
    motion_mode: str,
) -> dict[str, object]:
    return {
        "schema": VIDEO_RECORD_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "source_commit": source_commit,
        "split_file": split_filename,
        "split_sha256": split_sha256,
        "motion_mode": motion_mode,
        "contact_labels_read": False,
        "video": {
            "name": video.fixture,
            "video_id": video.video_id,
            "fps": video.fps,
            "width": video.width,
            "height": video.height,
            "role": video.role.value,
        },
        "feature_file": feature_path.relative_to(output_root).as_posix(),
        "feature_sha256": _sha256(feature_path),
        "row_count": len(rows),
        "feature_summary": summary,
        "input_files": input_files,
    }


def _run_record(
    videos: Sequence[VideoSpec],
    records: list[dict[str, object]],
    split_filename: str,
    split_sha256: str,
    source_commit: str,
    motion_mode: str,
    status: str,
) -> dict[str, object]:
    return {
        "schema": RUN_RECORD_SCHEMA,
        "status": status,
        "feature_schema": FEATURE_SCHEMA,
        "source_commit": source_commit,
        "split_file": split_filename,
        "split_sha256": split_sha256,
        "motion_mode": motion_mode,
        "contact_labels_read": False,
        "video_names": [video.fixture for video in videos],
        "video_count": len(videos),
        "completed_video_names": [video.fixture for video in videos[: len(records)]],
        "completed_video_count": len(records),
        "row_count": sum(int(record["row_count"]) for record in records),
        "feature_families": _feature_family_names(),
        "identity_fields": list(IDENTITY_FIELDS),
        "videos": records,
    }


def freeze_features(
    split_path: Path,
    video_metadata_path: Path,
    match_path: Path,
    data_root: Path,
    output_dir: Path,
    source_commit: str,
    motion_mode: str = "raw_per_frame",
    requested_names: Sequence[str] = (),
    row_builder: RowBuilder = _fixture_rows,
) -> Path:
    """Save one checked feature file per requested video and return the run record."""
    if not source_commit.strip():
        raise ValueError("source commit must be non-empty")
    _motion_scale_factor(30.0, motion_mode)

    split_file = Path(split_path)
    split = load_development_split(split_file)
    verify_accepted_development_split(split)
    verify_against_shuttleset_tables(split, video_metadata_path, match_path)
    videos = _selected_videos(split, requested_names)
    split_sha256 = _sha256(split_file)
    output_root = Path(output_dir)
    records: list[dict[str, object]] = []
    expected_dtype: np.dtype[Any] | None = None
    run_record_path = output_root / RUN_RECORD_FILENAME
    _write_json(
        run_record_path,
        _run_record(
            videos,
            records,
            split_file.name,
            split_sha256,
            source_commit,
            motion_mode,
            "running",
        ),
    )

    for video in videos:
        fixture = _as_fixture(video)
        rows, summary = row_builder(Path(data_root), fixture, motion_mode)
        _validate_feature_rows(video, rows, summary)
        if expected_dtype is None:
            expected_dtype = rows.dtype
        elif rows.dtype != expected_dtype:
            raise ValueError(f"{video.fixture}: feature fields differ from earlier videos")

        feature_path = output_root / "videos" / video.fixture / FEATURE_FILENAME
        _write_feature_file(feature_path, rows)
        record = _video_record(
            video,
            rows,
            summary,
            _input_files(Path(data_root), fixture),
            feature_path,
            output_root,
            split_file.name,
            split_sha256,
            source_commit,
            motion_mode,
        )
        _write_json(feature_path.with_name(VIDEO_RECORD_FILENAME), record)
        records.append(record)
        _write_json(
            run_record_path,
            _run_record(
                videos,
                records,
                split_file.name,
                split_sha256,
                source_commit,
                motion_mode,
                "running",
            ),
        )
        print(f"saved {video.fixture}: {len(rows)} feature rows", flush=True)

    _write_json(
        run_record_path,
        _run_record(
            videos,
            records,
            split_file.name,
            split_sha256,
            source_commit,
            motion_mode,
            "complete",
        ),
    )
    return run_record_path


def _load_feature_rows(path: Path) -> np.ndarray:
    from dataset_builder.vision import load_npy_xz

    return load_npy_xz(path)


def _assert_feature_rows_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.dtype != expected.dtype:
        raise AssertionError(f"feature field types differ: {actual.dtype!r} != {expected.dtype!r}")
    if actual.shape != expected.shape:
        raise AssertionError(f"feature row counts differ: {actual.shape!r} != {expected.shape!r}")
    field_names = actual.dtype.names
    if field_names is None:
        np.testing.assert_array_equal(actual, expected, strict=True)
        return

    for field_name in field_names:
        actual_field = actual[field_name]
        expected_field = expected[field_name]
        if np.issubdtype(actual_field.dtype, np.floating):
            missing = np.isnan(actual_field)
            np.testing.assert_array_equal(missing, np.isnan(expected_field), strict=True)
            np.testing.assert_array_equal(
                actual_field[~missing],
                expected_field[~missing],
                strict=True,
            )
        else:
            np.testing.assert_array_equal(actual_field, expected_field, strict=True)


def check_pilot_features(
    run_record_path: Path,
    saved_pilot_record_path: Path,
) -> dict[str, object]:
    """Check that the three pilot videos have exactly the saved pilot feature rows."""
    run_record_file = Path(run_record_path)
    run_record = json.loads(run_record_file.read_text(encoding="utf-8"))
    if run_record.get("schema") != RUN_RECORD_SCHEMA:
        raise ValueError("new feature record version differs")
    if run_record.get("status") != "complete":
        raise ValueError("new feature run is not complete")
    if tuple(run_record.get("video_names", ())) != PILOT_VIDEO_NAMES:
        raise ValueError("pilot check requires sset_01, sset_15 and sset_21 in that order")

    video_records = run_record.get("videos")
    if not isinstance(video_records, list) or len(video_records) != len(PILOT_VIDEO_NAMES):
        raise ValueError("new feature record has the wrong number of pilot videos")
    new_chunks: list[np.ndarray] = []
    for record in video_records:
        if not isinstance(record, dict):
            raise TypeError("new feature record has an invalid video entry")
        feature_filename = record.get("feature_file")
        feature_sha256 = record.get("feature_sha256")
        if not isinstance(feature_filename, str) or not isinstance(feature_sha256, str):
            raise TypeError("new feature record has no checked feature file")
        feature_path = run_record_file.parent / feature_filename
        if _sha256(feature_path) != feature_sha256:
            raise ValueError("new feature file checksum differs")
        new_chunks.append(_load_feature_rows(feature_path))
    new_rows = np.concatenate(new_chunks)

    saved_record_file = Path(saved_pilot_record_path)
    saved_record = json.loads(saved_record_file.read_text(encoding="utf-8"))
    saved_feature_filename = saved_record.get("feature_file")
    saved_feature_sha256 = saved_record.get("feature_sha256")
    if not isinstance(saved_feature_filename, str) or not isinstance(saved_feature_sha256, str):
        raise TypeError("saved pilot record has no checked feature file")
    saved_feature_path = saved_record_file.parent / saved_feature_filename
    if _sha256(saved_feature_path) != saved_feature_sha256:
        raise ValueError("saved pilot feature file checksum differs")
    saved_rows = _load_feature_rows(saved_feature_path)
    _assert_feature_rows_equal(new_rows, saved_rows)

    return {
        "schema": PILOT_CHECK_SCHEMA,
        "exact_match": True,
        "video_names": list(PILOT_VIDEO_NAMES),
        "row_count": len(new_rows),
        "new_run_record": run_record_file.name,
        "new_run_record_sha256": _sha256(run_record_file),
        "saved_pilot_record": saved_record_file.name,
        "saved_pilot_feature": saved_feature_filename,
        "saved_pilot_feature_sha256": saved_feature_sha256,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path)
    parser.add_argument("--video-metadata", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--motion-mode", choices=MOTION_MODES, default="raw_per_frame")
    parser.add_argument("--video", action="append", default=[], dest="video_names")
    parser.add_argument("--saved-pilot-record", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    run_record_path = freeze_features(
        arguments.split,
        arguments.video_metadata,
        arguments.matches,
        arguments.data_root,
        arguments.output_dir,
        arguments.source_commit,
        arguments.motion_mode,
        arguments.video_names,
    )
    if arguments.saved_pilot_record is not None:
        check = check_pilot_features(run_record_path, arguments.saved_pilot_record)
        _write_json(run_record_path.parent / PILOT_CHECK_FILENAME, check)
        print(f"pilot feature rows match exactly: {check['row_count']}", flush=True)
    print(run_record_path)


if __name__ == "__main__":
    main()
