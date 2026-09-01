from __future__ import annotations

import csv
import json
import lzma
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    load_development_split,
)
from scratch.contact_det_full_ds_fit.scripts.freeze_contact_features import (
    FEATURE_FILENAME,
    PILOT_VIDEO_NAMES,
    RUN_RECORD_FILENAME,
    RUN_RECORD_SCHEMA,
    VIDEO_RECORD_FILENAME,
    _as_fixture,
    _input_files,
    _selected_videos,
    _sha256,
    _stage_paths,
    _validate_feature_rows,
    _write_npy_xz,
    check_pilot_features,
    freeze_features,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = EXPERIMENT_ROOT / "records/shuttleset_development_split.json"
ANNOTATION_ROOT = REPO_ROOT / "training" / "data" / "shuttleset" / "annotations"
VIDEO_METADATA_PATH = ANNOTATION_ROOT / "video_metadata.csv"
MATCH_PATH = ANNOTATION_ROOT / "set" / "match.csv"


def _rows(video_name: str, fps: float, start: int = 10, end: int = 13) -> np.ndarray:
    dtype = np.dtype(
        [
            ("fixture", "S16"),
            ("interval_id", "<i4"),
            ("frame", "<i4"),
            ("fps", "<f4"),
            ("signal", "<f4"),
        ]
    )
    rows = np.zeros(end - start, dtype=dtype)
    rows["fixture"] = video_name.encode("ascii")
    rows["interval_id"] = 0
    rows["frame"] = np.arange(start, end)
    rows["fps"] = fps
    rows["signal"] = [1.0, np.nan, 3.0]
    return rows


def _summary(video_name: str, row_count: int, start: int = 10, end: int = 13) -> dict[str, Any]:
    return {
        "fixture": video_name,
        "row_count": row_count,
        "search_intervals": [[start, end]],
    }


def _write_required_inputs(data_root: Path, video_name: str) -> None:
    split = load_development_split(SPLIT_PATH)
    video = split.by_fixture[video_name]
    fixture = _as_fixture(video)
    paths = _stage_paths(data_root, fixture)
    paths["definitive_exclusion_mask"] = (
        data_root
        / "stages"
        / "annotation"
        / video_name
        / "definitive_exclusion_mask.npy.xz"
    )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("utf-8"))


def test_video_selection_keeps_split_order_and_rejects_duplicates() -> None:
    split = load_development_split(SPLIT_PATH)

    selected = _selected_videos(split, ["sset_21", "sset_01"])

    assert [video.fixture for video in selected] == ["sset_01", "sset_21"]
    with pytest.raises(ValueError, match="must be unique"):
        _selected_videos(split, ["sset_01", "sset_01"])
    with pytest.raises(ValueError, match="absent from the split"):
        _selected_videos(split, ["sset_99"])


def test_row_check_rejects_frames_outside_the_saved_range() -> None:
    split = load_development_split(SPLIT_PATH)
    video = split.by_fixture["sset_01"]
    rows = _rows(video.fixture, video.fps)
    rows["frame"] = [10, 12, 13]

    with pytest.raises(ValueError, match="frames differ"):
        _validate_feature_rows(video, rows, _summary(video.fixture, len(rows)))


def test_freeze_writes_checked_path_free_records(tmp_path: Path) -> None:
    data_root = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_required_inputs(data_root, "sset_01")

    def build_rows(_root: Path, fixture: Any, _motion_mode: str) -> tuple[np.ndarray, dict[str, Any]]:
        rows = _rows(fixture.name, fixture.fps)
        return rows, _summary(fixture.name, len(rows))

    record_path = freeze_features(
        SPLIT_PATH,
        VIDEO_METADATA_PATH,
        MATCH_PATH,
        data_root,
        output_dir,
        "abc123",
        requested_names=["sset_01"],
        row_builder=build_rows,
    )

    record = json.loads(record_path.read_text(encoding="utf-8"))
    video_record = record["videos"][0]
    assert record_path.name == RUN_RECORD_FILENAME
    assert record["schema"] == RUN_RECORD_SCHEMA
    assert record["status"] == "complete"
    assert record["video_names"] == ["sset_01"]
    assert record["completed_video_names"] == ["sset_01"]
    assert record["contact_labels_read"] is False
    assert video_record["feature_file"] == f"videos/sset_01/{FEATURE_FILENAME}"
    assert (output_dir / video_record["feature_file"]).is_file()
    assert (output_dir / "videos" / "sset_01" / VIDEO_RECORD_FILENAME).is_file()
    assert all(set(file_record) == {"role", "filename", "size_bytes", "sha256"} for file_record in video_record["input_files"])
    assert str(tmp_path) not in record_path.read_text(encoding="utf-8")


def test_pilot_check_accepts_exact_rows_and_rejects_a_change(tmp_path: Path) -> None:
    output_dir = tmp_path / "new"
    video_records: list[dict[str, object]] = []
    combined_rows: list[np.ndarray] = []
    for index, video_name in enumerate(PILOT_VIDEO_NAMES):
        rows = _rows(video_name, 25.0 if video_name != "sset_21" else 30.0, 10 + index * 3, 13 + index * 3)
        combined_rows.append(rows)
        feature_path = output_dir / "videos" / video_name / FEATURE_FILENAME
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        _write_npy_xz(feature_path, rows)
        video_records.append(
            {
                "feature_file": feature_path.relative_to(output_dir).as_posix(),
                "feature_sha256": _sha256(feature_path),
            }
        )

    run_record_path = output_dir / RUN_RECORD_FILENAME
    run_record_path.write_text(
        json.dumps(
            {
                "schema": RUN_RECORD_SCHEMA,
                "status": "complete",
                "video_names": list(PILOT_VIDEO_NAMES),
                "videos": video_records,
            }
        ),
        encoding="utf-8",
    )
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    saved_feature_path = saved_dir / "saved.npy.xz"
    _write_npy_xz(saved_feature_path, np.concatenate(combined_rows))
    saved_record_path = saved_dir / "saved.json"
    saved_record_path.write_text(
        json.dumps(
            {
                "feature_file": saved_feature_path.name,
                "feature_sha256": _sha256(saved_feature_path),
            }
        ),
        encoding="utf-8",
    )

    check = check_pilot_features(run_record_path, saved_record_path)

    assert check["exact_match"] is True
    changed_rows = combined_rows[0].copy()
    changed_rows["signal"][0] = 99.0
    changed_feature_path = output_dir / str(video_records[0]["feature_file"])
    _write_npy_xz(changed_feature_path, changed_rows)
    video_records[0]["feature_sha256"] = _sha256(changed_feature_path)
    run_record_path.write_text(
        json.dumps(
            {
                "schema": RUN_RECORD_SCHEMA,
                "status": "complete",
                "video_names": list(PILOT_VIDEO_NAMES),
                "videos": video_records,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        check_pilot_features(run_record_path, saved_record_path)


def test_pilot_check_rejects_a_changed_new_feature_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "new"
    video_records: list[dict[str, object]] = []
    combined_rows: list[np.ndarray] = []
    for video_name in PILOT_VIDEO_NAMES:
        rows = _rows(video_name, 25.0)
        combined_rows.append(rows)
        feature_path = output_dir / "videos" / video_name / FEATURE_FILENAME
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        _write_npy_xz(feature_path, rows)
        video_records.append(
            {
                "feature_file": feature_path.relative_to(output_dir).as_posix(),
                "feature_sha256": _sha256(feature_path),
            }
        )
    run_record_path = output_dir / RUN_RECORD_FILENAME
    run_record_path.write_text(
        json.dumps(
            {
                "schema": RUN_RECORD_SCHEMA,
                "status": "complete",
                "video_names": list(PILOT_VIDEO_NAMES),
                "videos": video_records,
            }
        ),
        encoding="utf-8",
    )
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    saved_feature_path = saved_dir / "saved.npy.xz"
    _write_npy_xz(saved_feature_path, np.concatenate(combined_rows))
    saved_record_path = saved_dir / "saved.json"
    saved_record_path.write_text(
        json.dumps(
            {
                "feature_file": saved_feature_path.name,
                "feature_sha256": _sha256(saved_feature_path),
            }
        ),
        encoding="utf-8",
    )

    first_feature_path = output_dir / str(video_records[0]["feature_file"])
    with lzma.open(first_feature_path, "wb", format=lzma.FORMAT_XZ, preset=0) as destination:
        np.save(destination, combined_rows[0], allow_pickle=False)

    with pytest.raises(ValueError, match="new feature file checksum differs"):
        check_pilot_features(run_record_path, saved_record_path)


def test_failed_rerun_replaces_an_old_complete_record(tmp_path: Path) -> None:
    data_root = tmp_path / "input"
    output_dir = tmp_path / "output"
    for video_name in ("sset_01", "sset_15"):
        _write_required_inputs(data_root, video_name)
    old_record_path = output_dir / RUN_RECORD_FILENAME
    old_record_path.parent.mkdir(parents=True)
    old_record_path.write_text('{"status": "complete", "old": true}', encoding="utf-8")

    def build_rows(_root: Path, fixture: Any, _motion_mode: str) -> tuple[np.ndarray, dict[str, Any]]:
        if fixture.name == "sset_15":
            raise RuntimeError("stopped test run")
        rows = _rows(fixture.name, fixture.fps)
        return rows, _summary(fixture.name, len(rows))

    with pytest.raises(RuntimeError, match="stopped test run"):
        freeze_features(
            SPLIT_PATH,
            VIDEO_METADATA_PATH,
            MATCH_PATH,
            data_root,
            output_dir,
            "abc123",
            requested_names=["sset_01", "sset_15"],
            row_builder=build_rows,
        )

    record = json.loads(old_record_path.read_text(encoding="utf-8"))
    assert record["status"] == "running"
    assert record["video_names"] == ["sset_01", "sset_15"]
    assert record["completed_video_names"] == ["sset_01"]
    assert record["completed_video_count"] == 1
    assert "old" not in record


def test_freeze_rejects_changed_saved_metadata_before_building_rows(tmp_path: Path) -> None:
    with VIDEO_METADATA_PATH.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    rows[0]["fps"] = "30"
    changed_metadata_path = tmp_path / "video_metadata.csv"
    with changed_metadata_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def must_not_run(_root: Path, _fixture: Any, _motion_mode: str) -> tuple[np.ndarray, dict[str, Any]]:
        raise AssertionError("row building started before metadata checks finished")

    with pytest.raises(ValueError, match="split details differ"):
        freeze_features(
            SPLIT_PATH,
            changed_metadata_path,
            MATCH_PATH,
            tmp_path / "input",
            tmp_path / "output",
            "abc123",
            requested_names=["sset_01"],
            row_builder=must_not_run,
        )


def test_input_file_records_do_not_save_the_data_root(tmp_path: Path) -> None:
    split = load_development_split(SPLIT_PATH)
    fixture = _as_fixture(split.by_fixture["sset_01"])
    _write_required_inputs(tmp_path, fixture.name)

    records = _input_files(tmp_path, fixture)

    assert all(str(tmp_path) not in json.dumps(record) for record in records)
