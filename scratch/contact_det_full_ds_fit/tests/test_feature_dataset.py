from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scratch.contact_det_full_ds_fit.scripts.freeze_contact_features as freezer
from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
    load_verified_feature_dataset,
)
from scratch.contact_det_full_ds_fit.scripts.freeze_contact_features import (
    FEATURE_FILENAME,
    RUN_RECORD_FILENAME,
    freeze_features,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = EXPERIMENT_ROOT / "records/shuttleset_development_split.json"
ANNOTATION_ROOT = REPO_ROOT / "training" / "data" / "shuttleset" / "annotations"
VIDEO_METADATA_PATH = ANNOTATION_ROOT / "video_metadata.csv"
MATCH_PATH = ANNOTATION_ROOT / "set" / "match.csv"


def _rows(video_name: str, fps: float, frame: int) -> np.ndarray:
    feature_families = freezer._feature_family_names()
    fields: list[tuple[str, str]] = [
        ("fixture", "S16"),
        ("interval_id", "<i4"),
        ("frame", "<i4"),
        ("fps", "<f4"),
    ]
    feature_names: list[str] = []
    for names in feature_families.values():
        for name in names:
            if name not in feature_names:
                feature_names.append(name)
    fields.extend((name, "<f4") for name in feature_names)
    rows = np.zeros(1, dtype=np.dtype(fields))
    rows["fixture"] = video_name.encode("ascii")
    rows["interval_id"] = 0
    rows["frame"] = frame
    rows["fps"] = fps
    return rows


def _summary(video_name: str, frame: int) -> dict[str, Any]:
    return {
        "fixture": video_name,
        "row_count": 1,
        "search_intervals": [[frame, frame + 1]],
    }


@pytest.fixture
def complete_feature_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(
        freezer,
        "_input_files",
        lambda _root, _fixture: [
            {
                "role": "test_input",
                "filename": "input.bin",
                "size_bytes": 1,
                "sha256": "0" * 64,
            }
        ],
    )

    def build_rows(_root: Path, fixture: Any, _motion_mode: str) -> tuple[np.ndarray, dict[str, Any]]:
        frame = fixture.video_id * 10
        return _rows(fixture.name, fixture.fps, frame), _summary(fixture.name, frame)

    output_dir = tmp_path / "features"
    record_path = freeze_features(
        SPLIT_PATH,
        VIDEO_METADATA_PATH,
        MATCH_PATH,
        tmp_path / "unused-input",
        output_dir,
        "deadbee",
        row_builder=build_rows,
    )
    return record_path, output_dir


def test_complete_feature_run_loads_in_split_order(
    complete_feature_run: tuple[Path, Path],
) -> None:
    record_path, _output_dir = complete_feature_run

    verified = load_verified_feature_dataset(record_path, SPLIT_PATH, "raw_per_frame")

    assert len(verified.rows) == 40
    assert list(verified.video_ranges) == [video.fixture for video in verified.split.videos]
    assert verified.video_ranges["sset_01"] == (0, 1)
    assert verified.video_ranges["sset_44"] == (39, 40)
    assert "shuttle_speed_t+0" in verified.model_input_fields
    assert "shuttle_visible_t+0" in verified.model_input_fields


def test_loader_rejects_a_running_record(complete_feature_run: tuple[Path, Path]) -> None:
    record_path, _output_dir = complete_feature_run
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["status"] = "running"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="not complete"):
        load_verified_feature_dataset(record_path, SPLIT_PATH, "raw_per_frame")


def test_loader_rejects_a_changed_feature_file_hash(
    complete_feature_run: tuple[Path, Path],
) -> None:
    record_path, output_dir = complete_feature_run
    feature_path = output_dir / "videos" / "sset_01" / FEATURE_FILENAME
    feature_path.write_bytes(feature_path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="feature file hash differs"):
        load_verified_feature_dataset(record_path, SPLIT_PATH, "raw_per_frame")


def test_loader_rejects_a_feature_path_outside_the_run(
    complete_feature_run: tuple[Path, Path],
) -> None:
    record_path, _output_dir = complete_feature_run
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["videos"][0]["feature_file"] = "../contact_features.npy.xz"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="stay inside"):
        load_verified_feature_dataset(record_path, SPLIT_PATH, "raw_per_frame")


def test_loader_rejects_a_changed_motion_mode(
    complete_feature_run: tuple[Path, Path],
) -> None:
    record_path, _output_dir = complete_feature_run

    with pytest.raises(ValueError, match="motion values differ"):
        load_verified_feature_dataset(record_path, SPLIT_PATH, "base30_per_frame")


def test_fixture_writes_the_expected_full_record_name(
    complete_feature_run: tuple[Path, Path],
) -> None:
    record_path, _output_dir = complete_feature_run

    assert record_path.name == RUN_RECORD_FILENAME
