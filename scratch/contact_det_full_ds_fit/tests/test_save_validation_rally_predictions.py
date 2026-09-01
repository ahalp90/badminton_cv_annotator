from __future__ import annotations

import builtins
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scratch.contact_det_full_ds_fit.scripts.save_validation_rally_predictions as saver
from scratch.contact_det.scripts.freeze_contact_evidence import (
    AnnotationData,
    _stage_paths,
)
from scratch.contact_det.scripts.freeze_tree_contact_features import _player_signals
from scratch.contact_det_full_ds_fit.scripts.baseline_config import FIXED_RUN_IDS
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import SCORE_DTYPE


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_inputs(frame_count: int = 8) -> tuple[np.ndarray, Any, Any, list[Any], Any, AnnotationData]:
    track = np.zeros((frame_count, 3), dtype=np.float32)
    track[:, 0] = np.linspace(0.1, 0.8, frame_count)
    track[:, 1] = np.linspace(0.8, 0.1, frame_count)
    track[:, 2] = 1
    pose_kps = np.zeros((frame_count, 2, 17, 2), dtype=np.float32)
    pose_kps[:, 0, :, :] = (200.0, 200.0)
    pose_kps[:, 1, :, :] = (1400.0, 800.0)
    pose = SimpleNamespace(
        kps=pose_kps,
        bboxes=np.tile(
            np.asarray([[100.0, 100.0, 300.0, 300.0], [1300.0, 700.0, 1500.0, 900.0]]),
            (frame_count, 1, 1),
        ),
    )
    sticky = SimpleNamespace(
        picks=np.tile(np.asarray([0, 1], dtype=np.int32), (frame_count, 1)),
        distances_per_slot=np.tile(np.asarray([0.2, 0.5]), (frame_count, 1)),
        ankle_pos=np.tile(np.asarray([[0.2, 0.2], [0.7, 0.8]]), (frame_count, 1, 1)),
        bbox_height=np.tile(np.asarray([200.0, 200.0]), (frame_count, 1)),
        standing_count=np.full(frame_count, 2, dtype=np.int32),
    )
    court = SimpleNamespace(evidence=SimpleNamespace(inputs=SimpleNamespace(net_band=(480.0, 600.0))))
    annotation = AnnotationData(
        spans=((0, 4), (4, 7)),
        contacts=(),
        filtered_contacts=(),
        filtered_by_rally={},
        striker_halves=(None, None),
        fitted_first_all=(None, None),
    )
    return track, pose, court, [], sticky, annotation


def _feature_rows(
    fixture: str,
    frames: np.ndarray,
    inputs: tuple[np.ndarray, Any, Any, list[Any], Any, AnnotationData],
) -> np.ndarray:
    track, pose, _court, _intervals, sticky, _annotation = inputs
    dtype = np.dtype(
        [("fixture", "S7"), ("frame", "<i4")]
        + [(field, "<f4") for field in saver.CENTRE_FEATURE_FIELDS]
    )
    rows = np.zeros(len(frames), dtype=dtype)
    rows["fixture"] = fixture.encode("ascii")
    rows["frame"] = frames
    visible = track[:, 2] == 1
    expected = {
        "shuttle_x": np.where(visible, track[:, 0], np.nan),
        "shuttle_y": np.where(visible, track[:, 1], np.nan),
        "shuttle_visible": visible.astype(np.float32),
        "standing_count": np.asarray(sticky.standing_count, dtype=np.float32),
    }
    expected.update(_player_signals(track, pose.kps, sticky, (1920.0, 1080.0)))
    for feature_field, expected_name in saver.CENTRE_FEATURE_FIELDS.items():
        rows[feature_field] = expected[expected_name][frames]
    return rows


def _stage_records(tmp_path: Path, fixture: Any) -> list[dict[str, object]]:
    records = []
    for index, (role, path) in enumerate(_stage_paths(tmp_path, fixture).items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{role}-{index}".encode("ascii"))
        records.append(
            {
                "role": role,
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    records.append(
        {
            "role": "definitive_exclusion_mask",
            "filename": "definitive_exclusion_mask.npy.xz",
            "size_bytes": 1,
            "sha256": "0" * 64,
        }
    )
    return records


def _checked_run(run_id: str, fixture: str, frames: tuple[int, ...]) -> Any:
    scores = np.zeros(len(frames), dtype=SCORE_DTYPE)
    scores["fixture"] = fixture.encode("ascii")
    scores["frame"] = frames
    scores["fps"] = 30.0
    scores["contact_score"] = np.linspace(0.8, 0.9, len(frames))
    scores["kept"] = True
    return SimpleNamespace(
        run=SimpleNamespace(run_id=run_id),
        score_rows=scores,
        kept=scores["kept"].copy(),
        predictions={fixture: np.asarray(frames, dtype=np.int32)},
    )


def _verified(tmp_path: Path) -> tuple[Any, Any, tuple[Any, ...]]:
    video = SimpleNamespace(
        fixture="sset_18",
        video_id=18,
        fps=30.0,
        width=1920,
        height=1080,
    )
    fixture = saver._fixture(video)
    inputs = _fake_inputs()
    frames = np.asarray([3, 4, 7], dtype=np.int32)
    rows = _feature_rows(video.fixture, frames, inputs)
    record_path = tmp_path / "contact_features_record.json"
    record_path.write_text("{}", encoding="utf-8")
    feature_record = {
        "video": {"name": video.fixture},
        "feature_summary": {"frame_count": 8, "rally_span_count": 2},
        "input_files": _stage_records(tmp_path, fixture),
    }
    menu_path = tmp_path / "baseline_menu_result.json"
    menu_path.write_text("{}", encoding="utf-8")
    runs = (
        _checked_run(FIXED_RUN_IDS[0], video.fixture, (3, 4, 7)),
        _checked_run(FIXED_RUN_IDS[1], video.fixture, (4,)),
    )
    verified = SimpleNamespace(
        menu_path=menu_path,
        menu={
            "split_file": "split.json",
            "split_sha256": "1" * 64,
            "contact_label_file": "shots_master.csv",
            "contact_label_sha256": "2" * 64,
        },
        split=SimpleNamespace(dataset="ShuttleSet", validation_videos=(video,)),
        raw_features=SimpleNamespace(
            record_path=record_path,
            record={"videos": [feature_record]},
            rows=rows,
            video_ranges={video.fixture: (0, len(rows))},
        ),
        runs=runs,
    )
    return verified, inputs, runs


def test_build_checks_inputs_and_replays_each_distinct_frame_once(tmp_path: Path) -> None:
    verified, inputs, _runs = _verified(tmp_path)
    calls: list[int] = []

    def load_inputs(_root: Path, _fixture: Any) -> Any:
        return inputs

    def attribute(frame: int, _track: Any, _sticky: Any, _bboxes: Any, _net: Any) -> str:
        calls.append(frame)
        return "Top" if frame < 4 else "Bot"

    result = saver.build_validation_rally_predictions(
        verified,
        tmp_path,
        "deadbee",
        input_loader=load_inputs,
        side_attributor=attribute,
    )

    assert result["labels_read"] is False
    assert calls == [3, 4, 7]
    first_contacts = result["runs"][0]["videos"][0]["contacts"]
    assert [contact["span_id"] for contact in first_contacts] == [0, 1, None]
    assert [contact["predicted_side"] for contact in first_contacts] == ["Top", "Bot", "Bot"]
    assert result["videos"][0]["spans"] == [
        {"span_id": 0, "start_frame": 0, "end_frame": 4},
        {"span_id": 1, "start_frame": 4, "end_frame": 7},
    ]


def test_default_replay_calls_the_shipped_player_side_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    from annotator import point_winner

    verified, inputs, _runs = _verified(tmp_path)
    calls: list[int] = []

    def attribute(frame: int, _track: Any, _sticky: Any, _bboxes: Any, _net: Any) -> str:
        calls.append(frame)
        return "Top"

    monkeypatch.setattr(point_winner, "attribute_half", attribute)
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("contact labels must not be read")
        ),
    )
    saver.build_validation_rally_predictions(
        verified,
        tmp_path,
        "deadbee",
        input_loader=lambda *_args: inputs,
    )

    assert calls == [3, 4, 7]


def test_changed_stage_file_is_rejected_before_inputs_are_loaded(tmp_path: Path) -> None:
    verified, inputs, _runs = _verified(tmp_path)
    fixture = saver._fixture(verified.split.validation_videos[0])
    _stage_paths(tmp_path, fixture)["annotation"].write_bytes(b"changed")
    loaded = False

    def load_inputs(_root: Path, _fixture: Any) -> Any:
        nonlocal loaded
        loaded = True
        return inputs

    with pytest.raises(ValueError, match="annotation file size differs"):
        saver.build_validation_rally_predictions(
            verified,
            tmp_path,
            "deadbee",
            input_loader=load_inputs,
            side_attributor=lambda *_args: "Top",
        )
    assert not loaded


def test_changed_centre_feature_is_rejected(tmp_path: Path) -> None:
    verified, inputs, _runs = _verified(tmp_path)
    verified.raw_features.rows["shuttle_x"][0] += 1.0
    with pytest.raises(ValueError, match="shuttle_x differs from the replay input"):
        saver.build_validation_rally_predictions(
            verified,
            tmp_path,
            "deadbee",
            input_loader=lambda *_args: inputs,
            side_attributor=lambda *_args: "Top",
        )


def test_wrong_rally_span_count_is_rejected(tmp_path: Path) -> None:
    verified, inputs, _runs = _verified(tmp_path)
    verified.raw_features.record["videos"][0]["feature_summary"]["rally_span_count"] = 3
    with pytest.raises(ValueError, match="rally span count differs"):
        saver.build_validation_rally_predictions(
            verified,
            tmp_path,
            "deadbee",
            input_loader=lambda *_args: inputs,
            side_attributor=lambda *_args: "Top",
        )


def test_save_does_not_import_pandas_and_is_byte_repeatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, inputs, _runs = _verified(tmp_path)
    files = {
        name: tmp_path / name
        for name in ("config.json", "split.json", "raw.json", "common.json", "shots_master.csv")
    }
    for name, path in files.items():
        path.write_bytes(name.encode("ascii"))
    old_import = builtins.__import__

    def fail_pandas(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pandas" or name.startswith("pandas."):
            raise AssertionError("pandas must not be imported")
        return old_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_pandas)
    output = tmp_path / "predictions.json.gz"

    def save_once() -> bytes:
        saver.save_validation_rally_predictions(
            verified.menu_path,
            files["config.json"],
            files["split.json"],
            files["raw.json"],
            files["common.json"],
            files["shots_master.csv"],
            tmp_path,
            output,
            "deadbee",
            menu_loader=lambda *_args: verified,
            input_loader=lambda *_args: inputs,
            side_attributor=lambda *_args: "Top",
        )
        return output.read_bytes()

    first = save_once()
    second = save_once()
    assert first == second
    with gzip.open(output, "rt", encoding="utf-8") as source:
        result = json.load(source)
    assert result["status"] == "complete"
    assert result["labels_read"] is False
    assert result["contact_label_file"] == "shots_master.csv"


def test_failed_check_replaces_old_complete_output_with_running(tmp_path: Path) -> None:
    output = tmp_path / "predictions.json.gz"
    saver._write_json(output, {"schema": saver.PREDICTION_SCHEMA, "status": "complete"})
    input_file = tmp_path / "input"
    input_file.write_bytes(b"input")

    with pytest.raises(RuntimeError, match="stopped"):
        saver.save_validation_rally_predictions(
            input_file,
            input_file,
            input_file,
            input_file,
            input_file,
            input_file,
            tmp_path,
            output,
            "deadbee",
            menu_loader=lambda *_args: (_ for _ in ()).throw(RuntimeError("stopped")),
        )

    with gzip.open(output, "rt", encoding="utf-8") as source:
        running = json.load(source)
    assert running == {
        "labels_read": False,
        "schema": saver.PREDICTION_SCHEMA,
        "source_commit": "deadbee",
        "status": "running",
    }


def test_bad_source_commit_also_replaces_old_complete_output(tmp_path: Path) -> None:
    output = tmp_path / "predictions.json.gz"
    saver._write_json(output, {"schema": saver.PREDICTION_SCHEMA, "status": "complete"})
    input_file = tmp_path / "input"
    input_file.write_bytes(b"input")

    with pytest.raises(ValueError, match="source commit"):
        saver.save_validation_rally_predictions(
            input_file,
            input_file,
            input_file,
            input_file,
            input_file,
            input_file,
            tmp_path,
            output,
            "not-a-commit",
        )

    with gzip.open(output, "rt", encoding="utf-8") as source:
        assert json.load(source)["status"] == "running"
