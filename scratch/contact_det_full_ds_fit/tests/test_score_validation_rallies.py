from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import scratch.contact_det_full_ds_fit.scripts.score_validation_rallies as scorer
from scratch.contact_det_full_ds_fit.scripts.baseline_config import FIXED_RUN_IDS
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    load_development_split,
)
from scratch.contact_det_full_ds_fit.scripts.save_validation_rally_predictions import (
    CENTRE_FEATURE_FIELDS,
    PREDICTION_SCHEMA,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = EXPERIMENT_ROOT / "records/shuttleset_development_split.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_prediction(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0
    ) as zipped:
        zipped.write(encoded)


def _prediction_files(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    menu_path = tmp_path / "baseline_menu_result.json"
    raw_path = tmp_path / "contact_features_record.json"
    labels_path = tmp_path / "shots_master.csv"
    menu_path.write_bytes(b"menu")
    labels_path.write_bytes(b"contact labels")
    split = load_development_split(SPLIT_PATH)
    videos = []
    for video in split.validation_videos:
        input_files = [
            {
                "role": role,
                "filename": f"{role}.bin",
                "size_bytes": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index, role in enumerate(scorer.EXPECTED_INPUT_ROLES)
        ]
        videos.append(
            {
                "fixture": video.fixture,
                "video_id": video.video_id,
                "fps": video.fps,
                "frame_count": 30,
                "spans": [{"span_id": 0, "start_frame": 0, "end_frame": 20}],
                "replayed_contact_count": 1,
                "input_files": input_files,
            }
        )
    raw_path.write_text(
        json.dumps(
            {
                "videos": [
                    {
                        "video": {"name": video["fixture"]},
                        "input_files": video["input_files"],
                        "feature_summary": {
                            "frame_count": video["frame_count"],
                            "rally_span_count": len(video["spans"]),
                        },
                    }
                    for video in videos
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    runs = [
        {
            "run_id": run_id,
            "videos": [
                {
                    "fixture": video.fixture,
                    "contacts": [
                        {
                            "frame": 10,
                            "timing_score": 0.8,
                            "predicted_side": "Top",
                            "span_id": 0,
                        }
                    ],
                }
                for video in split.validation_videos
            ],
        }
        for run_id in FIXED_RUN_IDS
    ]
    payload = {
        "schema": PREDICTION_SCHEMA,
        "status": "complete",
        "source_commit": "deadbee",
        "labels_read": False,
        "menu_result_file": menu_path.name,
        "menu_result_sha256": _sha256(menu_path),
        "split_file": SPLIT_PATH.name,
        "split_sha256": _sha256(SPLIT_PATH),
        "raw_feature_record_file": raw_path.name,
        "raw_feature_record_sha256": _sha256(raw_path),
        "contact_label_file": labels_path.name,
        "contact_label_sha256": _sha256(labels_path),
        "validation_videos": [video.fixture for video in split.validation_videos],
        "centre_feature_fields_checked": list(CENTRE_FEATURE_FIELDS),
        "videos": videos,
        "runs": runs,
    }
    prediction_path = tmp_path / "predictions.json.gz"
    _write_prediction(prediction_path, payload)
    return prediction_path, menu_path, raw_path, labels_path, payload


def _load(tmp_path: Path) -> scorer.VerifiedRallyPredictions:
    prediction_path, menu_path, raw_path, labels_path, _payload = _prediction_files(tmp_path)
    return scorer.load_validation_rally_predictions(
        prediction_path,
        menu_path,
        SPLIT_PATH,
        raw_path,
        labels_path,
    )


def _label_table() -> pd.DataFrame:
    split = load_development_split(SPLIT_PATH)
    rows = [
        {
            "vid": video.video_id,
            "set_id": "set1",
            "rally": 1,
            "frame_num": 10,
            "player_side": "Top",
        }
        for video in split.validation_videos
    ]
    rows.append(
        {
            "vid": split.validation_videos[0].video_id,
            "set_id": "set2",
            "rally": 1,
            "frame_num": 15,
            "player_side": "Bot",
        }
    )
    return pd.DataFrame(rows)


def test_valid_prediction_file_loads_fixed_runs_and_half_open_spans(tmp_path: Path) -> None:
    loaded = _load(tmp_path)

    assert [run.run_id for run in loaded.runs] == list(FIXED_RUN_IDS)
    assert loaded.spans_by_fixture["sset_18"] == (
        {"span_id": 0, "start_frame": 0, "end_frame": 20},
    )


def test_contact_at_span_end_must_be_unassigned(tmp_path: Path) -> None:
    prediction_path, menu_path, raw_path, labels_path, payload = _prediction_files(tmp_path)
    contact = payload["runs"][0]["videos"][0]["contacts"][0]
    contact["frame"] = 20
    _write_prediction(prediction_path, payload)

    with pytest.raises(ValueError, match="span ID differs"):
        scorer.load_validation_rally_predictions(
            prediction_path,
            menu_path,
            SPLIT_PATH,
            raw_path,
            labels_path,
        )


def test_player_side_answer_cannot_change_between_runs(tmp_path: Path) -> None:
    prediction_path, menu_path, raw_path, labels_path, payload = _prediction_files(tmp_path)
    payload["runs"][1]["videos"][0]["contacts"][0]["predicted_side"] = "Bot"
    _write_prediction(prediction_path, payload)

    with pytest.raises(ValueError, match="answers differ between runs"):
        scorer.load_validation_rally_predictions(
            prediction_path,
            menu_path,
            SPLIT_PATH,
            raw_path,
            labels_path,
        )


def test_span_and_contact_frames_must_stay_inside_the_video(tmp_path: Path) -> None:
    prediction_path, menu_path, raw_path, labels_path, payload = _prediction_files(tmp_path)
    payload["videos"][0]["spans"][0]["end_frame"] = 31
    _write_prediction(prediction_path, payload)

    with pytest.raises(ValueError, match="span exceeds the video frame count"):
        scorer.load_validation_rally_predictions(
            prediction_path,
            menu_path,
            SPLIT_PATH,
            raw_path,
            labels_path,
        )

    payload["videos"][0]["spans"][0]["end_frame"] = 20
    payload["runs"][0]["videos"][0]["contacts"][0]["frame"] = 30
    payload["runs"][0]["videos"][0]["contacts"][0]["span_id"] = None
    _write_prediction(prediction_path, payload)
    with pytest.raises(ValueError, match="contact frames differ"):
        scorer.load_validation_rally_predictions(
            prediction_path,
            menu_path,
            SPLIT_PATH,
            raw_path,
            labels_path,
        )


def test_saved_file_hash_change_is_rejected(tmp_path: Path) -> None:
    prediction_path, menu_path, raw_path, labels_path, _payload = _prediction_files(tmp_path)
    raw_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="raw feature record file hash differs"):
        scorer.load_validation_rally_predictions(
            prediction_path,
            menu_path,
            SPLIT_PATH,
            raw_path,
            labels_path,
        )


def test_timing_then_player_side_loads_separate_columns_and_groups_by_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _load(tmp_path)
    table = _label_table()
    monkeypatch.setattr(scorer, "EXPECTED_RALLY_COUNT", 9)
    monkeypatch.setattr(scorer, "EXPECTED_CONTACT_COUNT", 9)
    calls: list[tuple[str, ...]] = []

    def read_table(_path: Path, *, usecols: list[str]) -> pd.DataFrame:
        calls.append(tuple(usecols))
        return table.loc[:, usecols].copy()

    timing = scorer.load_timing_labels(
        tmp_path / "shots_master.csv", loaded.split, read_table
    )
    sides = scorer.load_player_side_labels(
        tmp_path / "shots_master.csv",
        loaded.split,
        timing.identities,
        read_table,
    )

    assert calls == [scorer.TIMING_COLUMNS, scorer.SIDE_COLUMNS]
    assert len(timing.rallies["sset_18"]) == 2
    assert sides[("sset_18", 10)] == "Top"
    assert sides[("sset_18", 15)] == "Bot"


def test_player_side_rows_must_match_timing_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _load(tmp_path)
    table = _label_table()
    monkeypatch.setattr(scorer, "EXPECTED_RALLY_COUNT", 9)
    monkeypatch.setattr(scorer, "EXPECTED_CONTACT_COUNT", 9)
    timing = scorer.load_timing_labels(
        tmp_path / "shots_master.csv",
        loaded.split,
        lambda _path, *, usecols: table.loc[:, usecols].copy(),
    )

    def missing_side_row(_path: Path, *, usecols: list[str]) -> pd.DataFrame:
        return table.iloc[:-1].loc[:, usecols].copy()

    with pytest.raises(ValueError, match="identities differ"):
        scorer.load_player_side_labels(
            tmp_path / "shots_master.csv",
            loaded.split,
            timing.identities,
            missing_side_row,
        )


def test_full_score_is_repeatable_and_uses_predictions_before_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_path, menu_path, raw_path, labels_path, _payload = _prediction_files(tmp_path)
    table = _label_table().iloc[:-1].copy()
    monkeypatch.setattr(scorer, "EXPECTED_RALLY_COUNT", 8)
    monkeypatch.setattr(scorer, "EXPECTED_CONTACT_COUNT", 8)
    calls: list[tuple[str, ...]] = []

    def read_table(_path: Path, *, usecols: list[str]) -> pd.DataFrame:
        calls.append(tuple(usecols))
        return table.loc[:, usecols].copy()

    output = tmp_path / "rally_result.json.gz"

    def run_once() -> bytes:
        calls.clear()
        scorer.score_validation_rallies(
            prediction_path,
            menu_path,
            SPLIT_PATH,
            raw_path,
            labels_path,
            output,
            "deadbee",
            table_reader=read_table,
        )
        assert calls == [scorer.TIMING_COLUMNS, scorer.SIDE_COLUMNS]
        return output.read_bytes()

    first = run_once()
    second = run_once()
    assert first == second
    with gzip.open(output, "rt", encoding="utf-8") as source:
        result = json.load(source)
    assert result["status"] == "complete"
    assert len(result["runs"]) == 9
    assert result["runs"][0]["primary"]["confidence_curve"][0][
        "fully_correct_kept_rallies"
    ] == 8
    assert result["runs"][0]["player_side"]["by_timing_tolerance"]["10"] == {
        "tolerance_at_30_fps": 10,
        "timing_matched_contacts": 8,
        "player_side_answers": 8,
        "correct_player_sides": 8,
        "answer_rate_for_timing_matches": 1.0,
        "accuracy_when_answered": 1.0,
        "correct_rate_for_timing_matches": 1.0,
    }


def test_bad_prediction_prevents_any_label_table_read(tmp_path: Path) -> None:
    prediction_path, menu_path, raw_path, labels_path, payload = _prediction_files(tmp_path)
    payload["runs"][0]["run_id"] = "changed"
    _write_prediction(prediction_path, payload)
    calls = 0

    def read_table(_path: Path, *, usecols: list[str]) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return _label_table().loc[:, usecols].copy()

    with pytest.raises(ValueError, match="saved run order differs"):
        scorer.score_validation_rallies(
            prediction_path,
            menu_path,
            SPLIT_PATH,
            raw_path,
            labels_path,
            tmp_path / "result.json.gz",
            "deadbee",
            table_reader=read_table,
        )
    assert calls == 0


def test_label_file_is_rechecked_after_the_timing_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_path, menu_path, raw_path, labels_path, _payload = _prediction_files(tmp_path)
    table = _label_table().iloc[:-1].copy()
    monkeypatch.setattr(scorer, "EXPECTED_RALLY_COUNT", 8)
    monkeypatch.setattr(scorer, "EXPECTED_CONTACT_COUNT", 8)
    calls = 0

    def change_after_read(_path: Path, *, usecols: list[str]) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        labels_path.write_bytes(b"changed during scoring")
        return table.loc[:, usecols].copy()

    with pytest.raises(ValueError, match="changed during the timing-label read"):
        scorer.score_validation_rallies(
            prediction_path,
            menu_path,
            SPLIT_PATH,
            raw_path,
            labels_path,
            tmp_path / "result.json.gz",
            "deadbee",
            table_reader=change_after_read,
        )
    assert calls == 1


def test_bad_source_commit_replaces_old_result_with_running(tmp_path: Path) -> None:
    prediction_path, menu_path, raw_path, labels_path, _payload = _prediction_files(tmp_path)
    output = tmp_path / "rally_result.json.gz"
    scorer._write_json(output, {"schema": scorer.RALLY_RESULT_SCHEMA, "status": "complete"})

    with pytest.raises(ValueError, match="source commit"):
        scorer.score_validation_rallies(
            prediction_path,
            menu_path,
            SPLIT_PATH,
            raw_path,
            labels_path,
            output,
            "bad",
        )

    with gzip.open(output, "rt", encoding="utf-8") as source:
        result = json.load(source)
    assert result["status"] == "running"
