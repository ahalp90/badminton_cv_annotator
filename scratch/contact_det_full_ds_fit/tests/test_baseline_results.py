from __future__ import annotations

import builtins
import hashlib
import json
import lzma
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scratch.contact_det_full_ds_fit.scripts.baseline_results as results
from scratch.contact_det.scripts.freeze_tree_contact_features import REGION_FIELDS
from scratch.contact_det_full_ds_fit.scripts.baseline_config import (
    FIXED_RUN_IDS,
    MotionMode,
    load_baseline_config,
)
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    load_development_split,
)
from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
    VerifiedFeatureDataset,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    RESULT_FILE,
    SCORE_FILE,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXPERIMENT_ROOT / "records/baseline_runs.json"
SPLIT_PATH = EXPERIMENT_ROOT / "records/shuttleset_development_split.json"
SOURCE_COMMIT = "deadbee"
FEATURE_SOURCE_COMMIT = "featurebee"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_features(record_path: Path) -> VerifiedFeatureDataset:
    split = load_development_split(SPLIT_PATH)
    rows = np.zeros(
        len(split.videos),
        dtype=np.dtype(
            [
                ("fixture", "S16"),
                ("interval_id", "<i4"),
                ("frame", "<i4"),
                ("fps", "<f4"),
                ("model_value", "<f4"),
                *((field, "u1") for field in REGION_FIELDS),
            ]
        ),
    )
    video_ranges: dict[str, tuple[int, int]] = {}
    for index, video in enumerate(split.videos):
        rows["fixture"][index] = video.fixture.encode("ascii")
        rows["interval_id"][index] = 0
        rows["frame"][index] = video.video_id * 10
        rows["fps"][index] = video.fps
        rows["model_value"][index] = 1.0
        rows[REGION_FIELDS[0]][index] = 1
        video_ranges[video.fixture] = (index, index + 1)
    return VerifiedFeatureDataset(
        record_path,
        {
            "source_commit": FEATURE_SOURCE_COMMIT,
            "videos": [
                {
                    "video": {"name": video.fixture, "video_id": video.video_id},
                    "input_files": [
                        {
                            "role": "annotation",
                            "filename": "annotator_result.json.gz",
                            "size_bytes": video.video_id,
                            "sha256": f"{video.video_id:064x}",
                        }
                    ],
                }
                for video in split.videos
            ],
        },
        split,
        rows,
        video_ranges,
        ("model_value",),
    )


def _write_scores(path: Path, features: VerifiedFeatureDataset) -> np.ndarray:
    validation_rows = results._validation_candidates(features).identities
    scores = np.empty(len(validation_rows), dtype=results.SCORE_DTYPE)
    for field in results.IDENTITY_FIELDS:
        scores[field] = validation_rows[field]
    scores["contact_score"] = 0.9
    scores["kept"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(path, "wb") as target:
        np.save(target, scores, allow_pickle=False)
    return scores


def _child_result(
    run_id: str,
    config_path: Path,
    split_path: Path,
    label_path: Path,
    feature_path: Path,
    features: VerifiedFeatureDataset,
    scores: np.ndarray,
    result_path: Path,
) -> dict[str, Any]:
    config = load_baseline_config(config_path)
    split = features.split
    run = next(item for item in config.runs if item.run_id == run_id)
    video_metrics = {
        "matched": 1,
        "contact_count": 1,
        "prediction_count": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "first_contact_matched": 1,
        "first_contact_count": 1,
        "first_contact_recall": 1.0,
        "other_contact_matched": 0,
        "other_contact_count": 0,
        "other_contact_recall": 0.0,
        "median_absolute_frame_error": 0.0,
    }
    aggregate_metrics = {
        **video_metrics,
        "matched": len(split.validation_videos),
        "contact_count": len(split.validation_videos),
        "prediction_count": len(split.validation_videos),
        "first_contact_matched": len(split.validation_videos),
        "first_contact_count": len(split.validation_videos),
    }
    result = {
        "schema": results.RESULT_SCHEMA,
        "status": "complete",
        "run_id": run_id,
        "source_commit": SOURCE_COMMIT,
        "config_file": config_path.name,
        "config_sha256": _sha256(config_path),
        "feature_record_file": feature_path.name,
        "feature_record_sha256": _sha256(feature_path),
        "feature_source_commit": FEATURE_SOURCE_COMMIT,
        "split_file": split_path.name,
        "split_sha256": _sha256(split_path),
        "contact_label_file": label_path.name,
        "contact_label_sha256": _sha256(label_path),
        "training_videos": [video.fixture for video in split.training_videos],
        "validation_videos": [video.fixture for video in split.validation_videos],
        "feature_row_count": len(features.rows),
        "candidate_row_count": results._validation_candidates(features).full_count,
        "feature_names": list(features.model_input_fields),
        "model": {
            "name": run.model_name,
            "kind": run.model_kind.value,
            "settings": dict(run.model_settings),
            "class_weight": run.class_weight,
            "class_weight_value": run.class_weight_value,
        },
        "training_selection": {
            "positive_radius_at_30_fps": 1,
            "ignored_radius_at_30_fps": 4,
            "nearby_negative_radius_at_30_fps": 15,
            "negative_rule": run.negative_rule,
            "maximum_negatives_per_positive": run.negative_limit,
            "selected_row_count": len(split.training_videos) * 2,
            "positive_row_count": len(split.training_videos),
            "videos": {
                video.fixture: {
                    "positive": 1,
                    "nearby_negative": 1,
                    "sampled_other_negative": 0,
                    "selected": 2,
                }
                for video in split.training_videos
            },
        },
        "selected_score_cutoff": 0.05,
        "selected_duplicate_distance_at_30_fps": 4,
        "selection_metrics": aggregate_metrics,
        "metrics": {tolerance: aggregate_metrics for tolerance in results.REPORT_TOLERANCES},
        "videos": {
            video.fixture: {
                "rally_count": 1,
                "prediction_frames": [video.video_id * 10],
                "metrics": {
                    tolerance: video_metrics for tolerance in results.REPORT_TOLERANCES
                },
            }
            for video in split.validation_videos
        },
        "validation_score_file": SCORE_FILE,
        "validation_score_sha256": _sha256(result_path.parent / SCORE_FILE),
        "validation_score_row_count": len(scores),
    }
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return result


def _build_valid_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    raw_record = tmp_path / "raw" / "contact_features_record.json"
    common30_record = tmp_path / "common30" / "contact_features_record.json"
    label_file = tmp_path / "shots_master.csv"
    raw_record.parent.mkdir(parents=True)
    common30_record.parent.mkdir(parents=True)
    raw_record.write_bytes(b"raw feature record")
    common30_record.write_bytes(b"common30 feature record")
    label_file.write_bytes(b"labels are only hashed")
    raw_features = _fake_features(raw_record)
    common30_features = _fake_features(common30_record)

    def fake_loader(_record_path: Path, _split_path: Path, motion_mode: str) -> VerifiedFeatureDataset:
        if motion_mode == MotionMode.RAW_PER_FRAME.value:
            return raw_features
        if motion_mode == MotionMode.BASE30_PER_FRAME.value:
            return common30_features
        raise AssertionError(motion_mode)

    monkeypatch.setattr(results, "load_verified_feature_dataset", fake_loader)
    output_dir = tmp_path / "results"
    menu_path = output_dir / "baseline_menu_result.json"
    config = load_baseline_config(CONFIG_PATH)
    completed: list[dict[str, str]] = []
    for run in config.runs:
        feature_path = raw_record if run.motion_mode is MotionMode.RAW_PER_FRAME else common30_record
        feature_set = raw_features if run.motion_mode is MotionMode.RAW_PER_FRAME else common30_features
        run_dir = output_dir / run.run_id
        run_dir.mkdir(parents=True)
        scores = _write_scores(run_dir / SCORE_FILE, feature_set)
        result_path = run_dir / RESULT_FILE
        _child_result(run.run_id, CONFIG_PATH, SPLIT_PATH, label_file, feature_path, feature_set, scores, result_path)
        completed.append(
            {
                "run_id": run.run_id,
                "result_file": f"{run.run_id}/{RESULT_FILE}",
                "result_sha256": _sha256(result_path),
            }
        )
    menu = {
        "schema": "full-dataset-contact-baseline-menu-result/1",
        "status": "complete",
        "source_commit": SOURCE_COMMIT,
        "config_file": CONFIG_PATH.name,
        "config_sha256": _sha256(CONFIG_PATH),
        "split_file": SPLIT_PATH.name,
        "split_sha256": _sha256(SPLIT_PATH),
        "contact_label_file": label_file.name,
        "contact_label_sha256": _sha256(label_file),
        "feature_records": {
            MotionMode.RAW_PER_FRAME.value: {
                "filename": raw_record.name,
                "sha256": _sha256(raw_record),
            },
            MotionMode.BASE30_PER_FRAME.value: {
                "filename": common30_record.name,
                "sha256": _sha256(common30_record),
            },
        },
        "run_ids": list(FIXED_RUN_IDS),
        "completed_runs": completed,
    }
    menu_path.parent.mkdir(parents=True, exist_ok=True)
    menu_path.write_text(json.dumps(menu, sort_keys=True), encoding="utf-8")
    return menu_path, menu


def test_valid_nine_run_menu_is_loaded_without_reading_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    menu_path, _menu = _build_valid_menu(tmp_path, monkeypatch)
    label_path = tmp_path / "shots_master.csv"
    old_read_text = Path.read_text

    def fail_label_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == label_path:
            raise AssertionError("label text must not be read")
        return old_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_label_text)
    old_import = builtins.__import__

    def fail_pandas(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pandas" or name.startswith("pandas."):
            raise AssertionError("pandas must not be imported")
        return old_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_pandas)
    loaded = results.load_completed_baseline_menu(
        menu_path,
        CONFIG_PATH,
        SPLIT_PATH,
        tmp_path / "raw" / "contact_features_record.json",
        tmp_path / "common30" / "contact_features_record.json",
        label_path,
    )

    assert loaded.raw_features.record["source_commit"] == FEATURE_SOURCE_COMMIT
    assert [run.run.run_id for run in loaded.runs] == list(FIXED_RUN_IDS)
    assert loaded.results == loaded.runs
    assert tuple(loaded.runs_by_id) == FIXED_RUN_IDS
    assert all(run.score_rows.dtype == results.SCORE_DTYPE for run in loaded.runs)
    assert all(run.kept.all() for run in loaded.runs)
    assert not hasattr(loaded, "common30_features")


def _reload_menu_hash(menu_path: Path, result_path: Path) -> None:
    menu = json.loads(menu_path.read_text(encoding="utf-8"))
    for item in menu["completed_runs"]:
        if item["run_id"] == result_path.parent.name:
            item["result_sha256"] = _sha256(result_path)
    menu_path.write_text(json.dumps(menu, sort_keys=True), encoding="utf-8")


def _load_with_expected_error(
    menu_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        results.load_completed_baseline_menu(
            menu_path,
            CONFIG_PATH,
            SPLIT_PATH,
            tmp_path / "raw" / "contact_features_record.json",
            tmp_path / "common30" / "contact_features_record.json",
            tmp_path / "shots_master.csv",
        )


def test_changed_child_hash_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    menu_path, _menu = _build_valid_menu(tmp_path, monkeypatch)
    result_path = menu_path.parent / FIXED_RUN_IDS[0] / RESULT_FILE
    result_path.write_text(result_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _load_with_expected_error(menu_path, tmp_path, monkeypatch, "result hash differs")


def test_non_complete_child_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    menu_path, _menu = _build_valid_menu(tmp_path, monkeypatch)
    result_path = menu_path.parent / FIXED_RUN_IDS[0] / RESULT_FILE
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["status"] = "running"
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    _reload_menu_hash(menu_path, result_path)
    _load_with_expected_error(menu_path, tmp_path, monkeypatch, "result is not complete")


def test_score_identity_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    menu_path, _menu = _build_valid_menu(tmp_path, monkeypatch)
    result_path = menu_path.parent / FIXED_RUN_IDS[0] / RESULT_FILE
    score_path = result_path.parent / SCORE_FILE
    with lzma.open(score_path, "rb") as source:
        scores = np.load(source, allow_pickle=False)
    scores["frame"][0] += 1
    with lzma.open(score_path, "wb") as target:
        np.save(target, scores, allow_pickle=False)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["validation_score_sha256"] = _sha256(score_path)
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    _reload_menu_hash(menu_path, result_path)
    _load_with_expected_error(menu_path, tmp_path, monkeypatch, "score identities differ")


def test_kept_mask_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    menu_path, _menu = _build_valid_menu(tmp_path, monkeypatch)
    result_path = menu_path.parent / FIXED_RUN_IDS[0] / RESULT_FILE
    score_path = result_path.parent / SCORE_FILE
    with lzma.open(score_path, "rb") as source:
        scores = np.load(source, allow_pickle=False)
    scores["kept"][0] = False
    with lzma.open(score_path, "wb") as target:
        np.save(target, scores, allow_pickle=False)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["validation_score_sha256"] = _sha256(score_path)
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    _reload_menu_hash(menu_path, result_path)
    _load_with_expected_error(menu_path, tmp_path, monkeypatch, "saved kept flags differ")


def test_wrong_prediction_frames_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    menu_path, _menu = _build_valid_menu(tmp_path, monkeypatch)
    result_path = menu_path.parent / FIXED_RUN_IDS[0] / RESULT_FILE
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["videos"]["sset_18"]["prediction_frames"] = [999]
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    _reload_menu_hash(menu_path, result_path)
    _load_with_expected_error(menu_path, tmp_path, monkeypatch, "prediction frames differ")


def test_wrong_training_video_set_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    menu_path, _menu = _build_valid_menu(tmp_path, monkeypatch)
    result_path = menu_path.parent / FIXED_RUN_IDS[0] / RESULT_FILE
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["training_selection"]["videos"].pop("sset_01")
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    _reload_menu_hash(menu_path, result_path)
    _load_with_expected_error(menu_path, tmp_path, monkeypatch, "training selection videos differ")


def test_different_raw_and_common30_inputs_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    menu_path, _menu = _build_valid_menu(tmp_path, monkeypatch)
    raw_path = tmp_path / "raw" / "contact_features_record.json"
    common30_path = tmp_path / "common30" / "contact_features_record.json"
    raw_features = _fake_features(raw_path)
    common30_features = _fake_features(common30_path)
    common30_features.record["videos"][0]["input_files"][0]["sha256"] = "f" * 64

    def fake_loader(_record: Path, _split: Path, motion_mode: str) -> VerifiedFeatureDataset:
        return raw_features if motion_mode == MotionMode.RAW_PER_FRAME.value else common30_features

    monkeypatch.setattr(results, "load_verified_feature_dataset", fake_loader)
    _load_with_expected_error(menu_path, tmp_path, monkeypatch, "feature input files differ")


def test_non_finite_contact_score_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    menu_path, _menu = _build_valid_menu(tmp_path, monkeypatch)
    result_path = menu_path.parent / FIXED_RUN_IDS[0] / RESULT_FILE
    score_path = result_path.parent / SCORE_FILE
    with lzma.open(score_path, "rb") as source:
        scores = np.load(source, allow_pickle=False)
    scores["contact_score"][0] = np.nan
    with lzma.open(score_path, "wb") as target:
        np.save(target, scores, allow_pickle=False)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["validation_score_sha256"] = _sha256(score_path)
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    _reload_menu_hash(menu_path, result_path)
    _load_with_expected_error(menu_path, tmp_path, monkeypatch, "finite values from 0 to 1")


def test_changed_training_rule_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    menu_path, _menu = _build_valid_menu(tmp_path, monkeypatch)
    result_path = menu_path.parent / FIXED_RUN_IDS[0] / RESULT_FILE
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["training_selection"]["maximum_negatives_per_positive"] = 99
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    _reload_menu_hash(menu_path, result_path)
    _load_with_expected_error(menu_path, tmp_path, monkeypatch, "training selection settings differ")


def test_metric_rate_that_disagrees_with_counts_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    menu_path, _menu = _build_valid_menu(tmp_path, monkeypatch)
    result_path = menu_path.parent / FIXED_RUN_IDS[0] / RESULT_FILE
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["metrics"]["5"]["precision"] = 0.5
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    _reload_menu_hash(menu_path, result_path)
    _load_with_expected_error(menu_path, tmp_path, monkeypatch, "precision differs")
