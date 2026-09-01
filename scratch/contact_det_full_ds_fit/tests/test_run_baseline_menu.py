from __future__ import annotations

import json
from pathlib import Path

import pytest

from scratch.contact_det_full_ds_fit.scripts.run_baseline_menu import (
    MENU_RESULT_FILE,
    run_menu,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import RESULT_FILE

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXPERIMENT_ROOT / "records/baseline_runs.json"
SPLIT_PATH = EXPERIMENT_ROOT / "records/shuttleset_development_split.json"
EXPECTED_RUN_IDS = [
    "hgb_reference_raw_balanced",
    "hgb_reference_common30_balanced",
    "rf_reference_raw_balanced",
    "rf_reference_common30_balanced",
    "hgb_reference_raw_no_weight",
    "rf_reference_raw_no_weight",
    "hgb_15_leaves_raw_balanced",
    "hgb_learning_rate_004_raw_balanced",
    "hgb_reference_raw_more_negatives",
]


def _input_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw_record = tmp_path / "raw" / "contact_features_record.json"
    common30_record = tmp_path / "common30" / "contact_features_record.json"
    labels = tmp_path / "shots_master.csv"
    for path, contents in (
        (raw_record, "raw"),
        (common30_record, "common30"),
        (labels, "labels"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    return raw_record, common30_record, labels


def test_menu_runs_the_fixed_order_with_the_right_motion_files(tmp_path: Path) -> None:
    raw_record, common30_record, labels = _input_files(tmp_path)
    output_dir = tmp_path / "results"
    calls: list[tuple[str, Path]] = []

    def run_one(
        _config: Path,
        run_id: str,
        feature_record: Path,
        _split: Path,
        _labels: Path,
        result_root: Path,
        _source_commit: str,
    ) -> Path:
        calls.append((run_id, feature_record))
        result_path = result_root / run_id / RESULT_FILE
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "schema": "full-dataset-contact-baseline-result/1",
                    "status": "complete",
                    "run_id": run_id,
                    "source_commit": "deadbee",
                }
            ),
            encoding="utf-8",
        )
        return result_path

    menu_path = run_menu(
        CONFIG_PATH,
        SPLIT_PATH,
        raw_record,
        common30_record,
        labels,
        output_dir,
        "deadbee",
        run_one=run_one,
    )

    result = json.loads(menu_path.read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert [run_id for run_id, _path in calls] == EXPECTED_RUN_IDS
    assert result["run_ids"] == EXPECTED_RUN_IDS
    assert [path for _run_id, path in calls] == [
        raw_record,
        common30_record,
        raw_record,
        common30_record,
        raw_record,
        raw_record,
        raw_record,
        raw_record,
        raw_record,
    ]
    assert len(result["completed_runs"]) == 9
    assert str(tmp_path) not in menu_path.read_text(encoding="utf-8")


def test_menu_records_the_failed_run_without_a_machine_path(tmp_path: Path) -> None:
    raw_record, common30_record, labels = _input_files(tmp_path)
    output_dir = tmp_path / "results"
    call_count = 0

    def run_one(*_args: object) -> Path:
        nonlocal call_count
        call_count += 1
        raise RuntimeError(f"failure at {tmp_path}")

    with pytest.raises(RuntimeError, match="failure at"):
        run_menu(
            CONFIG_PATH,
            SPLIT_PATH,
            raw_record,
            common30_record,
            labels,
            output_dir,
            "deadbee",
            run_one=run_one,
        )

    menu_path = output_dir / MENU_RESULT_FILE
    result_text = menu_path.read_text(encoding="utf-8")
    result = json.loads(result_text)
    assert call_count == 1
    assert result["status"] == "failed"
    assert result["failed_run_id"] == "hgb_reference_raw_balanced"
    assert result["error_type"] == "RuntimeError"
    assert str(tmp_path) not in result_text
    old_second_result = output_dir / EXPECTED_RUN_IDS[1] / RESULT_FILE
    assert json.loads(old_second_result.read_text(encoding="utf-8"))["status"] == "running"


def test_menu_marks_an_old_result_running_before_input_checks(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    menu_path = output_dir / MENU_RESULT_FILE
    menu_path.parent.mkdir(parents=True)
    menu_path.write_text('{"status": "complete"}', encoding="utf-8")
    old_child = output_dir / EXPECTED_RUN_IDS[0] / RESULT_FILE
    old_child.parent.mkdir(parents=True)
    old_child.write_text('{"status": "complete"}', encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        run_menu(
            tmp_path / "missing-config.json",
            SPLIT_PATH,
            tmp_path / "raw.json",
            tmp_path / "common30.json",
            tmp_path / "shots_master.csv",
            output_dir,
            "deadbee",
        )

    result = json.loads(menu_path.read_text(encoding="utf-8"))
    assert result == {
        "error_type": "FileNotFoundError",
        "failed_stage": "setup",
        "schema": "full-dataset-contact-baseline-menu-result/1",
        "source_commit": "deadbee",
        "status": "failed",
    }
    assert json.loads(old_child.read_text(encoding="utf-8"))["status"] == "running"


def test_menu_rejects_a_child_result_that_is_not_complete(tmp_path: Path) -> None:
    raw_record, common30_record, labels = _input_files(tmp_path)
    output_dir = tmp_path / "results"

    def run_one(
        _config: Path,
        run_id: str,
        _feature_record: Path,
        _split: Path,
        _labels: Path,
        result_root: Path,
        _source_commit: str,
    ) -> Path:
        result_path = result_root / run_id / RESULT_FILE
        result_path.write_text('{"status": "running"}', encoding="utf-8")
        return result_path

    with pytest.raises(ValueError, match="not the expected complete run"):
        run_menu(
            CONFIG_PATH,
            SPLIT_PATH,
            raw_record,
            common30_record,
            labels,
            output_dir,
            "deadbee",
            run_one=run_one,
        )

    result = json.loads((output_dir / MENU_RESULT_FILE).read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["failed_run_id"] == EXPECTED_RUN_IDS[0]
