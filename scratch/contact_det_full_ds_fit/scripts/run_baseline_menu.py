"""Run the fixed nine contact-model comparisons in their written order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from scratch.contact_det_full_ds_fit.scripts.baseline_config import (
    FIXED_RUN_IDS,
    BaselineRun,
    MotionMode,
    load_baseline_config,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    RESULT_FILE,
    RESULT_SCHEMA,
    run_baseline,
)

MENU_RESULT_SCHEMA = "full-dataset-contact-baseline-menu-result/1"
MENU_RESULT_FILE = "baseline_menu_result.json"

RunOne = Callable[[Path, str, Path, Path, Path, Path, str], Path]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _feature_record_for_run(
    run: BaselineRun,
    raw_feature_record: Path,
    common30_feature_record: Path,
) -> Path:
    if run.motion_mode is MotionMode.RAW_PER_FRAME:
        return Path(raw_feature_record)
    if run.motion_mode is MotionMode.BASE30_PER_FRAME:
        return Path(common30_feature_record)
    raise ValueError(f"unsupported motion choice: {run.motion_mode}")


def _verify_completed_result(path: Path, run_id: str, source_commit: str) -> None:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{run_id}: result file must contain an object")
    expected = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "run_id": run_id,
        "source_commit": source_commit,
    }
    if any(value.get(field) != expected_value for field, expected_value in expected.items()):
        raise ValueError(f"{run_id}: result file is not the expected complete run")


def run_menu(
    config_path: Path,
    split_path: Path,
    raw_feature_record: Path,
    common30_feature_record: Path,
    shots_master_path: Path,
    output_dir: Path,
    source_commit: str,
    *,
    run_one: RunOne = run_baseline,
) -> Path:
    """Run every fixed comparison and save path-free progress."""
    config_file = Path(config_path)
    split_file = Path(split_path)
    raw_record = Path(raw_feature_record)
    common30_record = Path(common30_feature_record)
    label_file = Path(shots_master_path)
    result_root = Path(output_dir)
    menu_result_path = result_root / MENU_RESULT_FILE
    _write_json(
        menu_result_path,
        {
            "schema": MENU_RESULT_SCHEMA,
            "status": "running",
            "source_commit": source_commit,
        },
    )
    for fixed_run_id in FIXED_RUN_IDS:
        _write_json(
            result_root / fixed_run_id / RESULT_FILE,
            {
                "schema": RESULT_SCHEMA,
                "status": "running",
                "run_id": fixed_run_id,
                "source_commit": source_commit,
            },
        )
    try:
        config = load_baseline_config(config_file)
        shared: dict[str, object] = {
            "schema": MENU_RESULT_SCHEMA,
            "source_commit": source_commit,
            "config_file": config_file.name,
            "config_sha256": _sha256(config_file),
            "split_file": split_file.name,
            "split_sha256": _sha256(split_file),
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
            "run_ids": [run.run_id for run in config.runs],
        }
    except Exception as error:
        _write_json(
            menu_result_path,
            {
                "schema": MENU_RESULT_SCHEMA,
                "status": "failed",
                "source_commit": source_commit,
                "failed_stage": "setup",
                "error_type": type(error).__name__,
            },
        )
        raise
    completed: list[dict[str, str]] = []
    _write_json(menu_result_path, {**shared, "status": "running", "completed_runs": completed})
    for run in config.runs:
        feature_record = _feature_record_for_run(run, raw_record, common30_record)
        try:
            result_path = run_one(
                config_file,
                run.run_id,
                feature_record,
                split_file,
                label_file,
                result_root,
                source_commit,
            )
            expected_result_path = result_root / run.run_id / RESULT_FILE
            if result_path != expected_result_path:
                raise ValueError(f"{run.run_id}: result file is outside its expected run folder")
            _verify_completed_result(result_path, run.run_id, source_commit)
        except Exception as error:
            _write_json(
                menu_result_path,
                {
                    **shared,
                    "status": "failed",
                    "completed_runs": completed,
                    "failed_run_id": run.run_id,
                    "error_type": type(error).__name__,
                },
            )
            raise
        completed.append(
            {
                "run_id": run.run_id,
                "result_file": f"{run.run_id}/{RESULT_FILE}",
                "result_sha256": _sha256(result_path),
            }
        )
        _write_json(
            menu_result_path,
            {**shared, "status": "running", "completed_runs": completed},
        )
        print(f"finished {run.run_id}", flush=True)

    _write_json(
        menu_result_path,
        {**shared, "status": "complete", "completed_runs": completed},
    )
    return menu_result_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--raw-feature-record", type=Path, required=True)
    parser.add_argument("--common30-feature-record", type=Path, required=True)
    parser.add_argument("--shots-master", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    result_path = run_menu(
        arguments.config,
        arguments.split,
        arguments.raw_feature_record,
        arguments.common30_feature_record,
        arguments.shots_master,
        arguments.output_dir,
        arguments.source_commit,
    )
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
