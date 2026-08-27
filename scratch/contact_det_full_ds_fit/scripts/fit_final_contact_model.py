"""Fit and reload-check the fixed contact model on all 40 videos."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from scratch.contact_det_full_ds_fit.scripts.baseline_config import (
    BaselineConfig,
    BaselineRun,
)
from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
    VerifiedFeatureDataset,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    CandidateRows,
    ContactLabels,
    _feature_matrix,
    _sha256,
    _write_json,
    collect_candidate_rows,
    make_model,
)
from scratch.contact_det_full_ds_fit.scripts.score_final_contact_groups import (
    COMBINED_RESULT_SCHEMA,
    EXPECTED_SCORE_ROW_COUNT,
    GROUP_NAMES,
    SOURCE_COMMIT,
    InputFiles,
    LabelLoader,
    _check_candidate_rows,
    _expected_input_hash,
    _final_scores,
    _input_records,
    _load_fixed_inputs,
    _load_json,
    _mapping,
    _read_final_scores,
    _read_raw_scores,
    choose_final_settings,
    load_contact_labels_for_videos,
    load_final_score_groups,
)
from scratch.contact_det_full_ds_fit.scripts.score_training_videos import (
    choose_training_rows_for_videos,
)

RESULT_SCHEMA = "final-contact-model-fit/1"
RESULT_FILE = "final_contact_model_result.json"
MODEL_FILE = "contact_model.joblib"
EXPECTED_VIDEO_COUNT = 40
EXPECTED_CHECK_ROW_COUNT = 80
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class FinalFitFiles:
    """Fixed experiment files and the completed held-out setting result."""

    inputs: InputFiles
    combined_result: Path
    combined_raw_scores: Path
    final_scores: Path


def _check_complete_setting_result(
    files: FinalFitFiles,
    features: VerifiedFeatureDataset,
    candidates: CandidateRows,
    score_source_commit: str,
) -> tuple[Mapping[str, Any], np.ndarray]:
    result = _load_json(files.combined_result, "final contact setting result")
    expected_groups = load_final_score_groups(files.inputs.groups, features.split)
    expected_videos = [
        video.fixture
        for group_name in GROUP_NAMES
        for video in expected_groups[group_name].scored_videos
    ]
    expected_fields = {
        "schema": COMBINED_RESULT_SCHEMA,
        "status": "complete",
        "source_commit": score_source_commit,
        "labels_read": True,
        "input_list_file": files.inputs.input_list.name,
        "input_list_sha256": _sha256(files.inputs.input_list),
        "input_files": _input_records(files.inputs),
        "groups": list(GROUP_NAMES),
        "videos": expected_videos,
        "combined_raw_score_file": files.combined_raw_scores.name,
        "combined_raw_score_sha256": _sha256(files.combined_raw_scores),
        "combined_raw_score_row_count": EXPECTED_SCORE_ROW_COUNT,
        "final_score_file": files.final_scores.name,
        "final_score_sha256": _sha256(files.final_scores),
    }
    if any(result.get(field) != value for field, value in expected_fields.items()):
        raise ValueError("final contact setting result differs")
    group_files = _mapping(result.get("group_files"), "final contact group files")
    if set(group_files) != set(GROUP_NAMES):
        raise ValueError("final contact group file records differ")

    raw_scores = _read_raw_scores(files.combined_raw_scores)
    if len(raw_scores) != EXPECTED_SCORE_ROW_COUNT:
        raise ValueError("combined raw contact score count differs")
    expected_rows = np.concatenate(
        [
            candidates.rows[
                candidates.video_ranges[video.fixture][0] : candidates.video_ranges[
                    video.fixture
                ][1]
            ]
            for group_name in GROUP_NAMES
            for video in expected_groups[group_name].scored_videos
        ]
    )
    identity_fields = ("fixture", "interval_id", "frame", "fps")
    if len(raw_scores) != len(expected_rows) or any(
        not np.array_equal(raw_scores[field], expected_rows[field])
        for field in identity_fields
    ):
        raise ValueError("combined raw contact score order differs")
    expected_sources = np.concatenate(
        [
            np.full(
                sum(
                    candidates.video_ranges[video.fixture][1]
                    - candidates.video_ranges[video.fixture][0]
                    for video in expected_groups[group_name].scored_videos
                ),
                group_name.encode("ascii"),
                dtype="S1",
            )
            for group_name in GROUP_NAMES
        ]
    )
    if not np.array_equal(raw_scores["source_group"], expected_sources):
        raise ValueError("combined raw contact score sources differ")
    return result, raw_scores


def _check_and_recount_settings(
    result: Mapping[str, Any],
    raw_scores: np.ndarray,
    final_score_path: Path,
    features: VerifiedFeatureDataset,
    labels: ContactLabels,
    config: BaselineConfig,
) -> tuple[float, int]:
    setting_results, cutoff, distance, metrics, _predictions, kept = (
        choose_final_settings(
            raw_scores,
            features.split.videos,
            labels,
            config,
        )
    )
    expected = {
        "setting_results": setting_results,
        "selected_score_cutoff": cutoff,
        "selected_duplicate_distance_at_30_fps": distance,
        "selected_metrics_at_5_frames": metrics,
        "kept_contact_count": int(kept.sum()),
    }
    if any(result.get(field) != value for field, value in expected.items()):
        raise ValueError("final contact setting choice differs on recount")
    final_scores = _read_final_scores(final_score_path)
    if not np.array_equal(final_scores, _final_scores(raw_scores, kept)):
        raise ValueError("final kept-contact scores differ on recount")
    return cutoff, distance


def model_check_rows(
    candidates: CandidateRows,
    features: VerifiedFeatureDataset,
) -> np.ndarray:
    """Return the first and last candidate row from each development video."""
    positions: list[int] = []
    for video in features.split.videos:
        row_start, row_end = candidates.video_ranges[video.fixture]
        if row_end <= row_start:
            raise ValueError(f"{video.fixture}: there is no contact candidate row")
        positions.extend((row_start, row_end - 1))
    rows = candidates.rows[np.asarray(positions, dtype=np.int64)]
    if len(rows) != EXPECTED_CHECK_ROW_COUNT:
        raise ValueError("final model check row count differs")
    return rows


def _contact_probabilities(
    model: Any,
    rows: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    if not np.array_equal(np.asarray(model.classes_), np.asarray([0, 1])):
        raise ValueError("contact model classes differ")
    probabilities = np.asarray(
        model.predict_proba(_feature_matrix(rows, feature_names))[:, 1],
        dtype=np.float64,
    )
    if len(probabilities) != len(rows) or not np.isfinite(probabilities).all():
        raise ValueError("final model check probabilities differ")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("final model check probabilities are outside zero and one")
    return probabilities


def model_check_records(
    rows: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, int | float | str]]:
    """Make a path-free record of the fixed 80-row model check."""
    if len(rows) != EXPECTED_CHECK_ROW_COUNT or len(probabilities) != len(rows):
        raise ValueError("final model check rows differ")
    return [
        {
            "fixture": bytes(row["fixture"]).decode("ascii"),
            "interval_id": int(row["interval_id"]),
            "frame": int(row["frame"]),
            "fps": float(row["fps"]),
            "contact_score": float(probability),
        }
        for row, probability in zip(rows, probabilities, strict=True)
    ]


def _feature_hashes(features: VerifiedFeatureDataset) -> dict[str, str]:
    raw_records = features.record.get("videos")
    if not isinstance(raw_records, list):
        raise TypeError("feature video records must be a list")
    hashes: dict[str, str] = {}
    for index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record, f"feature video record {index}")
        video = _mapping(record.get("video"), f"feature video identity {index}")
        fixture = video.get("name")
        feature_hash = record.get("feature_sha256")
        if not isinstance(fixture, str) or not isinstance(feature_hash, str):
            raise TypeError("feature video hash record differs")
        hashes[fixture] = feature_hash
    expected_names = [video.fixture for video in features.split.videos]
    if list(hashes) != expected_names:
        raise ValueError("feature video hash order differs")
    return hashes


def save_and_reload_model(
    model: Any,
    model_path: Path,
    check_rows: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[str, np.ndarray]:
    """Save one model file and require the same check scores after loading."""
    import joblib

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    written_files = joblib.dump(model, temporary, compress=3)
    if written_files != [str(temporary)]:
        raise ValueError("final model save produced more than one file")
    os.replace(temporary, destination)
    model_hash = _sha256(destination)
    expected_probabilities = _contact_probabilities(model, check_rows, feature_names)
    loaded_model = joblib.load(destination)
    actual_probabilities = _contact_probabilities(
        loaded_model,
        check_rows,
        feature_names,
    )
    if not np.array_equal(actual_probabilities, expected_probabilities):
        raise ValueError("loaded final model probabilities differ")
    if _sha256(destination) != model_hash:
        raise ValueError("final model file changed while it was checked")
    return model_hash, actual_probabilities


def _library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": version("scikit-learn"),
        "joblib": version("joblib"),
    }


def fit_final_model(
    files: FinalFitFiles,
    output_dir: Path,
    score_source_commit: str,
    fit_source_commit: str,
    *,
    model_factory: Callable[[BaselineRun, int], Any] = make_model,
    label_loader: LabelLoader = load_contact_labels_for_videos,
    feature_loader: Callable[..., VerifiedFeatureDataset] | None = None,
) -> Path:
    """Recount the fixed choice, fit all 40 videos and check the saved model."""
    destination = Path(output_dir)
    result_path = destination / RESULT_FILE
    labels_read = False
    _write_json(
        result_path,
        {
            "schema": RESULT_SCHEMA,
            "status": "running",
            "score_source_commit": score_source_commit,
            "fit_source_commit": fit_source_commit,
            "labels_read": False,
        },
    )
    try:
        if SOURCE_COMMIT.fullmatch(score_source_commit) is None:
            raise ValueError("score source commit must be a short or full Git commit")
        if SOURCE_COMMIT.fullmatch(fit_source_commit) is None:
            raise ValueError("fit source commit must be a short or full Git commit")
        source_root = str(REPO_ROOT / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        inputs, config, run = _load_fixed_inputs(files.inputs)
        if feature_loader is None:
            from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
                load_verified_feature_dataset,
            )

            feature_loader = load_verified_feature_dataset
        features = feature_loader(
            files.inputs.feature_record,
            files.inputs.split,
            run.motion_mode.value,
        )
        candidates = collect_candidate_rows(features)
        _check_candidate_rows(features, candidates, inputs)
        setting_result, raw_scores = _check_complete_setting_result(
            files,
            features,
            candidates,
            score_source_commit,
        )

        labels_read = True
        labels = label_loader(files.inputs.contact_labels, features.split.videos)
        expected_label_hash = _expected_input_hash(inputs, "contact labels")
        if _sha256(files.inputs.contact_labels) != expected_label_hash:
            raise ValueError("contact label file changed while the final model was fit")
        cutoff, distance = _check_and_recount_settings(
            setting_result,
            raw_scores,
            files.final_scores,
            features,
            labels,
            config,
        )
        training = choose_training_rows_for_videos(
            candidates,
            features.split.videos,
            labels,
            config,
            run,
            expected_video_count=EXPECTED_VIDEO_COUNT,
        )
        selected_names = set(
            np.char.decode(candidates.rows[training.selected]["fixture"], "ascii")
        )
        expected_names = {video.fixture for video in features.split.videos}
        if selected_names != expected_names:
            raise ValueError("final training examples contain the wrong videos")

        model = model_factory(run, config.random_seed)
        model.fit(
            _feature_matrix(
                candidates.rows[training.selected],
                features.model_input_fields,
            ),
            training.labels[training.selected],
        )
        check_rows = model_check_rows(candidates, features)
        probabilities_before_save = _contact_probabilities(
            model,
            check_rows,
            features.model_input_fields,
        )
        model_path = destination / MODEL_FILE
        model_hash, probabilities_after_load = save_and_reload_model(
            model,
            model_path,
            check_rows,
            features.model_input_fields,
        )
        if not np.array_equal(probabilities_before_save, probabilities_after_load):
            raise ValueError("final model scores changed during its save check")

        result: dict[str, object] = {
            "schema": RESULT_SCHEMA,
            "status": "complete",
            "score_source_commit": score_source_commit,
            "fit_source_commit": fit_source_commit,
            "labels_read": True,
            "input_list_file": files.inputs.input_list.name,
            "input_list_sha256": _sha256(files.inputs.input_list),
            "input_files": _input_records(files.inputs),
            "setting_result_file": files.combined_result.name,
            "setting_result_sha256": _sha256(files.combined_result),
            "combined_raw_score_file": files.combined_raw_scores.name,
            "combined_raw_score_sha256": _sha256(files.combined_raw_scores),
            "model": {
                "name": run.model_name,
                "kind": run.model_kind.value,
                "settings": dict(run.model_settings),
                "class_weight": run.class_weight,
                "class_weight_value": run.class_weight_value,
            },
            "training_selection": {
                "positive_radius_at_30_fps": config.positive_radius_at_30_fps,
                "ignored_radius_at_30_fps": config.ignored_radius_at_30_fps,
                "nearby_negative_radius_at_30_fps": config.hard_negative_radius_at_30_fps,
                "negative_rule": run.negative_rule,
                "maximum_negatives_per_positive": run.negative_limit,
                "selected_row_count": int(training.selected.sum()),
                "positive_row_count": int(training.labels[training.selected].sum()),
                "videos": training.video_counts,
            },
            "selected_score_cutoff": cutoff,
            "selected_duplicate_distance_at_30_fps": distance,
            "feature_record_file": files.inputs.feature_record.name,
            "feature_record_sha256": _sha256(files.inputs.feature_record),
            "feature_file_sha256_by_video": _feature_hashes(features),
            "model_input_fields": list(features.model_input_fields),
            "model_file": model_path.name,
            "model_sha256": model_hash,
            "model_check_rows": model_check_records(
                check_rows,
                probabilities_after_load,
            ),
            "library_versions": _library_versions(),
        }
        _write_json(result_path, result)
        return result_path
    except Exception as error:
        _write_json(
            result_path,
            {
                "schema": RESULT_SCHEMA,
                "status": "failed",
                "score_source_commit": score_source_commit,
                "fit_source_commit": fit_source_commit,
                "labels_read": labels_read,
                "error_type": type(error).__name__,
            },
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-record", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--chosen-baseline-result", type=Path, required=True)
    parser.add_argument("--contact-labels", type=Path, required=True)
    parser.add_argument("--chosen-validation-scores", type=Path, required=True)
    parser.add_argument("--combined-result", type=Path, required=True)
    parser.add_argument("--combined-raw-scores", type=Path, required=True)
    parser.add_argument("--final-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-source-commit", required=True)
    parser.add_argument("--fit-source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    input_files = InputFiles(
        arguments.input_list,
        arguments.groups,
        arguments.split,
        arguments.config,
        arguments.feature_record,
        arguments.baseline_summary,
        arguments.chosen_baseline_result,
        arguments.contact_labels,
        arguments.chosen_validation_scores,
    )
    result_path = fit_final_model(
        FinalFitFiles(
            input_files,
            arguments.combined_result,
            arguments.combined_raw_scores,
            arguments.final_scores,
        ),
        arguments.output_dir,
        arguments.score_source_commit,
        arguments.fit_source_commit,
    )
    print(result_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
