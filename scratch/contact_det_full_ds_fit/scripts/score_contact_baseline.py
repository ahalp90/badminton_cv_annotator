"""Train and check one fixed contact-model run on the development split."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scratch.contact_det.scripts.freeze_tree_contact_features import REGION_FIELDS
from scratch.contact_det_full_ds_fit.scripts.baseline_config import (
    BaselineConfig,
    BaselineRun,
    ModelKind,
    load_baseline_config,
)
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    DevelopmentSplit,
    VideoSpec,
)
from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
    VerifiedFeatureDataset,
    load_verified_feature_dataset,
)

RESULT_SCHEMA = "full-dataset-contact-baseline-result/1"
SCORE_FILE = "validation_contact_scores.npy.xz"
RESULT_FILE = "baseline_result.json"
SHOTS_MASTER_MD5 = "4c356bfb9809d08338b31e45d0b995b2"
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
RUN_ID = re.compile(r"^[a-z0-9_]+$")
REPORT_TOLERANCES_AT_30_FPS = (5, 10, 15)
SCORE_DTYPE = np.dtype(
    [
        ("fixture", "S7"),
        ("interval_id", "<i4"),
        ("frame", "<i4"),
        ("fps", "<f4"),
        ("contact_score", "<f8"),
        ("kept", "?"),
    ]
)
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ContactLabels:
    """Contact frames and first contacts, grouped by video."""

    frames: Mapping[str, np.ndarray]
    first_contacts: Mapping[str, frozenset[int]]
    rally_counts: Mapping[str, int]


@dataclass(frozen=True)
class CandidateRows:
    """Rows inside the label-blind search areas, grouped by video."""

    rows: np.ndarray
    video_ranges: Mapping[str, tuple[int, int]]


@dataclass(frozen=True)
class TrainingSelection:
    """The chosen training rows, their labels and saved counts."""

    selected: np.ndarray
    labels: np.ndarray
    video_counts: Mapping[str, Mapping[str, int]]


ModelFactory = Callable[[BaselineRun, int], Any]
LabelLoader = Callable[[Path, DevelopmentSplit], ContactLabels]


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


def _write_scores(path: Path, rows: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    with lzma.open(temporary, "wb") as target:
        np.save(target, rows, allow_pickle=False)
    os.replace(temporary, destination)


def _scaled_frames(value_at_30_fps: int, fps: float) -> int:
    from annotator.fps_constants import ScalingKind

    return int(ScalingKind.FRAME_COUNT.scale(value_at_30_fps, fps))


def _find_run(config: BaselineConfig, run_id: str) -> BaselineRun:
    matches = [run for run in config.runs if run.run_id == run_id]
    if len(matches) != 1:
        raise ValueError(f"unknown run ID: {run_id}")
    return matches[0]


def _seeded_rows(rows: np.ndarray) -> np.ndarray:
    selected = np.zeros(len(rows), dtype=bool)
    for field in REGION_FIELDS:
        selected |= rows[field].astype(bool)
    return selected


def collect_candidate_rows(features: VerifiedFeatureDataset) -> CandidateRows:
    """Keep rows selected by at least one label-blind search rule."""
    chunks: list[np.ndarray] = []
    video_ranges: dict[str, tuple[int, int]] = {}
    row_start = 0
    for video in features.split.videos:
        full_start, full_end = features.video_ranges[video.fixture]
        video_rows = features.rows[full_start:full_end]
        chosen_rows = video_rows[_seeded_rows(video_rows)]
        row_end = row_start + len(chosen_rows)
        video_ranges[video.fixture] = (row_start, row_end)
        chunks.append(chosen_rows)
        row_start = row_end
    return CandidateRows(np.concatenate(chunks), video_ranges)


def load_contact_labels(path: Path, split: DevelopmentSplit) -> ContactLabels:
    """Read the pinned ShuttleSet contact file after feature checks pass."""
    source_path = Path(path)
    if source_path.name != "shots_master.csv":
        raise ValueError("contact label file must be named shots_master.csv")
    actual_md5 = hashlib.md5(source_path.read_bytes()).hexdigest()
    if actual_md5 != SHOTS_MASTER_MD5:
        raise ValueError("shots_master.csv differs from the pinned contact labels")

    import pandas as pd

    from annotator.calibration.scoring import load_gt_rallies

    table = pd.read_csv(source_path, usecols=["vid", "set_id", "rally", "frame_num"])
    frames: dict[str, np.ndarray] = {}
    first_contacts: dict[str, frozenset[int]] = {}
    rally_counts: dict[str, int] = {}
    for video in split.videos:
        rallies = load_gt_rallies(table, video.video_id)
        frames[video.fixture] = np.asarray(
            [frame for rally in rallies for frame in rally.stroke_frames],
            dtype=np.int32,
        )
        first_contacts[video.fixture] = frozenset(rally.stroke_frames[0] for rally in rallies)
        rally_counts[video.fixture] = len(rallies)
    return ContactLabels(frames, first_contacts, rally_counts)


def _nearest_distances(frames: np.ndarray, contacts: np.ndarray) -> np.ndarray:
    if not len(contacts):
        return np.full(len(frames), np.iinfo(np.int32).max, dtype=np.int32)
    positions = np.searchsorted(contacts, frames)
    left_positions = np.maximum(positions - 1, 0)
    right_positions = np.minimum(positions, len(contacts) - 1)
    left_distances = np.abs(frames - contacts[left_positions])
    right_distances = np.abs(frames - contacts[right_positions])
    return np.minimum(left_distances, right_distances).astype(np.int32)


def choose_training_rows(
    candidates: CandidateRows,
    split: DevelopmentSplit,
    contact_labels: ContactLabels,
    config: BaselineConfig,
    run: BaselineRun,
) -> TrainingSelection:
    """Choose positive, nearby negative and sampled other training rows."""
    selected = np.zeros(len(candidates.rows), dtype=bool)
    labels = np.zeros(len(candidates.rows), dtype=np.uint8)
    video_counts: dict[str, dict[str, int]] = {}
    random = np.random.default_rng(config.random_seed)

    for video in split.training_videos:
        row_start, row_end = candidates.video_ranges[video.fixture]
        video_rows = candidates.rows[row_start:row_end]
        distances = _nearest_distances(video_rows["frame"], contact_labels.frames[video.fixture])
        positive = distances <= _scaled_frames(config.positive_radius_at_30_fps, video.fps)
        ignored = (~positive) & (
            distances <= _scaled_frames(config.ignored_radius_at_30_fps, video.fps)
        )
        negative = ~positive & ~ignored
        nearby_negative = negative & (
            distances <= _scaled_frames(config.hard_negative_radius_at_30_fps, video.fps)
        )

        positive_positions = np.flatnonzero(positive)
        nearby_positions = np.flatnonzero(nearby_negative)
        other_positions = np.flatnonzero(negative & ~nearby_negative)
        negative_limit = run.negative_limit * len(positive_positions)
        other_count = max(0, negative_limit - len(nearby_positions))
        if len(other_positions) > other_count:
            other_positions = random.choice(other_positions, size=other_count, replace=False)
        chosen_positions = np.concatenate(
            [positive_positions, nearby_positions, np.sort(other_positions)]
        )
        selected[row_start + chosen_positions] = True
        labels[row_start + positive_positions] = 1
        video_counts[video.fixture] = {
            "positive": len(positive_positions),
            "nearby_negative": len(nearby_positions),
            "sampled_other_negative": len(other_positions),
            "selected": len(chosen_positions),
        }

    if not np.any(labels[selected] == 1) or not np.any(labels[selected] == 0):
        raise ValueError("training rows must contain both contact and non-contact examples")
    return TrainingSelection(selected, labels, video_counts)


def _feature_matrix(rows: np.ndarray, names: Sequence[str]) -> np.ndarray:
    return np.column_stack([rows[name].astype(np.float32, copy=False) for name in names])


def make_model(run: BaselineRun, random_seed: int) -> Any:
    """Create the model named by one fixed run."""
    settings = dict(run.model_settings)
    if run.model_kind is ModelKind.HISTOGRAM_GRADIENT_BOOSTING:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            **settings,
            class_weight=run.class_weight_value,
            random_state=random_seed,
        )
    if run.model_kind is ModelKind.RANDOM_FOREST:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            **settings,
            class_weight=run.class_weight_value,
            n_jobs=-1,
            random_state=random_seed,
        )
    raise ValueError(f"unsupported model: {run.model_kind}")


def remove_nearby_contacts(
    frames: np.ndarray,
    interval_ids: np.ndarray,
    scores: np.ndarray,
    cutoff: float,
    distance: int,
) -> np.ndarray:
    """Keep the strongest contact within each nearby group and search interval."""
    kept: list[int] = []
    for interval_id in np.unique(interval_ids):
        possible = np.flatnonzero((interval_ids == interval_id) & (scores >= cutoff))
        strongest_first = sorted(possible, key=lambda index: (-scores[index], frames[index]))
        interval_kept: list[int] = []
        for index in strongest_first:
            if all(abs(int(frames[index]) - int(frames[other])) > distance for other in interval_kept):
                interval_kept.append(int(index))
        kept.extend(interval_kept)
    return np.asarray(sorted(kept, key=lambda index: frames[index]), dtype=np.int32)


def _match_contacts(
    contact_frames: np.ndarray,
    predicted_frames: np.ndarray,
    tolerance: int,
) -> list[tuple[int, int, int]]:
    possible: list[tuple[int, int, int, int, int, int]] = []
    for contact_index, contact_frame in enumerate(contact_frames):
        nearby = np.flatnonzero(np.abs(predicted_frames - contact_frame) <= tolerance)
        for prediction_index in nearby:
            offset = int(predicted_frames[prediction_index] - contact_frame)
            possible.append(
                (
                    abs(offset),
                    int(contact_frame),
                    int(predicted_frames[prediction_index]),
                    contact_index,
                    int(prediction_index),
                    offset,
                )
            )
    possible.sort()
    used_contacts: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[int, int, int]] = []
    for _, _, _, contact_index, prediction_index, offset in possible:
        if contact_index in used_contacts or prediction_index in used_predictions:
            continue
        used_contacts.add(contact_index)
        used_predictions.add(prediction_index)
        matches.append((contact_index, prediction_index, offset))
    return matches


def contact_counts(
    contact_labels: ContactLabels,
    predictions: Mapping[str, np.ndarray],
    videos: Sequence[VideoSpec],
    tolerance_at_30_fps: int,
) -> dict[str, int | float | None]:
    """Count one-to-one contact matches across the named videos."""
    matched = 0
    contact_total = 0
    prediction_total = 0
    first_contact_matched = 0
    first_contact_total = 0
    other_contact_matched = 0
    other_contact_total = 0
    absolute_offsets: list[int] = []
    for video in videos:
        expected = contact_labels.frames[video.fixture]
        predicted = predictions[video.fixture]
        tolerance = _scaled_frames(tolerance_at_30_fps, video.fps)
        matches = _match_contacts(expected, predicted, tolerance)
        matched += len(matches)
        contact_total += len(expected)
        prediction_total += len(predicted)
        first_contacts = contact_labels.first_contacts[video.fixture]
        first_contact_total += len(first_contacts)
        other_contact_total += len(expected) - len(first_contacts)
        for contact_index, _, offset in matches:
            absolute_offsets.append(abs(offset))
            if int(expected[contact_index]) in first_contacts:
                first_contact_matched += 1
            else:
                other_contact_matched += 1

    precision = matched / prediction_total if prediction_total else 0.0
    recall = matched / contact_total if contact_total else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "matched": matched,
        "contact_count": contact_total,
        "prediction_count": prediction_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "first_contact_matched": first_contact_matched,
        "first_contact_count": first_contact_total,
        "first_contact_recall": (
            first_contact_matched / first_contact_total if first_contact_total else 0.0
        ),
        "other_contact_matched": other_contact_matched,
        "other_contact_count": other_contact_total,
        "other_contact_recall": (
            other_contact_matched / other_contact_total if other_contact_total else 0.0
        ),
        "median_absolute_frame_error": (
            float(np.median(absolute_offsets)) if absolute_offsets else None
        ),
    }


def _validation_rows_and_scores(
    candidates: CandidateRows,
    split: DevelopmentSplit,
    model: Any,
    feature_names: Sequence[str],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for video in split.validation_videos:
        row_start, row_end = candidates.video_ranges[video.fixture]
        video_rows = candidates.rows[row_start:row_end]
        scores = model.predict_proba(_feature_matrix(video_rows, feature_names))[:, 1]
        output = np.empty(len(video_rows), dtype=SCORE_DTYPE)
        output["fixture"] = video_rows["fixture"]
        output["interval_id"] = video_rows["interval_id"]
        output["frame"] = video_rows["frame"]
        output["fps"] = video_rows["fps"]
        output["contact_score"] = scores
        output["kept"] = False
        chunks.append(output)
    return np.concatenate(chunks)


def predictions_for_settings(
    scores: np.ndarray,
    videos: Sequence[VideoSpec],
    cutoff: float,
    distance_at_30_fps: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Apply one cutoff and nearby-contact distance to validation scores."""
    kept = np.zeros(len(scores), dtype=bool)
    predictions: dict[str, np.ndarray] = {}
    names = np.char.decode(scores["fixture"], "ascii")
    for video in videos:
        positions = np.flatnonzero(names == video.fixture)
        video_rows = scores[positions]
        chosen = remove_nearby_contacts(
            video_rows["frame"],
            video_rows["interval_id"],
            video_rows["contact_score"],
            cutoff,
            _scaled_frames(distance_at_30_fps, video.fps),
        )
        kept[positions[chosen]] = True
        predictions[video.fixture] = video_rows["frame"][chosen].astype(np.int32)
    return predictions, kept


def choose_validation_settings(
    scores: np.ndarray,
    split: DevelopmentSplit,
    contact_labels: ContactLabels,
    config: BaselineConfig,
) -> tuple[float, int, dict[str, int | float | None], dict[str, np.ndarray], np.ndarray]:
    """Choose the fixed pair with the best timing result on validation videos."""
    best: tuple[
        tuple[float, float, float, int, float],
        float,
        int,
        dict[str, int | float | None],
        dict[str, np.ndarray],
        np.ndarray,
    ] | None = None
    for cutoff in config.score_cutoffs:
        for distance in config.duplicate_distances_at_30_fps:
            predictions, kept = predictions_for_settings(
                scores,
                split.validation_videos,
                cutoff,
                distance,
            )
            counts = contact_counts(
                contact_labels,
                predictions,
                split.validation_videos,
                config.timing_tolerance_at_30_fps,
            )
            key = (
                float(counts["f1"]),
                float(counts["recall"]),
                float(counts["precision"]),
                distance,
                cutoff,
            )
            if best is None or key > best[0]:
                best = (key, cutoff, distance, counts, predictions, kept)
    if best is None:
        raise ValueError("run menu has no validation settings")
    _, cutoff, distance, counts, predictions, kept = best
    return cutoff, distance, counts, predictions, kept


def _metrics_at_report_tolerances(
    labels: ContactLabels,
    predictions: Mapping[str, np.ndarray],
    videos: Sequence[VideoSpec],
) -> dict[str, dict[str, int | float | None]]:
    return {
        str(tolerance): contact_counts(labels, predictions, videos, tolerance)
        for tolerance in REPORT_TOLERANCES_AT_30_FPS
    }


def _per_video_metrics(
    labels: ContactLabels,
    predictions: Mapping[str, np.ndarray],
    videos: Sequence[VideoSpec],
) -> dict[str, object]:
    return {
        video.fixture: {
            "rally_count": labels.rally_counts[video.fixture],
            "prediction_frames": predictions[video.fixture].tolist(),
            "metrics": _metrics_at_report_tolerances(labels, predictions, [video]),
        }
        for video in videos
    }


def run_baseline(
    config_path: Path,
    run_id: str,
    feature_record_path: Path,
    split_path: Path,
    shots_master_path: Path,
    output_root: Path,
    source_commit: str,
    *,
    label_loader: LabelLoader = load_contact_labels,
    model_factory: ModelFactory = make_model,
) -> Path:
    """Run one named model and save its checked validation result."""
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a short or full Git commit")
    if RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run ID must contain only lower-case letters, numbers and underscores")
    source_root = str(REPO_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    config_file = Path(config_path)
    split_file = Path(split_path)
    feature_record_file = Path(feature_record_path)
    output_dir = Path(output_root) / run_id
    result_path = output_dir / RESULT_FILE
    _write_json(
        result_path,
        {
            "schema": RESULT_SCHEMA,
            "status": "running",
            "run_id": run_id,
            "source_commit": source_commit,
        },
    )
    config = load_baseline_config(config_file)
    run = _find_run(config, run_id)
    features = load_verified_feature_dataset(
        feature_record_file,
        split_file,
        run.motion_mode.value,
    )
    candidates = collect_candidate_rows(features)

    running_result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "running",
        "run_id": run.run_id,
        "source_commit": source_commit,
        "config_file": config_file.name,
        "config_sha256": _sha256(config_file),
        "feature_record_file": feature_record_file.name,
        "feature_record_sha256": _sha256(feature_record_file),
        "feature_source_commit": features.record["source_commit"],
        "split_file": split_file.name,
        "split_sha256": _sha256(split_file),
    }
    _write_json(result_path, running_result)

    contact_labels = label_loader(Path(shots_master_path), features.split)
    training = choose_training_rows(candidates, features.split, contact_labels, config, run)
    model = model_factory(run, config.random_seed)
    model.fit(
        _feature_matrix(candidates.rows[training.selected], features.model_input_fields),
        training.labels[training.selected],
    )
    scores = _validation_rows_and_scores(
        candidates,
        features.split,
        model,
        features.model_input_fields,
    )
    cutoff, distance, selection_counts, predictions, kept = choose_validation_settings(
        scores,
        features.split,
        contact_labels,
        config,
    )
    scores["kept"] = kept

    score_path = output_dir / SCORE_FILE
    _write_scores(score_path, scores)
    complete_result: dict[str, object] = {
        **running_result,
        "status": "complete",
        "contact_label_file": Path(shots_master_path).name,
        "contact_label_sha256": _sha256(Path(shots_master_path)),
        "training_videos": [video.fixture for video in features.split.training_videos],
        "validation_videos": [video.fixture for video in features.split.validation_videos],
        "feature_row_count": len(features.rows),
        "candidate_row_count": len(candidates.rows),
        "feature_names": list(features.model_input_fields),
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
        "selection_metrics": selection_counts,
        "metrics": _metrics_at_report_tolerances(
            contact_labels,
            predictions,
            features.split.validation_videos,
        ),
        "videos": _per_video_metrics(
            contact_labels,
            predictions,
            features.split.validation_videos,
        ),
        "validation_score_file": score_path.name,
        "validation_score_sha256": _sha256(score_path),
        "validation_score_row_count": len(scores),
    }
    _write_json(result_path, complete_result)
    return result_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--feature-record", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--shots-master", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    result_path = run_baseline(
        arguments.config,
        arguments.run_id,
        arguments.feature_record,
        arguments.split,
        arguments.shots_master,
        arguments.output_dir,
        arguments.source_commit,
    )
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
