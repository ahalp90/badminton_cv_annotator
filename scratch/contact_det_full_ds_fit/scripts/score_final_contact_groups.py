"""Score five held-out video groups and choose the final contact settings."""

from __future__ import annotations

import argparse
import csv
import json
import lzma
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
    load_baseline_config,
)
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    DevelopmentSplit,
    VideoSpec,
)
from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
    VerifiedFeatureDataset,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    SCORE_DTYPE,
    CandidateRows,
    ContactLabels,
    _feature_matrix,
    _find_run,
    _sha256,
    _write_json,
    _write_scores,
    collect_candidate_rows,
    contact_counts,
    make_model,
    predictions_for_settings,
)
from scratch.contact_det_full_ds_fit.scripts.score_training_videos import (
    CHOSEN_RUN_ID,
    DUPLICATE_DISTANCE_AT_30_FPS,
    SCORE_CUTOFF,
    _check_fixed_model,
    choose_training_rows_for_videos,
)

GROUP_SCHEMA = "final-contact-video-score-groups/1"
INPUT_SCHEMA = "final-contact-score-inputs/1"
GROUP_RESULT_SCHEMA = "final-contact-group-score/1"
COMBINED_RESULT_SCHEMA = "final-contact-setting-result/1"
GROUP_NAMES = ("A", "B", "C", "D", "V")
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
EXPECTED_VIDEO_COUNT = 40
EXPECTED_TRAINING_VIDEO_COUNT = 32
EXPECTED_SCORED_VIDEO_COUNT = 8
EXPECTED_SCORE_ROW_COUNT = 1_477_290
GROUP_SCORE_FILE = "raw_contact_scores.npy.xz"
GROUP_RESULT_FILE = "group_score_result.json"
COMBINED_RAW_SCORE_FILE = "combined_raw_contact_scores.npy.xz"
FINAL_SCORE_FILE = "final_kept_contact_scores.npy.xz"
COMBINED_RESULT_FILE = "final_contact_setting_result.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_ROW_CONDITION = "At least one listed unsigned-byte field equals 1"
RAW_SCORE_DTYPE = np.dtype(
    [
        ("fixture", "S8"),
        ("interval_id", "<i4"),
        ("frame", "<i4"),
        ("fps", "<f8"),
        ("contact_score", "<f8"),
        ("source_group", "S1"),
    ]
)
FINAL_SCORE_DTYPE = np.dtype(RAW_SCORE_DTYPE.descr + [("kept", "?")])


@dataclass(frozen=True)
class InputFiles:
    """Fixed files checked before final held-out scoring."""

    input_list: Path
    groups: Path
    split: Path
    config: Path
    feature_record: Path
    baseline_summary: Path
    chosen_baseline_result: Path
    contact_labels: Path
    chosen_validation_scores: Path


@dataclass(frozen=True)
class FinalScoreGroup:
    """Eight scored videos and the other 32 training videos."""

    name: str
    scored_videos: tuple[VideoSpec, ...]
    training_videos: tuple[VideoSpec, ...]


LabelLoader = Callable[[Path, Sequence[VideoSpec]], ContactLabels]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be an object with string keys")
    return value


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    return _mapping(json.loads(Path(path).read_text(encoding="utf-8")), label)


def _files_by_role(files: InputFiles) -> dict[str, Path]:
    return {
        "final score groups": files.groups,
        "development split": files.split,
        "baseline settings": files.config,
        "raw feature record": files.feature_record,
        "baseline summary": files.baseline_summary,
        "chosen baseline result": files.chosen_baseline_result,
        "contact labels": files.contact_labels,
        "chosen validation scores": files.chosen_validation_scores,
    }


def check_input_files(files: InputFiles) -> Mapping[str, Any]:
    """Check every fixed filename and hash before loading experiment data."""
    payload = _load_json(files.input_list, "final contact input list")
    expected_fields = {
        "schema",
        "expected_video_count",
        "expected_candidate_score_row_count",
        "candidate_row_condition",
        "candidate_row_fields",
        "expected_score_rows_by_video",
        "files",
    }
    if set(payload) != expected_fields or payload.get("schema") != INPUT_SCHEMA:
        raise ValueError("final contact input fields differ")
    raw_records = payload.get("files")
    if not isinstance(raw_records, list):
        raise TypeError("final contact input files must be a list")
    records: dict[str, Mapping[str, Any]] = {}
    for index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record, f"input file {index}")
        if set(record) != {"role", "filename", "sha256"}:
            raise ValueError(f"input file {index} fields differ")
        role = record.get("role")
        if not isinstance(role, str) or role in records:
            raise ValueError("input file roles differ")
        records[role] = record
    expected_paths = _files_by_role(files)
    if set(records) != set(expected_paths):
        raise ValueError("input file roles differ")
    for role, path in expected_paths.items():
        record = records[role]
        if record.get("filename") != Path(path).name:
            raise ValueError(f"{role} filename differs")
        if record.get("sha256") != _sha256(path):
            raise ValueError(f"{role} file hash differs")
    return payload


def _expected_input_hash(inputs: Mapping[str, Any], role: str) -> str:
    raw_records = inputs.get("files")
    if not isinstance(raw_records, list):
        raise TypeError("final contact input files must be a list")
    matches = [
        _mapping(record, f"{role} input record")
        for record in raw_records
        if isinstance(record, Mapping) and record.get("role") == role
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise ValueError(f"{role} input record differs")
    return str(matches[0]["sha256"])


def load_final_score_groups(
    path: Path,
    split: DevelopmentSplit,
) -> dict[str, FinalScoreGroup]:
    """Check the fixed A, B, C, D and V groups against the 40-video split."""
    payload = _load_json(path, "final score groups")
    if set(payload) != {"schema", "groups"} or payload.get("schema") != GROUP_SCHEMA:
        raise ValueError("final score group fields differ")
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or len(raw_groups) != len(GROUP_NAMES):
        raise ValueError("there must be five final score groups")
    by_fixture = split.by_fixture
    scored_by_group: dict[str, tuple[VideoSpec, ...]] = {}
    all_names: list[str] = []
    for expected_name, raw_group in zip(GROUP_NAMES, raw_groups, strict=True):
        group = _mapping(raw_group, f"group {expected_name}")
        if set(group) != {"group", "videos"} or group.get("group") != expected_name:
            raise ValueError(f"group {expected_name} fields differ")
        raw_names = group.get("videos")
        if not isinstance(raw_names, list) or any(
            not isinstance(name, str) for name in raw_names
        ):
            raise TypeError(f"group {expected_name} videos must be names")
        if len(raw_names) != EXPECTED_SCORED_VIDEO_COUNT or len(set(raw_names)) != len(
            raw_names
        ):
            raise ValueError(f"group {expected_name} must have eight distinct videos")
        if any(name not in by_fixture for name in raw_names):
            raise ValueError(f"group {expected_name} has an unknown video")
        scored_by_group[expected_name] = tuple(by_fixture[name] for name in raw_names)
        all_names.extend(raw_names)
    expected_names = {video.fixture for video in split.videos}
    if len(all_names) != EXPECTED_VIDEO_COUNT or set(all_names) != expected_names:
        raise ValueError("final groups must cover each development video once")

    result: dict[str, FinalScoreGroup] = {}
    for group_name in GROUP_NAMES:
        scored_names = {video.fixture for video in scored_by_group[group_name]}
        training = tuple(
            video for video in split.videos if video.fixture not in scored_names
        )
        if len(training) != EXPECTED_TRAINING_VIDEO_COUNT:
            raise ValueError(f"group {group_name} training video count differs")
        result[group_name] = FinalScoreGroup(
            group_name,
            scored_by_group[group_name],
            training,
        )
    return result


def _check_candidate_rows(
    features: VerifiedFeatureDataset,
    candidates: CandidateRows,
    inputs: Mapping[str, Any],
) -> None:
    if inputs.get("expected_video_count") != EXPECTED_VIDEO_COUNT:
        raise ValueError("expected development video count differs")
    if inputs.get("candidate_row_condition") != CANDIDATE_ROW_CONDITION:
        raise ValueError("candidate row condition differs")
    if inputs.get("candidate_row_fields") != list(REGION_FIELDS):
        raise ValueError("candidate row fields differ")
    for field in REGION_FIELDS:
        if features.rows.dtype[field] != np.dtype("u1"):
            raise ValueError(f"{field} must use unsigned-byte values")
        if not np.isin(features.rows[field], (0, 1)).all():
            raise ValueError(f"{field} must contain only zero and one")
    expected_counts = _mapping(
        inputs.get("expected_score_rows_by_video"),
        "expected score rows by video",
    )
    actual_counts = {
        video.fixture: candidates.video_ranges[video.fixture][1]
        - candidates.video_ranges[video.fixture][0]
        for video in features.split.videos
    }
    if dict(expected_counts) != actual_counts:
        raise ValueError("candidate score row counts differ")
    if (
        inputs.get("expected_candidate_score_row_count") != EXPECTED_SCORE_ROW_COUNT
        or len(candidates.rows) != EXPECTED_SCORE_ROW_COUNT
    ):
        raise ValueError("candidate score row total differs")


def _integer(value: str, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error


def load_contact_labels_for_videos(
    path: Path,
    videos: Sequence[VideoSpec],
) -> ContactLabels:
    """Read contact timing for exactly the named video IDs."""
    allowed_ids = {video.video_id for video in videos}
    if len(allowed_ids) != len(videos):
        raise ValueError("allowed label video identities repeat")
    rallies: dict[tuple[int, str, int], list[int]] = {}
    seen_frames: set[tuple[int, int]] = set()
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("contact label file is empty") from error
        required = ("vid", "set_id", "rally", "frame_num")
        if len(set(header)) != len(header) or any(field not in header for field in required):
            raise ValueError("contact label columns differ")
        positions = {field: header.index(field) for field in required}
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"contact label row {row_number} has the wrong length")
            video_id = _integer(
                row[positions["vid"]], f"contact label row {row_number} video ID"
            )
            if video_id not in allowed_ids:
                continue
            set_id = row[positions["set_id"]]
            rally = _integer(
                row[positions["rally"]], f"contact label row {row_number} rally"
            )
            frame = _integer(
                row[positions["frame_num"]], f"contact label row {row_number} frame"
            )
            if not set_id:
                raise ValueError(f"contact label row {row_number} set ID is empty")
            identity = (video_id, frame)
            if identity in seen_frames:
                raise ValueError("contact label video-frame identities repeat")
            seen_frames.add(identity)
            rallies.setdefault((video_id, set_id, rally), []).append(frame)
    if {video_id for video_id, _set_id, _rally in rallies} != allowed_ids:
        raise ValueError("contact label video coverage differs")

    frames: dict[str, np.ndarray] = {}
    first_contacts: dict[str, frozenset[int]] = {}
    rally_counts: dict[str, int] = {}
    for video in videos:
        video_rallies = [
            tuple(sorted(rally_frames))
            for (video_id, _set_id, _rally), rally_frames in sorted(rallies.items())
            if video_id == video.video_id
        ]
        if any(not rally or len(rally) != len(set(rally)) for rally in video_rallies):
            raise ValueError(f"{video.fixture}: contact label rally frames differ")
        frames[video.fixture] = np.asarray(
            sorted(frame for rally in video_rallies for frame in rally),
            dtype=np.int32,
        )
        first_contacts[video.fixture] = frozenset(rally[0] for rally in video_rallies)
        rally_counts[video.fixture] = len(video_rallies)
    return ContactLabels(frames, first_contacts, rally_counts)


def _load_fixed_inputs(
    files: InputFiles,
) -> tuple[Mapping[str, Any], BaselineConfig, BaselineRun]:
    inputs = check_input_files(files)
    config = load_baseline_config(files.config)
    run = _find_run(config, CHOSEN_RUN_ID)
    summary = _load_json(files.baseline_summary, "baseline summary")
    baseline_result = _load_json(files.chosen_baseline_result, "chosen baseline result")
    _check_fixed_model(config, run, summary, baseline_result)
    return inputs, config, run


def _score_group_rows(
    candidates: CandidateRows,
    group: FinalScoreGroup,
    model: Any,
    feature_names: Sequence[str],
) -> np.ndarray:
    if not np.array_equal(np.asarray(model.classes_), np.asarray([0, 1])):
        raise ValueError("contact model classes differ")
    chunks: list[np.ndarray] = []
    for video in group.scored_videos:
        row_start, row_end = candidates.video_ranges[video.fixture]
        video_rows = candidates.rows[row_start:row_end]
        probabilities = model.predict_proba(
            _feature_matrix(video_rows, feature_names)
        )[:, 1]
        output = np.empty(len(video_rows), dtype=RAW_SCORE_DTYPE)
        for field in ("fixture", "interval_id", "frame", "fps"):
            output[field] = video_rows[field]
        output["contact_score"] = probabilities
        output["source_group"] = group.name.encode("ascii")
        chunks.append(output)
    return np.concatenate(chunks)


def _read_old_validation_scores(path: Path) -> np.ndarray:
    with lzma.open(path, "rb") as source:
        scores = np.load(source, allow_pickle=False)
    if scores.dtype != SCORE_DTYPE:
        raise ValueError("chosen validation score fields differ")
    return scores


def _check_group_v_scores(
    scores: np.ndarray,
    old_scores: np.ndarray,
    group: FinalScoreGroup,
) -> None:
    shared_fields = ("fixture", "interval_id", "frame", "fps", "contact_score")
    if len(scores) != len(old_scores) or any(
        not np.array_equal(scores[field], old_scores[field]) for field in shared_fields
    ):
        raise ValueError("group V scores differ from the chosen validation scores")
    _predictions, expected_kept = predictions_for_settings(
        old_scores,
        group.scored_videos,
        SCORE_CUTOFF,
        DUPLICATE_DISTANCE_AT_30_FPS,
    )
    if not np.array_equal(old_scores["kept"], expected_kept):
        raise ValueError("chosen validation kept flags differ")


def _input_records(files: InputFiles) -> dict[str, dict[str, str]]:
    return {
        role: {"filename": path.name, "sha256": _sha256(path)}
        for role, path in _files_by_role(files).items()
    }


def score_group(
    files: InputFiles,
    group_name: str,
    output_dir: Path,
    source_commit: str,
    *,
    model_factory: Callable[[BaselineRun, int], Any] = make_model,
    label_loader: LabelLoader = load_contact_labels_for_videos,
    feature_loader: Callable[..., VerifiedFeatureDataset] | None = None,
) -> Path:
    """Train on 32 videos and save raw scores for the held-out eight."""
    destination = Path(output_dir)
    result_path = destination / GROUP_RESULT_FILE
    _write_json(
        result_path,
        {
            "schema": GROUP_RESULT_SCHEMA,
            "status": "running",
            "group": group_name,
            "source_commit": source_commit,
            "training_labels_read": False,
        },
    )
    training_labels_read = False
    try:
        if group_name not in GROUP_NAMES:
            raise ValueError("group must be A, B, C, D or V")
        if SOURCE_COMMIT.fullmatch(source_commit) is None:
            raise ValueError("source commit must be a short or full Git commit")
        source_root = str(REPO_ROOT / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        inputs, config, run = _load_fixed_inputs(files)
        if feature_loader is None:
            from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
                load_verified_feature_dataset,
            )

            feature_loader = load_verified_feature_dataset
        features = feature_loader(files.feature_record, files.split, run.motion_mode.value)
        groups = load_final_score_groups(files.groups, features.split)
        group = groups[group_name]
        candidates = collect_candidate_rows(features)
        _check_candidate_rows(features, candidates, inputs)

        training_labels_read = True
        training_labels = label_loader(files.contact_labels, group.training_videos)
        expected_label_hash = _expected_input_hash(inputs, "contact labels")
        if _sha256(files.contact_labels) != expected_label_hash:
            raise ValueError("contact label file changed while training labels were read")
        training = choose_training_rows_for_videos(
            candidates,
            group.training_videos,
            training_labels,
            config,
            run,
            expected_video_count=EXPECTED_TRAINING_VIDEO_COUNT,
        )
        selected_names = set(
            np.char.decode(candidates.rows[training.selected]["fixture"], "ascii")
        )
        expected_training_names = {video.fixture for video in group.training_videos}
        scored_names = {video.fixture for video in group.scored_videos}
        if selected_names != expected_training_names or selected_names & scored_names:
            raise ValueError("training examples contain the wrong videos")

        model = model_factory(run, config.random_seed)
        model.fit(
            _feature_matrix(
                candidates.rows[training.selected], features.model_input_fields
            ),
            training.labels[training.selected],
        )
        scores = _score_group_rows(
            candidates,
            group,
            model,
            features.model_input_fields,
        )
        expected_rows = sum(
            int(
                _mapping(
                    inputs["expected_score_rows_by_video"],
                    "expected score rows by video",
                )[video.fixture]
            )
            for video in group.scored_videos
        )
        if len(scores) != expected_rows:
            raise ValueError("group score row count differs")
        if not np.isfinite(scores["contact_score"]).all() or np.any(
            (scores["contact_score"] < 0.0) | (scores["contact_score"] > 1.0)
        ):
            raise ValueError("group contact scores differ")
        if group_name == "V":
            _check_group_v_scores(
                scores,
                _read_old_validation_scores(files.chosen_validation_scores),
                group,
            )
        score_path = destination / GROUP_SCORE_FILE
        _write_scores(score_path, scores)
        result: dict[str, object] = {
            "schema": GROUP_RESULT_SCHEMA,
            "status": "complete",
            "group": group.name,
            "source_commit": source_commit,
            "training_labels_read": True,
            "input_list_file": files.input_list.name,
            "input_list_sha256": _sha256(files.input_list),
            "input_files": _input_records(files),
            "run_id": run.run_id,
            "training_videos": [video.fixture for video in group.training_videos],
            "scored_videos": [video.fixture for video in group.scored_videos],
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
            "raw_score_file": score_path.name,
            "raw_score_sha256": _sha256(score_path),
            "raw_score_row_count": len(scores),
            "group_v_matches_chosen_validation_scores": group_name == "V",
        }
        _write_json(result_path, result)
        return result_path
    except Exception as error:
        _write_json(
            result_path,
            {
                "schema": GROUP_RESULT_SCHEMA,
                "status": "failed",
                "group": group_name,
                "source_commit": source_commit,
                "training_labels_read": training_labels_read,
                "error_type": type(error).__name__,
            },
        )
        raise


def _read_raw_scores(path: Path) -> np.ndarray:
    with lzma.open(path, "rb") as source:
        scores = np.load(source, allow_pickle=False)
    if scores.dtype != RAW_SCORE_DTYPE:
        raise ValueError("raw contact score fields differ")
    return scores


def _expected_group_rows(
    candidates: CandidateRows,
    group: FinalScoreGroup,
) -> np.ndarray:
    return np.concatenate(
        [
            candidates.rows[
                candidates.video_ranges[video.fixture][0] : candidates.video_ranges[
                    video.fixture
                ][1]
            ]
            for video in group.scored_videos
        ]
    )


def _check_group_result(
    result: Mapping[str, Any],
    group: FinalScoreGroup,
    files: InputFiles,
    config: BaselineConfig,
    run: BaselineRun,
    source_commit: str,
) -> None:
    expected_model = {
        "name": run.model_name,
        "kind": run.model_kind.value,
        "settings": dict(run.model_settings),
        "class_weight": run.class_weight,
        "class_weight_value": run.class_weight_value,
    }
    expected = {
        "schema": GROUP_RESULT_SCHEMA,
        "status": "complete",
        "group": group.name,
        "source_commit": source_commit,
        "training_labels_read": True,
        "input_list_file": files.input_list.name,
        "input_list_sha256": _sha256(files.input_list),
        "input_files": _input_records(files),
        "run_id": run.run_id,
        "training_videos": [video.fixture for video in group.training_videos],
        "scored_videos": [video.fixture for video in group.scored_videos],
        "model": expected_model,
        "group_v_matches_chosen_validation_scores": group.name == "V",
    }
    if any(result.get(field) != value for field, value in expected.items()):
        raise ValueError(f"group {group.name} result differs")
    selection = _mapping(result.get("training_selection"), "training selection")
    expected_settings = {
        "positive_radius_at_30_fps": config.positive_radius_at_30_fps,
        "ignored_radius_at_30_fps": config.ignored_radius_at_30_fps,
        "nearby_negative_radius_at_30_fps": config.hard_negative_radius_at_30_fps,
        "negative_rule": run.negative_rule,
        "maximum_negatives_per_positive": run.negative_limit,
    }
    if any(selection.get(field) != value for field, value in expected_settings.items()):
        raise ValueError(f"group {group.name} training settings differ")
    video_counts = _mapping(selection.get("videos"), "training video counts")
    if set(video_counts) != {video.fixture for video in group.training_videos}:
        raise ValueError(f"group {group.name} training count videos differ")
    selected_total = 0
    positive_total = 0
    expected_count_fields = {
        "positive",
        "nearby_negative",
        "sampled_other_negative",
        "selected",
    }
    for video_name, raw_counts in video_counts.items():
        counts = _mapping(raw_counts, f"{video_name} training counts")
        if set(counts) != expected_count_fields or any(
            type(count) is not int for count in counts.values()
        ):
            raise ValueError(f"group {group.name} training counts differ")
        if any(count < 0 for count in counts.values()):
            raise ValueError(f"group {group.name} training counts must be non-negative")
        if counts["selected"] != (
            counts["positive"]
            + counts["nearby_negative"]
            + counts["sampled_other_negative"]
        ):
            raise ValueError(f"group {group.name} training counts do not add up")
        selected_total += counts["selected"]
        positive_total += counts["positive"]
    if selection.get("selected_row_count") != selected_total:
        raise ValueError(f"group {group.name} selected training count differs")
    if selection.get("positive_row_count") != positive_total:
        raise ValueError(f"group {group.name} positive training count differs")


def choose_final_settings(
    scores: np.ndarray,
    videos: Sequence[VideoSpec],
    labels: ContactLabels,
    config: BaselineConfig,
) -> tuple[
    list[dict[str, object]],
    float,
    int,
    dict[str, int | float | None],
    dict[str, np.ndarray],
    np.ndarray,
]:
    """Score the fixed 57 pairs and choose with the written tie order."""
    results: list[dict[str, object]] = []
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
                videos,
                cutoff,
                distance,
            )
            metrics = contact_counts(
                labels,
                predictions,
                videos,
                config.timing_tolerance_at_30_fps,
            )
            results.append(
                {
                    "score_cutoff": cutoff,
                    "duplicate_distance_at_30_fps": distance,
                    "metrics": metrics,
                }
            )
            key = (
                float(metrics["f1"]),
                float(metrics["recall"]),
                float(metrics["precision"]),
                distance,
                cutoff,
            )
            if best is None or key > best[0]:
                best = (key, cutoff, distance, metrics, predictions, kept)
    if best is None:
        raise ValueError("final setting menu is empty")
    _key, cutoff, distance, metrics, predictions, kept = best
    return results, cutoff, distance, metrics, predictions, kept


def _metrics_at_tolerances(
    labels: ContactLabels,
    predictions: Mapping[str, np.ndarray],
    videos: Sequence[VideoSpec],
) -> dict[str, dict[str, int | float | None]]:
    return {
        str(tolerance): contact_counts(labels, predictions, videos, tolerance)
        for tolerance in (1, 2, 5, 10)
    }


def _per_video_metrics(
    labels: ContactLabels,
    predictions: Mapping[str, np.ndarray],
    videos: Sequence[VideoSpec],
) -> list[dict[str, object]]:
    return [
        {
            "fixture": video.fixture,
            "metrics_at_5_frames": contact_counts(
                labels,
                predictions,
                [video],
                5,
            ),
        }
        for video in videos
    ]


def _final_scores(scores: np.ndarray, kept: np.ndarray) -> np.ndarray:
    if kept.dtype != np.dtype("?") or len(kept) != len(scores):
        raise ValueError("final kept-contact flags differ")
    output = np.empty(len(scores), dtype=FINAL_SCORE_DTYPE)
    for field in RAW_SCORE_DTYPE.names or ():
        output[field] = scores[field]
    output["kept"] = kept
    return output


def combine_groups(
    files: InputFiles,
    group_directories: Mapping[str, Path],
    output_dir: Path,
    source_commit: str,
    *,
    label_loader: LabelLoader = load_contact_labels_for_videos,
    feature_loader: Callable[..., VerifiedFeatureDataset] | None = None,
) -> Path:
    """Combine five raw score files, then choose and save the final pair."""
    destination = Path(output_dir)
    result_path = destination / COMBINED_RESULT_FILE
    _write_json(
        result_path,
        {
            "schema": COMBINED_RESULT_SCHEMA,
            "status": "running",
            "source_commit": source_commit,
            "labels_read": False,
        },
    )
    labels_read = False
    try:
        if set(group_directories) != set(GROUP_NAMES):
            raise ValueError("one directory is required for each final score group")
        if SOURCE_COMMIT.fullmatch(source_commit) is None:
            raise ValueError("source commit must be a short or full Git commit")
        inputs, config, run = _load_fixed_inputs(files)
        if feature_loader is None:
            from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
                load_verified_feature_dataset,
            )

            feature_loader = load_verified_feature_dataset
        features = feature_loader(files.feature_record, files.split, run.motion_mode.value)
        groups = load_final_score_groups(files.groups, features.split)
        candidates = collect_candidate_rows(features)
        _check_candidate_rows(features, candidates, inputs)

        chunks: list[np.ndarray] = []
        child_files: dict[str, object] = {}
        seen_identities: set[tuple[bytes, int, int, float]] = set()
        for group_name in GROUP_NAMES:
            group = groups[group_name]
            group_dir = Path(group_directories[group_name])
            child_result_path = group_dir / GROUP_RESULT_FILE
            child_result = _load_json(child_result_path, f"group {group_name} result")
            _check_group_result(
                child_result,
                group,
                files,
                config,
                run,
                source_commit,
            )
            score_path = group_dir / GROUP_SCORE_FILE
            if (
                child_result.get("raw_score_file") != score_path.name
                or child_result.get("raw_score_sha256") != _sha256(score_path)
            ):
                raise ValueError(f"group {group_name} score file differs")
            scores = _read_raw_scores(score_path)
            if child_result.get("raw_score_row_count") != len(scores):
                raise ValueError(f"group {group_name} score count differs")
            expected_rows = _expected_group_rows(candidates, group)
            identity_fields = ("fixture", "interval_id", "frame", "fps")
            if len(scores) != len(expected_rows) or any(
                not np.array_equal(scores[field], expected_rows[field])
                for field in identity_fields
            ):
                raise ValueError(f"group {group_name} score order differs")
            if not np.all(scores["source_group"] == group_name.encode("ascii")):
                raise ValueError(f"group {group_name} source fields differ")
            identities = {
                (
                    bytes(row["fixture"]),
                    int(row["interval_id"]),
                    int(row["frame"]),
                    float(row["fps"]),
                )
                for row in scores
            }
            if len(identities) != len(scores) or identities & seen_identities:
                raise ValueError(f"group {group_name} score identities repeat")
            seen_identities.update(identities)
            chunks.append(scores)
            child_files[group_name] = {
                "result_file": child_result_path.name,
                "result_sha256": _sha256(child_result_path),
                "raw_score_file": score_path.name,
                "raw_score_sha256": _sha256(score_path),
                "raw_score_row_count": len(scores),
            }

        combined = np.concatenate(chunks)
        if len(combined) != EXPECTED_SCORE_ROW_COUNT:
            raise ValueError("combined raw score count differs")
        raw_score_path = destination / COMBINED_RAW_SCORE_FILE
        _write_scores(raw_score_path, combined)
        saved_combined = _read_raw_scores(raw_score_path)
        if not np.array_equal(saved_combined, combined):
            raise ValueError("saved combined raw contact scores differ")
        _write_json(
            result_path,
            {
                "schema": COMBINED_RESULT_SCHEMA,
                "status": "running",
                "source_commit": source_commit,
                "labels_read": False,
                "combined_raw_score_file": raw_score_path.name,
                "combined_raw_score_sha256": _sha256(raw_score_path),
                "combined_raw_score_row_count": len(combined),
            },
        )

        labels_read = True
        all_labels = label_loader(files.contact_labels, features.split.videos)
        expected_label_hash = _expected_input_hash(inputs, "contact labels")
        if _sha256(files.contact_labels) != expected_label_hash:
            raise ValueError("contact label file changed while final settings were chosen")
        setting_results, cutoff, distance, chosen_metrics, predictions, kept = (
            choose_final_settings(
                combined,
                features.split.videos,
                all_labels,
                config,
            )
        )
        final_scores = _final_scores(combined, kept)
        final_score_path = destination / FINAL_SCORE_FILE
        _write_scores(final_score_path, final_scores)
        saved_final_scores = _read_final_scores(final_score_path)
        if not np.array_equal(saved_final_scores, final_scores):
            raise ValueError("saved final contact scores differ")

        old_predictions, _old_kept = predictions_for_settings(
            combined,
            features.split.videos,
            SCORE_CUTOFF,
            DUPLICATE_DISTANCE_AT_30_FPS,
        )
        result: dict[str, object] = {
            "schema": COMBINED_RESULT_SCHEMA,
            "status": "complete",
            "source_commit": source_commit,
            "labels_read": True,
            "input_list_file": files.input_list.name,
            "input_list_sha256": _sha256(files.input_list),
            "input_files": _input_records(files),
            "groups": list(GROUP_NAMES),
            "videos": [
                video.fixture
                for group_name in GROUP_NAMES
                for video in groups[group_name].scored_videos
            ],
            "group_files": child_files,
            "combined_raw_score_file": raw_score_path.name,
            "combined_raw_score_sha256": _sha256(raw_score_path),
            "combined_raw_score_row_count": len(combined),
            "setting_results": setting_results,
            "selected_score_cutoff": cutoff,
            "selected_duplicate_distance_at_30_fps": distance,
            "selected_metrics_at_5_frames": chosen_metrics,
            "selected_metrics_by_tolerance": _metrics_at_tolerances(
                all_labels,
                predictions,
                features.split.videos,
            ),
            "selected_per_video": _per_video_metrics(
                all_labels,
                predictions,
                features.split.videos,
            ),
            "old_setting": {
                "score_cutoff": SCORE_CUTOFF,
                "duplicate_distance_at_30_fps": DUPLICATE_DISTANCE_AT_30_FPS,
                "metrics_by_tolerance": _metrics_at_tolerances(
                    all_labels,
                    old_predictions,
                    features.split.videos,
                ),
            },
            "final_score_file": final_score_path.name,
            "final_score_sha256": _sha256(final_score_path),
            "kept_contact_count": int(kept.sum()),
        }
        _write_json(result_path, result)
        return result_path
    except Exception as error:
        _write_json(
            result_path,
            {
                "schema": COMBINED_RESULT_SCHEMA,
                "status": "failed",
                "source_commit": source_commit,
                "labels_read": labels_read,
                "error_type": type(error).__name__,
            },
        )
        raise


def _read_final_scores(path: Path) -> np.ndarray:
    with lzma.open(path, "rb") as source:
        scores = np.load(source, allow_pickle=False)
    if scores.dtype != FINAL_SCORE_DTYPE:
        raise ValueError("final contact score fields differ")
    return scores


def _input_files_from_args(arguments: argparse.Namespace) -> InputFiles:
    return InputFiles(
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


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-record", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--chosen-baseline-result", type=Path, required=True)
    parser.add_argument("--contact-labels", type=Path, required=True)
    parser.add_argument("--chosen-validation-scores", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    score_parser = actions.add_parser("score", help="score one held-out group")
    _add_input_arguments(score_parser)
    score_parser.add_argument("--group", choices=GROUP_NAMES, required=True)
    score_parser.add_argument("--output-dir", type=Path, required=True)

    combine_parser = actions.add_parser(
        "combine", help="combine five groups and choose final settings"
    )
    _add_input_arguments(combine_parser)
    for group_name in GROUP_NAMES:
        combine_parser.add_argument(
            f"--group-{group_name.lower()}-dir",
            type=Path,
            required=True,
        )
    combine_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    files = _input_files_from_args(arguments)
    if arguments.action == "score":
        result_path = score_group(
            files,
            arguments.group,
            arguments.output_dir,
            arguments.source_commit,
        )
    else:
        directories = {
            group_name: getattr(arguments, f"group_{group_name.lower()}_dir")
            for group_name in GROUP_NAMES
        }
        result_path = combine_groups(
            files,
            directories,
            arguments.output_dir,
            arguments.source_commit,
        )
    print(result_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
