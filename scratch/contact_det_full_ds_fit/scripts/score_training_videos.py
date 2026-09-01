"""Score each training video with a contact model that did not train on it."""

from __future__ import annotations

import argparse
import json
import lzma
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scratch.contact_det.scripts.freeze_tree_contact_features import REGION_FIELDS
from scratch.contact_det_full_ds_fit.scripts.baseline_config import (
    BaselineConfig,
    BaselineRun,
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
    TrainingSelection,
    _feature_matrix,
    _find_run,
    _nearest_distances,
    _scaled_frames,
    _sha256,
    _write_json,
    _write_scores,
    collect_candidate_rows,
    load_contact_labels,
    make_model,
    predictions_for_settings,
)

GROUP_SCHEMA = "contact-training-video-score-groups/1"
INPUT_SCHEMA = "contact-training-video-score-inputs/1"
RESULT_SCHEMA = "contact-training-video-score-result/1"
COMBINED_RESULT_SCHEMA = "contact-training-video-scores/1"
CHOSEN_RUN_ID = "hgb_reference_raw_more_negatives"
SCORE_CUTOFF = 0.9
DUPLICATE_DISTANCE_AT_30_FPS = 6
GROUP_NAMES = ("A", "B", "C", "D")
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
GROUP_SCORE_FILE = "training_video_scores.npy.xz"
GROUP_RESULT_FILE = "training_video_score_result.json"
COMBINED_SCORE_FILE = "training_video_scores.npy.xz"
COMBINED_RESULT_FILE = "training_video_score_result.json"
COMBINED_SCORE_DTYPE = np.dtype(SCORE_DTYPE.descr + [("group", "S1")])
REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_ROW_CONDITION = "At least one listed unsigned-byte field equals 1"


@dataclass(frozen=True)
class InputFiles:
    """Files checked against the fixed input list."""

    input_list: Path
    groups: Path
    split: Path
    config: Path
    feature_record: Path
    baseline_summary: Path
    chosen_baseline_result: Path
    contact_labels: Path


@dataclass(frozen=True)
class ScoreGroup:
    """One fixed group of videos to score."""

    name: str
    scored_videos: tuple[VideoSpec, ...]
    training_videos: tuple[VideoSpec, ...]


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    return value


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    return _mapping(json.loads(Path(path).read_text(encoding="utf-8")), name)


def _files_by_role(files: InputFiles) -> dict[str, Path]:
    return {
        "training score groups": files.groups,
        "development split": files.split,
        "baseline settings": files.config,
        "raw feature record": files.feature_record,
        "baseline summary": files.baseline_summary,
        "chosen baseline result": files.chosen_baseline_result,
        "contact labels": files.contact_labels,
    }


def check_input_files(files: InputFiles) -> Mapping[str, Any]:
    """Check every fixed filename and hash before reading experiment data."""
    inputs = _load_json(files.input_list, "input list")
    if inputs.get("schema") != INPUT_SCHEMA:
        raise ValueError("input list version differs")
    raw_records = inputs.get("files")
    if not isinstance(raw_records, list):
        raise TypeError("input list files must be a list")
    records: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(raw_records):
        record = _mapping(value, f"files[{index}]")
        if set(record) != {"role", "filename", "sha256"}:
            raise ValueError(f"files[{index}] fields differ")
        role = record.get("role")
        if not isinstance(role, str) or role in records:
            raise ValueError("input file roles must be strings and unique")
        records[role] = record

    expected_paths = _files_by_role(files)
    if set(records) != set(expected_paths):
        raise ValueError("input file roles differ")
    for role, path in expected_paths.items():
        record = records[role]
        if record.get("filename") != path.name:
            raise ValueError(f"{role} filename differs")
        if record.get("sha256") != _sha256(path):
            raise ValueError(f"{role} file hash differs")
    return inputs


def _input_file_hash(inputs: Mapping[str, Any], role: str) -> str:
    raw_records = inputs.get("files")
    if not isinstance(raw_records, list):
        raise TypeError("input list files must be a list")
    matches = [
        _mapping(value, "input file")
        for value in raw_records
        if isinstance(value, Mapping) and value.get("role") == role
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise ValueError(f"input list must contain one {role} file hash")
    return str(matches[0]["sha256"])


def _fixture_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a list of video names")
    return value


def load_score_groups(
    path: Path,
    split: DevelopmentSplit,
) -> dict[str, ScoreGroup]:
    """Check the four fixed groups against the accepted development split."""
    payload = _load_json(path, "score groups")
    if payload.get("schema") != GROUP_SCHEMA:
        raise ValueError("score group version differs")
    expected_validation = [video.fixture for video in split.validation_videos]
    if _fixture_list(payload.get("fixed_validation_videos"), "fixed validation videos") != (
        expected_validation
    ):
        raise ValueError("fixed validation videos differ from the development split")

    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or len(raw_groups) != len(GROUP_NAMES):
        raise ValueError("there must be four score groups")
    by_fixture = split.by_fixture
    groups: dict[str, tuple[VideoSpec, ...]] = {}
    for expected_name, raw_group in zip(GROUP_NAMES, raw_groups, strict=True):
        group = _mapping(raw_group, f"group {expected_name}")
        if set(group) != {"group", "videos"} or group.get("group") != expected_name:
            raise ValueError(f"score group {expected_name} fields differ")
        raw_videos = group.get("videos")
        if not isinstance(raw_videos, list) or len(raw_videos) != 8:
            raise ValueError(f"score group {expected_name} must contain eight videos")
        videos: list[VideoSpec] = []
        women_match_count = 0
        for index, raw_video in enumerate(raw_videos):
            video_record = _mapping(raw_video, f"group {expected_name} video {index}")
            expected_fields = {"fixture", "video_id", "fps", "women_match", "tournament"}
            if set(video_record) != expected_fields:
                raise ValueError(f"group {expected_name} video {index} fields differ")
            fixture = video_record.get("fixture")
            if not isinstance(fixture, str) or fixture not in by_fixture:
                raise ValueError(f"group {expected_name} has an unknown video")
            video = by_fixture[fixture]
            if video not in split.training_videos:
                raise ValueError(f"group {expected_name} contains a non-training video")
            expected_identity = (video.fixture, video.video_id, video.fps, video.tournament)
            actual_identity = (
                video_record.get("fixture"),
                video_record.get("video_id"),
                float(video_record.get("fps", math.nan)),
                video_record.get("tournament"),
            )
            if actual_identity != expected_identity:
                raise ValueError(f"{fixture}: group details differ from the development split")
            women_match = video_record.get("women_match")
            if type(women_match) is not bool:
                raise TypeError(f"{fixture}: women_match must be true or false")
            women_match_count += int(women_match)
            videos.append(video)
        if len({video.fixture for video in videos}) != 8:
            raise ValueError(f"score group {expected_name} contains a duplicate video")
        if sum(video.fps == 25.0 for video in videos) != 4:
            raise ValueError(f"score group {expected_name} must have four 25 fps videos")
        if sum(video.fps == 30.0 for video in videos) != 4:
            raise ValueError(f"score group {expected_name} must have four 30 fps videos")
        expected_women_matches = 1 if expected_name == "D" else 2
        if women_match_count != expected_women_matches:
            raise ValueError(f"score group {expected_name} has the wrong women's match count")
        groups[expected_name] = tuple(videos)

    all_scored = [video.fixture for name in GROUP_NAMES for video in groups[name]]
    expected_training = [video.fixture for video in split.training_videos]
    if len(set(all_scored)) != 32 or set(all_scored) != set(expected_training):
        raise ValueError("score groups must cover each training video once")

    result: dict[str, ScoreGroup] = {}
    for scored_name in GROUP_NAMES:
        training = tuple(
            video
            for name in GROUP_NAMES
            if name != scored_name
            for video in groups[name]
        )
        result[scored_name] = ScoreGroup(scored_name, groups[scored_name], training)
    return result


def _check_fixed_model(
    config: BaselineConfig,
    run: BaselineRun,
    summary: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
) -> None:
    if config.random_seed != 20260824:
        raise ValueError("model random seed differs from the fixed run")
    if run.motion_mode.value != "raw_per_frame":
        raise ValueError("motion values differ from the fixed run")
    if run.model_kind.value != "histogram_gradient_boosting":
        raise ValueError("model kind differs from the fixed run")
    if summary.get("chosen_run_id") != CHOSEN_RUN_ID:
        raise ValueError("baseline summary names a different chosen run")
    if baseline_result.get("status") != "complete" or baseline_result.get("run_id") != run.run_id:
        raise ValueError("chosen baseline result differs from the requested run")
    expected_model = {
        "name": run.model_name,
        "kind": run.model_kind.value,
        "settings": dict(run.model_settings),
        "class_weight": run.class_weight,
        "class_weight_value": run.class_weight_value,
    }
    if baseline_result.get("model") != expected_model:
        raise ValueError("chosen model settings differ from the fixed run")
    training = _mapping(baseline_result.get("training_selection"), "training selection")
    expected_training = {
        "positive_radius_at_30_fps": config.positive_radius_at_30_fps,
        "ignored_radius_at_30_fps": config.ignored_radius_at_30_fps,
        "nearby_negative_radius_at_30_fps": config.hard_negative_radius_at_30_fps,
        "negative_rule": run.negative_rule,
        "maximum_negatives_per_positive": run.negative_limit,
    }
    if any(training.get(name) != value for name, value in expected_training.items()):
        raise ValueError("training example settings differ from the fixed run")
    if baseline_result.get("selected_score_cutoff") != SCORE_CUTOFF:
        raise ValueError("chosen score cut-off differs")
    if (
        baseline_result.get("selected_duplicate_distance_at_30_fps")
        != DUPLICATE_DISTANCE_AT_30_FPS
    ):
        raise ValueError("chosen nearby-contact distance differs")


def _check_candidate_rows(
    features: VerifiedFeatureDataset,
    candidates: CandidateRows,
    inputs: Mapping[str, Any],
) -> None:
    if inputs.get("expected_training_video_count") != 32:
        raise ValueError("expected training video count differs")
    if inputs.get("candidate_row_condition") != CANDIDATE_ROW_CONDITION:
        raise ValueError("candidate row condition differs")
    for field in REGION_FIELDS:
        if features.rows.dtype[field] != np.dtype("u1"):
            raise ValueError(f"{field} must use unsigned byte values")
        if not np.isin(features.rows[field], (0, 1)).all():
            raise ValueError(f"{field} must contain only zero and one")
    if inputs.get("candidate_row_fields") != list(REGION_FIELDS):
        raise ValueError("candidate row fields differ")
    expected_counts = _mapping(inputs.get("expected_score_rows_by_video"), "video row counts")
    actual_counts = {
        video.fixture: candidates.video_ranges[video.fixture][1]
        - candidates.video_ranges[video.fixture][0]
        for video in features.split.training_videos
    }
    if dict(expected_counts) != actual_counts:
        raise ValueError("candidate score row counts differ")
    expected_total = inputs.get("expected_candidate_score_row_count")
    if expected_total != sum(actual_counts.values()):
        raise ValueError("candidate score row total differs")


def _labels_for_videos(labels: ContactLabels, videos: Sequence[VideoSpec]) -> ContactLabels:
    fixtures = [video.fixture for video in videos]
    return ContactLabels(
        frames={fixture: labels.frames[fixture] for fixture in fixtures},
        first_contacts={fixture: labels.first_contacts[fixture] for fixture in fixtures},
        rally_counts={fixture: labels.rally_counts[fixture] for fixture in fixtures},
    )


def choose_training_rows_for_videos(
    candidates: CandidateRows,
    videos: Sequence[VideoSpec],
    contact_labels: ContactLabels,
    config: BaselineConfig,
    run: BaselineRun,
    *,
    expected_video_count: int = 24,
) -> TrainingSelection:
    """Choose training examples from only the named videos."""
    expected_names = [video.fixture for video in videos]
    label_names = set(contact_labels.frames)
    if expected_video_count <= 0:
        raise ValueError("expected training video count must be positive")
    if (
        len(expected_names) != expected_video_count
        or len(set(expected_names)) != expected_video_count
    ):
        raise ValueError(
            f"each fit must have {expected_video_count} distinct training videos"
        )
    if label_names != set(expected_names):
        raise ValueError(
            "contact labels must contain exactly the expected training videos"
        )
    if set(contact_labels.first_contacts) != label_names or set(contact_labels.rally_counts) != (
        label_names
    ):
        raise ValueError("contact label groups differ")

    selected = np.zeros(len(candidates.rows), dtype=bool)
    labels = np.zeros(len(candidates.rows), dtype=np.uint8)
    video_counts: dict[str, dict[str, int]] = {}
    random = np.random.default_rng(config.random_seed)
    for video in videos:
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
        raise ValueError("training examples must include contacts and non-contacts")
    return TrainingSelection(selected, labels, video_counts)


def _score_videos(
    candidates: CandidateRows,
    videos: Sequence[VideoSpec],
    model: Any,
    feature_names: Sequence[str],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for video in videos:
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


def _load_fixed_inputs(files: InputFiles) -> tuple[
    Mapping[str, Any],
    BaselineConfig,
    BaselineRun,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    inputs = check_input_files(files)
    from scratch.contact_det_full_ds_fit.scripts.baseline_config import (
        load_baseline_config,
    )

    config = load_baseline_config(files.config)
    run = _find_run(config, CHOSEN_RUN_ID)
    summary = _load_json(files.baseline_summary, "baseline summary")
    baseline_result = _load_json(files.chosen_baseline_result, "chosen baseline result")
    _check_fixed_model(config, run, summary, baseline_result)
    return inputs, config, run, summary, baseline_result


def score_group(
    files: InputFiles,
    group_name: str,
    output_dir: Path,
    source_commit: str,
    *,
    model_factory: Any = make_model,
    label_loader: Any = load_contact_labels,
    feature_loader: Any = None,
) -> Path:
    """Train one fixed model, score one group and save the checked result."""
    if group_name not in GROUP_NAMES:
        raise ValueError("group must be A, B, C or D")
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a short or full Git commit")
    source_root = str(REPO_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    destination = Path(output_dir)
    result_path = destination / GROUP_RESULT_FILE
    _write_json(
        result_path,
        {
            "schema": RESULT_SCHEMA,
            "status": "running",
            "group": group_name,
            "source_commit": source_commit,
        },
    )
    inputs, config, run, _summary, _baseline_result = _load_fixed_inputs(files)
    if feature_loader is None:
        from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
            load_verified_feature_dataset,
        )

        feature_loader = load_verified_feature_dataset
    features = feature_loader(files.feature_record, files.split, run.motion_mode.value)
    groups = load_score_groups(files.groups, features.split)
    group = groups[group_name]
    candidates = collect_candidate_rows(features)
    _check_candidate_rows(features, candidates, inputs)

    all_labels = label_loader(files.contact_labels, features.split)
    expected_label_hash = _input_file_hash(inputs, "contact labels")
    if _sha256(files.contact_labels) != expected_label_hash:
        raise ValueError("contact label file changed while it was read")
    training_labels = _labels_for_videos(all_labels, group.training_videos)
    training = choose_training_rows_for_videos(
        candidates,
        group.training_videos,
        training_labels,
        config,
        run,
    )
    scored_names = {video.fixture for video in group.scored_videos}
    validation_names = {video.fixture for video in features.split.validation_videos}
    selected_names = set(np.char.decode(candidates.rows[training.selected]["fixture"], "ascii"))
    if selected_names != {video.fixture for video in group.training_videos}:
        raise ValueError("selected training examples come from the wrong videos")
    if selected_names & (scored_names | validation_names):
        raise ValueError("a scored or validation video entered the training examples")

    model = model_factory(run, config.random_seed)
    model.fit(
        _feature_matrix(candidates.rows[training.selected], features.model_input_fields),
        training.labels[training.selected],
    )
    scores = _score_videos(
        candidates,
        group.scored_videos,
        model,
        features.model_input_fields,
    )
    _predictions, kept = predictions_for_settings(
        scores,
        group.scored_videos,
        SCORE_CUTOFF,
        DUPLICATE_DISTANCE_AT_30_FPS,
    )
    scores["kept"] = kept
    score_path = destination / GROUP_SCORE_FILE
    _write_scores(score_path, scores)

    expected_group_rows = sum(
        int(_mapping(inputs["expected_score_rows_by_video"], "video row counts")[video.fixture])
        for video in group.scored_videos
    )
    if len(scores) != expected_group_rows:
        raise ValueError("saved group score count differs")
    if not np.isfinite(scores["contact_score"]).all():
        raise ValueError("contact scores must be finite")
    if np.any((scores["contact_score"] < 0.0) | (scores["contact_score"] > 1.0)):
        raise ValueError("contact scores must be between zero and one")

    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "group": group.name,
        "source_commit": source_commit,
        "input_list_file": files.input_list.name,
        "input_list_sha256": _sha256(files.input_list),
        "input_files": {
            role: {"filename": path.name, "sha256": _sha256(path)}
            for role, path in _files_by_role(files).items()
        },
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
        "score_cutoff": SCORE_CUTOFF,
        "duplicate_distance_at_30_fps": DUPLICATE_DISTANCE_AT_30_FPS,
        "score_file": score_path.name,
        "score_sha256": _sha256(score_path),
        "score_row_count": len(scores),
        "kept_contact_count": int(scores["kept"].sum()),
    }
    _write_json(result_path, result)
    return result_path


def _read_scores(path: Path) -> np.ndarray:
    with lzma.open(path, "rb") as source:
        rows = np.load(source, allow_pickle=False)
    if rows.dtype != SCORE_DTYPE:
        raise ValueError(f"{path.parent.name}: score fields differ")
    return rows


def _score_identities(rows: np.ndarray) -> set[tuple[bytes, int, int, float]]:
    return {
        (bytes(row["fixture"]), int(row["interval_id"]), int(row["frame"]), float(row["fps"]))
        for row in rows
    }


def _check_saved_training_selection(
    value: object,
    group: ScoreGroup,
    config: BaselineConfig,
    run: BaselineRun,
) -> None:
    selection = _mapping(value, f"group {group.name} training selection")
    expected_settings = {
        "positive_radius_at_30_fps": config.positive_radius_at_30_fps,
        "ignored_radius_at_30_fps": config.ignored_radius_at_30_fps,
        "nearby_negative_radius_at_30_fps": config.hard_negative_radius_at_30_fps,
        "negative_rule": run.negative_rule,
        "maximum_negatives_per_positive": run.negative_limit,
    }
    if any(selection.get(name) != expected for name, expected in expected_settings.items()):
        raise ValueError(f"group {group.name} training settings differ")
    video_counts = _mapping(selection.get("videos"), f"group {group.name} training videos")
    expected_names = {video.fixture for video in group.training_videos}
    if set(video_counts) != expected_names:
        raise ValueError(f"group {group.name} training count videos differ")
    selected_total = 0
    positive_total = 0
    for video_name, value_by_kind in video_counts.items():
        counts = _mapping(value_by_kind, f"{group.name} {video_name} training counts")
        expected_fields = {
            "positive",
            "nearby_negative",
            "sampled_other_negative",
            "selected",
        }
        if set(counts) != expected_fields or any(type(count) is not int for count in counts.values()):
            raise ValueError(f"{group.name} {video_name} training counts differ")
        if any(count < 0 for count in counts.values()):
            raise ValueError(f"{group.name} {video_name} training counts must be non-negative")
        if counts["selected"] != (
            counts["positive"]
            + counts["nearby_negative"]
            + counts["sampled_other_negative"]
        ):
            raise ValueError(f"{group.name} {video_name} training counts do not add up")
        selected_total += counts["selected"]
        positive_total += counts["positive"]
    if selection.get("selected_row_count") != selected_total:
        raise ValueError(f"group {group.name} selected training count differs")
    if selection.get("positive_row_count") != positive_total:
        raise ValueError(f"group {group.name} positive training count differs")


def combine_groups(
    files: InputFiles,
    group_directories: Mapping[str, Path],
    output_dir: Path,
    source_commit: str,
    *,
    feature_loader: Any = None,
) -> Path:
    """Combine four complete group results and recheck every scored row."""
    if set(group_directories) != set(GROUP_NAMES):
        raise ValueError("one group directory is required for A, B, C and D")
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a short or full Git commit")
    destination = Path(output_dir)
    combined_result_path = destination / COMBINED_RESULT_FILE
    _write_json(
        combined_result_path,
        {
            "schema": COMBINED_RESULT_SCHEMA,
            "status": "running",
            "source_commit": source_commit,
        },
    )
    inputs, config, run, _summary, _baseline_result = _load_fixed_inputs(files)
    if feature_loader is None:
        from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
            load_verified_feature_dataset,
        )

        feature_loader = load_verified_feature_dataset
    features = feature_loader(files.feature_record, files.split, run.motion_mode.value)
    groups = load_score_groups(files.groups, features.split)
    candidates = collect_candidate_rows(features)
    _check_candidate_rows(features, candidates, inputs)

    chunks: list[np.ndarray] = []
    child_files: dict[str, object] = {}
    seen_identities: set[tuple[bytes, int, int, float]] = set()
    for group_name in GROUP_NAMES:
        group_dir = Path(group_directories[group_name])
        group_result_path = group_dir / GROUP_RESULT_FILE
        result = _load_json(group_result_path, f"group {group_name} result")
        if result.get("schema") != RESULT_SCHEMA or result.get("status") != "complete":
            raise ValueError(f"group {group_name} result is not complete")
        if result.get("group") != group_name or result.get("source_commit") != source_commit:
            raise ValueError(f"group {group_name} result identity differs")
        group = groups[group_name]
        expected_model = {
            "name": run.model_name,
            "kind": run.model_kind.value,
            "settings": dict(run.model_settings),
            "class_weight": run.class_weight,
            "class_weight_value": run.class_weight_value,
        }
        if result.get("run_id") != run.run_id or result.get("model") != expected_model:
            raise ValueError(f"group {group_name} model differs")
        _check_saved_training_selection(result.get("training_selection"), group, config, run)
        if result.get("training_videos") != [video.fixture for video in group.training_videos]:
            raise ValueError(f"group {group_name} training videos differ")
        if result.get("scored_videos") != [video.fixture for video in group.scored_videos]:
            raise ValueError(f"group {group_name} scored videos differ")
        if result.get("input_list_file") != files.input_list.name:
            raise ValueError(f"group {group_name} input list filename differs")
        if result.get("input_list_sha256") != _sha256(files.input_list):
            raise ValueError(f"group {group_name} input list differs")
        expected_input_files = {
            role: {"filename": path.name, "sha256": _sha256(path)}
            for role, path in _files_by_role(files).items()
        }
        if result.get("input_files") != expected_input_files:
            raise ValueError(f"group {group_name} input files differ")
        if result.get("score_cutoff") != SCORE_CUTOFF:
            raise ValueError(f"group {group_name} score cut-off differs")
        if result.get("duplicate_distance_at_30_fps") != DUPLICATE_DISTANCE_AT_30_FPS:
            raise ValueError(f"group {group_name} nearby-contact distance differs")

        score_path = group_dir / GROUP_SCORE_FILE
        if result.get("score_file") != score_path.name:
            raise ValueError(f"group {group_name} score filename differs")
        if result.get("score_sha256") != _sha256(score_path):
            raise ValueError(f"group {group_name} score file hash differs")
        rows = _read_scores(score_path)
        if result.get("score_row_count") != len(rows):
            raise ValueError(f"group {group_name} score row count differs")
        identities = _score_identities(rows)
        if len(identities) != len(rows):
            raise ValueError(f"group {group_name} has repeated score rows")
        if identities & seen_identities:
            raise ValueError(f"group {group_name} repeats rows from an earlier group")
        seen_identities.update(identities)
        expected_group_chunks: list[np.ndarray] = []
        for video in group.scored_videos:
            row_start, row_end = candidates.video_ranges[video.fixture]
            expected_group_chunks.append(candidates.rows[row_start:row_end])
        expected_group_rows = np.concatenate(expected_group_chunks)
        identity_fields = ("fixture", "interval_id", "frame", "fps")
        if len(rows) != len(expected_group_rows) or any(
            not np.array_equal(rows[field], expected_group_rows[field])
            for field in identity_fields
        ):
            raise ValueError(f"group {group_name} score rows differ from its candidates")
        if not np.isfinite(rows["contact_score"]).all():
            raise ValueError(f"group {group_name} has non-finite scores")
        if np.any((rows["contact_score"] < 0.0) | (rows["contact_score"] > 1.0)):
            raise ValueError(f"group {group_name} scores fall outside zero to one")
        _predictions, expected_kept = predictions_for_settings(
            rows,
            group.scored_videos,
            SCORE_CUTOFF,
            DUPLICATE_DISTANCE_AT_30_FPS,
        )
        if not np.array_equal(rows["kept"], expected_kept):
            raise ValueError(f"group {group_name} kept-contact flags differ")
        if result.get("kept_contact_count") != int(rows["kept"].sum()):
            raise ValueError(f"group {group_name} kept-contact count differs")
        combined_rows = np.empty(len(rows), dtype=COMBINED_SCORE_DTYPE)
        for field in SCORE_DTYPE.names or ():
            combined_rows[field] = rows[field]
        combined_rows["group"] = group_name.encode("ascii")
        chunks.append(combined_rows)
        child_files[group_name] = {
            "result_file": group_result_path.name,
            "result_sha256": _sha256(group_result_path),
            "score_file": score_path.name,
            "score_sha256": _sha256(score_path),
            "score_row_count": len(rows),
        }

    combined = np.concatenate(chunks)
    expected_candidate_chunks: list[np.ndarray] = []
    for group_name in GROUP_NAMES:
        for video in groups[group_name].scored_videos:
            row_start, row_end = candidates.video_ranges[video.fixture]
            expected_candidate_chunks.append(candidates.rows[row_start:row_end])
    expected_candidates = np.concatenate(expected_candidate_chunks)
    expected_identities = {
        (
            bytes(row["fixture"]),
            int(row["interval_id"]),
            int(row["frame"]),
            float(row["fps"]),
        )
        for row in expected_candidates
    }
    if len(combined) != inputs.get("expected_candidate_score_row_count"):
        raise ValueError("combined score row count differs")
    if seen_identities != expected_identities or len(expected_identities) != len(combined):
        raise ValueError("combined score rows differ from the fixed candidate rows")

    score_path = destination / COMBINED_SCORE_FILE
    _write_scores(score_path, combined)
    result: dict[str, object] = {
        "schema": COMBINED_RESULT_SCHEMA,
        "status": "complete",
        "source_commit": source_commit,
        "input_list_file": files.input_list.name,
        "input_list_sha256": _sha256(files.input_list),
        "groups": list(GROUP_NAMES),
        "videos": [
            video.fixture
            for group_name in GROUP_NAMES
            for video in groups[group_name].scored_videos
        ],
        "score_cutoff": SCORE_CUTOFF,
        "duplicate_distance_at_30_fps": DUPLICATE_DISTANCE_AT_30_FPS,
        "group_files": child_files,
        "score_file": score_path.name,
        "score_sha256": _sha256(score_path),
        "score_row_count": len(combined),
        "kept_contact_count": int(combined["kept"].sum()),
    }
    _write_json(combined_result_path, result)
    return combined_result_path


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
    parser.add_argument("--source-commit", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    score_parser = actions.add_parser("score", help="score one group")
    _add_input_arguments(score_parser)
    score_parser.add_argument("--group", choices=GROUP_NAMES, required=True)
    score_parser.add_argument("--output-dir", type=Path, required=True)

    combine_parser = actions.add_parser("combine", help="combine four group results")
    _add_input_arguments(combine_parser)
    combine_parser.add_argument("--group-a-dir", type=Path, required=True)
    combine_parser.add_argument("--group-b-dir", type=Path, required=True)
    combine_parser.add_argument("--group-c-dir", type=Path, required=True)
    combine_parser.add_argument("--group-d-dir", type=Path, required=True)
    combine_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
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
        group_directories = {
            "A": arguments.group_a_dir,
            "B": arguments.group_b_dir,
            "C": arguments.group_c_dir,
            "D": arguments.group_d_dir,
        }
        result_path = combine_groups(
            files,
            group_directories,
            arguments.output_dir,
            arguments.source_commit,
        )
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
