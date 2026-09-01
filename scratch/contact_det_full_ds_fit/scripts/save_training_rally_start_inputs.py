"""Save label-free rally-start inputs for the 32 training videos."""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scratch.contact_det.scripts.freeze_contact_evidence import (
    FixtureSpec,
    _load_inputs,
)
from scratch.contact_det_full_ds_fit.scripts.baseline_results import (
    VerifiedBaselineMenu,
    load_completed_baseline_menu,
)
from scratch.contact_det_full_ds_fit.scripts.check_rally_start_candidates import (
    DUPLICATE_DISTANCE_AT_30_FPS,
    build_video_candidate_lists,
)
from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
    VerifiedFeatureDataset,
)
from scratch.contact_det_full_ds_fit.scripts.save_validation_rally_predictions import (
    _check_centre_feature_values,
    _checked_stage_files,
    _normalise_side,
    _span_id,
    _spans,
    build_validation_rally_predictions,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    collect_candidate_rows,
    predictions_for_settings,
)
from scratch.contact_det_full_ds_fit.scripts.score_training_videos import (
    CHOSEN_RUN_ID,
    COMBINED_RESULT_FILE,
    COMBINED_RESULT_SCHEMA,
    COMBINED_SCORE_DTYPE,
    COMBINED_SCORE_FILE,
    GROUP_NAMES,
    GROUP_RESULT_FILE,
    GROUP_SCORE_FILE,
    RESULT_SCHEMA,
    SCORE_CUTOFF,
    SCORE_DTYPE,
    SOURCE_COMMIT,
    InputFiles,
    ScoreGroup,
    _check_candidate_rows,
    _check_saved_training_selection,
    _files_by_role,
    _load_fixed_inputs,
    _load_json,
    _read_scores,
    _sha256,
    load_score_groups,
)

SCORE_SUMMARY_SCHEMA = "contact-training-video-score-summary/1"
VIDEO_SCHEMA = "contact-rally-start-training-video/1"
COMBINED_SCHEMA = "contact-rally-start-training-inputs/1"
VIDEO_FILENAME = "rally_start_training_input.json.gz"
COMBINED_FILENAME = "rally_start_training_inputs.json.gz"
EXPECTED_SCORE_ROWS = 1_193_927
REPO_ROOT = Path(__file__).resolve().parents[3]

InputLoader = Callable[
    [Path, FixtureSpec],
    tuple[np.ndarray, Any, Any, list[tuple[int, int]], Any, Any],
]
SideAttributor = Callable[
    [int, np.ndarray, Any, np.ndarray, tuple[float, float]],
    object,
]


@dataclass(frozen=True)
class TrainingScorePaths:
    """Saved held-out scores and their tracked summary."""

    summary: Path
    root: Path

    @property
    def combined_result(self) -> Path:
        return self.root / "combined" / COMBINED_RESULT_FILE

    @property
    def combined_scores(self) -> Path:
        return self.root / "combined" / COMBINED_SCORE_FILE

    def group_result(self, group_name: str) -> Path:
        return self.root / f"group_{group_name}" / GROUP_RESULT_FILE

    def group_scores(self, group_name: str) -> Path:
        return self.root / f"group_{group_name}" / GROUP_SCORE_FILE


@dataclass(frozen=True)
class ValidationPaths:
    """Saved validation files used to prove that the rule is unchanged."""

    menu_result: Path
    common30_feature_record: Path
    rally_predictions: Path
    rally_result: Path
    chosen_scores: Path
    candidate_summary: Path
    frozen_candidates: Path


@dataclass(frozen=True)
class CheckedTrainingScores:
    """Held-out score rows checked against their four model sources."""

    rows: np.ndarray
    groups: Mapping[str, ScoreGroup]
    features: VerifiedFeatureDataset
    fixed_inputs: tuple[dict[str, object], ...]
    group_results: Mapping[str, Mapping[str, Any]]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be an object with string keys")
    return value


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    encoded = _json_bytes(value)
    if destination.name.endswith(".gz"):
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                compresslevel=9,
                mtime=0,
            ) as zipped,
        ):
            zipped.write(encoded)
    else:
        temporary.write_bytes(encoded)
    os.replace(temporary, destination)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    source_path = Path(path)
    if source_path.name.endswith(".gz"):
        with gzip.open(source_path, "rt", encoding="utf-8") as source:
            value = json.load(source)
    else:
        value = json.loads(source_path.read_text(encoding="utf-8"))
    return dict(_mapping(value, label))


def _file_record(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "filename": Path(path).name,
        "sha256": _sha256(path),
    }


def _read_combined_scores(path: Path) -> np.ndarray:
    with lzma.open(path, "rb") as source:
        rows = np.load(source, allow_pickle=False)
    if rows.dtype != COMBINED_SCORE_DTYPE:
        raise ValueError("combined training score fields differ")
    return rows


def _summary_group(
    summary: Mapping[str, Any],
    group_name: str,
) -> Mapping[str, Any]:
    raw_groups = summary.get("groups")
    if not isinstance(raw_groups, list):
        raise TypeError("training score summary groups must be a list")
    matches = [
        _mapping(value, f"summary group {group_name}")
        for value in raw_groups
        if isinstance(value, Mapping) and value.get("group") == group_name
    ]
    if len(matches) != 1:
        raise ValueError(f"training score summary group {group_name} differs")
    return matches[0]


def _expected_model(run: Any) -> dict[str, object]:
    return {
        "name": run.model_name,
        "kind": run.model_kind.value,
        "settings": dict(run.model_settings),
        "class_weight": run.class_weight,
        "class_weight_value": run.class_weight_value,
    }


def _check_group_result(
    group_name: str,
    group: ScoreGroup,
    result: Mapping[str, Any],
    summary_group: Mapping[str, Any],
    combined_child: Mapping[str, Any],
    paths: TrainingScorePaths,
    files: InputFiles,
    config: Any,
    run: Any,
    source_commit: str,
) -> None:
    result_path = paths.group_result(group_name)
    score_path = paths.group_scores(group_name)
    expected_training = [video.fixture for video in group.training_videos]
    expected_scored = [video.fixture for video in group.scored_videos]
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("group") != group_name
        or result.get("source_commit") != source_commit
        or result.get("run_id") != CHOSEN_RUN_ID
        or result.get("training_videos") != expected_training
        or result.get("scored_videos") != expected_scored
        or set(expected_training) & set(expected_scored)
    ):
        raise ValueError(f"group {group_name} model separation differs")
    if result.get("model") != _expected_model(run):
        raise ValueError(f"group {group_name} model differs")
    _check_saved_training_selection(
        result.get("training_selection"), group, config, run
    )
    if result.get("score_cutoff") != SCORE_CUTOFF:
        raise ValueError(f"group {group_name} score cut-off differs")
    if result.get("duplicate_distance_at_30_fps") != DUPLICATE_DISTANCE_AT_30_FPS:
        raise ValueError(f"group {group_name} nearby-contact distance differs")
    expected_input_files = {
        role: {"filename": path.name, "sha256": _sha256(path)}
        for role, path in _files_by_role(files).items()
    }
    if (
        result.get("input_list_file") != files.input_list.name
        or result.get("input_list_sha256") != _sha256(files.input_list)
        or result.get("input_files") != expected_input_files
    ):
        raise ValueError(f"group {group_name} fixed inputs differ")
    result_hash = _sha256(result_path)
    score_hash = _sha256(score_path)
    expected_shared = {
        "result_file": result_path.name,
        "result_sha256": result_hash,
        "score_file": score_path.name,
        "score_sha256": score_hash,
        "score_row_count": result.get("score_row_count"),
    }
    if dict(combined_child) != expected_shared:
        raise ValueError(f"group {group_name} combined child record differs")
    if (
        summary_group.get("result_sha256") != result_hash
        or summary_group.get("score_sha256") != score_hash
        or summary_group.get("training_video_count") != len(expected_training)
        or summary_group.get("scored_video_count") != len(expected_scored)
        or summary_group.get("score_row_count") != result.get("score_row_count")
        or summary_group.get("kept_contact_count") != result.get("kept_contact_count")
    ):
        raise ValueError(f"group {group_name} summary differs")


def _expected_score_rows(
    features: VerifiedFeatureDataset,
    groups: Mapping[str, ScoreGroup],
) -> np.ndarray:
    candidates = collect_candidate_rows(features)
    chunks: list[np.ndarray] = []
    for group_name in GROUP_NAMES:
        for video in groups[group_name].scored_videos:
            start, end = candidates.video_ranges[video.fixture]
            chunks.append(candidates.rows[start:end])
    return np.concatenate(chunks)


def _check_combined_rows(
    rows: np.ndarray,
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
    groups: Mapping[str, ScoreGroup],
    features: VerifiedFeatureDataset,
) -> None:
    if len(rows) != EXPECTED_SCORE_ROWS or result.get("score_row_count") != len(rows):
        raise ValueError("combined training score row count differs")
    if summary.get("score_row_count") != len(rows):
        raise ValueError("training score summary row count differs")
    identities = {
        (
            bytes(row["fixture"]),
            int(row["interval_id"]),
            int(row["frame"]),
            float(row["fps"]),
        )
        for row in rows
    }
    if len(identities) != len(rows):
        raise ValueError("combined training score identities repeat")
    if not np.isfinite(rows["contact_score"]).all():
        raise ValueError("combined training scores must be finite")
    if np.any((rows["contact_score"] < 0.0) | (rows["contact_score"] > 1.0)):
        raise ValueError("combined training scores must be between zero and one")

    expected_rows = _expected_score_rows(features, groups)
    identity_fields = ("fixture", "interval_id", "frame", "fps")
    if len(expected_rows) != len(rows) or any(
        not np.array_equal(rows[field], expected_rows[field])
        for field in identity_fields
    ):
        raise ValueError("combined training scores differ from the raw feature rows")

    candidates = collect_candidate_rows(features)
    start = 0
    for group_name in GROUP_NAMES:
        group = groups[group_name]
        group_count = sum(
            candidates.video_ranges[video.fixture][1]
            - candidates.video_ranges[video.fixture][0]
            for video in group.scored_videos
        )
        group_rows = rows[start : start + group_count]
        if not np.all(group_rows["group"] == group_name.encode("ascii")):
            raise ValueError(f"group {group_name} score marker differs")
        _predictions, expected_kept = predictions_for_settings(
            group_rows,
            group.scored_videos,
            SCORE_CUTOFF,
            DUPLICATE_DISTANCE_AT_30_FPS,
        )
        if not np.array_equal(group_rows["kept"], expected_kept):
            raise ValueError(f"group {group_name} kept-contact flags differ")
        start += group_count
    if start != len(rows):
        raise ValueError("combined training score group lengths differ")
    kept_count = int(rows["kept"].sum())
    if (
        result.get("kept_contact_count") != kept_count
        or summary.get("kept_contact_count") != kept_count
    ):
        raise ValueError("combined kept-contact count differs")


def _check_child_score_rows(rows: np.ndarray, paths: TrainingScorePaths) -> None:
    """Require the combined rows to equal the four checked child score files."""
    for group_name in GROUP_NAMES:
        combined_rows = rows[rows["group"] == group_name.encode("ascii")]
        child_rows = _read_scores(paths.group_scores(group_name))
        if len(combined_rows) != len(child_rows) or any(
            not np.array_equal(combined_rows[field], child_rows[field])
            for field in SCORE_DTYPE.names or ()
        ):
            raise ValueError(f"group {group_name} combined score rows differ")


def check_training_scores(
    paths: TrainingScorePaths,
    files: InputFiles,
    features: VerifiedFeatureDataset,
) -> CheckedTrainingScores:
    """Check the four model sources and their combined held-out scores."""
    inputs, config, run, _baseline_summary, _baseline_result = _load_fixed_inputs(files)
    if features.record_path.resolve() != files.feature_record.resolve():
        raise ValueError("raw feature record differs from the checked validation menu")
    groups = load_score_groups(files.groups, features.split)
    candidates = collect_candidate_rows(features)
    _check_candidate_rows(features, candidates, inputs)

    summary = _load_json(paths.summary, "training score summary")
    result = _load_json(paths.combined_result, "combined training score result")
    if (
        summary.get("schema") != SCORE_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("model") != CHOSEN_RUN_ID
        or summary.get("score_cutoff") != SCORE_CUTOFF
        or summary.get("duplicate_distance_at_30_fps") != DUPLICATE_DISTANCE_AT_30_FPS
        or summary.get("training_video_count") != 32
        or summary.get("group_count") != len(GROUP_NAMES)
    ):
        raise ValueError("training score summary differs")
    source_commit = summary.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or SOURCE_COMMIT.fullmatch(source_commit) is None
    ):
        raise ValueError("training score source commit differs")
    if summary.get("combined_result_sha256") != _sha256(
        paths.combined_result
    ) or summary.get("combined_score_sha256") != _sha256(paths.combined_scores):
        raise ValueError("combined training score hashes differ")
    expected_videos = [
        video.fixture
        for group_name in GROUP_NAMES
        for video in groups[group_name].scored_videos
    ]
    raw_children = _mapping(result.get("group_files"), "combined group files")
    if (
        result.get("schema") != COMBINED_RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("source_commit") != source_commit
        or result.get("groups") != list(GROUP_NAMES)
        or result.get("videos") != expected_videos
        or result.get("score_cutoff") != SCORE_CUTOFF
        or result.get("duplicate_distance_at_30_fps") != DUPLICATE_DISTANCE_AT_30_FPS
        or result.get("score_file") != paths.combined_scores.name
        or result.get("score_sha256") != _sha256(paths.combined_scores)
        or set(raw_children) != set(GROUP_NAMES)
    ):
        raise ValueError("combined training score result differs")

    group_results: dict[str, Mapping[str, Any]] = {}
    for group_name in GROUP_NAMES:
        group_result = _load_json(
            paths.group_result(group_name), f"group {group_name} result"
        )
        _check_group_result(
            group_name,
            groups[group_name],
            group_result,
            _summary_group(summary, group_name),
            _mapping(raw_children[group_name], f"combined group {group_name}"),
            paths,
            files,
            config,
            run,
            source_commit,
        )
        group_results[group_name] = group_result

    rows = _read_combined_scores(paths.combined_scores)
    _check_combined_rows(rows, result, summary, groups, features)
    _check_child_score_rows(rows, paths)
    for group_name in GROUP_NAMES:
        group_rows = rows[rows["group"] == group_name.encode("ascii")]
        group_result = group_results[group_name]
        if group_result.get("score_row_count") != len(group_rows) or group_result.get(
            "kept_contact_count"
        ) != int(group_rows["kept"].sum()):
            raise ValueError(f"group {group_name} saved score counts differ")
    fixed_inputs = (
        _file_record("training score summary", paths.summary),
        _file_record("combined training score result", paths.combined_result),
        _file_record("combined training scores", paths.combined_scores),
        *(
            record
            for group_name in GROUP_NAMES
            for record in (
                _file_record(
                    f"group {group_name} result", paths.group_result(group_name)
                ),
                _file_record(
                    f"group {group_name} scores", paths.group_scores(group_name)
                ),
            )
        ),
        *(_file_record(role, path) for role, path in _files_by_role(files).items()),
    )
    return CheckedTrainingScores(rows, groups, features, fixed_inputs, group_results)


def _video_record(
    features: VerifiedFeatureDataset,
    video_name: str,
) -> Mapping[str, Any]:
    raw_videos = features.record.get("videos")
    if not isinstance(raw_videos, list):
        raise TypeError("raw feature video records must be a list")
    matches = [
        _mapping(value, f"{video_name}: raw feature video")
        for value in raw_videos
        if isinstance(value, Mapping)
        and isinstance(value.get("video"), Mapping)
        and value["video"].get("name") == video_name
    ]
    if len(matches) != 1:
        raise ValueError(f"{video_name}: raw feature video record differs")
    return matches[0]


def _search_intervals(
    feature_record: Mapping[str, Any],
    video_name: str,
) -> tuple[tuple[int, int], ...]:
    summary = _mapping(
        feature_record.get("feature_summary"), f"{video_name}: feature summary"
    )
    raw_intervals = summary.get("search_intervals")
    if not isinstance(raw_intervals, list):
        raise TypeError(f"{video_name}: search intervals must be a list")
    intervals: list[tuple[int, int]] = []
    previous_end = -1
    for raw_interval in raw_intervals:
        if (
            not isinstance(raw_interval, list)
            or len(raw_interval) != 2
            or type(raw_interval[0]) is not int
            or type(raw_interval[1]) is not int
        ):
            raise ValueError(f"{video_name}: search interval differs")
        start, end = raw_interval
        if start < 0 or end <= start or start < previous_end:
            raise ValueError(
                f"{video_name}: search intervals must be ordered and separate"
            )
        intervals.append((start, end))
        previous_end = end
    return tuple(intervals)


def _video_feature_rows(
    features: VerifiedFeatureDataset,
    video_name: str,
) -> np.ndarray:
    start, end = features.video_ranges[video_name]
    rows = features.rows[start:end]
    if len(np.unique(rows["frame"])) != len(rows):
        raise ValueError(f"{video_name}: raw feature frames repeat")
    return rows


def _rows_for_frames(
    video_name: str,
    video_rows: np.ndarray,
    frames: np.ndarray,
) -> np.ndarray:
    order = np.argsort(video_rows["frame"])
    ordered = video_rows[order]
    positions = np.searchsorted(ordered["frame"], frames)
    if np.any(positions >= len(ordered)) or not np.array_equal(
        ordered["frame"][positions], frames
    ):
        raise ValueError(f"{video_name}: a replay frame has no raw feature row")
    return ordered[positions]


def _saved_validation_run(
    payload: Mapping[str, Any],
    run_id: str,
) -> Mapping[str, Any]:
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list):
        raise TypeError("saved validation runs must be a list")
    matches = [
        _mapping(value, f"validation run {run_id}")
        for value in raw_runs
        if isinstance(value, Mapping) and value.get("run_id") == run_id
    ]
    if len(matches) != 1:
        raise ValueError(f"saved validation run {run_id} differs")
    return matches[0]


def _saved_validation_videos(
    payload: Mapping[str, Any],
    key: str,
) -> dict[str, Mapping[str, Any]]:
    raw_videos = payload.get(key)
    if not isinstance(raw_videos, list):
        raise TypeError(f"saved validation {key} must be a list")
    output: dict[str, Mapping[str, Any]] = {}
    for raw_video in raw_videos:
        video = _mapping(raw_video, f"saved validation {key} video")
        fixture = video.get("fixture")
        if not isinstance(fixture, str) or fixture in output:
            raise ValueError(f"saved validation {key} video identities differ")
        output[fixture] = video
    return output


def _build_validation_candidate_lists(
    verified: VerifiedBaselineMenu,
    saved_predictions: Mapping[str, Any],
) -> tuple[list[dict[str, object]], int]:
    matching_runs = [run for run in verified.runs if run.run.run_id == CHOSEN_RUN_ID]
    if len(matching_runs) != 1:
        raise ValueError("chosen validation run differs")
    checked_run = matching_runs[0]
    saved_run = _saved_validation_run(saved_predictions, CHOSEN_RUN_ID)
    saved_videos = _saved_validation_videos(saved_predictions, "videos")
    saved_contacts = _saved_validation_videos(saved_run, "videos")
    score_names = np.char.decode(checked_run.score_rows["fixture"], "ascii")
    candidate_lists: list[dict[str, object]] = []
    skipped = 0

    for video in verified.split.validation_videos:
        feature_record = _video_record(verified.raw_features, video.fixture)
        intervals = _search_intervals(feature_record, video.fixture)
        raw_spans = saved_videos[video.fixture].get("spans")
        if not isinstance(raw_spans, list):
            raise TypeError(f"{video.fixture}: saved validation spans must be a list")
        spans = [
            _mapping(span, f"{video.fixture}: saved validation span")
            for span in raw_spans
        ]
        raw_contacts = saved_contacts[video.fixture].get("contacts")
        if not isinstance(raw_contacts, list):
            raise TypeError(
                f"{video.fixture}: saved validation contacts must be a list"
            )
        kept_frames = [
            int(_mapping(contact, f"{video.fixture}: validation contact")["frame"])
            for contact in raw_contacts
        ]
        expected_frames = checked_run.predictions[video.fixture].tolist()
        if kept_frames != expected_frames:
            raise ValueError(f"{video.fixture}: saved validation contacts differ")
        video_lists, video_skipped = build_video_candidate_lists(
            video.fixture,
            video.fps,
            checked_run.score_rows[score_names == video.fixture],
            kept_frames,
            spans,
            intervals,
            DUPLICATE_DISTANCE_AT_30_FPS,
        )
        candidate_lists.extend(video_lists)
        skipped += video_skipped
    return candidate_lists, skipped


def check_validation_reproduction(
    verified: VerifiedBaselineMenu,
    paths: ValidationPaths,
    baseline_summary_path: Path,
    data_root: Path,
    source_commit: str,
) -> tuple[dict[str, object], ...]:
    """Reproduce the frozen validation list and player-side answers."""
    baseline_summary = _load_json(baseline_summary_path, "baseline summary")
    chosen_files = _mapping(
        baseline_summary.get("chosen_run_files"),
        "chosen validation files",
    )
    if (
        baseline_summary.get("chosen_run_id") != CHOSEN_RUN_ID
        or baseline_summary.get("prediction_file") != paths.rally_predictions.name
        or baseline_summary.get("prediction_sha256") != _sha256(paths.rally_predictions)
        or baseline_summary.get("result_file") != paths.rally_result.name
        or baseline_summary.get("result_sha256") != _sha256(paths.rally_result)
        or chosen_files.get("score_file") != paths.chosen_scores.name
        or chosen_files.get("score_sha256") != _sha256(paths.chosen_scores)
    ):
        raise ValueError("saved validation file hashes differ")
    candidate_summary = _load_json(
        paths.candidate_summary,
        "validation candidate summary",
    )
    if (
        candidate_summary.get("run_id") != CHOSEN_RUN_ID
        or candidate_summary.get("construction_file") != paths.frozen_candidates.name
        or candidate_summary.get("construction_sha256")
        != _sha256(paths.frozen_candidates)
    ):
        raise ValueError("frozen validation candidate hash differs")
    saved_predictions = _read_json(
        paths.rally_predictions, "saved validation predictions"
    )
    replay = build_validation_rally_predictions(verified, data_root, source_commit)
    for field in (
        "labels_read",
        "menu_result_file",
        "menu_result_sha256",
        "split_file",
        "split_sha256",
        "raw_feature_record_file",
        "raw_feature_record_sha256",
        "contact_label_file",
        "contact_label_sha256",
        "validation_videos",
        "centre_feature_fields_checked",
        "videos",
        "runs",
    ):
        if replay.get(field) != saved_predictions.get(field):
            raise ValueError(f"validation player-side replay {field} differs")

    candidate_lists, skipped = _build_validation_candidate_lists(
        verified,
        saved_predictions,
    )
    frozen = _read_json(paths.frozen_candidates, "frozen validation candidates")
    raw_frozen_lists = frozen.get("candidate_lists")
    frozen_counts = _mapping(frozen.get("counts"), "frozen validation counts")
    if (
        not isinstance(raw_frozen_lists, list)
        or candidate_lists != raw_frozen_lists
        or len(candidate_lists) != frozen_counts.get("candidate_lists")
        or sum(len(row["candidates"]) for row in candidate_lists) != 1_845
        or skipped != frozen_counts.get("sections_without_kept_contact")
    ):
        raise ValueError("frozen validation candidate construction differs")
    return (
        _file_record("validation menu result", paths.menu_result),
        _file_record("validation rally predictions", paths.rally_predictions),
        _file_record("validation rally result", paths.rally_result),
        _file_record("chosen validation scores", paths.chosen_scores),
        _file_record("validation candidate summary", paths.candidate_summary),
        _file_record("frozen validation candidates", paths.frozen_candidates),
        _file_record("common-30 feature record", paths.common30_feature_record),
    )


def _fixture(video: Any) -> FixtureSpec:
    return FixtureSpec(
        name=video.fixture,
        video_id=video.video_id,
        fps=video.fps,
        width=float(video.width),
        height=float(video.height),
    )


def _score_rows_for_video(rows: np.ndarray, video_name: str) -> np.ndarray:
    selected = rows[rows["fixture"] == video_name.encode("ascii")]
    if len(selected) == 0:
        raise ValueError(f"{video_name}: held-out score rows are missing")
    if len(np.unique(selected["frame"])) != len(selected):
        raise ValueError(f"{video_name}: held-out score frames repeat")
    return selected


def _candidate_frames(candidate_lists: Sequence[Mapping[str, object]]) -> set[int]:
    frames: set[int] = set()
    for candidate_list in candidate_lists:
        raw_candidates = candidate_list.get("candidates")
        if not isinstance(raw_candidates, list):
            raise TypeError("candidate entries must be a list")
        for raw_candidate in raw_candidates:
            candidate = _mapping(raw_candidate, "candidate entry")
            frames.add(int(candidate["frame"]))
    return frames


def _enriched_candidates(
    candidate_lists: Sequence[Mapping[str, object]],
    rows_by_frame: Mapping[int, np.void],
    sides: Mapping[int, str | None],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for raw_list in candidate_lists:
        candidate_list = dict(raw_list)
        raw_candidates = candidate_list.get("candidates")
        if not isinstance(raw_candidates, list):
            raise TypeError("candidate entries must be a list")
        enriched: list[dict[str, object]] = []
        for raw_candidate in raw_candidates:
            candidate = dict(_mapping(raw_candidate, "candidate entry"))
            frame = int(candidate["frame"])
            row = rows_by_frame[frame]
            if float(candidate["contact_score"]) != float(row["contact_score"]):
                raise ValueError(f"candidate {frame}: contact score differs")
            candidate["kept"] = bool(row["kept"])
            candidate["predicted_side"] = sides[frame]
            enriched.append(candidate)
        candidate_list["candidates"] = enriched
        output.append(candidate_list)
    return output


def _kept_contacts(
    video_rows: np.ndarray,
    group_name: str,
    spans: Sequence[Mapping[str, int]],
    sides: Mapping[int, str | None],
) -> list[dict[str, object]]:
    return [
        {
            "frame": int(row["frame"]),
            "interval_id": int(row["interval_id"]),
            "contact_score": float(row["contact_score"]),
            "group": group_name,
            "span_id": _span_id(int(row["frame"]), spans),
            "predicted_side": sides[int(row["frame"])],
        }
        for row in video_rows
        if bool(row["kept"])
    ]


def _assemble_video_value(
    source_commit: str,
    video: Any,
    group_name: str,
    training_videos: Sequence[str],
    feature_record: Mapping[str, Any],
    fixed_inputs: Sequence[Mapping[str, object]],
    checked_stage_files: Sequence[Mapping[str, object]],
    spans: Sequence[Mapping[str, int]],
    video_rows: np.ndarray,
    sides: Mapping[int, str | None],
    candidate_lists: Sequence[Mapping[str, object]],
    skipped_section_count: int,
) -> dict[str, object]:
    rows_by_frame = {int(row["frame"]): row for row in video_rows}
    kept_contacts = _kept_contacts(video_rows, group_name, spans, sides)
    saved_candidates = _enriched_candidates(candidate_lists, rows_by_frame, sides)
    earlier_count = sum(
        not bool(candidate["is_fixed_contact"])
        for candidate_list in saved_candidates
        for candidate in candidate_list["candidates"]
    )
    feature_summary = _mapping(
        feature_record.get("feature_summary"),
        f"{video.fixture}: feature summary",
    )
    return {
        "schema": VIDEO_SCHEMA,
        "status": "complete",
        "source_commit": source_commit,
        "labels_read": False,
        "group": group_name,
        "model_training_videos": list(training_videos),
        "video": {
            "fixture": video.fixture,
            "video_id": video.video_id,
            "fps": video.fps,
            "frame_count": feature_summary["frame_count"],
        },
        "inputs": {
            "fixed_files": list(fixed_inputs),
            "feature_file": {
                "filename": Path(str(feature_record["feature_file"])).name,
                "sha256": feature_record["feature_sha256"],
            },
            "stage_files": list(checked_stage_files),
        },
        "spans": list(spans),
        "kept_contacts": kept_contacts,
        "candidate_lists": saved_candidates,
        "counts": {
            "detected_sections": len(spans),
            "sections_without_kept_contact": skipped_section_count,
            "kept_contacts": len(kept_contacts),
            "candidate_lists": len(saved_candidates),
            "candidate_entries": sum(
                len(candidate_list["candidates"]) for candidate_list in saved_candidates
            ),
            "earlier_candidate_entries": earlier_count,
            "distinct_replayed_frames": len(sides),
        },
    }


def _check_span_bounds(
    video_name: str,
    spans: Sequence[Mapping[str, int]],
    frame_count: int,
) -> None:
    previous_end = -1
    for expected_id, span in enumerate(spans):
        span_id = int(span["span_id"])
        start = int(span["start_frame"])
        end = int(span["end_frame"])
        if (
            span_id != expected_id
            or start < previous_end
            or start < 0
            or end <= start
            or end > frame_count
        ):
            raise ValueError(f"{video_name}: detected section bounds differ")
        previous_end = end


def _saved_inputs_match(
    saved: Mapping[str, Any],
    source_commit: str,
    fixed_inputs: Sequence[Mapping[str, object]],
    feature_record: Mapping[str, Any],
    checked_stage_files: Sequence[Mapping[str, object]],
) -> bool:
    if saved.get("schema") != VIDEO_SCHEMA or saved.get("status") != "complete":
        return False
    if saved.get("source_commit") != source_commit:
        return False
    inputs = _mapping(saved.get("inputs"), "saved training video inputs")
    expected_feature = {
        "filename": Path(str(feature_record["feature_file"])).name,
        "sha256": feature_record["feature_sha256"],
    }
    return (
        inputs.get("fixed_files") == list(fixed_inputs)
        and inputs.get("feature_file") == expected_feature
        and inputs.get("stage_files") == list(checked_stage_files)
    )


def _checked_side(value: object, label: str) -> str | None:
    if value is None or value in {"Top", "Bot"}:
        return value
    raise ValueError(f"{label}: predicted side differs")


def _check_saved_video_contents(
    saved: Mapping[str, Any],
    checked: CheckedTrainingScores,
    video: Any,
    group_name: str,
    feature_record: Mapping[str, Any],
) -> None:
    identity = _mapping(saved.get("video"), f"{video.fixture}: saved video identity")
    expected_training = [
        training_video.fixture
        for training_video in checked.groups[group_name].training_videos
    ]
    if (
        saved.get("schema") != VIDEO_SCHEMA
        or saved.get("status") != "complete"
        or saved.get("labels_read") is not False
        or saved.get("group") != group_name
        or saved.get("model_training_videos") != expected_training
        or identity.get("fixture") != video.fixture
        or identity.get("video_id") != video.video_id
        or identity.get("fps") != video.fps
    ):
        raise ValueError(f"{video.fixture}: saved video identity differs")
    frame_count = int(
        _mapping(
            feature_record["feature_summary"], f"{video.fixture}: feature summary"
        )["frame_count"]
    )
    if identity.get("frame_count") != frame_count:
        raise ValueError(f"{video.fixture}: saved frame count differs")
    raw_spans = saved.get("spans")
    if not isinstance(raw_spans, list):
        raise TypeError(f"{video.fixture}: saved spans must be a list")
    spans = [_mapping(span, f"{video.fixture}: saved span") for span in raw_spans]
    _check_span_bounds(video.fixture, spans, frame_count)

    video_rows = _score_rows_for_video(checked.rows, video.fixture)
    kept_rows = [row for row in video_rows if bool(row["kept"])]
    raw_contacts = saved.get("kept_contacts")
    if not isinstance(raw_contacts, list) or len(raw_contacts) != len(kept_rows):
        raise ValueError(f"{video.fixture}: saved kept-contact count differs")
    side_by_frame: dict[int, str | None] = {}
    for raw_contact, row in zip(raw_contacts, kept_rows, strict=True):
        contact = _mapping(raw_contact, f"{video.fixture}: saved kept contact")
        frame = int(row["frame"])
        expected = {
            "frame": frame,
            "interval_id": int(row["interval_id"]),
            "contact_score": float(row["contact_score"]),
            "group": group_name,
            "span_id": _span_id(frame, spans),
        }
        if any(contact.get(field) != value for field, value in expected.items()):
            raise ValueError(f"{video.fixture}/{frame}: saved kept contact differs")
        side_by_frame[frame] = _checked_side(
            contact.get("predicted_side"),
            f"{video.fixture}/{frame}",
        )

    expected_lists, skipped = build_video_candidate_lists(
        video.fixture,
        video.fps,
        video_rows,
        [int(row["frame"]) for row in kept_rows],
        spans,
        _search_intervals(feature_record, video.fixture),
        DUPLICATE_DISTANCE_AT_30_FPS,
    )
    raw_lists = saved.get("candidate_lists")
    if not isinstance(raw_lists, list) or len(raw_lists) != len(expected_lists):
        raise ValueError(f"{video.fixture}: saved candidate-list count differs")
    for raw_list, expected_list in zip(raw_lists, expected_lists, strict=True):
        candidate_list = _mapping(raw_list, f"{video.fixture}: saved candidate list")
        for field, value in expected_list.items():
            if field != "candidates" and candidate_list.get(field) != value:
                raise ValueError(f"{video.fixture}: saved candidate-list fields differ")
        raw_candidates = candidate_list.get("candidates")
        expected_candidates = expected_list["candidates"]
        if not isinstance(raw_candidates, list) or len(raw_candidates) != len(
            expected_candidates
        ):
            raise ValueError(f"{video.fixture}: saved candidate entries differ")
        for raw_candidate, expected_candidate in zip(
            raw_candidates,
            expected_candidates,
            strict=True,
        ):
            candidate = _mapping(raw_candidate, f"{video.fixture}: saved candidate")
            if any(
                candidate.get(field) != value
                for field, value in expected_candidate.items()
            ):
                raise ValueError(f"{video.fixture}: saved candidate fields differ")
            frame = int(candidate["frame"])
            matching_rows = video_rows[video_rows["frame"] == frame]
            if len(matching_rows) != 1 or candidate.get("kept") is not bool(
                matching_rows[0]["kept"]
            ):
                raise ValueError(
                    f"{video.fixture}/{frame}: saved candidate kept flag differs"
                )
            side = _checked_side(
                candidate.get("predicted_side"),
                f"{video.fixture}/{frame}",
            )
            if frame in side_by_frame and side_by_frame[frame] != side:
                raise ValueError(
                    f"{video.fixture}/{frame}: saved player sides disagree"
                )
            side_by_frame[frame] = side

    counts = _mapping(saved.get("counts"), f"{video.fixture}: saved counts")
    expected_counts = {
        "detected_sections": len(spans),
        "sections_without_kept_contact": skipped,
        "kept_contacts": len(kept_rows),
        "candidate_lists": len(expected_lists),
        "candidate_entries": sum(len(row["candidates"]) for row in expected_lists),
        "earlier_candidate_entries": sum(
            len(row["candidates"]) - 1 for row in expected_lists
        ),
        "distinct_replayed_frames": len(side_by_frame),
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"{video.fixture}: saved counts differ")


def save_training_video(
    checked: CheckedTrainingScores,
    video: Any,
    group_name: str,
    data_root: Path,
    output_path: Path,
    source_commit: str,
    *,
    resume: bool,
    input_loader: InputLoader = _load_inputs,
    side_attributor: SideAttributor | None = None,
) -> dict[str, Any]:
    """Build and save one checked video input file."""
    if side_attributor is None:
        from annotator.point_winner import attribute_half

        side_attributor = attribute_half
    feature_record = _video_record(checked.features, video.fixture)
    fixture = _fixture(video)
    saved_for_resume: dict[str, Any] | None = None
    if resume and Path(output_path).is_file():
        try:
            saved_for_resume = _read_json(
                output_path,
                f"{video.fixture}: saved training input",
            )
        except Exception:
            _write_json(
                output_path,
                {
                    "schema": VIDEO_SCHEMA,
                    "status": "running",
                    "source_commit": source_commit,
                    "labels_read": False,
                    "video": {"fixture": video.fixture},
                },
            )
            raise

    _write_json(
        output_path,
        {
            "schema": VIDEO_SCHEMA,
            "status": "running",
            "source_commit": source_commit,
            "labels_read": False,
            "video": {"fixture": video.fixture},
        },
    )
    checked_files = _checked_stage_files(data_root, fixture, feature_record)
    if saved_for_resume is not None and _saved_inputs_match(
        saved_for_resume,
        source_commit,
        checked.fixed_inputs,
        feature_record,
        checked_files,
    ):
        _check_saved_video_contents(
            saved_for_resume,
            checked,
            video,
            group_name,
            feature_record,
        )
        _write_json(output_path, saved_for_resume)
        return saved_for_resume
    track, pose, court, _tracker_intervals, sticky, annotation = input_loader(
        data_root,
        fixture,
    )
    feature_summary = _mapping(
        feature_record.get("feature_summary"),
        f"{video.fixture}: feature summary",
    )
    frame_count = int(feature_summary["frame_count"])
    if len(track) != frame_count:
        raise ValueError(f"{video.fixture}: replay frame count differs")
    court_inputs = getattr(getattr(court, "evidence", None), "inputs", None)
    if court_inputs is None:
        raise ValueError(f"{video.fixture}: court inputs are unavailable")
    net_band = tuple(float(value) for value in court_inputs.net_band)
    if (
        len(net_band) != 2
        or not np.all(np.isfinite(net_band))
        or net_band[0] > net_band[1]
    ):
        raise ValueError(f"{video.fixture}: net band differs")
    spans = _spans(annotation, video.fixture)
    _check_span_bounds(video.fixture, spans, frame_count)
    if len(spans) != feature_summary.get("rally_span_count"):
        raise ValueError(f"{video.fixture}: detected section count differs")

    video_scores = _score_rows_for_video(checked.rows, video.fixture)
    intervals = _search_intervals(feature_record, video.fixture)
    kept_frames = [int(row["frame"]) for row in video_scores if bool(row["kept"])]
    candidate_lists, skipped = build_video_candidate_lists(
        video.fixture,
        video.fps,
        video_scores,
        kept_frames,
        spans,
        intervals,
        DUPLICATE_DISTANCE_AT_30_FPS,
    )
    replay_frames = np.asarray(
        sorted(set(kept_frames) | _candidate_frames(candidate_lists)),
        dtype=np.int32,
    )
    feature_rows = _rows_for_frames(
        video.fixture,
        _video_feature_rows(checked.features, video.fixture),
        replay_frames,
    )
    _check_centre_feature_values(
        video.fixture,
        feature_rows,
        replay_frames,
        track,
        pose,
        sticky,
        (float(video.width), float(video.height)),
    )
    sides = {
        int(frame): _normalise_side(
            side_attributor(int(frame), track, sticky, pose.bboxes, net_band),
            f"{video.fixture}/{frame}",
        )
        for frame in replay_frames
    }
    training_videos = [
        training_video.fixture
        for training_video in checked.groups[group_name].training_videos
    ]
    first = _assemble_video_value(
        source_commit,
        video,
        group_name,
        training_videos,
        feature_record,
        checked.fixed_inputs,
        checked_files,
        spans,
        video_scores,
        sides,
        candidate_lists,
        skipped,
    )
    second = _assemble_video_value(
        source_commit,
        video,
        group_name,
        training_videos,
        feature_record,
        checked.fixed_inputs,
        checked_files,
        spans,
        video_scores,
        sides,
        candidate_lists,
        skipped,
    )
    if _json_bytes(first) != _json_bytes(second):
        raise ValueError(f"{video.fixture}: repeated training input build differs")
    _write_json(output_path, first)
    saved = _read_json(output_path, f"{video.fixture}: saved training input")
    if _json_bytes(saved) != _json_bytes(first):
        raise ValueError(f"{video.fixture}: saved training input differs")
    _check_saved_video_contents(saved, checked, video, group_name, feature_record)
    return saved


def _video_output_path(output_dir: Path, video_name: str) -> Path:
    return Path(output_dir) / "videos" / video_name / VIDEO_FILENAME


def _sum_count(videos: Sequence[Mapping[str, Any]], name: str) -> int:
    return sum(int(_mapping(video["counts"], "video counts")[name]) for video in videos)


def _check_combined_video_values(
    videos: Sequence[Mapping[str, Any]],
    expected_names: Sequence[str],
    expected_groups: Sequence[str],
) -> None:
    names = [
        str(_mapping(video.get("video"), "saved video identity")["fixture"])
        for video in videos
    ]
    if names != list(expected_names):
        raise ValueError("saved training video order differs")
    if [video.get("group") for video in videos] != list(expected_groups):
        raise ValueError("saved training video groups differ")
    candidate_identities: set[tuple[str, int]] = set()
    for video in videos:
        video_name = str(_mapping(video["video"], "saved video identity")["fixture"])
        raw_lists = video.get("candidate_lists")
        if not isinstance(raw_lists, list):
            raise TypeError(f"{video_name}: candidate lists must be a list")
        for raw_list in raw_lists:
            candidate_list = _mapping(raw_list, f"{video_name}: candidate list")
            raw_candidates = candidate_list.get("candidates")
            if not isinstance(raw_candidates, list) or len(raw_candidates) > 3:
                raise ValueError(f"{video_name}: candidate list size differs")
            for raw_candidate in raw_candidates:
                candidate = _mapping(raw_candidate, f"{video_name}: candidate")
                identity = (video_name, int(candidate["frame"]))
                if identity in candidate_identities:
                    raise ValueError(
                        "a candidate frame appears in more than one section"
                    )
                candidate_identities.add(identity)


def _assemble_combined_value(
    checked: CheckedTrainingScores,
    videos: Sequence[Mapping[str, Any]],
    validation_inputs: Sequence[Mapping[str, object]],
    source_commit: str,
) -> dict[str, Any]:
    """Assemble all 32 checked video values in their fixed group order."""
    expected_names = [
        video.fixture
        for group_name in GROUP_NAMES
        for video in checked.groups[group_name].scored_videos
    ]
    expected_groups = [
        group_name
        for group_name in GROUP_NAMES
        for _video in checked.groups[group_name].scored_videos
    ]
    _check_combined_video_values(videos, expected_names, expected_groups)
    group_counts = {
        group_name: {
            "videos": len(checked.groups[group_name].scored_videos),
            "kept_contacts": sum(
                int(_mapping(video["counts"], "video counts")["kept_contacts"])
                for video in videos
                if video.get("group") == group_name
            ),
            "candidate_lists": sum(
                int(_mapping(video["counts"], "video counts")["candidate_lists"])
                for video in videos
                if video.get("group") == group_name
            ),
            "candidate_entries": sum(
                int(_mapping(video["counts"], "video counts")["candidate_entries"])
                for video in videos
                if video.get("group") == group_name
            ),
        }
        for group_name in GROUP_NAMES
    }
    value: dict[str, Any] = {
        "schema": COMBINED_SCHEMA,
        "status": "complete",
        "source_commit": source_commit,
        "labels_read": False,
        "groups": list(GROUP_NAMES),
        "inputs": {
            "fixed_files": list(checked.fixed_inputs),
            "validation_files": list(validation_inputs),
        },
        "counts": {
            "videos": len(videos),
            "detected_sections": _sum_count(videos, "detected_sections"),
            "sections_without_kept_contact": _sum_count(
                videos,
                "sections_without_kept_contact",
            ),
            "kept_contacts": _sum_count(videos, "kept_contacts"),
            "candidate_lists": _sum_count(videos, "candidate_lists"),
            "candidate_entries": _sum_count(videos, "candidate_entries"),
            "earlier_candidate_entries": _sum_count(
                videos,
                "earlier_candidate_entries",
            ),
        },
        "group_counts": group_counts,
        "videos": list(videos),
    }
    if value["counts"]["kept_contacts"] != int(checked.rows["kept"].sum()):
        raise ValueError("combined saved kept-contact count differs")
    return value


def combine_training_videos(
    checked: CheckedTrainingScores,
    videos: Sequence[Mapping[str, Any]],
    validation_inputs: Sequence[Mapping[str, object]],
    output_path: Path,
    source_commit: str,
) -> dict[str, Any]:
    """Build the full value twice, then save and read it back."""
    first = _assemble_combined_value(
        checked,
        videos,
        validation_inputs,
        source_commit,
    )
    second = _assemble_combined_value(
        checked,
        videos,
        validation_inputs,
        source_commit,
    )
    if _json_bytes(first) != _json_bytes(second):
        raise ValueError("repeated combined training input build differs")
    _write_json(output_path, first)
    saved = _read_json(output_path, "saved combined training inputs")
    if _json_bytes(saved) != _json_bytes(first):
        raise ValueError("saved combined training inputs differ")
    return saved


def save_training_rally_start_inputs(
    files: InputFiles,
    score_paths: TrainingScorePaths,
    validation_paths: ValidationPaths,
    data_root: Path,
    output_dir: Path,
    source_commit: str,
    *,
    selected_video_names: Sequence[str] = (),
    resume: bool = False,
    menu_loader: Callable[..., VerifiedBaselineMenu] = load_completed_baseline_menu,
) -> Path | None:
    """Run the validation gate, then save selected or all training videos."""
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a short or full Git commit")
    source_root = str(REPO_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    destination = Path(output_dir)
    combined_path = destination / COMBINED_FILENAME
    _write_json(
        combined_path,
        {
            "schema": COMBINED_SCHEMA,
            "status": "running",
            "source_commit": source_commit,
            "labels_read": False,
        },
    )
    verified = menu_loader(
        validation_paths.menu_result,
        files.config,
        files.split,
        files.feature_record,
        validation_paths.common30_feature_record,
        files.contact_labels,
    )
    validation_inputs = check_validation_reproduction(
        verified,
        validation_paths,
        files.baseline_summary,
        data_root,
        source_commit,
    )
    checked = check_training_scores(score_paths, files, verified.raw_features)
    expected_videos = [
        video
        for group_name in GROUP_NAMES
        for video in checked.groups[group_name].scored_videos
    ]
    expected_names = [video.fixture for video in expected_videos]
    if selected_video_names:
        requested = list(selected_video_names)
        if len(set(requested)) != len(requested) or not set(requested) <= set(
            expected_names
        ):
            raise ValueError("requested training video names differ")
        videos_to_save = [
            video for video in expected_videos if video.fixture in set(requested)
        ]
    else:
        videos_to_save = expected_videos

    saved_by_name: dict[str, dict[str, Any]] = {}
    group_by_video = {
        video.fixture: group_name
        for group_name in GROUP_NAMES
        for video in checked.groups[group_name].scored_videos
    }
    for video in videos_to_save:
        output_path = _video_output_path(destination, video.fixture)
        saved_by_name[video.fixture] = save_training_video(
            checked,
            video,
            group_by_video[video.fixture],
            Path(data_root),
            output_path,
            source_commit,
            resume=resume,
        )
        print(f"saved {video.fixture}", flush=True)

    if selected_video_names:
        return None
    saved_videos = [saved_by_name[name] for name in expected_names]
    combine_training_videos(
        checked,
        saved_videos,
        validation_inputs,
        combined_path,
        source_commit,
    )
    return combined_path


def _input_files_from_args(arguments: argparse.Namespace) -> InputFiles:
    return InputFiles(
        arguments.input_list,
        arguments.groups,
        arguments.split,
        arguments.config,
        arguments.raw_feature_record,
        arguments.baseline_summary,
        arguments.chosen_baseline_result,
        arguments.contact_labels,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-feature-record", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--chosen-baseline-result", type=Path, required=True)
    parser.add_argument("--contact-labels", type=Path, required=True)
    parser.add_argument("--training-score-summary", type=Path, required=True)
    parser.add_argument("--training-score-root", type=Path, required=True)
    parser.add_argument("--menu-result", type=Path, required=True)
    parser.add_argument("--common30-feature-record", type=Path, required=True)
    parser.add_argument("--validation-rally-predictions", type=Path, required=True)
    parser.add_argument("--validation-rally-result", type=Path, required=True)
    parser.add_argument("--chosen-validation-scores", type=Path, required=True)
    parser.add_argument("--validation-candidate-summary", type=Path, required=True)
    parser.add_argument("--frozen-validation-candidates", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    output = save_training_rally_start_inputs(
        _input_files_from_args(arguments),
        TrainingScorePaths(
            arguments.training_score_summary,
            arguments.training_score_root,
        ),
        ValidationPaths(
            arguments.menu_result,
            arguments.common30_feature_record,
            arguments.validation_rally_predictions,
            arguments.validation_rally_result,
            arguments.chosen_validation_scores,
            arguments.validation_candidate_summary,
            arguments.frozen_validation_candidates,
        ),
        arguments.data_root,
        arguments.output_dir,
        arguments.source_commit,
        selected_video_names=arguments.video,
        resume=arguments.resume,
    )
    if output is not None:
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
