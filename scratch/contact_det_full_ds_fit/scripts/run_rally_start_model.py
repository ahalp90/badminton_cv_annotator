"""Choose and check the fixed rally-start contact model."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from scratch.contact_det.scripts.score_contact_rallies import RallyReference
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    DevelopmentSplit,
    VideoSpec,
    load_development_split,
    verify_accepted_development_split,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    CandidateRow,
    ContactStreams,
    HumanLabels,
    TargetAssignments,
    apply_selected_candidates,
    assign_candidate_targets,
    build_candidate_rows,
    fit_final_candidate_model,
    held_out_candidate_scores,
    model_choice_key,
    passes_result_gate,
    predict_candidate_scores,
    score_candidate_choice,
    select_candidates,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model_config import (
    RallyStartModelConfig,
    load_rally_start_model_config,
)

RESULT_SCHEMA = "rally-start-contact-model-result/1"
SCORE_SCHEMA = "rally-start-validation-candidate-scores/1"
RESULT_FILENAME = "rally_start_model_result.json.gz"
SCORE_FILENAME = "validation_candidate_scores.json.gz"
TRAINING_INPUT_SCHEMA = "contact-rally-start-training-inputs/1"
VALIDATION_INPUT_SCHEMA = "contact-rally-start-validation-inputs/1"
TRAINING_SUMMARY_SCHEMA = "contact-rally-start-training-input-summary/1"
VALIDATION_SUMMARY_SCHEMA = "contact-rally-start-validation-input-summary/1"
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class TimingLabelSet:
    """Rally timing labels and their exact video-frame identities."""

    rallies: Mapping[str, tuple[RallyReference, ...]]
    identities: frozenset[tuple[str, int]]


TimingLabelLoader = Callable[[Path, Sequence[VideoSpec]], TimingLabelSet]
SideLabelLoader = Callable[
    [Path, Sequence[VideoSpec], frozenset[tuple[str, int]]],
    Mapping[tuple[str, int], str],
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _write_json(path: Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    if destination.suffix == ".gz":
        temporary.write_bytes(gzip.compress(_json_bytes(value), mtime=0))
    else:
        temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, destination)


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    source = Path(path)
    raw = gzip.decompress(source.read_bytes()) if source.suffix == ".gz" else source.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must contain an object")
    return value


def _checked_input(
    summary_path: Path,
    input_path: Path,
    *,
    summary_schema: str,
    input_schema: str,
    expected_videos: Sequence[str],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    summary = _read_json(summary_path, "input summary")
    payload = _read_json(input_path, "saved candidate input")
    if (
        summary.get("schema") != summary_schema
        or summary.get("status") != "complete"
        or summary.get("result_file") != Path(input_path).name
        or summary.get("result_sha256") != _sha256(input_path)
        or payload.get("schema") != input_schema
        or payload.get("status") != "complete"
        or payload.get("labels_read") is not False
    ):
        raise ValueError("saved candidate input checks failed")
    raw_videos = payload.get("videos")
    if not isinstance(raw_videos, list) or any(
        not isinstance(video, Mapping) for video in raw_videos
    ):
        raise TypeError("saved candidate videos must be a list of objects")
    videos = [dict(video) for video in raw_videos]
    fixtures = []
    for video in videos:
        raw_identity = video.get("video")
        identity = raw_identity if isinstance(raw_identity, Mapping) else video
        fixture = identity.get("fixture")
        if not isinstance(fixture, str):
            raise TypeError("saved candidate video identity must be text")
        fixtures.append(fixture)
    if fixtures != list(expected_videos):
        raise ValueError("saved candidate video order differs")
    return payload, videos


def _checked_groups(
    path: Path,
    split: DevelopmentSplit,
    config: RallyStartModelConfig,
) -> dict[str, str]:
    payload = _read_json(path, "training groups")
    if payload.get("schema") != "contact-training-video-score-groups/1":
        raise ValueError("training group file version differs")
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list):
        raise TypeError("training groups must be a list")
    group_by_fixture: dict[str, str] = {}
    for expected_group, raw_group in zip(config.training_groups, raw_groups, strict=True):
        if not isinstance(raw_group, Mapping) or raw_group.get("group") != expected_group:
            raise ValueError("training group order differs")
        raw_videos = raw_group.get("videos")
        if not isinstance(raw_videos, list) or len(raw_videos) != 8:
            raise ValueError(f"group {expected_group}: video list differs")
        for raw_video in raw_videos:
            if not isinstance(raw_video, Mapping):
                raise TypeError(f"group {expected_group}: video must be an object")
            fixture = raw_video.get("fixture")
            if not isinstance(fixture, str) or fixture in group_by_fixture:
                raise ValueError("training group video identity differs")
            group_by_fixture[fixture] = expected_group
    training_names = [video.fixture for video in split.training_videos]
    if set(group_by_fixture) != set(training_names):
        raise ValueError("training group coverage differs")
    fixed_validation = payload.get("fixed_validation_videos")
    if fixed_validation != [video.fixture for video in split.validation_videos]:
        raise ValueError("fixed validation videos differ")
    return group_by_fixture


def _training_names_in_group_order(
    split: DevelopmentSplit,
    config: RallyStartModelConfig,
    group_by_fixture: Mapping[str, str],
) -> list[str]:
    names = [
        video.fixture
        for group in config.training_groups
        for video in split.training_videos
        if group_by_fixture.get(video.fixture) == group
    ]
    if len(names) != len(split.training_videos) or len(set(names)) != len(names):
        raise ValueError("training group video order differs")
    return names


def _integer(value: str, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    return result


def _normalise_side(value: str, label: str) -> str:
    if value == "Top":
        return "Top"
    if value in {"Bot", "Bottom"}:
        return "Bot"
    raise ValueError(f"{label}: player side differs")


def _allowed_video_maps(
    videos: Sequence[VideoSpec],
) -> tuple[set[int], dict[int, str]]:
    allowed_ids = {video.video_id for video in videos}
    fixture_by_id = {video.video_id: video.fixture for video in videos}
    if len(allowed_ids) != len(videos):
        raise ValueError("allowed label video identities repeat")
    return allowed_ids, fixture_by_id


def load_timing_labels(path: Path, videos: Sequence[VideoSpec]) -> TimingLabelSet:
    """Read timing labels for exactly the allowed videos."""
    allowed_ids, fixture_by_id = _allowed_video_maps(videos)
    grouped: dict[tuple[int, str, int], list[int]] = {}
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
            video_id = _integer(row[positions["vid"]], f"contact label row {row_number} video ID")
            if video_id not in allowed_ids:
                continue
            set_id = row[positions["set_id"]]
            rally_number = _integer(
                row[positions["rally"]], f"contact label row {row_number} rally"
            )
            frame = _integer(
                row[positions["frame_num"]], f"contact label row {row_number} frame"
            )
            if not set_id:
                raise ValueError(f"contact label row {row_number} set ID is empty")
            grouped.setdefault((video_id, set_id, rally_number), []).append(frame)

    if {video_id for video_id, _set_id, _rally in grouped} != allowed_ids:
        raise ValueError("contact label video coverage differs")
    rallies_by_fixture: dict[str, list[RallyReference]] = {
        video.fixture: [] for video in videos
    }
    for video_id, set_id, rally_number in sorted(grouped):
        frames = tuple(sorted(grouped[(video_id, set_id, rally_number)]))
        if not frames or len(frames) != len(set(frames)):
            raise ValueError("contact label rally frames repeat")
        fixture = fixture_by_id[video_id]
        fixture_rallies = rallies_by_fixture[fixture]
        fixture_rallies.append(
            RallyReference(
                fixture,
                len(fixture_rallies),
                f"{set_id}:{rally_number}",
                frames,
            )
        )
    rallies = MappingProxyType(
        {fixture: tuple(values) for fixture, values in rallies_by_fixture.items()}
    )
    identities = frozenset(
        (fixture, frame)
        for fixture, fixture_rallies in rallies.items()
        for rally in fixture_rallies
        for frame in rally.frames
    )
    return TimingLabelSet(rallies, identities)


def load_side_labels(
    path: Path,
    videos: Sequence[VideoSpec],
    expected_identities: frozenset[tuple[str, int]],
) -> Mapping[tuple[str, int], str]:
    """Read player sides for exactly the checked timing-label rows."""
    allowed_ids, fixture_by_id = _allowed_video_maps(videos)
    sides: dict[tuple[str, int], str] = {}
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("contact label file is empty") from error
        required = ("vid", "frame_num", "player_side")
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
            frame = _integer(
                row[positions["frame_num"]], f"contact label row {row_number} frame"
            )
            fixture = fixture_by_id[video_id]
            identity = (fixture, frame)
            if identity in sides:
                raise ValueError(f"{fixture}/{frame}: contact label repeats")
            sides[identity] = _normalise_side(
                row[positions["player_side"]], f"{fixture}/{frame}"
            )
    if set(sides) != set(expected_identities):
        raise ValueError("player-side identities differ from timing labels")
    return MappingProxyType(sides)


def load_human_labels(path: Path, videos: Sequence[VideoSpec]) -> HumanLabels:
    """Read timing, then player side, for exactly the allowed videos."""
    timing = load_timing_labels(path, videos)
    sides = load_side_labels(path, videos, timing.identities)
    return HumanLabels(
        timing.rallies,
        sides,
    )


def _load_checked_human_labels(
    path: Path,
    videos: Sequence[VideoSpec],
    expected_sha256: str,
    timing_loader: TimingLabelLoader,
    side_loader: SideLabelLoader,
) -> HumanLabels:
    timing = timing_loader(path, videos)
    if _sha256(path) != expected_sha256:
        raise ValueError("contact label file changed after timing labels were read")
    sides = side_loader(path, videos, timing.identities)
    if _sha256(path) != expected_sha256:
        raise ValueError("contact label file changed after player sides were read")
    return HumanLabels(
        timing.rallies,
        MappingProxyType(dict(sides)),
    )


def _score_rows(scores: Mapping[tuple[str, int, int], float]) -> list[dict[str, object]]:
    return [
        {"fixture": fixture, "span_id": span_id, "frame": frame, "score": score}
        for (fixture, span_id, frame), score in sorted(scores.items())
    ]


def _target_rows(
    rows: Sequence[CandidateRow],
    targets: TargetAssignments,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        target = targets.by_candidate[row.identity]
        output.append(
            {
                "fixture": row.fixture,
                "group": row.group,
                "span_id": row.span_id,
                "frame": row.frame,
                "section_status": target.section_status,
                "included_in_training": target.included_in_training,
                "rally_id": target.rally_id,
                "first_contact_frame": target.first_contact_frame,
                "timing_match": target.timing_match,
                "side_match": target.side_match,
                "correct_action": target.positive,
                "candidate_already_kept": row.kept,
            }
        )
    return output


def _stream_value(streams: ContactStreams) -> dict[str, object]:
    return {
        "spans": [
            {
                "fixture": span.fixture,
                "span_id": span.span_id,
                "start_frame": span.start_frame,
                "end_frame": span.end_frame,
                "events": [
                    {
                        "fixture": event.fixture,
                        "frame": event.frame,
                        "timing_score": event.timing_score,
                        "predicted_side": event.predicted_side,
                    }
                    for event in span.events
                ],
            }
            for span in streams.spans
        ],
        "events_by_fixture": {
            fixture: [
                {
                    "frame": event.frame,
                    "timing_score": event.timing_score,
                    "predicted_side": event.predicted_side,
                }
                for event in events
            ]
            for fixture, events in streams.events_by_fixture.items()
        },
    }


def _checked_apply(
    videos: Sequence[Mapping[str, Any]],
    selections: Mapping[tuple[str, int], CandidateRow],
    *,
    default_group: str,
) -> ContactStreams:
    first = apply_selected_candidates(videos, selections, default_group=default_group)
    second = apply_selected_candidates(videos, selections, default_group=default_group)
    if _json_bytes(_stream_value(first)) != _json_bytes(_stream_value(second)):
        raise ValueError("repeated contact-stream build differs")
    return first


def _verify_action_counts(
    detail_path: Path,
    choices: Sequence[Mapping[str, Any]],
) -> None:
    detail = _read_json(detail_path, "candidate detail")
    raw_candidates = detail.get("candidates")
    if not isinstance(raw_candidates, list):
        raise TypeError("candidate detail rows must be a list")
    by_identity: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            raise TypeError("candidate detail row must be an object")
        identity = (
            str(raw_candidate.get("fixture")),
            int(raw_candidate["span_id"]),
            int(raw_candidate["frame"]),
        )
        if identity in by_identity:
            raise ValueError("candidate detail identities repeat")
        by_identity[identity] = raw_candidate
    for choice in choices:
        metrics = choice.get("metrics")
        if not isinstance(metrics, Mapping):
            raise TypeError("choice metrics must be an object")
        raw_selected = metrics.get("selected_candidate_identities")
        if not isinstance(raw_selected, list):
            raise TypeError("selected candidate identities must be a list")
        selected = [
            by_identity[(str(identity[0]), int(identity[1]), int(identity[2]))]
            for identity in raw_selected
        ]
        correct = [row for row in selected if row.get("correct_action") is True]
        recovered = {
            (str(row["fixture"]), int(row["span_id"])) for row in correct
        }
        expected = (
            len(selected),
            len(correct),
            sum(row.get("candidate_already_kept") is False for row in selected),
            len(recovered),
        )
        actual = (
            metrics.get("selected_actions"),
            metrics.get("correct_additions"),
            metrics.get("newly_added_contacts"),
            metrics.get("recovered_sections"),
        )
        if actual != expected:
            raise ValueError("saved candidate detail does not reproduce action counts")


def _result_file_record(path: Path) -> dict[str, str]:
    return {"filename": Path(path).name, "sha256": _sha256(path)}


def run_rally_start_model(
    config_path: Path,
    split_path: Path,
    group_path: Path,
    training_summary_path: Path,
    training_input_path: Path,
    validation_summary_path: Path,
    validation_input_path: Path,
    shots_master_path: Path,
    output_dir: Path,
    source_commit: str,
    *,
    timing_label_loader: TimingLabelLoader = load_timing_labels,
    side_label_loader: SideLabelLoader = load_side_labels,
) -> Path:
    """Choose on training videos, freeze validation scores, then check validation."""
    result_root = Path(output_dir)
    result_path = result_root / RESULT_FILENAME
    score_path = result_root / SCORE_FILENAME
    _write_json(
        result_path,
        {"schema": RESULT_SCHEMA, "status": "running", "source_commit": source_commit},
    )
    try:
        if SOURCE_COMMIT.fullmatch(source_commit) is None:
            raise ValueError("source commit must be a short or full Git commit")
        config = load_rally_start_model_config(config_path)
        split = load_development_split(split_path)
        verify_accepted_development_split(split)
        group_by_fixture = _checked_groups(group_path, split, config)
        training_names = _training_names_in_group_order(
            split,
            config,
            group_by_fixture,
        )
        validation_names = [video.fixture for video in split.validation_videos]
        _training_payload, training_videos = _checked_input(
            training_summary_path,
            training_input_path,
            summary_schema=TRAINING_SUMMARY_SCHEMA,
            input_schema=TRAINING_INPUT_SCHEMA,
            expected_videos=training_names,
        )
        _validation_payload, validation_videos = _checked_input(
            validation_summary_path,
            validation_input_path,
            summary_schema=VALIDATION_SUMMARY_SCHEMA,
            input_schema=VALIDATION_INPUT_SCHEMA,
            expected_videos=validation_names,
        )
        for video in training_videos:
            identity = video.get("video")
            fixture = identity.get("fixture") if isinstance(identity, Mapping) else None
            group = video.get("group")
            if group != group_by_fixture.get(str(fixture)):
                raise ValueError("saved training video group differs")
            expected_first_model_training = {
                name
                for name in training_names
                if group_by_fixture[name] != group
            }
            raw_first_model_training = video.get("model_training_videos")
            if (
                not isinstance(raw_first_model_training, list)
                or set(raw_first_model_training) != expected_first_model_training
            ):
                raise ValueError("first contact model training videos differ")

        label_record = _result_file_record(shots_master_path)
        shared: dict[str, object] = {
            "schema": RESULT_SCHEMA,
            "source_commit": source_commit,
            "inputs": {
                "model_config": _result_file_record(config_path),
                "development_split": _result_file_record(split_path),
                "training_groups": _result_file_record(group_path),
                "training_input_summary": _result_file_record(training_summary_path),
                "training_inputs": _result_file_record(training_input_path),
                "validation_input_summary": _result_file_record(validation_summary_path),
                "validation_inputs": _result_file_record(validation_input_path),
                "contact_labels": label_record,
            },
        }
        training_rows = build_candidate_rows(training_videos, default_group="training")
        training_labels = _load_checked_human_labels(
            shots_master_path,
            split.training_videos,
            label_record["sha256"],
            timing_label_loader,
            side_label_loader,
        )
        training_targets = assign_candidate_targets(
            training_rows,
            training_videos,
            training_labels,
            default_group="training",
        )
        held_out_scores = held_out_candidate_scores(training_rows, training_targets, config)
        training_baseline = _checked_apply(
            training_videos, {}, default_group="training"
        )
        training_fps = {video.fixture: video.fps for video in split.training_videos}
        choices: list[dict[str, object]] = []
        passing: list[
            tuple[tuple[int, int, float, int], float, str, dict[str, object]]
        ] = []
        for spec in config.models:
            scores = held_out_scores[spec.model_id]
            for cutoff in config.selection_cutoffs:
                selections = select_candidates(training_rows, scores, cutoff)
                alternative = _checked_apply(
                    training_videos, selections, default_group="training"
                )
                metrics = score_candidate_choice(
                    training_baseline,
                    alternative,
                    training_labels,
                    training_fps,
                    training_targets,
                    selections,
                )
                passed = passes_result_gate(metrics, config.training_gate, validation=False)
                choice = {
                    "model_id": spec.model_id,
                    "cutoff": cutoff,
                    "passed": passed,
                    "metrics": metrics,
                }
                choices.append(choice)
                if passed:
                    passing.append(
                        (
                            model_choice_key(spec.model_id, metrics),
                            cutoff,
                            spec.model_id,
                            choice,
                        )
                    )

        training_score_file = result_root / "training_candidate_scores.json.gz"
        _write_json(
            training_score_file,
            {
                "schema": "rally-start-training-candidate-scores/1",
                "status": "complete",
                "source_commit": source_commit,
                "models": {
                    model_id: _score_rows(scores)
                    for model_id, scores in held_out_scores.items()
                },
                "candidates": _target_rows(training_rows, training_targets),
            },
        )
        _verify_action_counts(training_score_file, choices)
        training_result = {
            "choices": choices,
            "held_out_scores": _result_file_record(training_score_file),
        }
        if not passing:
            _write_json(
                result_path,
                {
                    **shared,
                    "status": "complete",
                    "outcome": "stopped_at_training_gate",
                    "training": training_result,
                    "validation_labels_read": False,
                },
            )
            return result_path

        _key, chosen_cutoff, chosen_model_id, chosen_training = max(passing)
        chosen_spec = next(
            spec for spec in config.models if spec.model_id == chosen_model_id
        )
        final_training_model = fit_final_candidate_model(
            chosen_spec, training_rows, training_targets
        )
        validation_rows = build_candidate_rows(validation_videos, default_group="V")
        validation_scores = predict_candidate_scores(final_training_model, validation_rows)
        _write_json(
            score_path,
            {
                "schema": SCORE_SCHEMA,
                "status": "complete",
                "source_commit": source_commit,
                "model_id": chosen_model_id,
                "cutoff": chosen_cutoff,
                "config": _result_file_record(config_path),
                "training_inputs": _result_file_record(training_input_path),
                "validation_inputs": _result_file_record(validation_input_path),
                "scores": _score_rows(validation_scores),
            },
        )

        validation_labels = _load_checked_human_labels(
            shots_master_path,
            split.validation_videos,
            label_record["sha256"],
            timing_label_loader,
            side_label_loader,
        )
        validation_targets = assign_candidate_targets(
            validation_rows,
            validation_videos,
            validation_labels,
            default_group="V",
        )
        validation_baseline = _checked_apply(
            validation_videos, {}, default_group="V"
        )
        validation_selections = select_candidates(
            validation_rows, validation_scores, chosen_cutoff
        )
        validation_alternative = _checked_apply(
            validation_videos, validation_selections, default_group="V"
        )
        validation_metrics = score_candidate_choice(
            validation_baseline,
            validation_alternative,
            validation_labels,
            {video.fixture: video.fps for video in split.validation_videos},
            validation_targets,
            validation_selections,
        )
        validation_passed = passes_result_gate(
            validation_metrics, config.validation_gate, validation=True
        )
        validation_detail_file = result_root / "validation_candidate_details.json.gz"
        _write_json(
            validation_detail_file,
            {
                "schema": "rally-start-validation-candidate-details/1",
                "status": "complete",
                "source_commit": source_commit,
                "candidates": _target_rows(validation_rows, validation_targets),
            },
        )
        _verify_action_counts(
            validation_detail_file,
            [{"metrics": validation_metrics}],
        )
        _write_json(
            result_path,
            {
                **shared,
                "status": "complete",
                "outcome": "passed_validation" if validation_passed else "failed_validation",
                "training": {
                    **training_result,
                    "chosen": {
                        "model_id": chosen_model_id,
                        "cutoff": chosen_cutoff,
                        "metrics": chosen_training["metrics"],
                    },
                },
                "validation_candidate_scores": _result_file_record(score_path),
                "validation_candidate_details": _result_file_record(
                    validation_detail_file
                ),
                "validation_labels_read": True,
                "validation": {
                    "passed": validation_passed,
                    "metrics": validation_metrics,
                },
            },
        )
        return result_path
    except Exception as error:
        _write_json(
            result_path,
            {
                "schema": RESULT_SCHEMA,
                "status": "failed",
                "source_commit": source_commit,
                "error_type": type(error).__name__,
            },
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--training-inputs", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--validation-inputs", type=Path, required=True)
    parser.add_argument("--shots-master", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    result_path = run_rally_start_model(
        arguments.config,
        arguments.split,
        arguments.groups,
        arguments.training_summary,
        arguments.training_inputs,
        arguments.validation_summary,
        arguments.validation_inputs,
        arguments.shots_master,
        arguments.output_dir,
        arguments.source_commit,
    )
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
