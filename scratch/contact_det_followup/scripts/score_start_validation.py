"""Score the frozen B4 first-contact action choice on the V labels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scratch.contact_det_followup.scripts import score_start_model as _start_model
from scratch.contact_det_followup.scripts import (
    write_start_validation_predictions as _prediction_writer,
)
from scratch.contact_det_followup.scripts.prediction_io import (
    REPO_ROOT,
    load_development_predictions,
    read_json,
)

PREDICTION_INPUT_PATH = _prediction_writer.PREDICTION_OUTPUT_PATH
OUTPUT_PATH = (
    REPO_ROOT / "scratch/contact_det_followup/results/start_model_validation.json"
)
LABEL_PATH = REPO_ROOT / "training/data/shuttleset/annotations/shots_master.csv"
EXPECTED_MODEL_ID = "shallow_hgb"
EXPECTED_CUTOFF = 0.9
EXPECTED_SCHEMA = "contact-detector-start-action-validation-predictions/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}


def _verify_source_files(payload: Mapping[str, Any]) -> None:
    raw_sources = payload.get("source_files")
    if not isinstance(raw_sources, Mapping):
        raise TypeError("frozen prediction source files must be an object")
    expected_paths = {
        "development_result": _prediction_writer.DEVELOPMENT_RESULT_PATH,
        "model_config": _prediction_writer.CONFIG_PATH,
        "training_inputs": _start_model.TRAINING_INPUT_PATH,
        "validation_input_summary": _prediction_writer.VALIDATION_INPUT_SUMMARY_PATH,
        "validation_inputs": _prediction_writer.VALIDATION_INPUT_PATH,
    }
    if set(raw_sources) != set(expected_paths):
        raise ValueError("frozen prediction source files differ")
    for name, path in expected_paths.items():
        raw_record = raw_sources[name]
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"{name}: source file record must be an object")
        if raw_record.get("path") != str(path.relative_to(REPO_ROOT)):
            raise ValueError(f"{name}: source file path differs")
        if raw_record.get("sha256") != _sha256(path):
            raise ValueError(f"{name}: source file changed")


def _score_mapping(raw_score: object) -> tuple[tuple[str, int, int, str], float]:
    if not isinstance(raw_score, Mapping):
        raise TypeError("frozen action score must be an object")
    required = {"fixture", "span_id", "frame", "action", "score"}
    if set(raw_score) != required:
        raise ValueError("frozen action score fields differ")
    fixture = raw_score["fixture"]
    action = raw_score["action"]
    if not isinstance(fixture, str) or action not in _start_model.ACTION_KINDS:
        raise ValueError("frozen action score identity differs")
    if type(raw_score["span_id"]) is not int or type(raw_score["frame"]) is not int:
        raise ValueError("frozen action score frame identity differs")
    score = raw_score["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("frozen action score must be numeric")
    score_float = float(score)
    if not 0.0 <= score_float <= 1.0:
        raise ValueError("frozen action score is outside zero to one")
    return (fixture, raw_score["span_id"], raw_score["frame"], action), score_float


def _selected_identity(raw_identity: object) -> tuple[str, int, int, str]:
    if not isinstance(raw_identity, list) or len(raw_identity) != 4:
        raise ValueError("frozen selected action identity differs")
    fixture, span_id, frame, action = raw_identity
    if (
        not isinstance(fixture, str)
        or type(span_id) is not int
        or type(frame) is not int
        or action not in _start_model.ACTION_KINDS
    ):
        raise ValueError("frozen selected action identity differs")
    return fixture, span_id, frame, action


def _rebuild_frozen_choice() -> tuple[
    Mapping[str, Any],
    tuple[Any, ...],
    dict[str, tuple[Any, ...]],
    tuple[_start_model.ActionRow, ...],
    dict[_start_model.ActionIdentity, float],
    dict[_start_model.SectionIdentity, _start_model.ActionRow],
]:
    """Load the record and independently rebuild its V scores and selection."""
    payload = read_json(PREDICTION_INPUT_PATH)
    if payload.get("schema") != EXPECTED_SCHEMA or payload.get("status") != "complete":
        raise ValueError("frozen validation prediction record is incomplete")
    if payload.get("validation_labels_read") is not False:
        raise ValueError("frozen prediction record labels_read flag differs")
    if (
        payload.get("model_id") != EXPECTED_MODEL_ID
        or float(payload.get("cutoff")) != EXPECTED_CUTOFF
        or tuple(payload.get("feature_names", ())) != _start_model.ACTION_FEATURE_NAMES
    ):
        raise ValueError("frozen validation model choice differs")
    _verify_source_files(payload)
    _prediction_writer._validate_development_choice()
    config = _prediction_writer.load_rally_start_model_config(
        _prediction_writer.CONFIG_PATH
    )
    if EXPECTED_CUTOFF not in config.selection_cutoffs:
        raise ValueError("frozen validation cut-off is not in the fixed config")

    validation_videos = _prediction_writer._load_validation_videos()
    predictions = load_development_predictions()
    expected_fixtures = tuple(
        fixture
        for fixture, group in predictions.group_by_fixture.items()
        if group == "V"
    )
    actual_fixtures = tuple(str(video["fixture"]) for video in validation_videos)
    if actual_fixtures != expected_fixtures:
        raise ValueError("frozen validation fixtures differ")
    if payload.get("validation_videos") != list(expected_fixtures):
        raise ValueError("frozen validation video coverage differs")
    spans = tuple(
        span for span in predictions.spans if span.fixture in expected_fixtures
    )
    events = {
        fixture: fixture_events
        for fixture, fixture_events in predictions.events_by_fixture.items()
        if fixture in expected_fixtures
    }
    rows = _start_model.build_candidate_rows(validation_videos, default_group="V")
    action_rows = _start_model.build_action_rows(rows)
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, list):
        raise TypeError("frozen action scores must be a list")
    scores: dict[_start_model.ActionIdentity, float] = {}
    for raw_score in raw_scores:
        identity, score = _score_mapping(raw_score)
        if identity in scores:
            raise ValueError(f"{identity}: frozen action score repeats")
        scores[identity] = score
    expected_identities = {row.identity for row in action_rows}
    if set(scores) != expected_identities:
        raise ValueError("frozen action score coverage differs")

    selections = _start_model.select_actions(
        action_rows,
        scores,
        EXPECTED_CUTOFF,
    )
    raw_selected = payload.get("selected_action_identities")
    if not isinstance(raw_selected, list):
        raise TypeError("frozen selected action identities must be a list")
    selected_identities = [_selected_identity(identity) for identity in raw_selected]
    if len(selected_identities) != len(set(selected_identities)):
        raise ValueError("frozen selected action identities repeat")
    rebuilt_identities = sorted(row.identity for row in selections.values())
    if selected_identities != rebuilt_identities:
        raise ValueError("frozen selections do not reproduce from saved scores")
    return payload, spans, events, action_rows, scores, selections


def _per_video_changes(
    baseline: frozenset[_start_model.SectionIdentity],
    revised: frozenset[_start_model.SectionIdentity],
    fixtures: tuple[str, ...],
) -> list[dict[str, int | str]]:
    repaired = revised - baseline
    broken = baseline - revised
    return [
        {
            "fixture": fixture,
            "repaired_sections": sum(identity[0] == fixture for identity in repaired),
            "broken_sections": sum(identity[0] == fixture for identity in broken),
            "net_sections": sum(identity[0] == fixture for identity in repaired)
            - sum(identity[0] == fixture for identity in broken),
        }
        for fixture in fixtures
    ]


def run_score() -> dict[str, object]:
    """Score the already-frozen V selection without changing it."""
    payload, spans, events, action_rows, scores, selections = _rebuild_frozen_choice()
    del action_rows, scores
    baseline = _start_model.apply_selected_actions(spans, events, {})
    revised = _start_model.apply_selected_actions(spans, events, selections)
    baseline_scored = _start_model.apply_whole_rally_alternation(baseline)
    revised_scored = _start_model.apply_whole_rally_alternation(revised)

    predictions = load_development_predictions()
    fixtures = tuple(
        fixture
        for fixture, group in predictions.group_by_fixture.items()
        if group == "V"
    )
    validation_videos = _prediction_writer._load_validation_videos()
    labels = _start_model.load_human_labels(
        LABEL_PATH,
        tuple(video for video in predictions.videos if video.fixture in fixtures),
    )
    fps_by_fixture = {
        video.fixture: video.fps
        for video in predictions.videos
        if video.fixture in fixtures
    }
    from scratch.contact_det_followup.scripts.score_development_sides import (
        _timed_side_counts,
    )

    tolerance_results: dict[str, dict[str, object]] = {}
    for tolerance in (5, 10):
        baseline_ids = _start_model._fully_correct_ids(
            baseline_scored,
            labels,
            fps_by_fixture,
            tolerance_at_30_fps=tolerance,
        )
        revised_ids = _start_model._fully_correct_ids(
            revised_scored,
            labels,
            fps_by_fixture,
            tolerance_at_30_fps=tolerance,
        )
        repaired = revised_ids - baseline_ids
        broken = baseline_ids - revised_ids
        baseline_contact = _timed_side_counts(
            baseline_scored.events_by_fixture,
            labels,
            fps_by_fixture,
            tolerance_at_30_fps=tolerance,
        )
        revised_contact = _timed_side_counts(
            revised_scored.events_by_fixture,
            labels,
            fps_by_fixture,
            tolerance_at_30_fps=tolerance,
        )
        tolerance_results[str(tolerance)] = {
            "baseline_fully_correct": len(baseline_ids),
            "revised_fully_correct": len(revised_ids),
            "repaired_sections": len(repaired),
            "broken_sections": len(broken),
            "net_sections": len(repaired) - len(broken),
            "per_video": _per_video_changes(baseline_ids, revised_ids, fixtures),
            "contact_and_side_f1": {
                "baseline": baseline_contact["contact_and_side_f1"],
                "revised": revised_contact["contact_and_side_f1"],
                "change": float(revised_contact["contact_and_side_f1"])
                - float(baseline_contact["contact_and_side_f1"]),
            },
            "repaired_identities": sorted(repaired),
            "broken_identities": sorted(broken),
        }
    primary = tolerance_results["5"]
    action_counts = Counter(row.action for row in selections.values())
    return {
        "schema": "contact-detector-start-action-validation/1",
        "status": "complete",
        "source_prediction_record": _file_record(PREDICTION_INPUT_PATH),
        "model_id": payload["model_id"],
        "cutoff": payload["cutoff"],
        "feature_names": payload["feature_names"],
        "validation_videos": list(fixtures),
        "baseline_fully_correct": primary["baseline_fully_correct"],
        "revised_fully_correct": primary["revised_fully_correct"],
        "repaired_sections": primary["repaired_sections"],
        "broken_sections": primary["broken_sections"],
        "net_sections": primary["net_sections"],
        "per_video": primary["per_video"],
        "number_changed": len(selections),
        "action_counts": {
            action: action_counts[action] for action in _start_model.ACTION_KINDS
        },
        "contact_and_side_f1": primary["contact_and_side_f1"],
        "repaired_identities": primary["repaired_identities"],
        "broken_identities": primary["broken_identities"],
        "by_tolerance_at_30_fps": tolerance_results,
        "validation_labels_read": True,
        "validated_candidate_video_count": len(validation_videos),
    }


def main() -> None:
    """Run and save the frozen V validation score."""
    result = run_score()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "model_id",
                    "cutoff",
                    "baseline_fully_correct",
                    "revised_fully_correct",
                    "repaired_sections",
                    "broken_sections",
                    "net_sections",
                    "number_changed",
                    "action_counts",
                    "contact_and_side_f1",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
