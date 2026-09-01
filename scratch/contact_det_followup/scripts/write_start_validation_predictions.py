"""Freeze the B4 first-contact action choice on the label-free V inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scratch.contact_det_followup.scripts import score_start_model as _start_model
from scratch.contact_det_followup.scripts.prediction_io import (
    REPO_ROOT,
    load_development_predictions,
    read_json,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    build_candidate_rows,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model_config import (
    load_rally_start_model_config,
)

DEVELOPMENT_RESULT_PATH = (
    REPO_ROOT / "scratch/contact_det_followup/results/start_model_development.json"
)
VALIDATION_INPUT_PATH = (
    REPO_ROOT
    / "scratch/contact_det_full_ds_fit/raw/validation_rally_start_inputs/rally_start_validation_inputs.json.gz"
)
VALIDATION_INPUT_SUMMARY_PATH = (
    REPO_ROOT / "scratch/contact_det_full_ds_fit/records/validation_rally_start_input_summary.json"
)
CONFIG_PATH = REPO_ROOT / "scratch/contact_det_full_ds_fit/records/rally_start_model_runs.json"
PREDICTION_OUTPUT_PATH = (
    REPO_ROOT / "scratch/contact_det_followup/results/start_model_validation_predictions.json"
)
EXPECTED_MODEL_ID = "shallow_hgb"
EXPECTED_CUTOFF = 0.9
EXPECTED_VALIDATION_SCHEMA = "contact-rally-start-validation-inputs/1"
EXPECTED_SUMMARY_SCHEMA = "contact-rally-start-validation-input-summary/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}


def _validate_development_choice() -> Mapping[str, Any]:
    payload = read_json(DEVELOPMENT_RESULT_PATH)
    if payload.get("schema") != "contact-detector-start-action-model/1":
        raise ValueError("development result schema differs")
    if payload.get("status") != "complete":
        raise ValueError("development result is not complete")
    if tuple(payload.get("feature_names", ())) != _start_model.ACTION_FEATURE_NAMES:
        raise ValueError("development action feature names differ")
    chosen = payload.get("chosen")
    if not isinstance(chosen, Mapping):
        raise TypeError("development result chosen configuration must be an object")
    if (
        chosen.get("model_id") != EXPECTED_MODEL_ID
        or float(chosen.get("cutoff")) != EXPECTED_CUTOFF
    ):
        raise ValueError("development result is not the frozen B4 choice")
    if payload.get("development_gate") != {
        "maximum_breaks_per_repair": 0.2,
        "minimum_net_sections": 20,
    }:
        raise ValueError("development result gate differs")
    repaired = int(chosen["repaired_sections"])
    broken = int(chosen["broken_sections"])
    net = int(chosen["net_sections"])
    if net < 20 or 5 * broken > repaired:
        raise ValueError("development B4 choice does not pass its gate")
    return payload


def _load_validation_videos() -> tuple[Mapping[str, Any], ...]:
    """Load and verify V candidate inputs without opening any label file."""
    summary = read_json(VALIDATION_INPUT_SUMMARY_PATH)
    payload = read_json(VALIDATION_INPUT_PATH)
    if (
        summary.get("schema") != EXPECTED_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("result_file") != VALIDATION_INPUT_PATH.name
        or summary.get("result_sha256") != _sha256(VALIDATION_INPUT_PATH)
        or payload.get("schema") != EXPECTED_VALIDATION_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("labels_read") is not False
    ):
        raise ValueError("saved validation candidate input checks failed")
    expected_fixtures = payload.get("validation_videos")
    raw_videos = payload.get("videos")
    if not isinstance(expected_fixtures, list) or not isinstance(raw_videos, list):
        raise TypeError("saved validation candidate videos must be lists")
    if any(not isinstance(fixture, str) for fixture in expected_fixtures):
        raise TypeError("saved validation video identities must be text")
    videos: list[Mapping[str, Any]] = []
    fixtures: list[str] = []
    for raw_video in raw_videos:
        if not isinstance(raw_video, Mapping):
            raise TypeError("saved validation video must be an object")
        if raw_video.get("labels_read") is not False:
            raise ValueError("saved validation video labels_read flag differs")
        fixture = raw_video.get("fixture")
        if not isinstance(fixture, str):
            raise TypeError("saved validation video fixture must be text")
        videos.append(raw_video)
        fixtures.append(fixture)
    if fixtures != expected_fixtures:
        raise ValueError("saved validation video order differs")
    return tuple(videos)


def _score_rows(scores: Mapping[tuple[str, int, int, str], float]) -> list[dict[str, object]]:
    return [
        {
            "fixture": fixture,
            "span_id": span_id,
            "frame": frame,
            "action": action,
            "score": score,
        }
        for (fixture, span_id, frame, action), score in sorted(scores.items())
    ]


def run_prediction() -> dict[str, object]:
    """Fit B4 on A-D and freeze its label-free V action scores and selection."""
    development = _validate_development_choice()
    config = load_rally_start_model_config(CONFIG_PATH)
    chosen_spec = next(
        (spec for spec in config.models if spec.model_id == EXPECTED_MODEL_ID),
        None,
    )
    if chosen_spec is None or EXPECTED_CUTOFF not in config.selection_cutoffs:
        raise ValueError("fixed B4 model configuration differs")

    predictions = load_development_predictions()
    training_fixtures = {
        fixture
        for fixture, group in predictions.group_by_fixture.items()
        if group in _start_model.GROUPS
    }
    training_videos = tuple(
        video for video in predictions.videos if video.fixture in training_fixtures
    )
    training_spans = tuple(
        span for span in predictions.spans if span.fixture in training_fixtures
    )
    training_events = {
        fixture: events
        for fixture, events in predictions.events_by_fixture.items()
        if fixture in training_fixtures
    }
    training_saved_videos = _start_model._candidate_videos()
    training_rows = build_candidate_rows(training_saved_videos, default_group="V")
    training_action_rows = _start_model.build_action_rows(training_rows)
    training_labels = _start_model.load_human_labels(
        _start_model.LABEL_PATH,
        training_videos,
    )
    fps_by_fixture = {video.fixture: video.fps for video in training_videos}
    targets = _start_model.assign_action_targets(
        training_action_rows,
        training_spans,
        training_events,
        training_saved_videos,
        training_labels,
        fps_by_fixture,
        default_group="V",
    )
    model = _start_model.fit_action_model(
        chosen_spec,
        training_action_rows,
        targets,
    )

    validation_saved_videos = _load_validation_videos()
    validation_fixtures = tuple(str(video["fixture"]) for video in validation_saved_videos)
    if validation_fixtures != tuple(
        fixture
        for fixture, group in predictions.group_by_fixture.items()
        if group == "V"
    ):
        raise ValueError("saved validation fixtures differ from frozen predictions")
    validation_rows = build_candidate_rows(validation_saved_videos, default_group="V")
    validation_action_rows = _start_model.build_action_rows(validation_rows)
    validation_scores = _start_model.predict_action_scores(
        model,
        validation_action_rows,
    )
    selections = _start_model.select_actions(
        validation_action_rows,
        validation_scores,
        EXPECTED_CUTOFF,
    )
    selected_identities = sorted(row.identity for row in selections.values())
    return {
        "schema": "contact-detector-start-action-validation-predictions/1",
        "status": "complete",
        "model_id": EXPECTED_MODEL_ID,
        "cutoff": EXPECTED_CUTOFF,
        "feature_names": list(_start_model.ACTION_FEATURE_NAMES),
        "source_files": {
            "development_result": _file_record(DEVELOPMENT_RESULT_PATH),
            "model_config": _file_record(CONFIG_PATH),
            "training_inputs": _file_record(_start_model.TRAINING_INPUT_PATH),
            "validation_input_summary": _file_record(VALIDATION_INPUT_SUMMARY_PATH),
            "validation_inputs": _file_record(VALIDATION_INPUT_PATH),
        },
        "development_choice": {
            "model_id": development["chosen"]["model_id"],
            "cutoff": development["chosen"]["cutoff"],
        },
        "validation_videos": list(validation_fixtures),
        "scores": _score_rows(validation_scores),
        "selected_action_identities": [list(identity) for identity in selected_identities],
        "validation_labels_read": False,
    }


def main() -> None:
    """Run and save the frozen V prediction record."""
    payload = run_prediction()
    PREDICTION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTION_OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model_id": payload["model_id"],
                "cutoff": payload["cutoff"],
                "score_count": len(payload["scores"]),
                "selected_action_count": len(payload["selected_action_identities"]),
                "validation_labels_read": payload["validation_labels_read"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
