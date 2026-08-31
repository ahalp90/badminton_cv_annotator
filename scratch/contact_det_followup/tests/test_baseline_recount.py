"""Checks for the shared record loader and baseline recount."""

import json

import pytest

from scratch.contact_det_followup.scripts.prediction_io import (
    DEFAULT_PREDICTIONS,
    load_frozen_test_predictions,
    read_json,
)
from scratch.contact_det_followup.scripts.recount_baseline import recount


def test_recount_matches_released_headlines() -> None:
    result = recount()

    assert result["metrics"] == {
        "predicted_contacts": 39_994,
        "labelled_contacts": 38_218,
        "matched_contacts": 32_243,
        "timing_precision": 0.806_195_929_389_408_4,
        "timing_recall": 0.843_660_055_471_243_9,
        "timing_f1": 0.824_502_633_866_925_8,
        "correct_player_sides": 29_620,
        "answered_player_sides": 32_188,
        "player_side_accuracy": 0.920_218_715_049_086_6,
        "predicted_sections": 3_982,
        "old_fully_correct_sections": 493,
        "strict_fully_correct_sections": 483,
        "strict_fully_correct_precision": 483 / 3_982,
    }


def test_prediction_loader_rejects_another_video_set(tmp_path) -> None:
    payload = read_json(DEFAULT_PREDICTIONS)
    payload["video_ids"][0] = 999
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed 47-video test set"):
        load_frozen_test_predictions(path)
