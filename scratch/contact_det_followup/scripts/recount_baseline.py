"""Recount the main 47-video baseline from the released record pack."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from scratch.contact_det_followup.scripts.prediction_io import (
    REPO_ROOT,
    load_frozen_test_predictions,
)
from scratch.contact_det_followup.scripts.score_followup import (
    clean_section_ids,
    fully_correct_ids,
    load_saved_test_labels,
    score_streams,
)

OUTPUT_PATH = REPO_ROOT / "scratch/contact_det_followup/results/baseline_recount.json"
EXPECTED = {
    "predicted_contacts": 39_994,
    "labelled_contacts": 38_218,
    "matched_contacts": 32_243,
    "timing_f1": 0.824_502_633_866_925_8,
    "correct_player_sides": 29_620,
    "answered_player_sides": 32_188,
    "old_fully_correct_sections": 493,
    "strict_fully_correct_sections": 483,
    "predicted_sections": 3_982,
}


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def recount() -> dict[str, Any]:
    """Load the saved rows once and return the headline counts."""
    predictions = load_frozen_test_predictions()
    saved_labels = load_saved_test_labels()
    if set(predictions.events_by_fixture) != set(
        saved_labels.labels.rallies_by_fixture
    ):
        raise ValueError("Prediction and label fixtures differ")
    result = score_streams(
        saved_labels.labels,
        predictions.spans,
        predictions.events_by_fixture,
    )
    strict_ids = clean_section_ids(predictions, saved_labels)
    metrics_by_tolerance: dict[str, dict[str, int | float]] = {}
    for tolerance in (5, 10):
        timing = result["contact_timing"][str(tolerance)]["total"]
        player_side = result["player_side"][str(tolerance)]["total"]
        old_correct = fully_correct_ids(result, tolerance=tolerance)
        strict_correct = old_correct & strict_ids
        metrics_by_tolerance[str(tolerance)] = {
            "predicted_contacts": int(timing["predicted_contacts"]),
            "labelled_contacts": int(timing["labelled_contacts"]),
            "matched_contacts": int(timing["matched_contacts"]),
            "timing_precision": float(timing["precision"]),
            "timing_recall": float(timing["recall"]),
            "timing_f1": float(timing["f1"]),
            "correct_player_sides": int(player_side["correct_player_sides"]),
            "answered_player_sides": int(player_side["predicted_side_answers"]),
            "player_side_accuracy": float(player_side["accuracy_when_both_answered"]),
            "predicted_sections": len(predictions.spans),
            "old_fully_correct_sections": len(old_correct),
            "strict_fully_correct_sections": len(strict_correct),
            "strict_fully_correct_precision": len(strict_correct)
            / len(predictions.spans),
        }
    metrics = metrics_by_tolerance["5"]
    differences = {
        name: (metrics[name], expected)
        for name, expected in EXPECTED.items()
        if metrics[name] != expected
    }
    if differences:
        raise AssertionError(f"Saved baseline recount differs: {differences}")
    return {
        "schema": "contact-detector-followup-baseline/1",
        "run_id": "baseline-recount",
        "repository_commit": _repository_commit(),
        "prediction_source_commit": predictions.source_commit,
        "command": "python -m scratch.contact_det_followup.scripts.recount_baseline",
        "inputs": {
            "predictions": str(predictions.path.relative_to(REPO_ROOT)),
            "labels": str(saved_labels.path.relative_to(REPO_ROOT)),
        },
        "metrics": metrics,
        "metrics_by_tolerance_at_30_fps": metrics_by_tolerance,
    }


def main() -> None:
    """Write the compact baseline recount."""
    payload = recount()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
