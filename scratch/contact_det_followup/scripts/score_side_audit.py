"""Measure whole-rally side headroom and the fixed simple vote."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

from scratch.contact_det_followup.scripts.prediction_io import (
    REPO_ROOT,
    FrozenPredictionPack,
    load_frozen_test_predictions,
    read_json,
)
from scratch.contact_det_followup.scripts.score_followup import (
    SavedLabels,
    clean_section_ids,
    fully_correct_ids,
    load_saved_test_labels,
    score_streams,
)
from scratch.contact_det_followup.scripts.side_rules import (
    apply_side_decisions,
    side_decisions_from_payload,
)

DECISION_PATH = (
    REPO_ROOT / "scratch/contact_det_followup/results/simple_side_decisions.json.gz"
)
OUTPUT_PATH = REPO_ROOT / "scratch/contact_det_followup/results/side_audit.json"
CONFIG_PATH = REPO_ROOT / "scratch/contact_det_followup/configs/side_rule.json"


@dataclass(frozen=True)
class CeilingCounts:
    """Best-case counts for the two alternating player patterns."""

    wrong_side_sections: int
    unanswered_side_sections: int
    repairable_wrong_side_sections: int
    repairable_unanswered_sections: int
    strict_repairable_wrong_side_sections: int
    strict_repairable_unanswered_sections: int
    strict_repairable_side_sections: int


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_decision_record(
    decision_payload: Mapping[str, object],
    config: Mapping[str, object],
    prediction_source_commit: str,
    section_count: int,
) -> None:
    """Check that the saved decisions came from the chosen rule and contact stream."""
    expected_config_path = str(CONFIG_PATH.relative_to(REPO_ROOT))
    if (
        decision_payload.get("status") != "complete"
        or decision_payload.get("labels_read") is not False
        or decision_payload.get("config") != expected_config_path
        or decision_payload.get("prediction_source_commit") != prediction_source_commit
        or decision_payload.get("sections_seen") != section_count
        or decision_payload.get("rule") != config.get("rule")
        or decision_payload.get("minimum_vote_gap") != config.get("minimum_vote_gap")
        or config.get("schema") != "contact-detector-side-rule/1"
    ):
        raise ValueError("Side decisions do not match the chosen rule and predictions")


def _section_outcomes(
    result: Mapping[str, object],
    tolerance: int,
) -> dict[tuple[str, int], str]:
    whole = result["whole_rallies"]
    rows = whole["by_tolerance"][str(tolerance)]["sections"]
    return {
        (str(row["fixture"]), int(row["span_id"])): str(row["outcome"]) for row in rows
    }


def _human_sides(
    saved_labels: SavedLabels,
) -> dict[tuple[str, str], tuple[str | None, ...]]:
    return {
        (fixture, f"{rally.set_id}:{rally.rally}"): tuple(
            contact.side for contact in rally.contacts
        )
        for fixture, rallies in saved_labels.labels.rallies_by_fixture.items()
        for rally in rallies
    }


def _rally_ids(
    result: Mapping[str, object],
    tolerance: int,
) -> dict[tuple[str, int], str]:
    rows = result["whole_rallies"]["by_tolerance"][str(tolerance)]["sections"]
    return {
        (str(row["fixture"]), int(row["span_id"])): str(row["rally_id"]) for row in rows
    }


def _alternates(sides: tuple[str | None, ...]) -> bool:
    return (
        bool(sides)
        and all(side is not None for side in sides)
        and all(left != right for left, right in pairwise(sides))
    )


def side_pattern_ceiling(
    baseline_result: Mapping[str, object],
    saved_labels: SavedLabels,
    strict_ids: set[tuple[str, int]],
    tolerance: int = 5,
) -> CeilingCounts:
    """Count side-only failures that either alternating pattern can repair."""
    outcomes = _section_outcomes(baseline_result, tolerance)
    rally_ids = _rally_ids(baseline_result, tolerance)
    human_sides = _human_sides(saved_labels)
    wrong = {
        identity
        for identity, outcome in outcomes.items()
        if outcome == "wrong_predicted_side"
    }
    unanswered = {
        identity
        for identity, outcome in outcomes.items()
        if outcome == "predicted_side_unanswered"
    }
    repairable_wrong = {
        identity
        for identity in wrong
        if _alternates(human_sides[(identity[0], rally_ids[identity])])
    }
    repairable_unanswered = {
        identity
        for identity in unanswered
        if _alternates(human_sides[(identity[0], rally_ids[identity])])
    }
    strict_repairable_wrong = repairable_wrong & strict_ids
    strict_repairable_unanswered = repairable_unanswered & strict_ids
    return CeilingCounts(
        wrong_side_sections=len(wrong),
        unanswered_side_sections=len(unanswered),
        repairable_wrong_side_sections=len(repairable_wrong),
        repairable_unanswered_sections=len(repairable_unanswered),
        strict_repairable_wrong_side_sections=len(strict_repairable_wrong),
        strict_repairable_unanswered_sections=len(strict_repairable_unanswered),
        strict_repairable_side_sections=len(
            strict_repairable_wrong | strict_repairable_unanswered
        ),
    )


def _strict_correct(
    result: Mapping[str, object],
    strict_ids: set[tuple[str, int]],
    tolerance: int = 5,
) -> set[tuple[str, int]]:
    return fully_correct_ids(result, tolerance=tolerance) & strict_ids


def _per_video_changes(
    repaired: set[tuple[str, int]],
    broken: set[tuple[str, int]],
) -> list[dict[str, object]]:
    repaired_counts = Counter(fixture for fixture, _span_id in repaired)
    broken_counts = Counter(fixture for fixture, _span_id in broken)
    return [
        {
            "fixture": fixture,
            "repaired_sections": repaired_counts[fixture],
            "broken_sections": broken_counts[fixture],
            "net_sections": repaired_counts[fixture] - broken_counts[fixture],
        }
        for fixture in sorted(repaired_counts.keys() | broken_counts.keys(), key=int)
    ]


def _tolerance_payload(
    tolerance: int,
    baseline: Mapping[str, object],
    revised: Mapping[str, object],
    predictions: FrozenPredictionPack,
    saved_labels: SavedLabels,
    strict_ids: set[tuple[str, int]],
    decision_payload: Mapping[str, object],
) -> dict[str, object]:
    baseline_correct = _strict_correct(baseline, strict_ids, tolerance)
    revised_correct = _strict_correct(revised, strict_ids, tolerance)
    repaired = revised_correct - baseline_correct
    broken = baseline_correct - revised_correct
    baseline_side = baseline["player_side"][str(tolerance)]["total"]
    revised_side = revised["player_side"][str(tolerance)]["total"]
    timing = baseline["contact_timing"][str(tolerance)]["total"]
    contact_and_side_denominator = int(timing["labelled_contacts"]) + int(
        timing["predicted_contacts"]
    )
    span_count = len(predictions.spans)
    return {
        "tolerance_at_30_fps": tolerance,
        "ceiling": side_pattern_ceiling(
            baseline,
            saved_labels,
            strict_ids,
            tolerance,
        ).__dict__,
        "simple_vote": {
            "sections_changed": int(decision_payload["sections_changed"]),
            "contacts_changed": int(decision_payload["contacts_changed"]),
            "baseline_strict_fully_correct": len(baseline_correct),
            "revised_strict_fully_correct": len(revised_correct),
            "repaired_sections": len(repaired),
            "broken_sections": len(broken),
            "net_sections": len(repaired) - len(broken),
            "baseline_full_output_precision": len(baseline_correct) / span_count,
            "revised_full_output_precision": len(revised_correct) / span_count,
            "baseline_side_accuracy": float(
                baseline_side["accuracy_when_both_answered"]
            ),
            "revised_side_accuracy": float(revised_side["accuracy_when_both_answered"]),
            "baseline_contact_and_side_f1": 2
            * int(baseline_side["correct_player_sides"])
            / contact_and_side_denominator,
            "revised_contact_and_side_f1": 2
            * int(revised_side["correct_player_sides"])
            / contact_and_side_denominator,
            "per_video": _per_video_changes(repaired, broken),
            "repaired_identities": sorted(repaired),
            "broken_identities": sorted(broken),
        },
    }


def run_audit() -> dict[str, object]:
    """Score the ceiling and saved label-free side decisions."""
    predictions = load_frozen_test_predictions()
    saved_labels = load_saved_test_labels()
    baseline = score_streams(
        saved_labels.labels, predictions.spans, predictions.events_by_fixture
    )
    strict_ids = clean_section_ids(predictions, saved_labels)
    decision_payload = read_json(DECISION_PATH)
    config = read_json(CONFIG_PATH)
    validate_decision_record(
        decision_payload,
        config,
        predictions.source_commit,
        len(predictions.spans),
    )
    decisions = side_decisions_from_payload(
        decision_payload,
        "contact-detector-side-decisions/1",
    )
    revised_spans, revised_events = apply_side_decisions(
        predictions.spans,
        predictions.events_by_fixture,
        decisions,
    )
    revised = score_streams(saved_labels.labels, revised_spans, revised_events)

    tolerance_results = {
        str(tolerance): _tolerance_payload(
            tolerance,
            baseline,
            revised,
            predictions,
            saved_labels,
            strict_ids,
            decision_payload,
        )
        for tolerance in (5, 10)
    }
    primary = tolerance_results["5"]
    return {
        "schema": "contact-detector-side-audit/1",
        "run_id": "simple-alternation-vote",
        "repository_commit": _repository_commit(),
        "result_type": {
            "ceiling": "Best-case check that uses labels",
            "simple_vote": "Fixed label-free rule scored once on the 47-video test set",
        },
        "labels_used": "Saved 47-video clean labels, for scoring only",
        "when_rule_acts": (
            "One alternating pattern beats the other by at least "
            f"{decision_payload['minimum_vote_gap']} vote."
        ),
        "ceiling": primary["ceiling"],
        "simple_vote": primary["simple_vote"],
        "results_by_tolerance_at_30_fps": tolerance_results,
    }


def main() -> None:
    """Write the player-side audit result."""
    payload = run_audit()
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"ceiling": payload["ceiling"], "simple_vote": payload["simple_vote"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
