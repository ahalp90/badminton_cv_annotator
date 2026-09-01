"""Score the fixed side vote on the 40 out-of-fold development videos."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent
from scratch.contact_det_followup.scripts.prediction_io import (
    REPO_ROOT,
    DevelopmentPredictionPack,
    load_development_predictions,
    read_json,
)
from scratch.contact_det_followup.scripts.side_rules import (
    SideDecision,
    apply_side_decisions,
    side_decisions_from_payload,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    HumanLabels,
    _fully_correct,
    scale_base30_frames,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    _match_contacts,
)

LABEL_PATH = REPO_ROOT / "training/data/shuttleset/annotations/shots_master.csv"
DECISION_PATH = (
    REPO_ROOT
    / "scratch/contact_det_followup/results/simple_side_decisions_development.json.gz"
)
OUTPUT_PATH = REPO_ROOT / "scratch/contact_det_followup/results/side_development.json"
CONFIG_PATH = REPO_ROOT / "scratch/contact_det_followup/configs/side_rule.json"


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _changes_by_group(
    repaired: set[tuple[str, int]],
    broken: set[tuple[str, int]],
    group_by_fixture: Mapping[str, str],
) -> list[dict[str, int | str]]:
    repaired_counts = Counter(group_by_fixture[fixture] for fixture, _span in repaired)
    broken_counts = Counter(group_by_fixture[fixture] for fixture, _span in broken)
    return [
        {
            "group": group,
            "repaired_sections": repaired_counts[group],
            "broken_sections": broken_counts[group],
            "net_sections": repaired_counts[group] - broken_counts[group],
        }
        for group in ("A", "B", "C", "D", "V")
    ]


def _changes_by_video(
    repaired: set[tuple[str, int]],
    broken: set[tuple[str, int]],
) -> list[dict[str, int | str]]:
    repaired_counts = Counter(fixture for fixture, _span in repaired)
    broken_counts = Counter(fixture for fixture, _span in broken)
    fixtures = sorted(repaired_counts.keys() | broken_counts.keys())
    return [
        {
            "fixture": fixture,
            "repaired_sections": repaired_counts[fixture],
            "broken_sections": broken_counts[fixture],
            "net_sections": repaired_counts[fixture] - broken_counts[fixture],
        }
        for fixture in fixtures
    ]


def _timed_side_counts(
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    *,
    tolerance_at_30_fps: int = 5,
) -> dict[str, int | float]:
    labelled = predicted = matched = answered = correct = 0
    first_contacts = matched_first = correct_first_side = 0
    for fixture, events in events_by_fixture.items():
        fixture_rallies = labels.rallies[fixture]
        label_frames = np.asarray(
            [frame for rally in fixture_rallies for frame in rally.frames],
            dtype=np.int32,
        )
        predicted_frames = np.asarray([event.frame for event in events], dtype=np.int32)
        matches = _match_contacts(
            label_frames,
            predicted_frames,
            scale_base30_frames(tolerance_at_30_fps, fps_by_fixture[fixture]),
        )
        first_frames = {rally.frames[0] for rally in fixture_rallies}
        labelled += len(label_frames)
        predicted += len(events)
        matched += len(matches)
        first_contacts += len(first_frames)
        for label_index, event_index, _offset in matches:
            label_frame = int(label_frames[label_index])
            event = events[event_index]
            if label_frame in first_frames:
                matched_first += 1
            if event.predicted_side is None:
                continue
            answered += 1
            side_correct = (
                event.predicted_side == labels.target_sides[(fixture, label_frame)]
            )
            correct += side_correct
            if label_frame in first_frames:
                correct_first_side += side_correct
    return {
        "labelled_contacts": labelled,
        "predicted_contacts": predicted,
        "timing_matches": matched,
        "side_answers": answered,
        "correct_sides": correct,
        "side_accuracy": correct / answered,
        "contact_and_side_f1": 2 * correct / (labelled + predicted),
        "first_contacts": first_contacts,
        "matched_first_contacts": matched_first,
        "first_contact_recall": matched_first / first_contacts,
        "correct_first_contact_sides": correct_first_side,
        "server_side_accuracy_when_start_matched": correct_first_side / matched_first,
    }


def _score_choice(
    minimum_vote_gap: int,
    all_decisions: Sequence[SideDecision],
    predictions: DevelopmentPredictionPack,
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    baseline_ids: set[tuple[str, int]],
) -> dict[str, object]:
    decisions = tuple(
        decision for decision in all_decisions if decision.score_gap >= minimum_vote_gap
    )
    revised_spans, revised_events = apply_side_decisions(
        predictions.spans,
        predictions.events_by_fixture,
        decisions,
    )
    revised_ids, _revised_rallies = _fully_correct(
        ContactStreams(revised_spans, revised_events),
        labels,
        fps_by_fixture,
        tolerance_at_30_fps=5,
        confidence_requirement=0.0,
    )
    repaired = revised_ids - baseline_ids
    broken = baseline_ids - revised_ids
    revised_side = _timed_side_counts(revised_events, labels, fps_by_fixture)
    return {
        "minimum_vote_gap": minimum_vote_gap,
        "sections_changed": len(decisions),
        "contacts_changed": sum(
            before != after
            for decision in decisions
            for before, after in zip(
                decision.sides_before,
                decision.sides_after,
                strict=True,
            )
        ),
        "revised_fully_correct": len(revised_ids),
        "repaired_sections": len(repaired),
        "broken_sections": len(broken),
        "net_sections": len(repaired) - len(broken),
        "revised_contact_and_side": revised_side,
        "by_group": _changes_by_group(repaired, broken, predictions.group_by_fixture),
        "by_video": _changes_by_video(repaired, broken),
        "repaired_identities": sorted(repaired),
        "broken_identities": sorted(broken),
    }


def main() -> None:
    """Choose a vote-gap cut-off on development videos and save the result."""
    predictions = load_development_predictions()
    decision_payload = read_json(DECISION_PATH)
    if decision_payload.get("labels_read") is not False:
        raise ValueError("Development side decisions were not made label-free")
    all_decisions = side_decisions_from_payload(
        decision_payload,
        "contact-detector-development-side-decisions/1",
    )
    labels = load_human_labels(LABEL_PATH, predictions.videos)
    fps_by_fixture = {video.fixture: video.fps for video in predictions.videos}
    baseline_ids, _baseline_rallies = _fully_correct(
        ContactStreams(predictions.spans, predictions.events_by_fixture),
        labels,
        fps_by_fixture,
        tolerance_at_30_fps=5,
        confidence_requirement=0.0,
    )
    baseline_side = _timed_side_counts(
        predictions.events_by_fixture,
        labels,
        fps_by_fixture,
    )
    choices = [
        _score_choice(
            minimum_vote_gap,
            all_decisions,
            predictions,
            labels,
            fps_by_fixture,
            baseline_ids,
        )
        for minimum_vote_gap in range(1, 7)
    ]
    selected = max(
        choices,
        key=lambda choice: (
            int(choice["net_sections"]),
            float(choice["revised_contact_and_side"]["contact_and_side_f1"]),
            float(
                choice["revised_contact_and_side"][
                    "server_side_accuracy_when_start_matched"
                ]
            ),
            -int(choice["contacts_changed"]),
        ),
    )
    payload = {
        "schema": "contact-detector-development-side-result/1",
        "run_id": "simple-alternation-vote-development",
        "repository_commit": _repository_commit(),
        "result_type": "Fixed label-free rule on out-of-fold development predictions",
        "labels_used": "40-video development labels, for scoring only",
        "video_groups": "A, B, C, D, and V; each contact model excluded the scored video group",
        "sections": len(predictions.spans),
        "baseline_fully_correct": len(baseline_ids),
        "baseline_contact_and_side": baseline_side,
        "choices": choices,
        "selected_minimum_vote_gap": selected["minimum_vote_gap"],
        "selected": selected,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = {
        "schema": "contact-detector-side-rule/1",
        "rule": "simple_alternation_vote",
        "minimum_vote_gap": selected["minimum_vote_gap"],
        "chosen_on": "40 development videos with each contact model held out by group",
        "development_result": {
            name: selected[name]
            for name in (
                "revised_fully_correct",
                "repaired_sections",
                "broken_sections",
                "net_sections",
            )
        },
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline_fully_correct": len(baseline_ids),
                "selected": selected,
                "choice_summary": [
                    {
                        name: choice[name]
                        for name in (
                            "minimum_vote_gap",
                            "repaired_sections",
                            "broken_sections",
                            "net_sections",
                            "contacts_changed",
                        )
                    }
                    for choice in choices
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
