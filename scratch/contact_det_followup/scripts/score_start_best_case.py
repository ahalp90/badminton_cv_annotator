"""Measure the best possible gain from saved earlier contact candidates."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    evaluate_span,
)
from scratch.contact_det_followup.scripts.prediction_io import (
    REPO_ROOT,
    load_development_predictions,
    read_json,
)
from scratch.contact_det_followup.scripts.score_setting_sweep import (
    timing_complete_ids,
)
from scratch.contact_det_followup.scripts.side_rules import choose_simple_alternation
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    HumanLabels,
    _fully_correct,
    scale_base30_frames,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

LABEL_PATH = REPO_ROOT / "training/data/shuttleset/annotations/shots_master.csv"
TRAINING_INPUT_PATH = (
    REPO_ROOT
    / "scratch/contact_det_full_ds_fit/raw/training_rally_start_inputs/rally_start_training_inputs.json.gz"
)
OUTPUT_PATH = REPO_ROOT / "scratch/contact_det_followup/results/start_best_case.json"


@dataclass(frozen=True)
class StartAction:
    """One allowed edit to the first event in a section."""

    kind: str
    candidate_frame: int | None
    span: FixedSpan


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _candidate_lists() -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for path, schema in (
        (TRAINING_INPUT_PATH, "contact-rally-start-training-inputs/1"),
    ):
        payload = read_json(path)
        if (
            payload.get("schema") != schema
            or payload.get("status") != "complete"
            or payload.get("labels_read") is not False
        ):
            raise ValueError(f"{path.name}: candidate input record differs")
        raw_videos = payload.get("videos")
        if not isinstance(raw_videos, list):
            raise TypeError(f"{path.name}: videos must be a list")
        for raw_video in raw_videos:
            if not isinstance(raw_video, Mapping):
                raise TypeError(f"{path.name}: video must be an object")
            raw_lists = raw_video.get("candidate_lists")
            if not isinstance(raw_lists, list):
                raise TypeError(f"{path.name}: candidate lists must be a list")
            records.extend(raw_lists)
    return tuple(records)


def start_actions(
    span: FixedSpan,
    candidate_list: Mapping[str, object] | None,
    fixture_events: Sequence[FixedEvent],
    previous_span_end: int,
) -> tuple[StartAction, ...]:
    """Build keep, add, and replace actions from one saved candidate list."""
    actions = [StartAction("keep", None, span)]
    if candidate_list is None:
        return tuple(actions)
    if (
        candidate_list.get("fixture") != span.fixture
        or int(candidate_list["span_id"]) != span.span_id
        or int(candidate_list["section_start_frame"]) != span.start_frame
        or int(candidate_list["section_end_frame"]) != span.end_frame
    ):
        raise ValueError(f"{span.fixture}/{span.span_id}: candidate section differs")
    if not span.events:
        raise ValueError(f"{span.fixture}/{span.span_id}: candidate section has no event")
    fixed_frame = int(candidate_list["fixed_contact_frame"])
    if span.events[0].frame != fixed_frame:
        raise ValueError(f"{span.fixture}/{span.span_id}: fixed first contact differs")
    raw_candidates = candidate_list.get("candidates")
    if not isinstance(raw_candidates, list):
        raise TypeError(f"{span.fixture}/{span.span_id}: candidates must be a list")
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            raise TypeError(f"{span.fixture}/{span.span_id}: candidate must be an object")
        if raw_candidate.get("is_fixed_contact") is True:
            continue
        frame = int(raw_candidate["frame"])
        if frame >= fixed_frame or any(event.frame == frame for event in span.events):
            raise ValueError(f"{span.fixture}/{span.span_id}/{frame}: earlier candidate differs")
        side = raw_candidate.get("predicted_side")
        if side not in {"Top", "Bot", None}:
            raise ValueError(f"{span.fixture}/{span.span_id}/{frame}: candidate side differs")
        event = FixedEvent(
            fixture=span.fixture,
            frame=frame,
            timing_score=float(raw_candidate["contact_score"]),
            predicted_side=None if side is None else str(side),
        )
        new_start = min(span.start_frame, frame)
        if new_start < previous_span_end:
            continue
        expanded_events = tuple(
            fixture_event
            for fixture_event in fixture_events
            if new_start <= fixture_event.frame < span.end_frame
        )
        existing_candidate = tuple(
            fixture_event
            for fixture_event in expanded_events
            if fixture_event.frame == frame
        )
        if existing_candidate:
            if len(existing_candidate) != 1 or existing_candidate[0] != event:
                raise ValueError(
                    f"{span.fixture}/{span.span_id}/{frame}: kept candidate differs"
                )
            added_events = expanded_events
        else:
            added_events = tuple(
                sorted((*expanded_events, event), key=lambda item: item.frame)
            )
        replaced_events = tuple(
            candidate_event
            for candidate_event in added_events
            if candidate_event.frame != fixed_frame
        )
        actions.append(
            StartAction(
                "add",
                frame,
                FixedSpan(
                    span.fixture,
                    span.span_id,
                    new_start,
                    span.end_frame,
                    added_events,
                ),
            )
        )
        actions.append(
            StartAction(
                "replace",
                frame,
                FixedSpan(
                    span.fixture,
                    span.span_id,
                    new_start,
                    span.end_frame,
                    replaced_events,
                ),
            )
        )
    return tuple(actions)


def _with_alternating_sides(span: FixedSpan) -> FixedSpan:
    decision = choose_simple_alternation(span)
    if decision is None or decision.score_gap < 1:
        return span
    events = tuple(
        FixedEvent(event.fixture, event.frame, event.timing_score, side)
        for event, side in zip(span.events, decision.sides_after, strict=True)
    )
    return FixedSpan(
        span.fixture,
        span.span_id,
        span.start_frame,
        span.end_frame,
        events,
    )


def _score_action(
    action: StartAction,
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
) -> tuple[bool, bool, bool]:
    span = action.span
    tolerance = scale_base30_frames(5, fps_by_fixture[span.fixture])
    current = evaluate_span(
        span,
        labels.rallies[span.fixture],
        labels.target_sides,
        tolerance,
        confidence_requirement=0.0,
    )
    timing_complete = (
        current.rally_id is not None
        and current.event_count == current.ground_truth_contacts
        and current.timing_matches == current.event_count
    )
    revised_span = _with_alternating_sides(span)
    revised = evaluate_span(
        revised_span,
        labels.rallies[span.fixture],
        labels.target_sides,
        tolerance,
        confidence_requirement=0.0,
    )
    return timing_complete, current.fully_correct, revised.fully_correct


def _counts_by_group(
    identities: set[tuple[str, int]],
    group_by_fixture: Mapping[str, str],
) -> list[dict[str, int | str]]:
    counts = Counter(group_by_fixture[fixture] for fixture, _span_id in identities)
    return [{"group": group, "sections": counts[group]} for group in ("A", "B", "C", "D")]


def run_audit() -> dict[str, object]:
    """Score the label-guided best allowed start edit on A-D videos."""
    all_predictions = load_development_predictions()
    training_fixtures = {
        fixture
        for fixture, group in all_predictions.group_by_fixture.items()
        if group != "V"
    }
    spans = tuple(
        span for span in all_predictions.spans if span.fixture in training_fixtures
    )
    events_by_fixture = {
        fixture: events
        for fixture, events in all_predictions.events_by_fixture.items()
        if fixture in training_fixtures
    }
    videos = tuple(
        video for video in all_predictions.videos if video.fixture in training_fixtures
    )
    raw_lists = _candidate_lists()
    lists_by_span: dict[tuple[str, int], Mapping[str, object]] = {}
    for candidate_list in raw_lists:
        identity = (str(candidate_list["fixture"]), int(candidate_list["span_id"]))
        if identity in lists_by_span:
            raise ValueError(f"{identity}: duplicate candidate list")
        lists_by_span[identity] = candidate_list
    labels = load_human_labels(LABEL_PATH, videos)
    fps_by_fixture = {video.fixture: video.fps for video in videos}
    baseline_timing = set(
        timing_complete_ids(spans, labels, fps_by_fixture)
    )
    baseline_full, _baseline_rallies = _fully_correct(
        ContactStreams(spans, events_by_fixture),
        labels,
        fps_by_fixture,
        tolerance_at_30_fps=5,
        confidence_requirement=0.0,
    )
    side_spans = tuple(_with_alternating_sides(span) for span in spans)
    side_events = {
        fixture: tuple(
            event
            for span in side_spans
            if span.fixture == fixture
            for event in span.events
        )
        for fixture in events_by_fixture
    }
    baseline_side_full, _baseline_side_rallies = _fully_correct(
        ContactStreams(side_spans, side_events),
        labels,
        fps_by_fixture,
        tolerance_at_30_fps=5,
        confidence_requirement=0.0,
    )

    best_timing: set[tuple[str, int]] = set()
    best_current_side: set[tuple[str, int]] = set()
    best_rally_side: set[tuple[str, int]] = set()
    timing_by_action: dict[str, set[tuple[str, int]]] = {
        "add": set(),
        "replace": set(),
    }
    rally_side_by_action: dict[str, set[tuple[str, int]]] = {
        "add": set(),
        "replace": set(),
    }
    sections_with_candidates = 0
    previous_end_by_fixture: dict[str, int] = {}
    for span in spans:
        identity = (span.fixture, span.span_id)
        candidate_list = lists_by_span.get(identity)
        actions = start_actions(
            span,
            candidate_list,
            events_by_fixture[span.fixture],
            previous_end_by_fixture.get(span.fixture, -1),
        )
        previous_end_by_fixture[span.fixture] = span.end_frame
        sections_with_candidates += candidate_list is not None
        for action in actions:
            timing_complete, current_full, rally_side_full = _score_action(
                action,
                labels,
                fps_by_fixture,
            )
            if timing_complete:
                best_timing.add(identity)
                if action.kind in timing_by_action:
                    timing_by_action[action.kind].add(identity)
            if current_full:
                best_current_side.add(identity)
            if rally_side_full:
                best_rally_side.add(identity)
                if action.kind in rally_side_by_action:
                    rally_side_by_action[action.kind].add(identity)

    timing_repairs = best_timing - baseline_timing
    current_side_repairs = best_current_side - baseline_full
    rally_side_repairs = best_rally_side - baseline_side_full
    decision = "continue" if len(timing_repairs) >= 40 else "stop"
    return {
        "schema": "contact-detector-start-best-case/1",
        "status": "complete",
        "run_id": "start-keep-add-replace-best-case",
        "repository_commit": _repository_commit(),
        "result_type": "Label-guided best-case check on 32 A-D development videos",
        "labels_used": "A-D labels choose the best allowed action for this ceiling only; V labels are not read",
        "finished_rule_inputs": "Saved section bounds, current contacts, and two earlier candidates per eligible section",
        "inputs": [str(TRAINING_INPUT_PATH.relative_to(REPO_ROOT))],
        "sections": len(spans),
        "sections_with_candidate_lists": sections_with_candidates,
        "candidate_lists": len(raw_lists),
        "timing_only": {
            "baseline_sections": len(baseline_timing),
            "best_sections": len(best_timing),
            "repaired_sections": len(timing_repairs),
            "repairs_by_group": _counts_by_group(
                timing_repairs,
                all_predictions.group_by_fixture,
            ),
            "add_repair_pool": len(timing_by_action["add"] - baseline_timing),
            "replace_repair_pool": len(timing_by_action["replace"] - baseline_timing),
            "repaired_identities": sorted(timing_repairs),
        },
        "timing_and_current_side": {
            "baseline_sections": len(baseline_full),
            "best_sections": len(best_current_side),
            "repaired_sections": len(current_side_repairs),
            "repairs_by_group": _counts_by_group(
                current_side_repairs,
                all_predictions.group_by_fixture,
            ),
            "repaired_identities": sorted(current_side_repairs),
        },
        "timing_then_rally_side": {
            "baseline_sections": len(baseline_side_full),
            "best_sections": len(best_rally_side),
            "repaired_sections": len(rally_side_repairs),
            "repairs_by_group": _counts_by_group(
                rally_side_repairs,
                all_predictions.group_by_fixture,
            ),
            "add_repair_pool": len(rally_side_by_action["add"] - baseline_side_full),
            "replace_repair_pool": len(
                rally_side_by_action["replace"] - baseline_side_full
            ),
            "repaired_identities": sorted(rally_side_repairs),
        },
        "decision": decision,
        "decision_reason": (
            "The timing-only best case clears the 40-section signal."
            if decision == "continue"
            else "The timing-only best case does not clear the 40-section signal."
        ),
    }


def main() -> None:
    """Run and save the first-contact best-case audit."""
    payload = run_audit()
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    headline: dict[str, object] = {}
    for name in (
        "timing_only",
        "timing_and_current_side",
        "timing_then_rally_side",
    ):
        values = dict(payload[name])
        values.pop("repaired_identities")
        headline[name] = values
    headline["decision"] = payload["decision"]
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
