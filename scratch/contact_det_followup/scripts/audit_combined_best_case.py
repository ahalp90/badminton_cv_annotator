"""Measure the label-guided ceiling from bounded combined event edits."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

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
from scratch.contact_det_followup.scripts.score_start_best_case import (
    _with_alternating_sides,
    start_actions,
)
from scratch.contact_det_followup.scripts.side_rules import alternating_pattern
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
OUTPUT_PATH = REPO_ROOT / "scratch/contact_det_followup/results/combined_best_case.json"
GROUPS = ("A", "B", "C", "D")
ACTION_KINDS = (
    "keep",
    "add",
    "replace",
    "delete",
    "add_delete",
    "replace_delete",
)
START_ACTION_KINDS = ("add", "replace")
COMBINED_ACTION_KINDS = ("add_delete", "replace_delete")


class _AlternatingHalf(StrEnum):
    """The two side phases accepted by the existing alternation helper."""

    TOP = "Top"
    BOT = "Bot"


@dataclass(frozen=True)
class CombinedAction:
    """One allowed section edit in the combined label-guided ceiling."""

    kind: str
    candidate_frame: int | None
    deleted_frame: int | None
    span: FixedSpan

    @property
    def identity(self) -> tuple[str, int]:
        return self.span.fixture, self.span.span_id


@dataclass(frozen=True)
class CeilingEvaluation:
    """The section changes from one allowed action pool."""

    name: str
    baseline_count: int
    revised_count: int
    repaired: frozenset[tuple[str, int]]
    broken: frozenset[tuple[str, int]]
    selected: Mapping[tuple[str, int], CombinedAction]

    @property
    def net(self) -> int:
        return len(self.repaired) - len(self.broken)


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
    payload = read_json(TRAINING_INPUT_PATH)
    if (
        payload.get("schema") != "contact-rally-start-training-inputs/1"
        or payload.get("status") != "complete"
        or payload.get("labels_read") is not False
    ):
        raise ValueError("saved candidate input record differs")
    raw_videos = payload.get("videos")
    counts = payload.get("counts")
    if (
        not isinstance(raw_videos, list)
        or not isinstance(counts, Mapping)
        or payload.get("source_commit") != "f08621a8"
        or payload.get("groups") != list(GROUPS)
        or counts.get("videos") != 32
        or counts.get("detected_sections") != 2850
        or counts.get("candidate_lists") != 2621
    ):
        raise TypeError("saved candidate videos must be a list")
    output: list[Mapping[str, object]] = []
    for raw_video in raw_videos:
        if not isinstance(raw_video, Mapping):
            raise TypeError("saved candidate video must be an object")
        if raw_video.get("group") not in GROUPS:
            raise ValueError("saved candidate input contains another group")
        raw_lists = raw_video.get("candidate_lists")
        if not isinstance(raw_lists, list):
            raise TypeError("saved candidate lists must be a list")
        for raw_list in raw_lists:
            if not isinstance(raw_list, Mapping):
                raise TypeError("saved candidate list must be an object")
            output.append(raw_list)
    return tuple(output)


def delete_actions(span: FixedSpan) -> tuple[CombinedAction, ...]:
    """Build one delete option for each existing event in a section."""
    actions: list[CombinedAction] = []
    for event in span.events:
        revised = FixedSpan(
            span.fixture,
            span.span_id,
            span.start_frame,
            span.end_frame,
            tuple(
                candidate for candidate in span.events if candidate.frame != event.frame
            ),
        )
        actions.append(CombinedAction("delete", None, event.frame, revised))
    return tuple(actions)


def section_actions(
    span: FixedSpan,
    candidate_list: Mapping[str, object] | None,
    fixture_events: Sequence[FixedEvent],
    previous_span_end: int,
) -> tuple[CombinedAction, ...]:
    """Return start-only, delete-only, and one-start-plus-one-delete options."""
    actions: list[CombinedAction] = [CombinedAction("keep", None, None, span)]
    start_options = tuple(
        CombinedAction(action.kind, action.candidate_frame, None, action.span)
        for action in start_actions(
            span,
            candidate_list,
            fixture_events,
            previous_span_end,
        )
        if action.kind in START_ACTION_KINDS
    )
    actions.extend(start_options)
    actions.extend(delete_actions(span))
    seen_combined: set[tuple[str, int, int, tuple[int, ...]]] = set()
    simple_event_lists = {
        tuple(event.frame for event in action.span.events)
        for action in actions
    }
    for start_option in start_options:
        if start_option.candidate_frame is None:
            raise ValueError("start option candidate is missing")
        combined_kind = f"{start_option.kind}_delete"
        for event in start_option.span.events:
            # Removing the inserted candidate would only undo this option.
            if event.frame == start_option.candidate_frame:
                continue
            # Add-then-delete-fixed is exactly the simpler replace action.
            if (
                start_option.kind == "add"
                and span.events
                and event.frame == span.events[0].frame
            ):
                continue
            revised_events = tuple(
                candidate for candidate in start_option.span.events if candidate.frame != event.frame
            )
            frame_list = tuple(candidate.frame for candidate in revised_events)
            if frame_list in simple_event_lists:
                continue
            identity = (
                combined_kind,
                start_option.candidate_frame,
                event.frame,
                frame_list,
            )
            if identity in seen_combined:
                continue
            seen_combined.add(identity)
            actions.append(
                CombinedAction(
                    combined_kind,
                    start_option.candidate_frame,
                    event.frame,
                    FixedSpan(
                        start_option.span.fixture,
                        start_option.span.span_id,
                        start_option.span.start_frame,
                        start_option.span.end_frame,
                        revised_events,
                    ),
                )
            )
    return tuple(actions)


def _apply_actions(
    spans: Sequence[FixedSpan],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
    selected: Mapping[tuple[str, int], CombinedAction],
) -> ContactStreams:
    """Apply one selected action per section and rebuild the whole event stream."""
    span_lookup = {(span.fixture, span.span_id): span for span in spans}
    if len(span_lookup) != len(spans):
        raise ValueError("section identities repeat")
    for identity, action in selected.items():
        if identity not in span_lookup or action.identity != identity:
            raise ValueError(f"{identity}: selected action identity differs")
        if action.kind not in ACTION_KINDS:
            raise ValueError(f"{identity}: unknown combined action")

    fixture_events: dict[str, list[FixedEvent]] = {
        fixture: list(events) for fixture, events in events_by_fixture.items()
    }
    for fixture, events in fixture_events.items():
        if any(event.fixture != fixture for event in events):
            raise ValueError(f"{fixture}: event fixture differs")
        if tuple(events) != tuple(sorted(events, key=lambda event: event.frame)):
            raise ValueError(f"{fixture}: event order differs")

    for identity, action in selected.items():
        if action.kind == "keep":
            continue
        fixture = action.span.fixture
        current_events = fixture_events.setdefault(fixture, [])
        has_start = action.kind in START_ACTION_KINDS or action.kind in COMBINED_ACTION_KINDS
        start_kind = action.kind.removesuffix("_delete") if has_start else None
        has_delete = action.kind == "delete" or action.kind in COMBINED_ACTION_KINDS
        if has_start:
            if action.candidate_frame is None:
                raise ValueError(f"{identity}: start action candidate is missing")
            candidate_events = [
                event
                for event in action.span.events
                if event.frame == action.candidate_frame
            ]
            if len(candidate_events) != 1:
                raise ValueError(f"{identity}: start action candidate differs")
            candidate_event = candidate_events[0]
            existing = [
                event
                for event in current_events
                if event.frame == candidate_event.frame
            ]
            if existing:
                if len(existing) != 1 or existing[0] != candidate_event:
                    raise ValueError(f"{identity}: candidate event differs")
            else:
                current_events.append(candidate_event)
            if start_kind == "replace":
                fixed_frame = span_lookup[identity].events[0].frame
                fixed_events = [
                    event for event in current_events if event.frame == fixed_frame
                ]
                if len(fixed_events) != 1:
                    raise ValueError(f"{identity}: fixed event differs")
                current_events = [
                    event for event in current_events if event.frame != fixed_frame
                ]
                fixture_events[fixture] = current_events
        if has_delete:
            if action.deleted_frame is None:
                raise ValueError(f"{identity}: delete frame is missing")
            deleted = [
                event for event in current_events if event.frame == action.deleted_frame
            ]
            if len(deleted) != 1:
                raise ValueError(f"{identity}: delete event differs")
            fixture_events[fixture] = [
                event for event in current_events if event.frame != action.deleted_frame
            ]

    frozen_events = {
        fixture: tuple(sorted(events, key=lambda event: event.frame))
        for fixture, events in fixture_events.items()
    }
    output_spans: list[FixedSpan] = []
    previous_end_by_fixture: dict[str, int] = {}
    assigned_events: set[tuple[str, int]] = set()
    for span in spans:
        identity = (span.fixture, span.span_id)
        action = selected.get(identity)
        output_start = span.start_frame
        if action is not None and (
            action.kind in START_ACTION_KINDS or action.kind in COMBINED_ACTION_KINDS
        ):
            output_start = action.span.start_frame
            if output_start < previous_end_by_fixture.get(span.fixture, -1):
                raise ValueError(f"{identity}: moved section overlaps predecessor")
        section_events = tuple(
            event
            for event in frozen_events.get(span.fixture, ())
            if output_start <= event.frame < span.end_frame
        )
        for event in section_events:
            event_identity = (event.fixture, event.frame)
            if event_identity in assigned_events:
                raise ValueError(f"{event_identity}: event belongs to two sections")
            assigned_events.add(event_identity)
        output_spans.append(
            FixedSpan(
                span.fixture,
                span.span_id,
                output_start,
                span.end_frame,
                section_events,
            )
        )
        previous_end_by_fixture[span.fixture] = span.end_frame
    return ContactStreams(tuple(output_spans), frozen_events)


def _option_fully_correct(
    action: CombinedAction,
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    tolerance_at_30_fps: int,
) -> bool:
    tolerance = scale_base30_frames(tolerance_at_30_fps, fps_by_fixture[action.span.fixture])
    return any(
        evaluate_span(
            _with_alternating_phase(action.span, final_half),
            labels.rallies.get(action.span.fixture, ()),
            labels.target_sides,
            tolerance,
            confidence_requirement=0.0,
        ).fully_correct
        for final_half in (_AlternatingHalf.TOP, _AlternatingHalf.BOT)
    )


def _with_alternating_phase(
    span: FixedSpan,
    final_half: _AlternatingHalf,
) -> FixedSpan:
    """Assign one of the two possible alternating side phases."""
    sides = alternating_pattern(final_half, len(span.events))
    events = tuple(
        FixedEvent(event.fixture, event.frame, event.timing_score, side)
        for event, side in zip(span.events, sides, strict=True)
    )
    return FixedSpan(
        span.fixture,
        span.span_id,
        span.start_frame,
        span.end_frame,
        events,
    )


def _label_guided_full_ids(
    streams: ContactStreams,
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    *,
    tolerance_at_30_fps: int,
) -> frozenset[tuple[str, int]]:
    """Score each revised event list under either label-guided side phase."""
    identities: set[tuple[str, int]] = set()
    rally_ids: dict[tuple[str, int], tuple[str, str]] = {}
    for span in streams.spans:
        tolerance = scale_base30_frames(
            tolerance_at_30_fps,
            fps_by_fixture[span.fixture],
        )
        scores = tuple(
            evaluate_span(
                _with_alternating_phase(span, final_half),
                labels.rallies.get(span.fixture, ()),
                labels.target_sides,
                tolerance,
                confidence_requirement=0.0,
            )
            for final_half in (_AlternatingHalf.TOP, _AlternatingHalf.BOT)
        )
        correct = next((score for score in scores if score.fully_correct), None)
        if correct is not None:
            identity = (span.fixture, span.span_id)
            if correct.rally_id is None:
                raise ValueError("fully correct section has no labelled rally")
            identities.add(identity)
            rally_ids[identity] = (span.fixture, correct.rally_id)
    if any(count > 1 for count in Counter(rally_ids.values()).values()):
        raise ValueError("one labelled rally is fully correct in more than one section")
    return frozenset(identities)


def _action_priority(action: CombinedAction) -> tuple[int, int, int]:
    kind_priority = {
        "keep": 0,
        "add": 1,
        "replace": 2,
        "delete": 3,
        "add_delete": 4,
        "replace_delete": 5,
    }
    candidate_frame = (
        action.candidate_frame
        if action.candidate_frame is not None
        else action.deleted_frame
    )
    return (
        kind_priority[action.kind],
        -1 if candidate_frame is None else candidate_frame,
        -1 if action.deleted_frame is None else action.deleted_frame,
    )


def choose_section_actions(
    options_by_section: Mapping[tuple[str, int], Sequence[CombinedAction]],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    *,
    allowed_kinds: frozenset[str],
    tolerance_at_30_fps: int,
) -> dict[tuple[str, int], CombinedAction]:
    """Choose the first deterministic fully-correct option for every section."""
    selected: dict[tuple[str, int], CombinedAction] = {}
    for identity, options in options_by_section.items():
        allowed = [option for option in options if option.kind in allowed_kinds]
        if not allowed:
            raise ValueError(f"{identity}: action pool is empty")
        correct = [
            option
            for option in allowed
            if _option_fully_correct(
                option,
                labels,
                fps_by_fixture,
                tolerance_at_30_fps,
            )
        ]
        selected[identity] = min(correct or allowed, key=_action_priority)
    return selected


def _score_selection(
    name: str,
    baseline: ContactStreams,
    spans: Sequence[FixedSpan],
    selected: Mapping[tuple[str, int], CombinedAction],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    baseline_ids: frozenset[tuple[str, int]],
    tolerance_at_30_fps: int,
) -> CeilingEvaluation:
    revised = _apply_actions(spans, baseline.events_by_fixture, selected)
    revised_ids = _label_guided_full_ids(
        revised,
        labels,
        fps_by_fixture,
        tolerance_at_30_fps=tolerance_at_30_fps,
    )
    repaired = frozenset(revised_ids - baseline_ids)
    broken = frozenset(baseline_ids - revised_ids)
    return CeilingEvaluation(
        name,
        len(baseline_ids),
        len(revised_ids),
        repaired,
        broken,
        selected,
    )


def _side_voted_stream(streams: ContactStreams) -> ContactStreams:
    spans = tuple(_with_alternating_sides(span) for span in streams.spans)
    events_by_fixture = {
        fixture: tuple(
            event for span in spans if span.fixture == fixture for event in span.events
        )
        for fixture in streams.events_by_fixture
    }
    return ContactStreams(spans, events_by_fixture)


def _groups(
    identities: Sequence[tuple[str, int]],
    group_by_fixture: Mapping[str, str],
) -> list[dict[str, int | str]]:
    counts = Counter(group_by_fixture[fixture] for fixture, _span_id in identities)
    return [{"group": group, "sections": counts[group]} for group in GROUPS]


def _changes_by_group(
    repaired: Sequence[tuple[str, int]],
    broken: Sequence[tuple[str, int]],
    group_by_fixture: Mapping[str, str],
) -> list[dict[str, int | str]]:
    repaired_counts = Counter(
        group_by_fixture[fixture] for fixture, _span_id in repaired
    )
    broken_counts = Counter(group_by_fixture[fixture] for fixture, _span_id in broken)
    return [
        {
            "group": group,
            "repaired_sections": repaired_counts[group],
            "broken_sections": broken_counts[group],
            "net_sections": repaired_counts[group] - broken_counts[group],
        }
        for group in GROUPS
    ]


def _change_payload(
    evaluation: CeilingEvaluation,
    group_by_fixture: Mapping[str, str],
) -> dict[str, object]:
    repaired = sorted(evaluation.repaired)
    broken = sorted(evaluation.broken)
    counts = Counter(action.kind for action in evaluation.selected.values())
    edited_identities = {
        identity
        for identity, action in evaluation.selected.items()
        if action.kind != "keep"
    }
    event_edit_repairs = sorted(evaluation.repaired & edited_identities)
    side_phase_only_repairs = sorted(evaluation.repaired - edited_identities)
    return {
        "baseline_fully_correct": evaluation.baseline_count,
        "revised_fully_correct": evaluation.revised_count,
        "repaired_sections": len(repaired),
        "broken_sections": len(broken),
        "net_sections": evaluation.net,
        "changes_by_group": _changes_by_group(repaired, broken, group_by_fixture),
        "repairs_by_group": _groups(repaired, group_by_fixture),
        "breaks_by_group": _groups(broken, group_by_fixture),
        "action_counts": {kind: counts[kind] for kind in ACTION_KINDS},
        "number_changed": sum(
            count for kind, count in counts.items() if kind != "keep"
        ),
        "event_edit_repaired_sections": len(event_edit_repairs),
        "side_phase_only_repaired_sections": len(side_phase_only_repairs),
        "event_edit_repaired_identities": event_edit_repairs,
        "side_phase_only_repaired_identities": side_phase_only_repairs,
        "repaired_identities": repaired,
        "broken_identities": broken,
    }


def _tolerance_result(
    tolerance_at_30_fps: int,
    raw_baseline: ContactStreams,
    spans: Sequence[FixedSpan],
    options_by_section: Mapping[tuple[str, int], Sequence[CombinedAction]],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    group_by_fixture: Mapping[str, str],
) -> dict[str, object]:
    """Score all three label-guided action pools at one timing tolerance."""
    baseline = _side_voted_stream(raw_baseline)
    baseline_ids, _baseline_rallies = _fully_correct(
        baseline,
        labels,
        fps_by_fixture,
        tolerance_at_30_fps=tolerance_at_30_fps,
        confidence_requirement=0.0,
    )
    start_selected = choose_section_actions(
        options_by_section,
        labels,
        fps_by_fixture,
        allowed_kinds=frozenset({"keep", "add", "replace"}),
        tolerance_at_30_fps=tolerance_at_30_fps,
    )
    delete_selected = choose_section_actions(
        options_by_section,
        labels,
        fps_by_fixture,
        allowed_kinds=frozenset({"keep", "delete"}),
        tolerance_at_30_fps=tolerance_at_30_fps,
    )
    combined_selected = choose_section_actions(
        options_by_section,
        labels,
        fps_by_fixture,
        allowed_kinds=frozenset(ACTION_KINDS),
        tolerance_at_30_fps=tolerance_at_30_fps,
    )
    start_evaluation = _score_selection(
        "start_only",
        raw_baseline,
        spans,
        start_selected,
        labels,
        fps_by_fixture,
        frozenset(baseline_ids),
        tolerance_at_30_fps,
    )
    delete_evaluation = _score_selection(
        "delete_only",
        raw_baseline,
        spans,
        delete_selected,
        labels,
        fps_by_fixture,
        frozenset(baseline_ids),
        tolerance_at_30_fps,
    )
    combined_evaluation = _score_selection(
        "combined",
        raw_baseline,
        spans,
        combined_selected,
        labels,
        fps_by_fixture,
        frozenset(baseline_ids),
        tolerance_at_30_fps,
    )
    extra_repairs = combined_evaluation.repaired - start_evaluation.repaired
    extra_fully_correct = (
        combined_evaluation.revised_count - start_evaluation.revised_count
    )
    denominator = start_evaluation.revised_count
    extra_rate = extra_fully_correct / denominator if denominator else 0.0
    decision = "continue" if extra_rate >= 0.05 else "stop"
    return {
        "tolerance_at_30_fps": tolerance_at_30_fps,
        "baseline_after_side_vote": {
            "fully_correct_sections": len(baseline_ids),
            "by_group": _groups(sorted(baseline_ids), group_by_fixture),
        },
        "start_only": _change_payload(start_evaluation, group_by_fixture),
        "delete_only": _change_payload(delete_evaluation, group_by_fixture),
        "combined": _change_payload(combined_evaluation, group_by_fixture),
        "combined_repairs_extra_beyond_start_only": {
            "sections": extra_fully_correct,
            "identities": sorted(extra_repairs),
            "denominator_start_only_fully_correct": denominator,
            "rate": extra_rate,
            "calculation": (
                f"{extra_fully_correct} extra combined fully-correct sections / {denominator} start-only fully-correct sections "
                f"= {extra_rate:.6f}"
            ),
        },
        "decision": decision,
        "decision_reason": (
            "Combined repairs add at least five percent over the start-only fully-correct denominator."
            if decision == "continue"
            else "Combined repairs add less than five percent over the start-only fully-correct denominator."
        ),
    }


def run_audit() -> dict[str, object]:
    """Score keep, start, delete, and combined label-guided ceilings."""
    predictions = load_development_predictions()
    fixtures = {
        fixture
        for fixture, group in predictions.group_by_fixture.items()
        if group in GROUPS
    }
    spans = tuple(span for span in predictions.spans if span.fixture in fixtures)
    events = {
        fixture: fixture_events
        for fixture, fixture_events in predictions.events_by_fixture.items()
        if fixture in fixtures
    }
    videos = tuple(video for video in predictions.videos if video.fixture in fixtures)
    labels = load_human_labels(LABEL_PATH, videos)
    fps_by_fixture = {video.fixture: video.fps for video in videos}
    raw_baseline = ContactStreams(spans, events)

    lists_by_span: dict[tuple[str, int], Mapping[str, object]] = {}
    for candidate_list in _candidate_lists():
        identity = (str(candidate_list["fixture"]), int(candidate_list["span_id"]))
        if identity in lists_by_span:
            raise ValueError(f"{identity}: candidate list repeats")
        lists_by_span[identity] = candidate_list

    options_by_section: dict[tuple[str, int], tuple[CombinedAction, ...]] = {}
    previous_end_by_fixture: dict[str, int] = {}
    for span in spans:
        identity = (span.fixture, span.span_id)
        options_by_section[identity] = section_actions(
            span,
            lists_by_span.get(identity),
            events[span.fixture],
            previous_end_by_fixture.get(span.fixture, -1),
        )
        previous_end_by_fixture[span.fixture] = span.end_frame

    group_by_fixture = dict(predictions.group_by_fixture)
    pool_counts = Counter(
        option.kind for options in options_by_section.values() for option in options
    )
    tolerance_results = {
        str(tolerance): _tolerance_result(
            tolerance,
            raw_baseline,
            spans,
            options_by_section,
            labels,
            fps_by_fixture,
            group_by_fixture,
        )
        for tolerance in (5, 10)
    }
    primary = tolerance_results["5"]
    return {
        "schema": "contact-detector-combined-best-case/2",
        "status": "complete",
        "run_id": "combined-start-delete-best-case",
        "repository_commit": _repository_commit(),
        "result_type": "Label-guided combined event-edit ceiling on all A-D development sections",
        "labels_used": "A-D labels choose each section edit and either alternating side phase, then score A-D; this is a descriptive ceiling",
        "inputs": {
            "candidate_inputs": str(TRAINING_INPUT_PATH.relative_to(REPO_ROOT)),
            "source_commit": "f08621a8",
            "videos": 32,
            "sections": 2850,
            "candidate_lists": 2621,
        },
        "sections": len(spans),
        "action_pool_counts": {kind: pool_counts[kind] for kind in ACTION_KINDS},
        "primary_tolerance_at_30_fps": 5,
        "baseline_after_side_vote": primary["baseline_after_side_vote"],
        "start_only": primary["start_only"],
        "delete_only": primary["delete_only"],
        "combined": primary["combined"],
        "combined_repairs_extra_beyond_start_only": primary[
            "combined_repairs_extra_beyond_start_only"
        ],
        "decision": primary["decision"],
        "decision_reason": primary["decision_reason"],
        "results_by_tolerance": tolerance_results,
    }


def main() -> None:
    """Run and save the combined label-guided ceiling."""
    payload = run_audit()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                tolerance: {
                    "baseline_fully_correct": result["baseline_after_side_vote"][
                        "fully_correct_sections"
                    ],
                    "start_only_fully_correct": result["start_only"][
                        "revised_fully_correct"
                    ],
                    "delete_only_fully_correct": result["delete_only"][
                        "revised_fully_correct"
                    ],
                    "combined_fully_correct": result["combined"][
                        "revised_fully_correct"
                    ],
                    "extra_combined_repairs": result[
                        "combined_repairs_extra_beyond_start_only"
                    ]["sections"],
                    "decision": result["decision"],
                }
                for tolerance, result in payload["results_by_tolerance"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
