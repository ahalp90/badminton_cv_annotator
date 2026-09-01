"""Fit and score a small held-out first-contact action chooser."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
    _greedy_matches,
)
from scratch.contact_det_followup.scripts.prediction_io import (
    REPO_ROOT,
    load_development_predictions,
    read_json,
)
from scratch.contact_det_followup.scripts.side_rules import choose_simple_alternation
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    CandidateRow,
    ContactStreams,
    HumanLabels,
    build_candidate_rows,
    scale_base30_frames,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model_config import (
    ModelKind,
    ModelSpec,
    RallyStartModelConfig,
    load_rally_start_model_config,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

LABEL_PATH = REPO_ROOT / "training/data/shuttleset/annotations/shots_master.csv"
TRAINING_INPUT_PATH = (
    REPO_ROOT
    / "scratch/contact_det_full_ds_fit/raw/training_rally_start_inputs/rally_start_training_inputs.json.gz"
)
CONFIG_PATH = REPO_ROOT / "scratch/contact_det_full_ds_fit/records/rally_start_model_runs.json"
OUTPUT_PATH = (
    REPO_ROOT / "scratch/contact_det_followup/results/start_model_development.json"
)
GROUPS = ("A", "B", "C", "D")
ACTION_KINDS = ("add", "replace")
ACTION_FEATURE_NAMES = (
    "candidate_contact_score",
    "fixed_contact_score",
    "frames_before_fixed_at_30_fps",
    "candidate_from_section_start_at_30_fps",
    "section_length_at_30_fps",
    "candidate_already_kept",
    "candidate_side_known",
    "fixed_side_known",
    "candidate_and_fixed_side_match",
    "action_is_replace",
)

CandidateIdentity = tuple[str, int, int]
SectionIdentity = tuple[str, int]
ActionIdentity = tuple[str, int, int, str]


@dataclass(frozen=True)
class ActionRow:
    """One candidate/action pair and its ten fixed model inputs."""

    candidate: CandidateRow
    action: str
    features: tuple[float, ...]

    @property
    def identity(self) -> ActionIdentity:
        return (*self.candidate.identity, self.action)

    @property
    def section_identity(self) -> SectionIdentity:
        return self.candidate.section_identity


@dataclass(frozen=True)
class ActionTarget:
    """The label-derived training answer for one action row."""

    included_in_training: bool
    positive: bool
    section_status: str
    baseline_timing_complete: bool
    action_timing_complete: bool


@dataclass(frozen=True)
class ActionTargetAssignments:
    """Action targets and the section-level exclusion statuses."""

    by_action: Mapping[ActionIdentity, ActionTarget]
    section_statuses: Mapping[SectionIdentity, str]


@dataclass(frozen=True)
class AppliedAction:
    """One selected edit and its revised section span."""

    kind: str
    candidate: CandidateRow
    span: FixedSpan


@dataclass(frozen=True)
class StartChoiceEvaluation:
    """Fully-correct changes produced by one model and cut-off."""

    model_id: str
    cutoff: float
    baseline_count: int
    revised_count: int
    repaired: frozenset[SectionIdentity]
    broken: frozenset[SectionIdentity]
    changed: int
    action_counts: Mapping[str, int]

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


def _candidate_videos() -> tuple[Mapping[str, Any], ...]:
    """Load only the saved A-D candidate rows and their label-free inputs."""
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
        or payload.get("groups") != list(GROUPS)
        or payload.get("source_commit") != "f08621a8"
        or counts.get("videos") != 32
        or counts.get("candidate_lists") != 2621
        or counts.get("earlier_candidate_entries") != 5242
    ):
        raise TypeError("saved candidate videos must be a list")
    videos: list[Mapping[str, Any]] = []
    for raw_video in raw_videos:
        if not isinstance(raw_video, Mapping):
            raise TypeError("saved candidate video must be an object")
        group = raw_video.get("group")
        if group not in GROUPS:
            raise ValueError("saved candidate input contains a non-development group")
        videos.append(raw_video)
    if {str(video["group"]) for video in videos} != set(GROUPS):
        raise ValueError("saved candidate input groups differ")
    return tuple(videos)


def build_action_rows(rows: Sequence[CandidateRow]) -> tuple[ActionRow, ...]:
    """Duplicate every candidate into add and replace action rows."""
    output: list[ActionRow] = []
    for candidate in rows:
        if len(candidate.features) != 9 or not np.isfinite(candidate.features).all():
            raise ValueError(f"{candidate.identity}: candidate features differ")
        for action in ACTION_KINDS:
            output.append(
                ActionRow(
                    candidate=candidate,
                    action=action,
                    features=(*candidate.features, float(action == "replace")),
                )
            )
    return tuple(output)


def _event_for_candidate(candidate: CandidateRow) -> FixedEvent:
    return FixedEvent(
        fixture=candidate.fixture,
        frame=candidate.frame,
        timing_score=candidate.contact_score,
        predicted_side=candidate.predicted_side,
    )


def _action_span(
    span: FixedSpan,
    candidate: CandidateRow,
    action: str,
    fixture_events: Sequence[FixedEvent],
    previous_span_end: int,
) -> FixedSpan | None:
    """Build one add/replace span, returning None for predecessor overlap."""
    if action not in ACTION_KINDS:
        raise ValueError(f"unknown first-contact action {action!r}")
    if candidate.section_identity != (span.fixture, span.span_id):
        raise ValueError("candidate section identity differs")
    if not span.events or span.events[0].frame != candidate.fixed_contact_frame:
        raise ValueError(f"{span.fixture}/{span.span_id}: fixed first contact differs")
    if candidate.frame >= candidate.fixed_contact_frame:
        raise ValueError(
            f"{candidate.identity}: candidate is not earlier than the fixed contact"
        )
    expanded_start = min(span.start_frame, candidate.frame)
    if expanded_start < previous_span_end:
        return None
    expanded_events = tuple(
        event
        for event in fixture_events
        if expanded_start <= event.frame < span.end_frame
    )
    candidate_event = _event_for_candidate(candidate)
    existing_candidate = tuple(
        event for event in expanded_events if event.frame == candidate.frame
    )
    if existing_candidate:
        if len(existing_candidate) != 1 or existing_candidate[0] != candidate_event:
            raise ValueError(f"{candidate.identity}: saved candidate event differs")
        events_with_candidate = expanded_events
    else:
        events_with_candidate = tuple(
            sorted((*expanded_events, candidate_event), key=lambda event: event.frame)
        )
    if action == "replace":
        fixed_events = tuple(
            event
            for event in events_with_candidate
            if event.frame == candidate.fixed_contact_frame
        )
        if len(fixed_events) != 1:
            raise ValueError(
                f"{span.fixture}/{span.span_id}: fixed contact event differs"
            )
        final_events = tuple(
            event
            for event in events_with_candidate
            if event.frame != candidate.fixed_contact_frame
        )
    else:
        final_events = events_with_candidate
    return FixedSpan(
        fixture=span.fixture,
        span_id=span.span_id,
        start_frame=expanded_start,
        end_frame=span.end_frame,
        events=final_events,
    )


def start_actions(
    span: FixedSpan,
    candidates: Sequence[CandidateRow],
    fixture_events: Sequence[FixedEvent],
    previous_span_end: int,
) -> tuple[AppliedAction, ...]:
    """Build valid add and replace actions for one section."""
    actions: list[AppliedAction] = []
    for candidate in candidates:
        for action in ACTION_KINDS:
            revised = _action_span(
                span,
                candidate,
                action,
                fixture_events,
                previous_span_end,
            )
            if revised is not None:
                actions.append(AppliedAction(action, candidate, revised))
    return tuple(actions)


def _span_lookup(spans: Sequence[FixedSpan]) -> dict[SectionIdentity, FixedSpan]:
    output: dict[SectionIdentity, FixedSpan] = {}
    for span in spans:
        identity = (span.fixture, span.span_id)
        if identity in output:
            raise ValueError(f"{identity}: section identity repeats")
        output[identity] = span
    return output


def apply_selected_actions(
    spans: Sequence[FixedSpan],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
    selections: Mapping[SectionIdentity, ActionRow],
) -> ContactStreams:
    """Apply selected actions while retaining all events in an expanded prefix."""
    span_by_identity = _span_lookup(spans)
    fixture_events: dict[str, list[FixedEvent]] = {}
    for fixture, events in events_by_fixture.items():
        ordered = tuple(events)
        if tuple(sorted(ordered, key=lambda event: event.frame)) != ordered:
            raise ValueError(f"{fixture}: event order differs")
        if any(event.fixture != fixture for event in ordered):
            raise ValueError(f"{fixture}: event fixture differs")
        fixture_events[fixture] = list(ordered)
    for identity, action_row in selections.items():
        if identity not in span_by_identity:
            raise ValueError(f"{identity}: selected section is missing")
        if (
            action_row.section_identity != identity
            or action_row.action not in ACTION_KINDS
        ):
            raise ValueError(f"{identity}: selected action identity differs")
        candidate = action_row.candidate
        span = span_by_identity[identity]
        if not span.events or span.events[0].frame != candidate.fixed_contact_frame:
            raise ValueError(f"{identity}: fixed first contact differs")
        if candidate.frame >= candidate.fixed_contact_frame:
            raise ValueError(
                f"{candidate.identity}: candidate is not earlier than the fixed contact"
            )
        fixture_events.setdefault(candidate.fixture, [])
        existing = [
            event
            for event in fixture_events[candidate.fixture]
            if event.frame == candidate.frame
        ]
        candidate_event = _event_for_candidate(candidate)
        if existing:
            if len(existing) != 1 or existing[0] != candidate_event:
                raise ValueError(f"{candidate.identity}: saved candidate event differs")
        else:
            fixture_events[candidate.fixture].append(candidate_event)
        if action_row.action == "replace":
            fixed_frame = candidate.fixed_contact_frame
            fixed_events = [
                event
                for event in fixture_events[candidate.fixture]
                if event.frame == fixed_frame
            ]
            if len(fixed_events) != 1:
                raise ValueError(f"{identity}: fixed contact event is missing")
            fixture_events[candidate.fixture] = [
                event
                for event in fixture_events[candidate.fixture]
                if event.frame != fixed_frame
            ]

    frozen_events = {
        fixture: tuple(sorted(events, key=lambda event: event.frame))
        for fixture, events in fixture_events.items()
    }
    previous_end_by_fixture: dict[str, int] = {}
    output_spans: list[FixedSpan] = []
    assigned_events: set[tuple[str, int]] = set()
    for span in spans:
        previous_end = previous_end_by_fixture.get(span.fixture, -1)
        action_row = selections.get((span.fixture, span.span_id))
        output_start = span.start_frame
        if action_row is not None:
            output_start = min(span.start_frame, action_row.candidate.frame)
            if output_start < previous_end:
                raise ValueError(
                    f"{span.fixture}/{span.span_id}: moved section overlaps its predecessor"
                )
        section_events = tuple(
            event
            for event in frozen_events.get(span.fixture, ())
            if output_start <= event.frame < span.end_frame
        )
        for event in section_events:
            identity = (event.fixture, event.frame)
            if identity in assigned_events:
                raise ValueError(f"{identity}: event belongs to two sections")
            assigned_events.add(identity)
        output_spans.append(
            FixedSpan(
                fixture=span.fixture,
                span_id=span.span_id,
                start_frame=output_start,
                end_frame=span.end_frame,
                events=section_events,
            )
        )
        previous_end_by_fixture[span.fixture] = span.end_frame
    return ContactStreams(tuple(output_spans), frozen_events)


def _section_statuses(
    rows: Sequence[CandidateRow],
    videos: Sequence[Mapping[str, Any]],
    labels: HumanLabels,
    *,
    default_group: str,
    tolerance_at_30_fps: int,
) -> dict[SectionIdentity, str]:
    """Copy the existing ambiguous-rally exclusions without side labels."""
    spans_by_fixture: dict[str, tuple[Mapping[str, Any], ...]] = {}
    contacts_by_fixture: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for video in videos:
        identity = video.get("video")
        if not isinstance(identity, Mapping):
            identity = video
        fixture = identity.get("fixture")
        if not isinstance(fixture, str):
            raise TypeError("saved video fixture differs")
        raw_spans = video.get("spans")
        raw_contacts = video.get("kept_contacts")
        if not isinstance(raw_spans, list) or not isinstance(raw_contacts, list):
            raise TypeError(f"{fixture}: spans and kept contacts must be lists")
        spans_by_fixture[fixture] = tuple(
            span for span in raw_spans if isinstance(span, Mapping)
        )
        contacts_by_fixture[fixture] = tuple(
            contact for contact in raw_contacts if isinstance(contact, Mapping)
        )

    rally_sections: Counter[tuple[str, str]] = Counter()
    matching_by_section: dict[SectionIdentity, tuple[RallyReference, ...]] = {}
    for fixture, raw_spans in spans_by_fixture.items():
        rallies = labels.rallies.get(fixture, ())
        for raw_span in raw_spans:
            span_id = int(raw_span["span_id"])
            start = int(raw_span["start_frame"])
            end = int(raw_span["end_frame"])
            matching = tuple(
                rally
                for rally in rallies
                if any(start <= frame < end for frame in rally.frames)
            )
            identity = (fixture, span_id)
            matching_by_section[identity] = matching
            for rally in matching:
                rally_sections[(fixture, rally.rally_id)] += 1

    rows_by_section: dict[SectionIdentity, list[CandidateRow]] = {}
    for row in rows:
        rows_by_section.setdefault(row.section_identity, []).append(row)
    statuses: dict[SectionIdentity, str] = {}
    for identity, section_rows in rows_by_section.items():
        fixture, span_id = identity
        matching = matching_by_section.get(identity)
        if matching is None:
            raise ValueError(f"{identity}: candidate section is missing")
        if not matching:
            statuses[identity] = "no_labelled_rally"
            continue
        if len(matching) > 1:
            statuses[identity] = "more_than_one_labelled_rally"
            continue
        rally = matching[0]
        if rally_sections[(fixture, rally.rally_id)] > 1:
            statuses[identity] = "labelled_rally_touches_more_than_one_section"
            continue
        first_contact = rally.frames[0]
        tolerance = scale_base30_frames(tolerance_at_30_fps, section_rows[0].fps)
        section_start = section_rows[0].section_start_frame
        section_end = section_rows[0].section_end_frame
        existing_frames = [
            int(contact["frame"])
            for contact in contacts_by_fixture[fixture]
            if section_start <= int(contact["frame"]) < section_end
        ]
        if any(abs(frame - first_contact) <= tolerance for frame in existing_frames):
            statuses[identity] = "first_contact_already_matched"
        elif any(abs(row.frame - first_contact) <= tolerance for row in section_rows):
            # Side agreement is deliberately absent from this action target.
            statuses[identity] = "usable_candidate"
        else:
            statuses[identity] = "no_usable_candidate"
    return statuses


def _timing_complete(
    span: FixedSpan,
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    *,
    tolerance_at_30_fps: int,
) -> bool:
    """Return exact one-rally timing status without consulting player sides."""
    rallies = tuple(
        rally
        for rally in labels.rallies.get(span.fixture, ())
        if any(span.start_frame <= frame < span.end_frame for frame in rally.frames)
    )
    if len(rallies) != 1:
        return False
    rally = rallies[0]
    tolerance = scale_base30_frames(tolerance_at_30_fps, fps_by_fixture[span.fixture])
    matches = _greedy_matches(
        rally.frames,
        tuple(event.frame for event in span.events),
        tolerance,
    )
    return len(span.events) == len(rally.frames) == len(matches)


def assign_action_targets(
    action_rows: Sequence[ActionRow],
    spans: Sequence[FixedSpan],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
    videos: Sequence[Mapping[str, Any]],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    *,
    default_group: str,
) -> ActionTargetAssignments:
    """Assign one timing-only positive action per eligible section."""
    candidate_rows: dict[CandidateIdentity, CandidateRow] = {}
    for action_row in action_rows:
        if action_row.action not in ACTION_KINDS:
            raise ValueError(f"unknown action {action_row.action!r}")
        identity = action_row.candidate.identity
        previous = candidate_rows.setdefault(identity, action_row.candidate)
        if previous != action_row.candidate:
            raise ValueError(f"{identity}: candidate row differs")
    statuses = _section_statuses(
        tuple(candidate_rows.values()),
        videos,
        labels,
        default_group=default_group,
        tolerance_at_30_fps=10,
    )
    rows_by_section: dict[SectionIdentity, list[ActionRow]] = {}
    for action_row in action_rows:
        rows_by_section.setdefault(action_row.section_identity, []).append(action_row)
    previous_end_by_fixture: dict[str, int] = {}
    action_spans: dict[ActionIdentity, FixedSpan | None] = {}
    baseline_complete: dict[SectionIdentity, bool] = {}
    for span in spans:
        identity = (span.fixture, span.span_id)
        if identity in rows_by_section:
            baseline_complete[identity] = _timing_complete(
                span,
                labels,
                fps_by_fixture,
                tolerance_at_30_fps=5,
            )
            for action_row in rows_by_section[identity]:
                action_spans[action_row.identity] = _action_span(
                    span,
                    action_row.candidate,
                    action_row.action,
                    events_by_fixture[span.fixture],
                    previous_end_by_fixture.get(span.fixture, -1),
                )
        previous_end_by_fixture[span.fixture] = span.end_frame
    if set(baseline_complete) != set(rows_by_section):
        raise ValueError("action target section coverage differs")

    by_action: dict[ActionIdentity, ActionTarget] = {}
    for section_identity, section_rows in rows_by_section.items():
        section_status = statuses[section_identity]
        baseline = baseline_complete[section_identity]
        eligible_for_training = section_status not in {
            "more_than_one_labelled_rally",
            "labelled_rally_touches_more_than_one_section",
        }
        correctness: dict[ActionIdentity, bool] = {}
        first_contact_by_action: dict[ActionIdentity, int] = {}
        for action_row in section_rows:
            revised = action_spans[action_row.identity]
            correctness[action_row.identity] = (
                revised is not None
                and not baseline
                and _timing_complete(
                    revised,
                    labels,
                    fps_by_fixture,
                    tolerance_at_30_fps=5,
                )
            )
            if correctness[action_row.identity]:
                assert revised is not None
                matching_rallies = tuple(
                    rally
                    for rally in labels.rallies.get(revised.fixture, ())
                    if any(
                        revised.start_frame <= frame < revised.end_frame
                        for frame in rally.frames
                    )
                )
                if len(matching_rallies) != 1:
                    raise ValueError(
                        f"{action_row.identity}: repairing action has no single rally"
                    )
                first_contact_by_action[action_row.identity] = matching_rallies[
                    0
                ].frames[0]
        positive_rows = [
            action_row
            for action_row in section_rows
            if correctness[action_row.identity]
        ]
        positive_row = min(
            positive_rows,
            key=lambda identity: (
                0 if identity.action == "add" else 1,
                abs(
                    identity.candidate.frame
                    - first_contact_by_action[identity.identity]
                ),
                -identity.candidate.contact_score,
                identity.candidate.frame,
            ),
            default=None,
        )
        positive_identity = None if positive_row is None else positive_row.identity
        for action_row in section_rows:
            revised = action_spans[action_row.identity]
            by_action[action_row.identity] = ActionTarget(
                included_in_training=eligible_for_training and revised is not None,
                positive=eligible_for_training
                and action_row.identity == positive_identity,
                section_status=section_status,
                baseline_timing_complete=baseline,
                action_timing_complete=correctness[action_row.identity],
            )
    if set(by_action) != {row.identity for row in action_rows}:
        raise ValueError("action target coverage differs")
    return ActionTargetAssignments(by_action, statuses)


def _action_feature_array(rows: Sequence[ActionRow]) -> np.ndarray:
    values = np.asarray([row.features for row in rows], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 10 or not np.isfinite(values).all():
        raise ValueError("action model inputs differ")
    return values


def make_action_model(spec: ModelSpec) -> Any:
    """Construct one model from the reviewed fixed model settings."""
    settings = dict(spec.settings)
    if spec.kind is ModelKind.LOGISTIC_REGRESSION:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if settings.pop("standardise_numeric_inputs") is not True:
            raise ValueError("logistic regression scaling setting differs")
        return make_pipeline(StandardScaler(), LogisticRegression(**settings))
    if spec.kind is ModelKind.HISTOGRAM_GRADIENT_BOOSTING:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(**settings)
    raise ValueError(f"unsupported action model: {spec.kind}")


def fit_action_model(
    spec: ModelSpec,
    training_rows: Sequence[ActionRow],
    targets: ActionTargetAssignments,
) -> Any:
    """Fit one action model on included rows and their saved targets."""
    included_rows = [
        row
        for row in training_rows
        if targets.by_action[row.identity].included_in_training
    ]
    target_values = np.asarray(
        [targets.by_action[row.identity].positive for row in included_rows],
        dtype=np.uint8,
    )
    if not included_rows or set(target_values.tolist()) != {0, 1}:
        raise ValueError("action model training needs positive and negative examples")
    model = make_action_model(spec)
    model.fit(_action_feature_array(included_rows), target_values)
    return model


def predict_action_scores(
    model: Any,
    rows: Sequence[ActionRow],
) -> dict[ActionIdentity, float]:
    """Predict positive probabilities without labels or target metadata."""
    if not rows:
        return {}
    classes = np.asarray(model.classes_)
    positive_positions = np.flatnonzero(classes == 1)
    if len(positive_positions) != 1:
        raise ValueError("action model positive class differs")
    probabilities = np.asarray(model.predict_proba(_action_feature_array(rows)))
    scores = probabilities[:, int(positive_positions[0])]
    if len(scores) != len(rows) or not np.isfinite(scores).all():
        raise ValueError("action model scores differ")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("action model scores must be between zero and one")
    return {row.identity: float(score) for row, score in zip(rows, scores, strict=True)}


def held_out_action_scores(
    rows: Sequence[ActionRow],
    targets: ActionTargetAssignments,
    config: RallyStartModelConfig,
) -> dict[str, dict[ActionIdentity, float]]:
    """Fit on three groups and predict every action in the held-out group."""
    if {row.candidate.group for row in rows} != set(config.training_groups):
        raise ValueError("action row groups differ")
    output: dict[str, dict[ActionIdentity, float]] = {
        spec.model_id: {} for spec in config.models
    }
    for held_out_group in config.training_groups:
        training_rows = [row for row in rows if row.candidate.group != held_out_group]
        held_out_rows = [row for row in rows if row.candidate.group == held_out_group]
        for spec in config.models:
            model = fit_action_model(spec, training_rows, targets)
            group_scores = predict_action_scores(model, held_out_rows)
            if set(output[spec.model_id]) & set(group_scores):
                raise ValueError("held-out action score identities repeat")
            output[spec.model_id].update(group_scores)
    expected = {row.identity for row in rows}
    if any(set(scores) != expected for scores in output.values()):
        raise ValueError("held-out action score coverage differs")
    return output


def select_actions(
    rows: Sequence[ActionRow],
    scores: Mapping[ActionIdentity, float],
    cutoff: float,
) -> dict[SectionIdentity, ActionRow]:
    """Choose at most one scored action per section without reading labels."""
    if not 0.0 < cutoff < 1.0:
        raise ValueError("action selection cut-off must be between zero and one")
    if set(scores) != {row.identity for row in rows}:
        raise ValueError("action selection score coverage differs")
    rows_by_section: dict[SectionIdentity, list[ActionRow]] = {}
    for row in rows:
        rows_by_section.setdefault(row.section_identity, []).append(row)
    selected: dict[SectionIdentity, ActionRow] = {}
    for section_identity, section_rows in rows_by_section.items():
        eligible = [row for row in section_rows if scores[row.identity] >= cutoff]
        if eligible:
            selected[section_identity] = max(
                eligible,
                key=lambda row: (
                    scores[row.identity],
                    int(row.action == "add"),
                    row.candidate.contact_score,
                    -row.candidate.frame,
                ),
            )
    return selected


def _with_alternating_sides(span: FixedSpan) -> FixedSpan:
    decision = choose_simple_alternation(span)
    if decision is None or decision.score_gap < 1:
        return span
    events = tuple(
        FixedEvent(event.fixture, event.frame, event.timing_score, side)
        for event, side in zip(span.events, decision.sides_after, strict=True)
    )
    return FixedSpan(
        span.fixture, span.span_id, span.start_frame, span.end_frame, events
    )


def apply_whole_rally_alternation(streams: ContactStreams) -> ContactStreams:
    """Apply the existing minimum-one-vote-gap alternation to every span."""
    overrides: dict[tuple[str, int], str] = {}
    for span in streams.spans:
        revised = _with_alternating_sides(span)
        for before, after in zip(span.events, revised.events, strict=True):
            identity = (before.fixture, before.frame)
            if identity in overrides:
                raise ValueError(f"{identity}: side changed twice")
            overrides[identity] = after.predicted_side
    revised_events = {
        fixture: tuple(
            FixedEvent(
                event.fixture,
                event.frame,
                event.timing_score,
                overrides.get((fixture, event.frame), event.predicted_side),
            )
            for event in events
        )
        for fixture, events in streams.events_by_fixture.items()
    }
    event_lookup = {
        (event.fixture, event.frame): event
        for events in revised_events.values()
        for event in events
    }
    revised_spans = tuple(
        FixedSpan(
            span.fixture,
            span.span_id,
            span.start_frame,
            span.end_frame,
            tuple(event_lookup[(event.fixture, event.frame)] for event in span.events),
        )
        for span in streams.spans
    )
    return ContactStreams(revised_spans, revised_events)


def _fully_correct_ids(
    streams: ContactStreams,
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    *,
    tolerance_at_30_fps: int = 5,
) -> frozenset[SectionIdentity]:
    from scratch.contact_det_full_ds_fit.scripts.rally_start_model import _fully_correct

    identities, _rallies = _fully_correct(
        streams,
        labels,
        fps_by_fixture,
        tolerance_at_30_fps=tolerance_at_30_fps,
        confidence_requirement=0.0,
    )
    return frozenset(identities)


def _choice_key(evaluation: StartChoiceEvaluation) -> tuple[int, int, int, int, float]:
    return (
        evaluation.net,
        -len(evaluation.broken),
        -evaluation.changed,
        int(evaluation.model_id == "logistic_regression"),
        evaluation.cutoff,
    )


def choose_best_configuration(
    evaluations: Sequence[StartChoiceEvaluation],
) -> StartChoiceEvaluation:
    """Choose net gain, then breaks, changes, model, and cut-off."""
    if not evaluations:
        raise ValueError("no start-model configurations were evaluated")
    return max(evaluations, key=_choice_key)


def passes_development_gate(evaluation: StartChoiceEvaluation) -> bool:
    """Keep choices with at least 20 net repairs and at most one break per five repairs."""
    return evaluation.net >= 20 and 5 * len(evaluation.broken) <= len(
        evaluation.repaired
    )


def choose_deployable_configuration(
    evaluations: Sequence[StartChoiceEvaluation],
) -> StartChoiceEvaluation:
    """Choose the strongest result that meets the brief's development gate."""
    passing = [
        evaluation for evaluation in evaluations if passes_development_gate(evaluation)
    ]
    if not passing:
        raise ValueError("no start-model configuration passed the development gate")
    return max(passing, key=_choice_key)


def _choose_inner_configuration(
    evaluations: Sequence[StartChoiceEvaluation],
) -> StartChoiceEvaluation | None:
    """Choose a safe inner-group setting, or keep the baseline when none helps."""
    passing = [
        evaluation for evaluation in evaluations if passes_development_gate(evaluation)
    ]
    return max(passing, key=_choice_key) if passing else None


def nested_held_out_evaluation(
    action_rows: Sequence[ActionRow],
    targets: ActionTargetAssignments,
    config: RallyStartModelConfig,
    held_out_scores: Mapping[str, Mapping[ActionIdentity, float]],
    baseline: ContactStreams,
    spans: Sequence[FixedSpan],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    group_by_fixture: Mapping[str, str],
) -> tuple[
    StartChoiceEvaluation,
    StartChoiceEvaluation,
    list[dict[str, object]],
]:
    """Choose each outer group's setting without using that group's labels."""
    selections: dict[SectionIdentity, ActionRow] = {}
    choices: list[dict[str, object]] = []
    for outer_group in config.training_groups:
        inner_groups = tuple(
            group for group in config.training_groups if group != outer_group
        )
        inner_scores: dict[str, dict[ActionIdentity, float]] = {
            spec.model_id: {} for spec in config.models
        }
        for inner_group in inner_groups:
            training_rows = [
                row
                for row in action_rows
                if row.candidate.group not in {outer_group, inner_group}
            ]
            scoring_rows = [
                row for row in action_rows if row.candidate.group == inner_group
            ]
            for spec in config.models:
                model = fit_action_model(spec, training_rows, targets)
                inner_scores[spec.model_id].update(
                    predict_action_scores(model, scoring_rows)
                )

        inner_fixtures = {
            fixture
            for fixture, group in group_by_fixture.items()
            if group in inner_groups
        }
        inner_rows = [row for row in action_rows if row.candidate.group in inner_groups]
        inner_spans = [span for span in spans if span.fixture in inner_fixtures]
        inner_events = {
            fixture: events
            for fixture, events in baseline.events_by_fixture.items()
            if fixture in inner_fixtures
        }
        inner_baseline = apply_selected_actions(inner_spans, inner_events, {})
        inner_evaluations = tuple(
            _evaluate_configuration(
                spec.model_id,
                cutoff,
                inner_rows,
                inner_scores[spec.model_id],
                inner_baseline,
                inner_spans,
                labels,
                fps_by_fixture,
                group_by_fixture,
            )
            for spec in config.models
            for cutoff in config.selection_cutoffs
        )
        chosen = _choose_inner_configuration(inner_evaluations)
        if chosen is None:
            choices.append({"group": outer_group, "action": "keep"})
            continue
        outer_rows = [row for row in action_rows if row.candidate.group == outer_group]
        outer_scores = {
            row.identity: held_out_scores[chosen.model_id][row.identity]
            for row in outer_rows
        }
        outer_selections = select_actions(outer_rows, outer_scores, chosen.cutoff)
        selections.update(outer_selections)
        choices.append(
            {
                "group": outer_group,
                "action": "model",
                "model_id": chosen.model_id,
                "cutoff": chosen.cutoff,
                "inner_net_sections": chosen.net,
                "inner_repaired_sections": len(chosen.repaired),
                "inner_broken_sections": len(chosen.broken),
            }
        )

    revised = apply_selected_actions(spans, baseline.events_by_fixture, selections)
    baseline_scored = apply_whole_rally_alternation(baseline)
    revised_scored = apply_whole_rally_alternation(revised)
    action_counts = Counter(row.action for row in selections.values())
    evaluations: list[StartChoiceEvaluation] = []
    for tolerance in (5, 10):
        baseline_ids = _fully_correct_ids(
            baseline_scored,
            labels,
            fps_by_fixture,
            tolerance_at_30_fps=tolerance,
        )
        revised_ids = _fully_correct_ids(
            revised_scored,
            labels,
            fps_by_fixture,
            tolerance_at_30_fps=tolerance,
        )
        evaluations.append(
            StartChoiceEvaluation(
                model_id="nested_group_choice",
                cutoff=0.0,
                baseline_count=len(baseline_ids),
                revised_count=len(revised_ids),
                repaired=frozenset(revised_ids - baseline_ids),
                broken=frozenset(baseline_ids - revised_ids),
                changed=len(selections),
                action_counts={
                    action: action_counts[action] for action in ACTION_KINDS
                },
            )
        )
    return evaluations[0], evaluations[1], choices


def _groups_count(
    identities: Sequence[SectionIdentity],
    group_by_fixture: Mapping[str, str],
) -> list[dict[str, int | str]]:
    counts = Counter(group_by_fixture[fixture] for fixture, _span_id in identities)
    return [{"group": group, "sections": counts[group]} for group in GROUPS]


def _change_counts(
    repaired: Sequence[SectionIdentity],
    broken: Sequence[SectionIdentity],
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


def _evaluate_configuration(
    model_id: str,
    cutoff: float,
    action_rows: Sequence[ActionRow],
    scores: Mapping[ActionIdentity, float],
    baseline: ContactStreams,
    spans: Sequence[FixedSpan],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    group_by_fixture: Mapping[str, str],
    *,
    tolerance_at_30_fps: int = 5,
) -> StartChoiceEvaluation:
    selections = select_actions(action_rows, scores, cutoff)
    revised = apply_selected_actions(spans, baseline.events_by_fixture, selections)
    baseline_scored = apply_whole_rally_alternation(baseline)
    revised_scored = apply_whole_rally_alternation(revised)
    baseline_ids = _fully_correct_ids(
        baseline_scored,
        labels,
        fps_by_fixture,
        tolerance_at_30_fps=tolerance_at_30_fps,
    )
    revised_ids = _fully_correct_ids(
        revised_scored,
        labels,
        fps_by_fixture,
        tolerance_at_30_fps=tolerance_at_30_fps,
    )
    repaired = revised_ids - baseline_ids
    broken = baseline_ids - revised_ids
    action_counts = Counter(row.action for row in selections.values())
    return StartChoiceEvaluation(
        model_id=model_id,
        cutoff=cutoff,
        baseline_count=len(baseline_ids),
        revised_count=len(revised_ids),
        repaired=frozenset(repaired),
        broken=frozenset(broken),
        changed=len(selections),
        action_counts={action: action_counts[action] for action in ACTION_KINDS},
    )


def _evaluation_payload(
    evaluation: StartChoiceEvaluation,
    group_by_fixture: Mapping[str, str],
) -> dict[str, object]:
    repaired = sorted(evaluation.repaired)
    broken = sorted(evaluation.broken)
    return {
        "model_id": evaluation.model_id,
        "cutoff": evaluation.cutoff,
        "baseline_fully_correct": evaluation.baseline_count,
        "revised_fully_correct": evaluation.revised_count,
        "repaired_sections": len(repaired),
        "broken_sections": len(broken),
        "net_sections": evaluation.net,
        "number_changed": evaluation.changed,
        "action_counts": dict(evaluation.action_counts),
        "changes_by_group": _change_counts(repaired, broken, group_by_fixture),
        "repairs_by_group": _groups_count(repaired, group_by_fixture),
        "breaks_by_group": _groups_count(broken, group_by_fixture),
        "repaired_identities": repaired,
        "broken_identities": broken,
    }


def run_experiment() -> dict[str, object]:
    """Run the four-group action comparison on A-D development predictions."""
    config = load_rally_start_model_config(CONFIG_PATH)
    predictions = load_development_predictions()
    training_fixtures = {
        fixture
        for fixture, group in predictions.group_by_fixture.items()
        if group in GROUPS
    }
    spans = tuple(
        span for span in predictions.spans if span.fixture in training_fixtures
    )
    events_by_fixture = {
        fixture: events
        for fixture, events in predictions.events_by_fixture.items()
        if fixture in training_fixtures
    }
    videos = tuple(
        video for video in predictions.videos if video.fixture in training_fixtures
    )
    raw_videos = _candidate_videos()
    rows = build_candidate_rows(raw_videos, default_group="V")
    if {row.group for row in rows} != set(GROUPS):
        raise ValueError("candidate rows do not cover A-D")
    action_rows = build_action_rows(rows)
    labels = load_human_labels(LABEL_PATH, videos)
    fps_by_fixture = {video.fixture: video.fps for video in videos}
    targets = assign_action_targets(
        action_rows,
        spans,
        events_by_fixture,
        raw_videos,
        labels,
        fps_by_fixture,
        default_group="V",
    )
    held_out_scores = held_out_action_scores(action_rows, targets, config)
    baseline = apply_selected_actions(spans, events_by_fixture, {})
    group_by_fixture = dict(predictions.group_by_fixture)
    evaluations = tuple(
        _evaluate_configuration(
            spec.model_id,
            cutoff,
            action_rows,
            held_out_scores[spec.model_id],
            baseline,
            spans,
            labels,
            fps_by_fixture,
            group_by_fixture,
        )
        for spec in config.models
        for cutoff in config.selection_cutoffs
    )
    nested_evaluation, nested_at_10, nested_choices = nested_held_out_evaluation(
        action_rows,
        targets,
        config,
        held_out_scores,
        baseline,
        spans,
        labels,
        fps_by_fixture,
        group_by_fixture,
    )
    descriptive_best = choose_best_configuration(evaluations)
    chosen = choose_deployable_configuration(evaluations)
    descriptive_best_at_10 = _evaluate_configuration(
        descriptive_best.model_id,
        descriptive_best.cutoff,
        action_rows,
        held_out_scores[descriptive_best.model_id],
        baseline,
        spans,
        labels,
        fps_by_fixture,
        group_by_fixture,
        tolerance_at_30_fps=10,
    )
    chosen_at_10 = _evaluate_configuration(
        chosen.model_id,
        chosen.cutoff,
        action_rows,
        held_out_scores[chosen.model_id],
        baseline,
        spans,
        labels,
        fps_by_fixture,
        group_by_fixture,
        tolerance_at_30_fps=10,
    )
    return {
        "schema": "contact-detector-start-action-model/1",
        "status": "complete",
        "run_id": "start-action-keep-add-replace-held-out",
        "repository_commit": _repository_commit(),
        "labels_used": "A-D labels assign action targets and score A-D only; V labels are not read",
        "inputs": {
            "candidate_inputs": str(TRAINING_INPUT_PATH.relative_to(REPO_ROOT)),
            "candidate_source_commit": "f08621a8",
            "candidate_videos": 32,
            "candidate_lists": 2621,
            "candidate_rows": 5242,
            "model_config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        },
        "groups": list(GROUPS),
        "candidate_rows": len(rows),
        "action_rows": len(action_rows),
        "feature_names": list(ACTION_FEATURE_NAMES),
        "config_count": len(evaluations),
        "development_gate": {
            "minimum_net_sections": 20,
            "maximum_breaks_per_repair": 0.2,
        },
        "nested_held_out_estimate": {
            **_evaluation_payload(nested_evaluation, group_by_fixture),
            "at_10_frames": _evaluation_payload(
                nested_at_10,
                group_by_fixture,
            ),
            "choices_by_group": nested_choices,
            "selection_note": (
                "Each group's model and cut-off were chosen on the other three "
                "groups with an inner grouped split."
            ),
        },
        "descriptive_best": _evaluation_payload(
            descriptive_best,
            group_by_fixture,
        ),
        "descriptive_best_at_10_frames": _evaluation_payload(
            descriptive_best_at_10,
            group_by_fixture,
        ),
        "chosen": _evaluation_payload(chosen, group_by_fixture),
        "chosen_at_10_frames": _evaluation_payload(
            chosen_at_10,
            group_by_fixture,
        ),
        "chosen_note": (
            "This A-D result selected the fixed configuration on the same pooled "
            "out-of-fold scores. The untouched V group provides the held-out check."
        ),
        "decision": "validate_once_then_stop_first_contact_line",
        "decision_reason": (
            "The safe choice clears the 20-section minimum but captures less than "
            "one third of the 300-section full-rally best case."
        ),
        "configurations": [
            _evaluation_payload(evaluation, group_by_fixture)
            for evaluation in evaluations
        ],
    }


def main() -> None:
    """Run the development comparison and write its compact result."""
    payload = run_experiment()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "chosen": payload["chosen"],
                "config_count": payload["config_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
