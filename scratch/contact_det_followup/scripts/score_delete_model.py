"""Fit and score a small held-out chooser for deleting one event per section."""

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
    evaluate_span,
)
from scratch.contact_det_followup.scripts.prediction_io import (
    REPO_ROOT,
    load_development_predictions,
    read_json,
)
from scratch.contact_det_followup.scripts.score_start_model import (
    _fully_correct_ids,
    apply_whole_rally_alternation,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    HumanLabels,
    scale_base30_frames,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model_config import (
    CONFIG_SHA256,
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
COMBINED_RESULT_PATH = REPO_ROOT / "scratch/contact_det_followup/results/combined_best_case.json"
OUTPUT_PATH = REPO_ROOT / "scratch/contact_det_followup/results/delete_model_development.json"
GROUPS = ("A", "B", "C", "D")
DELETE_FEATURE_NAMES = (
    "event_contact_score",
    "event_count",
    "section_duration_at_30_fps",
    "event_index_from_start",
    "event_index_from_end",
    "event_is_first",
    "event_is_last",
    "score_minus_section_minimum",
    "score_minus_section_median",
    "previous_gap_at_30_fps",
    "next_gap_at_30_fps",
    "side_known",
)

DeleteIdentity = tuple[str, int, int]
SectionIdentity = tuple[str, int]


@dataclass(frozen=True)
class DeleteRow:
    """One existing event and the fixed inputs for deleting it."""

    fixture: str
    group: str
    fps: float
    span_id: int
    section_start_frame: int
    section_end_frame: int
    frame: int
    contact_score: float
    predicted_side: str | None
    features: tuple[float, ...]

    @property
    def identity(self) -> DeleteIdentity:
        return self.fixture, self.span_id, self.frame

    @property
    def section_identity(self) -> SectionIdentity:
        return self.fixture, self.span_id


@dataclass(frozen=True)
class DeleteTarget:
    """The label-derived training answer for one delete row."""

    included_in_training: bool
    positive: bool
    section_status: str
    baseline_fully_correct: bool
    deletion_fully_correct: bool


@dataclass(frozen=True)
class DeleteTargetAssignments:
    """Delete targets and the status assigned to each event section."""

    by_event: Mapping[DeleteIdentity, DeleteTarget]
    section_statuses: Mapping[SectionIdentity, str]


@dataclass(frozen=True)
class DeleteEvaluation:
    """Strict fully-correct changes from one model and cut-off."""

    model_id: str
    cutoff: float
    baseline_count: int
    revised_count: int
    repaired: frozenset[SectionIdentity]
    broken: frozenset[SectionIdentity]
    selected: Mapping[SectionIdentity, DeleteRow]
    scored_rows: int
    sections_with_rows: int

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


def _candidate_input_record() -> Mapping[str, Any]:
    """Read and pin the saved label-free candidate input record."""
    payload = read_json(TRAINING_INPUT_PATH)
    counts = payload.get("counts")
    if (
        payload.get("schema") != "contact-rally-start-training-inputs/1"
        or payload.get("status") != "complete"
        or payload.get("labels_read") is not False
        or payload.get("source_commit") != "f08621a8"
        or payload.get("groups") != list(GROUPS)
        or not isinstance(counts, Mapping)
        or counts.get("videos") != 32
        or counts.get("detected_sections") != 2850
        or counts.get("kept_contacts") != 26459
    ):
        raise ValueError("saved candidate input record differs")
    return payload


def build_delete_rows(
    spans: Sequence[FixedSpan],
    group_by_fixture: Mapping[str, str],
    fps_by_fixture: Mapping[str, float],
) -> tuple[DeleteRow, ...]:
    """Build the twelve fixed, label-free inputs for every in-section event."""
    rows: list[DeleteRow] = []
    identities: set[DeleteIdentity] = set()
    for span in spans:
        group = group_by_fixture[span.fixture]
        fps = fps_by_fixture[span.fixture]
        event_count = len(span.events)
        if event_count == 0:
            continue
        scores = np.asarray([event.timing_score for event in span.events], dtype=float)
        section_minimum = float(np.min(scores))
        section_median = float(np.median(scores))
        section_duration = (span.end_frame - span.start_frame) * 30.0 / fps
        for event_index, event in enumerate(span.events):
            identity = (span.fixture, span.span_id, event.frame)
            if identity in identities:
                raise ValueError(f"{identity}: delete event identity repeats")
            identities.add(identity)
            previous_gap = (
                0.0
                if event_index == 0
                else (event.frame - span.events[event_index - 1].frame) * 30.0 / fps
            )
            next_gap = (
                0.0
                if event_index == event_count - 1
                else (span.events[event_index + 1].frame - event.frame) * 30.0 / fps
            )
            features = (
                event.timing_score,
                float(event_count),
                section_duration,
                float(event_index),
                float(event_count - event_index - 1),
                float(event_index == 0),
                float(event_index == event_count - 1),
                event.timing_score - section_minimum,
                event.timing_score - section_median,
                previous_gap,
                next_gap,
                float(event.predicted_side is not None),
            )
            if not np.isfinite(features).all():
                raise ValueError(f"{identity}: delete features are not finite")
            rows.append(
                DeleteRow(
                    span.fixture,
                    group,
                    fps,
                    span.span_id,
                    span.start_frame,
                    span.end_frame,
                    event.frame,
                    event.timing_score,
                    event.predicted_side,
                    features,
                )
            )
    return tuple(rows)


def _span_lookup(spans: Sequence[FixedSpan]) -> dict[SectionIdentity, FixedSpan]:
    output: dict[SectionIdentity, FixedSpan] = {}
    for span in spans:
        identity = (span.fixture, span.span_id)
        if identity in output:
            raise ValueError(f"{identity}: section identity repeats")
        output[identity] = span
    return output


def apply_selected_deletions(
    spans: Sequence[FixedSpan],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
    selections: Mapping[SectionIdentity, DeleteRow],
) -> ContactStreams:
    """Delete exactly the selected existing event while retaining all others."""
    span_by_identity = _span_lookup(spans)
    fixture_events: dict[str, list[FixedEvent]] = {
        fixture: list(events) for fixture, events in events_by_fixture.items()
    }
    for fixture, events in fixture_events.items():
        ordered = tuple(events)
        if tuple(sorted(ordered, key=lambda event: event.frame)) != ordered:
            raise ValueError(f"{fixture}: event order differs")
        if any(event.fixture != fixture for event in ordered):
            raise ValueError(f"{fixture}: event fixture differs")

    for identity, row in selections.items():
        if identity not in span_by_identity or row.section_identity != identity:
            raise ValueError(f"{identity}: selected delete identity differs")
        span_events = span_by_identity[identity].events
        matching_span = [event for event in span_events if event.frame == row.frame]
        if len(matching_span) != 1:
            raise ValueError(f"{identity}: selected event is not in its section")
        current_events = fixture_events.get(row.fixture, [])
        matching_stream = [event for event in current_events if event.frame == row.frame]
        expected_event = matching_span[0]
        if len(matching_stream) != 1 or matching_stream[0] != expected_event:
            raise ValueError(f"{identity}: selected event differs from stream")
        fixture_events[row.fixture] = [
            event for event in current_events if event.frame != row.frame
        ]

    frozen_events = {
        fixture: tuple(sorted(events, key=lambda event: event.frame))
        for fixture, events in fixture_events.items()
    }
    output_spans: list[FixedSpan] = []
    assigned_events: set[tuple[str, int]] = set()
    previous_end_by_fixture: dict[str, int] = {}
    for span in spans:
        previous_end = previous_end_by_fixture.get(span.fixture, -1)
        if span.start_frame < previous_end:
            raise ValueError(f"{span.fixture}/{span.span_id}: spans overlap")
        section_events = tuple(
            event
            for event in frozen_events.get(span.fixture, ())
            if span.start_frame <= event.frame < span.end_frame
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
                span.start_frame,
                span.end_frame,
                section_events,
            )
        )
        previous_end_by_fixture[span.fixture] = span.end_frame
    return ContactStreams(tuple(output_spans), frozen_events)


def _section_label_statuses(
    spans: Sequence[FixedSpan],
    labels: HumanLabels,
) -> dict[SectionIdentity, str]:
    """Mark no-rally and ambiguous sections as explicit negative statuses."""
    rally_sections: Counter[tuple[str, str]] = Counter()
    matching_by_section: dict[SectionIdentity, tuple[RallyReference, ...]] = {}
    for span in spans:
        matching = tuple(
            rally
            for rally in labels.rallies.get(span.fixture, ())
            if any(span.start_frame <= frame < span.end_frame for frame in rally.frames)
        )
        identity = (span.fixture, span.span_id)
        matching_by_section[identity] = matching
        for rally in matching:
            rally_sections[(span.fixture, rally.rally_id)] += 1

    statuses: dict[SectionIdentity, str] = {}
    for identity, matching in matching_by_section.items():
        if not matching:
            statuses[identity] = "no_labelled_rally"
        elif len(matching) > 1:
            statuses[identity] = "more_than_one_labelled_rally"
        elif rally_sections[(identity[0], matching[0].rally_id)] > 1:
            statuses[identity] = "labelled_rally_touches_more_than_one_section"
        else:
            statuses[identity] = "single_labelled_rally"
    return statuses


def _side_voted_span(span: FixedSpan) -> FixedSpan:
    """Apply the existing minimum-one-vote-gap side rule to one span."""
    streams = ContactStreams((span,), {span.fixture: span.events})
    return apply_whole_rally_alternation(streams).spans[0]


def _span_fully_correct(
    span: FixedSpan,
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    *,
    tolerance_at_30_fps: int,
) -> bool:
    revised = _side_voted_span(span)
    tolerance = scale_base30_frames(
        tolerance_at_30_fps,
        fps_by_fixture[span.fixture],
    )
    return evaluate_span(
        revised,
        labels.rallies.get(span.fixture, ()),
        labels.target_sides,
        tolerance,
        confidence_requirement=0.0,
    ).fully_correct


def assign_delete_targets(
    rows: Sequence[DeleteRow],
    spans: Sequence[FixedSpan],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
) -> DeleteTargetAssignments:
    """Assign one timing-and-side deletion positive per section at ±5 frames."""
    rows_by_section: dict[SectionIdentity, list[DeleteRow]] = {}
    for row in rows:
        rows_by_section.setdefault(row.section_identity, []).append(row)
    span_by_identity = _span_lookup(spans)
    statuses = _section_label_statuses(spans, labels)
    by_event: dict[DeleteIdentity, DeleteTarget] = {}
    for section_identity, section_rows in rows_by_section.items():
        span = span_by_identity[section_identity]
        section_status = statuses[section_identity]
        baseline = _span_fully_correct(
            span,
            labels,
            fps_by_fixture,
            tolerance_at_30_fps=5,
        )
        deletion_correct: dict[DeleteIdentity, bool] = {}
        for row in section_rows:
            revised = FixedSpan(
                span.fixture,
                span.span_id,
                span.start_frame,
                span.end_frame,
                tuple(event for event in span.events if event.frame != row.frame),
            )
            deletion_correct[row.identity] = (
                section_status == "single_labelled_rally"
                and not baseline
                and _span_fully_correct(
                    revised,
                    labels,
                    fps_by_fixture,
                    tolerance_at_30_fps=5,
                )
            )
        positive_rows = [row for row in section_rows if deletion_correct[row.identity]]
        positive_identity = (
            min(positive_rows, key=lambda row: (row.contact_score, row.frame)).identity
            if positive_rows
            else None
        )
        for row in section_rows:
            by_event[row.identity] = DeleteTarget(
                included_in_training=True,
                positive=(
                    section_status == "single_labelled_rally"
                    and row.identity == positive_identity
                ),
                section_status=section_status,
                baseline_fully_correct=baseline,
                deletion_fully_correct=deletion_correct[row.identity],
            )
    if set(by_event) != {row.identity for row in rows}:
        raise ValueError("delete target coverage differs")
    return DeleteTargetAssignments(by_event, statuses)


def _feature_array(rows: Sequence[DeleteRow]) -> np.ndarray:
    values = np.asarray([row.features for row in rows], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(DELETE_FEATURE_NAMES):
        raise ValueError("delete model feature shape differs")
    if not np.isfinite(values).all():
        raise ValueError("delete model features are not finite")
    return values


def make_delete_model(spec: ModelSpec) -> Any:
    """Construct one model from the reviewed fixed rally-start settings."""
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
    raise ValueError(f"unsupported delete model: {spec.kind}")


def fit_delete_model(
    spec: ModelSpec,
    training_rows: Sequence[DeleteRow],
    targets: DeleteTargetAssignments,
) -> Any:
    """Fit one delete model on included rows and label-derived targets."""
    included_rows = [
        row
        for row in training_rows
        if targets.by_event[row.identity].included_in_training
    ]
    values = np.asarray(
        [targets.by_event[row.identity].positive for row in included_rows],
        dtype=np.uint8,
    )
    if not included_rows or set(values.tolist()) != {0, 1}:
        raise ValueError("delete model training needs positive and negative examples")
    model = make_delete_model(spec)
    model.fit(_feature_array(included_rows), values)
    return model


def predict_delete_scores(
    model: Any,
    rows: Sequence[DeleteRow],
) -> dict[DeleteIdentity, float]:
    """Predict delete probabilities without accepting labels or targets."""
    if not rows:
        return {}
    classes = np.asarray(model.classes_)
    positive_positions = np.flatnonzero(classes == 1)
    if len(positive_positions) != 1:
        raise ValueError("delete model positive class differs")
    probabilities = np.asarray(model.predict_proba(_feature_array(rows)))
    scores = probabilities[:, int(positive_positions[0])]
    if len(scores) != len(rows) or not np.isfinite(scores).all():
        raise ValueError("delete model scores differ")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("delete model scores must be between zero and one")
    return {row.identity: float(score) for row, score in zip(rows, scores, strict=True)}


def held_out_delete_scores(
    rows: Sequence[DeleteRow],
    targets: DeleteTargetAssignments,
    config: RallyStartModelConfig,
) -> dict[str, dict[DeleteIdentity, float]]:
    """Fit on three groups and predict every event in the held-out group."""
    if {row.group for row in rows} != set(config.training_groups):
        raise ValueError("delete row groups differ")
    output: dict[str, dict[DeleteIdentity, float]] = {
        spec.model_id: {} for spec in config.models
    }
    for held_out_group in config.training_groups:
        training_rows = [row for row in rows if row.group != held_out_group]
        held_out_rows = [row for row in rows if row.group == held_out_group]
        for spec in config.models:
            model = fit_delete_model(spec, training_rows, targets)
            scores = predict_delete_scores(model, held_out_rows)
            if set(output[spec.model_id]) & set(scores):
                raise ValueError("held-out delete score identities repeat")
            output[spec.model_id].update(scores)
    expected = {row.identity for row in rows}
    if any(set(scores) != expected for scores in output.values()):
        raise ValueError("held-out delete score coverage differs")
    return output


def fit_final_delete_model(
    spec: ModelSpec,
    rows: Sequence[DeleteRow],
    targets: DeleteTargetAssignments,
) -> Any:
    """Fit one delete model on all supplied development rows."""
    return fit_delete_model(spec, rows, targets)


def select_deletions(
    rows: Sequence[DeleteRow],
    scores: Mapping[DeleteIdentity, float],
    cutoff: float,
) -> dict[SectionIdentity, DeleteRow]:
    """Select at most one deletion per section without accepting labels."""
    if not 0.0 < cutoff < 1.0:
        raise ValueError("delete selection cut-off must be between zero and one")
    if set(scores) != {row.identity for row in rows}:
        raise ValueError("delete selection score coverage differs")
    rows_by_section: dict[SectionIdentity, list[DeleteRow]] = {}
    for row in rows:
        rows_by_section.setdefault(row.section_identity, []).append(row)
    selected: dict[SectionIdentity, DeleteRow] = {}
    for section_identity, section_rows in rows_by_section.items():
        eligible = [row for row in section_rows if scores[row.identity] >= cutoff]
        if eligible:
            selected[section_identity] = max(
                eligible,
                key=lambda row: (
                    scores[row.identity],
                    -row.contact_score,
                    -row.frame,
                ),
            )
    return selected


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
    repaired_counts = Counter(group_by_fixture[fixture] for fixture, _span_id in repaired)
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
    rows: Sequence[DeleteRow],
    scores: Mapping[DeleteIdentity, float],
    baseline: ContactStreams,
    spans: Sequence[FixedSpan],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    *,
    tolerance_at_30_fps: int,
) -> DeleteEvaluation:
    selections = select_deletions(rows, scores, cutoff)
    revised = apply_selected_deletions(spans, baseline.events_by_fixture, selections)
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
    return DeleteEvaluation(
        model_id,
        cutoff,
        len(baseline_ids),
        len(revised_ids),
        frozenset(revised_ids - baseline_ids),
        frozenset(baseline_ids - revised_ids),
        selections,
        len(rows),
        len({row.section_identity for row in rows}),
    )


def _evaluation_payload(
    evaluation: DeleteEvaluation,
    group_by_fixture: Mapping[str, str],
    ceiling_recoverable: int | None = None,
) -> dict[str, object]:
    repaired = sorted(evaluation.repaired)
    broken = sorted(evaluation.broken)
    selected = sorted(evaluation.selected.values(), key=lambda row: row.identity)
    payload: dict[str, object] = {
        "model_id": evaluation.model_id,
        "cutoff": evaluation.cutoff,
        "baseline_fully_correct": evaluation.baseline_count,
        "revised_fully_correct": evaluation.revised_count,
        "repaired_sections": len(repaired),
        "broken_sections": len(broken),
        "net_sections": evaluation.net,
        "number_changed": len(selected),
        "action_counts": {"delete": len(selected)},
        "action_coverage": {
            "scored_rows": evaluation.scored_rows,
            "sections_with_rows": evaluation.sections_with_rows,
            "selected_deletions": len(selected),
            "selection_rate": (
                len(selected) / evaluation.sections_with_rows
                if evaluation.sections_with_rows
                else 0.0
            ),
        },
        "changes_by_group": _change_counts(repaired, broken, group_by_fixture),
        "repairs_by_group": _groups_count(repaired, group_by_fixture),
        "breaks_by_group": _groups_count(broken, group_by_fixture),
        "selected_action_identities": [list(row.identity) for row in selected],
        "repaired_identities": repaired,
        "broken_identities": broken,
    }
    if ceiling_recoverable is not None:
        payload["ceiling_recoverable_sections"] = ceiling_recoverable
        payload["recovery_share_of_ceiling"] = (
            len(repaired) / ceiling_recoverable if ceiling_recoverable else 0.0
        )
    return payload


def _choice_key(evaluation: DeleteEvaluation) -> tuple[int, int, int, int, float]:
    return (
        evaluation.net,
        -len(evaluation.broken),
        -len(evaluation.selected),
        int(evaluation.model_id == "logistic_regression"),
        evaluation.cutoff,
    )


def choose_best_configuration(
    evaluations: Sequence[DeleteEvaluation],
) -> DeleteEvaluation:
    """Choose the descriptive best result with deterministic tie ordering."""
    if not evaluations:
        raise ValueError("no delete-model configurations were evaluated")
    return max(evaluations, key=_choice_key)


def passes_development_gate(evaluation: DeleteEvaluation) -> bool:
    """Require at least 30 net sections and no more than one break per five repairs."""
    return evaluation.net >= 30 and 5 * len(evaluation.broken) <= len(evaluation.repaired)


def choose_deployable_configuration(
    evaluations: Sequence[DeleteEvaluation],
) -> DeleteEvaluation | None:
    """Return the strongest gated pooled choice, or stop when none passes."""
    passing = [evaluation for evaluation in evaluations if passes_development_gate(evaluation)]
    return max(passing, key=_choice_key) if passing else None


def _read_combined_ceiling() -> dict[int, dict[str, object]]:
    """Read delete-only ceilings from the corrected combined audit result."""
    payload = read_json(COMBINED_RESULT_PATH)
    if (
        payload.get("schema") != "contact-detector-combined-best-case/2"
        or payload.get("status") != "complete"
    ):
        raise ValueError("combined best-case result is not the corrected schema")
    raw_results = payload.get("results_by_tolerance")
    if not isinstance(raw_results, Mapping):
        raise TypeError("combined best-case tolerance results must be an object")
    output: dict[int, dict[str, object]] = {}
    for tolerance in (5, 10):
        raw_result = raw_results.get(str(tolerance))
        if not isinstance(raw_result, Mapping):
            raise TypeError(f"combined best-case {tolerance}-frame result is missing")
        raw_delete = raw_result.get("delete_only")
        if not isinstance(raw_delete, Mapping):
            raise TypeError(f"combined best-case {tolerance}-frame delete result differs")
        recoverable = raw_delete.get("event_edit_repaired_sections")
        if type(recoverable) is not int or recoverable < 0:
            raise ValueError(f"combined best-case {tolerance}-frame ceiling differs")
        output[tolerance] = {
            "recoverable_sections": recoverable,
            "source": str(COMBINED_RESULT_PATH.relative_to(REPO_ROOT)),
            "schema": payload["schema"],
            "repository_commit": payload.get("repository_commit"),
            "delete_only_event_edit_repaired_sections": recoverable,
        }
    return output


def nested_held_out_evaluation(
    rows: Sequence[DeleteRow],
    targets: DeleteTargetAssignments,
    config: RallyStartModelConfig,
    held_out_scores: Mapping[str, Mapping[DeleteIdentity, float]],
    baseline: ContactStreams,
    spans: Sequence[FixedSpan],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    group_by_fixture: Mapping[str, str],
) -> tuple[DeleteEvaluation, DeleteEvaluation, list[dict[str, object]]]:
    """Estimate outer-group performance after an inner-group gate choice."""
    selections: dict[SectionIdentity, DeleteRow] = {}
    choices: list[dict[str, object]] = []
    for outer_group in config.training_groups:
        inner_groups = tuple(
            group for group in config.training_groups if group != outer_group
        )
        inner_scores: dict[str, dict[DeleteIdentity, float]] = {
            spec.model_id: {} for spec in config.models
        }
        for inner_group in inner_groups:
            fit_rows = [
                row
                for row in rows
                if row.group not in {outer_group, inner_group}
            ]
            score_rows = [row for row in rows if row.group == inner_group]
            for spec in config.models:
                model = fit_delete_model(spec, fit_rows, targets)
                inner_scores[spec.model_id].update(
                    predict_delete_scores(model, score_rows)
                )

        inner_fixtures = {
            fixture
            for fixture, group in group_by_fixture.items()
            if group in inner_groups
        }
        inner_rows = [row for row in rows if row.group in inner_groups]
        inner_spans = [span for span in spans if span.fixture in inner_fixtures]
        inner_events = {
            fixture: events
            for fixture, events in baseline.events_by_fixture.items()
            if fixture in inner_fixtures
        }
        inner_baseline = ContactStreams(tuple(inner_spans), inner_events)
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
                tolerance_at_30_fps=5,
            )
            for spec in config.models
            for cutoff in config.selection_cutoffs
        )
        inner_choice = choose_deployable_configuration(inner_evaluations)
        if inner_choice is None:
            choices.append({"group": outer_group, "action": "keep"})
            continue
        outer_rows = [row for row in rows if row.group == outer_group]
        outer_scores = {
            row.identity: held_out_scores[inner_choice.model_id][row.identity]
            for row in outer_rows
        }
        selections.update(select_deletions(outer_rows, outer_scores, inner_choice.cutoff))
        choices.append(
            {
                "group": outer_group,
                "action": "model",
                "model_id": inner_choice.model_id,
                "cutoff": inner_choice.cutoff,
                "inner_net_sections": inner_choice.net,
                "inner_repaired_sections": len(inner_choice.repaired),
                "inner_broken_sections": len(inner_choice.broken),
            }
        )

    revised = apply_selected_deletions(spans, baseline.events_by_fixture, selections)
    baseline_scored = apply_whole_rally_alternation(baseline)
    revised_scored = apply_whole_rally_alternation(revised)
    evaluations: list[DeleteEvaluation] = []
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
            DeleteEvaluation(
                "nested_group_choice",
                0.0,
                len(baseline_ids),
                len(revised_ids),
                frozenset(revised_ids - baseline_ids),
                frozenset(baseline_ids - revised_ids),
                selections,
                len(rows),
                len({row.section_identity for row in rows}),
            )
        )
    return evaluations[0], evaluations[1], choices


def run_experiment() -> dict[str, object]:
    """Run pooled and nested held-out delete-model comparisons on A-D."""
    config = load_rally_start_model_config(CONFIG_PATH)
    candidate_record = _candidate_input_record()
    predictions = load_development_predictions()
    training_fixtures = {
        fixture
        for fixture, group in predictions.group_by_fixture.items()
        if group in GROUPS
    }
    spans = tuple(span for span in predictions.spans if span.fixture in training_fixtures)
    events_by_fixture = {
        fixture: events
        for fixture, events in predictions.events_by_fixture.items()
        if fixture in training_fixtures
    }
    videos = tuple(video for video in predictions.videos if video.fixture in training_fixtures)
    group_by_fixture = dict(predictions.group_by_fixture)
    fps_by_fixture = {video.fixture: video.fps for video in videos}
    rows = build_delete_rows(spans, group_by_fixture, fps_by_fixture)
    if {row.group for row in rows} != set(GROUPS):
        raise ValueError("delete rows do not cover A-D")
    labels = load_human_labels(LABEL_PATH, videos)
    targets = assign_delete_targets(rows, spans, labels, fps_by_fixture)
    held_out_scores = held_out_delete_scores(rows, targets, config)
    baseline = ContactStreams(spans, events_by_fixture)
    evaluations_at_5 = tuple(
        _evaluate_configuration(
            spec.model_id,
            cutoff,
            rows,
            held_out_scores[spec.model_id],
            baseline,
            spans,
            labels,
            fps_by_fixture,
            tolerance_at_30_fps=5,
        )
        for spec in config.models
        for cutoff in config.selection_cutoffs
    )
    evaluations_at_10 = tuple(
        _evaluate_configuration(
            evaluation.model_id,
            evaluation.cutoff,
            rows,
            held_out_scores[evaluation.model_id],
            baseline,
            spans,
            labels,
            fps_by_fixture,
            tolerance_at_30_fps=10,
        )
        for evaluation in evaluations_at_5
    )
    by_config_at_10 = {
        (evaluation.model_id, evaluation.cutoff): evaluation
        for evaluation in evaluations_at_10
    }
    descriptive_best = choose_best_configuration(evaluations_at_5)
    pooled_choice = choose_deployable_configuration(evaluations_at_5)
    nested_at_5, nested_at_10, nested_choices = nested_held_out_evaluation(
        rows,
        targets,
        config,
        held_out_scores,
        baseline,
        spans,
        labels,
        fps_by_fixture,
        group_by_fixture,
    )
    ceilings = _read_combined_ceiling()
    ceiling_5 = int(ceilings[5]["recoverable_sections"])
    ceiling_10 = int(ceilings[10]["recoverable_sections"])
    source_counts = dict(candidate_record["counts"])
    config_count = len(evaluations_at_5)
    pooled_choice_at_10 = (
        None
        if pooled_choice is None
        else by_config_at_10[(pooled_choice.model_id, pooled_choice.cutoff)]
    )
    decision = "continue" if pooled_choice is not None else "stop"
    return {
        "schema": "contact-detector-delete-action-model/1",
        "status": "complete",
        "run_id": "delete-action-held-out-and-nested",
        "repository_commit": _repository_commit(),
        "labels_used": "A-D labels assign deletion targets and score A-D only; runtime prediction and selection accept no labels",
        "inputs": {
            "candidate_inputs": str(TRAINING_INPUT_PATH.relative_to(REPO_ROOT)),
            "candidate_source_commit": candidate_record["source_commit"],
            "candidate_counts": source_counts,
            "development_prediction_inputs": [
                str(path.relative_to(REPO_ROOT)) for path in predictions.paths
            ],
            "model_config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "model_config_sha256": CONFIG_SHA256,
            "combined_best_case": ceilings,
        },
        "groups": list(GROUPS),
        "sections": len(spans),
        "events_in_sections": sum(len(span.events) for span in spans),
        "delete_rows": len(rows),
        "feature_names": list(DELETE_FEATURE_NAMES),
        "config_count": config_count,
        "development_gate": {
            "minimum_net_sections": 30,
            "maximum_breaks_per_repair": 0.2,
        },
        "ceiling_recoverable": {
            "at_5_frames": ceiling_5,
            "at_10_frames": ceiling_10,
        },
        "descriptive_best": _evaluation_payload(
            descriptive_best,
            group_by_fixture,
            ceiling_5,
        ),
        "descriptive_best_at_10_frames": _evaluation_payload(
            by_config_at_10[(descriptive_best.model_id, descriptive_best.cutoff)],
            group_by_fixture,
            ceiling_10,
        ),
        "gated_pooled_choice": (
            None
            if pooled_choice is None
            else _evaluation_payload(pooled_choice, group_by_fixture, ceiling_5)
        ),
        "gated_pooled_choice_at_10_frames": (
            None
            if pooled_choice_at_10 is None
            else _evaluation_payload(pooled_choice_at_10, group_by_fixture, ceiling_10)
        ),
        "nested_held_out_estimate": {
            "at_5_frames": _evaluation_payload(nested_at_5, group_by_fixture, ceiling_5),
            "at_10_frames": _evaluation_payload(nested_at_10, group_by_fixture, ceiling_10),
            "choices_by_outer_group": nested_choices,
            "selection_note": "Each outer group used an inner two-group fit and three-group gate choice; the outer group was scored only from its other-three-group fit.",
        },
        "decision": decision,
        "decision_reason": (
            "A pooled configuration passed net >=30 and breaks <= repairs/5; proceed to the bounded validation step."
            if decision == "continue"
            else "No pooled configuration passed net >=30 and breaks <= repairs/5; stop the delete-model line."
        ),
        "configurations": [
            {
                "at_5_frames": _evaluation_payload(evaluation, group_by_fixture, ceiling_5),
                "at_10_frames": _evaluation_payload(
                    by_config_at_10[(evaluation.model_id, evaluation.cutoff)],
                    group_by_fixture,
                    ceiling_10,
                ),
            }
            for evaluation in evaluations_at_5
        ],
    }


def main() -> None:
    """Run and save the development delete-model comparison."""
    payload = run_experiment()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "descriptive_best": {
                    key: payload["descriptive_best"][key]
                    for key in (
                        "model_id",
                        "cutoff",
                        "baseline_fully_correct",
                        "revised_fully_correct",
                        "repaired_sections",
                        "broken_sections",
                        "net_sections",
                        "number_changed",
                    )
                },
                "gated_pooled_choice": payload["gated_pooled_choice"],
                "decision": payload["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
