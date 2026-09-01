"""Fit a small keep-or-review model on the A-D sections.

Strict five-frame results provide the training answers. The model only sees
facts from the predicted section. Each group is scored by a model trained on
the other three groups.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    evaluate_span,
)
from scratch.contact_det_followup.scripts.prediction_io import (
    REPO_ROOT,
    DevelopmentPredictionPack,
    load_development_predictions,
    read_json,
)
from scratch.contact_det_followup.scripts.side_rules import (
    apply_side_decisions,
    choose_simple_alternation,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

GROUPS = ("A", "B", "C", "D")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
EVENT_COUNT_BINS = ("0", "1", "2-3", "4-6", "7+")
FEATURE_NAMES = (
    "predicted_event_count",
    "section_duration_seconds",
    "minimum_score",
    "median_score",
    "mean_weakest_three",
    "shortest_gap_seconds",
    "longest_gap_seconds",
    "start_to_first_seconds",
    "last_to_end_seconds",
    "unanswered_side_count",
)
OUTPUT_PATH = (
    REPO_ROOT / "scratch/contact_det_followup/results/keep_review_development.json"
)
LABEL_PATH = REPO_ROOT / "training/data/shuttleset/annotations/shots_master.csv"

SectionIdentity = tuple[str, int]
ModelFactory = Callable[[], Any]


class ProbabilityModel(Protocol):
    """Minimal fitted-model interface used by the label-free prediction step."""

    classes_: np.ndarray

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class probabilities for feature rows."""


@dataclass(frozen=True)
class KeepReviewRow:
    """One predicted section and its ten label-free model inputs."""

    fixture: str
    group: str
    span_id: int
    fps: float
    features: tuple[float, ...]

    @property
    def identity(self) -> SectionIdentity:
        return (self.fixture, self.span_id)

    @property
    def predicted_event_count(self) -> int:
        return int(self.features[0])


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _event_count_bin(event_count: int) -> str:
    if event_count < 0:
        raise ValueError("predicted event count cannot be negative")
    if event_count <= 1:
        return str(event_count)
    if event_count <= 3:
        return "2-3"
    if event_count <= 6:
        return "4-6"
    return "7+"


def build_feature_vector(span: FixedSpan, fps: float) -> tuple[float, ...]:
    """Build the ten label-free features for one section.

    Empty sections use zero for score and timing summaries.  A one-event
    section has zero contact gaps, while its event score is retained in the
    score summaries.
    """
    frames_per_second = _finite(fps, "fps")
    if frames_per_second <= 0:
        raise ValueError("fps must be positive")
    if span.end_frame < span.start_frame:
        raise ValueError(f"{span.fixture}/{span.span_id}: section bounds are reversed")

    events = span.events
    scores = tuple(_finite(event.timing_score, "contact score") for event in events)
    event_count = len(events)
    if event_count:
        ordered_scores = tuple(sorted(scores))
        minimum_score = ordered_scores[0]
        median_score = float(np.median(np.asarray(scores, dtype=np.float64)))
        mean_weakest_three = float(np.mean(ordered_scores[:3]))
        start_to_first = (events[0].frame - span.start_frame) / frames_per_second
        last_to_end = (span.end_frame - events[-1].frame) / frames_per_second
        unanswered_sides = sum(event.predicted_side is None for event in events)
    else:
        minimum_score = 0.0
        median_score = 0.0
        mean_weakest_three = 0.0
        start_to_first = 0.0
        last_to_end = 0.0
        unanswered_sides = 0

    if event_count >= 2:
        gaps = tuple(
            (later.frame - earlier.frame) / frames_per_second
            for earlier, later in pairwise(events)
        )
        shortest_gap = min(gaps)
        longest_gap = max(gaps)
    else:
        shortest_gap = 0.0
        longest_gap = 0.0

    features = (
        float(event_count),
        (span.end_frame - span.start_frame) / frames_per_second,
        minimum_score,
        median_score,
        mean_weakest_three,
        shortest_gap,
        longest_gap,
        start_to_first,
        last_to_end,
        float(unanswered_sides),
    )
    if len(features) != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError(f"{span.fixture}/{span.span_id}: feature vector is not finite")
    return features


def build_feature_rows(
    spans: Sequence[FixedSpan],
    fps_by_fixture: Mapping[str, float],
    group_by_fixture: Mapping[str, str],
) -> tuple[KeepReviewRow, ...]:
    """Build one feature row for every predicted section."""
    rows: list[KeepReviewRow] = []
    seen: set[SectionIdentity] = set()
    for span in spans:
        identity = (span.fixture, span.span_id)
        if identity in seen:
            raise ValueError(f"duplicate section identity {identity}")
        seen.add(identity)
        group = group_by_fixture.get(span.fixture)
        if group not in GROUPS:
            raise ValueError(f"{span.fixture}: section is not in A-D")
        if span.fixture not in fps_by_fixture:
            raise KeyError(f"missing fps for {span.fixture}")
        rows.append(
            KeepReviewRow(
                fixture=span.fixture,
                group=group,
                span_id=span.span_id,
                fps=float(fps_by_fixture[span.fixture]),
                features=build_feature_vector(span, fps_by_fixture[span.fixture]),
            )
        )
    return tuple(rows)


def make_keep_review_model() -> Pipeline:
    """Construct the fixed StandardScaler plus balanced logistic model."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=20260824,
                ),
            ),
        ]
    )


def _feature_matrix(rows: Sequence[KeepReviewRow]) -> np.ndarray:
    matrix = np.asarray([row.features for row in rows], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("feature rows have the wrong shape")
    if not np.isfinite(matrix).all():
        raise ValueError("feature rows contain a non-finite value")
    return matrix


def predict_keep_probabilities(
    model: ProbabilityModel,
    rows: Sequence[KeepReviewRow],
) -> dict[SectionIdentity, float]:
    """Predict one positive-class keep probability per feature row.

    This function accepts only predictions and derived features.  It does not
    accept labels, so callers cannot accidentally make runtime decisions from
    the scoring target.
    """
    if not rows:
        return {}
    identities = [row.identity for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("prediction rows contain duplicate section identities")
    probabilities = np.asarray(
        model.predict_proba(_feature_matrix(rows)), dtype=np.float64
    )
    classes = np.asarray(model.classes_)
    positive_indices = np.flatnonzero(classes == 1)
    if positive_indices.size != 1:
        raise ValueError("keep model must expose exactly one positive class")
    positive = probabilities[:, int(positive_indices[0])]
    if probabilities.shape[0] != len(rows) or not np.isfinite(positive).all():
        raise ValueError("keep model returned malformed probabilities")
    if np.any((positive < 0.0) | (positive > 1.0)):
        raise ValueError("keep model returned probabilities outside zero to one")
    return {
        identity: float(probability)
        for identity, probability in zip(identities, positive, strict=True)
    }


def _target_values(
    rows: Sequence[KeepReviewRow],
    targets: Mapping[SectionIdentity, bool] | Sequence[bool],
) -> np.ndarray:
    if isinstance(targets, Mapping):
        missing = [row.identity for row in rows if row.identity not in targets]
        if missing:
            raise KeyError(f"missing keep targets for {missing[0]}")
        values = [bool(targets[row.identity]) for row in rows]
    else:
        if len(targets) != len(rows):
            raise ValueError("target count does not match feature rows")
        values = [bool(value) for value in targets]
    return np.asarray(values, dtype=np.int8)


def cross_fit_probabilities(
    rows: Sequence[KeepReviewRow],
    targets: Mapping[SectionIdentity, bool] | Sequence[bool],
    *,
    model_factory: ModelFactory | None = None,
) -> dict[SectionIdentity, float]:
    """Fit on three groups and predict the fourth group in every fold."""
    if not rows:
        return {}
    groups = {row.group for row in rows}
    if groups != set(GROUPS):
        raise ValueError(f"cross-fit rows must cover exactly {GROUPS}")
    identities = [row.identity for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("cross-fit rows contain duplicate section identities")
    target_values = _target_values(rows, targets)
    factory = make_keep_review_model if model_factory is None else model_factory
    probabilities: dict[SectionIdentity, float] = {}
    for held_out_group in GROUPS:
        training_rows = tuple(row for row in rows if row.group != held_out_group)
        held_out_rows = tuple(row for row in rows if row.group == held_out_group)
        if not held_out_rows:
            raise ValueError(f"held-out group {held_out_group} has no sections")
        training_indices = [
            index for index, row in enumerate(rows) if row.group != held_out_group
        ]
        model = factory()
        model.fit(_feature_matrix(training_rows), target_values[training_indices])
        fold_probabilities = predict_keep_probabilities(model, held_out_rows)
        overlap = probabilities.keys() & fold_probabilities.keys()
        if overlap:
            raise ValueError(
                f"cross-fit prediction repeated section {next(iter(overlap))}"
            )
        probabilities.update(fold_probabilities)
    if set(probabilities) != set(identities):
        raise ValueError("cross-fit probabilities do not cover every section")
    return probabilities


def strict_keep_targets(
    spans: Sequence[FixedSpan],
    labels: HumanLabels,
    fps_by_fixture: Mapping[str, float],
    tolerance_at_30_fps: int = 5,
) -> dict[SectionIdentity, bool]:
    """Score strict labels at one base-30 frame tolerance."""
    # score_contact_rallies establishes the repository's ``src`` import path
    # before this sibling scorer is loaded in a direct module invocation.
    from scratch.contact_det.scripts.score_contact_evidence import scale_base30_frames

    if isinstance(tolerance_at_30_fps, bool) or tolerance_at_30_fps < 0:
        raise ValueError("tolerance must be a non-negative integer")
    targets: dict[SectionIdentity, bool] = {}
    for span in spans:
        identity = (span.fixture, span.span_id)
        if identity in targets:
            raise ValueError(f"duplicate target identity {identity}")
        tolerance = scale_base30_frames(tolerance_at_30_fps, fps_by_fixture[span.fixture])
        score = evaluate_span(
            span,
            labels.rallies.get(span.fixture, ()),
            labels.target_sides,
            tolerance,
            confidence_requirement=0.0,
        )
        targets[identity] = score.fully_correct
    return targets


def _metrics(
    rows: Sequence[KeepReviewRow],
    probabilities: Mapping[SectionIdentity, float],
    targets: Mapping[SectionIdentity, bool],
    threshold: float,
) -> dict[str, int | float | None]:
    accepted = [row for row in rows if probabilities[row.identity] >= threshold]
    correct = sum(bool(targets[row.identity]) for row in accepted)
    return {
        "population_count": len(rows),
        "accepted_count": len(accepted),
        "fully_correct_accepted": correct,
        "precision": correct / len(accepted) if accepted else None,
        "coverage": len(accepted) / len(rows) if rows else 0.0,
    }


def keep_review_curve(
    rows: Sequence[KeepReviewRow],
    probabilities: Mapping[SectionIdentity, float],
    targets: Mapping[SectionIdentity, bool],
    thresholds: Sequence[float] = THRESHOLDS,
) -> tuple[dict[str, Any], ...]:
    """Report pooled, per-group and event-count-bin keep metrics."""
    values = tuple(float(threshold) for threshold in thresholds)
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError("thresholds must be non-empty, sorted, and unique")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("thresholds must be finite values between zero and one")
    expected = {row.identity for row in rows}
    if set(probabilities) != expected or set(targets) != expected:
        raise ValueError(
            "probabilities and targets must cover exactly the feature rows"
        )

    curve: list[dict[str, Any]] = []
    for threshold in values:
        by_group = {
            group: _metrics(
                tuple(row for row in rows if row.group == group),
                probabilities,
                targets,
                threshold,
            )
            for group in GROUPS
        }
        by_event_count = {
            event_bin: _metrics(
                tuple(
                    row
                    for row in rows
                    if _event_count_bin(row.predicted_event_count) == event_bin
                ),
                probabilities,
                targets,
                threshold,
            )
            for event_bin in EVENT_COUNT_BINS
        }
        curve.append(
            {
                "threshold": threshold,
                **_metrics(rows, probabilities, targets, threshold),
                "by_group": by_group,
                "by_event_count": by_event_count,
            }
        )
    return tuple(curve)


def choose_candidate_threshold(
    curve: Sequence[Mapping[str, Any]],
    *,
    minimum_precision: float = 0.90,
    minimum_coverage: float = 0.10,
) -> float | None:
    """Return the lowest fixed threshold satisfying both development gates."""
    if not 0.0 <= minimum_precision <= 1.0 or not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("threshold gates must lie between zero and one")
    eligible: list[float] = []
    for row in curve:
        threshold = float(row["threshold"])
        precision = row.get("precision")
        coverage = row.get("coverage")
        if (
            precision is not None
            and float(precision) >= minimum_precision
            and float(coverage) >= minimum_coverage
        ):
            eligible.append(threshold)
    return min(eligible) if eligible else None


def apply_fixed_side_vote(
    spans: Sequence[FixedSpan],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
    minimum_vote_gap: int = 1,
) -> tuple[tuple[FixedSpan, ...], Mapping[str, tuple[FixedEvent, ...]]]:
    """Apply the already-fixed whole-rally vote before labels are read."""
    if minimum_vote_gap < 0:
        raise ValueError("minimum vote gap cannot be negative")
    decisions = tuple(
        decision
        for span in spans
        for decision in (choose_simple_alternation(span),)
        if decision is not None and decision.score_gap >= minimum_vote_gap
    )
    return apply_side_decisions(spans, events_by_fixture, decisions)


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _prediction_input_provenance(
    path: Path,
    expected_schema: str,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Verify and summarise one saved label-free prediction input record."""
    payload = read_json(path)
    if (
        payload.get("schema") != expected_schema
        or payload.get("status") != "complete"
        or payload.get("labels_read") is not False
    ):
        raise ValueError(f"prediction input {path} is incomplete or not label-free")
    if payload.get("source_commit") != expected_source_commit:
        raise ValueError(f"prediction input {path} has an unexpected source commit")
    raw_videos = payload.get("videos")
    if not isinstance(raw_videos, list):
        raise TypeError(f"prediction input {path} videos must be a list")
    section_count = 0
    for raw_video in raw_videos:
        if not isinstance(raw_video, Mapping):
            raise TypeError(f"prediction input {path} contains a malformed video")
        raw_spans = raw_video.get("spans")
        if not isinstance(raw_spans, list):
            raise TypeError(f"prediction input {path} contains malformed spans")
        section_count += len(raw_spans)
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        raise TypeError(f"prediction input {path} counts must be an object")
    if counts.get("videos") != len(raw_videos) or counts.get("detected_sections") != section_count:
        raise ValueError(f"prediction input {path} counts do not match its rows")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "source_commit": expected_source_commit,
        "videos": len(raw_videos),
        "sections": section_count,
        "labels_read": False,
    }


def _development_streams(
    predictions: DevelopmentPredictionPack,
) -> tuple[tuple[FixedSpan, ...], Mapping[str, tuple[FixedEvent, ...]]]:
    fixtures = {
        fixture
        for fixture, group in predictions.group_by_fixture.items()
        if group in GROUPS
    }
    if len(fixtures) != 32:
        raise ValueError(f"expected 32 A-D videos, found {len(fixtures)}")
    spans = tuple(span for span in predictions.spans if span.fixture in fixtures)
    events = {
        fixture: values
        for fixture, values in predictions.events_by_fixture.items()
        if fixture in fixtures
    }
    return spans, events


def _fold_report(rows: Sequence[KeepReviewRow]) -> list[dict[str, Any]]:
    return [
        {
            "held_out_group": group,
            "training_groups": [other for other in GROUPS if other != group],
            "training_sections": sum(row.group != group for row in rows),
            "held_out_sections": sum(row.group == group for row in rows),
        }
        for group in GROUPS
    ]


def main() -> None:
    """Fit the descriptive A-D OOF curve and save its compact report."""
    predictions = load_development_predictions()
    training_input_provenance = _prediction_input_provenance(
        predictions.paths[0],
        "contact-rally-start-training-inputs/1",
        "f08621a8",
    )
    validation_input_provenance = _prediction_input_provenance(
        predictions.paths[1],
        "contact-rally-start-validation-inputs/1",
        "90bdcda2",
    )
    development_spans, development_events = _development_streams(predictions)
    revised_spans, _revised_events = apply_fixed_side_vote(
        development_spans,
        development_events,
        minimum_vote_gap=1,
    )
    development_videos = tuple(
        video
        for video in predictions.videos
        if predictions.group_by_fixture[video.fixture] in GROUPS
    )
    labels = load_human_labels(LABEL_PATH, development_videos)
    fps_by_fixture = {video.fixture: video.fps for video in development_videos}
    groups_by_fixture = {
        fixture: predictions.group_by_fixture[fixture] for fixture in fps_by_fixture
    }
    rows = build_feature_rows(revised_spans, fps_by_fixture, groups_by_fixture)
    targets_5 = strict_keep_targets(revised_spans, labels, fps_by_fixture, 5)
    targets_10 = strict_keep_targets(revised_spans, labels, fps_by_fixture, 10)
    probabilities = cross_fit_probabilities(rows, targets_5)
    curve_at_5_frames = keep_review_curve(rows, probabilities, targets_5)
    curve_at_10_frames = keep_review_curve(rows, probabilities, targets_10)
    candidate_threshold = choose_candidate_threshold(curve_at_5_frames)
    payload = {
        "schema": "contact-detector-keep-review-development/1",
        "run_id": "logistic-keep-review-a-d-oof",
        "repository_commit": _repository_commit(),
        "result_type": "Descriptive pooled out-of-fold curve on A-D sections",
        "labels_used": (
            "A-D labels set the five-frame training target and decision. The same "
            "held-out scores are also measured at ten frames."
        ),
        "runtime_inputs": list(FEATURE_NAMES),
        "video_groups": "A, B, C, and D; each held-out group's probabilities use the other three groups",
        "side_vote": {"rule": "simple_alternation_vote", "minimum_vote_gap": 1},
        "prediction_inputs": {
            "training": training_input_provenance,
            "validation": {
                **validation_input_provenance,
                "used_for_keep_review": False,
            },
        },
        "sections": len(rows),
        "baseline_strict_correct": sum(targets_5.values()),
        "baseline_strict_correct_at_5_frames": sum(targets_5.values()),
        "baseline_strict_correct_at_10_frames": sum(targets_10.values()),
        "model": {
            "pipeline": "StandardScaler + LogisticRegression",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 20260824,
        },
        "folds": _fold_report(rows),
        "thresholds": list(THRESHOLDS),
        "curve_at_5_frames": list(curve_at_5_frames),
        "curve_at_10_frames": list(curve_at_10_frames),
        "candidate_threshold_at_5_frames": candidate_threshold,
        "decision_at_5_frames": "continue" if candidate_threshold is not None else "stop",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
