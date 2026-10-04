"""Pick one option per rally sequence from chooser scores.

The first chooser switches from keep to an edit only when the edit scores
strictly higher. Later choosers keep the previous stage's choice unless their
best option beats it by a fixed margin.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from annotator.sequence.edits import (
    EditKind,
    SequenceOption,
    edit_priority,
    option_priority,
)

MIN_EDIT_ADVANTAGE = 0.05


def choose_edits(
    options: Sequence[SequenceOption], scores: np.ndarray, rows: Sequence[int], minimum_score: float,
) -> dict[int, int]:
    """Choose one edit per sequence, falling back to keep.

    An edit must have a finite score, reach ``minimum_score`` and strictly beat
    keep. Equal scores go to the simpler kind, then the earlier frame.

    :param options: The shared option pool.
    :param scores: One score per pool option.
    :param rows: Pool rows this chooser scored, in pool order.
    :param minimum_score: Lowest score an edit may have.
    :return: Chosen pool row by span ID.
    """
    rows_by_span: dict[int, list[int]] = {}
    for row in rows:
        rows_by_span.setdefault(options[row].span_id, []).append(row)
    chosen = {}
    for span_id, span_rows in rows_by_span.items():
        keeps = [row for row in span_rows if options[row].edit.kind == EditKind.KEEP]
        if len(keeps) != 1:
            raise ValueError(f"span {span_id}: exactly one keep option is required")
        keep_score = float(scores[keeps[0]])
        better = []
        for row in span_rows:
            score = float(scores[row])
            is_edit = options[row].edit.kind != EditKind.KEEP
            if is_edit and np.isfinite(score) and score >= minimum_score and score > keep_score:
                better.append(row)
        chosen[span_id] = keeps[0]
        if better:
            chosen[span_id] = min(better, key=lambda row: (-float(scores[row]), edit_priority(options[row].edit)))
    return chosen


def choose_best_options(options: Sequence[SequenceOption], scores: np.ndarray) -> dict[int, int]:
    """Choose each sequence's highest-scored option.

    Ties prefer fewer added contacts, then the simpler edit, then the earlier option.
    """
    if len(options) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("option scores are incomplete")
    best: dict[int, int] = {}
    for row, option in enumerate(options):
        previous = best.get(option.span_id)
        if previous is not None:
            score = scores[row]
            previous_score = scores[previous]
            is_tie = score == previous_score
            if score < previous_score or (is_tie and option_priority(option) >= option_priority(options[previous])):
                continue
        best[option.span_id] = row
    return best


def choose_with_margin(
    options: Sequence[SequenceOption],
    scores: np.ndarray,
    previous: Mapping[int, int],
    minimum_advantage: float = MIN_EDIT_ADVANTAGE,
) -> dict[int, int]:
    """Keep each previous choice unless the best option clearly outscores it.

    The previous choice's score is the highest score of any option producing
    the same sequence, so an equal output reached by another edit counts.

    :param options: The shared option pool.
    :param scores: One score per pool option.
    :param previous: Previous stage's chosen pool row by span ID.
    :param minimum_advantage: Score margin the best option must reach.
    :return: Chosen pool row by span ID.
    """
    chosen = choose_best_options(options, scores)
    if set(previous) != set(chosen):
        raise ValueError("previous choices do not cover the option pool")
    best_scores: dict[int, float] = {}
    previous_scores: dict[int, float] = {}
    for row, option in enumerate(options):
        span_id = option.span_id
        score = float(scores[row])
        best_scores[span_id] = max(score, best_scores.get(span_id, -np.inf))
        if option.sequence == options[previous[span_id]].sequence:
            previous_scores[span_id] = max(score, previous_scores.get(span_id, -np.inf))
    for span_id, best_score in best_scores.items():
        advantage = best_score - previous_scores[span_id]
        if advantage <= 0.0 or advantage < minimum_advantage:
            chosen[span_id] = previous[span_id]
    return chosen
