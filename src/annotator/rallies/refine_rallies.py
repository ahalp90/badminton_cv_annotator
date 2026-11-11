"""Choose rally contacts, finish their bounds and assign alternating players.

Review scoring uses the selected proposal's hitter guesses and the finished
rally's bounds. This preserves player uncertainty when ranking rallies.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from annotator.contacts.player_attribution import fit_rally_sides
from annotator.contacts.score_candidates import positive_probabilities
from annotator.rallies.build_contact_proposals import (
    CONTACT_SEQUENCE_FEATURE_NAMES,
    FEATURE_NAMES,
    HITTER_ALTERNATION_FEATURE_NAMES,
    RallyOptions,
    contact_sequence_features,
    events_between,
)
from annotator.types import ContactEvent, ContactSequence, FinishedRally
from shared.frame_timing import frames_at_fps

BOUNDARY_PADDING_AT_30_FPS = 10

REVIEW_OPTION_NAMES = (
    'added_count', 'deleted_count', *(f'after__{name}' for name in HITTER_ALTERNATION_FEATURE_NAMES),
    'unused_candidate_count', 'usable_candidate_count', 'positive_gap_count',
    'gaps_without_candidates', 'longest_gap_seconds', 'last_gap_has_candidate',
    'gap_probability_max', 'strongest_unused_left_gap_seconds', 'strongest_unused_right_gap_seconds',
)
REVIEW_OPTION_COLUMNS = tuple(FEATURE_NAMES.index(name) for name in REVIEW_OPTION_NAMES)
REVIEW_FEATURE_NAMES = (
    *REVIEW_OPTION_NAMES,
    'option_score', 'top_margin',
    *(f'final__{name}' for name in CONTACT_SEQUENCE_FEATURE_NAMES),
)


@dataclass(frozen=True)
class RallySelector:
    """Fitted rally selector, with the contact-edit limits used during training."""

    selector: Any
    max_edits: int = 3
    deletion_candidates: int = 6

    def __post_init__(self) -> None:
        if self.max_edits < 0 or self.deletion_candidates < 0:
            raise ValueError('edit and deletion budgets must be non-negative')


@dataclass(frozen=True)
class RallyPrediction:
    """Selected rallies alongside the alternatives and scores used to choose them.

    ``chosen`` maps initial rally IDs to rows in ``pool`` and ``scores``.
    ``rallies`` has final bounds and fitted player sides; ``pool`` keeps the
    earlier bounds and individual player guesses for review features.
    ``events`` is the finished contact list, including hits outside rallies.
    """

    rallies: tuple[FinishedRally, ...]
    events: tuple[ContactEvent, ...]
    pool: RallyOptions
    scores: np.ndarray
    chosen: dict[int, int]

    @property
    def sequences(self) -> tuple[ContactSequence, ...]:
        """Finished contact sequences in rally order."""
        return tuple(rally.sequence for rally in self.rallies)


@dataclass(frozen=True)
class RallyReviewFeatures:
    """Feature rows paired with the IDs of the predicted rallies being reviewed."""

    span_ids: tuple[int, ...]
    matrix: np.ndarray


@dataclass(frozen=True)
class RallyReview:
    """Correctness score for each rally in features.span_ids; review low scores first."""

    features: RallyReviewFeatures
    scores: np.ndarray


def predict_rally_options(models: RallySelector, pool: RallyOptions) -> RallyPrediction:
    """Score alternative contact lists, choose one per rally, and finish bounds and hitters."""
    scores = positive_probabilities(models.selector, pool.matrix)
    chosen = choose_rows(pool, scores)
    rallies, events = finish_rallies(pool, chosen)
    return RallyPrediction(rallies, events, pool, scores, chosen)


def choose_rows(pool: RallyOptions, scores: np.ndarray) -> dict[int, int]:
    """Choose each rally's highest-scoring option; first in pool order wins ties."""
    if scores.shape != (len(pool.sequences),) or not np.isfinite(scores).all():
        raise ValueError('scores must be finite and align with every pool option')
    chosen = {}
    for span_id, rows in pool.span_rows.items():
        chosen[span_id] = int(rows[np.argmax(scores[rows])])
    return chosen


def finish_rallies(
    pool: RallyOptions, rows: dict[int, int],
) -> tuple[tuple[FinishedRally, ...], tuple[ContactEvent, ...]]:
    """Finish the chosen contact lists, rally bounds and alternating player sides.

    Replace contacts inside each chosen range with that proposal's contacts.
    Preserve selected contacts outside all chosen ranges. Extend discovered
    bounds where possible, then assign player sides in the full contact list.
    """
    context = pool.context
    sequences = tuple(pool.sequences[rows[initial.span_id]] for initial in context.initial)
    events = [
        event for event in context.events
        if not any(sequence.start_frame <= event.frame < sequence.end_frame for sequence in sequences)
    ]
    events.extend(event for sequence in sequences for event in sequence.events)
    events.sort(key=lambda event: event.frame)
    if not context.bounds_supplied:
        sequences = extend_boundaries(
            sequences, events, context.fps, frame_count=context.frame_count, piece_bounds=context.piece_bounds,
        )
    rallies = fit_rally_sides(sequences)
    fitted = {event.frame: event for rally in rallies for event in rally.sequence.events}
    return rallies, tuple(fitted.get(event.frame, event) for event in events)


def extend_boundaries(
    sequences: Sequence[ContactSequence], events: Sequence[ContactEvent], fps: float,
    *, frame_count: int, piece_bounds: dict[int, tuple[int, int]] | None = None,
) -> tuple[ContactSequence, ...]:
    """Add space before the first hit and after the last, keeping the same contacts.

    Extend outwards only, stopping at neighbouring rallies and video end.
    If both neighbours extend into the same gap, divide it at the midpoint.
    Keep the old bounds if an extension would change which contacts belong
    to the rally.

    :param sequences: Non-overlapping rallies in time order.
    :param events: Full contact list in frame order, including hits outside rallies.
    :param fps: Video frame rate.
    :param frame_count: Video length; bounds exclude this endpoint.
    :return: Same rallies and contacts, with permitted extensions.
    """
    piece_bounds = piece_bounds if piece_bounds is not None else {}
    padding = frames_at_fps(BOUNDARY_PADDING_AT_30_FPS, fps)
    bounds = [proposed_bounds(sequence, padding) for sequence in sequences]
    for position, sequence in enumerate(sequences):
        earliest, latest = piece_bounds.get(sequence.span_id, (0, frame_count))
        bounds[position][0] = max(bounds[position][0], earliest)
        bounds[position][1] = min(bounds[position][1], latest)
        if position:
            bounds[position][0] = max(bounds[position][0], sequences[position - 1].end_frame)
        if position + 1 < len(sequences):
            bounds[position][1] = min(bounds[position][1], sequences[position + 1].start_frame)
    for left_position in range(len(sequences) - 1):
        right_position = left_position + 1
        if bounds[left_position][1] <= bounds[right_position][0]:
            continue
        midpoint = (sequences[left_position].end_frame + sequences[right_position].start_frame) // 2
        bounds[left_position][1] = min(bounds[left_position][1], midpoint)
        bounds[right_position][0] = max(bounds[right_position][0], midpoint)

    extended = []
    for sequence, (start_frame, end_frame) in zip(sequences, bounds, strict=True):
        if events_between(events, start_frame, end_frame) == sequence.events:
            extended.append(replace(sequence, start_frame=start_frame, end_frame=end_frame))
        else:
            extended.append(sequence)
    return tuple(extended)


def proposed_bounds(sequence: ContactSequence, padding: int) -> list[int]:
    """Extend bounds to pad the first/last hit; keep existing bounds for an empty rally."""
    if not sequence.events:
        return [sequence.start_frame, sequence.end_frame]
    first_frame = min(event.frame for event in sequence.events)
    last_frame = max(event.frame for event in sequence.events)
    start = max(0, min(sequence.start_frame, first_frame - padding))
    end = max(sequence.end_frame, last_frame + padding + 1)
    return [start, end]


# Rank finished rallies for review

def score_rally_review(prediction: RallyPrediction, model: Any) -> RallyReview:
    """Estimate each finished rally's probability of being correct."""
    features = build_review_features(prediction)
    scores = positive_probabilities(model, features.matrix)
    return RallyReview(features, scores)


def build_review_features(prediction: RallyPrediction) -> RallyReviewFeatures:
    """Measure each selected proposal and its finished rally for review scoring.

    Keep the original hitter guesses so the model can see uncertainty hidden
    by the fitted alternating order. Use final bounds for timing measurements.
    """
    pool = prediction.pool
    span_ids = tuple(prediction.chosen)
    final_by_span = {sequence.span_id: sequence for sequence in prediction.sequences}
    rows = []
    for span_id, row in prediction.chosen.items():
        proposed_sequence = pool.sequences[row]
        final_sequence = final_by_span[span_id]
        proposal_with_final_bounds = replace(
            proposed_sequence, start_frame=final_sequence.start_frame, end_frame=final_sequence.end_frame,
        )
        values = prediction.scores[pool.span_rows[span_id]]
        best = np.partition(values, -2)[-2:] if len(values) > 1 else values
        features = dict(zip(REVIEW_OPTION_NAMES, pool.matrix[row, REVIEW_OPTION_COLUMNS], strict=True))
        features.update({
            'option_score': float(prediction.scores[row]),
            'top_margin': float(best.max() - best.min()),
        })
        sequence_features = contact_sequence_features(proposal_with_final_bounds, pool.context.fps)
        features.update((f'final__{name}', value) for name, value in sequence_features.items())
        rows.append([features[name] for name in REVIEW_FEATURE_NAMES])
    matrix = np.asarray(rows, dtype=np.float64).reshape(len(span_ids), len(REVIEW_FEATURE_NAMES))
    return RallyReviewFeatures(span_ids, matrix)
