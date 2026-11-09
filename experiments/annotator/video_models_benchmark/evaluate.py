"""Decode validation probabilities and reuse the benchmark's event scorer."""
from __future__ import annotations

import numpy as np

from annotator.contacts.player_attribution import alternating_player_sides
from annotator.contacts.score_candidates import (
    NEARBY_CONTACT_DISTANCE_AT_30_FPS,
    remove_nearby_contacts,
)
from annotator.types import ContactEvent, ContactSequence, CourtHalf
from experiments.annotator.contact_records import stream_payload
from shared.frame_timing import frames_at_fps

from .data import ClipDataset
from .scoring import COUNT_NAMES, TOLERANCES, add_rates, score_video_rules

CLASS_SIDES = {1: CourtHalf.BOT, 2: CourtHalf.TOP}
SIDE_CLASSES = {side: label for label, side in CLASS_SIDES.items()}


def fit_alternation(guesses: list[CourtHalf]) -> CourtHalf | None:
    """Keep the benchmark's original majority vote, leaving equal votes unresolved."""
    scores = {
        half: sum(guess == assigned for guess, assigned in zip(
            guesses, alternating_player_sides(half, len(guesses)), strict=True,
        ))
        for half in (CourtHalf.TOP, CourtHalf.BOT)
    }
    if scores[CourtHalf.TOP] == scores[CourtHalf.BOT]:
        return None
    return max(scores, key=scores.__getitem__)


def decode_span(
    span: dict, probabilities: np.ndarray, fps: float, *, break_alternation_ties: bool = False,
) -> ContactSequence:
    """Alternate hitters by majority vote, optionally resolving tied votes by probability."""
    classes = probabilities.argmax(axis=1)
    candidates = np.flatnonzero(classes != 0)
    frames = candidates + span["start_frame"]
    scores = probabilities[candidates, classes[candidates]]
    kept = remove_nearby_contacts(
        frames, np.zeros(len(frames), dtype=int), scores, 0,
        frames_at_fps(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps),
    )
    guesses = [CLASS_SIDES[int(classes[candidates[index]])] for index in kept]
    final_side = fit_alternation(guesses)
    sides = guesses if final_side is None else alternating_player_sides(final_side, len(guesses))
    if break_alternation_ties and final_side is None and guesses:
        alternatives = [alternating_player_sides(half, len(guesses)) for half in (CourtHalf.TOP, CourtHalf.BOT)]
        kept_probabilities = probabilities[candidates[kept]]
        sides = max(alternatives, key=lambda order: (
            sum(float(row[SIDE_CLASSES[side]]) for row, side in zip(kept_probabilities, order, strict=True)),
            order[0] == guesses[0],
        ))
    events = tuple(ContactEvent(int(frames[index]), float(scores[index]), side)
                   for index, side in zip(kept, sides, strict=True))
    return ContactSequence(span["span_id"], span["start_frame"], span["end_frame"], events)


def score_probabilities(
    dataset: ClipDataset, probabilities: np.ndarray, *, break_alternation_ties: bool = False,
) -> dict:
    """Score exhaustive, ordered inference once at both existing tolerances."""
    if len(probabilities) != len(dataset):
        raise ValueError("Predictions and exhaustive validation centres differ")
    streams = {}
    totals = {str(tolerance): dict.fromkeys(COUNT_NAMES, 0) for tolerance in TOLERANCES}
    offset = 0
    for video in dataset.videos:
        sequences = []
        for span in video["spans"]:
            count = span["end_frame"] - span["start_frame"]
            sequences.append(decode_span(span, probabilities[offset:offset + count], video["fps"],
                                         break_alternation_ties=break_alternation_ties))
            offset += count
        stream = stream_payload(sequences, [event for sequence in sequences for event in sequence.events])
        streams[video["item_id"]] = stream
        for tolerance in TOLERANCES:
            scored = score_video_rules(video, {"decoded": sequences}, tolerance)["decoded"]
            for name in COUNT_NAMES:
                totals[str(tolerance)][name] += scored[name]
    for counts in totals.values():
        rates = add_rates(counts)
        for name in ("contact_f1", "correct_player_f1", "fully_correct_percent"):
            counts[name] = rates[name]
    return {"metrics": totals, "streams": streams,
            "hitter_rule": "forced_alternation_probability_ties" if break_alternation_ties else "alternation_raw_ties"}
