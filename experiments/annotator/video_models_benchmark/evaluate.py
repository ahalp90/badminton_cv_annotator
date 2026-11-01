"""Decode validation probabilities and reuse the benchmark's event scorer."""
from __future__ import annotations

import numpy as np

from annotator.contacts.model import (
    NEARBY_CONTACT_DISTANCE_AT_30_FPS,
    remove_nearby_contacts,
)
from annotator.outcomes.point_winner import (
    Half,
    alternating_player_sides,
    fit_alternation,
)
from annotator.sequence.contacts import (
    ContactEvent,
    ContactSequence,
    scale_frames,
    stream_payload,
)
from annotator.training.sequences import LabelledRally
from experiments.annotator.annotator_closeout_new_courts.scripts.evaluate_saved import (
    score_video,
)

from .data import ClipDataset

CLASS_SIDES = {1: Half.BOT, 2: Half.TOP}


def decode_span(span: dict, probabilities: np.ndarray, fps: float) -> ContactSequence:
    classes = probabilities.argmax(axis=1)
    candidates = np.flatnonzero(classes != 0)
    frames = candidates + span["start_frame"]
    scores = probabilities[candidates, classes[candidates]]
    kept = remove_nearby_contacts(
        frames, np.zeros(len(frames), dtype=int), scores, 0,
        scale_frames(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps),
    )
    guesses = [CLASS_SIDES[int(classes[candidates[index]])] for index in kept]
    final_side = fit_alternation(guesses)
    sides = guesses if final_side is None else alternating_player_sides(final_side, len(guesses))
    events = tuple(ContactEvent(int(frames[index]), float(scores[index]), side)
                   for index, side in zip(kept, sides, strict=True))
    return ContactSequence(span["span_id"], span["start_frame"], span["end_frame"], events)


def score_probabilities(dataset: ClipDataset, probabilities: np.ndarray) -> dict:
    """Score exhaustive, ordered inference once at both existing tolerances."""
    if len(probabilities) != len(dataset):
        raise ValueError("Predictions and exhaustive validation centres differ")
    streams = {}
    totals = {str(tolerance): {"labelled": 0, "predicted": 0, "matched": 0, "correct_player": 0, "rallies": 0, "fully_correct": 0}
              for tolerance in (10, 5)}
    offset = 0
    for video in dataset.videos:
        sequences, rallies = [], []
        for span in video["spans"]:
            count = span["end_frame"] - span["start_frame"]
            sequences.append(decode_span(span, probabilities[offset:offset + count], video["fps"]))
            rallies.append(LabelledRally(span["rally_id"], tuple(span["label_frames"]),
                                        tuple(Half(side) for side in span["label_sides"])))
            offset += count
        stream = stream_payload(sequences, [event for sequence in sequences for event in sequence.events])
        streams[video["item_id"]] = stream
        for tolerance in (10, 5):
            _proposals, contacts, predictions, rally_rows = score_video(
                video["item_id"], video["role"], stream, tuple(rallies), video["fps"], tolerance,
            )
            counts = totals[str(tolerance)]
            counts["labelled"] += len(contacts)
            counts["predicted"] += len(predictions)
            counts["matched"] += sum(row["matched"] for row in contacts)
            counts["correct_player"] += sum(row["player_correct"] for row in contacts)
            counts["rallies"] += len(rally_rows)
            counts["fully_correct"] += sum(row["fully_correct"] for row in rally_rows)
    for counts in totals.values():
        denominator = counts["labelled"] + counts["predicted"]
        counts["contact_f1"] = 200 * counts["matched"] / denominator
        counts["correct_player_f1"] = 200 * counts["correct_player"] / denominator
        counts["fully_correct_percent"] = 100 * counts["fully_correct"] / counts["rallies"]
    return {"metrics": totals, "streams": streams}
