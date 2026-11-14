"""Decode original-frame probabilities with the shared event benchmark rules."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from annotator.training.sequences import judge_rally_boundaries, match_contacts
from annotator.types import ContactSequence
from experiments.annotator.contact_records import stream_payload
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    labels_for_item,
)
from experiments.annotator.video_models_benchmark.evaluate import decode_span
from experiments.annotator.video_models_benchmark.scoring import (
    COUNT_NAMES as CONTACT_COUNT_NAMES,
)
from experiments.annotator.video_models_benchmark.scoring import (
    add_rates,
    left_out_windows_for,
    score_video_rules,
    sequences_to_score,
)
from shared.frame_timing import frames_at_fps

COUNT_NAMES = (*CONTACT_COUNT_NAMES, "predicted_rallies", "rally_fp", "rally_fn")
TOLERANCES = (1, 2, 5, 10)


def fully_correct_rallies(video: dict, sequences: list[ContactSequence], tolerance_base30: int) -> dict:
    """Match non-empty predictions to GT rallies at most once on either side.

    The boundary judge gives each prediction at most one eligible GT rally.
    Taking one prediction per reachable GT therefore gives maximum cardinality;
    later duplicate predictions remain false positives.
    """
    rallies = labels_for_item(video)
    tolerance = frames_at_fps(tolerance_base30, video["fps"])
    matched = set()
    nonempty = 0
    for sequence in sequences:
        if not sequence.events:
            continue
        nonempty += 1
        boundary = judge_rally_boundaries(sequence, rallies, tolerance=tolerance)
        if boundary.outcome != "boundaries_right":
            continue
        rally = boundary.rally
        assert rally is not None
        pairs = match_contacts(rally.frames, [event.frame for event in sequence.events], tolerance)
        if len(pairs) != len(rally.frames) or len(sequence.events) != len(rally.frames):
            continue
        if all(rally.sides[label] is not None and sequence.events[prediction].side == rally.sides[label]
               for label, prediction, _distance in pairs):
            matched.add(rally.rally_id)
    return {"fully_correct": len(matched), "predicted_rallies": nonempty,
            "rally_fp": nonempty - len(matched), "rally_fn": len(rallies) - len(matched)}


def rally_rates(counts: dict) -> dict:
    rates = add_rates(counts)
    denominator = counts["rallies"] + counts["predicted_rallies"]
    rates.update(fully_correct_precision=100 * counts["fully_correct"] / counts["predicted_rallies"]
                 if counts["predicted_rallies"] else 0.0,
                 fully_correct_recall=rates["fully_correct_percent"],
                 fully_correct_f1=200 * counts["fully_correct"] / denominator if denominator else 0.0)
    return rates


def discovery_intervals(probabilities: np.ndarray) -> list[tuple[int, int]]:
    rally = 1 - probabilities[:, 3] >= 0.5
    edges = np.diff(np.concatenate(([False], rally, [False])).astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True))


def score_predictions(videos: list[dict], probabilities: list[np.ndarray], mode: str) -> dict:
    left_out = left_out_windows_for(videos)
    totals = {str(tolerance): dict.fromkeys(COUNT_NAMES, 0) for tolerance in TOLERANCES}
    streams = {}
    for video, rows in zip(videos, probabilities, strict=True):
        if mode == "supplied":
            spans = video["spans"]
        else:
            spans = [{"span_id": f"discovery-{index}", "start_frame": start, "end_frame": end}
                     for index, (start, end) in enumerate(discovery_intervals(rows))]
        sequences = []
        for span in spans:
            contact_rows = rows[span["start_frame"]:span["end_frame"], :3].copy()
            # Preserve the four-way argmax: NOT_RALLY never becomes a contact when
            # the decoder sees its three-class input. Keep original probabilities.
            if mode == "discovery":
                not_rally = rows[span["start_frame"]:span["end_frame"]].argmax(axis=1) == 3
                contact_rows[not_rally, 0] = 1
            sequence = decode_span(span, contact_rows, video["fps"], break_alternation_ties=True)
            if sequence.events:
                sequences.append(sequence)
        streams[video["item_id"]] = stream_payload(sequences, [event for sequence in sequences for event in sequence.events])
        sequences = sequences_to_score(video, {"decoded": sequences}, left_out)["decoded"]
        sequences = [sequence for sequence in sequences if sequence.events]
        if mode == "discovery":
            sequences = [replace(sequence, start_frame=sequence.events[0].frame,
                                 end_frame=sequence.events[-1].frame + 1) for sequence in sequences]
        for tolerance in TOLERANCES:
            counts = score_video_rules(video, {"decoded": sequences}, tolerance)["decoded"]
            counts.update(fully_correct_rallies(video, sequences, tolerance))
            for name in COUNT_NAMES:
                totals[str(tolerance)][name] += counts[name]
    return {"metrics": {key: rally_rates(value) for key, value in totals.items()}, "streams": streams,
            "hitter_rule": "forced_alternation_probability_ties", "mode": mode}
