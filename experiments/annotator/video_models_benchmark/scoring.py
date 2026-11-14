"""Shared contact and rally scoring for the video-model benchmark."""
from dataclasses import replace

from annotator.training.sequences import (
    LabelledRally,
    judge_rally_boundaries,
    match_contacts,
)
from annotator.types import ContactSequence, CourtHalf
from shared.frame_timing import frames_at_fps

COUNT_NAMES = ("labelled", "predicted", "matched", "correct_player", "rallies", "fully_correct")
TOLERANCES = (10, 5)  # frames at 30 fps, scaled to each video's frame rate


def left_out_windows_for(videos: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """Windows around each test video's left-out rallies, keyed by item ID; none without test videos."""
    if not any(video["role"] == "test" for video in videos):
        return {}
    # Imported here so training and validation never load the report scorer or the test labels.
    from experiments.annotator.monotrack_benchmark.writeup_scripts import metrics

    return {item_id: metrics.left_out_windows(rallies) for item_id, rallies in metrics.test_left_out().items()}


def ignore_left_out_predictions(
    video: dict, sequences: list[ContactSequence], windows: list[tuple[int, int]],
) -> list[ContactSequence]:
    """Drop predictions inside left-out rallies, keeping those near a scored hit and every supplied window."""
    from experiments.annotator.monotrack_benchmark.writeup_scripts import metrics

    scored_frames = [frame for span in video["spans"] for frame in span["label_frames"]]
    margin = frames_at_fps(metrics.IGNORE_MARGIN, video["fps"])
    return [
        replace(sequence, events=tuple(
            event for event in sequence.events
            if not metrics.inside_left_out_rally(event.frame, windows, scored_frames, margin)
        ))
        for sequence in sequences
    ]


def sequences_to_score(
    video: dict, sequences: dict[str, list[ContactSequence]], left_out: dict[str, list[tuple[int, int]]],
) -> dict[str, list[ContactSequence]]:
    """Each hitter rule's sequences as scored: test videos ignore predictions inside left-out rallies,
    as the annotator and MonoTrack benchmark does; validation videos score every prediction."""
    if video["role"] != "test":
        return sequences
    return {rule: ignore_left_out_predictions(video, rule_sequences, left_out[video["item_id"]])
            for rule, rule_sequences in sequences.items()}


def score_video_rules(video: dict, sequences: dict[str, list[ContactSequence]], tolerance_base30: int) -> dict:
    """Score hitter rules sharing the same contacts, without changing their sides.

    Matching uses one reference rule because every rule has identical contact
    frames and rally bounds. Unknown hitters count in the contact denominator
    but receive no player credit and prevent whole-rally credit.
    """
    rallies = []
    for span in video["spans"]:
        sides = tuple(None if side is None else CourtHalf(side) for side in span["label_sides"])
        rallies.append(LabelledRally(span["rally_id"], tuple(span["label_frames"]), sides))
    # Same order as the shared scorer, so one-to-one matching pairs agree.
    labels = sorted(
        (frame, rally.rally_id, index, side)
        for rally in rallies for index, (frame, side) in enumerate(zip(rally.frames, rally.sides, strict=True))
    )
    events = {rule: [event for sequence in items for event in sequence.events] for rule, items in sequences.items()}
    tolerance = frames_at_fps(tolerance_base30, video["fps"])
    reference_events = next(iter(events.values()))
    matches = match_contacts([label[0] for label in labels], [event.frame for event in reference_events], tolerance)
    counts = {}
    for rule, predictions in events.items():
        correct_player = 0
        for label, prediction, _distance in matches:
            side = labels[label][3]
            correct_player += side is not None and predictions[prediction].side == side
        counts[rule] = {"labelled": len(labels), "predicted": len(predictions), "matched": len(matches),
                        "correct_player": correct_player, "rallies": len(rallies), "fully_correct": 0}
    for aligned_sequences in zip(*sequences.values(), strict=True):
        reference = aligned_sequences[0]
        boundary = judge_rally_boundaries(reference, rallies, tolerance=tolerance)
        if boundary.outcome != "boundaries_right":
            continue
        rally = boundary.rally
        assert rally is not None
        pairs = match_contacts(rally.frames, [event.frame for event in reference.events], tolerance)
        if len(pairs) != len(rally.frames) or len(reference.events) != len(rally.frames):
            continue
        for rule, sequence in zip(sequences, aligned_sequences, strict=True):
            if all(rally.sides[label] is not None and sequence.events[prediction].side == rally.sides[label]
                   for label, prediction, _distance in pairs):
                counts[rule]["fully_correct"] += 1
    return counts


def add_rates(counts: dict) -> dict:
    denominator = counts["labelled"] + counts["predicted"]
    return {
        **counts,
        "contact_recall": 100 * counts["matched"] / counts["labelled"],
        "contact_precision": 100 * counts["matched"] / counts["predicted"] if counts["predicted"] else 0.0,
        "contact_f1": 200 * counts["matched"] / denominator,
        "correct_player_f1": 200 * counts["correct_player"] / denominator,
        "fully_correct_percent": 100 * counts["fully_correct"] / counts["rallies"],
    }
