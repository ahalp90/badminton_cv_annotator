"""Shared contact and rally scoring for the video-model benchmark."""
from annotator.training.sequences import (
    LabelledRally,
    judge_rally_boundaries,
    match_contacts,
)
from annotator.types import ContactSequence, CourtHalf
from shared.frame_timing import frames_at_fps

COUNT_NAMES = ("labelled", "predicted", "matched", "correct_player", "rallies", "fully_correct")
TOLERANCES = (10, 5)  # frames at 30 fps, scaled to each video's frame rate


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
