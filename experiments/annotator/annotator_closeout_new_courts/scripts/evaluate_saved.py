"""Rescore the selected new-court annotator's saved streams into row-level tables.

Reuses the project's one-to-one contact matcher and whole-rally rule, so the
±10-frame totals must reproduce the published model-selection result. The
±5-frame view repeats the same rule with the narrower allowance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from annotator.sequence.contacts import ContactSequence, scale_frames
from annotator.sequence.sides import alternate_sides
from annotator.training.sequences import (
    LabelledRally,
    match_contacts,
    overlapping_rallies,
)
from annotator.training.workflow import load_labels, whole_rally_correctness
from dataset_builder.vision import load_json_gz
from experiments.annotator.old_court_regression.retrain import stream_from_payload

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "raw/run/output"
REPOSITORY = ROOT.parents[2]
TOLERANCES_BASE30 = (10, 5)
# The historical review cutoff; no new cutoff was chosen for the retrained confidence model.
HISTORICAL_THRESHOLD = 0.7570784853533734
OUTCOME_NAMES = {1: "correct", 0: "wrong", -1: "unknown"}
PUBLISHED_TEST_TOTALS = {
    "labelled_rallies": 3327, "labelled_contacts": 37184, "predicted_contacts": 40246,
    "matched_contacts": 34200, "correct_side_contacts": 33156, "fully_correct_rallies": 1744,
    "sections_correct": 1744, "sections_wrong": 1273, "sections_unknown": 1004,
    "reached_not_correct": 1355, "wholly_missed": 228,
    "selected": 725, "selected_correct": 591, "selected_wrong": 117, "selected_unknown": 17,
}


def judge_section(sequence: ContactSequence, rallies: tuple[LabelledRally, ...], tolerance: int) -> dict[str, object]:
    """Repeat ``whole_rally_correctness`` at any allowance and keep each reason it fails."""
    overlaps = overlapping_rallies(sequence, rallies)
    record: dict[str, object] = {
        "overlapping_rallies": len(overlaps), "rally_id": None, "matched": None, "missing": None,
        "extra": None, "wrong_player": None, "boundary_error": False, "exact_sequence": False,
    }
    if not overlaps:
        return {**record, "outcome": -1}
    if len(overlaps) > 1:
        return {**record, "outcome": 0}
    rally = overlaps[0]
    contained = all(sequence.start_frame <= frame < sequence.end_frame for frame in rally.frames)
    matches = match_contacts(rally.frames, [event.frame for event in sequence.events], tolerance)
    revised = alternate_sides((sequence,), sequence.events)[0][0]
    wrong_player = sum(rally.sides[label] is not None and revised.events[prediction].side != rally.sides[label]
                       for label, prediction, _offset in matches)
    missing = len(rally.frames) - len(matches)
    extra = len(sequence.events) - len(matches)
    record.update(rally_id=rally.identity, matched=len(matches), missing=missing, extra=extra,
                  wrong_player=wrong_player, boundary_error=not contained,
                  exact_sequence=contained and missing == 0 and extra == 0,
                  first_label_frame=rally.frames[0], last_label_frame=rally.frames[-1])
    if not contained or missing or wrong_player or extra:
        return {**record, "outcome": 0}
    if any(side is None for side in rally.sides):
        return {**record, "outcome": -1}
    return {**record, "outcome": 1}


def score_video(video: str, split: str, stream: dict, rallies: tuple[LabelledRally, ...], fps: float,
                tolerance_base30: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    sequences, events = stream_from_payload(stream)
    confidence = stream.get("confidence")
    tolerance = scale_frames(tolerance_base30, fps)
    common = {"split": split, "video": video, "fps": fps, "tolerance_base30": tolerance_base30}

    proposals = []
    for sequence in sequences:
        judgement = judge_section(sequence, rallies, tolerance)
        if tolerance_base30 == 10:
            assert judgement["outcome"] == whole_rally_correctness(sequence, rallies, fps), (video, sequence.span_id)
        score = None if confidence is None else confidence[str(sequence.span_id)]
        proposals.append({
            **common, "span_id": sequence.span_id, "start_frame": sequence.start_frame,
            "end_frame": sequence.end_frame, "events": len(sequence.events), "confidence": score,
            "selected": score is not None and score >= HISTORICAL_THRESHOLD,
            **judgement, "outcome": OUTCOME_NAMES[judgement["outcome"]],
        })

    # Full-video matching, ordered as in the refit comparison so correct-side counts agree.
    labels = sorted(
        (frame, rally.identity, index, side, len(rally.frames))
        for rally in rallies for index, (frame, side) in enumerate(zip(rally.frames, rally.sides, strict=True))
    )
    pairs = match_contacts([row[0] for row in labels], [event.frame for event in events], tolerance)
    matched_label = {label: (prediction, offset) for label, prediction, offset in pairs}
    matched_prediction = {prediction: label for label, prediction, _offset in pairs}
    contacts = []
    for label_position, (frame, rally_id, index, side, rally_length) in enumerate(labels):
        pair = matched_label.get(label_position)
        event = events[pair[0]] if pair is not None else None
        predicted_side = None if event is None or event.side is None else event.side.value
        position = "serve" if index == 0 else "last" if index == rally_length - 1 else "middle"
        contacts.append({
            **common, "rally_id": rally_id, "label_index": index, "source_frame": frame,
            "labelled_contacts": rally_length, "position": position,
            "target_side": None if side is None else side.value, "matched": pair is not None,
            "player_correct": event is not None and side is not None and event.side == side,
            "prediction_frame": None if event is None else event.frame, "predicted_side": predicted_side,
            "offset_base30": None if pair is None else pair[1] * 30 / fps,
        })

    section_by_frame = {event.frame: sequence.span_id for sequence in sequences for event in sequence.events}
    predictions = []
    for position, event in enumerate(events):
        label_position = matched_prediction.get(position)
        predictions.append({
            **common, "source_frame": event.frame, "probability": event.probability,
            "predicted_side": None if event.side is None else event.side.value,
            "span_id": section_by_frame.get(event.frame), "matched": label_position is not None,
            "rally_id": None if label_position is None else labels[label_position][1],
            "label_index": None if label_position is None else labels[label_position][2],
        })

    reached = {contact["rally_id"] for contact in contacts if contact["matched"]}
    rally_rows = []
    for rally in rallies:
        own = [row for row in proposals if row["rally_id"] == rally.identity]
        touching = [sequence for sequence in sequences
                    if any(sequence.start_frame <= frame < sequence.end_frame for frame in rally.frames)]
        rally_rows.append({
            **common, "rally_id": rally.identity, "first_frame": rally.frames[0], "last_frame": rally.frames[-1],
            "labelled_contacts": len(rally.frames),
            "fully_correct": any(row["outcome"] == "correct" for row in own),
            "selected_correct": any(row["outcome"] == "correct" and row["selected"] for row in own),
            "exact_sequence": any(row["exact_sequence"] for row in own),
            "overlapping_proposals": len(touching),
            "contained": any(all(sequence.start_frame <= frame < sequence.end_frame for frame in rally.frames)
                             for sequence in touching),
            "timing_reached": rally.identity in reached,
        })
    return proposals, contacts, predictions, rally_rows


def test_inputs() -> list[tuple[str, str, dict, tuple[LabelledRally, ...], float]]:
    root = RUN / "base/eval/test"
    videos = sorted((path.name for path in root.iterdir()), key=int)
    assert len(videos) == 46 and "15" not in videos
    inputs = []
    for video in videos:
        scores = load_json_gz(root / video / "scores.json.gz")
        rallies = load_labels(RUN / "shared/labels/test" / f"{video}.csv", scores["frame_count"])
        inputs.append((video, "test", load_json_gz(root / video / "stream.json.gz"), rallies, scores["fps"]))
    return inputs


def development_inputs() -> list[tuple[str, str, dict, tuple[LabelledRally, ...], float]]:
    """V uses the 32-video contact tree; A-D use predictions made with their own group held out."""
    split = json.loads((REPOSITORY / "scratch/contact_det_full_ds_fit/records/shuttleset_development_split.json").read_text())
    fps_by_video = {record["fixture"]: float(record["fps"]) for record in split["videos"]}
    inputs = []
    for path in sorted((RUN / "base/eval/validation").iterdir()):
        scores = load_json_gz(path / "scores.json.gz")
        assert scores["fps"] == fps_by_video[path.name]
        rallies = load_labels(RUN / "shared/labels/development" / f"{path.name}.csv", scores["frame_count"])
        inputs.append((path.name, "validation", load_json_gz(path / "stream.json.gz"), rallies, scores["fps"]))
    for path in sorted((RUN / "base/dev_heldout").glob("*.json.gz")):
        video = path.name.removesuffix(".json.gz")
        # The labels' own bound check needs a frame count; the held-out stream does not record one.
        rallies = load_labels(RUN / "shared/labels/development" / f"{video}.csv", 10**8)
        inputs.append((video, "development_heldout", load_json_gz(path), rallies, fps_by_video[video]))
    return inputs


def check_test_totals(proposals: pd.DataFrame, contacts: pd.DataFrame, predictions: pd.DataFrame,
                      rallies: pd.DataFrame) -> dict[str, int]:
    proposals, contacts = proposals[proposals.tolerance_base30 == 10], contacts[contacts.tolerance_base30 == 10]
    predictions, rallies = predictions[predictions.tolerance_base30 == 10], rallies[rallies.tolerance_base30 == 10]
    selected = proposals[proposals.selected]
    totals = {
        "labelled_rallies": len(rallies), "labelled_contacts": len(contacts), "predicted_contacts": len(predictions),
        "matched_contacts": int(contacts.matched.sum()), "correct_side_contacts": int(contacts.player_correct.sum()),
        "fully_correct_rallies": int(rallies.fully_correct.sum()),
        "sections_correct": int(sum(proposals.outcome == "correct")),
        "sections_wrong": int(sum(proposals.outcome == "wrong")),
        "sections_unknown": int(sum(proposals.outcome == "unknown")),
        "reached_not_correct": int(sum(rallies.timing_reached & ~rallies.fully_correct)),
        "wholly_missed": int(sum(~rallies.timing_reached)),
        "selected": len(selected), "selected_correct": int(sum(selected.outcome == "correct")),
        "selected_wrong": int(sum(selected.outcome == "wrong")),
        "selected_unknown": int(sum(selected.outcome == "unknown")),
    }
    mismatched = {name: (value, PUBLISHED_TEST_TOTALS[name]) for name, value in totals.items()
                  if value != PUBLISHED_TEST_TOTALS[name]}
    assert not mismatched, mismatched
    return totals


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict]] = {"proposals": [], "contacts": [], "predictions": [], "rallies": []}
    for video, split, stream, rallies, fps in test_inputs() + development_inputs():
        for tolerance in TOLERANCES_BASE30:
            for name, rows in zip(tables, score_video(video, split, stream, rallies, fps, tolerance), strict=True):
                tables[name].extend(rows)
    frames = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    test_frames = {name: frame[frame.split == "test"] for name, frame in frames.items()}
    totals = check_test_totals(**test_frames)
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv.gz", index=False)
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    run(parser.parse_args().output)
