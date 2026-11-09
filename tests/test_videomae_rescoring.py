"""Check that raw and alternating hitter rules share contacts and handle unknown hitters."""
import numpy as np

from experiments.annotator.video_models_benchmark.scoring import score_video_rules
from experiments.annotator.video_models_benchmark.writeup_scripts.rescore_videomae import (
    decode_video,
)

BOTTOM, TOP = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def one_rally(start: int, end: int, frames: list[int], sides: list[str | None]) -> dict:
    span = {"span_id": 0, "rally_id": "example", "start_frame": start, "end_frame": end,
            "label_frames": frames, "label_sides": sides}
    return {"item_id": "video", "fps": 30, "spans": [span]}


def contact_probabilities(video: dict, peaks: dict[int, tuple[float, float, float]]) -> np.ndarray:
    span = video["spans"][0]
    probabilities = np.tile([1.0, 0.0, 0.0], (span["end_frame"] - span["start_frame"], 1))
    for frame, row in peaks.items():
        probabilities[frame - span["start_frame"]] = row
    return probabilities


def test_raw_and_alternating_hitters_share_contacts() -> None:
    video = one_rally(0, 41, [10, 20, 30], ["Bot", "Top", "Bot"])
    probabilities = contact_probabilities(video, {10: BOTTOM, 20: BOTTOM, 30: BOTTOM})
    scores = score_video_rules(video, decode_video(video, probabilities, 1, 0), 5)
    assert scores["raw"]["matched"] == scores["alternating"]["matched"] == 3
    assert (scores["raw"]["correct_player"], scores["raw"]["fully_correct"]) == (2, 0)
    assert (scores["alternating"]["correct_player"], scores["alternating"]["fully_correct"]) == (3, 1)


def test_unknown_hitter_counts_as_contact_but_blocks_rally_credit() -> None:
    video = one_rally(0, 41, [10, 20, 30], ["Bot", None, "Bot"])
    probabilities = contact_probabilities(video, {10: BOTTOM, 20: TOP, 30: BOTTOM})
    scores = score_video_rules(video, decode_video(video, probabilities, 1, 0), 5)["alternating"]
    assert (scores["labelled"], scores["matched"], scores["correct_player"], scores["fully_correct"]) == (3, 3, 2, 0)
