"""Check VideoMAE stride-three rescoring and the sampled training-loss reconstruction."""
import numpy as np
import pytest

from experiments.annotator.video_models_benchmark.evaluate import decode_span
from experiments.annotator.video_models_benchmark.scoring import score_video_rules
from experiments.annotator.video_models_benchmark.writeup_scripts.make_videomae_figures import (
    binned_training_loss,
    training_windows,
)
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


def test_stride_offsets_follow_video_frame_numbers() -> None:
    # The rally starts at frame 7, so offsets must use video frames rather than positions in the rally.
    video = one_rally(7, 60, [20, 40], ["Bot", "Top"])
    peaks = {frame: BOTTOM for frame in range(18, 23)} | {frame: TOP for frame in range(38, 43)}
    probabilities = contact_probabilities(video, peaks)
    dense = decode_video(video, probabilities, 1, 0)
    assert dense["alternating"][0] == decode_span(video["spans"][0], probabilities, 30)
    for offset in range(3):
        frames = [event.frame for event in decode_video(video, probabilities, 3, offset)["raw"][0].events]
        assert len(frames) == 2 and all(frame % 3 == offset for frame in frames)


def test_raw_and_alternating_hitters_share_contacts() -> None:
    video = one_rally(0, 41, [10, 20, 30], ["Bot", "Top", "Bot"])
    probabilities = contact_probabilities(video, {10: BOTTOM, 20: BOTTOM, 30: BOTTOM})
    scores = score_video_rules(video, decode_video(video, probabilities, 1, 0), 5)
    assert scores["raw"]["matched"] == scores["alternating"]["matched"] == 3
    assert (scores["raw"]["correct_player"], scores["raw"]["fully_correct"]) == (2, 0)
    assert (scores["alternating"]["correct_player"], scores["alternating"]["fully_correct"]) == (3, 1)


@pytest.mark.parametrize("rule", ["raw", "alternating"])
def test_unknown_hitter_counts_as_contact_but_blocks_rally_credit(rule: str) -> None:
    video = one_rally(0, 41, [10, 20, 30], ["Bot", None, "Bot"])
    probabilities = contact_probabilities(video, {10: BOTTOM, 20: TOP, 30: BOTTOM})
    scores = score_video_rules(video, decode_video(video, probabilities, 1, 0), 5)[rule]
    assert (scores["labelled"], scores["matched"], scores["correct_player"], scores["fully_correct"]) == (3, 3, 2, 0)


def test_training_windows_drop_replayed_updates() -> None:
    # Session 1 validated at update 100, then stopped mid-interval at 150; session 2 replays from 100.
    progress = [
        {"session_started": 1.0, "interval_start_step": 0, "step": 50, "train_loss": 1.0, "lr": 0.1},
        {"session_started": 1.0, "interval_start_step": 0, "step": 100, "train_loss": 0.8, "lr": 0.2},
        {"session_started": 1.0, "interval_start_step": 100, "step": 150, "train_loss": 0.5, "lr": 0.3},
        {"session_started": 2.0, "interval_start_step": 100, "step": 160, "train_loss": 0.4, "lr": 0.3},
    ]
    windows = training_windows(progress)
    assert [(window.first, window.last) for window in windows] == [(0, 50), (50, 100), (100, 160)]
    np.testing.assert_allclose([window.loss for window in windows], [1.0, 0.6, 0.4])
    centres, losses = binned_training_loss(windows, 100)
    np.testing.assert_allclose(centres, [50, 150])
    np.testing.assert_allclose(losses, [0.8, 0.4])
