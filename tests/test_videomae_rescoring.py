"""Check hitter rules, unknown hitters and left-out rallies when rescoring saved probabilities."""
import lzma
from pathlib import Path

import numpy as np
import pytest

from experiments.annotator.video_models_benchmark.data import write_json
from experiments.annotator.video_models_benchmark.scoring import score_video_rules
from experiments.annotator.video_models_benchmark.writeup_scripts import (
    rescore_videomae,
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


def test_saved_test_run_rescores_with_left_out_rallies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = one_rally(0, 90, [30, 60], ["Bot", "Top"]) | {"role": "test"}
    # The hit at 81 lies in a left-out rally's window, 21 frames from the last scored hit.
    probabilities = contact_probabilities(video, {30: BOTTOM, 60: TOP, 81: BOTTOM})
    run = {"name": "videomae-seed0", "seed": 0, "learning_rate": 1e-4, "best_step": 1000}
    test_run = tmp_path / "test" / "seed0"
    for directory in (tmp_path / run["name"], test_run):
        directory.mkdir(parents=True)
    write_json(tmp_path / "selection.json.gz", {"ranking": [{"runs": [run]}]})
    write_json(tmp_path / run["name"] / "history.json.gz", [{"steps": 1000, "validation": {"loss": 0.2}}])
    write_json(test_run / "status.json.gz", {"phase": "complete"})
    saved_counts = {"labelled": 2, "predicted": 3, "matched": 2, "correct_player": 2, "rallies": 1, "fully_correct": 0}
    write_json(test_run / "events.json.gz", {"metrics": {"10": saved_counts, "5": saved_counts},
                                             "sampling": {"stride": 3, "offset": 0}})
    with lzma.open(test_run / "probabilities.npy.xz", "wb") as stream:
        np.save(stream, probabilities)
    monkeypatch.setattr(rescore_videomae, "left_out_windows_for", lambda videos: {"video": [(75, 95)]})

    rows = rescore_videomae.rescore(rescore_videomae.test_jobs(tmp_path, [video], None)[0])
    assert len(rows) == 4  # both hitter rules at both tolerances, at the evaluated sampling only
    for row in rows:
        assert (row["predicted"], row["matched"], row["fully_correct"]) == (2, 2, 1)
