"""Tests for classifier dataset helpers."""

from pathlib import Path

import pandas as pd

from classifier_shared.dataset import (
    compute_clip_bounds,
    compute_temporal_bounds,
    parse_flaw_records,
)


def test_parse_flaw_records_uses_explicit_path(tmp_path: Path):
    flaw_csv = tmp_path / "flaws.csv"
    flaw_csv.write_text(
        "match,set,rally,ball_round,stroke_type,measure\n"
        "1,1,1,1,whole,removed\n"
        "2,3,4,5,smash,removed\n"
        "3,1,1,1,drop,kept\n"
    )

    excluded, removed = parse_flaw_records(flaw_csv)

    assert excluded == {1}
    assert removed == {(2, 3, 4, 5)}


def test_compute_clip_bounds_clamps_negative_start():
    row = pd.Series({"frame_num": 3, "start_f": -1, "end_f": 20})

    start, end = compute_clip_bounds(
        row,
        "between_2_hits_with_max_limits",
        fps=30.0,
    )

    assert start == 0
    assert end == 27


def test_compute_temporal_bounds_uses_adjacent_hits(tmp_path: Path):
    pd.DataFrame(
        {
            "rally": [1, 1, 1],
            "ball_round": [1, 2, 3],
            "frame_num": [10, 20, 30],
        }
    ).to_csv(tmp_path / "set1.csv", index=False)
    shots = pd.DataFrame(
        {
            "set": [1, 1, 1],
            "rally": [1, 1, 1],
            "ball_round": [1, 2, 3],
            "frame_num": [10, 20, 30],
            "roundscore_A": [0, 0, 0],
            "roundscore_B": [0, 0, 0],
            "player": ["A", "B", "A"],
            "type": ["clear", "drop", "smash"],
        }
    )

    bounded = compute_temporal_bounds(tmp_path, shots)

    assert bounded["start_f"].tolist() == [-1.0, 10.0, 20.0]
    assert bounded["end_f"].tolist() == [20.0, 30.0, -1.0]
