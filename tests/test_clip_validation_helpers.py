"""Regression tests for validation helpers that derive classifier clip bounds."""

from pathlib import Path

import pandas as pd

from src.bst_x.validation_scripts.compute_clip_length_stats import (
    compute_clip_lengths_for_video,
)
from src.bst_x.validation_scripts.hit_frame_lookup import build_hit_frame_lookup


def _write_early_hit_annotations(set_dir: Path) -> None:
    match_name = 'early_hit_match'
    pd.DataFrame(
        {'id': [1], 'video': [match_name], 'set': [1]},
    ).to_csv(set_dir / 'match.csv', index=False)
    match_dir = set_dir / match_name
    match_dir.mkdir()
    pd.DataFrame(
        {
            'rally': [1],
            'ball_round': [1],
            'frame_num': [3],
            'time': ['00:00:01'],
        }
    ).to_csv(match_dir / 'set1.csv', index=False)


def test_hit_frame_lookup_clamps_early_clip_start(tmp_path: Path):
    set_dir = tmp_path / 'set'
    set_dir.mkdir()
    _write_early_hit_annotations(set_dir)
    metadata = tmp_path / 'video_metadata.csv'
    pd.DataFrame({'id': [1], 'fps': [30]}).to_csv(metadata, index=False)

    lookup = build_hit_frame_lookup(set_dir, metadata)

    assert lookup['1_1_1_1'] == 3


def test_clip_length_stats_clamps_early_clip_start(tmp_path: Path, monkeypatch):
    set_dir = tmp_path / 'set'
    set_dir.mkdir()
    _write_early_hit_annotations(set_dir)
    monkeypatch.setattr(
        'src.bst_x.validation_scripts.compute_clip_length_stats.estimate_fps',
        lambda *_args: 30,
    )

    lengths, fps = compute_clip_lengths_for_video(
        set_dir,
        vid=1,
        video_name='early_hit_match',
        n_sets=1,
        removed_shots=set(),
    )

    assert fps == 30
    assert lengths.tolist() == [18]
