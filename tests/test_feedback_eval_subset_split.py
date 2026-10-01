"""Seeded subsets of a pinned split.

The failure this guards is a subset that scores differently from run to run
because it was redrawn, or one `score_cli` cannot use because it is shaped like
a smaller reference file instead of a smaller split.
"""
import json

import pytest

from feedback_eval import subset_split as cli
from feedback_eval.contracts import ReferenceRecord
from feedback_eval.splits import Split, SplitError, load_split, players_by_clip, select


def _reference(clip_id, players):
    return ReferenceRecord(
        clip_id=clip_id,
        player_ids=tuple(players),
        references=("prepare the racket earlier",),
        source="template",
    )


REFERENCES = (
    _reference("t1", ("A", "B")),
    _reference("t2", ("C", "D")),
    _reference("x1", ("P", "Q")),
    _reference("x2", ("P", "Q")),
    _reference("x3", ("P", "R")),
    _reference("x4", ("S", "T")),
)

SPLIT = Split(
    train_players=("A", "B", "C", "D"),
    test_players=("P", "Q", "R", "S", "T"),
    train_clips=("t1", "t2"),
    test_clips=("x1", "x2", "x3", "x4"),
)


def _subset(size, seed=7, side="test"):
    return cli.subset_split(SPLIT, players_by_clip(REFERENCES), size=size, seed=seed, side=side)


def test_keeps_the_train_side_untouched():
    subset = _subset(2)
    assert subset.train_clips == SPLIT.train_clips
    assert subset.train_players == SPLIT.train_players


def test_same_seed_draws_the_same_clips():
    assert _subset(2, seed=3).test_clips == _subset(2, seed=3).test_clips


def test_clips_keep_their_original_order():
    clips = _subset(3).test_clips
    assert list(clips) == [c for c in SPLIT.test_clips if c in clips]


def test_test_players_narrow_to_those_in_the_sampled_clips():
    subset = _subset(4)
    assert subset.test_players == SPLIT.test_players
    for seed in range(20):
        subset = _subset(1, seed=seed)
        present = {p for c in subset.test_clips for p in players_by_clip(REFERENCES)[c]}
        assert set(subset.test_players) == present


def test_full_references_select_exactly_the_subset():
    # The reason the subset is a split file: score_cli selects from the full
    # reference set, and a smaller reference file would be refused.
    subset = _subset(2)
    assert {r.clip_id for r in select(REFERENCES, subset, "test")} == set(subset.test_clips)


def test_refuses_more_clips_than_the_side_has():
    with pytest.raises(SplitError, match="only 4"):
        _subset(5)


def test_refuses_a_non_positive_size():
    with pytest.raises(SplitError, match="positive"):
        _subset(0)


def test_refuses_clips_missing_from_the_references():
    with pytest.raises(SplitError, match="not in the reference set"):
        cli.subset_split(SPLIT, players_by_clip(REFERENCES[:3]), size=1, seed=0)


def _write_inputs(tmp_path):
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "schema": "feedback-eval-split/1",
                "seed": None,
                "train": {"players": list(SPLIT.train_players), "clips": list(SPLIT.train_clips)},
                "test": {"players": list(SPLIT.test_players), "clips": list(SPLIT.test_clips)},
                "discarded_clips": [],
            }
        )
    )
    refs_path = tmp_path / "refs.jsonl"
    refs_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "clip_id": r.clip_id,
                    "player_ids": list(r.player_ids),
                    "references": list(r.references),
                    "source": r.source,
                }
            )
            for r in REFERENCES
        )
        + "\n"
    )
    return split_path, refs_path


def test_cli_writes_a_loadable_split_with_provenance(tmp_path):
    split_path, refs_path = _write_inputs(tmp_path)
    out = tmp_path / "subset.json"
    argv = ["--split", str(split_path), "--references", str(refs_path),
            "--size", "2", "--seed", "11", "--out", str(out)]
    assert cli.main(argv) == 0

    assert len(load_split(out).test_clips) == 2
    assert json.loads(out.read_text())["subset"] == {
        "of": str(split_path), "side": "test", "size": 2, "seed": 11,
    }


def test_cli_refuses_to_overwrite_without_force(tmp_path):
    split_path, refs_path = _write_inputs(tmp_path)
    out = tmp_path / "subset.json"
    argv = ["--split", str(split_path), "--references", str(refs_path),
            "--size", "2", "--seed", "11", "--out", str(out)]
    cli.main(argv)
    with pytest.raises(SplitError, match="already exists"):
        cli.main(argv)
    assert cli.main([*argv, "--force"]) == 0
