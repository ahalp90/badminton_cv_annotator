"""Player-disjoint splits.

The failure this guards is a score that looks like generalisation and is not:
the same player on both sides, a split silently covering a subset of the
reference set, or a split quietly rebuilt between two runs that are then
compared.
"""
import json

import pytest

from feedback_eval.contracts import PredictionRecord, ReferenceRecord
from feedback_eval.splits import (
    LeakageError,
    Split,
    SplitError,
    assert_player_disjoint,
    build_split,
    clips_by_player,
    load_split,
    save_split,
    select,
)


def _reference(clip_id, players):
    return ReferenceRecord(
        clip_id=clip_id,
        player_ids=(players,) if isinstance(players, str) else tuple(players),
        references=("prepare the racket earlier",),
        source="commentary",
    )


def _records(*pairs):
    return tuple(_reference(clip_id, players) for clip_id, players in pairs)


def _three_players():
    return _records(
        ("c1", "alice"), ("c2", "alice"), ("c3", "alice"),
        ("c4", "bob"), ("c5", "bob"),
        ("c6", "carol"),
    )


def test_clips_by_player_groups_in_file_order():
    grouped = clips_by_player(_three_players())
    assert grouped == {"alice": ("c1", "c2", "c3"), "bob": ("c4", "c5"), "carol": ("c6",)}


def test_assert_player_disjoint_accepts_a_clean_split():
    train = _records(("c1", "alice"))
    test = _records(("c2", "bob"))
    assert_player_disjoint(train, test)


def test_assert_player_disjoint_names_the_leaked_player():
    train = _records(("c1", "alice"), ("c2", "bob"))
    test = _records(("c3", "bob"))
    with pytest.raises(LeakageError, match="bob"):
        assert_player_disjoint(train, test)


def test_build_split_assigns_whole_players_never_partial_ones():
    """The point of the split: a player's clips never straddle the boundary."""
    records = _three_players()
    split = build_split(records, test_fraction=0.3, seed=7)
    grouped = clips_by_player(records)
    for player in split.test_players:
        assert set(grouped[player]) <= set(split.test_clips)
        assert not set(grouped[player]) & set(split.train_clips)


def test_build_split_is_deterministic_under_a_seed():
    records = _three_players()
    assert build_split(records, seed=7) == build_split(records, seed=7)


def test_build_split_ignores_reference_file_order():
    """Otherwise reordering the reference file silently changes the split."""
    forward = _three_players()
    reversed_records = tuple(reversed(forward))
    assert build_split(forward, seed=7).test_players == build_split(reversed_records, seed=7).test_players


def test_build_split_pins_named_test_players():
    split = build_split(_three_players(), test_players=("bob",))
    assert split.test_players == ("bob",)
    assert set(split.test_clips) == {"c4", "c5"}
    assert "bob" not in split.train_players


def test_build_split_forgets_the_seed_when_players_are_pinned():
    """A recorded seed would imply the split could be regenerated from it."""
    assert build_split(_three_players(), test_players=("bob",), seed=7).seed is None


def test_build_split_rejects_an_unknown_pinned_player():
    with pytest.raises(SplitError, match="not present"):
        build_split(_three_players(), test_players=("dave",))


def test_build_split_rejects_taking_every_player_for_test():
    with pytest.raises(SplitError, match="nothing to train on"):
        build_split(_three_players(), test_players=("alice", "bob", "carol"))


def test_build_split_never_empties_the_train_side_by_fraction():
    """A high fraction must still leave a player to train on."""
    split = build_split(_three_players(), test_fraction=0.99, seed=1)
    assert split.train_players
    assert split.test_players


def test_build_split_requires_a_seed_when_splitting_by_fraction():
    with pytest.raises(SplitError, match="explicit seed"):
        build_split(_three_players(), test_fraction=0.3)


def test_build_split_rejects_a_single_player():
    with pytest.raises(SplitError, match="at least 2 players"):
        build_split(_records(("c1", "alice")), seed=1)


def test_build_split_rejects_an_out_of_range_fraction():
    with pytest.raises(SplitError, match="strictly between 0 and 1"):
        build_split(_three_players(), test_fraction=1.0, seed=1)


def test_split_rejects_a_player_on_both_sides():
    with pytest.raises(LeakageError, match="alice"):
        Split(
            train_players=("alice",),
            test_players=("alice",),
            train_clips=("c1",),
            test_clips=("c2",),
        )


def test_select_takes_only_the_named_side():
    records = _three_players()
    split = build_split(records, test_players=("bob",))
    assert [record.clip_id for record in select(records, split, "test")] == ["c4", "c5"]
    assert [record.clip_id for record in select(records, split, "train")] == ["c1", "c2", "c3", "c6"]


def test_select_works_on_predictions_too():
    """Predictions must be filtered to the same side or align() sees a mismatch."""
    split = build_split(_three_players(), test_players=("bob",))
    predictions = tuple(
        PredictionRecord(clip_id=clip_id, feedback="text")
        for clip_id in ("c1", "c2", "c3", "c4", "c5", "c6")
    )
    assert [record.clip_id for record in select(predictions, split, "test")] == ["c4", "c5"]


def test_select_rejects_records_missing_a_split_clip():
    """A model that answered only some test clips has not been evaluated on the test set."""
    split = build_split(_three_players(), test_players=("bob",))
    partial = tuple(PredictionRecord(clip_id="c4", feedback="text") for _ in range(1))
    with pytest.raises(SplitError, match="absent from the records"):
        select(partial, split, "test")


def test_select_rejects_an_unknown_side():
    split = build_split(_three_players(), test_players=("bob",))
    with pytest.raises(SplitError, match="is not one of"):
        select(_three_players(), split, "validation")


def test_split_round_trips_through_disk(tmp_path):
    split = build_split(_three_players(), test_fraction=0.3, seed=11)
    path = save_split(tmp_path / "split.json", split)
    assert load_split(path) == split


def test_saved_split_records_the_seed_for_reproduction(tmp_path):
    save_split(tmp_path / "split.json", build_split(_three_players(), seed=11))
    payload = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
    assert payload["seed"] == 11
    assert payload["schema"] == "feedback-eval-split/1"


def test_load_split_rejects_an_unknown_schema(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"schema": "something-else"}), encoding="utf-8")
    with pytest.raises(SplitError, match="schema"):
        load_split(path)


def test_load_split_rejects_a_leaky_file(tmp_path):
    """A hand-edited split file gets the same guard as a generated one."""
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "schema": "feedback-eval-split/1",
                "seed": None,
                "train": {"players": ["alice"], "clips": ["c1"]},
                "test": {"players": ["alice"], "clips": ["c2"]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LeakageError, match="alice"):
        load_split(path)


def test_load_split_reports_a_missing_file(tmp_path):
    with pytest.raises(SplitError, match="no such split file"):
        load_split(tmp_path / "absent.json")


# --- two-player rallies: the construction that is actually disjoint ---------


def _doubles_up_records():
    """Rallies between pairs of players, the real singles-match shape."""
    return _records(
        ("c1", ("alice", "bob")),
        ("c2", ("alice", "bob")),
        ("c3", ("carol", "dave")),
        ("c4", ("carol", "dave")),
        ("c5", ("alice", "carol")),  # crosses any alice/carol partition
    )


def test_a_clip_is_kept_only_when_every_player_is_on_one_side():
    split = build_split(_doubles_up_records(), test_players=("carol", "dave"))
    assert set(split.test_clips) == {"c3", "c4"}
    assert set(split.train_clips) == {"c1", "c2"}


def test_cross_side_clips_are_discarded_not_assigned():
    """c5 involves a train player and a test player, so neither side may use it."""
    split = build_split(_doubles_up_records(), test_players=("carol", "dave"))
    assert split.discarded_clips == ("c5",)
    assert "c5" not in split.train_clips
    assert "c5" not in split.test_clips


def test_the_resulting_sides_are_genuinely_player_disjoint():
    """The property the whole module exists for, checked end to end."""
    records = _doubles_up_records()
    split = build_split(records, test_players=("carol", "dave"))
    assert_player_disjoint(select(records, split, "train"), select(records, split, "test"))


def test_a_pair_key_would_have_leaked():
    """Why player_ids is a set: pair AB in train and AC in test both contain A."""
    records = _records(("c1", ("alice", "bob")), ("c2", ("alice", "carol")))
    with pytest.raises(LeakageError, match="alice"):
        assert_player_disjoint(records[:1], records[1:])


def test_discarded_clips_survive_a_round_trip(tmp_path):
    split = build_split(_doubles_up_records(), test_players=("carol", "dave"))
    path = save_split(tmp_path / "split.json", split)
    assert load_split(path).discarded_clips == ("c5",)


def test_build_split_refuses_a_partition_that_empties_the_test_side():
    """Every test-player clip shared with a train player leaves nothing to score."""
    records = _records(("c1", ("alice", "bob")), ("c2", ("alice", "carol")))
    with pytest.raises(SplitError, match="test side is empty"):
        build_split(records, test_players=("bob", "carol"))


def test_clips_by_player_lists_a_clip_under_each_of_its_players():
    grouped = clips_by_player(_records(("c1", ("alice", "bob"))))
    assert grouped == {"alice": ("c1",), "bob": ("c1",)}


def test_a_bare_string_of_players_is_rejected_not_split_into_letters():
    with pytest.raises(TypeError, match="not the string"):
        ReferenceRecord(clip_id="c1", player_ids="alice", references=("x",), source="expert")


def test_a_clip_with_no_players_is_rejected():
    with pytest.raises(ValueError, match="at least one player"):
        ReferenceRecord(clip_id="c1", player_ids=(), references=("x",), source="expert")
