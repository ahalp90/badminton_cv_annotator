"""Deriving per-clip faults from ShuttleSet's terminal-shot annotations.

The exclusions are the thing under test as much as the join is. Every rally this
module drops is a rally the reference set silently does not cover, and a
reference set missing a third of its rallies still produces a perfectly
plausible-looking mean.
"""
import csv
from pathlib import Path

import pytest

from feedback_eval.shuttleset_faults import (
    OPPONENT_WINNER,
    STROKE_FAMILIES,
    ShuttleSetFaultError,
    derive_faults,
    load_match_players,
    write_clip_faults,
    write_clip_players,
)
from feedback_eval.templates import load_templates

REPO = Path(__file__).resolve().parent.parent
SHUTTLESET_LIBRARY = (
    REPO / "data" / "feedback_eval" / "templates" / "badminton_singles_shuttleset_v1.jsonl"
)

COLUMNS = [
    "rally",
    "ball_round",
    "player",
    "type",
    "lose_reason",
    "getpoint_player",
]


def _row(rally="1", player="A", type="殺球", lose_reason="", getpoint_player=""):
    return {
        "rally": rally,
        "ball_round": "1.0",
        "player": player,
        "type": type,
        "lose_reason": lose_reason,
        "getpoint_player": getpoint_player,
    }


def _corpus(tmp_path, rows, video="Alice_Bob_Open_2026_Finals", set_name="set1.csv"):
    """Write a minimal ShuttleSet tree and its match.csv."""
    root = tmp_path / "set"
    (root / video).mkdir(parents=True, exist_ok=True)
    with (root / video / set_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    match = root / "match.csv"
    if not match.is_file():
        with match.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["video", "winner", "loser"])
            writer.writerow([video, "Alice", "Bob"])
    return root


def _derive(tmp_path, rows, **kwargs):
    root = _corpus(tmp_path, rows, **kwargs)
    return derive_faults(root, load_match_players(root / "match.csv"))


# --- the join ---------------------------------------------------------------


def test_an_out_error_becomes_a_hit_out_fault(tmp_path):
    faults, report = _derive(
        tmp_path, [_row(type="殺球", lose_reason="出界", getpoint_player="B")]
    )
    assert report.kept == 1
    assert faults[0].template_id == "smash_hit_out"
    assert faults[0].erring_player == "A"


@pytest.mark.parametrize("lose_reason", ["掛網", "未過網"])
def test_both_net_reasons_merge_into_one_fault(tmp_path, lose_reason):
    """A coach says the same thing whether the shuttle caught the tape or fell short."""
    faults, _ = _derive(
        tmp_path, [_row(type="放小球", lose_reason=lose_reason, getpoint_player="B")]
    )
    assert faults[0].template_id == "net_play_into_net"


def test_the_clip_id_carries_the_set_because_rally_numbers_restart(tmp_path):
    faults, _ = _derive(
        tmp_path,
        [_row(rally="7", lose_reason="出界", getpoint_player="B")],
        set_name="set3.csv",
    )
    assert faults[0].clip_id == "Alice_Bob_Open_2026_Finals_s3_r7"


def test_both_match_players_are_carried_not_just_the_one_who_erred(tmp_path):
    """splits.build_split needs every player in the rally to keep the split disjoint."""
    faults, _ = _derive(tmp_path, [_row(lose_reason="出界", getpoint_player="B")])
    assert set(faults[0].player_ids) == {"Alice", "Bob"}


def test_non_terminal_rows_are_not_faults(tmp_path):
    faults, report = _derive(
        tmp_path,
        [
            _row(rally="1", lose_reason=""),
            _row(rally="1", lose_reason="出界", getpoint_player="B"),
        ],
    )
    assert len(faults) == 1
    assert report.total == 1


# --- the exclusions ---------------------------------------------------------


def test_a_rally_ended_by_a_winner_is_excluded(tmp_path):
    """The loser made no recorded error, so any fault would have to be invented."""
    faults, report = _derive(
        tmp_path, [_row(lose_reason=OPPONENT_WINNER, getpoint_player="A")]
    )
    assert faults == ()
    assert report.opponent_winner == 1


@pytest.mark.parametrize("lose_reason", ["落點判斷失誤", "犯規"])
def test_an_ambiguous_lose_reason_is_excluded(tmp_path, lose_reason):
    _, report = _derive(tmp_path, [_row(lose_reason=lose_reason, getpoint_player="B")])
    assert report.ambiguous_reason == 1
    assert report.kept == 0


def test_an_error_credited_to_the_player_who_made_it_is_excluded(tmp_path):
    """You cannot lose the point you erred on and also win it."""
    _, report = _derive(tmp_path, [_row(lose_reason="出界", getpoint_player="A")])
    assert report.attribution_inconsistent == 1
    assert report.kept == 0


def test_an_unmapped_stroke_type_is_counted_not_guessed(tmp_path):
    _, report = _derive(
        tmp_path, [_row(type="未知球種", lose_reason="出界", getpoint_player="B")]
    )
    assert report.unmapped_stroke == {"未知球種": 1}
    assert report.kept == 0


def test_every_excluded_rally_is_still_counted_in_the_total(tmp_path):
    """The drop counts are the audit trail; a silent drop is the failure mode."""
    _, report = _derive(
        tmp_path,
        [
            _row(rally="1", lose_reason="出界", getpoint_player="B"),
            _row(rally="2", lose_reason=OPPONENT_WINNER, getpoint_player="A"),
            _row(rally="3", lose_reason="犯規", getpoint_player="B"),
            _row(rally="4", lose_reason="出界", getpoint_player="A"),
            _row(rally="5", type="未知球種", lose_reason="出界", getpoint_player="B"),
        ],
    )
    assert report.total == 5
    assert report.kept == 1


def test_a_match_with_no_match_csv_row_is_counted(tmp_path):
    root = _corpus(tmp_path, [_row(lose_reason="出界", getpoint_player="B")])
    _, report = derive_faults(root, {"Some_Other_Match": ("X", "Y")})
    assert sum(report.missing_match.values()) == 1


# --- guards -----------------------------------------------------------------


def test_an_empty_set_root_is_refused(tmp_path):
    (tmp_path / "set").mkdir()
    with pytest.raises(ShuttleSetFaultError, match="no \\*/set\\*.csv"):
        derive_faults(tmp_path / "set", {})


def test_a_missing_match_file_is_refused(tmp_path):
    with pytest.raises(ShuttleSetFaultError, match="no such match file"):
        load_match_players(tmp_path / "nope.csv")


def test_two_terminal_rows_in_one_rally_are_refused(tmp_path):
    """One rally ends once; a duplicate clip_id would silently overwrite a fault."""
    with pytest.raises(ShuttleSetFaultError, match="derived more than once"):
        _derive(
            tmp_path,
            [
                _row(rally="1", lose_reason="出界", getpoint_player="B"),
                _row(rally="1", lose_reason="掛網", getpoint_player="B"),
            ],
        )


# --- the library the derivation targets -------------------------------------


def test_every_derivable_template_id_exists_in_the_shipped_library():
    """The deriver names templates by convention, so the convention is under test."""
    library = load_templates(SHUTTLESET_LIBRARY)
    expected = {
        f"{family}_{mode}"
        for family in set(STROKE_FAMILIES.values())
        for mode in ("hit_out", "into_net")
    }
    assert expected == set(library)


def test_the_shuttleset_library_is_honest_that_its_corrections_are_drafted():
    """Only the fault statement comes from ShuttleSet; the coaching text does not."""
    library = load_templates(SHUTTLESET_LIBRARY)
    assert {t.provenance.kind for t in library.values()} == {"drafted"}


def test_every_shuttleset_template_offers_several_phrasings():
    library = load_templates(SHUTTLESET_LIBRARY)
    assert all(len(t.corrections) >= 3 for t in library.values())


# --- output -----------------------------------------------------------------


def test_written_faults_round_trip_through_the_reference_builder(tmp_path):
    from feedback_eval.template_references import load_clip_faults, load_clip_players

    faults, _ = _derive(tmp_path, [_row(lose_reason="出界", getpoint_player="B")])
    faults_csv = write_clip_faults(tmp_path / "faults.csv", faults)
    players_csv = write_clip_players(tmp_path / "players.csv", faults)

    assert load_clip_faults(faults_csv) == {
        "Alice_Bob_Open_2026_Finals_s1_r1": ("smash_hit_out",)
    }
    assert load_clip_players(players_csv) == {
        "Alice_Bob_Open_2026_Finals_s1_r1": ("Alice", "Bob")
    }


def test_the_faults_csv_records_who_erred_for_audit(tmp_path):
    """The one thing the derivation knows that the reference set cannot express."""
    faults, _ = _derive(tmp_path, [_row(player="B", lose_reason="出界", getpoint_player="A")])
    path = write_clip_faults(tmp_path / "faults.csv", faults)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["erring_player"] == "B"
