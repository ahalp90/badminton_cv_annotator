from __future__ import annotations

import pytest

from dataset_builder.degradation import (
    DEGRADATION_TEMPERATURE,
    MATCH_SCOPE_ID,
    least_squares_trend,
    player_trend_rows,
    trendable_features,
)
from dataset_builder.schema_v1 import PLAYER_RALLIES


IDENTITY = ("run1", "shuttleset", "0001")


def _rally(
    rally_id: int, origin: str, source_set: int | None = None, source_rally: int | None = None
) -> dict[str, object]:
    return {
        "rally_id": rally_id,
        "rally_origin": origin,
        "source_set": source_set,
        "source_rally": source_rally,
    }


def _player_row(
    rally_id: int, origin: str, player_id: str | None, posture_mad: float | None
) -> dict[str, object]:
    return {
        "rally_id": rally_id,
        "rally_origin": origin,
        "player_id": player_id,
        "posture_mad": posture_mad,
    }


def test_trendable_features_discovers_float_columns() -> None:
    # posture_mad is the only float-valued player_rallies column on main today;
    # a future float feature is picked up with no change to this module.
    assert trendable_features(PLAYER_RALLIES) == ("posture_mad",)


def test_least_squares_trend_known_sequence() -> None:
    slope, intercept = least_squares_trend([10.0, 12.0, 14.0, 16.0])
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(10.0)


def test_fewer_than_three_points_emits_no_row() -> None:
    rallies = [
        _rally(0, "source_contacts", 1, 1),
        _rally(1, "source_contacts", 1, 2),
    ]
    player_rallies = [
        _player_row(0, "source_contacts", "alice", 1.0),
        _player_row(1, "source_contacts", "alice", 2.0),
    ]

    rows, population = player_trend_rows(IDENTITY, rallies, player_rallies)

    assert rows == []
    # One skip for the (only) set, one for the match, since both see the same 2 points.
    assert population == {"fits_written": 0, "fits_skipped_insufficient_points": 2}


def test_null_feature_values_are_ignored() -> None:
    rallies = [_rally(i, "source_contacts", 1, i + 1) for i in range(4)]
    player_rallies = [
        _player_row(0, "source_contacts", "alice", 1.0),
        _player_row(1, "source_contacts", "alice", None),
        _player_row(2, "source_contacts", "alice", 3.0),
        _player_row(3, "source_contacts", "alice", 5.0),
    ]

    rows, population = player_trend_rows(IDENTITY, rallies, player_rallies)

    match_row = next(row for row in rows if row["scope"] == "match")
    # The null drops out entirely rather than counting as a point or a gap:
    # 3 remaining values (1, 3, 5) at positions (0, 1, 2), slope 2.
    assert match_row["n_points"] == 3
    assert match_row["slope"] == pytest.approx(2.0)
    assert population["fits_skipped_insufficient_points"] == 0


def test_missing_player_id_is_ignored() -> None:
    rallies = [_rally(i, "source_contacts", 1, i + 1) for i in range(3)]
    player_rallies = [_player_row(i, "source_contacts", None, float(i)) for i in range(3)]

    rows, population = player_trend_rows(IDENTITY, rallies, player_rallies)

    assert rows == []
    assert population == {"fits_written": 0, "fits_skipped_insufficient_points": 0}


def test_annotator_rallies_produce_no_rows() -> None:
    rallies = [_rally(i, "annotator") for i in range(5)]
    player_rallies = [_player_row(i, "annotator", "alice", float(i)) for i in range(5)]

    rows, population = player_trend_rows(IDENTITY, rallies, player_rallies)

    assert rows == []
    assert population == {"fits_written": 0, "fits_skipped_insufficient_points": 0}


def test_slope_tanh_saturates_for_a_large_slope_and_stays_open_for_a_small_one() -> None:
    rallies = [_rally(i, "source_contacts", 1, i + 1) for i in range(3)]
    steep = [
        _player_row(0, "source_contacts", "alice", 0.0),
        _player_row(1, "source_contacts", "alice", 1000.0),
        _player_row(2, "source_contacts", "alice", 2000.0),
    ]
    shallow = [
        _player_row(0, "source_contacts", "bob", 1.0),
        _player_row(1, "source_contacts", "bob", 1.5),
        _player_row(2, "source_contacts", "bob", 2.0),
    ]

    steep_row = next(
        row
        for row in player_trend_rows(IDENTITY, rallies, steep)[0]
        if row["scope"] == "match"
    )
    shallow_row = next(
        row
        for row in player_trend_rows(IDENTITY, rallies, shallow)[0]
        if row["scope"] == "match"
    )

    assert steep_row["temperature"] == DEGRADATION_TEMPERATURE
    assert steep_row["slope_tanh"] == pytest.approx(1.0, abs=1e-6)
    # A gentle slope stays well inside (-1, 1), so direction and magnitude both survive.
    assert 0.0 < shallow_row["slope_tanh"] < 0.3


def test_set_and_match_scopes_both_fit_with_different_slopes_across_a_set_boundary() -> None:
    rallies = [
        _rally(0, "source_contacts", 1, 1),
        _rally(1, "source_contacts", 1, 2),
        _rally(2, "source_contacts", 1, 3),
        _rally(3, "source_contacts", 2, 1),
        _rally(4, "source_contacts", 2, 2),
        _rally(5, "source_contacts", 2, 3),
    ]
    # Rising through set 1, falling through set 2: the whole-match fit sees a
    # different slope than either set-scoped fit.
    values = [1.0, 2.0, 3.0, 6.0, 5.0, 4.0]
    player_rallies = [
        _player_row(i, "source_contacts", "alice", value) for i, value in enumerate(values)
    ]

    rows, population = player_trend_rows(IDENTITY, rallies, player_rallies)

    by_scope = {(row["scope"], row["scope_id"]): row for row in rows}
    assert set(by_scope) == {("set", 1), ("set", 2), ("match", MATCH_SCOPE_ID)}
    assert by_scope[("set", 1)]["slope"] == pytest.approx(1.0)
    assert by_scope[("set", 1)]["n_points"] == 3
    assert by_scope[("set", 2)]["slope"] == pytest.approx(-1.0)
    assert by_scope[("match", MATCH_SCOPE_ID)]["n_points"] == 6
    assert population == {"fits_written": 3, "fits_skipped_insufficient_points": 0}
    for row in rows:
        assert row["run_id"] == "run1"
        assert row["source_dataset"] == "shuttleset"
        assert row["video_id"] == "0001"
        assert row["player_id"] == "alice"
        assert row["feature"] == "posture_mad"
