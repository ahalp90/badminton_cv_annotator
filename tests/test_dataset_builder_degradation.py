from __future__ import annotations

import pytest

from dataset_builder import degradation
from dataset_builder.degradation import (
    DEGRADATION_TEMPERATURE,
    MATCH_SCOPE_ID,
    least_squares_trend,
    player_trend_rows,
    rally_level_features,
    trendable_features,
    trended_features,
)
from dataset_builder.schema_v1 import PLAYER_RALLIES, RALLIES


IDENTITY = ("run1", "shuttleset", "0001")


def _rally(
    rally_id: int,
    origin: str,
    source_set: int | None = None,
    source_rally: int | None = None,
    top_player_id: str | None = None,
    bottom_player_id: str | None = None,
    duration_seconds: float | None = None,
) -> dict[str, object]:
    return {
        "rally_id": rally_id,
        "rally_origin": origin,
        "source_set": source_set,
        "source_rally": source_rally,
        "top_player_id": top_player_id,
        "bottom_player_id": bottom_player_id,
        "duration_seconds": duration_seconds,
    }


def _player_row(
    rally_id: int,
    origin: str,
    player_id: str | None,
    posture_mad: float | None,
    recovery_distance_median: float | None = None,
    movement_inefficiency_median: float | None = None,
) -> dict[str, object]:
    return {
        "rally_id": rally_id,
        "rally_origin": origin,
        "player_id": player_id,
        "posture_mad": posture_mad,
        "recovery_distance_median": recovery_distance_median,
        "movement_inefficiency_median": movement_inefficiency_median,
    }


def test_trendable_features_discovers_float_columns() -> None:
    # Issue #142 added recovery_distance_median and movement_inefficiency_median
    # as float player_rallies columns alongside posture_mad; a future float
    # feature is picked up with no change to this module.
    assert trendable_features(PLAYER_RALLIES) == (
        "posture_mad", "recovery_distance_median", "movement_inefficiency_median",
    )


def test_rally_level_features_returns_every_declared_candidate() -> None:
    # Issue #142 added rallies.shots_per_rally, so both RALLY_LEVEL_FEATURES
    # candidates are now declared on the frozen `rallies` table.
    assert rally_level_features(RALLIES) == ("duration_seconds", "shots_per_rally")


def test_rally_level_features_skips_a_name_not_on_the_frozen_table(monkeypatch) -> None:
    # A candidate not yet on the frozen `rallies` table must be skipped
    # cleanly rather than raising: the same guard that let shots_per_rally
    # sit here safely before issue #142 added the column.
    monkeypatch.setattr(
        degradation, "RALLY_LEVEL_FEATURES", ("duration_seconds", "not_a_real_column")
    )
    assert rally_level_features(RALLIES) == ("duration_seconds",)


def test_trended_features_is_the_union_used_for_documentation() -> None:
    assert trended_features(PLAYER_RALLIES, RALLIES) == (
        "posture_mad", "recovery_distance_median", "movement_inefficiency_median",
        "duration_seconds", "shots_per_rally",
    )


def test_least_squares_trend_known_sequence() -> None:
    slope = least_squares_trend([0.0, 1.0, 2.0, 3.0], [10.0, 12.0, 14.0, 16.0])
    assert slope == pytest.approx(2.0)


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
    # Set scope: 2 rally points, below the 3-point floor. Match scope: the
    # only set collapses to 1 median point, below the 2-point floor.
    assert population == {
        "fits_written": 0,
        "fits_skipped_insufficient_points_set": 1,
        "fits_skipped_insufficient_points_match": 1,
    }


def test_null_feature_values_are_ignored() -> None:
    rallies = [_rally(i, "source_contacts", 1, i + 1) for i in range(4)]
    player_rallies = [
        _player_row(0, "source_contacts", "alice", 2.0),
        _player_row(1, "source_contacts", "alice", None),
        _player_row(2, "source_contacts", "alice", 6.0),
        _player_row(3, "source_contacts", "alice", 8.0),
    ]

    rows, population = player_trend_rows(IDENTITY, rallies, player_rallies)

    set_row = next(row for row in rows if row["scope"] == "set")
    # The null drops out entirely rather than counting as a point or a gap:
    # 3 remaining points at source_rally (1, 3, 4), values (2, 6, 8) sit
    # exactly on a line of slope 2 despite the gap at source_rally 2.
    assert set_row["n_points"] == 3
    assert set_row["slope"] == pytest.approx(2.0)
    assert population["fits_skipped_insufficient_points_set"] == 0
    # Only one set exists, so scope=match sees a single median point and is
    # skipped rather than written.
    assert population["fits_skipped_insufficient_points_match"] == 1


def test_missing_player_id_is_ignored() -> None:
    rallies = [_rally(i, "source_contacts", 1, i + 1) for i in range(3)]
    player_rallies = [_player_row(i, "source_contacts", None, float(i)) for i in range(3)]

    rows, population = player_trend_rows(IDENTITY, rallies, player_rallies)

    assert rows == []
    assert population == {
        "fits_written": 0,
        "fits_skipped_insufficient_points_set": 0,
        "fits_skipped_insufficient_points_match": 0,
    }


def test_annotator_rallies_produce_no_rows() -> None:
    rallies = [_rally(i, "annotator") for i in range(5)]
    player_rallies = [_player_row(i, "annotator", "alice", float(i)) for i in range(5)]

    rows, population = player_trend_rows(IDENTITY, rallies, player_rallies)

    assert rows == []
    assert population == {
        "fits_written": 0,
        "fits_skipped_insufficient_points_set": 0,
        "fits_skipped_insufficient_points_match": 0,
    }


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
        if row["scope"] == "set"
    )
    shallow_row = next(
        row
        for row in player_trend_rows(IDENTITY, rallies, shallow)[0]
        if row["scope"] == "set"
    )

    assert steep_row["temperature"] == DEGRADATION_TEMPERATURE
    assert steep_row["slope_tanh"] == pytest.approx(1.0, abs=1e-6)
    # A gentle slope stays well inside (-1, 1), so direction and magnitude both survive.
    assert 0.0 < shallow_row["slope_tanh"] < 0.3


def test_set_scope_uses_source_rally_as_x_and_preserves_a_gap() -> None:
    # source_rally jumps from 2 to 5: a missing rally in between. Fit against
    # the real rally numbers, y = x, gives an exact slope of 1. Renumbering
    # to consecutive positions (0, 1, 2) would instead read a slope of 2.
    rallies = [
        _rally(0, "source_contacts", 1, 1),
        _rally(1, "source_contacts", 1, 2),
        _rally(2, "source_contacts", 1, 5),
    ]
    player_rallies = [
        _player_row(0, "source_contacts", "alice", 1.0),
        _player_row(1, "source_contacts", "alice", 2.0),
        _player_row(2, "source_contacts", "alice", 5.0),
    ]

    rows, _ = player_trend_rows(IDENTITY, rallies, player_rallies)

    set_row = next(row for row in rows if row["scope"] == "set")
    assert set_row["slope"] == pytest.approx(1.0)


def test_match_scope_has_one_point_per_set_and_fits_with_two_sets() -> None:
    rallies = [
        _rally(0, "source_contacts", 1, 1),
        _rally(1, "source_contacts", 1, 2),
        _rally(2, "source_contacts", 1, 3),
        _rally(3, "source_contacts", 2, 1),
        _rally(4, "source_contacts", 2, 2),
        _rally(5, "source_contacts", 2, 3),
    ]
    # Set 1 values (1, 2, 3) median to 2.0; set 2 values (4, 5, 6) median to
    # 5.0. A match match has 2 or 3 sets, so a 2-set match must still fit.
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    player_rallies = [
        _player_row(i, "source_contacts", "alice", value) for i, value in enumerate(values)
    ]

    rows, population = player_trend_rows(IDENTITY, rallies, player_rallies)

    match_row = next(row for row in rows if row["scope"] == "match")
    assert match_row["n_points"] == 2
    assert match_row["slope"] == pytest.approx(3.0)
    assert population["fits_skipped_insufficient_points_match"] == 0


def test_set_and_match_scopes_both_fit_with_different_slopes_across_a_set_boundary() -> None:
    rallies = [
        _rally(0, "source_contacts", 1, 1),
        _rally(1, "source_contacts", 1, 2),
        _rally(2, "source_contacts", 1, 3),
        _rally(3, "source_contacts", 2, 1),
        _rally(4, "source_contacts", 2, 2),
        _rally(5, "source_contacts", 2, 3),
    ]
    # Rising through set 1, falling through set 2: the two set-scoped fits
    # disagree in sign, and the match fit (one median point per set, 2.0 then
    # 5.0) sees a third slope.
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
    assert by_scope[("match", MATCH_SCOPE_ID)]["n_points"] == 2
    assert by_scope[("match", MATCH_SCOPE_ID)]["slope"] == pytest.approx(3.0)
    assert population == {
        "fits_written": 3,
        "fits_skipped_insufficient_points_set": 0,
        "fits_skipped_insufficient_points_match": 0,
    }
    for row in rows:
        assert row["run_id"] == "run1"
        assert row["source_dataset"] == "shuttleset"
        assert row["video_id"] == "0001"
        assert row["player_id"] == "alice"
        assert row["feature"] == "posture_mad"


def test_rally_level_feature_is_trended_from_rallies() -> None:
    # duration_seconds lives on `rallies`, not `player_rallies`. Both
    # top_player_id and bottom_player_id take one point per rally they played.
    rallies = [
        _rally(
            i, "source_contacts", 1, i + 1,
            top_player_id="alice", bottom_player_id="bob",
            duration_seconds=float(10 + i),
        )
        for i in range(3)
    ]

    rows, _ = player_trend_rows(IDENTITY, rallies, [])

    alice_row = next(
        row for row in rows
        if row["player_id"] == "alice" and row["feature"] == "duration_seconds"
        and row["scope"] == "set"
    )
    bob_row = next(
        row for row in rows
        if row["player_id"] == "bob" and row["feature"] == "duration_seconds"
        and row["scope"] == "set"
    )
    assert alice_row["slope"] == pytest.approx(1.0)
    assert bob_row["slope"] == pytest.approx(1.0)
    assert alice_row["n_points"] == 3


def test_shots_per_rally_is_trended_now_that_the_column_exists() -> None:
    # Issue #142 added rallies.shots_per_rally, so it is discovered and
    # trended the same way duration_seconds always has been.
    rallies = [
        _rally(
            i, "source_contacts", 1, i + 1,
            top_player_id="alice", duration_seconds=1.0,
        )
        for i in range(3)
    ]
    for row, value in zip(rallies, [2, 3, 4]):
        row["shots_per_rally"] = value

    rows, _ = player_trend_rows(IDENTITY, rallies, [])

    shots_row = next(
        row for row in rows
        if row["feature"] == "shots_per_rally" and row["scope"] == "set"
    )
    assert shots_row["n_points"] == 3
    assert shots_row["slope"] == pytest.approx(1.0)


def test_undeclared_rally_level_column_is_ignored_without_error() -> None:
    # A made-up column name that is not one of RALLY_LEVEL_FEATURES must
    # never surface as a trended feature, whether or not the row dict
    # happens to carry it.
    rallies = [
        _rally(i, "source_contacts", 1, i + 1, top_player_id="alice", duration_seconds=1.0)
        for i in range(3)
    ]
    for row in rallies:
        row["not_a_real_column"] = 4

    rows, _ = player_trend_rows(IDENTITY, rallies, [])

    assert all(row["feature"] != "not_a_real_column" for row in rows)
