"""Player degradation trends over source-scoped rallies (issue #138).

Issue #22 asked for a least-squares trend of each kept per-rally feature
across a player's rallies, tanh-compressed to (-1, 1) so an improving and a
declining player read as opposite signs on one comparable scale. Issue #104
could not fix the tanh temperature, so both the raw slope and its
tanh-normalised form sat in ``schema_v1.FEATURE_DISPOSITIONS`` as unresolved.
Issue #138 asked to sweep the temperature if that was cheap, and otherwise
pick 2 and accept the risk. The sweep was skipped, so 2 is that fallback. The
raw slope is stored beside the compressed one, so anyone who dislikes that
scaling can undo it exactly.

Only ``source_contacts`` rallies carry a player identity a reader can trust.
An ``annotator`` row's identity is a guess resolved from an overlapping side
phase (see the ``by_rally_origin`` reliability note in
``docs/dataset_v1_schema.md``), so those rallies take no part in a trend.

Two progressions are fit, both read from issue #138's "Progression over
{set,rally}": ``scope="set"`` fits a feature's rally-by-rally values against
the rally's number within one ShuttleSet set, so a rally missing from the
middle of a set keeps its gap in the fit. ``scope="match"`` fits one point
per set, the median of the feature over the player's rallies in that set,
against the set number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from dataset_builder.schema_v1 import PLAYER_RALLIES, RALLIES, ColumnType, RallyOrigin, TableSpec


# Issue #138 asked to sweep a range of temperatures if that was cheap, and
# otherwise pick a magic number like 2. The sweep was skipped, so 2 is that
# named fallback, not a considered choice over the sweep.
# slope_tanh alone would lose information once it saturates near +/-1, so
# player_trends also stores the raw slope and this constant. That means
# anyone who wants different scaling can recover the exact slope:
# slope = temperature * arctanh(slope_tanh).
DEGRADATION_TEMPERATURE = 2.0

# ShuttleSet sets run to dozens of rallies; fewer than 3 points makes a
# rally-by-rally line noise, not a trend. This floor is our choice, not
# something issue #138 sets.
MIN_TREND_POINTS_SET = 3

# A ShuttleSet match has 2 or 3 sets. Requiring 3 points would silently drop
# the whole-match trend for every 2-set match; a line through 2 points is an
# exact fit, not a guess, so 2 is enough here.
MIN_TREND_POINTS_MATCH = 2

SCOPE_SET = "set"
SCOPE_MATCH = "match"
# ShuttleSet sets are numbered from 1, so 0 can never collide with a real
# set number and is a safe sentinel scope_id for the whole-match scope.
MATCH_SCOPE_ID = 0

# Rally-level columns of `rallies` trended per player, when the column
# exists. Both are present since issue #142 added `shots_per_rally`; a future
# rally-level column is picked up the same way, with no change here, and a
# column that is later removed is skipped cleanly instead of raising.
RALLY_LEVEL_FEATURES = ("duration_seconds", "shots_per_rally")


def trendable_features(table: TableSpec) -> tuple[str, ...]:
    """Return the table's float-valued feature columns, in column order.

    :param table: a TableSpec whose float columns should be trended per
        player and court side, for example ``player_rallies``.
    :return: the float column names, discovered by frozen type rather than a
        hardcoded list, so a new float feature is picked up with no change
        here.
    """
    return tuple(column.name for column in table.columns if column.type is ColumnType.FLOAT)


def rally_level_features(table: TableSpec) -> tuple[str, ...]:
    """Return which of RALLY_LEVEL_FEATURES ``table`` currently declares.

    :param table: the ``rallies`` TableSpec to check for candidate columns.
    :return: the candidate names present on ``table``, in RALLY_LEVEL_FEATURES
        order, so a candidate not yet on the frozen schema is skipped cleanly
        rather than raising.
    """
    declared = table.column_names()
    return tuple(name for name in RALLY_LEVEL_FEATURES if name in declared)


def trended_features(player_rallies: TableSpec, rallies: TableSpec) -> tuple[str, ...]:
    """Return every per-player feature this module fits a degradation trend for.

    :param player_rallies: the ``player_rallies`` TableSpec.
    :param rallies: the ``rallies`` TableSpec.
    :return: the union of ``player_rallies``' float columns and ``rallies``'
        discovered rally-level columns, so documentation can list the exact
        trended set without hand-copying it.
    """
    return trendable_features(player_rallies) + rally_level_features(rallies)


def least_squares_trend(x: Sequence[float], y: Sequence[float]) -> float:
    """Return the ordinary least squares slope of ``y`` against ``x``.

    :param x: ordered fit positions: a rally's ``source_rally`` number for
        ``scope=set``, or a set's ``source_set`` number for ``scope=match``.
        Explicit positions keep a gap's spacing in the fit rather than
        renumbering it away.
    :param y: values matched to ``x``, same length.
    :return: the slope of the best-fit line.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    slope, _intercept = np.polyfit(x_arr, y_arr, 1)
    return float(slope)


def player_trend_rows(
    identity: tuple[str, str, str],
    rallies: Sequence[Mapping[str, object]],
    player_rallies: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Fit degradation trends for one video's source_contacts rallies.

    Values come from two sources: ``player_rallies``' float columns, read per
    court side and joined to a player through the matching ``rallies`` row;
    and named rally-level columns of ``rallies`` (``RALLY_LEVEL_FEATURES``),
    read once per player named on the rally (``top_player_id`` or
    ``bottom_player_id``). Both feed the same two progressions.

    :param identity: (run_id, source_dataset, video_id) written on every row.
    :param rallies: this video's rally rows of both origins; only
        source_contacts rows contribute.
    :param player_rallies: this video's player_rallies rows of both origins;
        only source_contacts rows with a resolved player_id contribute.
    :return: the player_trends rows, and a manifest-style population summary
        of fits written and, per scope, fits skipped for too few points.
    """
    order_by_rally = {
        int(row["rally_id"]): (int(row["source_set"]), int(row["source_rally"]))
        for row in rallies
        if row["rally_origin"] == RallyOrigin.SOURCE_CONTACTS.value
    }

    # (player_id, feature) -> {(source_set, source_rally): value}. A dict
    # keyed on the rally, not a list, so the same rally can only ever
    # contribute one point per feature.
    series: dict[tuple[str, str], dict[tuple[int, int], float]] = {}

    def _record(player_id: object, feature: str, source_set: int, source_rally: int, value: object) -> None:
        if player_id is None or value is None or (isinstance(value, float) and math.isnan(value)):
            return
        series.setdefault((player_id, feature), {})[(source_set, source_rally)] = float(value)

    for row in player_rallies:
        if row["rally_origin"] != RallyOrigin.SOURCE_CONTACTS.value or row["player_id"] is None:
            continue
        order = order_by_rally.get(int(row["rally_id"]))
        if order is None:
            continue
        source_set, source_rally = order
        for feature in trendable_features(PLAYER_RALLIES):
            _record(row["player_id"], feature, source_set, source_rally, row[feature])

    rally_features = rally_level_features(RALLIES)
    if rally_features:
        for row in rallies:
            if row["rally_origin"] != RallyOrigin.SOURCE_CONTACTS.value:
                continue
            source_set, source_rally = int(row["source_set"]), int(row["source_rally"])
            for side_column in ("top_player_id", "bottom_player_id"):
                player_id = row.get(side_column)
                for feature in rally_features:
                    _record(player_id, feature, source_set, source_rally, row.get(feature))

    rows: list[dict[str, object]] = []
    population = {
        "fits_written": 0,
        "fits_skipped_insufficient_points_set": 0,
        "fits_skipped_insufficient_points_match": 0,
    }
    for (player_id, feature), points_by_key in series.items():
        by_set: dict[int, list[tuple[int, float]]] = {}
        for (source_set, source_rally), value in points_by_key.items():
            by_set.setdefault(source_set, []).append((source_rally, value))

        for source_set, set_points in sorted(by_set.items()):
            set_points.sort()
            _fit_row(
                rows, population, identity, player_id, SCOPE_SET, source_set, feature,
                x=[float(source_rally) for source_rally, _ in set_points],
                y=[value for _, value in set_points],
                min_points=MIN_TREND_POINTS_SET,
                skip_key="fits_skipped_insufficient_points_set",
            )

        match_points = sorted(
            (source_set, float(np.median([value for _, value in set_points])))
            for source_set, set_points in by_set.items()
        )
        _fit_row(
            rows, population, identity, player_id, SCOPE_MATCH, MATCH_SCOPE_ID, feature,
            x=[float(source_set) for source_set, _ in match_points],
            y=[value for _, value in match_points],
            min_points=MIN_TREND_POINTS_MATCH,
            skip_key="fits_skipped_insufficient_points_match",
        )

    return rows, population


def _fit_row(
    rows: list[dict[str, object]],
    population: dict[str, int],
    identity: tuple[str, str, str],
    player_id: str,
    scope: str,
    scope_id: int,
    feature: str,
    x: list[float],
    y: list[float],
    min_points: int,
    skip_key: str,
) -> None:
    """Append one player_trends row when there are enough points, else tally the scope's skip.

    :param x: fit positions, matched to ``y``.
    :param y: feature values to fit.
    :param min_points: the scope's minimum point count.
    :param skip_key: the population counter to increment when there are too
        few points to fit.
    """
    if len(y) < min_points:
        population[skip_key] += 1
        return
    slope = least_squares_trend(x, y)
    run_id, source_dataset, video_id = identity
    rows.append(
        {
            "run_id": run_id,
            "source_dataset": source_dataset,
            "video_id": video_id,
            "player_id": player_id,
            "scope": scope,
            "scope_id": scope_id,
            "feature": feature,
            "n_points": len(y),
            "slope": slope,
            "slope_tanh": math.tanh(slope / DEGRADATION_TEMPERATURE),
            "temperature": DEGRADATION_TEMPERATURE,
        }
    )
    population["fits_written"] += 1
