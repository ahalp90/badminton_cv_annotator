"""Player degradation trends over source-scoped rallies (issue #138).

Issue #22 asked for a least-squares trend of each kept per-rally feature
across a player's rallies, tanh-compressed to (-1, 1) so an improving and a
declining player read as opposite signs on one comparable scale. Issue #104
could not fix the tanh temperature, so both the raw slope and its
tanh-normalised form sat in ``schema_v1.FEATURE_DISPOSITIONS`` as unresolved.
The feature's owner has since chosen a fixed temperature over a per-feature
sweep; see ``scripts/degradation_temperature_report.py`` for the evidence
that a temperature of 2.0 gives a reasonable spread without ideas saturating.

Only ``source_contacts`` rallies carry a player identity a reader can trust.
An ``annotator`` row's identity is a guess resolved from an overlapping side
phase (see the ``by_rally_origin`` reliability note in
``docs/dataset_v1_schema.md``), so those rallies take no part in a trend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from dataset_builder.schema_v1 import PLAYER_RALLIES, ColumnType, RallyOrigin, TableSpec


# The teammate who specified degradation (issue #138) chose this over a
# per-feature sweep: "pick a magic number like 2 and hope for the best."
# slope_tanh alone would lose information once it saturates near +/-1, so
# player_trends also stores the raw slope and this constant. That means
# anyone who wants different scaling can recover the exact slope:
# slope = temperature * arctanh(slope_tanh).
DEGRADATION_TEMPERATURE = 2.0

# Fewer points make a trend line noise, not a trend.
MIN_TREND_POINTS = 3

SCOPE_SET = "set"
SCOPE_MATCH = "match"
# ShuttleSet sets are numbered from 1, so 0 can never collide with a real
# set number and is a safe sentinel scope_id for the whole-match scope.
MATCH_SCOPE_ID = 0


def trendable_features(table: TableSpec) -> tuple[str, ...]:
    """Return the table's float-valued feature columns, in column order.

    Selecting by frozen type rather than a hardcoded name list means a new
    float feature added to ``player_rallies`` later, for example a movement
    quality median, is picked up automatically with no change here.
    """
    return tuple(column.name for column in table.columns if column.type is ColumnType.FLOAT)


def least_squares_trend(values: Sequence[float]) -> tuple[float, float]:
    """Return the ordinary least squares slope and intercept of ``values``.

    ``values`` are ordered observations fit against position 0..n-1 in that
    order; the caller decides what the order means (rally order within a
    set, or rally order across the whole match).
    """
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def player_trend_rows(
    identity: tuple[str, str, str],
    rallies: Sequence[Mapping[str, object]],
    player_rallies: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Fit degradation trends for one video's source_contacts rallies.

    Builds one row per (player, scope, feature) with at least
    ``MIN_TREND_POINTS`` non-null values. ``scope="set"`` fits across a
    player's rallies within one ShuttleSet set, ordered by the rally's
    number in that set. ``scope="match"`` fits across every one of that
    player's rallies in the video, ordered by (set, rally), so it does not
    stop at a set boundary. Both scopes fit the same per-rally feature
    values; match scope just does not split them by set.

    :return: the player_trends rows, and a manifest-style population
        summary of how many fits were written and how many were skipped
        for having fewer than ``MIN_TREND_POINTS`` values.
    """
    order_by_rally = {
        int(row["rally_id"]): (int(row["source_set"]), int(row["source_rally"]))
        for row in rallies
        if row["rally_origin"] == RallyOrigin.SOURCE_CONTACTS.value
    }
    features = trendable_features(PLAYER_RALLIES)
    # (player_id, feature) -> [(source_set, source_rally, value), ...]
    series: dict[tuple[str, str], list[tuple[int, int, float]]] = {}
    for row in player_rallies:
        if row["rally_origin"] != RallyOrigin.SOURCE_CONTACTS.value or row["player_id"] is None:
            continue
        order = order_by_rally.get(int(row["rally_id"]))
        if order is None:
            continue
        source_set, source_rally = order
        for feature in features:
            value = row[feature]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            series.setdefault((row["player_id"], feature), []).append(
                (source_set, source_rally, float(value))
            )

    rows: list[dict[str, object]] = []
    population = {"fits_written": 0, "fits_skipped_insufficient_points": 0}
    for (player_id, feature), points in series.items():
        points.sort()
        _fit_row(
            rows, population, identity, player_id, SCOPE_MATCH, MATCH_SCOPE_ID, feature,
            [value for _, _, value in points],
        )

        by_set: dict[int, list[tuple[int, float]]] = {}
        for source_set, source_rally, value in points:
            by_set.setdefault(source_set, []).append((source_rally, value))
        for source_set, set_points in sorted(by_set.items()):
            set_points.sort()
            _fit_row(
                rows, population, identity, player_id, SCOPE_SET, source_set, feature,
                [value for _, value in set_points],
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
    values: list[float],
) -> None:
    """Append one player_trends row when there are enough points, else tally the skip."""
    if len(values) < MIN_TREND_POINTS:
        population["fits_skipped_insufficient_points"] += 1
        return
    slope, intercept = least_squares_trend(values)
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
            "n_points": len(values),
            "slope": slope,
            "intercept": intercept,
            "slope_tanh": math.tanh(slope / DEGRADATION_TEMPERATURE),
            "temperature": DEGRADATION_TEMPERATURE,
        }
    )
    population["fits_written"] += 1
