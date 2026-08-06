"""Pure best-config selection rules for calibration score rows.

The calibration sweep must supply every score row as a dictionary with
``label`` (``"grid"`` for configs that may win), ``changed_from_defaults``,
and ``settings``. It must also supply the existing aggregate fields, including
``covered``, ``covered_fraction``, ``split``, ``missed``, ``spurious_spans``,
``recall_5``, ``precision_raw_5``, and ``start_alignment_median``. The strict boundary
fields are ``clean_covered`` (the strict true-positive count),
``swallowed_rallies`` (the sum, over merged spans, of contained rallies minus
one), ``max_rallies_in_one_span``, ``strict_align_median``, and
``strict_align_p90``.  Strict alignment fields are both ``None`` when no rally
is cleanly identified.

The sweep computes those diagnostics and orchestrates selection. If its quality
floor fails, it withholds the best-config verdict and reports no winner. An
empty requested grid is instead a sweep configuration error before either phase.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from annotator.calibration.scoring import safe_f1


ScoreRow = dict[str, Any]
SortKey = tuple[Any, ...]
GridKey = Callable[[ScoreRow], SortKey]

GRID_LABEL = "grid"
COVERAGE_ALLOWANCE_FRACTION = 1.0
MINIMUM_BEST_COVERED_FRACTION = 0.0


def standard_tail(row: ScoreRow) -> SortKey:
    """Return the deterministic final tie rule for every selection key.

    Fewer changed settings win first.  The supplied settings tuple settles an
    otherwise exact tie, so the caller must provide values that have a total
    order.
    """
    return (row["changed_from_defaults"], row["settings"])


def boundary_live_key_rally_id_f1(row: ScoreRow, n_gt_rallies: int) -> SortKey:
    """Return the strict live boundary key, with an exact tie settled by the tail.

    Strict F1 uses ``clean_covered`` as TP, ``n_gt_rallies - TP`` as FN, and
    ``spurious_spans`` as FP.  A ``None`` strict alignment median sorts after
    every measured median; p90 is report-only and does not affect this key.
    """
    true_positives = row["clean_covered"]
    false_negatives = n_gt_rallies - true_positives
    false_positives = row["spurious_spans"]
    denominator = (2 * true_positives) + false_negatives + false_positives
    f1 = 0.0 if denominator == 0 else (2 * true_positives) / denominator
    median = row["strict_align_median"]
    sortable_median = median if median is not None else float("inf")
    return (-f1, sortable_median, *standard_tail(row))


def boundary_report_key_fewest_swallowed_rallies(row: ScoreRow) -> SortKey:
    """Return the fewest-swallowed-rallies key, with an exact tie settled by the tail.

    All inputs are supplied counts, so this key has no ``None`` conversion.
    """
    return (
        row["swallowed_rallies"],
        row["split"],
        row["missed"],
        row["spurious_spans"],
        *standard_tail(row),
    )


def boundary_report_key_coverage_first(row: ScoreRow) -> SortKey:
    """Return the coverage-first report key, with an exact tie settled by the tail.

    A missing covered fraction sorts below every valid fraction by mapping to
    ``-1.0`` before the descending comparison.
    """
    covered_fraction = row["covered_fraction"]
    sortable_fraction = covered_fraction if covered_fraction is not None else -1.0
    return (
        -sortable_fraction,
        row["split"],
        row["missed"],
        row["spurious_spans"],
        row["swallowed_rallies"],
        *standard_tail(row),
    )


def boundary_report_key_tightest_start(row: ScoreRow) -> SortKey:
    """Return the tightest-start report key, with an exact tie settled by the tail.

    A missing start alignment median has no measurable tightness, so it maps to
    positive infinity and sorts last.
    """
    median = row["start_alignment_median"]
    absolute_median = abs(median) if median is not None else float("inf")
    return (
        absolute_median,
        row["swallowed_rallies"],
        row["split"],
        row["missed"],
        row["spurious_spans"],
        *standard_tail(row),
    )


def f1_raw_5(row: ScoreRow) -> float | None:
    """Return raw +/-5 F1, or ``None`` when either required metric is missing.

    The harmonic mean uses ``recall_5`` and ``precision_raw_5`` only.  A zero
    sum yields zero F1 rather than a division error.
    """
    recall = row["recall_5"]
    precision = row["precision_raw_5"]
    if recall is None or precision is None:
        return None
    return safe_f1(precision, recall)


def contact_meets_floors(
    row: ScoreRow,
    minimum_precision: float | None = None,
    minimum_recall: float | None = None,
) -> bool:
    """Return whether a contact row clears optional metric floors.

    An omitted floor does not constrain selection.  If a configured floor lacks
    its row operand, eligibility fails closed; an exact threshold tie passes.
    """
    precision = row["precision_raw_5"]
    recall = row["recall_5"]
    if minimum_precision is not None and (
        precision is None or precision < minimum_precision
    ):
        return False
    if minimum_recall is not None and (recall is None or recall < minimum_recall):
        return False
    return True


def contact_live_key_raw_f1(row: ScoreRow) -> SortKey:
    """Return the live contact key, with an exact tie settled by the tail.

    Callers pass only rows that clear configured floors and have a raw +/-5 F1.
    Missing F1 is rejected with ``ValueError`` so a caller cannot accidentally
    sort an ineligible row as though it had a score.
    """
    raw_f1 = f1_raw_5(row)
    if raw_f1 is None:
        raise ValueError("contact live key requires recall_5 and precision_raw_5")
    return (-raw_f1, *standard_tail(row))


def grid_rows(rows: Iterable[ScoreRow]) -> list[ScoreRow]:
    """Return grid rows only, preserving input order before deterministic sorting.

    Rows with any other label are references and cannot win.  Missing labels are
    invalid input and raise through normal dictionary access.
    """
    return [row for row in rows if row["label"] == GRID_LABEL]


def coverage_allowance_rows(
    rows: Iterable[ScoreRow],
    n_gt_rallies: int,
    coverage_allowance_fraction: float = COVERAGE_ALLOWANCE_FRACTION,
) -> list[ScoreRow]:
    """Return grid rows within the fractional coverage allowance of the best.

    Reference rows are excluded.  The allowance is multiplied by the supplied
    GT-rally count without integer rounding; exact allowance-boundary rows pass.
    An empty grid raises ``ValueError`` because selection has no candidate.
    """
    candidates = grid_rows(rows)
    if not candidates:
        raise ValueError("no grid rows to select from")
    best_covered = max(row["covered"] for row in candidates)
    allowance = coverage_allowance_fraction * n_gt_rallies
    return [
        row for row in candidates if best_covered - row["covered"] <= allowance
    ]


def select_contact_live_winner(
    rows: Iterable[ScoreRow],
    minimum_precision: float | None = None,
    minimum_recall: float | None = None,
) -> ScoreRow | None:
    """Return the floored-F1 contact winner, or ``None`` when none is eligible.

    Reference rows are excluded.  Floors fail closed on missing operands and a
    missing raw F1 is ineligible; the remaining key's standard tail settles ties.
    """
    candidates = [
        row
        for row in grid_rows(rows)
        if contact_meets_floors(row, minimum_precision, minimum_recall)
        and f1_raw_5(row) is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=contact_live_key_raw_f1)


def best_config_clears_quality_floor(
    best_grid_row: ScoreRow,
    minimum_best_covered_fraction: float = MINIMUM_BEST_COVERED_FRACTION,
) -> bool:
    """Return whether the supplied best grid row clears the quality floor.

    A missing covered fraction fails closed. Equality clears the floor; the
    caller withholds the verdict when this predicate is false.
    """
    covered_fraction = best_grid_row["covered_fraction"]
    return (
        covered_fraction is not None
        and covered_fraction >= minimum_best_covered_fraction
    )
