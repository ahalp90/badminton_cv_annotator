"""Fixed calibration output schemas; every writer emits its header when empty.

Contact frontier rows use raw ``recall_5`` and ``precision_raw_5`` axes.  The
runner decides frontier membership; this module only supplies its fixed header.
"""

from __future__ import annotations

from enum import Enum
from numbers import Real
from typing import Any, Mapping

from annotator.calibration.scoring import CONTACT_TOLERANCES_BASE30


FROZEN_ROW_COLUMNS = (
    "label",
    "rest_speed",
    "rest_window",
    "end_rest_frames",
    "start_speed",
    "start_min_frames",
    "smooth_window",
    "min_contact_speed",
    "n_spans",
    "covered",
    "covered_fraction",
    "split",
    "missed",
    "merged_spans",
    "spurious_spans",
    "start_alignment_mean",
    "start_alignment_median",
    "count_gate_covered_fraction",
    "count_gate_unmerged_fraction",
    "recall_1",
    "precision_1",
    "f1_1",
    "recall_2",
    "precision_2",
    "f1_2",
    "recall_5",
    "precision_5",
    "f1_5",
    "recall_10",
    "precision_10",
    "f1_10",
    "total_candidates",
    "precision_raw_1",
    "precision_raw_2",
    "precision_raw_5",
    "precision_raw_10",
)

TOLERANCE_FRAME_COLUMNS = (
    "tolerance_frames_1",
    "tolerance_frames_2",
    "tolerance_frames_5",
    "tolerance_frames_10",
)

BOUNDARY_DIAGNOSTIC_COLUMNS = (
    "clean_covered",
    "swallowed_rallies",
    "max_rallies_in_one_span",
    "strict_align_median",
    "strict_align_p90",
    "strict_f1",
)

BOUNDARY_SWEEP_COLUMNS = (
    *FROZEN_ROW_COLUMNS,
    *BOUNDARY_DIAGNOSTIC_COLUMNS,
    *TOLERANCE_FRAME_COLUMNS,
)

CONTACT_SWEEP_COLUMNS = (
    *FROZEN_ROW_COLUMNS,
    # The live impulse axes sit outside the frozen prefix so old and new contact
    # CSVs stay column-comparable in the frozen order; the swept smoothing value
    # rides the frozen prefix's smooth_window column.
    "impulse_floor_half_window_frames",
    "contact_dedup_radius_frames",
    "contact_impulse_multiple",
    "f1_raw_5",
    *TOLERANCE_FRAME_COLUMNS,
)

BEST_CONFIG_COMPARISON_COLUMNS = (
    "rule",
    *BOUNDARY_SWEEP_COLUMNS,
    "coverage_gap_from_best",
    "needed_allowance",
)

CONTACT_FRONTIER_COLUMNS = CONTACT_SWEEP_COLUMNS

CONTACT_STABILITY_COLUMNS = (
    "rule",
    *CONTACT_SWEEP_COLUMNS,
    "same_winner_as_live",
)

SPLIT_LOG_COLUMNS = (
    "video_id",
    "gt_rally_index",
    "gt_start",
    "gt_end",
    "piece_count",
    "piece_spans",
)

ALIGNMENT_COLUMNS = (
    "rule",
    "n_rallies",
    "median_abs_start_offset",
    "p90_abs_start_offset",
)

ALIGNMENT_OWN_COVERED_COLUMNS = ALIGNMENT_COLUMNS
ALIGNMENT_SHARED_COLUMNS = ALIGNMENT_COLUMNS

CSV_COLUMNS_BY_FILENAME = {
    "boundary_sweep.csv": BOUNDARY_SWEEP_COLUMNS,
    "contact_sweep.csv": CONTACT_SWEEP_COLUMNS,
    "best_config_comparison.csv": BEST_CONFIG_COMPARISON_COLUMNS,
    "contact_frontier.csv": CONTACT_FRONTIER_COLUMNS,
    "contact_stability.csv": CONTACT_STABILITY_COLUMNS,
    "split_log.csv": SPLIT_LOG_COLUMNS,
    "alignment_own_covered.csv": ALIGNMENT_OWN_COVERED_COLUMNS,
    "alignment_shared.csv": ALIGNMENT_SHARED_COLUMNS,
}

WINNER_JSON_META_KEY = "meta"
WINNER_JSON_BOUNDARY_KEY = "boundary"
WINNER_JSON_CONTACT_KEY = "contact"
WINNER_JSON_PHASE_KEYS = (WINNER_JSON_BOUNDARY_KEY, WINNER_JSON_CONTACT_KEY)
WINNER_JSON_META_KEYS = (
    "fixture",
    "phases_run",
    "verdict",
    "tolerances_base30",
)
WINNER_JSON_VERDICT_ISSUED = "issued"
WINNER_JSON_TOLERANCES_BASE30 = CONTACT_TOLERANCES_BASE30
WINNER_SPEC_OVERRIDES_KEY = "overrides_base30"
WINNER_SPEC_STRATEGIES_KEY = "strategies"
WINNER_SPEC_KEYS = (WINNER_SPEC_OVERRIDES_KEY, WINNER_SPEC_STRATEGIES_KEY)


def serialise_winner_strategy(value: object) -> object:
    """Return enum strategies by name so JSON never receives their integer value."""
    return value.name if isinstance(value, Enum) else value


def winner_spec(
    overrides_base30: Mapping[str, Real],
    strategies: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Build one phase's winner payload from its swept numeric and strategy axes."""
    return {
        WINNER_SPEC_OVERRIDES_KEY: dict(overrides_base30),
        WINNER_SPEC_STRATEGIES_KEY: {
            field: serialise_winner_strategy(value)
            for field, value in strategies.items()
        },
    }


def winner_document(
    fixture: str,
    phases_run: list[str],
    *,
    boundary: Mapping[str, Any] | None = None,
    contact: Mapping[str, Any] | None = None,
    schema_version: int | None = None,
    tuning_video_ids: list[int] | None = None,
    input_digests: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build config_winner.json; a missing phase mapping omits its key.

    Phase keys carry the specs the document depends on (a contact-only run
    passes its loaded boundary spec too); ``phases_run`` records only the work
    the invocation performed.
    """
    meta: dict[str, object] = {
        "fixture": fixture,
        "phases_run": phases_run,
        "verdict": WINNER_JSON_VERDICT_ISSUED,
        "tolerances_base30": list(WINNER_JSON_TOLERANCES_BASE30),
    }
    document: dict[str, object] = {WINNER_JSON_META_KEY: meta}
    if schema_version is not None or tuning_video_ids is not None or input_digests is not None:
        if schema_version is None or tuning_video_ids is None or input_digests is None:
            raise ValueError("winner provenance fields must be supplied together")
        meta.update({
            "schema_version": schema_version,
            "tuning_video_ids": tuning_video_ids,
            "input_digests": dict(input_digests),
        })
    if boundary is not None:
        document[WINNER_JSON_BOUNDARY_KEY] = dict(boundary)
    if contact is not None:
        document[WINNER_JSON_CONTACT_KEY] = dict(contact)
    return document
