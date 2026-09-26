"""Describe line directions and check generated court records."""


from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np


def validate_population(record: dict, case_id: str, run_w5: ModuleType, source: str) -> dict:
    run_w5.validate_generation_record(record, case_id, source, expected_stage="results", validate_entries=False)
    entries = record.get("entries")
    if not isinstance(entries, list) or len(entries) > 256:
        raise ValueError(f"{case_id}: {source} has an invalid entry count")
    candidate_ids = [entry.get("candidate_id") for entry in entries]
    if not all(isinstance(candidate_id, str) and candidate_id for candidate_id in candidate_ids):
        raise ValueError(f"{case_id}: {source} has an invalid candidate ID")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{case_id}: {source} candidate IDs are not unique")
    for winner_name in ("line_winner_id", "paint_winner_id"):
        winner_id = record.get(winner_name)
        if winner_id is not None and winner_id not in set(candidate_ids):
            raise ValueError(f"{case_id}: {source} {winner_name} is outside its entries")
    return record


def direction_record(context: Any, settings: dict, vp_pruning: ModuleType) -> dict:
    """Estimate the view's court directions with fixed search settings.

    :param settings: ``vp_pruning.Settings`` fields, as saved in a direction file.
    """
    working_size = tuple(context.size)
    _points, estimator = vp_pruning.estimate(
        np.asarray(context.segments, dtype=float), working_size, vp_pruning.Settings(**settings)
    )
    return {
        "schema": "wider-w5-direction/1",
        "case_id": context.case_id,
        "working_size": list(working_size),
        "settings": settings,
        "estimator": estimator,
    }
