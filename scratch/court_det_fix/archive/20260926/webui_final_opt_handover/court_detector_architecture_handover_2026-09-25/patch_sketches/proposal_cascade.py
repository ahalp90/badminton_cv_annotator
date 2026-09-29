"""Illustrative proposal-cascade seam; not a drop-in repository patch.

The important invariants are:
- all current geometry/player checks run;
- coarse rank is stable;
- exact-score subsets are restored to original proposal order;
- exact retain/ranking is unchanged;
- exact mode remains available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CascadeConfig:
    coarse_samples: int = 16
    proposal_k: int = 8192
    shadow_full: bool = False


@dataclass
class CascadeResult:
    exact_positions: np.ndarray
    exact_scores: np.ndarray
    retained_positions: np.ndarray
    coarse_scores: np.ndarray
    full_shortlist_match: bool | None


def stable_topk_in_proposal_order(values: np.ndarray, k: int) -> np.ndarray:
    """Choose by stable coarse rank, then restore baseline proposal order."""
    if k >= len(values):
        return np.arange(len(values), dtype=np.int64)
    ranked = np.argsort(-values, kind="stable")[:k]
    return np.sort(ranked)


def score_pair_cascade(
    homographies: np.ndarray,
    coarse_score: Callable[[np.ndarray, int], np.ndarray],
    exact_score: Callable[[np.ndarray], np.ndarray],
    exact_retain_positions: Callable[[np.ndarray, np.ndarray], np.ndarray],
    config: CascadeConfig,
) -> CascadeResult:
    coarse = coarse_score(homographies, config.coarse_samples)
    exact_positions = stable_topk_in_proposal_order(coarse, config.proposal_k)
    exact_values = exact_score(homographies[exact_positions])
    retained_subset = exact_retain_positions(exact_positions, exact_values)

    match: bool | None = None
    if config.shadow_full:
        all_positions = np.arange(len(homographies), dtype=np.int64)
        full_values = exact_score(homographies)
        retained_full = exact_retain_positions(all_positions, full_values)
        match = np.array_equal(retained_subset, retained_full)

    return CascadeResult(
        exact_positions=exact_positions,
        exact_scores=exact_values,
        retained_positions=retained_subset,
        coarse_scores=coarse,
        full_shortlist_match=match,
    )
