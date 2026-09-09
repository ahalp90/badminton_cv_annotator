"""Check separate refinement pools, renewed eligibility and empty-frame accounting."""

from experiments.annotator.independent_court.run_refit_selection import (
    eligible,
    rank_pools,
    summarise,
)


def candidate(identifier: str, model: str, score: float, allowed: bool = True) -> dict:
    return {"id": identifier, "model": model, "eligible": allowed, "stripe_score": score,
            "complete_agreements": 1, "disagreements": 0, "metrics": {"corner_max_error_px": 5}}


def test_each_refinement_pool_retains_starts_and_excludes_the_other_model() -> None:
    entries = [candidate("start", "start", 0.8), candidate("nominal", "nominal_centre", 0.9),
               candidate("fixed", "fixed_position", 1.0), candidate("failed_gate", "fixed_position", 1.1, False)]
    orders = rank_pools(entries)
    assert orders["starts"]["stripe_exclusive"] == ["start"]
    assert orders["nominal_centre"]["stripe_exclusive"] == ["nominal", "start"]
    assert orders["fixed_position"]["stripe_exclusive"] == ["fixed", "start"]
    assert rank_pools(entries[::-1]) == orders


def test_renewed_geometry_and_player_gates_both_apply() -> None:
    assert not eligible({"eligible": False, "reason": "geometry"})
    assert not eligible({"eligible": True, "scheme_eligible": {"original": False}})
    assert eligible({"eligible": True, "scheme_eligible": {"original": True}})


def test_empty_frame_counts_without_an_accurate_pick() -> None:
    entries = [candidate("start", "start", 0.8)]
    records = [{"id": "one", "entries": entries, "orders": rank_pools(entries)},
               {"id": "empty", "entries": [], "orders": rank_pools([])}]
    summary = summarise(records)
    assert summary["frames"] == 2
    assert all(value == 1 for value in summary["useful_eligible_pools"].values())
    assert all(count == 1 for pool in summary["accurate_picks"].values() for count in pool.values())
