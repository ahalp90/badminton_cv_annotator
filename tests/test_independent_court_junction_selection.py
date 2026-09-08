"""Check missing evidence, stable ties and evaluation denominators."""

from experiments.annotator.independent_court.run_junction_selection import (
    SCHEMES,
    rank,
    run_case,
    summarise,
)


def entry(identifier: str, score: float, usable: int = 0, complete: int = 0, disagreements: int = 0) -> dict:
    return {"id": identifier, "stripe_score": score, "usable_sites": usable,
            "complete_agreements": complete, "disagreements": disagreements}


def test_missing_evidence_is_neutral_but_never_a_complete_agreement() -> None:
    entries = [entry("unknown", 0.9), entry("contradiction", 1.0, 1, 0, 1), entry("supported", 0.8, 1, 1)]
    orders = rank(entries)
    assert orders["stripe_exclusive"][0] == "contradiction"
    assert orders["contradictions_first"][0] == "unknown"
    assert orders["complete_agreements_first"][0] == "supported"
    assert all(set(order) == {item["id"] for item in entries} for order in orders.values())


def test_ties_use_identity_independently_of_input_order() -> None:
    entries = [entry("b", 0.8, 1, 1), entry("a", 0.8, 1, 1)]
    assert rank(entries) == rank(entries[::-1]) == {scheme: ["a", "b"] for scheme in SCHEMES}


def test_metrics_do_not_select_and_empty_pool_stays_in_denominator() -> None:
    sources = [{"id": "poor", "eligible": True, "scores": {"stripe_exclusive": 1.0},
                "metrics": {"corner_max_error_px": 600}},
               {"id": "accurate", "eligible": True, "scores": {"stripe_exclusive": 0.8},
                "metrics": {"corner_max_error_px": 5}}]
    diagnostic = {"entries": [{"id": item["id"], "disagreements": 0, "usable_sites": 0, "sites": []}
                              for item in sources]}
    selected = run_case({"id": "case", "entries": sources}, diagnostic)
    assert all(order[0] == "poor" for order in selected["orders"].values())
    empty = run_case({"id": "empty", "entries": []}, {"entries": []})
    summary = summarise([selected, empty])
    assert summary["frames"] == 2
    assert summary["accurate_picks"] == dict.fromkeys(SCHEMES, 0)
    assert summary["winners_without_usable_sites"] == dict.fromkeys(SCHEMES, 1)
