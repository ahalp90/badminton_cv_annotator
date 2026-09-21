"""Check missing evidence, stable ties and evaluation denominators."""

import gzip
import hashlib
import json
import sys

import pytest

from experiments.annotator.independent_court.case_provenance import (
    PACK_MD5_BY_NAME,
    SIDECAR_MD5,
    SIDECAR_SCHEMA,
)
from experiments.annotator.independent_court.run_junction_selection import (
    SCHEMES,
    main,
    preflight_inputs,
    rank,
    run_case,
    summarise,
)


def pack_binding() -> dict[str, str]:
    pack_filename = next(iter(PACK_MD5_BY_NAME))
    return {
        "pack_filename": pack_filename,
        "pack_md5": PACK_MD5_BY_NAME[pack_filename],
        "sidecar_schema": SIDECAR_SCHEMA,
        "sidecar_md5": SIDECAR_MD5,
    }


def bound_stripes(records: list[dict]) -> dict:
    return {"schema": "frozen-stripe-observations/2", "input_provenance": pack_binding(), "records": records}


def bound_diagnostic(records: list[dict], stripe_md5: str = "a" * 32) -> dict:
    return {
        "schema": "frozen-junction-diagnostic/2",
        "provenance": {**pack_binding(), "stripe_artefact_md5": stripe_md5},
        "records": records,
    }


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


def test_old_unbound_junction_diagnostic_is_rejected_before_ranking() -> None:
    stripes = bound_stripes([{"id": "case", "entries": []}])
    old = {"schema": "frozen-junction-diagnostic/1", "records": [{"id": "case", "entries": []}]}
    with pytest.raises(ValueError, match="old or unsupported"):
        preflight_inputs(stripes, old, "a" * 32)


def test_bound_junction_diagnostic_requires_current_pack_and_sidecar() -> None:
    stripes = bound_stripes([{"id": "case", "entries": []}])
    diagnostic = bound_diagnostic([{"id": "case", "entries": []}])
    measured, binding = preflight_inputs(stripes, diagnostic, "a" * 32)
    assert list(measured) == ["case"]
    assert binding == diagnostic["provenance"]

    diagnostic["provenance"]["sidecar_md5"] = "0" * 32
    with pytest.raises(ValueError, match="current pinned pack/sidecar contract"):
        preflight_inputs(stripes, diagnostic, "a" * 32)


def test_substituted_stripe_artefact_is_rejected() -> None:
    stripes = bound_stripes([{"id": "case", "entries": []}])
    diagnostic = bound_diagnostic([{"id": "case", "entries": []}], "1" * 32)
    with pytest.raises(ValueError, match="bytes do not match"):
        preflight_inputs(stripes, diagnostic, "2" * 32)


def test_selection_output_propagates_validated_binding(monkeypatch, tmp_path) -> None:
    stripes = bound_stripes([{"id": "case", "entries": []}])
    stripe_path = tmp_path / "stripes.json.gz"
    stripe_path.write_bytes(gzip.compress(json.dumps(stripes).encode(), mtime=0))
    stripe_md5 = hashlib.md5(stripe_path.read_bytes(), usedforsecurity=False).hexdigest()
    diagnostic = bound_diagnostic([{"id": "case", "entries": []}], stripe_md5)
    junction_path = tmp_path / "junctions.json.gz"
    junction_path.write_bytes(gzip.compress(json.dumps(diagnostic).encode(), mtime=0))
    output_path = tmp_path / "selection.json.gz"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_junction_selection.py", "--stripes", str(stripe_path), "--junctions", str(junction_path),
         "--output", str(output_path)],
    )
    main()
    output = json.loads(gzip.decompress(output_path.read_bytes()))
    assert output["schema"] == "frozen-junction-selection/2"
    assert output["provenance"] == diagnostic["provenance"]
