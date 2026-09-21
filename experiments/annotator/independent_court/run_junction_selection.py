"""Compare two prespecified junction rankings on unchanged stripe candidates."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .case_provenance import PACK_MD5_BY_NAME, SIDECAR_MD5, SIDECAR_SCHEMA
from .run_assignment import ACCURATE_PX

SCHEMES = ("stripe_exclusive", "contradictions_first", "complete_agreements_first")


def validate_diagnostic_provenance(diagnostic: dict) -> dict[str, str]:
    """Reject diagnostics that are not bound to the current pinned provenance contract."""
    if diagnostic.get("schema") != "frozen-junction-diagnostic/2":
        raise ValueError("Junction diagnostic is old or unsupported; rerun run_junctions.py")
    provenance = diagnostic.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("Junction diagnostic provenance must be an object")
    pack_filename = provenance.get("pack_filename")
    if not isinstance(pack_filename, str):
        raise TypeError("Junction diagnostic pack_filename must be a string")
    expected_pack_md5 = PACK_MD5_BY_NAME.get(pack_filename)
    expected = {
        "pack_filename": pack_filename,
        "pack_md5": expected_pack_md5,
        "sidecar_schema": SIDECAR_SCHEMA,
        "sidecar_md5": SIDECAR_MD5,
        "stripe_artefact_md5": provenance.get("stripe_artefact_md5"),
    }
    stripe_artefact_md5 = provenance.get("stripe_artefact_md5")
    if (
        expected_pack_md5 is None
        or not isinstance(stripe_artefact_md5, str)
        or len(stripe_artefact_md5) != 32
        or provenance != expected
    ):
        raise ValueError("Junction diagnostic provenance does not match the current pinned pack/sidecar contract")
    return provenance


def preflight_inputs(
    stripes: dict, junctions: dict, stripe_artefact_md5: str,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Validate provenance and case identity before ranking any diagnostic."""
    binding = validate_diagnostic_provenance(junctions)
    stripe_binding = {key: value for key, value in binding.items() if key != "stripe_artefact_md5"}
    if stripes.get("schema") != "frozen-stripe-observations/2":
        raise ValueError("Stripe results are old or unsupported; rerun run_stripes.py with --provenance-pack")
    if stripes.get("input_provenance") != stripe_binding:
        raise ValueError("Stripe result provenance does not match the junction diagnostic")
    if stripe_artefact_md5 != binding["stripe_artefact_md5"]:
        raise ValueError("Stripe artefact bytes do not match those measured by the junction diagnostic")
    stripe_ids = [record["id"] for record in stripes["records"]]
    junction_ids = [record["id"] for record in junctions["records"]]
    if len(stripe_ids) != len(set(stripe_ids)):
        raise ValueError("Stripe results contain duplicate case IDs")
    if len(junction_ids) != len(set(junction_ids)):
        raise ValueError("Junction diagnostic contains duplicate case IDs")
    if set(stripe_ids) != set(junction_ids):
        raise ValueError(
            f"Stripe/junction case-set mismatch (stripes={sorted(stripe_ids)!r}, junctions={sorted(junction_ids)!r})"
        )
    return {record["id"]: record for record in junctions["records"]}, binding


def bytes_md5(value: bytes) -> str:
    """Return the MD5 used to bind the exact artefact bytes consumed."""
    return hashlib.md5(value, usedforsecurity=False).hexdigest()


def rank(entries: list[dict]) -> dict[str, list[str]]:
    """Keep missing observations neutral and retain every eligible alternative."""
    stripe = sorted(entries, key=lambda entry: (-entry["stripe_score"], entry["id"]))
    contradictions = sorted(entries, key=lambda entry: (
        entry["disagreements"], -entry["stripe_score"], entry["id"],
    ))
    agreements = sorted(entries, key=lambda entry: (
        -entry["complete_agreements"], entry["disagreements"], -entry["stripe_score"], entry["id"],
    ))
    return dict(zip(SCHEMES, ([entry["id"] for entry in order] for order in (stripe, contradictions, agreements))))


def run_case(frozen: dict, diagnostic: dict) -> dict:
    """Rank from observations and scores before attaching saved reference metrics."""
    measured = {entry["id"]: entry for entry in diagnostic["entries"]}
    entries = []
    for source in frozen["entries"]:
        if not source["eligible"]:
            continue
        junctions = measured[source["id"]]
        complete = sum(site["usable"] and len(site["agreements"]) == 2 for site in junctions["sites"])
        entries.append({"id": source["id"], "stripe_score": source["scores"]["stripe_exclusive"],
                        "disagreements": junctions["disagreements"], "usable_sites": junctions["usable_sites"],
                        "complete_agreements": complete})
    orders = rank(entries)
    metrics = {entry["id"]: entry["metrics"] for entry in frozen["entries"]}
    for entry in entries:
        entry["metrics"] = metrics[entry["id"]]
    return {"id": frozen["id"], "entries": entries, "orders": orders}


def summarise(records: list[dict]) -> dict:
    accurate, unobserved = dict.fromkeys(SCHEMES, 0), dict.fromkeys(SCHEMES, 0)
    cases = []
    for record in records:
        entries = {entry["id"]: entry for entry in record["entries"]}
        picks = {}
        for scheme, order in record["orders"].items():
            winner = entries[order[0]] if order else None
            picks[scheme] = winner
            if winner is not None:
                accurate[scheme] += winner["metrics"]["corner_max_error_px"] <= ACCURATE_PX
                unobserved[scheme] += winner["usable_sites"] == 0
        cases.append({"id": record["id"], "picks": picks})
    return {"frames": len(records), "accurate_picks": accurate,
            "winners_without_usable_sites": unobserved, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stripes", required=True, type=Path)
    parser.add_argument("--junctions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    stripe_bytes = args.stripes.read_bytes()
    stripes = json.loads(gzip.decompress(stripe_bytes))
    junctions = json.loads(gzip.decompress(args.junctions.read_bytes()))
    measured, binding = preflight_inputs(stripes, junctions, bytes_md5(stripe_bytes))
    records = [run_case(record, measured[record["id"]]) for record in stripes["records"]]
    summary = summarise(records)
    output = {"schema": "frozen-junction-selection/2", "development_data": True,
              "acceptance_evaluated": False, "provenance": binding,
              "summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(output, allow_nan=False).encode(), mtime=0))
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
