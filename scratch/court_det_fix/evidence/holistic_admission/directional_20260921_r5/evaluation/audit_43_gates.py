"""Audit saved 43-arm C ranks under two historical filters.

This is a read-only diagnostic. It reuses the saved ``C.provisional_rank`` order
and candidate records. It does not fit candidates, rescore evidence, or create
new thresholds.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

CASE_RECORD_SCHEMA = "w5-case-evidence/2"
ARM_NAME = "w5_directional_20260921_r5_43"


def read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def historical_fullcourt(candidate: dict[str, Any], camera_limit: float) -> bool:
    gates = candidate.get("gates", {})
    fractions = gates.get("player_fractions", [None, None])
    one_player = fractions[0] if len(fractions) > 0 else None
    two_player = fractions[1] if len(fractions) > 1 else None
    camera_error = gates.get("camera_error")
    camera_valid = camera_error is not None and camera_error <= camera_limit
    return bool(
        gates.get("geometry_valid", False)
        and one_player == 1.0
        and two_player is not None
        and two_player >= 0.5
        and camera_valid
    )


def source_memberships(candidate: dict[str, Any]) -> frozenset[str]:
    memberships = candidate.get("source_memberships")
    if not isinstance(memberships, list) or not all(isinstance(source, str) for source in memberships):
        raise ValueError(f"{candidate.get('origin_key')}: missing source_memberships")
    return frozenset(memberships)


def has_source_membership(candidate: dict[str, Any], source: str) -> bool:
    return source in source_memberships(candidate)


def has_legacy_membership(candidate: dict[str, Any]) -> bool:
    memberships = source_memberships(candidate)
    return bool(memberships & {"G0", "G1"})


def candidate_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = [*record["parents"], *record["valid_children"]]
    mapped = {candidate["origin_key"]: candidate for candidate in candidates}
    if len(mapped) != len(candidates):
        raise ValueError(f"{record['case_id']}: duplicate candidate origin keys")
    return mapped


def rank_filter(
    case_id: str,
    provisional_rank: list[str],
    candidates: dict[str, dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    gallery_dir: Path,
) -> dict[str, Any]:
    eligible = []
    for origin_key in provisional_rank:
        candidate = candidates.get(origin_key)
        if candidate is None:
            raise ValueError(f"{case_id}: rank names missing candidate {origin_key}")
        if predicate(candidate):
            eligible.append(origin_key)

    winner = eligible[0] if eligible else None
    if winner is None:
        return {
            "eligible_count": 0,
            "winner": None,
            "original_rank": None,
            "gallery_full": None,
            "gallery_status": "empty",
        }

    gallery_full = gallery_dir / f"{safe_name(f'{case_id}__{winner}')}__full.png"
    return {
        "eligible_count": len(eligible),
        "winner": winner,
        "original_rank": provisional_rank.index(winner) + 1,
        "gallery_full": gallery_full.name,
        "gallery_status": "existing" if gallery_full.is_file() else "needs_render",
    }


def read_case_ids(per_view_path: Path) -> list[str]:
    with per_view_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != 27 or len(set(case_ids)) != 27:
        raise ValueError(f"{per_view_path}: expected 27 distinct cases")
    return case_ids


def audit(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    camera_limit = float(manifest["global_parameters"]["camera_error_limit_historical"])
    case_ids = read_case_ids(run_dir / "per_view.csv")
    gallery_dir = run_dir / "gallery"

    cases: list[dict[str, Any]] = []
    for case_id in case_ids:
        record = read_json_gz(run_dir / "case_records" / f"{case_id}.json.gz")
        if record.get("schema") != CASE_RECORD_SCHEMA:
            raise ValueError(f"{case_id}: unexpected record schema {record.get('schema')!r}")
        candidates = candidate_map(record)
        provisional_rank = record["rankings"]["C"]["provisional_rank"]
        historical_subset = record["rankings"]["C"]["historical_fullcourt_subset_rank"]

        fullcourt = rank_filter(
            case_id,
            provisional_rank,
            candidates,
            lambda candidate: historical_fullcourt(candidate, camera_limit),
            gallery_dir,
        )
        g0_plus_g1 = rank_filter(
            case_id,
            provisional_rank,
            candidates,
            has_legacy_membership,
            gallery_dir,
        )
        g0_only = rank_filter(
            case_id,
            provisional_rank,
            candidates,
            lambda candidate: has_source_membership(candidate, "G0"),
            gallery_dir,
        )
        g1_only = rank_filter(
            case_id,
            provisional_rank,
            candidates,
            lambda candidate: has_source_membership(candidate, "G1"),
            gallery_dir,
        )
        expected_fullcourt = [
            origin_key
            for origin_key in provisional_rank
            if historical_fullcourt(candidates[origin_key], camera_limit)
        ]
        line_template_count = sum(
            candidates[origin_key].get("source") == "line_template"
            for origin_key in provisional_rank
        )
        cases.append(
            {
                "case_id": case_id,
                "provisional_count": len(provisional_rank),
                "original_c": record["rankings"]["C"].get("selected_origin_key"),
                "line_template_excluded_count": line_template_count,
                "historical_fullcourt": fullcourt,
                "historical_fullcourt_matches_saved_subset": set(expected_fullcourt) == set(historical_subset),
                "g0_plus_g1": g0_plus_g1,
                "g0_only": g0_only,
                "g1_only": g1_only,
            }
        )

    def summary(field: str) -> dict[str, Any]:
        rows = [case[field] for case in cases]
        winners = [row["winner"] for row in rows if row["winner"] is not None]
        return {
            "empty_count": sum(row["winner"] is None for row in rows),
            "winner_source_counts": {
                source: sum(winner.startswith(f"{source}:") for winner in winners)
                for source in ("G0", "G1", "line_template")
            },
            "gallery_existing_count": sum(row["gallery_status"] == "existing" for row in rows),
            "gallery_needs_render_count": sum(row["gallery_status"] == "needs_render" for row in rows),
            "gallery_missing_cases": [
                case["case_id"] for case in cases if case[field]["gallery_status"] == "needs_render"
            ],
        }

    return {
        "schema": "w5-43-gate-audit/1",
        "run": ARM_NAME,
        "camera_error_limit_historical": camera_limit,
        "predicate": "geometry_valid and player_fractions[0] == 1.0 and player_fractions[1] >= 0.5 and camera_error <= limit",
        "candidate_map_source": "case_records parents + valid_children",
        "rank_source": "case_records rankings.C.provisional_rank",
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "historical_fullcourt_subset_matches_saved_count": sum(
                case["historical_fullcourt_matches_saved_subset"] for case in cases
            ),
            "historical_fullcourt": summary("historical_fullcourt"),
            "g0_plus_g1": summary("g0_plus_g1"),
            "g0_only": summary("g0_only"),
            "g1_only": summary("g1_only"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.repo_root / "scratch/court_det_fix/evidence/holistic_admission/directional_20260921_r5" / ARM_NAME
    result = audit(run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
