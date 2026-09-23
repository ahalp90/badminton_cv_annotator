from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
EVIDENCE_RELATIVE = Path("scratch/court_det_fix/evidence")
APPROVED_RELATIVE = EVIDENCE_RELATIVE / "independent_proposals/history_audit_20260922/approved_winner_retention.json.gz"
REPLAY_RELATIVE = EVIDENCE_RELATIVE / "webui_followup3_20260922/review_20260923/replay/results.json.gz"
DEVELOPMENT_RELATIVE = EVIDENCE_RELATIVE / "independent_proposals/development/player_guided/20260914"
CACHE_ROOT_RELATIVE = DEVELOPMENT_RELATIVE / "automatic_axes/collected/all_camera"
OUTPUT_PATH = Path(__file__).with_name("approved_witnesses.json.gz")
CORNER_TOLERANCE_PX = 1e-8
RETAINED_RANK_LIMIT = 12


def read_json_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    approved = read_json_gzip(REPO_ROOT / APPROVED_RELATIVE)
    replay = read_json_gzip(REPO_ROOT / REPLAY_RELATIVE)
    replay_by_case = {case["case_id"]: case for case in replay["cases"]}
    assert len(replay_by_case) == len(replay["cases"]), "duplicate SVD replay case IDs"
    approved_cases = approved["cases"]
    expected_total = sum(len(case["witnesses"]) for case in approved_cases)
    assert expected_total == 8, f"expected 8 approved witnesses, found {expected_total}"

    output_cases = []
    count_retained = 0
    for approved_case in approved_cases:
        case_id = approved_case["case_id"]
        assert case_id in replay_by_case, f"missing SVD replay case: {case_id}"
        top_pencils = set(replay_by_case[case_id]["rank_order"][:RETAINED_RANK_LIMIT])
        cache_relative = CACHE_ROOT_RELATIVE / f"{case_id}.json.gz"
        cache = read_json_gzip(REPO_ROOT / cache_relative)
        entries = {entry["candidate_id"]: entry for entry in cache["entries"]}
        assert len(entries) == len(cache["entries"]), f"duplicate candidate IDs in {cache_relative}"
        pairs = {pair["pair_id"]: pair for pair in cache["pairs"]}
        assert len(pairs) == len(cache["pairs"]), f"duplicate pair IDs in {cache_relative}"
        output_witnesses = []
        for witness in approved_case["witnesses"]:
            candidate_id = witness["old_candidate_id"]
            assert candidate_id in entries, f"missing exact candidate {case_id}/{candidate_id}"
            entry = entries[candidate_id]
            assert entry["pair_id"] in pairs, f"missing pair {case_id}/{entry['pair_id']}"
            pencils = pairs[entry["pair_id"]]["pencils"]
            expected_corners = witness["old_corners_native_px"]
            cached_corners = entry["corners_px"]
            assert len(expected_corners) == len(cached_corners) == 4
            corner_difference = []
            for expected_corner, cached_corner in zip(expected_corners, cached_corners, strict=True):
                corner_difference.append(
                    [cached - expected for expected, cached in zip(expected_corner, cached_corner, strict=True)]
                )
            max_abs_difference = max(abs(delta) for corner in corner_difference for delta in corner)
            assert max_abs_difference <= CORNER_TOLERANCE_PX, (
                f"corner mismatch for {case_id}/{candidate_id}: {max_abs_difference} px"
            )
            retained = all(pencil_id in top_pencils for pencil_id in pencils)
            count_retained += retained
            output_witnesses.append(
                {
                    "candidate_id": candidate_id,
                    "pair_id": entry["pair_id"],
                    "pencils": pencils,
                    "retained": retained,
                    "exact_corner_difference_px": corner_difference,
                    "max_abs_corner_difference_px": max_abs_difference,
                }
            )
        output_cases.append(
            {"case_id": case_id, "automatic_cache_path": str(cache_relative), "witnesses": output_witnesses}
        )
    output = {
        "schema": "approved-witness-svd-retention/1",
        "source_paths": {
            "approved_winners": str(APPROVED_RELATIVE),
            "svd_replay_results": str(REPLAY_RELATIVE),
            "automatic_cache_root": str(CACHE_ROOT_RELATIVE),
        },
        "corner_check": "cached minus approved; absolute tolerance 1e-8 px, relative tolerance 0",
        "retention_check": "both original pencil IDs occur in rank_order[:12]",
        "total_witnesses": expected_total,
        "count_retained": count_retained,
        "cases": output_cases,
    }
    with OUTPUT_PATH.open("wb") as raw_file, gzip.GzipFile(
        filename="", fileobj=raw_file, mode="wb", mtime=0
    ) as compressed:
        compressed.write(json.dumps(output, indent=2, sort_keys=True, allow_nan=False).encode("utf-8"))
    print(f"Wrote {expected_total} witnesses; {count_retained} retained to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
