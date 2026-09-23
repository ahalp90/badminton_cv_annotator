"""Check archived automatic candidate retention under the saved 12-family screen."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "src"))
from courtkeynet.court_corners import CORNER_COURT_M

PREFIX = Path("scratch/court_det_fix")
REVIEW = PREFIX / "evidence/webui_followup3_20260922/review_20260923"
RUN = PREFIX / "direction_agreement/runs/direction_agreement_20260915_144900"
CACHE = PREFIX / "evidence/independent_proposals/development/player_guided/20260914/automatic_axes/collected/all_camera"
BEST_NEIGHBOURHOOD_PX = 1.0
EXACT_MATCH_PX = 1e-3
PROJECTION_TOLERANCE_PX = 1e-7


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def source(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "md5": hashlib.md5(path.read_bytes()).hexdigest()}


def project(homographies: np.ndarray) -> np.ndarray:
    # Preserve the producer's float32 court coordinates before projecting in float64.
    # Replacing them with exact decimal values amplifies differences near the horizon.
    court = np.column_stack((CORNER_COURT_M, np.ones(4)))
    projected = np.einsum("nj,kij->kni", court, homographies)
    require(np.isfinite(projected).all() and (np.abs(projected[:, :, 2]) > 1e-12).all(),
            "Invalid cached candidate homography projection")
    return projected[:, :, :2] / projected[:, :, 2, None]


def identity(case_id: str, cache: dict, e2: dict, e3: dict, baseline: dict, replay: dict) -> None:
    require(all(item["case_id"] == case_id for item in (cache, e2, e3, baseline, replay)),
            f"{case_id}: case identity mismatch")
    require(cache["automatic_directions"] is True and cache["label_guided_generation"] is False,
            f"{case_id}: unexpected automatic cache semantics")
    require(e2["controls_loaded"] is False and e3["automatic_detection"] is False,
            f"{case_id}: unexpected direction experiment semantics")
    require(e3["label_guided_evaluation"] is True and replay["input_identity_and_crosslink_checks_passed"],
            f"{case_id}: unverified screen inputs")
    require(cache["working_size"] == e2["working_size"] == e3["working_size"] ==
            baseline["working_size"] == replay["working_size"], f"{case_id}: working size mismatch")
    b = e2["arms"]["B"]
    cached = cache["estimator"]
    saved = baseline["estimator"]
    require(b["direction_count"] == 16 and b["matcher_eligible"], f"{case_id}: wrong B population")
    for left, right, description in (
        (cached["points_working"], b["points_working"], "direction points"),
        (cached["retained_support_masks"], b["support_masks"], "support masks"),
        (cached["retained_candidate_ids"], b["representative_candidate_ids"], "original group IDs"),
        (cached["direction_lines"], saved["direction_lines"], "direction lines"),
        (cached["normalised_to_working"], saved["normalised_to_working"], "normalisation"),
        (cached["points_working"], saved["points_working"], "baseline direction points"),
        (cached["retained_support_masks"], saved["retained_support_masks"], "baseline support masks"),
        (cached["retained_candidate_ids"], saved["retained_candidate_ids"], "baseline group IDs"),
        (b["points_working"], e3["sets"]["B"]["points_working"], "E3 direction points"),
    ):
        require(left == right, f"{case_id}: {description} mismatch")
    require(cache["estimator_settings"] == baseline["settings"], f"{case_id}: estimator settings mismatch")
    require(len(cached["direction_lines"]) == len(cached["retained_support_masks"][0]),
            f"{case_id}: support/line shape mismatch")
    require(replay["rank_order"] == sorted(range(16), key=lambda index: (
        replay["groups"][index]["algebraic_rms"], -replay["groups"][index]["line_count"], index)),
        f"{case_id}: replay order mismatch")
    expected = [[first, second] for first in range(16) for second in range(16) if first != second]
    require([pair["pair_id"] for pair in cache["pairs"]] == list(range(240)),
            f"{case_id}: pair ID sequence mismatch")
    require([pair["pencils"] for pair in cache["pairs"]] == expected,
            f"{case_id}: pair order/group identity mismatch")
    require([record["groups"] for record in e3["sets"]["B"]["fits"]["records"]] == expected,
            f"{case_id}: E3 pair order mismatch")
    require([pair["pair_id"] for pair in e3["sets"]["B"]["fits"]["records"]] == list(range(240)),
            f"{case_id}: E3 pair ID mismatch")


def brief(entry: dict, pencils: list[int]) -> dict:
    return {"candidate_id": entry["candidate_id"], "pair_id": entry["pair_id"],
            "pencils_original_group_ids": pencils, "corners_native_px": entry["corners_px"]}


def analyse(case: dict) -> dict:
    case_id = case["case_id"]
    paths = {"cache": ROOT / CACHE / f"{case_id}.json.gz", "e2": ROOT / RUN / "e2" / f"{case_id}.json.gz",
             "e3": ROOT / RUN / "e3" / f"{case_id}.json.gz",
             "baseline": ROOT / PREFIX / "frozen_views/baseline_directions" / f"{case_id}.json.gz"}
    data = {name: read(path) for name, path in paths.items()}
    cache, e2, e3, baseline = (data[name] for name in ("cache", "e2", "e3", "baseline"))
    identity(case_id, cache, e2, e3, baseline, case)
    require(case["control"] == e3["control"], f"{case_id}: control mismatch")
    require(source(paths["e2"])["md5"] == e3["e2_record_md5"], f"{case_id}: E2 digest mismatch")
    require(source(paths["baseline"])["md5"] == e3["saved_estimator_md5"],
            f"{case_id}: baseline digest mismatch")
    selection = next(item for item in case["budgets"] if item["budget"] == 12)
    require(selection["selected_groups_rank_order"] == case["rank_order"][:12],
            f"{case_id}: changed admission")
    selected = set(selection["selected_groups_rank_order"])
    pairs = cache["pairs"]
    retained_pairs = {pair["pair_id"] for pair in pairs if set(pair["pencils"]) <= selected}
    require(len(retained_pairs) == 132, f"{case_id}: wrong retained pair count")
    entries = cache["entries"]
    require(len({entry["candidate_id"] for entry in entries}) == len(entries),
            f"{case_id}: duplicate candidate ID")
    require(all(entry["pair_id"] in range(240) and entry["candidate_id"].startswith(
        f'{entry["pair_id"]}:') for entry in entries), f"{case_id}: candidate/pair mismatch")
    retained = [entry for entry in entries if entry["pair_id"] in retained_pairs]
    dropped = [entry for entry in entries if entry["pair_id"] not in retained_pairs]
    require(len(retained) + len(dropped) == len(entries), f"{case_id}: candidate partition mismatch")
    require(len(entries) == cache["camera_eligible_before_global"], f"{case_id}: cache pool mismatch")
    native = np.asarray(e3["control"]["native_size"], dtype=float)
    working = np.asarray(e3["control"]["working_size"], dtype=float)
    scale = native / working
    require(np.allclose(np.asarray(e3["control"]["corners_native_px"]) / scale,
                        e3["control"]["corners_working_px"], rtol=0, atol=1e-8),
            f"{case_id}: control scale mismatch")
    projected = project(np.asarray([entry["homography_working"] for entry in entries], dtype=float))
    cached_corners = np.asarray([entry["corners_px"] for entry in entries], dtype=float) / scale
    projection_error = np.linalg.norm(projected - cached_corners, axis=2).max(axis=1)
    require(np.allclose(projected, cached_corners, rtol=0,
                        atol=PROJECTION_TOLERANCE_PX),
            f"{case_id}: homography/cached-corner mismatch: {projection_error.max()}")
    control = np.asarray(e3["control"]["corners_working_px"], dtype=float)
    direct = np.linalg.norm(projected - control, axis=2).max(axis=1)
    rotated = np.linalg.norm(projected - np.roll(control, 2, axis=0), axis=2).max(axis=1)
    errors = np.minimum(direct, rotated)
    retained_mask = np.asarray([entry["pair_id"] in retained_pairs for entry in entries])
    full_best = int(np.argmin(errors))
    retained_best = int(np.argmin(np.where(retained_mask, errors, np.inf)))
    neighbourhood = errors <= errors[full_best] + BEST_NEIGHBOURHOOD_PX
    exact = errors <= EXACT_MATCH_PX
    require(np.isfinite(errors).all(), f"{case_id}: non-finite reference agreement")
    entry_by_id = {entry["candidate_id"]: entry for entry in entries}
    winners = {}
    for role in ("line", "paint"):
        candidate_id = cache[f"{role}_winner_id"]
        require(candidate_id in entry_by_id, f"{case_id}: missing {role} winner")
        entry = entry_by_id[candidate_id]
        winners[role] = {**brief(entry, pairs[entry["pair_id"]]["pencils"]),
                         "retained_12": entry["pair_id"] in retained_pairs}
    def diagnostics(mask: np.ndarray) -> dict:
        count = int(mask.sum())
        return {"count": count, "retained": int((mask & retained_mask).sum()),
                "dropped": int((mask & ~retained_mask).sum()),
                "matched_candidates": [{**brief(entry, pairs[entry["pair_id"]]["pencils"]),
                                        "retained_12": bool(keep)}
                                       for entry, matched, keep in zip(entries, mask, retained_mask, strict=True)
                                       if matched],
                "dropped_candidates": [brief(entry, pairs[entry["pair_id"]]["pencils"])
                                       for entry, matched, keep in zip(entries, mask, retained_mask, strict=True)
                                       if matched and not keep]}
    accounting = {}
    for label, chosen in (("full_16", pairs), ("retained_12", [pairs[index] for index in sorted(retained_pairs)]),
                          ("dropped", [pair for pair in pairs if pair["pair_id"] not in retained_pairs])):
        accounting[label] = {"pairs": len(chosen), "pair_status": dict(Counter(pair["status"] for pair in chosen)),
                             "cached_pair_elapsed_s_sum": sum(pair.get("elapsed_s", 0.0) for pair in chosen)}
    return {"case_id": case_id, "population": "visually_approved" if e3["control"]["visually_approved"] else "manual",
            "control_source": e3["control"]["control_source"], "sources": {name: source(path) for name, path in paths.items()},
            "input_identity_checks_passed": True, "selected_original_group_ids_rank_order": case["rank_order"][:12],
            "chosen_group_mask_original_order": [index in selected for index in range(16)],
            "full_16_original_group_ids": list(range(16)), "retained_pair_ids": sorted(retained_pairs),
            "candidate_counts": {"full_16": len(entries), "retained_12": len(retained), "dropped": len(dropped)},
            "cached_pair_accounting_not_measured_wall_clock_speedup": accounting, "winners": winners,
            "posthoc_reference_agreement": {
                "metric": "minimum over direct and 180-degree corner orders of maximum corner Euclidean distance, working px",
                "control_used_for_admission": False, "full_best_error_px": float(errors[full_best]),
                "retained_best_error_px": float(errors[retained_best]),
                "full_best": brief(entries[full_best], pairs[entries[full_best]["pair_id"]]["pencils"]),
                "retained_best": brief(entries[retained_best], pairs[entries[retained_best]["pair_id"]]["pencils"]),
                "exact_best_candidate_preserved": bool(retained_mask[full_best]),
                "best_neighbourhood_within_1_working_px": diagnostics(neighbourhood),
                "visually_approved_exact_geometry_matches_at_1e_minus_3_working_px": (
                    diagnostics(exact) if e3["control"]["visually_approved"] else None),
                "visually_approved_exact_match_status": (
                    "available" if exact.any() else "unavailable/no exact match")
                    if e3["control"]["visually_approved"] else "manual reference",
                "maximum_projected_vs_cached_corner_error_working_px": float(projection_error.max()),
            }}


def main() -> None:
    replay_path = ROOT / REVIEW / "replay/results.json.gz"
    replay = read(replay_path)
    require(len(replay["cases"]) == 9, "Expected nine replay cases")
    results = [analyse(case) for case in replay["cases"]]
    require(sum(case["population"] == "visually_approved" for case in results) == 6,
            "Expected six visually approved controls")
    require(sum(case["population"] == "manual" for case in results) == 3,
            "Expected three manual references")
    output = {"schema": "saved-automatic-candidate-retention/1", "replay_source": source(replay_path),
              "direction_replacement": False, "normalisation_change": False, "refit": False,
              "matcher_run": False, "threshold_or_cap_change": False,
              "projection_crosscheck": {"absolute_tolerance_working_px": PROJECTION_TOLERANCE_PX,
                                        "relative_tolerance": 0,
                                        "reason": "Reuse the producer's canonical float32 court coordinates"},
              "cached_elapsed_not_measured_wall_clock_speedup": True, "cases": results}
    destination = Path(__file__).with_name("results.json.gz")
    destination.write_bytes(gzip.compress((json.dumps(output, indent=2, allow_nan=False) + "\n").encode(), mtime=0))
    print(json.dumps({"cases": len(results), "candidate_counts": {
        label: sum(case["candidate_counts"][label] for case in results)
        for label in ("full_16", "retained_12", "dropped")},
        "winners_retained": {role: sum(case["winners"][role]["retained_12"] for case in results)
                             for role in ("line", "paint")},
        "exact_best_preserved": sum(case["posthoc_reference_agreement"]["exact_best_candidate_preserved"]
                                    for case in results)}, indent=2))


if __name__ == "__main__":
    main()
