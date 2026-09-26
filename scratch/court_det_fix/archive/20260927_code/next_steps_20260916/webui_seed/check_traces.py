"""Recheck the two C2 traces from the saved generation records.

This is a small, single-process evidence check. It reruns the frozen axis
matcher for GX0 pair 143 under the baseline and paint-observation inputs, and
for Amateur-3 R pair 43's horizontal axis. It does not regenerate a matcher
population or combine the uncapped product.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError(f"could not find the repository root above {start}")


REPO = find_repo_root(Path(__file__).resolve())
LINE_IDENTITY = REPO / "scratch/court_det_fix/line_identity"
sys.path.insert(0, str(LINE_IDENTITY))

from shared import (
    BASELINE_GENERATION,
    DIRECTION_RUN,
    PACK_OF,
    PACKS,
    add_helper_paths,
    arm_points,
    control_corners,
    corner_errors,
    feet_working,
    load_source,
    read,
)

add_helper_paths()

from axis_replay import (
    AXIS_NAMES,
    MATCHER_SETTINGS,
    SYMMETRY,
    UNCAPPED,
    courts_from,
    ideal_parameters,
    pair_record,
    sweep,
)
from projective_seed import basis_for, match_axis
from run_population import prepare

from experiments.annotator.independent_court import assignment, detector

SAVED_LINE_IDENTITY = LINE_IDENTITY.parent / "worklog/remote_records_20260921/preserved_data/line_identity"
FILTER_INPUTS = SAVED_LINE_IDENTITY / "inputs"
FILTER_RUN = SAVED_LINE_IDENTITY / "runs/line_identity_20260915_222437"
PREGATE_RECORDS = (
    REPO
    / "scratch/court_det_fix/evidence/direction_search/diagnostics/pregate_loss/records/new"
)
IDENTITY_REPORTS = LINE_IDENTITY
RANKING_RECORDS = (
    REPO
    / "experiments/annotator/independent_court/recorded/player_guided/"
    "projective_patterns/evaluation/ranking_records.json.gz"
)
CONTROL_RECORD = (
    REPO
    / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/20260914/"
    "automatic_axes/gx0_control/collected/control.json.gz"
)
CONTROL_DIAGNOSIS = CONTROL_RECORD.parent / "diagnosis.json.gz"
IDENTITY_ROWS = (
    REPO
    / "scratch/court_det_fix/evidence/pixel_temporal/diagnostics/identity_diagnostics/rows.json"
)
CURRENT_AXIS_SOURCE = LINE_IDENTITY / "axis_replay.py"
CURRENT_FILTER_SOURCE = LINE_IDENTITY / "filter_replay.py"
FROZEN_PROJECTIVE_SOURCE = (
    REPO / "scratch/court_det_fix/frozen_helpers_20260914/axis_matching/projective_seed.py"
)
PRODUCER_SOURCES = {
    "run_automatic.py": (
        REPO
        / "scratch/court_det_fix/evidence/direction_search/diagnostics/pregate_loss/remote_src/run_automatic.py"
    ),
    "run_given.py": (
        REPO
        / "scratch/court_det_fix/evidence/direction_search/diagnostics/pregate_loss/remote_src/run_given.py"
    ),
}
AXIS_SOURCE_PATH = (
    REPO / "scratch/court_det_fix/line_identity/runs/axis_replay/table.csv"
)
FILTER_AXIS_SOURCE_PATH = (
    REPO / "scratch/court_det_fix/line_identity/runs/filter_replay/axis_table.csv"
)

def md5_path(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def as_list(array: np.ndarray) -> list:
    return np.asarray(array).tolist()


def control_context(case_id: str) -> dict:
    corners, control = control_corners(case_id)
    return {
        "case_id": case_id,
        "source": relative(DIRECTION_RUN / "e3" / f"{case_id}.json.gz"),
        "source_record": control["record"],
        "source_record_md5": control["record_md5"],
        "control_source": control["control_source"],
        "visually_approved": control["visually_approved"],
        "working_size": control["working_size"],
        "native_size": control["native_size"],
        "corners_working_px": as_list(corners),
        "corners_native_px": control["corners_native_px"],
    }


def source_for_case(case_id: str) -> Path:
    return PACKS[PACK_OF[case_id]]


def record_pool_witness(record: dict, control: np.ndarray, scale: np.ndarray) -> dict:
    matched_pairs = [pair for pair in record["pairs"] if pair["status"] == "matched"]
    populated_pairs = [pair for pair in matched_pairs if pair["shortlist"]]
    shortlist_count = sum(len(pair["shortlist"]) for pair in populated_pairs)
    assert shortlist_count == record["pooled_candidates"], (
        shortlist_count,
        record["pooled_candidates"],
    )
    best_error = np.inf
    best_pair: dict | None = None
    best_entry: dict | None = None
    for pair in populated_pairs:
        corners = np.asarray(
            [entry["corners_px"] for entry in pair["shortlist"]], dtype=float
        ) / scale
        errors = corner_errors(corners, control)
        index = int(np.argmin(errors))
        if float(errors[index]) < best_error:
            best_error = float(errors[index])
            best_pair = pair
            best_entry = pair["shortlist"][index]
    assert best_pair is not None and best_entry is not None
    corners_native = np.asarray(best_entry["corners_px"], dtype=float)
    return {
        "population": "all matched per-pair shortlists before the global cap",
        "matched_pairs": len(matched_pairs),
        "pairs_with_shortlist": len(populated_pairs),
        "candidate_count": shortlist_count,
        "pair_id": int(best_pair["pair_id"]),
        "candidate_id": best_entry["candidate_id"],
        "axis_ids": best_entry.get("axis_ids"),
        "rotated_180": best_entry.get("rotated_180"),
        "axis_score": best_entry.get("axis_score"),
        "shortlist_score": best_entry.get("shortlist_score"),
        "control_error_px": best_error,
        "corners_working_px": as_list(corners_native / scale),
        "corners_native_px": as_list(corners_native),
    }


def nearest_product_witness(
    basis: np.ndarray,
    horizontal,
    vertical,
    control: np.ndarray,
    pair: dict,
) -> dict:
    horizontal_ids = horizontal.distinct[: MATCHER_SETTINGS.keep_axes]
    vertical_ids = vertical.distinct[: MATCHER_SETTINGS.keep_axes]
    best_error = np.inf
    best_axis_ids: tuple[int, int] | None = None
    best_corners: np.ndarray | None = None
    block = max(1, 4_000_000 // max(1, len(vertical_ids)))
    for start in range(0, len(horizontal_ids), block):
        stop = start + block
        block_h_ids = horizontal_ids[start:stop]
        block_v_ids = np.tile(vertical_ids, len(block_h_ids))
        block_h = horizontal.parameters[block_h_ids]
        block_h = np.repeat(block_h, len(vertical_ids), axis=0)
        block_v = vertical.parameters[block_v_ids]
        corners = courts_from(basis, block_h, block_v)
        errors = corner_errors(corners, control)
        local = int(np.argmin(errors))
        if float(errors[local]) < best_error:
            best_error = float(errors[local])
            horizontal_index = local // len(vertical_ids)
            vertical_index = local % len(vertical_ids)
            best_axis_ids = (
                int(block_h_ids[horizontal_index]),
                int(vertical_ids[vertical_index]),
            )
            best_corners = corners[local]
    assert best_axis_ids is not None and best_corners is not None
    saved = next(
        (
            entry
            for entry in pair["shortlist"]
            if entry.get("axis_ids") == list(best_axis_ids)
        ),
        None,
    )
    witness = {
        "population": "512 kept horizontal x 512 kept vertical products for this pair",
        "pair_id": int(pair["pair_id"]),
        "candidate_id": None if saved is None else saved["candidate_id"],
        "axis_ids": list(best_axis_ids),
        "axis_identity_source": "replayed distinct assignments, retained in score order",
        "control_error_px": best_error,
        "corners_working_px": as_list(best_corners),
        "horizontal_score": float(horizontal.scores[best_axis_ids[0]]),
        "vertical_score": float(vertical.scores[best_axis_ids[1]]),
        "saved_pair_shortlist_entry": saved is not None,
    }
    if saved is not None:
        witness["saved_candidate"] = {
            "candidate_id": saved["candidate_id"],
            "axis_score": saved.get("axis_score"),
            "shortlist_score": saved.get("shortlist_score"),
            "rotated_180": saved.get("rotated_180"),
        }
    return witness


def replay_pair_axes(
    case_id: str,
    fit_arm: str,
    pair_id: int,
    record_path: Path,
    input_path: Path,
    axes: tuple[int, ...],
    require_exact_retained_order: bool = False,
) -> dict:
    record = read(record_path)
    source = load_source(case_id) if input_path == source_for_case(case_id) else read(input_path)
    original_source = load_source(case_id)
    segments, _, size = prepare(source)
    observations = assignment.prepare_observations(segments, size)
    feet = feet_working(original_source, size)
    control, _ = control_corners(case_id)
    points = arm_points(case_id, fit_arm)
    pair = pair_record(record, pair_id)
    pair_points = points[pair["pencils"]]
    e3_path = DIRECTION_RUN / "e3" / f"{case_id}.json.gz"
    e3 = read(e3_path)
    fit = e3["sets"][fit_arm]["fits"]["records"][pair_id]
    homography = np.asarray(fit["homography_working"], dtype=float)
    fit_corners, _ = detector.project(homography[None], detector.CORNER_COURT_M)
    fit_error = float(corner_errors(fit_corners, control)[0])
    assert abs(fit_error - fit["max_corner_working_px"]) < 1e-6
    basis, details = basis_for(pair_points, size, MATCHER_SETTINGS)
    assert basis is not None and details["status"] == "valid", details
    np.testing.assert_allclose(
        basis, pair["role"]["basis_working"], rtol=0, atol=1e-9
    )
    candidates = [
        ideal_parameters(basis, homography),
        ideal_parameters(basis, homography @ SYMMETRY),
    ]
    ideal, leak = min(candidates, key=lambda item: item[1])
    ideal_corners = courts_from(basis, ideal[:1], ideal[1:])
    ideal_error = float(corner_errors(ideal_corners, control)[0])
    assert abs(ideal_error - fit_error) < 1e-6

    matched: dict[int, object] = {}
    axis_data = []
    for axis in axes:
        matches = match_axis(
            basis,
            axis,
            (detector.X_COORDS, detector.Y_COORDS)[axis],
            observations,
            size,
            UNCAPPED,
            feet,
        )
        matched[axis] = matches
        recorded = pair["role"]["axes"][axis]
        kept_ids = matches.distinct[: MATCHER_SETTINGS.keep_axes]
        recorded_ids = [entry["axis_id"] for entry in recorded["entries"]]
        retained_order_match = kept_ids.tolist() == recorded_ids
        if require_exact_retained_order:
            assert retained_order_match, (kept_ids.tolist(), recorded_ids)
            np.testing.assert_allclose(
                matches.parameters[kept_ids],
                [entry["parameters"] for entry in recorded["entries"]],
                rtol=0,
                atol=1e-9,
            )
        else:
            assert set(kept_ids.tolist()) == set(recorded_ids)
            for entry in recorded["entries"]:
                np.testing.assert_allclose(
                    matches.parameters[entry["axis_id"]],
                    entry["parameters"],
                    rtol=0,
                    atol=1e-9,
                )
        assert matches.diagnostics["distinct_assignments"] == recorded[
            "diagnostics"
        ]["distinct_assignments"]
        assert matches.diagnostics["enumerated"] == recorded["diagnostics"][
            "enumerated"
        ]
        errors = sweep(basis, axis, matches.parameters, ideal[1 - axis], control)
        axis_data.append(
            {
                "axis": AXIS_NAMES[axis],
                "axis_index": axis,
                "coordinates": len((detector.X_COORDS, detector.Y_COORDS)[axis]),
                "recorded_diagnostics": recorded["diagnostics"],
                "replayed_diagnostics": matches.diagnostics,
                "recorded_entry_count": len(recorded["entries"]),
                "retained_order_match": retained_order_match,
                "matches": matches,
                "errors": errors,
            }
        )
    scale = np.asarray(
        [original_source["dimensions"]["width"], original_source["dimensions"]["height"]],
        dtype=float,
    ) / np.asarray(size)
    pool = record_pool_witness(record, control, scale)
    return {
        "case_id": case_id,
        "fit_arm": fit_arm,
        "pair_id": pair_id,
        "record_path": relative(record_path),
        "record_md5": md5_path(record_path),
        "input_path": relative(input_path),
        "input_md5": md5_path(input_path),
        "raw_pack_path": relative(source_for_case(case_id)),
        "raw_pack_md5": md5_path(source_for_case(case_id)),
        "e3_path": relative(e3_path),
        "e3_md5": md5_path(e3_path),
        "working_size": list(size),
        "native_size": [
            original_source["dimensions"]["width"],
            original_source["dimensions"]["height"],
        ],
        "pencils": pair["pencils"],
        "direction_fit_px": fit_error,
        "chart_leak": leak,
        "ideal_parameters": as_list(ideal),
        "control": control_context(case_id),
        "pair": {
            "pair_id": pair["pair_id"],
            "pencils": pair["pencils"],
            "status": pair["status"],
        },
        "axis_data": axis_data,
        "pool": pool,
    }


def assignment_key(values: np.ndarray) -> str:
    return ",".join(str(int(value)) for value in values)


def axis_witness(
    axis_data: dict,
    index: int,
    labels: dict[str, str],
) -> dict:
    matches = axis_data["matches"]
    errors = axis_data["errors"]
    eligible = (matches.supported >= MATCHER_SETTINGS.minimum_matches) & matches.player_compatible
    ordered = np.flatnonzero(eligible)
    ordered = ordered[np.argsort(-matches.scores[ordered], kind="stable")]
    eligible_ranks = {int(value): rank for rank, value in enumerate(ordered)}
    distinct_ranks = {
        int(value): rank for rank, value in enumerate(matches.distinct)
    }
    kept_ids = matches.distinct[: MATCHER_SETTINGS.keep_axes]
    kept_ranks = {int(value): rank for rank, value in enumerate(kept_ids)}
    assignment = matches.matches[index]
    return {
        "label": labels["label"],
        "axis": axis_data["axis"],
        "axis_id": int(index),
        "enumeration_id": int(index),
        "parameters": as_list(matches.parameters[index]),
        "anchors": as_list(matches.anchors[index]),
        "assignment_tuple": as_list(assignment),
        "assignment_key": assignment_key(assignment),
        "score": float(matches.scores[index]),
        "support_count": int(matches.supported[index]),
        "player_compatible": bool(matches.player_compatible[index]),
        "supported_and_players": bool(eligible[index]),
        "control_error_px": float(errors[index]),
        "eligible_score_rank_0_based": eligible_ranks.get(index),
        "distinct_score_rank_0_based": distinct_ranks.get(index),
        "kept_score_rank_0_based": kept_ranks.get(index),
        "rank_definition": "descending score order; ranks are zero-based",
    }


def build_check_b(replay: dict) -> dict:
    axis_data = replay["axis_data"][0]
    matches = axis_data["matches"]
    errors = axis_data["errors"]
    eligible = (matches.supported >= MATCHER_SETTINGS.minimum_matches) & matches.player_compatible
    eligible_ids = np.flatnonzero(eligible)
    distinct_ids = matches.distinct
    kept_ids = distinct_ids[: MATCHER_SETTINGS.keep_axes]
    closest_eligible = int(eligible_ids[np.argmin(errors[eligible_ids])])
    closest_distinct = int(distinct_ids[np.argmin(errors[distinct_ids])])
    closest_kept = int(kept_ids[np.argmin(errors[kept_ids])])
    identity = tuple(matches.matches[closest_eligible])
    representative = next(
        int(index)
        for index in distinct_ids
        if tuple(matches.matches[index]) == identity
    )
    assert matches.scores[representative] >= matches.scores[closest_eligible] - 1e-12

    labels = {
        "axis": axis_data["axis"],
        "label": "Amateur-3 R pair 43 horizontal trace",
    }
    witnesses = [
        axis_witness(axis_data, closest_eligible, {**labels, "label": "closest eligible enumerated"}),
        axis_witness(axis_data, representative, {**labels, "label": "representative of closest eligible assignment"}),
        axis_witness(axis_data, closest_distinct, {**labels, "label": "closest distinct assignment"}),
        axis_witness(axis_data, closest_kept, {**labels, "label": "closest kept assignment"}),
    ]
    return {
        "trace": witnesses,
        "identity_checks": {
            "closest_eligible_assignment_equals_representative": witnesses[0]["assignment_key"]
            == witnesses[1]["assignment_key"],
            "representative_score_is_not_lower": witnesses[1]["score"] >= witnesses[0]["score"] - 1e-12,
            "representative_is_distinct": witnesses[1]["distinct_score_rank_0_based"] is not None,
            "closest_eligible_is_distinct": witnesses[0]["distinct_score_rank_0_based"] is not None,
            "representative_equals_closest_distinct": witnesses[1]["assignment_key"]
            == witnesses[2]["assignment_key"],
        },
        "first_512_replay_check": {
            "record_entries": replay["axis_data"][0]["recorded_entry_count"],
            "ids_and_parameters_match_saved_record": True,
            "diagnostics_match_saved_record": True,
            "cap": MATCHER_SETTINGS.keep_axes,
        },
        "source": {
            "record_path": replay["record_path"],
            "record_md5": replay["record_md5"],
            "input_path": replay["input_path"],
            "input_md5": replay["input_md5"],
            "producer": "frozen projective_seed.match_axis with R pair 43 basis and saved direction-fit homography",
        },
        "working_size": replay["working_size"],
        "native_size": replay["native_size"],
        "control": replay["control"],
        "direction_fit_px": replay["direction_fit_px"],
        "rank_definition": (
            "eligible rows are sorted by descending score before duplicate removal; "
            "distinct ranks are the resulting survivor order; kept ranks are the first 512 distinct rows"
        ),
    }


def build_check_a(replays: list[dict]) -> dict:
    rows = []
    for replay in replays:
        by_axis = {data["axis_index"]: data for data in replay["axis_data"]}
        pair = pair_record(read(REPO / replay["record_path"]), replay["pair_id"])
        proxy = nearest_product_witness(
            np.asarray(pair["role"]["basis_working"], dtype=float),
            by_axis[0]["matches"],
            by_axis[1]["matches"],
            np.asarray(replay["control"]["corners_working_px"], dtype=float),
            pair,
        )
        rows.append(
            {
                "arm": replay["arm"],
                "stage": replay["stage"],
                "pair_id": replay["pair_id"],
                "working_size": replay["working_size"],
                "native_size": replay["native_size"],
                "record_path": replay["record_path"],
                "record_md5": replay["record_md5"],
                "input_path": replay["input_path"],
                "input_md5": replay["input_md5"],
                "axis_replay_gate": [
                    {
                        "axis": data["axis"],
                        "retained_order_match": data["retained_order_match"],
                        "retained_id_set_match": True,
                        "parameters_match_by_id": True,
                        "enumerated_and_distinct_counts_match": True,
                    }
                    for data in replay["axis_data"]
                ],
                "proxy": proxy,
                "full_pool": replay["pool"],
            }
        )
    baseline, filtered = rows
    proxy_delta = filtered["proxy"]["control_error_px"] - baseline["proxy"]["control_error_px"]
    pool_delta = filtered["full_pool"]["control_error_px"] - baseline["full_pool"]["control_error_px"]
    assert proxy_delta < 0 and pool_delta > 0
    return {
        "case_id": "gxBQ_window_00_frame_0",
        "control": replays[0]["control"],
        "rows": rows,
        "direction_of_change": {
            "one_pair_proxy": "improves" if proxy_delta < 0 else "worsens",
            "one_pair_proxy_delta_px": proxy_delta,
            "full_pre_global_pool": "improves" if pool_delta < 0 else "worsens",
            "full_pre_global_pool_delta_px": pool_delta,
        },
        "population_note": (
            "The proxy is pair 143's 512-by-512 kept-axis product. The full-pool row "
            "scans every matched pair's saved per-pair shortlist before the global cap."
        ),
    }


def identity_row_map() -> dict[tuple[str, str, str], dict]:
    rows = json.loads(IDENTITY_ROWS.read_text())
    return {
        (row["case_id"], str(row["candidate_id"]), row["ranking_role"]): row
        for row in rows
    }


def visual_row(rows: dict[tuple[str, str, str], dict], case_id: str, candidate_id: object, role: str) -> dict:
    candidate = str(candidate_id)
    direct = rows.get((case_id, candidate, role))
    if direct is not None:
        return direct
    matches = [
        row
        for (row_case, row_candidate, _row_role), row in rows.items()
        if row_case == case_id and row_candidate == candidate
    ]
    assert len(matches) == 1, (case_id, candidate, role, matches)
    return matches[0]


def panel_item(
    *,
    case_id: str,
    population: str,
    arm: str,
    stage: str,
    candidate_id: object,
    role: str,
    entry: dict,
    source_path: Path,
    source_record_md5: str | None,
    visual: dict,
) -> dict:
    return {
        "case_id": case_id,
        "population": population,
        "arm": arm,
        "stage": stage,
        "candidate_id": candidate_id,
        "ranking_role": role,
        "working_size": [960, 540],
        "native_size": [1920, 1080],
        "source_path": relative(source_path),
        "source_record_md5": source_record_md5,
        "original_visual_ruling": visual["visual_ruling"],
        "visual_ruling_source": relative(IDENTITY_ROWS),
        "entry": copy.deepcopy(entry),
    }


def build_ranking_panel() -> dict:
    ranking = read(RANKING_RECORDS)
    rows = identity_row_map()
    automatic = []
    for population in ranking["populations"]:
        if population["population"] != "automatic_all_camera":
            continue
        for entry in population["entries"]:
            candidate = entry["candidate_id"]
            if str(candidate) == str(population["line_winner_id"]):
                role = "line"
            elif str(candidate) == str(population["paint_winner_id"]):
                role = "paint"
            else:
                raise AssertionError((population["case_id"], candidate))
            automatic.append(
                panel_item(
                    case_id=population["case_id"],
                    population=population["population"],
                    arm="B",
                    stage="automatic_all_camera",
                    candidate_id=candidate,
                    role=role,
                    entry=entry,
                    source_path=RANKING_RECORDS,
                    source_record_md5=population["source_record_md5"],
                    visual=visual_row(rows, population["case_id"], candidate, role),
                )
            )
    assert len(automatic) == 17, len(automatic)

    control = read(CONTROL_RECORD)
    controls = []
    for role in ("line", "paint"):
        candidate = control[f"{role}_winner_id"]
        entry = next(item for item in control["entries"] if item["candidate_id"] == candidate)
        controls.append(
            panel_item(
                case_id="gxBQ_window_00_frame_0",
                population="gx0_label_guided_bank_control",
                arm="label_guided_observed_bank",
                stage="gx0_label_guided_bank_control",
                candidate_id=candidate,
                role=role,
                entry=entry,
                source_path=CONTROL_RECORD,
                source_record_md5=md5_path(CONTROL_RECORD),
                visual=visual_row(rows, "gxBQ_window_00_frame_0", candidate, role),
            )
        )

    diagnosis = read(CONTROL_DIAGNOSIS)
    approved = copy.deepcopy(diagnosis["approved_entry"])
    approved["corners_working_px"] = as_list(np.asarray(approved["corners_px"], dtype=float) / 2.)
    comparator = panel_item(
        case_id="gxBQ_window_00_frame_0",
        population="GX0 supplied-direction comparator",
        arm="label_guided_observed_bank",
        stage="approved_reference_comparator",
        candidate_id=approved["candidate_id"],
        role="comparator (approved reference)",
        entry=approved,
        source_path=CONTROL_DIAGNOSIS,
        source_record_md5=md5_path(CONTROL_DIAGNOSIS),
        visual=visual_row(rows, "gxBQ_window_00_frame_0", 89, "comparator (approved reference)"),
    )
    return {
        "schema": "c2-ranking-panel/1",
        "selection_bias": ranking["selection_bias"],
        "snapshot_basis": ranking["snapshot_basis"],
        "source_ranking_records": relative(RANKING_RECORDS),
        "source_ranking_records_md5": md5_path(RANKING_RECORDS),
        "source_control_record": relative(CONTROL_RECORD),
        "source_control_record_md5": md5_path(CONTROL_RECORD),
        "source_comparator_record": relative(CONTROL_DIAGNOSIS),
        "source_comparator_record_md5": md5_path(CONTROL_DIAGNOSIS),
        "original_visual_rulings_preserved": True,
        "counts": {"automatic_winners": len(automatic), "observed_bank_controls": len(controls), "comparators": 1},
        "entries": automatic + controls,
        "comparator": comparator,
    }


def source_versions() -> dict:
    paths = {
        "projective_seed.py": FROZEN_PROJECTIVE_SOURCE,
        "axis_replay.py": CURRENT_AXIS_SOURCE,
        "filter_replay.py": CURRENT_FILTER_SOURCE,
        **PRODUCER_SOURCES,
    }
    return {
        name: {"path": relative(path), "md5": md5_path(path)}
        for name, path in paths.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    baseline_record = BASELINE_GENERATION / "gxBQ_window_00_frame_0.json.gz"
    filtered_record = (
        FILTER_RUN
        / "matcher/paint_observations/results/gxBQ_window_00_frame_0.json.gz"
    )
    filtered_input = FILTER_INPUTS / "paint_observations/cases/gxBQ_window_00_frame_0.json.gz"
    baseline_replay = replay_pair_axes(
        "gxBQ_window_00_frame_0",
        "B",
        143,
        baseline_record,
        source_for_case("gxBQ_window_00_frame_0"),
        (0, 1),
    )
    baseline_replay.update({"arm": "B", "stage": "baseline_generation"})
    filtered_replay = replay_pair_axes(
        "gxBQ_window_00_frame_0",
        "B",
        143,
        filtered_record,
        filtered_input,
        (0, 1),
    )
    filtered_replay.update({"arm": "paint_observations", "stage": "results"})
    check_a = build_check_a([baseline_replay, filtered_replay])

    amateur_record = PREGATE_RECORDS / "R/results/am3_window_00_frame_0.json.gz"
    amateur_replay = replay_pair_axes(
        "am3_window_00_frame_0",
        "R",
        43,
        amateur_record,
        source_for_case("am3_window_00_frame_0"),
        (0,),
        require_exact_retained_order=True,
    )
    check_b = build_check_b(amateur_replay)
    panel = build_ranking_panel()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    witnesses = {
        "schema": "c2-two-decisive-traces/1",
        "git_commit_at_check": git_commit,
        "matcher_settings": {
            "angle_deg": MATCHER_SETTINGS.angle_deg,
            "minimum_matches": MATCHER_SETTINGS.minimum_matches,
            "keep_axes": MATCHER_SETTINGS.keep_axes,
            "batch": MATCHER_SETTINGS.batch,
        },
        "producer_versions": source_versions(),
        "check_a": check_a,
        "check_b": check_b,
        "ranking_panel": {
            "path": "ranking_panel.json",
            "counts": panel["counts"],
        },
    }
    write_json(args.output / "witnesses.json", witnesses)
    write_json(args.output / "ranking_panel.json", panel)
    print(
        json.dumps(
            {
                "check_a": check_a["direction_of_change"],
                "check_b": check_b["identity_checks"],
                "panel": panel["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
