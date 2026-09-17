#!/usr/bin/env python3
"""Read-only W2 calculations: verified L2 table plus a finite ridge-support probe.

No repository imports, matcher execution, model fitting, acceptance rule, or image claims.
With --witnesses, reads the already-produced L2 trace; it never recomputes pixels.
Python 3.10+, standard library only. Writes only the chosen output directory.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

REVISION = "bb6787adeca9a19b5f1d3c3324d54cfcb2ea6e68"
TABLE_BLOB = "7b1716becb2fae350a41935600fcd343efdb20af"
WITNESS_BLOB = "739af3d1dacc7f3fedb551eab01a2b4bad94fd06"
OFFSETS = [-4.0, -2.0, 0.0, 2.0, 4.0]
N = 24
CONTRAST = 10.0
LEGACY_FRACTION = 0.4
# A diagnostic definition, NOT a calibrated spatial or court-acceptance tolerance.
MAX_OFFSET_STEP = 2.0
MARKINGS = ["left_doubles", "left_singles", "centre", "right_singles", "right_doubles",
            "far_baseline", "far_long_service", "far_short_service", "near_short_service",
            "near_long_service", "near_baseline"]
MARKING_INTERVALS = [[0], [1], [2, 3], [4], [5], [6], [7], [8], [9], [10], [11]]
INTERVAL_MARKING = {i: m for m, intervals in zip(MARKINGS, MARKING_INTERVALS) for i in intervals}
EXPECTED_PANEL = {
    ("am2_window_01_frame_28019", "184:4123"),
    ("am2_window_00_frame_150", "30:33"),
    ("shuttleset_03_scene_0019", "165:6702"),
    ("shuttleset_03_scene_0019", "1:60"),
}


def git_blob_hash(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def verified_read(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    actual = git_blob_hash(data)
    if actual != expected:
        raise ValueError(f"Immutable input mismatch: {path}: {actual} != {expected}. "
                         "Do not silently substitute a different producing scope.")
    return data


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def table_calculation(path: Path) -> dict:
    data = verified_read(path, TABLE_BLOB)
    rows = list(csv.DictReader(data.decode().splitlines()))
    if len(rows) != 24:
        raise ValueError("Expected 24 L2 cells")
    cases: dict[str, dict[str, dict]] = {}
    for row in rows:
        cases.setdefault(row["case_id"], {})[row["cell"]] = row
    comparisons = {"G0_rescoring": ("G0,S0", "G0,S1"),
                   "union_rescoring": ("U,S0", "U,S1"),
                   "population_substitution_S0": ("G0,S0", "G1,S0"),
                   "population_addition_S0": ("G0,S0", "U,S0")}
    output: dict[str, Any] = {"input_git_blob": TABLE_BLOB, "rows": len(rows),
                              "revision_read": REVISION, "comparisons": {}}
    for comparison, (first, second) in comparisons.items():
        records = []
        for case_id, cells in cases.items():
            a, b = cells[first], cells[second]
            record: dict[str, Any] = {"case_id": case_id, "label": a["label"]}
            for kind in ["line", "paint"]:
                ka, kb = a[f"{kind}_winner_origin"] or None, b[f"{kind}_winner_origin"] or None
                da, db = a[f"{kind}_distance_working_px"], b[f"{kind}_distance_working_px"]
                record[kind] = {"before": ka, "after": kb,
                                "origin_changed": ka != kb,
                                "distance_before": float(da) if da else None,
                                "distance_after": float(db) if db else None,
                                "delta_working_px": float(db) - float(da) if da and db else None}
            records.append(record)
        output["comparisons"][comparison] = {
            "rows": records,
            "line_origin_changes": sum(r["line"]["origin_changed"] for r in records),
            "paint_origin_changes": sum(r["paint"]["origin_changed"] for r in records),
            "caution": "Origin changes across populations need not be geometrically distinct courts."
        }
    for cells in cases.values():
        for population in ["G0", "G1", "U"]:
            a, b = cells[f"{population},S0"], cells[f"{population},S1"]
            for key in ["candidate_count", "eligible_count", "nearest_all_distance_working_px",
                        "nearest_camera_distance_working_px"]:
                if a[key] != b[key]:
                    raise ValueError(f"Fixed-population invariant failed: {key}")
    output["fixed_population_invariants"] = "all candidate/eligibility counts and nearest distances unchanged"
    output["diagonal_flags"] = all(r["g0_diagonal_match"] == r["g1_diagonal_match"] == "True" for r in rows)
    output["diagonal_flag_caution"] = "Eight case/arm checks, repeated on six rows per case; not 48 independent tests."
    return output


def longest_path(mask: list[list[bool]]) -> tuple[int, list[list[int]]]:
    """Longest run of adjacent stations, moving at most one offset-grid step.

    Dynamic program, O(N*5*3). The returned path contains (station, offset_index).
    It describes lattice connectivity, NOT physical ownership or continuous pixels
    between the 24 stations. Ties keep the first path in station/offset order.
    """
    previous: list[list[list[int]]] = [[] for _ in OFFSETS]
    best: list[list[int]] = []
    for s, row in enumerate(mask):
        current: list[list[list[int]]] = [[] for _ in OFFSETS]
        for o, allowed in enumerate(row):
            if not allowed:
                continue
            predecessors = [previous[k] for k in range(len(OFFSETS))
                            if abs(OFFSETS[k] - OFFSETS[o]) <= MAX_OFFSET_STEP]
            prefix = max(predecessors, key=len, default=[])
            current[o] = prefix + [[s, o]]
            if len(current[o]) > len(best):
                best = current[o]
        previous = current
    return len(best), best


def interval_probe(interval: dict) -> dict:
    """Keep known pass, known failure, and unknown offset tests separate.

    lower = witnessed passing run; upper = best possible run if unknown tests
    passed. The upper bound is not attained evidence. No scalar court score.
    """
    visible = bool(interval["visible"])
    if not visible:
        return {"interval": interval["interval"], "status": "geometrically_unavailable",
                "legacy_interval_pass": None, "observed_pass_stations": None,
                "run_lower_stations": None, "run_upper_stations": None,
                "run_lower_fraction": None, "run_upper_fraction": None,
                "lower_path": None, "unknown_offset_tests": None}
    available = interval["available"]
    values = interval["minimum_contrast"]
    if len(available) != N or len(values) != N or any(len(r) != len(OFFSETS) for r in available + values):
        raise ValueError("Expected a 24 by 5 trace")
    known = [[bool(a) and isinstance(v, (int, float)) and math.isfinite(v)
              for a, v in zip(ar, vr)] for ar, vr in zip(available, values)]
    passing = [[k and v >= CONTRAST for k, v in zip(kr, vr)] for kr, vr in zip(known, values)]
    possible = [[p or not k for p, k in zip(pr, kr)] for pr, kr in zip(passing, known)]
    legacy_hits = sum(any(row) for row in passing)
    if "pass_by_shift" in interval and passing != interval["pass_by_shift"]:
        raise ValueError("Recorded passing offsets disagree with availability/contrast values")
    lo, path = longest_path(passing)
    hi, _ = longest_path(possible)
    available_stations = sum(any(row) for row in known)
    answer = {
        "interval": interval["interval"],
        "status": "measured" if available_stations else "no_available_offset_tests",
        "legacy_interval_pass": legacy_hits / N >= LEGACY_FRACTION,
        "legacy_pass_station_fraction": legacy_hits / N,
        "observed_pass_stations": legacy_hits,
        "stations_with_any_measured_offset": available_stations,
        "unknown_offset_tests": sum(not k for row in known for k in row),
        "run_lower_stations": lo, "run_upper_stations": hi,
        "run_lower_fraction": lo / N, "run_upper_fraction": hi / N,
        "lower_path": path,
        "note": "Bounds are connectivity diagnostics, not court acceptance or a measured upper path."
    }
    points = interval.get("tested_sample_coordinates_working_px")
    if points is not None:
        start, end = points[0][2], points[-1][2]
        length = math.dist(start, end)
        answer["clipped_span_working_px"] = length
        answer["station_spacing_working_px"] = length / (N - 1)
        answer["lower_path_endpoint_span_working_px"] = max(0, lo - 1) * length / (N - 1)
    if "interval_pass" in interval and answer["legacy_interval_pass"] != interval["interval_pass"]:
        raise ValueError("Legacy interval pass does not reproduce")
    return answer


def constructed_interval(hits: dict[int, list[int]], unknown: set[int] | None = None,
                         visible: bool = True) -> dict:
    unknown = unknown or set()
    # Realisable one-dimensional intensity cross-sections: background 40,
    # brightness 60 at the requested integer normal offsets. Tests use the
    # detector's actual +/-6-pixel flank spacing, including overlapping samples.
    contrasts = []
    for s in range(N):
        bright = {OFFSETS[o] for o in hits.get(s, [])}
        def intensity(z: float) -> float:
            return 60.0 if z in bright else 40.0
        contrasts.append([min(intensity(o) - intensity(o - 6.0),
                              intensity(o) - intensity(o + 6.0)) for o in OFFSETS])
    return {"interval": 0, "visible": visible,
            "available": [[s not in unknown] * len(OFFSETS) for s in range(N)],
            "minimum_contrast": contrasts}


def exact_rank(matrix: list[list[int]]) -> int:
    rows = [[Fraction(v) for v in row] for row in matrix]
    rank = 0
    for col in range(len(rows[0])):
        pivot = next((r for r in range(rank, len(rows)) if rows[r][col]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [v / scale for v in rows[rank]]
        for r in range(rank + 1, len(rows)):
            factor = rows[r][col]
            rows[r] = [a - factor * b for a, b in zip(rows[r], rows[rank])]
        rank += 1
    return rank


def line_constraints(lines: list[list[int]]) -> list[list[int]]:
    # Constructed H=identity correspondences, l_model cross (H^T l_image)=0.
    # Nine columns are H in row-major order; each line contributes 3 redundant equations.
    result = []
    for l in lines:
        equations = [[0] * 9 for _ in range(3)]
        for r in range(3):
            for c in range(3):
                v = [0, 0, 0]
                v[c] = l[r]
                cross = [l[1]*v[2] - l[2]*v[1], l[2]*v[0] - l[0]*v[2], l[0]*v[1] - l[1]*v[0]]
                for k in range(3):
                    equations[k][3*r+c] = cross[k]
        result.extend(equations)
    return result


def constructed_rank_check() -> dict:
    base = [[1, 0, -x] for x in [0, 1, 3, 4]] + [[0, 1, -2]]
    ranks = {"four_longitudinal_one_transverse": exact_rank(line_constraints(base)),
             "plus_another_longitudinal": exact_rank(line_constraints(base + [[1, 0, -2]])),
             "plus_another_transverse": exact_rank(line_constraints(base + [[0, 1, -3]]))}
    assert list(ranks.values()) == [7, 7, 8]
    return {"kind": "constructed ideal line correspondences, NOT the recovered scene19 homography",
            "ranks": ranks, "preserving_transformation": "x'=x; y'=2+a*(y-2), a>0",
            "qualification": "Finite observed endpoints can add constraints; crop boundaries cannot stand in for them."}


def synthetic_calculation() -> dict:
    definitions = {
        "coherent": constructed_interval({s: [2] for s in range(N)}),
        "hopping": constructed_interval({s: [0 if s % 2 == 0 else 4] for s in range(N)}),
        "contiguous_ten": constructed_interval({s: [2] for s in range(10)}),
        "isolated_ten": constructed_interval({s: [2] for s in range(0, 20, 2)}),
        "half_unknown": constructed_interval({s: [2] for s in range(12)}, set(range(12, 24))),
        "all_unknown": constructed_interval({}, set(range(N))),
        "geometrically_unavailable": constructed_interval({}, visible=False),
        # Two projected markings borrowing the same spatial ridge. Constant offsets
        # on either side of one stripe BOTH look coherent to this proposal.
        "shared_ridge_marking_a": constructed_interval({s: [4] for s in range(N)}),
        "shared_ridge_marking_b": constructed_interval({s: [0] for s in range(N)}),
    }
    results = {name: interval_probe(value) for name, value in definitions.items()}
    assert results["coherent"]["legacy_interval_pass"] == results["hopping"]["legacy_interval_pass"] is True
    assert results["coherent"]["run_lower_stations"] == 24
    assert results["hopping"]["run_lower_stations"] == 1
    assert results["contiguous_ten"]["run_lower_stations"] == 10
    assert results["isolated_ten"]["run_lower_stations"] == 1
    assert results["half_unknown"]["run_lower_stations"] == 12
    assert results["half_unknown"]["run_upper_stations"] == 24
    assert results["all_unknown"]["status"] == "no_available_offset_tests"
    assert results["geometrically_unavailable"]["run_lower_fraction"] is None
    assert results["shared_ridge_marking_a"]["run_lower_stations"] == 24
    assert results["shared_ridge_marking_b"]["run_lower_stations"] == 24
    # Exact profile object produced by any all-available, all-passing construction.
    aggregate = {"score": 1.0, "marking_ridge": [1.0] * 11,
                 "interval_visible": [True] * 12, "interval_ridge": [True] * 12}
    return {"kind": "constructed numerical witnesses, NOT extracted from repository images",
            "definitions": definitions, "results": results,
            "all_pass_aggregate": aggregate,
            "shared_ridge_geometry": {
                "predicted_horizontal_lines_y": [100.0, 108.0],
                "one_actual_bright_ridge_y": 104.0,
                "passing_offsets_for_the_two_lines": [4.0, -4.0],
                "conclusion": "Coherence does not establish separate ownership; both full-span profiles pass."},
            "approved_contrary_example_limit":
                "The user and W report supply identical all-pass profiles for approved Am2-150 30:33 and false Am2-28019 184:4123. "
                "These constructed preimages prove their aggregate cannot predict this probe. "
                "Neither construction is asserted to be either real candidate's trace."}


def witness_calculation(path: Path, output: Path) -> dict:
    payload = json.loads(verified_read(path, WITNESS_BLOB))
    if payload.get("schema") != "l2-generation-scoring-witnesses/1":
        raise ValueError("Unexpected L2 witness schema")
    panel = payload["diagnostic_panel"]
    if {(r["case_id"], r["candidate_id"]) for r in panel} != EXPECTED_PANEL or len(panel) != 4:
        raise ValueError("Wrong diagnostic panel")
    rows = []
    exports = []
    for candidate in panel:
        trace = candidate["trace"]
        settings = trace["settings"]
        expected = {"samples_along": N, "centre_offsets_working_px": OFFSETS,
                    "side_distance_working_px": 6.0, "minimum_contrast": CONTRAST,
                    "minimum_fraction": LEGACY_FRACTION}
        if settings != expected:
            raise ValueError(f"Changed extractor settings: {settings}")
        for key in ["interval_pass_matches_detector", "saved_profile_reproduced", "existing_profile_reproduced"]:
            if trace.get(key) is not True:
                raise ValueError(f"Original trace replay check absent/failed: {key}")
        if len(trace["intervals"]) != 12:
            raise ValueError("Expected 12 finite painted intervals")
        compact_intervals = []
        for interval in trace["intervals"]:
            metrics = interval_probe(interval)
            identity = {"case_id": candidate["case_id"], "candidate_id": candidate["candidate_id"],
                        "population": "ORIGINAL automatic_all_camera", "existing_ruling": candidate["existing_ruling"],
                        "marking": INTERVAL_MARKING[interval["interval"]]}
            rows.append(identity | {k: v for k, v in metrics.items() if k not in ["lower_path", "note"]})
            # Small trace for review; no repeated all-three coordinate arrays, raw images,
            # 24x5x3 RGB data, or unrelated candidate entries.
            compact_intervals.append({"interval": interval["interval"], "visible": interval["visible"],
                "metrics": metrics, "available": interval["available"],
                "pass_by_shift": interval["pass_by_shift"], "passing_shifts": interval["passing_shifts"],
                "minimum_contrast": interval["minimum_contrast"],
                "centre_coordinates_at_zero_offset": [p[2] for p in interval["tested_sample_coordinates_working_px"]]})
        safe = candidate["case_id"] + "__" + candidate["candidate_id"].replace(":", "_")
        exported = {"input_blob": WITNESS_BLOB, "revision_read": REVISION,
                    "case_id": candidate["case_id"], "candidate_id": candidate["candidate_id"],
                    "existing_ruling": candidate["existing_ruling"], "source_record": candidate["source_record"],
                    "image": candidate["image"], "candidate": candidate["candidate"],
                    "settings": settings, "intervals": compact_intervals}
        file = output / "compact_traces" / f"{safe}.json"
        write_json(file, exported)
        exports.append(str(file))
        pair_intervals = {("am2_window_01_frame_28019", "184:4123"): [10, 11],
                          ("am2_window_00_frame_150", "30:33"): [6, 7]}.get(
                              (candidate["case_id"], candidate["candidate_id"]))
        if pair_intervals is not None:
            # Keep the actual side coordinates and both contrasts for pixel attribution.
            pair = {k: exported[k] for k in ["input_blob", "revision_read", "case_id", "candidate_id",
                                            "existing_ruling", "source_record", "image", "settings"]}
            pair["intervals"] = [trace["intervals"][i] for i in pair_intervals]
            pair["purpose"] = "Joint observed-support inspection; no new rank or acceptance rule."
            pair_file = output / "pair_traces" / f"{safe}.json"
            write_json(pair_file, pair)
            exports.append(str(pair_file))
    with (output / "real_interval_probe.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    return {"kind": "real saved-trace calculation, not a new pixel replay",
            "input_blob": WITNESS_BLOB, "candidates": 4, "interval_rows": len(rows),
            "exports": exports, "no_acceptance_rule": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path(__file__).parent / "inputs/l2_comparison.csv")
    parser.add_argument("--witnesses", type=Path, help="Optional exact local 3.99 MB L2 witnesses.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table = table_calculation(args.table)
    synthetic = synthetic_calculation()
    write_json(args.output / "l2_deltas.json", table)
    write_json(args.output / "synthetic_probe.json", synthetic)
    write_json(args.output / "constructed_rank.json", constructed_rank_check())
    status = {"real_table_calculation": "executed", "synthetic_probe": "executed",
              "real_pixel_trace_probe": "NOT RUN: --witnesses not supplied",
              "repository_detector_or_matcher": "not executed", "pixels_viewed": False}
    if args.witnesses:
        status["real_pixel_trace_probe"] = witness_calculation(args.witnesses, args.output)
    write_json(args.output / "execution_status.json", status)
    print(json.dumps({"status": status,
                      "table_changes": {k: {n: v[n] for n in ["line_origin_changes", "paint_origin_changes"]}
                                        for k, v in table["comparisons"].items()},
                      "synthetic_run_bounds": {k: [v["run_lower_stations"], v["run_upper_stations"]]
                                               for k, v in synthetic["results"].items()}}, indent=2))


if __name__ == "__main__":
    main()
