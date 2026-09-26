"""Replay a predeclared, bounded lower-post reward over frozen W5 candidate pools."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RUN = ROOT / "wider_evaluation/runs/20260922"
DEFAULT_OUTPUT = REPO / "local_scratch/net_recovery/20260923/bounded"
sys.path[:0] = [str(ROOT / "colour_consistency"), str(ROOT / "wider_evaluation")]

import am1_net_selection_trial as net  # pyrefly: ignore[missing-import]
from run_cases import load_runtime  # pyrefly: ignore[missing-import]

from experiments.annotator.independent_court import net_geometry
from scratch.court_det_fix.court_detector.net_choice import (
    LOWER_SAMPLE_COUNT,
    choose,
    post_features,
    project_pieces,
    reward,
)

SAVED_SCAN = ROOT / "net_recovery/saved_net_scan.json.gz"
AM1_SCAN = ROOT / "colour_consistency/am1_net_selection_trial.json.gz"
SPLIT = ROOT / "net_recovery/bounded_split.json.gz"
MANIFEST = RUN / "manifest.json.gz"
CONTROL_PACK = RUN / "control_inputs.json.gz"
SETTINGS = (("primary", 0.04, 4.0), ("weight_02", 0.02, 4.0),
            ("weight_08", 0.08, 4.0), ("overrun_02", 0.04, 2.0),
            ("overrun_08", 0.04, 8.0), ("zero", 0.0, 4.0))


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def selected_geometry(candidate: dict, context, projection: dict) -> dict:
    result = net.selection_geometry(candidate, context, projection)
    return net.verifier.jsonable(result)


def measure_case(case: dict, source_label: str, cohort: str, manifest: dict, verifier) -> dict:
    case_id = case["case_id"]
    context = verifier.prepare_view(ROOT, case_id)
    record_path = REPO / case["source_record"]
    record = read(record_path)
    if record["case_id"] != case_id or record["provenance"]["frame_path"] != context.frame_relative_path:
        raise ValueError(f"{case_id}: source record identity differs")
    if case["frame_path"] != context.frame_relative_path or manifest["image"] != context.frame_relative_path:
        raise ValueError(f"{case_id}: frame path differs")
    if (list(context.native_size) != case["native_size_wh"]
            or list(context.size) != case["working_size_wh"]
            or list(context.native_size) != record["provenance"]["native_dimensions"]
            or list(context.size) != record["provenance"]["working_dimensions"]):
        raise ValueError(f"{case_id}: dimensions differ")
    frame_md5 = hashlib.md5((ROOT / context.frame_relative_path).read_bytes()).hexdigest()
    if frame_md5 != manifest["image_md5"]:
        raise ValueError(f"{case_id}: frame MD5 differs from full manifest")

    candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
    ranking = record["rankings"]["C"]["provisional_rank"]
    criterion = record["rankings"]["C"]["r2_criterion"]
    if ranking != [row["origin_key"] for row in case["candidates"]]:
        raise ValueError(f"{case_id}: frozen camera ranking differs")
    rows = []
    projections = {}
    full_rank = 0
    for rank, key in enumerate(ranking, start=1):
        candidate = candidates[key]
        frozen = case["candidates"][rank - 1]
        gated = candidate["historical"]["historical_fullcourt"]
        paint_score = candidate["evidence"][criterion]
        if (frozen["original_rank"] != rank or frozen["historical_fullcourt"] != gated
                or ("paint_score" in frozen and frozen["paint_score"] != paint_score)):
            raise ValueError(f"{case_id}: frozen rank, gate or paint differs at {key}")
        if not gated:
            continue
        full_rank += 1
        if frozen["full_court_rank"] != full_rank:
            raise ValueError(f"{case_id}: full-court rank differs at {key}")
        projection = project_pieces(candidate["corners_px"], context)
        projections[key] = projection
        posts = {}
        if projection["state"] == "measured":
            for name, piece_index in (("post_left", 2), ("post_right", 3)):
                posts[name] = post_features(np.asarray(projection["pieces_working_px"][piece_index]),
                                            context.segments, context.size)
        focal = projection.get("focal_widths")
        at_focal_search_bound = bool(focal is not None and (
            np.isclose(focal, net_geometry.FOCAL_MIN_WIDTHS)
            or np.isclose(focal, net_geometry.FOCAL_MAX_WIDTHS)))
        rows.append({
            "origin_key": key, "candidate_id": candidate["candidate_id"], "source": candidate["source"],
            "original_rank": rank, "full_court_rank": full_rank, "historical_fullcourt": True,
            "camera_eligible": candidate["camera_eligible"], "gate_camera_error": candidate["gates"]["camera_error"],
            "paint_score": paint_score, "net_state": projection["state"],
            "net_camera_error": projection.get("camera_error"), "focal_widths": focal,
            "at_focal_search_bound": at_focal_search_bound,
            "projection_reason": projection.get("reason"), "posts": posts,
        })
    baseline_key = rows[0]["origin_key"] if rows else None
    saved_baseline = case["baseline"]
    if baseline_key != case["original_comparator_full"]["gated"]:
        raise ValueError(f"{case_id}: original comparator differs")
    if baseline_key != (saved_baseline and saved_baseline["origin_key"]):
        raise ValueError(f"{case_id}: original baseline differs")
    choices = {}
    scores = {}
    for name, weight, overrun in SETTINGS:
        key, scores[name] = choose(rows, weight, overrun)
        choices[name] = key
    if choices["zero"] != baseline_key:
        raise ValueError(f"{case_id}: zero weight changed baseline")
    if (all(reward(row["posts"], 4.0) == 0 for row in rows if row["net_state"] == "measured")
            and choices["primary"] != baseline_key):
        raise ValueError(f"{case_id}: no-positive-cue case changed")
    selected_keys = set(choices.values()) | {baseline_key, case["trial"] and case["trial"]["origin_key"]}
    selected_keys.discard(None)
    geometries = {}
    for key in selected_keys:
        if key not in projections:
            raise ValueError(f"{case_id}: saved choice is outside the full-court gate")
        geometries[key] = selected_geometry(candidates[key], context, projections[key])
    for role, key in (("baseline", baseline_key), ("trial", case["trial"] and case["trial"]["origin_key"])):
        saved = case[role]
        if saved is None:
            continue
        geometry = geometries[key]
        for field in ("corners_native_px", "corners_working_px", "homography_working"):
            if not np.array_equal(geometry[field], saved[field]):
                raise ValueError(f"{case_id}: {role} {field} differs from saved scan")
        if saved["net"]["state"] == "measured":
            for field in ("pieces_native_px", "pieces_working_px"):
                if not np.allclose(geometry["net"][field], saved["net"][field], rtol=0, atol=1e-9):
                    raise ValueError(f"{case_id}: {role} {field} differs from saved scan")
            native_per_working = np.asarray(context.native_size) / np.asarray(context.size)
            if not np.allclose(np.asarray(geometry["net"]["pieces_working_px"]) * native_per_working,
                               geometry["net"]["pieces_native_px"], rtol=0, atol=1e-9):
                raise ValueError(f"{case_id}: native-to-working scaling differs")
    return {
        "case_id": case_id, "source_label": source_label, "cohort": cohort,
        "group": case.get("group", manifest["group"]), "arm": case.get("arm", manifest["arm"]),
        "scan_label": case["label"], "source_scan": relative(SAVED_SCAN if source_label == "frozen71" else AM1_SCAN),
        "source_record": relative(record_path), "frame_path": context.frame_relative_path,
        "frame_md5": frame_md5, "native_size_wh": list(context.native_size),
        "working_size_wh": list(context.size), "segments_working_px": context.segments.tolist(),
        "ordering_criterion": criterion, "candidate_rows": rows, "baseline_key": baseline_key,
        "old_fixed_trial_key": case["trial"] and case["trial"]["origin_key"],
        "choices": choices, "scores": scores, "selected_geometry": geometries,
        "reference_status": case.get("reference_status"), "view_status": case.get("view_status"),
        "retrospective_saved_diagnostic_only": case.get("retrospective_reference_diagnostic_only"),
    }


def summary(cases: list[dict]) -> str:
    lines = ["# Bounded post-base scoring replay", "", (f"Cases: {len(cases)}; frozen saved: "
             f"{sum(case['source_label'] == 'frozen71' for case in cases)}; seeded Am1: "
             f"{sum(case['source_label'] == 'am1_seeded_pool' for case in cases)}."), "",
             "The reward is conservative evidence of a plausible projected post base, not verification of a real post.",
             ("Existing full-court gates fix acceptance; unchanged labelled non-court acceptance "
              "is not new specificity evidence."), ""]
    for section in ("Development", "Withheld"):
        subset = [case for case in cases if (case["cohort"] == "development") == (section == "Development")]
        lines += [f"## {section} ({len(subset)} cases)", "",
                  "| Arm (weight, overrun px) | Changed from paint baseline |", "|---|---:|"]
        for name, weight, overrun in SETTINGS:
            count = sum(case["choices"][name] != case["baseline_key"] for case in subset)
            lines.append(f"| {name} ({weight:g}, {overrun:g}) | {count} |")
        lines += ["", "Changes by source and cohort (primary):", "",
                  "| Source / group / cohort | Cases | Changed |", "|---|---:|---:|"]
        families = {(case["source_label"], case["group"], case["cohort"]) for case in subset}
        for source, group, cohort in sorted(families):
            members = [case for case in subset
                       if case["source_label"] == source and case["group"] == group and case["cohort"] == cohort]
            changed = sum(case["choices"]["primary"] != case["baseline_key"] for case in members)
            lines.append(f"| {source} / {group} / {cohort} | {len(members)} | {changed} |")
        lines += ["", "Primary changes:", "",
                  "| Case | Source / group | Baseline → selected rank | Paint gap | Reward | Combined score |",
                  "|---|---|---:|---:|---:|---:|"]
        for case in subset:
            if case["choices"]["primary"] == case["baseline_key"]:
                continue
            selected = next(row for row in case["scores"]["primary"]
                            if row["origin_key"] == case["choices"]["primary"])
            baseline = next(row for row in case["scores"]["primary"]
                            if row["origin_key"] == case["baseline_key"])
            gap = selected["paint_score"] - baseline["paint_score"]
            lines.append(f"| {case['case_id']} | {case['source_label']} / {case['group']} | "
                         f"{baseline['full_court_rank']} → {selected['full_court_rank']} | {gap:+.5f} | "
                         f"{selected['net_reward']:.1f} | {selected['combined_score']:.5f} |")
        lines += ["", "Sensitivity differences from primary:", ""]
        for name, _, _ in SETTINGS[1:]:
            differing = [case["case_id"] for case in subset if case["choices"][name] != case["choices"]["primary"]]
            lines.append(f"- {name}: {len(differing)} — {', '.join(differing) if differing else 'none'}")
        lines.append("")
    labelled = [case for case in cases if case["source_label"] == "frozen71"
                and case["reference_status"] == "non_court"]
    lines += [(f"Labelled non-court rows identified from stored status: {len(labelled)}; "
               f"baseline accepted {sum(case['baseline_key'] is not None for case in labelled)}; "
               f"primary accepted {sum(case['choices']['primary'] is not None for case in labelled)}."), ""]
    return "\n".join(lines)


def write_atomic(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb") as stream:
            stream.write(json.dumps(net.verifier.jsonable(payload), allow_nan=False).encode("utf-8"))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(output_dir: Path, case_filter: set[str] | None = None) -> None:
    cv2.setNumThreads(1)
    saved = read(SAVED_SCAN)
    seeded = [case for case in read(AM1_SCAN)["cases"] if case["label"] == "am1_seeded_pool"]
    split = read(SPLIT)
    manifest = {row["case_id"]: row for row in read(MANIFEST)["cases"]}
    if not saved["complete"] or len(saved["cases"]) != 71 or len(seeded) != 1 or len(split["cases"]) != 71:
        raise ValueError("Frozen pool or predeclared split is incomplete")
    cohorts = {row["case_id"]: row["cohort"] for row in split["cases"]}
    if set(cohorts) != {row["case_id"] for row in saved["cases"]}:
        raise ValueError("Split and saved scan case IDs differ")
    extra = split["additional_development"]
    if len(extra) != 1 or extra[0]["case_id"] != seeded[0]["case_id"]:
        raise ValueError("Seeded Am1 split declaration differs")
    _, verifier, _ = load_runtime(ROOT, CONTROL_PACK)
    inputs = [(case, "frozen71", cohorts[case["case_id"]]) for case in saved["cases"]]
    inputs.append((seeded[0], "am1_seeded_pool", "development"))
    if case_filter is not None:
        inputs = [item for item in inputs if item[0]["case_id"] in case_filter]
        if {item[0]["case_id"] for item in inputs} != case_filter:
            raise ValueError("Unknown requested case")
    cases = []
    for case, label, cohort in inputs:
        result = measure_case(case, label, cohort, manifest[case["case_id"]], verifier)
        cases.append(result)
        print(f"{len(cases)}/{len(inputs)} {result['case_id']}: {result['baseline_key']} -> "
              f"{result['choices']['primary']}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "bounded-post-base-scoring-replay/1", "complete": case_filter is None,
        "settings": [{"name": name, "weight": weight, "overrun_working_px": overrun}
                     for name, weight, overrun in SETTINGS],
        "measurement": {"source": "cached DeepLSD segments", "samples_per_post": LOWER_SAMPLE_COUNT,
                        "samples_from_original": net.SAMPLES_PER_PIECE,
                        "perpendicular_working_px": net.PERPENDICULAR_TOLERANCE_WORKING_PX,
                        "absolute_direction_degrees": net.DIRECTION_TOLERANCE_DEG,
                        "endpoint_extent_margin_working_px": net.EXTENT_MARGIN_WORKING_PX,
                        "focal_search_min_widths": net_geometry.FOCAL_MIN_WIDTHS,
                        "focal_search_max_widths": net_geometry.FOCAL_MAX_WIDTHS},
        "sources": {"saved_scan": relative(SAVED_SCAN), "seeded_scan": relative(AM1_SCAN),
                    "split": relative(SPLIT), "manifest": relative(MANIFEST)},
        "cases": cases,
    }
    write_atomic(output_dir / "bounded_trial.json.gz", payload)
    (output_dir / "summary.md").write_text(summary(cases), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", help="Case ID for a pilot replay; repeat to select several")
    arguments = parser.parse_args()
    main(arguments.output_dir, set(arguments.case) if arguments.case else None)
