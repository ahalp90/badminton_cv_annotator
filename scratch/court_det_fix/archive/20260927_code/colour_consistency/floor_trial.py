"""Automatic observed-internal-marking floor-colour trial for saved courts."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scratch/court_det_fix/w5_holistic")]

from experiments.annotator.independent_court import detector
from scratch.court_det_fix.colour_consistency import observed_colour as observed
from scratch.court_det_fix.w5_holistic import verifier

BASE = Path(__file__).resolve().parent
COURT = BASE.parent
WIDER = COURT / "wider_evaluation/runs/20260922"
SVD = COURT / "svd_search/run_20260923/cases"
OUTPUT = BASE / "floor_trial.json.gz"
CASES = ("am2_window_01_frame_28019", "gxBQ_window_00_frame_0", "gxBQ_window_00_frame_5")
OFFSETS_M = (0.15, 0.35, 0.60)
FRACTIONS = np.linspace(0.05, 0.95, 40)
CHROMA_FLOORS = (8.0, 12.0, 20.0)
PRIMARY_FLOOR = 12.0
DISPERSION_MULTIPLIER = 3.0
MIN_PAIRED_SAMPLES = 2
MIN_VALID_LOCATIONS = 20
MIN_CONTRADICTION_RUN = 8


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write(path: Path, value: dict) -> None:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(gzip.compress(payload, mtime=0))


def image_samples(lab: np.ndarray, points: np.ndarray, boxes: np.ndarray,
                  depth: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return native Lab samples and a mask; invalid pixels never enter decisions."""
    shape = points.shape[:-1]
    flat = points.reshape(-1, 2)
    valid = np.isfinite(flat).all(axis=1)
    valid &= verifier.observable_points(flat, (lab.shape[1], lab.shape[0]), boxes)
    valid &= ((flat >= 0) & (flat <= np.asarray([lab.shape[1] - 1, lab.shape[0] - 1]))).all(axis=1)
    if depth is not None:
        valid &= np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0)
    values = np.full((len(flat), 3), np.nan, dtype=float)
    if valid.any():
        safe = flat[valid].astype(np.float32)
        sampled = cv2.remap(lab, safe[:, 0], safe[:, 1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        values[valid] = sampled.reshape(-1, 3)
    return values.reshape(*shape, 3), valid.reshape(shape)


def anchor_evidence(fragments: list[dict]) -> tuple[dict, np.ndarray]:
    groups: dict[str, list[dict]] = {}
    for fragment in fragments:
        if fragment["usable"] and fragment["marking"] not in observed.OUTER_MARKINGS:
            groups.setdefault(fragment["marking"], []).append(fragment)
    modes = np.asarray([np.median([row["median_side_ab"] for row in rows], axis=0)
                        for rows in groups.values()], dtype=float).reshape(-1, 2)
    evidence = []
    distances = []
    weights = []
    for name, rows in groups.items():
        evidence.append({"marking": name, "mode_ab": np.median([row["median_side_ab"] for row in rows], axis=0).tolist(),
                         "fragments": [{"raw_fragment_id": row["raw_fragment_id"], "intervals": row["intervals"],
                                        "median_side_ab": row["median_side_ab"],
                                        "side_samples": [{"xy_native": sample["xy_native"][side],
                                                          "lab": sample["side_minus_lab"] if side == 0
                                                          else sample["side_plus_lab"]}
                                                         for sample in row["samples"] if sample["valid"]
                                                         for side in (0, 2)]} for row in rows]})
        for row in rows:
            distance = float(np.linalg.norm(modes - np.asarray(row["median_side_ab"]), axis=1).min())
            distances.append(distance)
            weights.append(1 / len(rows))
    dispersion = None
    if len(modes) >= 2:
        order = np.argsort(distances)
        cumulative = np.cumsum(np.asarray(weights)[order])
        dispersion = float(np.asarray(distances)[order][np.searchsorted(cumulative, 0.9 * cumulative[-1])])
    return {"candidate_conditioned": True, "markings": evidence,
            "valid_marking_count": len(modes), "dispersion_ab": dispersion,
            "modes_ab": modes.tolist()}, modes


def court_samples(corners: list, lab: np.ndarray, boxes: np.ndarray) -> list[dict]:
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, np.asarray(corners, np.float32))
    model = detector.CORNER_COURT_M.astype(float)
    normals = ((0, 1), (-1, 0), (0, -1), (1, 0))
    edges = []
    for edge_index in range(4):
        start, end = model[edge_index], model[(edge_index + 1) % 4]
        boundary = start + FRACTIONS[:, None] * (end - start)
        inward = np.asarray(normals[edge_index], dtype=float)
        offsets = np.asarray(OFFSETS_M)
        court_points = np.stack((boundary[:, None, :] + offsets[None, :, None] * inward,
                                 boundary[:, None, :] - offsets[None, :, None] * inward), axis=1)
        points, depth = detector.project(homography[None], court_points.reshape(-1, 2))
        points = points[0].reshape(40, 2, 3, 2)
        depth = depth[0].reshape(40, 2, 3)
        values, valid = image_samples(lab, points, boxes, depth)
        edges.append({"edge": ("far_baseline", "right_sideline", "near_baseline", "left_sideline")[edge_index],
                      "fractions": FRACTIONS.tolist(), "xy_native": points.tolist(),
                      "valid": valid.tolist(), "depth_positive": (depth > 0).tolist()})
        # Null values keep absent evidence distinct from a true dark Lab value.
        edges[-1]["lab"] = [[
            [values[position, side, offset].tolist() if valid[position, side, offset] else None
             for offset in range(3)] for side in range(2)] for position in range(40)]
        edges[-1]["xy_native"] = [[
            [points[position, side, offset].tolist() if np.isfinite(points[position, side, offset]).all() else None
             for offset in range(3)] for side in range(2)] for position in range(40)]
    return edges


def longest_run(flags: list[bool]) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def decide(edges: list[dict], modes: np.ndarray, dispersion: float | None, floor: float) -> dict:
    threshold = max(floor, DISPERSION_MULTIPLIER * dispersion) if dispersion is not None else None
    summaries = []
    for edge in edges:
        rows = []
        for fraction, locations, labs, masks in zip(edge["fractions"], edge["xy_native"],
                                                     edge["lab"], edge["valid"], strict=True):
            distances = []
            for side in range(2):
                side_distances = [float(np.linalg.norm(modes - np.asarray(labs[side][offset])[1:], axis=1).min())
                                  for offset in range(3) if masks[side][offset] and len(modes)]
                distances.append(side_distances)
            paired = all(len(side) >= MIN_PAIRED_SAMPLES for side in distances)
            inward = float(np.median(distances[0])) if paired else None
            outward = float(np.median(distances[1])) if paired else None
            contradiction = bool(paired and threshold is not None and inward > threshold and outward > threshold)
            rows.append({"fraction": fraction, "valid_counts": [len(side) for side in distances],
                         "paired": paired, "median_distance_ab": [inward, outward],
                         "contradiction": contradiction})
        run = longest_run([row["contradiction"] for row in rows])
        summaries.append({"edge": edge["edge"], "locations": rows,
                          "valid_paired_count": sum(row["paired"] for row in rows),
                          "contradictory_count": sum(row["contradiction"] for row in rows),
                          "max_contiguous_contradiction": run})
    if threshold is None:
        decision, reason = "no_decision", "fewer_than_two_observed_internal_markings"
    elif any(row["max_contiguous_contradiction"] >= MIN_CONTRADICTION_RUN for row in summaries):
        decision, reason = "abstain", "contiguous_floor_colour_contradiction"
    elif any(row["valid_paired_count"] >= MIN_VALID_LOCATIONS for row in summaries):
        decision, reason = "keep", "adequate_paired_support_without_long_contradiction"
    else:
        decision, reason = "no_decision", "insufficient_paired_boundary_support"
    return {"decision": decision, "reason": reason, "threshold_ab": threshold, "edges": summaries}


def selected_arms(case_id: str, numeric: dict) -> list[dict]:
    selected = numeric["selections"]["full"]["gated"]
    arms = [{"arm": "saved_w5", "source": "numeric_fit", "selected_key": selected,
             "corners_native_px": numeric["candidates"][selected]["corners_px"] if selected else None}]
    for kind in ("baseline", "deeper"):
        record = read(SVD / kind / f"{case_id}.json.gz")
        key = record["selected_key"]
        matches = [row for row in record["candidates"] if row["origin_key"] == key] if key else []
        if key is None:
            arms.append({"arm": f"svd_{kind}", "source": f"svd_search/{kind}",
                         "selected_key": None, "corners_native_px": None})
            continue
        if len(matches) != 1:
            raise ValueError(f"{case_id}/{kind}: expected one selected candidate, got {len(matches)}")
        arms.append({"arm": f"svd_{kind}", "source": f"svd_search/{kind}",
                     "selected_key": key, "corners_native_px": matches[0]["corners_px"]})
    return arms


def run() -> dict:
    manifest = {row["case_id"]: row for row in read(WIDER / "manifest.json.gz")["cases"]}
    numeric = {row["case_id"]: row for row in read(WIDER / "numeric_fit.json.gz")["cases"]}
    cases = []
    for case_id in CASES:
        item = manifest[case_id]
        record = numeric[case_id]
        context = verifier.prepare_view(COURT, case_id)
        image, lab, grey, boxes = observed.native_image(context, COURT)
        native_size = [image.shape[1], image.shape[0]]
        if native_size != record["native_size"]:
            raise ValueError(f"{case_id}: native size differs from saved record")
        if item["same_image_boxes"] != context.same_image_mask_available:
            raise ValueError(f"{case_id}: manifest box provenance differs from verifier")
        measured = {}
        arms = []
        for arm in selected_arms(case_id, record):
            if arm["corners_native_px"] is None:
                arms.append({**arm, "measurement_id": None, "before": "keep", "after": "no_decision",
                             "reason": "no_saved_candidate"})
                continue
            geometry_key = json.dumps(arm["corners_native_px"], separators=(",", ":"))
            if geometry_key not in measured:
                observed_rows = observed.observed_fragments(context, arm["corners_native_px"], lab, grey, boxes)
                anchors, modes = anchor_evidence(observed_rows["fragments"])
                edges = court_samples(arm["corners_native_px"], lab, boxes)
                dispersion = anchors["dispersion_ab"]
                measured[geometry_key] = {
                    "anchors": anchors, "observed": observed_rows,
                    "samples": edges,
                    "primary": decide(edges, modes, dispersion, PRIMARY_FLOOR),
                    "sensitivity": {str(int(floor)): decide(edges, modes, dispersion, floor)
                                    for floor in CHROMA_FLOORS},
                }
            measurement_id = list(measured).index(geometry_key)
            decision = measured[geometry_key]["primary"]
            arms.append({**arm, "measurement_id": measurement_id,
                         "before": "keep", "after": decision["decision"], "reason": decision["reason"]})
            print(case_id, arm["arm"], arm["selected_key"], "keep ->", decision["decision"],
                  decision["reason"])
        cases.append({"case_id": case_id, "image": item["image"], "native_size": native_size,
                      "working_size": record["working_size"],
                      "anchor_status": "candidate_conditioned_observed_internal_markings",
                      "box_mask": {"same_image_available": context.same_image_mask_available,
                                   "box_count": len(boxes),
                                   "unavailable_reason": context.person_mask_unavailable_reason},
                      "arms": arms,
                      "measurements": list(measured.values())})
    return {
        "schema": "automatic-observed-floor-colour-trial/1",
        "inputs": {"manifest": str((WIDER / "manifest.json.gz").relative_to(COURT)),
                   "numeric_fit": str((WIDER / "numeric_fit.json.gz").relative_to(COURT)),
                   "svd": "svd_search/run_20260923/cases/{baseline,deeper}/<case>.json.gz"},
        "supervision": "Automatic candidate-conditioned observed internal-marking side colours; outer boundaries excluded.",
        "settings": {"colour_space": "OpenCV uint8 Lab", "decision_channels": "ab",
                     "distance_units": "Euclidean OpenCV Lab ab units", "minimum_valid_markings": 2,
                     "anchor_dispersion_percentile": 90, "dispersion_multiplier": DISPERSION_MULTIPLIER,
                     "primary_chroma_floor_ab": PRIMARY_FLOOR, "sensitivity_chroma_floors_ab": CHROMA_FLOORS,
                     "boundary_fractions": FRACTIONS.tolist(), "offsets_m_each_side": OFFSETS_M,
                     "minimum_valid_samples_per_side": MIN_PAIRED_SAMPLES,
                     "minimum_valid_paired_locations_on_one_edge": MIN_VALID_LOCATIONS,
                     "minimum_contiguous_contradictions_on_one_edge": MIN_CONTRADICTION_RUN,
                     "decision_note": "Exploratory floor uniformity hypothesis; false positives possible. "
                                      "No reference agreement used for threshold choice."},
        "cases": cases,
    }


if __name__ == "__main__":
    cv2.setNumThreads(1)
    write(OUTPUT, run())
