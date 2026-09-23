"""Measure colour increments on saved court candidates without changing geometry."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.annotator.independent_court import (
    assignment,
    detector,
    paint_geometry,
    stripe_observations,
)
from scratch.court_det_fix.w5_holistic.verifier import prepare_segments

BASE = Path(__file__).resolve().parent
WIDER = BASE.parent / "wider_evaluation/runs/20260922"
WEB = BASE.parent / "edge_polarity/webui_return_colour_consistency"
SVD = BASE.parent / "svd_search"
MARKINGS = assignment.MARKINGS
SAMPLE_COUNT = 32
MIN_WIDTH_PX = 1.5
GREY_CONTRAST = 10.0
MIN_STRONG = 6
MIN_GEOM_SUPPORT = 0.08
DISPLAY_SPREAD_AB = 20.0  # Exploratory OpenCV Lab display flag, never a court acceptance rule.
CHROMATIC_PIXEL_FRACTION = 0.01
CHROMATIC_CHANNEL_TOLERANCE = 4


def read(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(value, allow_nan=False, separators=(",", ":")).encode(), mtime=0))


def sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float32)
    values = cv2.remap(image, points[:, 0], points[:, 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return values.reshape(len(points), -1)


def geometry(corners: list) -> tuple[np.ndarray, np.ndarray]:
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M.astype(np.float32),
                                             np.asarray(corners, np.float32))
    centres = cv2.perspectiveTransform(paint_geometry.CENTRE_SEGMENTS_M.reshape(-1, 1, 2).astype(np.float32),
                                       homography).reshape(12, 2, 2)
    intervals = np.repeat(np.arange(12), 2)
    positions = np.tile([1, 2], 12)
    stripe_edges = paint_geometry.positioned_segments(paint_geometry.CENTRE_SEGMENTS_M, intervals, positions)
    edges = cv2.perspectiveTransform(stripe_edges.reshape(-1, 1, 2).astype(np.float32),
                                     homography).reshape(12, 2, 2, 2)
    return centres, edges


def perpendicular_stripe(centre: np.ndarray, edge_pair: np.ndarray, fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect projected edge lines with centre-line normals at each sample."""
    start, end = centre
    middle = start + fraction * (end - start)
    along = end - start
    normal = np.array([-along[1], along[0]], dtype=np.float32) / np.linalg.norm(along)
    offsets = []
    for edge in edge_pair:
        edge_start, edge_end = edge
        direction = edge_end - edge_start
        line_xy = np.array([-direction[1], direction[0]], dtype=np.float32)
        denominator = float(line_xy @ normal)
        if abs(denominator) < 1e-8:
            offsets.append(np.full(len(middle), np.nan, dtype=np.float32))
        else:
            offsets.append(-((middle - edge_start) @ line_xy) / denominator)
    widths = np.abs(offsets[1] - offsets[0])
    return middle, np.broadcast_to(normal, middle.shape), widths


def spread(samples: list[dict], key: str) -> float | None:
    if not samples:
        return None
    values = np.asarray([row[key][1:] for row in samples], dtype=float)
    centre = np.median(values, axis=0)
    return round(float(np.percentile(np.linalg.norm(values - centre, axis=1), 90)), 2)


def measure_marking(grey: np.ndarray, lab: np.ndarray, centres: np.ndarray,
                    edges: np.ndarray, intervals: tuple[int, ...], geom_support: float,
                    assigned_count: int, boxes: np.ndarray) -> dict:
    rows = []
    height, width = grey.shape
    for interval in intervals:
        fraction = np.linspace(0.08, 0.92, SAMPLE_COUNT, dtype=np.float32)[:, None]
        middle, normal, widths = perpendicular_stripe(centres[interval], edges[interval], fraction)
        valid_width = np.isfinite(widths) & (widths >= MIN_WIDTH_PX)
        for point, direction, stripe_width, wide_enough in zip(middle, normal, widths, valid_width, strict=True):
            if not wide_enough:
                rows.append({"state": "unresolved_width",
                             "width_px": float(stripe_width) if np.isfinite(stripe_width) else None})
                continue
            candidates = point + np.array([-0.4, 0.0, 0.4], np.float32)[:, None] * stripe_width * direction
            floor_minus = candidates - 1.5 * stripe_width * direction
            floor_plus = candidates + 1.5 * stripe_width * direction
            locations = np.concatenate([candidates, floor_minus, floor_plus])
            if len(boxes):
                covered = ((locations[:, None, 0] >= boxes[None, :, 0])
                           & (locations[:, None, 0] <= boxes[None, :, 2])
                           & (locations[:, None, 1] >= boxes[None, :, 1])
                           & (locations[:, None, 1] <= boxes[None, :, 3]))
                if covered.any():
                    rows.append({"state": "occluded", "width_px": float(stripe_width)})
                    continue
            inside = (locations[:, 0] >= 0) & (locations[:, 0] < width - 1)
            inside &= (locations[:, 1] >= 0) & (locations[:, 1] < height - 1)
            if not inside.all():
                rows.append({"state": "outside", "width_px": float(stripe_width)})
                continue
            values = sample(grey, locations).reshape(3, 3)
            contrast = np.minimum(values[0] - values[1], values[0] - values[2])
            best = int(np.argmax(contrast))
            if contrast[best] < GREY_CONTRAST:
                rows.append({"state": "weak_ridge", "width_px": float(stripe_width),
                             "grey_contrast": float(contrast[best])})
                continue
            paint = sample(lab, candidates[best:best + 1])[0]
            floor_left = sample(lab, floor_minus[best:best + 1])[0]
            floor_right = sample(lab, floor_plus[best:best + 1])[0]
            floor = (floor_left + floor_right) / 2
            rows.append({"state": "supported", "width_px": float(stripe_width),
                         "grey_contrast": float(contrast[best]),
                         "paint_lab": paint.round(2).tolist(),
                         "floor_minus_lab": floor_left.round(2).tolist(),
                         "floor_plus_lab": floor_right.round(2).tolist(),
                         "delta_lab": (paint - floor).round(2).tolist(),
                         "point": candidates[best].round(2).tolist()})
    state_counts = dict(Counter(row["state"] for row in rows))
    usable = [row for row in rows if row["state"] == "supported"]
    supported = geom_support >= MIN_GEOM_SUPPORT and assigned_count > 0 and len(usable) >= MIN_STRONG
    signatures = {}
    for key in ("paint_lab", "floor_minus_lab", "floor_plus_lab", "delta_lab"):
        signatures[f"median_{key}"] = (
            np.median([row[key] for row in usable], axis=0).round(2).tolist() if usable else None
        )
    return {"geom_support": round(geom_support, 4), "assigned_fragments": assigned_count,
            "sample_states": state_counts, "supported": bool(supported),
            "raw_paint_spread_ab": spread(usable, "paint_lab"),
            "relative_spread_ab": spread(usable, "delta_lab"),
            **signatures, "samples": rows}


def comparisons(markings: dict, chromatic: bool) -> dict:
    result = {}
    for name, target in markings.items():
        references = {other: row for other, row in markings.items()
                      if other != name and row["supported"]}
        if not target["supported"]:
            status = "missing_target_support"
        elif not references:
            status = "missing_reference_support"
        elif not chromatic:
            status = "greyscale_only"
        else:
            status = "evaluated"
        relative_distances = {}
        raw_distances = {}
        reference_spreads = {"relative": None, "raw_paint": None}
        if status == "evaluated":
            for key, distances, label in (("median_delta_lab", relative_distances, "relative"),
                                          ("median_paint_lab", raw_distances, "raw_paint")):
                target_ab = np.asarray(target[key][1:])
                for other, row in references.items():
                    ref_ab = np.asarray(row[key][1:])
                    distances[other] = round(float(np.linalg.norm(target_ab - ref_ab)), 2)
                signatures = [np.asarray(row[key][1:]) for row in references.values()]
                if len(signatures) >= 2:
                    reference_spreads[label] = round(max(
                        float(np.linalg.norm(left - right))
                        for index, left in enumerate(signatures)
                        for right in signatures[index + 1:]
                    ), 2)
        ambiguous = (status == "evaluated" and (
            len(references) == 1
            or target["raw_paint_spread_ab"] >= DISPLAY_SPREAD_AB
            or target["relative_spread_ab"] >= DISPLAY_SPREAD_AB
            or any(row["raw_paint_spread_ab"] >= DISPLAY_SPREAD_AB
                   or row["relative_spread_ab"] >= DISPLAY_SPREAD_AB for row in references.values())
            or any(value is not None and value >= DISPLAY_SPREAD_AB for value in reference_spreads.values())
        ))
        result[name] = {"status": status, "reference_markings": list(references),
                        "relative_ab_distances": relative_distances,
                        "raw_paint_ab_distances": raw_distances,
                        "nearest_relative_ab": min(relative_distances.values()) if relative_distances else None,
                        "nearest_raw_paint_ab": min(raw_distances.values()) if raw_distances else None,
                        "reference_spread_ab": reference_spreads,
                        "ambiguous_reference": bool(ambiguous)}
    return result


def measure_case(item: dict, record: dict, pack_case: dict, image: np.ndarray,
                 extra_candidates: dict[str, tuple[str | None, dict]], sampling_space: str) -> dict:
    size = tuple(record["working_size"])
    native_size = (image.shape[1], image.shape[0])
    if tuple(record["native_size"]) != native_size:
        raise ValueError(f"{item['case_id']}: native image size differs from saved record")
    frame = image if sampling_space == "native" else cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    channel_range = image.max(axis=2).astype(np.int16) - image.min(axis=2).astype(np.int16)
    chromatic_fraction = float(np.mean(channel_range > CHROMATIC_CHANNEL_TOLERANCE))
    chromatic = chromatic_fraction >= CHROMATIC_PIXEL_FRACTION
    segments, _, prepared_size = prepare_segments(pack_case)
    if prepared_size != size:
        raise ValueError(f"{item['case_id']}: working size differs from frozen preparation")
    observations = assignment.prepare_observations(segments, size)
    weights = stripe_observations.fragment_weights(observations)
    sample_scale = np.asarray(frame.shape[1::-1], dtype=float) / np.asarray(native_size, dtype=float)
    box_scale = np.tile(sample_scale, 2)
    boxes = np.asarray(pack_case.get("bbox_px", []), dtype=float).reshape(-1, 4) * box_scale
    outputs = {}
    measured_geometry = {}
    selections = {
        "saved_w5": (record["selections"]["full"]["gated"], record["candidates"]),
        "saved_g1_templates": (record["selections"]["g1_templates"]["gated"], record["candidates"]),
    }
    selections.update(extra_candidates)
    for source, (key, candidates) in selections.items():
        candidate = candidates.get(key) if key else None
        if candidate is None:
            outputs[source] = {"status": "no_saved_candidate", "selected_key": key}
            continue
        corners = candidate["corners_px"]
        native_corners = np.asarray(corners, dtype=float)
        scale = np.asarray(size, dtype=float) / np.asarray(native_size, dtype=float)
        working_corners = (native_corners * scale).tolist()
        cache_key = native_corners.tobytes()
        if cache_key in measured_geometry:
            outputs[source] = {**measured_geometry[cache_key], "selected_key": key}
            continue
        working_centres, working_edges = geometry(working_corners)
        sample_centres, sample_edges = geometry((native_corners * sample_scale).tolist())
        homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M.astype(np.float32),
                                                 np.asarray(working_corners, np.float32))
        evidence = stripe_observations.measure(homography, observations, size,
                                                centres=paint_geometry.CENTRE_SEGMENTS_M)
        score = stripe_observations.score_model(evidence, weights, 3)
        assigned = np.asarray(score["assignments"]["marking"])
        marking_rows = {}
        for index, name in enumerate(MARKINGS):
            marking_rows[name] = measure_marking(grey, lab, sample_centres, sample_edges,
                                                 assignment.MARKING_INTERVALS[index],
                                                 score["exclusive_per_marking"][index],
                                                 int(np.sum(assigned == index)), boxes)
        for row in marking_rows.values():
            for sample_row in row["samples"]:
                if sample_row["state"] == "supported":
                    sample_row["point"] = (np.asarray(sample_row["point"]) * scale / sample_scale).round(2).tolist()
        result = {"status": "measured", "selected_key": key,
                  "corners_working_px": working_corners, "centres_working_px": working_centres.tolist(),
                  "edges_working_px": working_edges.reshape(24, 2, 2).tolist(),
                  "markings": marking_rows, "comparisons": comparisons(marking_rows, chromatic)}
        measured_geometry[cache_key] = result
        outputs[source] = result
    return {"case_id": item["case_id"], "group": item["group"],
            "view_status": item.get("view_status"), "control_label": record.get("control_label"),
            "image": item["image"], "working_size": size, "sampling_space": sampling_space,
            "chromatic_fraction": round(chromatic_fraction, 5), "chromatic": chromatic,
            "sources": outputs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append")
    parser.add_argument("--output", type=Path, default=BASE / "measurements.json.gz")
    parser.add_argument("--sampling-space", choices=("working", "native"), default="native")
    args = parser.parse_args()
    manifest = read(WIDER / "manifest.json.gz")["cases"]
    numeric = {row["case_id"]: row for row in read(WIDER / "numeric_fit.json.gz")["cases"]}
    packs = {}
    cases = []
    for item in manifest:
        case_id = item["case_id"]
        if args.case and case_id not in args.case:
            continue
        pack_relative = item.get("observation_pack", item["source_pack"])
        pack_path = ROOT / "scratch/court_det_fix" / pack_relative
        if pack_path not in packs:
            packs[pack_path] = {row["id"]: row for row in read(pack_path)["cases"]}
        image = cv2.imread(str(ROOT / "scratch/court_det_fix" / item["image"]))
        if image is None:
            raise FileNotFoundError(item["image"])
        extra_candidates = {}
        for arm in ("baseline", "deeper", "shortlist"):
            summary_path = SVD / "run_20260923/cases" / arm / f"{case_id}.json.gz"
            if summary_path.exists():
                summary = read(summary_path)
                extra_candidates[f"svd_{arm}"] = (
                    summary["selected_key"],
                    {candidate["origin_key"]: candidate for candidate in summary["candidates"]},
                )
                refit = summary.get("oracle_refitted")
                if refit is not None:
                    extra_candidates[f"svd_{arm}_reference_best_refit_retrospective"] = (
                        refit["origin_key"],
                        {candidate["origin_key"]: candidate for candidate in summary["candidates"]},
                    )
        cases.append(measure_case(item, numeric[case_id], packs[pack_path][case_id], image,
                                  extra_candidates, args.sampling_space))
        print(case_id, flush=True)
    write(args.output, {"schema": "colour-consistency/2", "cases": cases,
                        "settings": {"sample_count_per_interval": SAMPLE_COUNT,
                                     "sampling_space": args.sampling_space,
                                     "min_projected_width_px": MIN_WIDTH_PX,
                                     "display_spread_ab": DISPLAY_SPREAD_AB,
                                     "chromatic_pixel_fraction": CHROMATIC_PIXEL_FRACTION,
                                     "chromatic_channel_tolerance": CHROMATIC_CHANNEL_TOLERANCE,
                                     "grey_contrast": GREY_CONTRAST,
                                     "min_strong_samples": MIN_STRONG,
                                     "min_geom_support": MIN_GEOM_SUPPORT}})


if __name__ == "__main__":
    main()
