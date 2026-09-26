#!/usr/bin/env python3
"""Bounded post-fit audit for court-boundary evidence versus net tape.

The script keeps the frozen parent, fit and fragment membership unchanged.  It
performs two local checks on the supplied Am1 and GX images:

1. reruns the previously proposed outward-attached-mesh falsifier; and
2. evaluates an exploratory same-paint chroma veto for fragments already
   assigned to the far baseline.

The chroma veto is deliberately one-sided: absence of far-baseline fragments is
not a rejection, so a cropped/partial court remains admissible.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
COURT_CORNERS_M = np.array(
    [[0.0, 0.0], [COURT_WIDTH_M, 0.0],
     [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
    dtype=np.float32,
)
INTERVAL_TO_MARKING = {
    0: "left_doubles", 1: "left_singles", 2: "centre", 3: "centre",
    4: "right_singles", 5: "right_doubles", 6: "far_baseline",
    7: "far_long_service", 8: "far_short_service", 9: "near_short_service",
    10: "near_long_service", 11: "near_baseline",
}


def load_packet(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def bilinear(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    values = cv2.remap(
        image.astype(np.float32),
        points[:, 0].reshape(-1, 1),
        points[:, 1].reshape(-1, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if image.ndim == 2:
        return values.reshape(len(points))
    return values.reshape(len(points), image.shape[2])


def project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    mapped = np.c_[points, np.ones(len(points))] @ H.T
    return mapped[:, :2] / mapped[:, 2:]


def homography(corners: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(COURT_CORNERS_M, np.asarray(corners, np.float32)).astype(float)


def fragment_intervals(case: dict) -> dict[int, int]:
    constraints = case["constraints_original"]
    by_fragment: dict[int, list[int]] = {}
    for fragment_id, interval in zip(constraints["fragment_ids"], constraints["intervals"], strict=True):
        by_fragment.setdefault(int(fragment_id), []).append(int(interval))
    return {
        fragment_id: Counter(intervals).most_common(1)[0][0]
        for fragment_id, intervals in by_fragment.items()
    }


def ridge_signature(
    image: np.ndarray,
    segment: np.ndarray,
    *,
    sample_count: int = 24,
    offsets: tuple[float, ...] = (-4.0, -2.0, 0.0, 2.0, 4.0),
    side_distance: float = 6.0,
    contrast_floor: float = 10.0,
) -> dict:
    """Mirror the existing five-centre, two-side greyscale ridge measurement.

    Additional output is the Lab colour increment at the winning centre:
    centre colour minus the mean of the two side colours.
    """
    p0, p1 = np.asarray(segment, dtype=float)
    vector = p1 - p0
    length = float(np.linalg.norm(vector))
    direction = vector / length
    normal = np.array([-direction[1], direction[0]])
    fractions = np.linspace(0.0, 1.0, sample_count)
    base = p0[None] + fractions[:, None] * vector[None]
    offsets_array = np.asarray(offsets, dtype=float)
    centres = base[:, None] + offsets_array[None, :, None] * normal
    minus = centres - side_distance * normal
    plus = centres + side_distance * normal

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    centre_grey = bilinear(grey, centres.reshape(-1, 2)).reshape(sample_count, len(offsets))
    minus_grey = bilinear(grey, minus.reshape(-1, 2)).reshape(sample_count, len(offsets))
    plus_grey = bilinear(grey, plus.reshape(-1, 2)).reshape(sample_count, len(offsets))
    contrast = np.minimum(centre_grey - minus_grey, centre_grey - plus_grey)
    best_index = contrast.argmax(axis=1)
    best_contrast = contrast[np.arange(sample_count), best_index]
    best_points = centres[np.arange(sample_count), best_index]
    keep = best_contrast >= contrast_floor
    if not keep.any():
        return {
            "strong_samples": 0,
            "strong_fraction": 0.0,
            "median_contrast": None,
            "median_delta_lab": None,
            "sample_points": [],
        }

    selected = best_points[keep]
    selected_normal = np.repeat(normal[None], len(selected), axis=0)
    centre_bgr = bilinear(image, selected)
    side_bgr = (
        bilinear(image, selected - side_distance * selected_normal)
        + bilinear(image, selected + side_distance * selected_normal)
    ) / 2.0
    centre_lab = cv2.cvtColor(
        np.clip(centre_bgr, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB,
    ).reshape(-1, 3).astype(float)
    side_lab = cv2.cvtColor(
        np.clip(side_bgr, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB,
    ).reshape(-1, 3).astype(float)
    delta_lab = centre_lab - side_lab
    return {
        "strong_samples": int(keep.sum()),
        "strong_fraction": float(keep.mean()),
        "median_contrast": float(np.median(best_contrast[keep])),
        "median_delta_lab": np.median(delta_lab, axis=0).tolist(),
        "sample_points": selected.tolist(),
    }


def paint_consistency_veto(case: dict, image_native: np.ndarray) -> dict:
    """Reject a supported far boundary whose paint increment is an outlier.

    Reference fragments are other transverse markings (intervals 7..11).  The
    rule is not evaluated unless both groups contain at least three fragments,
    each with at least eight strong ridge samples.  This means an unsupported or
    cropped far boundary is not rejected.
    """
    working_size = tuple(int(value) for value in case["working_size"])
    image = cv2.resize(image_native, working_size, interpolation=cv2.INTER_AREA)
    intervals = fragment_intervals(case)
    rows = []
    for fragment_id, segment in zip(
        case["retained_observations"]["fragment_ids"],
        case["retained_observations"]["segments"],
        strict=True,
    ):
        interval = intervals[int(fragment_id)]
        signature = ridge_signature(image, np.asarray(segment, dtype=float))
        rows.append({
            "fragment_id": int(fragment_id),
            "interval": interval,
            "marking": INTERVAL_TO_MARKING[interval],
            **signature,
        })

    usable = [row for row in rows if row["strong_samples"] >= 8]
    boundary = [row for row in usable if row["interval"] == 6]
    reference = [row for row in usable if 7 <= row["interval"] <= 11]
    if len(boundary) < 3:
        return {
            "decision": "no_rejection",
            "status": "untestable_no_supported_far_boundary",
            "usable_boundary_fragments": len(boundary),
            "usable_reference_fragments": len(reference),
            "fragments": rows,
        }
    if len(reference) < 3:
        return {
            "decision": "no_rejection",
            "status": "untestable_no_transverse_paint_reference",
            "usable_boundary_fragments": len(boundary),
            "usable_reference_fragments": len(reference),
            "fragments": rows,
        }

    boundary_ab = np.asarray([row["median_delta_lab"][1:] for row in boundary], dtype=float)
    reference_ab = np.asarray([row["median_delta_lab"][1:] for row in reference], dtype=float)
    boundary_median = np.median(boundary_ab, axis=0)
    reference_median = np.median(reference_ab, axis=0)
    separation = float(np.linalg.norm(boundary_median - reference_median))
    reference_distances = np.linalg.norm(reference_ab - reference_median, axis=1)
    robust_scale = float(1.4826 * np.median(reference_distances))

    # Engineering demonstration only; not calibrated from two views.
    reject = separation > 20.0 and separation > 4.0 * max(robust_scale, 1e-6)
    return {
        "decision": "reject_paint_mismatch" if reject else "no_rejection",
        "status": "evaluated",
        "usable_boundary_fragments": len(boundary),
        "usable_reference_fragments": len(reference),
        "boundary_fragment_ids": [row["fragment_id"] for row in boundary],
        "reference_fragment_ids": [row["fragment_id"] for row in reference],
        "boundary_median_delta_ab": boundary_median.tolist(),
        "reference_median_delta_ab": reference_median.tolist(),
        "opponent_colour_separation": separation,
        "reference_robust_scale": robust_scale,
        "separation_over_scale": separation / max(robust_scale, 1e-6),
        "rule": "separation > 20 and separation > 4 * reference_robust_scale",
        "fragments": rows,
    }


def nearest_polyline(points: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = points[:, None] - line[None]
    squared = np.einsum("...d,...d->...", delta, delta)
    index = squared.argmin(axis=1)
    return np.sqrt(squared[np.arange(len(points)), index]), index


def attached_mesh_test(image_native: np.ndarray, corners: np.ndarray, working_size: tuple[int, int]) -> dict:
    """Rerun the previously proposed outward attached-mesh falsifier."""
    width, height = working_size
    image = cv2.resize(image_native, (width, height), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H = homography(corners)
    x = np.linspace(0.05 * COURT_WIDTH_M, 0.95 * COURT_WIDTH_M, 256)
    outside_edge = project(H, np.c_[x, np.zeros_like(x)])
    interior = project(H, np.c_[x, np.full_like(x, 0.25)])
    inward = interior - outside_edge
    inward_length = np.linalg.norm(inward, axis=1)
    valid = np.isfinite(inward_length) & (inward_length > 1e-6)
    outward = np.zeros_like(inward)
    outward[valid] = -inward[valid] / inward_length[valid, None]
    in_frame = ((outside_edge >= 0) & (outside_edge < np.array([width, height]))).all(axis=1) & valid
    visible = np.flatnonzero(in_frame)
    if len(visible) < 32:
        return {"decision": "untestable", "reason": "insufficient_visible_far_boundary"}
    visible_span = float(np.linalg.norm(outside_edge[visible[-1]] - outside_edge[visible[0]]))
    if visible_span < 80.0:
        return {"decision": "untestable", "reason": "visible_span_under_80px", "visible_span_px": visible_span}

    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(grey)[0]
    if detected is None:
        return {"decision": "no_contradiction", "visible_span_px": visible_span, "attachment_count": 0}
    segments = detected.reshape(-1, 2, 2).astype(float)
    attachments = []
    cosine_limit = float(np.cos(np.deg2rad(15.0)))
    for segment in segments:
        if np.linalg.norm(segment[1] - segment[0]) < 12.0:
            continue
        distance, boundary_index = nearest_polyline(segment, outside_edge)
        endpoint = int(distance.argmin())
        if distance[endpoint] > 5.0:
            continue
        index = int(boundary_index[endpoint])
        if not in_frame[index]:
            continue
        vector = segment[1 - endpoint] - segment[endpoint]
        vector_length = float(np.linalg.norm(vector))
        if vector_length < 12.0:
            continue
        normal = outward[index]
        alignment = float(np.dot(vector / vector_length, normal))
        extent = float(np.dot(vector, normal))
        lateral = abs(float(normal[0] * vector[1] - normal[1] * vector[0]))
        if alignment < cosine_limit or extent < 12.0 or extent > 45.0:
            continue
        if lateral > extent * np.tan(np.deg2rad(15.0)):
            continue
        attachments.append((index / (len(outside_edge) - 1), extent, segment.tolist()))

    attachments.sort(key=lambda row: row[0])
    merged = []
    for row in attachments:
        if not merged or (row[0] - merged[-1][0]) * visible_span >= 8.0:
            merged.append(row)
        elif row[1] > merged[-1][1]:
            merged[-1] = row
    span_fraction = merged[-1][0] - merged[0][0] if len(merged) >= 2 else 0.0
    reject = len(merged) >= 3 and span_fraction >= 0.25
    return {
        "decision": "reject_net_attached" if reject else "no_contradiction",
        "visible_span_px": visible_span,
        "attachment_count": len(merged),
        "attachment_span_fraction": span_fraction,
        "attachments": [
            {"baseline_fraction": float(position), "outward_extent_px": float(extent), "segment": segment}
            for position, extent, segment in merged
        ],
    }


def render_overlay(case: dict, image_native: np.ndarray, result: dict, output: Path) -> None:
    working_size = tuple(int(value) for value in case["working_size"])
    image = cv2.resize(image_native, working_size, interpolation=cv2.INTER_AREA)
    corners = np.asarray(case["fits"]["strong_centres"]["corners_px"], dtype=float)
    cv2.line(image, tuple(np.rint(corners[0]).astype(int)), tuple(np.rint(corners[1]).astype(int)), (0, 0, 255), 2)
    intervals = fragment_intervals(case)
    boundary_ids = set(result.get("boundary_fragment_ids", []))
    reference_ids = set(result.get("reference_fragment_ids", []))
    for fragment_id, segment in zip(
        case["retained_observations"]["fragment_ids"], case["retained_observations"]["segments"], strict=True,
    ):
        fragment_id = int(fragment_id)
        if fragment_id in boundary_ids:
            colour = (0, 255, 255)
        elif fragment_id in reference_ids:
            colour = (255, 255, 0)
        else:
            continue
        p0, p1 = np.rint(np.asarray(segment)).astype(int)
        cv2.line(image, tuple(p0), tuple(p1), colour, 2)
        midpoint = tuple(np.rint((p0 + p1) / 2).astype(int))
        cv2.putText(image, str(fragment_id), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("--am1-image", type=Path, required=True)
    parser.add_argument("--gx-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path)
    args = parser.parse_args()

    packet = load_packet(args.packet)
    image_paths = {
        "am1_window_00_frame_54": args.am1_image,
        "gxBQ_window_00_frame_5": args.gx_image,
    }
    output = {
        "schema": "boundary-paint-followup/1",
        "source_revision": packet["source_revision"],
        "checks": {},
    }
    for case in packet["cases"]:
        case_id = case["case_id"]
        image = cv2.imread(str(image_paths[case_id]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_paths[case_id])
        corners = np.asarray(case["fits"]["strong_centres"]["corners_px"], dtype=float)
        working_size = tuple(int(value) for value in case["working_size"])
        paint = paint_consistency_veto(case, image)
        mesh = attached_mesh_test(image, corners, working_size)
        output["checks"][case_id] = {
            "attached_mesh": mesh,
            "paint_consistency": paint,
        }
        if args.overlay_dir is not None:
            render_overlay(case, image, paint, args.overlay_dir / f"{case_id}_paint_overlay.png")

    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
