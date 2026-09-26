#!/usr/bin/env python3
"""Bounded Am1/GX court-vs-net audit.

This script does not search candidates or change assignments. It reads the frozen
case packet, reports the recorded fixed-membership refits, demonstrates the
projective depth ambiguity, and runs one negative-control calculation: support
for the physical net projected by the accepted camera model using only the
packet's retained fragments.

If the two source PNGs are supplied, --mesh-test runs the proposed local
outward attached-mesh contradiction test on the already-fitted far boundary.
It is a falsifier only: a positive result rejects a boundary as net tape; a
negative/untestable result does not establish that the boundary is court paint.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
COURT_CORNERS_M = np.array(
    [[0.0, 0.0], [COURT_WIDTH_M, 0.0],
     [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
    dtype=np.float32,
)
MARKINGS = (
    "left_doubles", "left_singles", "centre", "right_singles", "right_doubles",
    "far_baseline", "far_long_service", "far_short_service",
    "near_short_service", "near_long_service", "near_baseline",
)
INTERVAL_TO_MARKING = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4,
                       6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10}
MESH_TEST_PREDICTION = {
    "am1_window_00_frame_54": "reject: repeated mesh attaches to the outward side",
    "gxBQ_window_00_frame_5": "retain/no contradiction: no outward attached mesh",
}
PROJECTED_NET_CONTROL_PREDICTION = {
    "am1_window_00_frame_54": "low retained-fragment support",
    "gxBQ_window_00_frame_5": "high retained-fragment support",
}


def load_packet(path: Path) -> tuple[dict, bytes, bytes]:
    raw = path.read_bytes()
    payload = gzip.decompress(raw) if path.suffix == ".gz" else raw
    return json.loads(payload), raw, payload


def project(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    mapped = np.c_[points, np.ones(len(points))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return mapped[:, :2] / mapped[:, 2:]


def homography_from_corners(corners: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(COURT_CORNERS_M, np.asarray(corners, np.float32)).astype(float)


def camera(corners: np.ndarray, size: tuple[int, int]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Exact bounded camera diagnostic used by the pinned helper."""
    width, height = size
    transform = homography_from_corners(corners)
    focal = np.geomspace(0.4 * width, 4.0 * width, 200)
    axes = np.broadcast_to(transform[:, :2], (len(focal), 3, 2)).copy()
    axes[:, :2] -= np.array([width / 2, height / 2])[None, :, None] * axes[:, 2:3]
    axes[:, :2] /= focal[:, None, None]
    norms = np.linalg.norm(axes, axis=1)
    cosine = (axes[:, :, 0] * axes[:, :, 1]).sum(axis=1) / np.prod(norms, axis=1)
    ratio = np.log(norms[:, 0] / norms[:, 1])
    error = np.hypot(cosine, ratio)
    best = int(error.argmin())
    intrinsic = np.array(
        [[focal[best], 0.0, width / 2], [0.0, focal[best], height / 2], [0.0, 0.0, 1.0]]
    )
    pose = np.linalg.inv(intrinsic) @ transform
    pose /= np.linalg.norm(pose[:, :2], axis=0).mean()
    vertical = np.cross(pose[:, 0], pose[:, 1])
    vertical /= np.linalg.norm(vertical)
    return float(error[best]), pose, intrinsic, vertical


def projected_net(corners: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    error, pose, intrinsic, vertical = camera(corners, size)
    ground = np.array([[0.0, 6.7, 1.0], [3.05, 6.7, 1.0], [6.1, 6.7, 1.0]]) @ pose.T
    top = ground - np.array([1.55, 1.524, 1.55])[:, None] * vertical
    ground_px = ground @ intrinsic.T
    top_px = top @ intrinsic.T
    ground_px = ground_px[:, :2] / ground_px[:, 2:]
    top_px = top_px[:, :2] / top_px[:, 2:]
    segments = np.array(
        [[top_px[0], top_px[1]], [top_px[1], top_px[2]],
         [ground_px[0], top_px[0]], [ground_px[2], top_px[2]]]
    )
    return segments, error


def visible_samples(endpoints: np.ndarray, size: tuple[int, int], count: int = 48):
    endpoints = np.asarray(endpoints, dtype=float)[None]
    starts = endpoints[:, :, 0]
    vectors = endpoints[:, :, 1] - starts
    lower = np.zeros(starts.shape[:2])
    upper = np.ones(starts.shape[:2])
    visible = np.ones(starts.shape[:2], dtype=bool)
    for axis, limit in enumerate(size):
        stationary = np.abs(vectors[..., axis]) < 1e-8
        visible &= ~stationary | ((starts[..., axis] >= 0) & (starts[..., axis] <= limit - 1))
        divisor = np.where(stationary, 1.0, vectors[..., axis])
        first = -starts[..., axis] / divisor
        last = (limit - 1 - starts[..., axis]) / divisor
        lower = np.maximum(lower, np.where(stationary, -np.inf, np.minimum(first, last)))
        upper = np.minimum(upper, np.where(stationary, np.inf, np.maximum(first, last)))
    visible &= upper > lower
    clipped_length = (upper - lower) * np.linalg.norm(vectors, axis=-1)
    visible &= clipped_length >= 12.0
    fractions = lower[..., None] + (upper - lower)[..., None] * np.linspace(0.0, 1.0, count)
    samples = starts[..., None, :] + fractions[..., None] * vectors[..., None, :]
    return samples[0], visible[0], clipped_length[0]


def retained_net_support(segments: np.ndarray, net: np.ndarray, size: tuple[int, int]):
    """Negative control mirroring the helper's Gaussian distance support.

    It deliberately uses only the retained packet fragments. This is not the
    proposed decision rule and cannot certify absence of a net.
    """
    width, height = size
    mask = np.full((height, width), 255, dtype=np.uint8)
    for segment in np.rint(segments).astype(int):
        cv2.line(mask, tuple(segment[0]), tuple(segment[1]), 0, 1)
    distances_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    samples, visible, lengths = visible_samples(net, size, 48)
    pixel_x = np.clip(samples[..., 0], 0, width - 1).astype(int)
    pixel_y = np.clip(samples[..., 1], 0, height - 1).astype(int)
    distances = distances_map[pixel_y, pixel_x]
    support = np.exp(-0.5 * (distances / 4.0) ** 2).mean(axis=-1) * visible
    score = float(support.sum() / max(int(visible.sum()), 1))
    return score, support, visible, lengths, distances


def point_to_polyline_distance(points: np.ndarray, line_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest sampled boundary point and distance for each point."""
    delta = points[:, None, :] - line_points[None, :, :]
    d2 = np.einsum("...d,...d->...", delta, delta)
    index = d2.argmin(axis=1)
    return np.sqrt(d2[np.arange(len(points)), index]), index


def attached_mesh_test(
    image_native: np.ndarray,
    corners_working: np.ndarray,
    working_size: tuple[int, int],
) -> dict:
    """Local outward attached-mesh contradiction on the fitted far boundary.

    Protocol constants are deliberately fixed before seeing either result:
      * ignore the endpoint 5% of the baseline;
      * segment touches the projected outside paint edge within 5 working px;
      * it extends at least 12 px outward and is within 15 degrees of the local
        outward normal;
      * reject only for >=3 distinct attachments spanning >=25% of the tested
        baseline. Insufficient visible baseline returns untestable.

    These constants are a bounded protocol, not a calibrated production gate.
    """
    width, height = working_size
    image = cv2.resize(image_native, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H = homography_from_corners(corners_working)

    x = np.linspace(0.05 * COURT_WIDTH_M, 0.95 * COURT_WIDTH_M, 256)
    outside_edge = project(H, np.c_[x, np.zeros_like(x)])
    interior = project(H, np.c_[x, np.full_like(x, 0.25)])
    inward = interior - outside_edge
    inward_norm = np.linalg.norm(inward, axis=1)
    valid_norm = np.isfinite(inward_norm) & (inward_norm > 1e-6)
    outward = np.zeros_like(inward)
    outward[valid_norm] = -inward[valid_norm] / inward_norm[valid_norm, None]

    in_frame = ((outside_edge >= 0) & (outside_edge < np.array([width, height]))).all(axis=1) & valid_norm
    visible_index = np.flatnonzero(in_frame)
    if len(visible_index) < 32:
        return {"decision": "untestable", "reason": "insufficient_visible_far_boundary"}
    visible_span = float(np.linalg.norm(outside_edge[visible_index[-1]] - outside_edge[visible_index[0]]))
    if visible_span < 80.0:
        return {"decision": "untestable", "reason": "visible_span_under_80px", "visible_span_px": visible_span}

    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(gray)[0]
    if detected is None:
        return {"decision": "no_contradiction", "attachments": [], "visible_span_px": visible_span}
    segments = detected.reshape(-1, 2, 2).astype(float)
    attachments: list[tuple[float, float, list[list[float]]]] = []
    cosine_limit = float(np.cos(np.deg2rad(15.0)))
    for segment in segments:
        length = float(np.linalg.norm(segment[1] - segment[0]))
        if length < 12.0:
            continue
        endpoint_distance, endpoint_index = point_to_polyline_distance(segment, outside_edge)
        attach_endpoint = int(endpoint_distance.argmin())
        if endpoint_distance[attach_endpoint] > 5.0:
            continue
        boundary_i = int(endpoint_index[attach_endpoint])
        if not in_frame[boundary_i]:
            continue
        attach = segment[attach_endpoint]
        other = segment[1 - attach_endpoint]
        vector = other - attach
        vector_length = float(np.linalg.norm(vector))
        if vector_length < 12.0:
            continue
        normal = outward[boundary_i]
        alignment = float(np.dot(vector / vector_length, normal))
        if alignment < cosine_limit:
            continue
        outward_extent = float(np.dot(vector, normal))
        lateral = abs(float(normal[0] * vector[1] - normal[1] * vector[0]))
        if outward_extent < 12.0 or outward_extent > 45.0 or lateral > outward_extent * np.tan(np.deg2rad(15.0)):
            continue
        # Parameter along the tested 5%-95% boundary span.
        s = boundary_i / (len(outside_edge) - 1)
        attachments.append((s, outward_extent, segment.tolist()))

    # Merge near-duplicate LSD edges from one strand in normalized baseline position.
    attachments.sort(key=lambda row: row[0])
    merged = []
    for row in attachments:
        if not merged or (row[0] - merged[-1][0]) * visible_span >= 8.0:
            merged.append(row)
        elif row[1] > merged[-1][1]:
            merged[-1] = row
    span_fraction = (merged[-1][0] - merged[0][0]) if len(merged) >= 2 else 0.0
    reject = len(merged) >= 3 and span_fraction >= 0.25
    return {
        "decision": "reject_net_attached" if reject else "no_contradiction",
        "visible_span_px": visible_span,
        "attachment_count": len(merged),
        "attachment_span_fraction": span_fraction,
        "attachments": [
            {"baseline_fraction": float(s), "outward_extent_px": float(extent), "segment": segment}
            for s, extent, segment in merged
        ],
    }


def assignment_summary(case: dict) -> dict[str, dict[str, object]]:
    constraints = case["constraints_original"]
    intervals = np.asarray(constraints["intervals"], dtype=int)
    fragment_ids = np.asarray(constraints["fragment_ids"], dtype=int)
    rows: dict[str, list[object]] = defaultdict(lambda: [0, set()])
    for interval, fragment_id in zip(intervals, fragment_ids, strict=True):
        marking = MARKINGS[INTERVAL_TO_MARKING[int(interval)]]
        rows[marking][0] += 1
        rows[marking][1].add(int(fragment_id))
    return {
        marking: {"points": int(value[0]), "fragments": sorted(value[1])}
        for marking, value in rows.items()
    }


def depth_family_demo() -> list[dict[str, object]]:
    """G_t fixes y=0 pointwise and preserves x=0 as a line while depth explodes."""
    corners = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]).T
    output = []
    for t in (0.0, -0.5, -0.9, -0.99):
        G = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, t, 1.0]])
        mapped = G @ corners
        mapped = (mapped[:2] / mapped[2:]).T
        output.append({"t": t, "corners": mapped.tolist()})
    return output


def resolve_image(case_id: str, image_args: dict[str, Path | None]) -> Path | None:
    return image_args.get(case_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path, nargs="?", default=Path("cases.json.gz"))
    parser.add_argument("--am1-image", type=Path)
    parser.add_argument("--gx-image", type=Path)
    parser.add_argument("--mesh-test", action="store_true")
    args = parser.parse_args()

    packet, raw, payload = load_packet(args.packet)
    print("packet_schema", packet.get("schema"))
    print("source_revision", packet.get("source_revision"))
    print("gzip_sha256", hashlib.sha256(raw).hexdigest())
    print("json_sha256", hashlib.sha256(payload).hexdigest())
    print("mesh_test_prediction", json.dumps(MESH_TEST_PREDICTION, sort_keys=True))
    print("projected_net_control_prediction", json.dumps(PROJECTED_NET_CONTROL_PREDICTION, sort_keys=True))
    print("depth_family", json.dumps(depth_family_demo(), separators=(",", ":")))

    image_args = {
        "am1_window_00_frame_54": args.am1_image,
        "gxBQ_window_00_frame_5": args.gx_image,
    }
    for case in packet["cases"]:
        case_id = case["case_id"]
        fit = case["fits"]["strong_centres"]
        corners = np.asarray(fit["corners_px"], dtype=float)
        size = tuple(int(value) for value in case["working_size"])
        width, height = size
        measurement = fit.get("measurement") or {}
        outside = (
            (corners[:, 0] < 0) | (corners[:, 0] > width - 1)
            | (corners[:, 1] < 0) | (corners[:, 1] > height - 1)
        )
        print("\ncase", case_id)
        print("assignment_summary", json.dumps(assignment_summary(case), sort_keys=True))
        print("fit", json.dumps({
            "status": fit.get("status"),
            "objective_before": fit.get("objective_before"),
            "objective_after": fit.get("objective_after"),
            "jacobian_rank": fit.get("jacobian_rank"),
            "jacobian_condition": fit.get("jacobian_condition"),
            "minimum_corner_denominator": fit.get("minimum_corner_denominator"),
            "corners_working": corners.tolist(),
            "off_frame_corners": outside.tolist(),
            "paint_score": measurement.get("paint_score"),
            "geometry_score": measurement.get("geometry_score"),
            "gates": measurement.get("gates"),
        }, sort_keys=True))

        net, camera_error = projected_net(corners, size)
        score, per_segment, visible, lengths, distances = retained_net_support(
            np.asarray(case["retained_observations"]["segments"], dtype=float), net, size
        )
        print("retained_projected_net_control", json.dumps({
            "camera_error": camera_error,
            "net_segments": net.tolist(),
            "visible": visible.tolist(),
            "clipped_lengths_px": lengths.tolist(),
            "per_segment_support": per_segment.tolist(),
            "mean_visible_support": score,
            "median_nearest_fragment_px": [
                float(np.median(distances[index])) if visible[index] else None for index in range(4)
            ],
        }, sort_keys=True))

        if args.mesh_test:
            image_path = resolve_image(case_id, image_args)
            if image_path is None:
                print("attached_mesh_test", json.dumps({"decision": "not_run", "reason": "image_path_not_supplied"}))
            else:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    print("attached_mesh_test", json.dumps({"decision": "not_run", "reason": "image_unreadable"}))
                else:
                    print("attached_mesh_test", json.dumps(
                        attached_mesh_test(image, corners, size), sort_keys=True
                    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
