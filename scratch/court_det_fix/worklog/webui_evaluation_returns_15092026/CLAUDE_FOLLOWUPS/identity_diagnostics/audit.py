#!/usr/bin/env python3
"""Offline, read-only identifiability audit of committed badminton-court evidence.

No image scoring, candidate search, direction fitting, score tuning or acceptance
changes are performed. Transformed homographies are mathematical witnesses, not
claims about generated candidates. Uses NumPy and OpenCV; no network access.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np

COMMIT = "7299ff3f74bf51dc3bc59582c79190dbc81666ca"
BLOBS = {
    "measurements.json.gz": "751132ecceba8277fa7c2fac211896c505ddab3a",
    "gx0_control_measurements.json.gz": "4d4da3984eb6b2d61b2aa0e30ab41cc7ab0d8ff5",
}
# Matches the float32 endpoints in court_corners.PAINTED_SEGMENTS_M and
# CORNER_COURT_M, promoted to float64 as in detector.SEGMENTS_M.
CORNERS = np.array([[0, 0], [6.10, 0], [6.10, 13.40], [0, 13.40]], np.float32).astype(float)
SEGMENTS = np.array([
    [[0, 0], [0, 13.40]], [[.46, 0], [.46, 13.40]],
    [[3.05, 0], [3.05, 4.72]], [[3.05, 8.68], [3.05, 13.40]],
    [[5.64, 0], [5.64, 13.40]], [[6.10, 0], [6.10, 13.40]],
    [[0, 0], [6.10, 0]], [[0, .76], [6.10, .76]],
    [[0, 4.72], [6.10, 4.72]], [[0, 8.68], [6.10, 8.68]],
    [[0, 12.64], [6.10, 12.64]], [[0, 13.40], [6.10, 13.40]],
], np.float32).astype(float)
MARKINGS = ["left_doubles", "left_singles", "centre", "right_singles", "right_doubles",
            "far_baseline", "far_long_service", "far_short_service",
            "near_short_service", "near_long_service", "near_baseline"]
INTERVAL_NAMES = ["left_doubles", "left_singles", "centre_far", "centre_near",
                  "right_singles", "right_doubles", "far_baseline", "far_long_service",
                  "far_short_service", "near_short_service", "near_long_service", "near_baseline"]
MARKING_INTERVALS = [(0,), (1,), (2, 3), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,)]
SIZE = (960, 540)
# Only the PNG header was retrieved, not its image pixels.
SCENE19_PNG_HEADER_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAA8AAAAIcCAAAAAAcNpinAAEAAElEQVR42oz9"
    "WY+l2ZIlhpnZHr7hzMdnjykjIuc737pdt6pusbu6m8VqspsSKJDQBAoCCL3o"
)


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def read_gzip(path: Path) -> tuple[dict, dict]:
    data = path.read_bytes()
    sha = git_blob_sha(data)
    if sha != BLOBS[path.name]:
        raise ValueError(f"Git blob mismatch for {path.name}: {sha}")
    value = json.loads(gzip.decompress(data))  # Also checks the gzip trailer.
    return value, {"file": path.name, "bytes": len(data), "git_blob_sha": sha,
                   "sha256": hashlib.sha256(data).hexdigest(), "verified": True}


def project(h: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, float)
    flat = points.reshape(-1, 2)
    q = np.column_stack([flat, np.ones(len(flat))]) @ h.T
    if not np.isfinite(q).all() or np.any(q[:, 2] == 0):
        raise ValueError("Non-finite projection or exactly zero denominator")
    return (q[:, :2] / q[:, 2, None]).reshape(points.shape), q[:, 2].reshape(points.shape[:-1])


def reconstruct_h(corners: np.ndarray) -> np.ndarray:
    """Exact float64 four-point reconstruction, with H33 fixed to one."""
    a, b = [], []
    for (x, y), (u, v) in zip(CORNERS, np.asarray(corners, float), strict=True):
        a.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y], [0, 0, 0, x, y, 1, -v*x, -v*y]])
        b.extend([u, v])
    return np.append(np.linalg.solve(np.asarray(a), np.asarray(b)), 1.).reshape(3, 3)


def clip_segments(endpoints: np.ndarray, size: tuple[int, int] = SIZE) -> dict[str, Any]:
    """Reproduce detector._visible_samples geometry; distinguish short intersections.

    Returns geometric spans, not a claim about occlusion or physical paint.
    The 12-pixel bound is the committed sampling rule, not a new quality cutoff.
    """
    starts = endpoints[:, 0]
    vectors = endpoints[:, 1] - starts
    lower, upper = np.zeros(len(starts)), np.ones(len(starts))
    intersects = np.ones(len(starts), dtype=bool)
    for axis, limit in enumerate(size):
        stationary = np.abs(vectors[:, axis]) < 1e-8
        intersects &= ~stationary | ((starts[:, axis] >= 0) & (starts[:, axis] <= limit-1))
        divisor = np.where(stationary, 1, vectors[:, axis])
        first = -starts[:, axis] / divisor
        last = (limit - 1 - starts[:, axis]) / divisor
        lower = np.maximum(lower, np.where(stationary, -np.inf, np.minimum(first, last)))
        upper = np.minimum(upper, np.where(stationary, np.inf, np.maximum(first, last)))
    intersects &= upper > lower
    raw_length = (upper - lower) * np.linalg.norm(vectors, axis=1)
    visible = intersects & (raw_length >= 12)
    fractions = lower[:, None] + (upper-lower)[:, None] * np.array([0., 1.])
    clipped = starts[:, None, :] + fractions[:, :, None] * vectors[:, None, :]
    reasons = ["available" if vis else "short_intersection" if inter else "no_intersection"
               for vis, inter in zip(visible, intersects, strict=True)]
    return {"visible": visible, "intersects": intersects, "clipped": clipped,
            "length": np.where(intersects, raw_length, 0.), "reasons": reasons}


def camera_components(h: np.ndarray, size: tuple[int, int] = SIZE) -> dict[str, np.ndarray]:
    """The objective and focal grid from net_geometry._camera_from_corners."""
    width, height = size
    focals = np.geomspace(.4*width, 4.*width, 200)
    axes = np.broadcast_to(h[:, :2], (len(focals), 3, 2)).copy()
    axes[:, :2] -= np.array([width/2, height/2])[None, :, None] * axes[:, 2:3]
    axes[:, :2] /= focals[:, None, None]
    norms = np.linalg.norm(axes, axis=1)
    cosine = (axes[:, :, 0]*axes[:, :, 1]).sum(axis=1) / norms.prod(axis=1)
    log_ratio = np.log(norms[:, 0]/norms[:, 1])
    return {"focals": focals, "cosine": cosine, "log_ratio": log_ratio,
            "errors": np.hypot(cosine, log_ratio)}


def source_camera_error(corners: np.ndarray, size: tuple[int, int]) -> float:
    # Exact source float32 OpenCV input conversion, separate from float64 geometry.
    h = cv2.getPerspectiveTransform(CORNERS.astype(np.float32), np.asarray(corners, np.float32)).astype(float)
    return float(camera_components(h, size)["errors"].min())


def constraint_rank(h: np.ndarray, selected_intervals: list[int]) -> dict:
    """Rank of ideal named supporting-line constraints, NOT statistical independence.

    Lines are generated from the frozen H as a mathematical thought experiment;
    this does not infer new pixel correspondences. Each endpoint on each model
    supporting line contributes one homogeneous linear constraint on H.
    """
    normalise_image = np.array([[1/960, 0, -.5], [0, 1/960, -540/1920], [0, 0, 1.]])
    rows = []
    for interval in selected_intervals:
        image, _ = project(h, SEGMENTS[interval])
        line = np.cross(np.append(image[0], 1), np.append(image[1], 1))
        line = np.linalg.inv(normalise_image).T @ line
        line /= np.linalg.norm(line)
        for point in SEGMENTS[interval]:
            row = np.kron(line, np.append(point, 1))
            rows.append(row / np.linalg.norm(row))
    matrix = np.array(rows)
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    return {"rank": rank, "homography_dof_remaining_after_scale": 8-rank,
            "singular_values": singular.tolist(), "selected_interval_indices": selected_intervals}


def interval_records(h: np.ndarray) -> tuple[list[dict], dict]:
    endpoints, _ = project(h, SEGMENTS)
    clipped = clip_segments(endpoints)
    rows = []
    for i, name in enumerate(INTERVAL_NAMES):
        rows.append({"interval_index": i, "interval_name": name,
                     "projected_endpoints_working_px": endpoints[i].tolist(),
                     "observation_state_geometric": clipped["reasons"][i],
                     "clipped_span_working_px": float(clipped["length"][i]),
                     "clipped_endpoints_working_px": clipped["clipped"][i].tolist()
                     if clipped["intersects"][i] else None})
    return rows, clipped


def line_separation(h: np.ndarray, source_interval: int, target_interval: int) -> dict:
    endpoints, _ = project(h, SEGMENTS)
    clipped = clip_segments(endpoints)
    if not clipped["visible"][source_interval]:
        raise ValueError("Source interval has no available span")
    target = endpoints[target_interval]
    v = target[1]-target[0]
    n = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    signed = (clipped["clipped"][source_interval]-target[0]) @ n
    # Linear signed distance along a straight clipped interval, extrema at ends.
    low = 0. if signed.prod() <= 0 else float(np.abs(signed).min())
    high = float(np.abs(signed).max())
    return {"source_interval": INTERVAL_NAMES[source_interval],
            "target_supporting_line": INTERVAL_NAMES[target_interval],
            "clipped_source_endpoints_working_px": clipped["clipped"][source_interval].tolist(),
            "perpendicular_distance_range_working_px": [low, high],
            "note": "Geometric predicted centre-line spacing, not measured stripe separation in pixels."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--inputs", type=Path, default=here/"inputs")
    parser.add_argument("--out", type=Path, default=here/"results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary, summary_prov = read_gzip(args.inputs/"measurements.json.gz")
    controls, control_prov = read_gzip(args.inputs/"gx0_control_measurements.json.gz")
    partial_path = args.inputs/"retrieved_ranking_entries.json"
    detailed = json.loads(partial_path.read_text())
    if detailed["evidence_commit"] != COMMIT:
        raise ValueError("Partial ranking extraction belongs to another commit")
    compact = {(case["case_id"], "automatic_all_camera", e["candidate_id"]): e
               for case in summary["automatic_arms"]["all_camera"]
               for e in case["winners"].values()}
    assert len(compact) == 17
    checks = []
    assignments = []
    geometry_pool = []
    for record in detailed["records"]:
        meta, entry = record["population_metadata"], record["entry"]
        ident = (meta["case_id"], meta["population"], entry["candidate_id"])
        old = compact[ident]
        for small_key, value in [("corners_px", entry["corners_px"]), ("gates", entry["gates"]),
                                 ("stripe_score", entry["stripe"]["exclusive"]["score"]),
                                 ("profile_score", entry["profile"]["score"])]:
            if old[small_key] != value:
                raise ValueError(f"Partial ranking/verified summary mismatch: {ident}/{small_key}")
        h = np.asarray(entry["homography_working"])
        actual_corners, denominator = project(h, CORNERS)
        expected = np.asarray(entry["corners_px"])/2
        corner_residual = float(np.abs(actual_corners-expected).max())
        if corner_residual > 1e-8:  # Numerical self-check, not court acceptance.
            raise ValueError("Saved homography does not reproduce saved corners")
        intervals, clipped = interval_records(h)
        if clipped["visible"].tolist() != entry["profile"]["interval_visible"]:
            raise ValueError("Visibility replay mismatch")
        checks.append({"identity": list(ident), "compact_fields_match": True,
                       "corner_reprojection_max_abs_working_px": corner_residual,
                       "visibility_exact_match": True})
        stripe = entry["stripe"]
        forward_i, forward_e = stripe["independent"]["forward"], stripe["exclusive"]["forward"]
        marking_table = [{"marking": name, "profile": entry["profile"]["marking_ridge"][i],
                          "independent_forward": stripe["independent_per_marking"][i],
                          "exclusive_forward": stripe["exclusive_per_marking"][i]}
                         for i, name in enumerate(MARKINGS)]
        pairs = []
        # Targeted examples, selected by identity rather than a fitted strength cutoff.
        if entry["candidate_id"] == "184:4123":
            selected_pairs = [(9, 10)]
            gap = line_separation(h, 11, 10)
        elif entry["candidate_id"] == "30:33":
            selected_pairs = [(5, 6)]
            gap = line_separation(h, 7, 6)
        else:
            selected_pairs, gap = [], None
        a = stripe["assignments"]
        for chosen, alternative in selected_pairs:
            found = [{"assignment_array_index_zero_based": i,
                      "chosen_marking": MARKINGS[chosen], "chosen_position_index": a["position"][i],
                      "chosen_reverse_strength": a["strength"][i],
                      "alternative_marking": MARKINGS[alternative],
                      "alternative_reverse_strength": a["alternative_strength"][i]}
                     for i, (m, alt) in enumerate(zip(a["marking"], a["alternative_marking"], strict=True))
                     if m == chosen and alt == alternative]
            found.sort(key=lambda x: -x["alternative_reverse_strength"])
            pairs.extend(found[:2])
        assignments.append({"identity": list(ident), "independent_forward": forward_i,
                            "exclusive_forward": forward_e, "forward_loss": forward_i-forward_e,
                            "relative_forward_loss": (forward_i-forward_e)/forward_i,
                            "per_marking": marking_table, "selected_assignment_examples": pairs,
                            "predicted_line_separation": gap})
        geometry_pool.append({"identity": list(ident), "homography_source": "saved detailed ranking record",
                              "intervals": intervals, "floor_score_recorded": entry["gates"]["floor_score"],
                              "positive_corner_denominators": bool((denominator > 0).all()),
                              "ideal_visible_line_constraint_rank": constraint_rank(h, np.flatnonzero(clipped["visible"]).tolist())})

    # Header proves native image dimensions independently of display scaling.
    header = base64.b64decode(SCENE19_PNG_HEADER_B64)
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    dimensions = struct.unpack(">II", header[16:24])
    assert dimensions == SIZE
    scene = "shuttleset_03_scene_0019"
    false = compact[(scene, "automatic_all_camera", "165:6702")]
    saved_profile = next(x["profile"] for x in summary["extreme_paint_winners"] if x["case_id"] == scene)
    h = reconstruct_h(np.asarray(false["corners_px"]))
    projected_corners, denom = project(h, CORNERS)
    intervals, clipped = interval_records(h)
    assert clipped["visible"].tolist() == saved_profile["interval_visible"]
    assert sum(clipped["visible"]) == 5
    assert all(x == "no_intersection" for x in np.array(clipped["reasons"])[~clipped["visible"]])
    source_error = source_camera_error(np.asarray(false["corners_px"]), SIZE)
    assert source_error == false["gates"]["camera_error"]
    components = camera_components(h)
    index = int(components["errors"].argmin())
    cosine = float(components["cosine"][index])
    ratio = float(components["log_ratio"][index])
    focal = float(components["focals"][index])
    limit = .1  # Existing camera-error limit, not a new threshold.
    halfwidth = float(np.sqrt(limit**2-cosine**2))
    scale_bounds = np.exp([ratio-halfwidth, ratio+halfwidth])
    y0 = float(SEGMENTS[8, 0, 1])
    witnesses = []
    # Symmetric +/-5% diagnostic witnesses; no optimization of these values.
    for scale in [.95, 1., 1.05]:
        transform = np.array([[1., 0., 0.], [0., scale, (1.-scale)*y0], [0., 0., 1.]])
        changed = h @ transform
        new_endpoints, _ = project(changed, SEGMENTS)
        new_clip = clip_segments(new_endpoints)
        new_corners, depth = project(changed, CORNERS)
        same_mask = np.array_equal(clipped["visible"], new_clip["visible"])
        delta = float(np.abs(clipped["clipped"][clipped["visible"]]-new_clip["clipped"][clipped["visible"]]).max())
        fixed_error = float(np.hypot(cosine, ratio-np.log(scale)))
        if not same_mask or delta > 1e-8 or fixed_error > limit:
            raise AssertionError("Witness did not preserve the claimed geometric/camera properties")
        witnesses.append({"longitudinal_scale_a": scale, "homography_working": changed.tolist(),
                          "visibility_mask_exact_match": same_mask,
                          "clipped_endpoint_max_abs_delta_working_px": delta,
                          "corners_working_px": new_corners.tolist(),
                          "max_corner_displacement_working_px": float(np.linalg.norm(new_corners-projected_corners, axis=1).max()),
                          "camera_error_at_unchanged_focal": fixed_error,
                          "source_float32_camera_grid_min_error": source_camera_error(new_corners, SIZE),
                          "positive_corner_denominators": bool((depth > 0).all()),
                          "candidate_pool_membership": "not asserted; analytic witness only"})
    selected = [0, 1, 4, 5, 8]
    ranks = {"observed_five_supporting_lines": constraint_rank(h, selected),
             "plus_centre_supporting_line_only_hypothetical": constraint_rank(h, selected+[2]),
             "plus_second_cross_marking_hypothetical": constraint_rank(h, selected+[11]),
             "all_eleven_supporting_lines_hypothetical": constraint_rank(h, [0,1,2,4,5,6,7,8,9,10,11])}
    assert [ranks[k]["rank"] for k in ranks] == [7,7,8,8]
    # Usable scene19 line winner is a contrary example in exactly the same image.
    for candidate in ["1:60", "165:6702"]:
        e = compact[(scene, "automatic_all_camera", candidate)]
        q = reconstruct_h(np.asarray(e["corners_px"]))
        rows, vis = interval_records(q)
        _, d = project(q, CORNERS)
        geometry_pool.append({"identity": [scene, "automatic_all_camera", candidate],
                              "homography_source": "float64 reconstruction from verified saved corners",
                              "intervals": rows, "floor_score_recorded": e["gates"]["floor_score"],
                              "positive_corner_denominators": bool((d > 0).all()),
                              "ideal_visible_line_constraint_rank": constraint_rank(q, np.flatnonzero(vis["visible"]).tolist())})
    for name, e in list(controls["winners"].items()) + [("approved", {
            "candidate_id": controls["approved_candidate_id"],
            "corners_native_px": controls["approved_corners_native_px"],
            "floor_score": controls["approved_floor_score"]})]:
        pop = "GX0 observed-bank control (descriptive namespace)" if name != "approved" else "GX0 supplied-direction comparator (descriptive namespace)"
        q = reconstruct_h(np.asarray(e["corners_native_px"])/2)
        rows, vis = interval_records(q)
        _, d = project(q, CORNERS)
        geometry_pool.append({"identity": [controls["case_id"], pop, e["candidate_id"]],
                              "homography_source": "float64 reconstruction from verified GX0 control corners",
                              "intervals": rows, "floor_score_recorded": e["floor_score"],
                              "positive_corner_denominators": bool((d > 0).all()),
                              "ideal_visible_line_constraint_rank": constraint_rank(q, np.flatnonzero(vis["visible"]).tolist())})
    # Only 9 exact candidates are in the extended geometry audit, not a full negative population.
    assert len(geometry_pool) == 9
    output = {
        "schema": "court-identifiability-extension/1", "commit": COMMIT,
        "purpose": "read-only geometry/assignment audit; no image rescoring, training or detector search",
        "source_verification": [summary_prov, control_prov],
        "partial_rankings_provenance": {"file": partial_path.name,
                                       "sha256": hashlib.sha256(partial_path.read_bytes()).hexdigest(),
                                       "scope": "four complete JSON entries from a partially recovered gzip; full ranking archive CRC/blob NOT verified",
                                       "cross_checks": checks},
        "scene19": {"identity": [scene, "automatic_all_camera", "165:6702"],
                    "native_and_working_size": list(dimensions),
                    "image_header_only": True, "images_visually_inspected": False,
                    "homography_float64_reconstructed_from_corners": h.tolist(),
                    "saved_camera_error": false["gates"]["camera_error"],
                    "exact_source_camera_replay_error": source_error,
                    "camera_reconstruction_delta_float64_minus_saved": float(components["errors"][index]-source_error),
                    "baseline_focal_grid_index_zero_based": index,
                    "baseline_focal_widths": focal/SIZE[0],
                    "cosine_at_fixed_focal": cosine, "log_axis_norm_ratio_at_fixed_focal": ratio,
                    "fixed_focal_scale_bounds_under_existing_camera_limit": scale_bounds.tolist(),
                    "invariant_y0_metres_source_float32": y0,
                    "intervals": intervals, "witnesses": witnesses,
                    "ideal_line_constraint_ranks": ranks,
                    "limits": ["not pixel-score replay", "not asserted to be saved/generated candidate IDs",
                               "not a player, floor, net or end-to-end gate replay",
                               "rank describes ideal named-line geometry, not independent noise or calibrated evidence"]},
        "assignment_audit": assignments,
        "nine_candidate_geometry_audit": geometry_pool,
        "all_camera_compact_count": len(compact),
        "checks_passed": ["Both compact gzip files match committed Git blob SHAs and decompress with valid trailers",
                          "Four detailed entries agree exactly with compact scores, gates and corners",
                          "Four saved homographies reproduce corners within numerical precision",
                          "All four detailed visibility masks match geometry replay",
                          "Scene19 PNG header dimensions are 960x540",
                          "Scene19 false visibility matches all twelve saved flags, with seven no-intersection intervals",
                          "Scene19 source float32 camera error replay equals committed error exactly",
                          "All three analytic witnesses preserve availability and clipped endpoints numerically",
                          "All three witnesses satisfy the existing error bound at the unchanged focal",
                          "Ideal named-line ranks are 7, 7, 8, 8",
                          "Nine-candidate geometry audit includes positive contrary controls"]}
    (args.out/"audit_results.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    compact_table = [{"case_id": c, "population": p, "candidate_id": i,
                      "stripe_score": e["stripe_score"], "profile_score": e["profile_score"],
                      "gates": e["gates"]} for (c,p,i),e in compact.items()]
    (args.out/"unchanged_automatic_winner_summary.json").write_text(json.dumps(compact_table, indent=2)+"\n")
    print(f"Wrote {args.out/'audit_results.json'}")
    print(f"Source camera replay: {source_error:.17g}; saved: {false['gates']['camera_error']:.17g}")
    print("Scene19 witnesses (a, max corner displacement working px, fixed-focal camera error):")
    for row in witnesses:
        print(row["longitudinal_scale_a"], row["max_corner_displacement_working_px"], row["camera_error_at_unchanged_focal"])
    print("Assignment comparison (candidate, relative independent-to-exclusive forward loss):")
    for row in assignments:
        print(row["identity"], row["relative_forward_loss"])
    print(f"All {len(output['checks_passed'])} documented checks passed.")


if __name__ == "__main__":
    main()
