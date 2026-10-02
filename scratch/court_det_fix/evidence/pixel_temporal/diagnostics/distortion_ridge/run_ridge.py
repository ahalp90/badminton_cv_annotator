"""Measure lens distortion on GX by sampling the painted court-line ridge.

Second attempt at the distortion check. The first attempt (distortion/)
fitted quadratics to merged-row endpoints, which mixes paint with mat edges
and people and cannot separate lens distortion from merge noise. This
attempt samples the paint itself along the perpendicular to each court
edge, finds the ridge (the brightest point after removing the local
median), and fits a line and a quadratic to the ridge offsets.

Read-only on every input. Writes only under OUT.
"""

import csv
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

# <repo> is an alias that must be resolved locally.
REPO_ROOT = Path("<repo>")
OUT = REPO_ROOT / "scratch/court_det_fix/evidence/pixel_temporal/diagnostics/distortion_ridge"

MARGIN_PX = 20
N_POSITIONS = 60
STEP_PX = 0.5
PEAK_THRESHOLD_GREY = 25.0

# view -> (frame path, e3 record path, role)
GX_FRAME = (
    "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/"
    "20260909/gx_extension/people/images/gxBQ_window_00_frame_00000000.png"
)
GX_E3 = (
    "scratch/court_det_fix/direction_agreement/runs/"
    "direction_agreement_20260915_144900/e3/gxBQ_window_00_frame_0.json.gz"
)
AM3_FRAME = (
    "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/"
    "20260908/people_original/images/am3_window_00_frame_00000000.png"
)
AM3_E3 = (
    "scratch/court_det_fix/direction_agreement/runs/"
    "direction_agreement_20260915_144900/e3/am3_window_00_frame_0.json.gz"
)
SHUTTLESET_FRAME = (
    "scratch/court_det_fix/evidence/independent_proposals/development/people_job/images/"
    "shuttleset_03_scene_0016.png"
)
SHUTTLESET_E3 = (
    "scratch/court_det_fix/direction_agreement/runs/"
    "direction_agreement_20260915_144900/e3/shuttleset_03_scene_0016.json.gz"
)

# view -> (frame path, e3 record path, role)
VIEWS = {
    "gxBQ_window_00_frame_0": (GX_FRAME, GX_E3, "GX0 (test)"),
    "am3_window_00_frame_0": (AM3_FRAME, AM3_E3, "Amateur-3 (control)"),
    "shuttleset_03_scene_0016": (
        SHUTTLESET_FRAME,
        SHUTTLESET_E3,
        "ShuttleSet-03 scene 16 (control)",
    ),
}

# corners_native_px order: far-left, far-right, near-right, near-left
EDGES = {
    "far_baseline": (0, 1),
    "near_baseline": (3, 2),
    "left_sideline": (0, 3),
    "right_sideline": (1, 2),
}


def load_e3(path: Path) -> dict:
    with gzip.open(path) as f:
        return json.load(f)


def clip_segment_to_rect(p0, p1, xmin, xmax, ymin, ymax):
    """Liang-Barsky clip of segment p0->p1 to an axis-aligned rectangle.

    Returns (q0, q1) as the clipped endpoints, or None if fully outside.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, p0[0] - xmin),
        (dx, xmax - p0[0]),
        (-dy, p0[1] - ymin),
        (dy, ymax - p0[1]),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    if t0 >= t1:
        return None
    q0 = (p0[0] + t0 * dx, p0[1] + t0 * dy)
    q1 = (p0[0] + t1 * dx, p0[1] + t1 * dy)
    return q0, q1


def outward_normal(p0, p1, image_centre):
    """Unit normal to the edge p0->p1, oriented away from image_centre."""
    d = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=float)
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    mid = np.array([(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0])
    to_mid = mid - np.array(image_centre)
    if np.dot(n, to_mid) < 0:
        n = -n
    return d, n


def sample_ridge_at(gray, point, normal, half_width, step):
    offsets = np.arange(-half_width, half_width + step / 2.0, step)
    xs = point[0] + offsets * normal[0]
    ys = point[1] + offsets * normal[1]
    profile = map_coordinates(
        gray, [ys, xs], order=1, mode="nearest"
    ).astype(float)
    return offsets, profile


def find_ridge_offset(profile, step):
    median = np.median(profile)
    sub = profile - median
    idx = int(np.argmax(sub))
    n = len(sub)
    if idx == 0 or idx == n - 1:
        return None, "peak_at_window_end"
    peak_height = sub[idx]
    if peak_height < PEAK_THRESHOLD_GREY:
        return None, "peak_below_threshold"
    fm1, f0, fp1 = profile[idx - 1], profile[idx], profile[idx + 1]
    denom = fm1 - 2.0 * f0 + fp1
    delta = 0.0 if denom == 0 else 0.5 * (fm1 - fp1) / denom
    refined_idx = idx + delta
    centre_idx = (n - 1) / 2.0
    offset_px = (refined_idx - centre_idx) * step
    return offset_px, "ok"


def process_edge(gray, image_centre, corner_a, corner_b, width, height, edge_name):
    half_width = 6.0 if width == 960 else 12.0
    min_len = 150.0 if width == 960 else 300.0

    clipped = clip_segment_to_rect(
        corner_a,
        corner_b,
        MARGIN_PX,
        width - MARGIN_PX,
        MARGIN_PX,
        height - MARGIN_PX,
    )
    if clipped is None:
        return {
            "status": "skipped_no_overlap",
            "clipped_length_px": 0.0,
            "half_width_px": half_width,
            "min_len_px": min_len,
        }
    q0, q1 = clipped
    clipped_length = float(np.hypot(q1[0] - q0[0], q1[1] - q0[1]))
    if clipped_length < min_len:
        return {
            "status": "skipped_too_short",
            "clipped_length_px": clipped_length,
            "half_width_px": half_width,
            "min_len_px": min_len,
        }

    edge_dir, normal = outward_normal(corner_a, corner_b, image_centre)

    positions_t = np.linspace(0.0, 1.0, N_POSITIONS)
    surviving_pos = []
    surviving_offset = []
    discarded_points = []  # native (x, y) on the projected edge
    survived_points = []  # native (x, y) of the refined ridge sample

    for t in positions_t:
        point = (q0[0] + t * (q1[0] - q0[0]), q0[1] + t * (q1[1] - q0[1]))
        s = t * clipped_length  # arc-length position along the clipped edge
        _, profile = sample_ridge_at(gray, point, normal, half_width, STEP_PX)
        offset_px, _reason = find_ridge_offset(profile, STEP_PX)
        if offset_px is None:
            discarded_points.append(point)
            continue
        surviving_pos.append(s)
        surviving_offset.append(offset_px)
        ridge_xy = (
            point[0] + offset_px * normal[0],
            point[1] + offset_px * normal[1],
        )
        survived_points.append(ridge_xy)

    n_survive = len(surviving_pos)
    result = {
        "status": "ok",
        "clipped_length_px": clipped_length,
        "half_width_px": half_width,
        "min_len_px": min_len,
        "n_survive": n_survive,
        "q0": q0,
        "q1": q1,
        "edge_dir": edge_dir,
        "normal": normal,
        "discarded_points": discarded_points,
        "survived_points": survived_points,
    }

    if n_survive < 3:
        result["status"] = "too_few_survivors"
        return result

    pos_arr = np.array(surviving_pos)
    off_arr = np.array(surviving_offset)

    line_coeffs = np.polyfit(pos_arr, off_arr, 1)
    line_fit = np.polyval(line_coeffs, pos_arr)
    rms_residual = float(np.sqrt(np.mean((off_arr - line_fit) ** 2)))

    quad_coeffs = np.polyfit(pos_arr, off_arr, 2)
    a, b, c = quad_coeffs
    span_lo, span_hi = 0.0, clipped_length
    mid = (span_lo + span_hi) / 2.0
    quad_mid = a * mid**2 + b * mid + c
    quad_lo = a * span_lo**2 + b * span_lo + c
    quad_hi = a * span_hi**2 + b * span_hi + c
    chord_mid = (quad_lo + quad_hi) / 2.0
    sagitta = float(quad_mid - chord_mid)

    result.update(
        {
            "rms_residual_line_px": rms_residual,
            "sagitta_px": sagitta,
            "line_coeffs": line_coeffs.tolist(),
            "quad_coeffs": quad_coeffs.tolist(),
        }
    )
    return result


def to_px(value):
    return round(value)


def draw_markers(canvas, res):
    for x, y in res.get("discarded_points", []):
        cv2.circle(canvas, (to_px(x), to_px(y)), 2, (150, 150, 150), -1, cv2.LINE_AA)
    for x, y in res.get("survived_points", []):
        cv2.circle(canvas, (to_px(x), to_px(y)), 2, (0, 200, 0), -1, cv2.LINE_AA)


def draw_full_frame(image, view_edges, corners, out_path):
    canvas = image.copy()
    for ia, ib in EDGES.values():
        pa = (to_px(corners[ia][0]), to_px(corners[ia][1]))
        pb = (to_px(corners[ib][0]), to_px(corners[ib][1]))
        cv2.line(canvas, pa, pb, (0, 255, 255), 1, cv2.LINE_AA)
    for res in view_edges.values():
        draw_markers(canvas, res)
    cv2.imwrite(str(out_path), canvas)


def draw_edge_crop(image, res, corner_a, corner_b, out_path, pad=40):
    xs = [corner_a[0], corner_b[0]]
    ys = [corner_a[1], corner_b[1]]
    h, w = image.shape[:2]
    x0 = max(0, int(min(xs)) - pad)
    x1 = min(w, int(max(xs)) + pad)
    y0 = max(0, int(min(ys)) - pad)
    y1 = min(h, int(max(ys)) + pad)
    if x1 <= x0 or y1 <= y0:
        return
    canvas = image.copy()
    pa = (to_px(corner_a[0]), to_px(corner_a[1]))
    pb = (to_px(corner_b[0]), to_px(corner_b[1]))
    cv2.line(canvas, pa, pb, (0, 255, 255), 1, cv2.LINE_AA)
    draw_markers(canvas, res)
    crop = canvas[y0:y1, x0:x1]
    cv2.imwrite(str(out_path), crop)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table_rows = []
    all_results = {}

    for case_id, (frame_rel, e3_rel, role) in VIEWS.items():
        frame_path = REPO_ROOT / frame_rel
        e3_path = REPO_ROOT / e3_rel

        e3 = load_e3(e3_path)
        control = e3["control"]
        native_w, native_h = control["native_size"]
        corners = control["corners_native_px"]

        image = cv2.imread(str(frame_path))
        assert image is not None, f"could not read frame: {frame_path}"
        h, w = image.shape[:2]
        assert (w, h) == (native_w, native_h), (
            f"{case_id}: decoded frame size {(w, h)} != "
            f"control.native_size {(native_w, native_h)}"
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
        image_centre = (w / 2.0, h / 2.0)

        view_edges = {}
        for edge_name, (ia, ib) in EDGES.items():
            corner_a = corners[ia]
            corner_b = corners[ib]
            res = process_edge(
                gray, image_centre, corner_a, corner_b, w, h, edge_name
            )
            view_edges[edge_name] = res
            table_rows.append(
                {
                    "case_id": case_id,
                    "role": role,
                    "edge": edge_name,
                    "status": res["status"],
                    "clipped_length_px": round(res.get("clipped_length_px", 0.0), 2),
                    "n_survive": res.get("n_survive", 0),
                    "rms_residual_line_px": round(res["rms_residual_line_px"], 4)
                    if "rms_residual_line_px" in res
                    else "",
                    "sagitta_px": round(res["sagitta_px"], 4)
                    if "sagitta_px" in res
                    else "",
                }
            )

        all_results[case_id] = {"view_edges": view_edges, "corners": corners, "role": role}

        draw_full_frame(image, view_edges, corners, OUT / f"{case_id}.png")
        for edge_name, (ia, ib) in EDGES.items():
            res = view_edges[edge_name]
            draw_edge_crop(
                image, res, corners[ia], corners[ib], OUT / f"{case_id}_{edge_name}.png"
            )

    # table.csv
    fieldnames = [
        "case_id",
        "role",
        "edge",
        "status",
        "clipped_length_px",
        "n_survive",
        "rms_residual_line_px",
        "sagitta_px",
    ]
    with open(OUT / "table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    return table_rows, all_results


if __name__ == "__main__":
    rows, results = main()
    for r in rows:
        print(r)
