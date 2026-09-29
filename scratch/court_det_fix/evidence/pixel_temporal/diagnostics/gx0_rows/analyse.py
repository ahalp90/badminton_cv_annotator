"""Read-only analysis script for the gx0_rows brief.

Reads E0 membership, E2 support masks, the input fragments, the frame and
the approved control court, checks the gate, draws the overlay and crops,
and prints the gate table for note.md. Writes only under OUT.
"""

import gzip
import json
from pathlib import Path

import cv2
import numpy as np

# <repo> is an alias that must be resolved locally.
REPO = Path("<repo>")
OUT = REPO / "scratch/court_det_fix/evidence/pixel_temporal/diagnostics/gx0_rows"

FRAME_PATH = (
    REPO / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/"
    "20260909/gx_extension/people/images/gxBQ_window_00_frame_00000000.png"
)
INPUTS_PATH = (
    REPO / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/"
    "20260909/gx_extension/inputs.json.gz"
)
E0_PATH = REPO / "scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/e0/gxBQ_window_00_frame_0.json.gz"
E2_PATH = REPO / "scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/e2/gxBQ_window_00_frame_0.json.gz"
CONTROL_PATH = REPO / "experiments/annotator/independent_court/recorded/player_guided/projective_patterns/gx0_control_measurements.json.gz"


def load_gz_json(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def main() -> None:
    inputs = load_gz_json(INPUTS_PATH)
    case = None
    for entry in inputs["cases"]:
        if entry.get("id") == "gxBQ_window_00_frame_0":
            case = entry
            break
    if case is None:
        raise SystemExit("gate failed: case id gxBQ_window_00_frame_0 not found in inputs.json.gz")
    segments_px = case["segments_px"]
    print(f"gate: found case gxBQ_window_00_frame_0 with {len(segments_px)} fragments")

    e0 = load_gz_json(E0_PATH)
    membership = e0["membership"]
    print(f"gate: E0 membership has {len(membership)} merged rows")

    e2 = load_gz_json(E2_PATH)
    arm_b = e2["arms"]["B"]
    leader_ids = arm_b["leader_candidate_ids"]
    support_masks = arm_b["support_masks"]
    if 737 not in leader_ids or 4104 not in leader_ids:
        raise SystemExit(f"gate failed: 737 or 4104 not in leader_candidate_ids {leader_ids}")
    idx_737 = leader_ids.index(737)
    idx_4104 = leader_ids.index(4104)
    mask_737 = support_masks[idx_737]
    mask_4104 = support_masks[idx_4104]
    print(f"gate: candidate 737 at leader index {idx_737}, candidate 4104 at leader index {idx_4104}")

    target_rows = [31, 51, 61]
    row_info = {}
    for row_id in target_rows:
        row = None
        for member in membership:
            if member["line_id"] == row_id:
                row = member
                break
        if row is None:
            raise SystemExit(f"gate failed: line_id {row_id} not found in E0 membership")
        row_info[row_id] = row
        print(
            f"row {row_id}: member_raw_fragment_ids={row['member_raw_fragment_ids']} "
            f"covered_length_px={row['covered_length_px']}"
        )

    # Gate check: row 31 and 61 in candidate 737's mask, row 51 in candidate 4104's mask.
    checks = [
        (31, mask_737, 737),
        (61, mask_737, 737),
        (51, mask_4104, 4104),
    ]
    gate_lines = []
    all_pass = True
    for row_id, mask, cand in checks:
        in_mask = bool(mask[row_id])
        gate_lines.append(f"row {row_id} in candidate {cand} mask: {in_mask}")
        if not in_mask:
            all_pass = False
    print("gate mask checks:")
    for line in gate_lines:
        print(f"  {line}")
    if not all_pass:
        raise SystemExit("gate failed: mask membership check did not pass; stopping per brief instructions")
    print("gate: PASS - all mask membership checks hold")

    # Load control court corners.
    control = load_gz_json(CONTROL_PATH)
    corners = control["approved_corners_native_px"]
    print(f"control corners: {corners}")

    # Load frame.
    frame = cv2.imread(str(FRAME_PATH))
    if frame is None:
        raise SystemExit(f"gate failed: could not read frame at {FRAME_PATH}")
    overlay = frame.copy()

    # Draw control court in yellow (thin). Corners assumed to be a 4-point
    # polygon (order as recorded); draw as a closed polyline.
    corner_pts = np.array(corners, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(overlay, [corner_pts], isClosed=True, color=(0, 255, 255), thickness=1)

    def draw_fragment(img, frag, color, thickness):
        x1, y1, x2, y2 = frag
        cv2.line(img, (round(x1), round(y1)), (round(x2), round(y2)), color, thickness)

    # Other support rows for candidate 737 (green) and candidate 4104 (blue),
    # excluding the three target rows themselves.
    membership_by_id = {m["line_id"]: m for m in membership}

    def other_support_fragment_ids(mask, exclude_ids):
        frag_ids = []
        for row_id, is_member in enumerate(mask):
            if not is_member or row_id in exclude_ids:
                continue
            row = membership_by_id.get(row_id)
            if row is None:
                continue
            frag_ids.extend(row["member_raw_fragment_ids"])
        return frag_ids

    green_frag_ids = other_support_fragment_ids(mask_737, set(target_rows))
    blue_frag_ids = other_support_fragment_ids(mask_4104, set(target_rows))

    for frag_id in green_frag_ids:
        draw_fragment(overlay, segments_px[frag_id], (0, 200, 0), 1)
    for frag_id in blue_frag_ids:
        draw_fragment(overlay, segments_px[frag_id], (255, 0, 0), 1)

    # Red target rows, thick, with label near midpoint.
    row_midpoints = {}
    for row_id in target_rows:
        row = row_info[row_id]
        pts = []
        for frag_id in row["member_raw_fragment_ids"]:
            frag = segments_px[frag_id]
            draw_fragment(overlay, frag, (0, 0, 255), 3)
            pts.append(frag)
        xs = [p[0] for p in pts] + [p[2] for p in pts]
        ys = [p[1] for p in pts] + [p[3] for p in pts]
        mid = (round(sum(xs) / len(xs)), round(sum(ys) / len(ys)))
        row_midpoints[row_id] = mid
        cv2.putText(
            overlay,
            str(row_id),
            (mid[0] + 8, mid[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "overlay.png"), overlay)
    print(f"wrote {OUT / 'overlay.png'}")

    # Crops: ~500x350 around each red row's midpoint, with and without drawing.
    h, w = frame.shape[:2]
    crop_w, crop_h = 500, 350
    for row_id in target_rows:
        mx, my = row_midpoints[row_id]
        x0 = max(0, min(w - crop_w, mx - crop_w // 2))
        y0 = max(0, min(h - crop_h, my - crop_h // 2))
        x1 = x0 + crop_w
        y1 = y0 + crop_h
        crop_drawn = overlay[y0:y1, x0:x1]
        crop_clean = frame[y0:y1, x0:x1]
        cv2.imwrite(str(OUT / f"row_{row_id}.png"), crop_drawn)
        cv2.imwrite(str(OUT / f"row_{row_id}_clean.png"), crop_clean)
        print(f"wrote row_{row_id}.png and row_{row_id}_clean.png, crop box=({x0},{y0},{x1},{y1})")

    # Print gate summary table data for note.md.
    print("\n--- table data for note.md ---")
    for row_id in target_rows:
        row = row_info[row_id]
        print(
            f"{row_id}\t{len(row['member_raw_fragment_ids'])}\t{row['covered_length_px']}"
        )


if __name__ == "__main__":
    main()
