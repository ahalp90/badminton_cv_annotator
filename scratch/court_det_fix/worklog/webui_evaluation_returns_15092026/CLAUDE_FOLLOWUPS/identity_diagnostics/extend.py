"""Identity diagnostics extended from the nine-candidate audit to every exported winner.

Read-only on every named input. Reuses audit.py's geometric functions
(reconstruct_h, clip_segments, constraint_rank, interval_records, line_separation)
unmodified. The only new logic here is a generic all-pairs marking-separation
scan (audit.py itself only ran line_separation for two hand-picked pairs), and
the table assembly, quoting and CSV/markdown writing.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import audit  # local copy, unmodified; importable because it lives next to this script
import numpy as np

REPO = Path(__file__).resolve().parents[6]
PROJ = REPO / "experiments/annotator/independent_court/recorded/player_guided/projective_patterns"
OUT = Path(__file__).resolve().parent

# Build interval_index -> marking name directly from MARKING_INTERVALS.
INTERVAL_TO_MARKING = {}
for marking_index, intervals in enumerate(audit.MARKING_INTERVALS):
    for interval_index in intervals:
        INTERVAL_TO_MARKING[interval_index] = audit.MARKINGS[marking_index]


def load_gzip_json(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


# SEGMENTS indices 0-5 are the five vertical (sideline/centre) markings; 6-11
# are the six horizontal (baseline/service-line) markings. A crossing pair
# (one vertical, one horizontal) always has an extended-line intersection
# somewhere, usually inside the clipped span, so audit.line_separation trivially
# returns 0 for almost every such pair -- that is a projective fact about
# perpendicular lines, not evidence of a shared ridge. Restricting the scan to
# same-family pairs (both vertical or both horizontal) is what makes the
# distance meaningful, and matches the two pairs audit.py itself picked out
# (near_baseline/near_long_service and far_long_service/far_baseline are both
# horizontal-family pairs).
VERTICAL_INTERVALS = list(range(6))
HORIZONTAL_INTERVALS = list(range(6, 12))


def minimal_marking_separation(h: np.ndarray, visible: np.ndarray) -> dict | None:
    """Minimum projected separation between any two distinct, same-family markings.

    For every ordered pair of same-family painted intervals (both vertical or
    both horizontal) whose source interval is available (>=12 working-pixel
    clipped span, per `visible`), compute the perpendicular distance from that
    clipped source span to the other interval's supporting line
    (audit.line_separation), and keep the smallest low-end distance seen
    across all pairs of DISTINCT markings. Centre's two intervals (indices 2
    and 3) map to the same marking and are skipped against each other.
    Returns None if no interval is available at all.
    """
    best = None
    for family in (VERTICAL_INTERVALS, HORIZONTAL_INTERVALS):
        for source in family:
            if not visible[source]:
                continue
            for target in family:
                if source == target:
                    continue
                source_marking = INTERVAL_TO_MARKING[source]
                target_marking = INTERVAL_TO_MARKING[target]
                if source_marking == target_marking:
                    continue
                result = audit.line_separation(h, source, target)
                low = result["perpendicular_distance_range_working_px"][0]
                if best is None or low < best[0]:
                    best = (low, source_marking, target_marking)
    if best is None:
        return None
    return {"minimum_working_px": best[0], "marking_a": best[1], "marking_b": best[2]}


def geometry_row(h: np.ndarray) -> dict:
    _intervals, clipped = audit.interval_records(h)
    available = int(sum(clipped["visible"]))
    rank = audit.constraint_rank(h, np.flatnonzero(clipped["visible"]).tolist())["rank"]
    separation = minimal_marking_separation(h, clipped["visible"])
    if separation is None:
        return {"available_interval_count": available, "constraint_rank": rank,
                "min_separation_working_px": None, "min_separation_pair": "no available interval"}
    return {"available_interval_count": available, "constraint_rank": rank,
            "min_separation_working_px": separation["minimum_working_px"],
            "min_separation_pair": f"{separation['marking_a']} vs {separation['marking_b']}"}


# --- visual rulings, hand-transcribed quotes keyed by (case_id, role) ---
# Source: automatic_axes_visual_judgements.md, one subsection per view.
AUTOMATIC_RULINGS = {
    ("gxBQ_window_00_frame_0", "line"):
        "not really usable. The top-left corner lies up and right of its expected "
        "position; the bottom-left lies too far left. The court looks sheared. The "
        "bottom-right corner seems perfect. Without the shear, the fit would probably "
        "be good",
    ("gxBQ_window_00_frame_0", "paint"):
        "worse, with the same shear and an incorrect bottom-right corner",
    ("gxBQ_window_00_frame_5", "line"):
        "rejected; it draws a court on the wall. The user suspects a far-left net "
        "post and a person were treated as the two net posts",
    ("gxBQ_window_00_frame_5", "paint"):
        "rejected; it draws a court over the seated children. The user associates "
        "its left margin with the space between two net posts",
    ("am2_window_00_frame_150", "line"):
        "mistakes the blue mat boundary for the court. The sidelines otherwise "
        "align with the outer edges of the white paint",
    ("am2_window_00_frame_150", "paint"):
        "described as perfect, including alignment with the outer paint edges. The "
        "extreme far end is uncertain because of perspective, but looks suitable",
    ("am2_window_01_frame_28019", "line"):
        "extends to the blue mat borders and appears sheared in several directions. "
        "It roughly follows the right painted sideline, but not adequately",
    ("am2_window_01_frame_28019", "paint"):
        "rejected as an incoherent court near the net top",
    ("am3_window_00_frame_0", "line"): "essentially perfect",
    ("am3_window_00_frame_0", "paint"):
        "very usable. The near baseline follows the inner paint edge. The first "
        "horizontal in from the net visibly biases downward along its length",
    ("shuttleset_03_scene_0017", "line"):
        "very usable with slight skew. Sidelines follow the inner paint edges. "
        "Horizontals appear to move from the top side of the paint on the left to "
        "the bottom side on the right",
    ("shuttleset_03_scene_0017", "paint"):
        "similar but worse. The far baseline overshoots by an estimated 20 cm",
    ("shuttleset_03_scene_0019", "line"):
        "usable. Sidelines are well aligned with the inner paint edges. The near "
        "baseline is perfect at the left and slightly inset at the right. The far "
        "baseline drifts from the inner paint edge on the left to the outer edge "
        "on the right",
    ("shuttleset_03_scene_0019", "paint"):
        "rejected as an unrelated, hallucinated court",
    ("shuttleset_03_scene_0016", "line"):
        "very usable. Sidelines follow the inner paint edges; the near baseline is "
        "essentially perfect. The far baseline drifts from the outer paint edge on "
        "the left to the inner edge on the right",
    ("shuttleset_03_scene_0016", "paint"):
        "usable. Sidelines and near baseline follow the inner paint edges. The far "
        "baseline slightly overshoots on the left and slopes towards the outer "
        "paint edge at the right",
    ("shuttleset_21_scene_0020", "line"):
        "very usable. Sidelines follow the inner paint edges. The near baseline "
        "follows the outer edge perfectly. The far baseline slightly overshoots at "
        "the top left and otherwise roughly follows the outer paint edge.",
    ("shuttleset_21_scene_0020", "paint"):
        "very usable. Sidelines follow the inner paint edges. The near baseline "
        "follows the outer edge perfectly. The far baseline slightly overshoots at "
        "the top left and otherwise roughly follows the outer paint edge.",
}
# Source: automatic_axes_results.md results table, "line / paint" corner error
# columns in display pixels (1280x720). All-camera-eligible column used, since
# these winners come from the automatic_all_camera population.
CORNER_ERROR_DISPLAY_PX = {
    ("am2_window_00_frame_150", "line"): 185.1,
    ("am2_window_00_frame_150", "paint"): 10.8,
    ("am2_window_01_frame_28019", "line"): 193.5,
    ("am2_window_01_frame_28019", "paint"): 1189.5,
    ("am3_window_00_frame_0", "line"): 7.7,
    ("am3_window_00_frame_0", "paint"): 11.3,
    ("gxBQ_window_00_frame_0", "line"): 18.8,
    ("gxBQ_window_00_frame_0", "paint"): 12.7,
    ("gxBQ_window_00_frame_5", "line"): 705.5,
    ("gxBQ_window_00_frame_5", "paint"): 704.4,
    ("shuttleset_03_scene_0016", "line"): 8.6,
    ("shuttleset_03_scene_0016", "paint"): 11.3,
    ("shuttleset_03_scene_0017", "line"): 11.1,
    ("shuttleset_03_scene_0017", "paint"): 10.2,
    ("shuttleset_03_scene_0019", "line"): 12.7,
    ("shuttleset_03_scene_0019", "paint"): 8390.9,
    ("shuttleset_21_scene_0020", "line"): 7.2,
    ("shuttleset_21_scene_0020", "paint"): 7.2,
}


def main() -> None:
    ranking_path = PROJ / "evaluation/ranking_records.json.gz"
    gx0_path = PROJ / "gx0_control_measurements.json.gz"
    ranking = load_gzip_json(ranking_path)
    gx0 = load_gzip_json(gx0_path)

    rows = []
    for pop in ranking["populations"]:
        case_id, population = pop["case_id"], pop["population"]
        line_id, paint_id = pop.get("line_winner_id"), pop.get("paint_winner_id")
        for entry in pop["entries"]:
            candidate_id = entry["candidate_id"]
            roles = []
            if candidate_id == line_id:
                roles.append("line")
            if candidate_id == paint_id:
                roles.append("paint")
            role_label = " and ".join(roles) if roles else "not a winner"
            h = np.asarray(entry["homography_working"])
            geometry = geometry_row(h)
            if population == "automatic_all_camera":
                ruling = " / ".join(AUTOMATIC_RULINGS[(case_id, r)] for r in roles) \
                    if roles else "no ruling recorded"
                corner_error = " / ".join(str(CORNER_ERROR_DISPLAY_PX[(case_id, r)])
                                           for r in roles) if roles else "no ruling recorded"
                population_label = population
            else:  # gx0_label_guided_bank_control
                key = "line" if candidate_id == gx0["winners"]["line"]["candidate_id"] else "paint"
                ruling = gx0["winners"][key]["visual_judgement"]
                corner_error = gx0["winners"][key]["approved_max_corner_display_px"]
                population_label = population
                role_label = key
            rows.append({
                "case_id": case_id, "population": population_label, "candidate_id": str(candidate_id),
                "ranking_role": role_label, "visual_ruling": ruling,
                "available_interval_count": geometry["available_interval_count"],
                "constraint_rank": geometry["constraint_rank"],
                "min_separation_working_px": geometry["min_separation_working_px"],
                "min_separation_pair": geometry["min_separation_pair"],
                "max_corner_error_display_px": corner_error,
            })

    # Comparator 89: reconstructed from its own native-pixel corners, not
    # exported as a ranking_records.json.gz entry (it has no homography_working
    # there; it is the GX0 reference the two bank-control winners are measured
    # against). Same reconstruction path audit.py uses for the GX0 controls.
    approved_corners_working = np.asarray(gx0["approved_corners_native_px"]) / 2
    h89 = audit.reconstruct_h(approved_corners_working)
    geometry89 = geometry_row(h89)
    rows.append({
        "case_id": gx0["case_id"], "population": "GX0 supplied-direction comparator",
        "candidate_id": str(gx0["approved_candidate_id"]), "ranking_role": "comparator (approved reference)",
        "visual_ruling": "no ruling recorded",
        "available_interval_count": geometry89["available_interval_count"],
        "constraint_rank": geometry89["constraint_rank"],
        "min_separation_working_px": geometry89["min_separation_working_px"],
        "min_separation_pair": geometry89["min_separation_pair"],
        "max_corner_error_display_px":
            "not recorded; candidate89 is itself the reference for winners 1864 and 5144",
    })

    fieldnames = ["case_id", "population", "candidate_id", "ranking_role", "visual_ruling",
                  "available_interval_count", "constraint_rank", "min_separation_working_px",
                  "min_separation_pair", "max_corner_error_display_px"]
    with open(OUT / "table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(OUT / "rows.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} rows.")
    for row in rows:
        print(row["case_id"], row["population"], row["candidate_id"], row["ranking_role"],
              "avail=", row["available_interval_count"], "rank=", row["constraint_rank"],
              "min_sep=", round(row["min_separation_working_px"], 3), row["min_separation_pair"])


if __name__ == "__main__":
    main()
