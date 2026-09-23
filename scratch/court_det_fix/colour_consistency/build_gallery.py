"""Build the colour diagnostic with the shared SVD court gallery renderer."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
TEMPLATE = BASE.parent / "svd_search/gallery_template.html"
sys.path.insert(0, str(BASE))
from run import read


def source_view(source: dict) -> dict:
    if source["status"] != "measured":
        return {"status": source["status"], "key": None, "geometry": None, "summary": None, "markings": []}
    geometry = {
        "corners": source["corners_working_px"],
        "centres": source["centres_working_px"],
        "edges": source["edges_working_px"],
    }
    expected = {"corners": (4, 2), "centres": (12, 2, 2), "edges": (24, 2, 2)}
    for name, shape in expected.items():
        if np.asarray(geometry[name]).shape != shape:
            raise ValueError(f"{source['selected_key']}: {name} must have shape {shape}")
    markings = []
    status_counts = {}
    raw_distances = []
    relative_distances = []
    for name, row in source["markings"].items():
        comparison = source["comparisons"][name]
        status = comparison["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        if comparison["nearest_raw_paint_ab"] is not None:
            raw_distances.append(comparison["nearest_raw_paint_ab"])
            relative_distances.append(comparison["nearest_relative_ab"])
        markings.append({
            "name": name,
            "status": status,
            "ambiguous": comparison.get("ambiguous_reference"),
            "support": row["geom_support"],
            "fragments": row["assigned_fragments"],
            "strong": row["sample_states"].get("supported", 0),
            "unresolved": row["sample_states"].get("unresolved_width", 0),
            "paint": row.get("median_paint_lab"),
            "floor_minus": row.get("median_floor_minus_lab"),
            "floor_plus": row.get("median_floor_plus_lab"),
            "delta": row.get("median_delta_lab"),
            "paint_spread": row.get("raw_paint_spread_ab"),
            "relative_spread": row.get("relative_spread_ab"),
            "nearest_raw": comparison["nearest_raw_paint_ab"],
            "nearest_relative": comparison["nearest_relative_ab"],
            "reference_spread": comparison.get("reference_spread_ab"),
            "references": comparison["reference_markings"],
        })
    return {
        "status": "measured",
        "key": source["selected_key"],
        "geometry": geometry,
        "summary": {
            "statuses": status_counts,
            "ambiguous": sum(bool(row["ambiguous"]) for row in markings),
            "supported": sum(row["strong"] for row in markings),
            "unresolved": sum(row["unresolved"] for row in markings),
            "median_raw": statistics.median(raw_distances) if raw_distances else None,
            "median_relative": statistics.median(relative_distances) if relative_distances else None,
        },
        "markings": markings,
    }


def gallery_case(case: dict, image_name: str) -> dict:
    sources = {name: source_view(source) for name, source in case["sources"].items()}
    saved = sources["saved_w5"]
    g1 = sources["saved_g1_templates"]
    arms = {}
    for arm in ("baseline", "deeper", "shortlist"):
        selected = sources.get(f"svd_{arm}")
        refitted = sources.get(f"svd_{arm}_reference_best_refit_retrospective")
        if selected is None:
            continue
        arms[arm] = {
            "geometry": {"selected": selected["geometry"], "oracle_refitted": refitted["geometry"]},
            "selected": selected["geometry"]["corners"] if selected["geometry"] else None,
            "oracle_refitted": refitted["geometry"]["corners"] if refitted["geometry"] else None,
        }
    return {
        "gallery_mode": "colour",
        "case_id": case["case_id"],
        "group": case["group"],
        "view_status": case["view_status"],
        "control_label": case["control_label"],
        "chromatic": case["chromatic"],
        "image": image_name,
        "size": case["working_size"],
        "saved_selected": saved["geometry"]["corners"] if saved["geometry"] else None,
        "saved_geometry": saved["geometry"],
        "g1_geometry": g1["geometry"],
        "arms": arms,
        "colour_sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE / "measurements.json.gz")
    parser.add_argument("--output", type=Path, default=BASE / "gallery")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    display = []
    for case in read(args.input)["cases"]:
        filename = f"{case['case_id']}.jpg"
        image_path = args.output / filename
        if not image_path.exists():
            image = cv2.imread(str(ROOT / "scratch/court_det_fix" / case["image"]))
            if image is None:
                raise FileNotFoundError(case["image"])
            frame = cv2.resize(image, tuple(case["working_size"]), interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise OSError(image_path)
        display.append(gallery_case(case, filename))
    payload = json.dumps(display, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text().replace("__GALLERY_DATA__", payload)
    (args.output / "index.html").write_text(page)
    print(f"Built {len(display)} cases: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
