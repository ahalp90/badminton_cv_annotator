"""Build an offline court overlay gallery and compact comparison from completed arms."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.annotator.independent_court import detector, paint_geometry

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
ARMS = ("baseline", "deeper", "shortlist")


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write(path: Path, value: dict) -> None:
    path.write_bytes(gzip.compress(json.dumps(value, allow_nan=False, separators=(",", ":")).encode(), mtime=0))


def candidate(summary: dict, key: str | None) -> dict | None:
    if key is None:
        return None
    return next((item for item in summary["candidates"] if item["origin_key"] == key), None)


def projected_geometry(corners: list | None) -> dict | None:
    if corners is None:
        return None
    points = np.asarray(corners)
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, points.astype(np.float32))
    reprojected = detector.project(homography[None], detector.CORNER_COURT_M)[0][0]
    np.testing.assert_allclose(reprojected, points, atol=0.01, rtol=0)
    centres = detector.project(homography[None], paint_geometry.CENTRE_SEGMENTS_M)[0][0]
    intervals = np.repeat(np.arange(12), 2)
    positions = np.tile([1, 2], 12)
    stripes = paint_geometry.positioned_segments(paint_geometry.CENTRE_SEGMENTS_M, intervals, positions)
    edges = detector.project(homography[None], stripes)[0][0]
    return {"corners": points.tolist(), "centres": centres.tolist(), "edges": edges.tolist()}


def arm_view(summary: dict, scale: np.ndarray) -> dict:
    selected = candidate(summary, summary["selected_key"])
    generated = candidate(summary, (summary["oracle_generated"] or {}).get("origin_key"))
    refitted = candidate(summary, (summary["oracle_refitted"] or {}).get("origin_key"))
    views = {
        "selected": (np.asarray(selected["corners_px"]) * scale).tolist() if selected else None,
        "oracle_generated": (np.asarray(generated["corners_px"]) * scale).tolist() if generated else None,
        "oracle_refitted": (np.asarray(refitted["corners_px"]) * scale).tolist() if refitted else None,
    }
    return {
        **views,
        "geometry": {name: projected_geometry(corners) for name, corners in views.items()},
        "selected_error": (summary["selected_reference_error"] or {}).get("maximum"),
        "selected_saved_error": (summary.get("selected_saved_error") or {}).get("maximum"),
        "oracle_generated_error": (summary["oracle_generated"] or {}).get("maximum_native_px"),
        "oracle_refitted_error": (summary["oracle_refitted"] or {}).get("maximum_native_px"),
        "counts": summary["counts"], "timing": summary["timing"],
        "ranking_status": summary["ranking_status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Build only named cases")
    parser.add_argument("--allow-smoke", action="store_true", help="Allow pair-limited data for local HTML smoke")
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read(BASE / "input_manifest.json.gz")
    saved = read(BASE / "saved_selected.json.gz")
    rows = []
    for item in manifest["cases"]:
        case_id = item["case_id"]
        if args.case and case_id not in args.case:
            continue
        summaries = {arm: read(source / "cases" / arm / f"{case_id}.json.gz") for arm in ARMS}
        if not args.allow_smoke and any(summary["max_matched_pairs"] is not None for summary in summaries.values()):
            raise ValueError(f"{case_id}: smoke-limited result cannot enter full gallery")
        frame = cv2.imread(str(ROOT / item["image"]))
        if frame is None:
            raise FileNotFoundError(ROOT / item["image"])
        generation = read(source / summaries["baseline"]["generation_record"])
        working = tuple(generation["working_size"])
        frame = cv2.resize(frame, working, interpolation=cv2.INTER_AREA)
        image_name = f"{case_id}.png"
        if not cv2.imwrite(str(output / image_name), frame):
            raise OSError(output / image_name)
        source_pack = read(ROOT / item["source_pack"])
        source_case = next(case for case in source_pack["cases"] if case["id"] == case_id)
        native = [source_case["dimensions"]["width"], source_case["dimensions"]["height"]]
        scale = np.asarray(working, dtype=float) / np.asarray(native, dtype=float)
        saved_corners = saved[case_id]["corners_native_px"]
        saved_working = (np.asarray(saved_corners) * scale).tolist() if saved_corners else None
        rows.append({
            "case_id": case_id, "image": image_name, "size": working,
            "saved_selected": saved_working,
            "saved_geometry": projected_geometry(saved_working),
            "saved_source": saved[case_id]["source"],
            "saved_selected_key": saved[case_id]["selected_key"],
            "reference": (np.asarray(summaries["baseline"]["reference_corners_native_px"]) * scale).tolist(),
            "arms": {arm: arm_view(summary, scale) for arm, summary in summaries.items()},
            "scale": scale.tolist(),
        })
    write(output / "comparison.json.gz", {"schema": "svd-search-comparison/1", "cases": rows})
    template = (BASE / "gallery_template.html").read_text()
    page = template.replace("__GALLERY_DATA__", json.dumps(rows, allow_nan=False))
    (output / "index.html").write_text(page)
    print(f"Built {len(rows)} cases: {output / 'index.html'}")


if __name__ == "__main__":
    main()
