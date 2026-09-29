"""Build a local visual comparison from saved centre-label diagnostics."""

from __future__ import annotations

import argparse
import gzip
import html
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.annotator.independent_court import detector, paint_geometry

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE / "diagnostics.json.gz")
    parser.add_argument("--output", type=Path, default=BASE / "gallery")
    parser.add_argument("--title", default="Court fit · four-case visual review")
    args = parser.parse_args()
    cases = read(args.input)["cases"]
    saved = {case["case_id"]: case for case in read(BASE.parent / "results.json.gz")["cases"]}
    manifest = {case["case_id"]: case for case in read(ROOT / "wider_evaluation/runs/20260922/manifest.json.gz")["cases"]}
    gallery = []
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        case_id = case["case_id"]
        previous = saved.get(case_id)
        frame = cv2.imread(str(ROOT / manifest[case_id]["image"]))
        assert frame is not None, case_id
        working_size = tuple(case["working_size"] if "working_size" in case else previous["working_size"])
        if (frame.shape[1], frame.shape[0]) != working_size:
            frame = cv2.resize(frame, working_size, interpolation=cv2.INTER_AREA)
        image_name = f"{case_id}.png"
        assert cv2.imwrite(str(output / image_name), frame)
        arms = dict(case["fits"])
        if "original" not in arms:
            arms["original"] = previous["fits"]["baseline"]
        geometries = {}
        for name, fit in arms.items():
            if not fit["successful"]:
                geometries[name] = {"corners": None, "status": fit["status"], "paint": None}
                continue
            corners = np.asarray(fit["corners_px"])
            homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, corners.astype(np.float32))
            reprojected = detector.project(homography[None], detector.CORNER_COURT_M)[0][0]
            # Bound float32 reconstruction error to a hundredth of a working pixel.
            np.testing.assert_allclose(reprojected, corners, atol=0.01, rtol=0)
            centres = detector.project(homography[None], paint_geometry.CENTRE_SEGMENTS_M)[0][0]
            intervals = np.repeat(np.arange(12), 2)
            positions = np.tile([1, 2], 12)
            stripes = paint_geometry.positioned_segments(paint_geometry.CENTRE_SEGMENTS_M, intervals, positions)
            edges = detector.project(homography[None], stripes)[0][0]
            geometries[name] = {"corners": corners.tolist(), "centres": centres.tolist(), "edges": edges.tolist(),
                                "paint": fit["measurement"]["paint_score"],
                                "camera_eligible": fit["measurement"]["camera_eligible"], "status": fit["status"]}
        title = case.get("title", case_id.replace("_", " "))
        if case_id.startswith("shuttleset"):
            title = f"SS{case_id.split('_')[1]} · scene {case_id[-2:]}"
        elif case_id.startswith("gx"):
            title = f"GX · frame {case_id.rsplit('_', 1)[1]}"
        gallery.append({"id": case_id, "title": title, "image": image_name, "size": working_size,
                        "arms": geometries, "changed_fragments": len(case["centre_choices"])})
    template = (BASE / "gallery_template.html").read_text()
    template = template.replace("__GALLERY_TITLE__", html.escape(args.title))
    (output / "index.html").write_text(template.replace("__GALLERY_DATA__", json.dumps(gallery, allow_nan=False)))
    print(f"Built {len(gallery)} cases: {output / 'index.html'}")


if __name__ == "__main__":
    main()
