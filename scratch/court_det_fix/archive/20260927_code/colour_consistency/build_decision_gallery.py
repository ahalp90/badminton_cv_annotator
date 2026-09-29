"""Build a compact decision review from frozen colour and edge trial records."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.annotator.independent_court import detector, paint_geometry

BASE = Path(__file__).resolve().parent
COURT = BASE.parent
OUTPUT = BASE / "decision_gallery"
TEMPLATE = COURT / "svd_search/gallery_template.html"
CASE_IDS = (
    "shuttleset_03_scene_0034", "shuttleset_03_scene_0029",
    "shuttleset_03_scene_0019", "gxBQ_window_00_frame_5",
    "am1_window_00_frame_54", "am4_window_00_frame_0",
)


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def projected_geometry(corners: list | None) -> dict | None:
    if corners is None:
        return None
    # Saved off-frame corners can exceed the shared helper's float32 round-trip tolerance.
    points = np.asarray(corners)
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, points.astype(np.float32))
    centres = detector.project(homography[None], paint_geometry.CENTRE_SEGMENTS_M)[0][0]
    intervals = np.repeat(np.arange(12), 2)
    positions = np.tile([1, 2], 12)
    stripes = paint_geometry.positioned_segments(paint_geometry.CENTRE_SEGMENTS_M, intervals, positions)
    edges = detector.project(homography[None], stripes)[0][0]
    return {"corners": points.tolist(), "centres": centres.reshape(-1, 2, 2).tolist(),
            "edges": edges.reshape(-1, 2, 2).tolist()}


def fit_view(fit: dict, accepted: bool) -> dict:
    measurement = fit.get("measurement")
    valid = bool(fit["successful"] and fit.get("valid_projection") and measurement
                 and measurement["camera_eligible"] and accepted)
    corners = fit["corners_px"] if valid else None
    return {"status": fit["status"], "valid": valid, "geometry": projected_geometry(corners)}


def observed_samples(case: dict) -> list[dict]:
    scale = np.asarray(case["working_size"], dtype=float) / np.asarray(case["native_size"], dtype=float)
    fragment_names = {}
    for name, marking in case["markings"].items():
        for fragment_id in marking["fragment_ids"]:
            fragment_names[fragment_id] = name
    points = []
    for fragment in case["observed"]["fragments"]:
        name = fragment_names.get(fragment["raw_fragment_id"])
        if name is None:
            continue
        valid = [sample for sample in fragment["samples"] if sample["valid"]]
        if not valid:
            continue
        centres = np.asarray([sample["xy_native"][1] for sample in valid], dtype=float)
        points.append({"name": name, "xy": (np.median(centres, axis=0) * scale).tolist()})
    return points


def paint_evidence(case: dict) -> dict:
    markings = []
    for name, row in case["markings"].items():
        colour = np.asarray(row["raw_ab"], dtype=float)
        offset = colour - 128.0
        markings.append({"name": name, "raw_ab": row["raw_ab"],
                         "hue": float(np.degrees(np.arctan2(offset[1], offset[0]))),
                         "chroma": float(np.linalg.norm(offset)), "fragments": len(row["fragment_ids"])})
    return {"raw": case["primary"], "hue_grouped_raw": case["balanced_hue"],
            "hue_only": case["hue_only"], "markings": markings,
            "samples": observed_samples(case)}


def movement(first: dict, second: dict) -> float | None:
    if not first["successful"] or not second["successful"]:
        return None
    corners = np.asarray(first["corners_px"], dtype=float)
    earlier = np.asarray(second["corners_px"], dtype=float)
    return float(np.max(np.linalg.norm(corners - earlier, axis=1)))


def main() -> None:
    paint = read(BASE / "paint_grouping_trial.json.gz")
    edge = read(BASE / "edge_auto_trial.json.gz")
    floor = read(BASE / "floor_trial.json.gz")
    paint_by_id = {case["case_id"]: case for case in paint["cases"]}
    edge_by_id = {case["case_id"]: case for case in edge["cases"]}
    floor_arms = [arm for case in floor["cases"] for arm in case["arms"]]
    summary = {
        "floor": {"changed": sum(arm["before"] != arm["after"] for arm in floor_arms), "total": len(floor_arms)},
        "raw": {"rejected": sum(case["primary"]["after"] == "abstain" for case in paint["cases"]),
                "total": len(paint["cases"])},
        "grouped": {"rejected": sum(case["balanced_hue"]["after"] == "abstain" for case in paint["cases"]),
                    "total": len(paint["cases"])},
        "hue": {decision: sum(case["hue_only"]["after"] == decision for case in paint["cases"])
                for decision in ("abstain", "keep", "no_decision")},
        "edge": {"valid": sum(case["automatic_valid"] for case in edge["cases"]),
                 "total": len(edge["cases"]),
                 "exact_reuse": sum(case["reused_saved_arm"] is not None for case in edge["cases"])},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in CASE_IDS:
        colour = paint_by_id[case_id]
        trial = edge_by_id[case_id]
        if not colour["selected_key"].startswith(trial["parent_key"]):
            raise ValueError(f"{case_id}: colour and edge parent keys differ")
        native = np.asarray(colour["native_size"], dtype=float)
        working = np.asarray(colour["working_size"], dtype=float)
        scale = working / native
        image_path = COURT / colour["image"]
        image = cv2.imread(str(image_path))
        if image is None or image.shape[:2] != tuple(native[::-1].astype(int)):
            raise ValueError(f"{case_id}: missing image or native size mismatch: {image_path}")
        image_name = f"{case_id}.jpg"
        frame = cv2.resize(image, tuple(working.astype(int)), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(OUTPUT / image_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise OSError(OUTPUT / image_name)
        original = trial["fits_saved"]["original"]
        earlier = trial["bright_centres_fit"]
        automatic = trial["automatic_fit"]
        views = {"original": fit_view(original, True),
                 "earlier": fit_view(earlier, True),
                 "automatic": fit_view(automatic, trial["automatic_valid"])}
        saved_w5 = (
            projected_geometry((np.asarray(colour["corners_native_px"]) * scale).tolist())
            if case_id.startswith("am1_") else None
        )
        rows.append({"gallery_mode": "decision_trials", "case_id": case_id, "image": image_name,
                     "size": colour["working_size"], "decision_views": views,
                     "saved_selected": views["original"]["geometry"]["corners"] if views["original"]["valid"] else None,
                     "saved_w5_geometry": saved_w5,
                     "movement_px": movement(automatic, earlier),
                     "reused_saved_arm": trial["reused_saved_arm"],
                     "paint": paint_evidence(colour) if case_id.startswith("am1_") else None,
                     "box_mask": colour["box_mask"],
                     "context": "Known wrong net fit" if case_id.startswith("am1_") else
                                "Unchanged control" if case_id.startswith("am4_") else
                                "Known good partial view" if case_id == "gxBQ_window_00_frame_5" else
                                "Initial reviewed case"})
    rows[0]["decision_summary"] = summary
    payload = json.dumps(rows, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    template = TEMPLATE.read_text()
    if template.count("__GALLERY_DATA__") != 1:
        raise ValueError("shared template has no unique data slot")
    template = template.replace("Maximum corner movement vs earlier rule",
                                "Maximum corner movement vs earlier rule (working px)")
    (OUTPUT / "index.html").write_text(template.replace("__GALLERY_DATA__", payload))
    print(f"Built {len(rows)} cases: {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    cv2.setNumThreads(1)
    main()
