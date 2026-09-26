"""Build a two-column gallery from the saved selected-court polarity fits."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import cv2
import numpy as np
from build_bounded_gallery import project_selection, read, replace_once

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
REPO = ROOT.parents[1]
OUTPUT = BASE / "polarity_gallery"
TEMPLATE = ROOT / "svd_search/gallery_template.html"
TRIAL = REPO / "local_scratch/net_recovery/20260923/bounded/bounded_trial.json.gz"
REQUESTS = REPO / "local_scratch/net_recovery/20260923/selected_polarity/bounded_requests.json.gz"
RESULTS = REPO / "local_scratch/net_recovery/20260923/selected_polarity/bounded_results.json.gz"
REPLAY_TOLERANCE_NATIVE_PX = 2.5e-9
CORNER_TOLERANCE_PX = 1e-6
CONTROLS = (
    "gxBQ_window_00_frame_0",
    "gxBQ_window_00_frame_5",
    "letterboxed_short_frame_78",
    "am2_window_00_frame_150",
    "am3_window_00_frame_0",
)


def identity(row: dict, label_key: str) -> tuple[str, str]:
    return row["case_id"], row[label_key]


def image_for_case(case: dict) -> tuple[str, list[int]]:
    path = ROOT / "colour_consistency/gallery" / f"{case['case_id']}.jpg"
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    actual = [image.shape[1], image.shape[0]]
    if actual != case["working_size_wh"]:
        raise ValueError(f"{case['case_id']}: image dimensions {actual} != {case['working_size_wh']}")
    return os.path.relpath(path, OUTPUT), actual


def equal_array(left: object, right: object, description: str) -> None:
    if not np.array_equal(left, right):
        raise ValueError(f"{description} differs")


def checked_case(case: dict, result: dict, request: dict) -> tuple[dict, dict]:
    case_id, label = identity(case, "source_label")
    key = case["choices"]["primary"]
    if key is None or key != result["selected_origin_key"] or key != request["origin_key"]:
        raise ValueError(f"{case_id}/{label}: polarity input is not the bounded primary selection")
    if case["source_record"] != request["source_record"]:
        raise ValueError(f"{case_id}/{label}: request source record differs")
    if Path(result["source_record"]).resolve() != (REPO / case["source_record"]).resolve():
        raise ValueError(f"{case_id}/{label}: result source record differs")
    if case["frame_md5"] != result["frame_md5"]:
        raise ValueError(f"{case_id}/{label}: source frame differs")

    selected = case["selected_geometry"][key]
    polarity_selected = result["selected_geometry"]
    for field in ("corners_native_px", "homography_working"):
        equal_array(selected[field], polarity_selected[field], f"{case_id}/{label}: selected {field}")
    expected_working = np.asarray(polarity_selected["corners_native_px"]) * (
        np.asarray(case["working_size_wh"]) / np.asarray(case["native_size_wh"])
    )
    if not np.allclose(expected_working, selected["corners_working_px"], rtol=0, atol=CORNER_TOLERANCE_PX):
        raise ValueError(f"{case_id}/{label}: selected native-to-working conversion differs")
    source_record = read(REPO / case["source_record"])
    source_candidates = {item["origin_key"]: item for item in source_record["parents"] + source_record["valid_children"]}
    if source_record["case_id"] != case_id or key not in source_candidates:
        raise ValueError(f"{case_id}/{label}: bounded primary source is missing")
    equal_array(source_candidates[key]["corners_px"], selected["corners_native_px"],
                f"{case_id}/{label}: bounded primary source corners")
    before = project_selection(selected, case)
    equal_array(before["corners"], selected["corners_working_px"], f"{case_id}/{label}: displayed before corners")

    corrected = result["corrected"]
    if corrected["status"] != corrected["fit"]["status"]:
        raise ValueError(f"{case_id}/{label}: corrected fit status differs")
    for field in ("corners_working_px", "corners_native_px"):
        equal_array(corrected[field], corrected["measurement"][field], f"{case_id}/{label}: corrected {field}")
    corrected_selection = {**corrected, "native_size_wh": case["native_size_wh"],
                           "working_size_wh": case["working_size_wh"]}
    after = project_selection(corrected_selection, case)
    equal_array(after["corners"], corrected["corners_working_px"], f"{case_id}/{label}: displayed after corners")
    if not np.allclose(corrected["fit"]["corners_px"], corrected["corners_working_px"],
                       rtol=0, atol=CORNER_TOLERANCE_PX):
        raise ValueError(f"{case_id}/{label}: fit and displayed corrected corners differ")
    replay_difference = result["original_replay"]["maximum_absolute_difference_native_px"]
    if replay_difference > REPLAY_TOLERANCE_NATIVE_PX:
        raise ValueError(f"{case_id}/{label}: original replay exceeds tolerance: {replay_difference}")
    automatic = result["automatic"]
    if not automatic["unchanged_constraint_arrays"]:
        raise ValueError(f"{case_id}/{label}: parent constraint arrays changed")
    movement = np.linalg.norm(np.asarray(after["corners"]) - np.asarray(before["corners"]), axis=1)
    image, image_size = image_for_case(case)
    row = {
        "case_id": case_id, "source_label": label, "cohort": case["cohort"],
        "image": image, "size": image_size, "before": before, "after": after,
        "changed_fragments": automatic["changed_fragment_count"],
        "unresolved_fragments": automatic["unresolved_fragment_count"],
        "fragment_count": automatic["fragment_count"], "fit_status": corrected["status"],
        "fit_valid": corrected["valid"], "camera_eligible": corrected["measurement"]["camera_eligible"],
        "maximum_corner_movement_working_px": float(movement.max()), "original_replay_matched": True,
    }
    audit = {
        "case_id": case_id, "source_label": label, "selected_origin_key": key,
        "source_record": case["source_record"], "frame_path": case["frame_path"],
        "frame_md5": case["frame_md5"], "image": image, "image_size_wh": image_size,
        "image_path_exists": True, "image_dimensions_equal": True,
        "bounded_primary_source_equal": True, "polarity_selected_equal": True,
        "before_geometry_equal": True, "corrected_fit_output_equal": True,
        "corrected_measurement_equal": True, "after_geometry_equal": True,
        "native_working_conversion_checked": True, "homography_projection_checked": True,
        "original_replay_maximum_absolute_difference_native_px": replay_difference,
    }
    return row, audit


def order_rows(rows: list[dict]) -> list[dict]:
    priority = {case_id: index + 1 for index, case_id in enumerate(CONTROLS)}
    return sorted(rows, key=lambda row: (0 if row["source_label"] == "am1_seeded_pool" else
                                         priority.get(row["case_id"], len(priority) + 1), row["case_id"]))


def page(rows: list[dict]) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    template = replace_once(template, "SVD12 search depth · court gallery", "Stripe-polarity correction · court gallery")
    template = replace_once(template, "</style>",
                            ".badge{display:inline-block;border:1px solid #8195a8;border-radius:4px;"
                            "padding:3px 7px;background:#253240}\n</style>")
    template = replace_once(template, "<h1>SVD12 search-depth comparison</h1>",
                            "<h1>Selected court · stripe-polarity correction</h1>")
    start = template.index("<p>Three independent automatic G0 searches")
    end = template.index('<nav id="nav"></nav>', start)
    template = template[:start] + (
        "<p>The left image shows the current bounded primary court selection. The right image shows the saved "
        "automatic stripe-polarity fit. Correction changes centre and edge labels of the same original parent "
        "constraints; it does not change net weight or court selection.</p>\n"
        "<p>Maximum corner movement is a working-pixel diagnostic, not an error or quality score. "
        "Fit status and camera gate are shown for every case. Court markings and net tape need not share a colour; "
        "matching uses local court-stripe contrast. Net and post overlays belong to the selection stage and were "
        "not recalculated after this fit.</p>\n"
        "<p>Click a full image to focus both enlarged crops. Select a corner to jump there. "
        "Hold Space to hide overlays temporarily.</p>\n"
        '<p><a href="../bounded_gallery/">View bounded court selection gallery</a></p>\n'
    ) + template[end:]
    start = template.index('<label>Show <select id="view">')
    end = template.index('<label>Overlay <select id="mode">', start)
    template = template[:start] + template[end:]
    template = replace_once(template,
                            '<label><input id="reference" type="checkbox"> Reference annotation (dashed)</label>', '')
    start = template.index('<script>')
    end = template.index('</script>', start) + len('</script>')
    script = (OUTPUT / "gallery.js").read_text(encoding="utf-8")
    script = replace_once(script, "__GALLERY_DATA__", json.dumps(rows, allow_nan=False, separators=(",", ":")))
    return template[:start] + '<script>\n' + script + '\n</script>' + template[end:]


def main() -> None:
    trial = read(TRIAL)
    requests = read(REQUESTS)["selections"]
    results = read(RESULTS)
    if trial["schema"] != "bounded-post-base-scoring-replay/1" or not trial["complete"]:
        raise ValueError("bounded trial is incomplete")
    if results["schema"] != "selected-polarity-refit/1":
        raise ValueError("unexpected polarity replay schema")
    selected = results["selections"]
    if len(selected) != 20 or len(requests) != 20:
        raise ValueError("expected 20 completed polarity requests and results")
    cases = {identity(case, "source_label"): case for case in trial["cases"]}
    requested = {identity(request, "label"): request for request in requests}
    fitted = {identity(result, "label"): result for result in selected}
    if len(cases) != len(trial["cases"]) or len(requested) != 20 or len(fitted) != 20 or set(requested) != set(fitted):
        raise ValueError("case identities are duplicated or differ")
    if not set(fitted) <= set(cases):
        raise ValueError("polarity case missing from bounded trial")
    OUTPUT.mkdir(exist_ok=True)
    rows, audits = [], []
    for case_identity, result in fitted.items():
        row, audit = checked_case(cases[case_identity], result, requested[case_identity])
        rows.append(row)
        audits.append(audit)
    rows = order_rows(rows)
    (OUTPUT / "index.html").write_text(page(rows), encoding="utf-8")
    audit = {"schema": "polarity-gallery-source-audit/1", "source": str(RESULTS.relative_to(REPO)),
             "cases": sorted(audits, key=lambda item: (item["case_id"], item["source_label"]))}
    audit_bytes = json.dumps(audit, allow_nan=False, separators=(",", ":")).encode()
    (OUTPUT / "source_audit.json.gz").write_bytes(gzip.compress(audit_bytes, mtime=0))
    print(f"Built {len(rows)} polarity rows")


if __name__ == "__main__":
    main()
