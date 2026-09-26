"""Build the review gallery from the completed bounded lower-post replay."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import cv2
import numpy as np

from experiments.annotator.independent_court import detector, paint_geometry

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
REPO = ROOT.parents[1]
OUTPUT = BASE / "bounded_gallery"
TRIAL = REPO / "local_scratch/net_recovery/20260923/bounded/bounded_trial.json.gz"
TEMPLATE = ROOT / "svd_search/gallery_template.html"
SCRIPT = OUTPUT / "gallery.js"
SAVED = BASE / "saved_net_scan.json.gz"
SEEDED = ROOT / "colour_consistency/am1_net_selection_trial.json.gz"
CORNER_TOLERANCE_PX = 1e-6
HOMOGRAPHY_CORNER_TOLERANCE_PX = 0.02


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def replace_once(page: str, old: str, new: str) -> str:
    if page.count(old) != 1:
        raise ValueError(f"shared gallery template changed around {old[:60]!r}")
    return page.replace(old, new)


def project_selection(selection: dict, case: dict) -> dict:
    native = np.asarray(selection["corners_native_px"], dtype=float)
    working = np.asarray(selection["corners_working_px"], dtype=float)
    native_size = np.asarray(selection["native_size_wh"], dtype=float)
    working_size = np.asarray(selection["working_size_wh"], dtype=float)
    if native.shape != (4, 2) or working.shape != (4, 2):
        raise ValueError(f"{case['case_id']}: expected four x/y court corners")
    if not np.array_equal(native_size, case["native_size_wh"]):
        raise ValueError(f"{case['case_id']}: native sizes differ")
    if not np.array_equal(working_size, case["working_size_wh"]):
        raise ValueError(f"{case['case_id']}: working sizes differ")
    expected = native * working_size / native_size
    if not np.allclose(working, expected, rtol=0, atol=CORNER_TOLERANCE_PX):
        raise ValueError(f"{case['case_id']}: source corners are not converted exactly once")
    homography = np.asarray(selection["homography_working"], dtype=float)
    projected_corners = detector.project(homography[None], detector.CORNER_COURT_M)[0][0]
    if not np.allclose(projected_corners, working, rtol=0, atol=HOMOGRAPHY_CORNER_TOLERANCE_PX):
        raise ValueError(f"{case['case_id']}: working homography differs from selected corners")
    centres = detector.project(homography[None], paint_geometry.CENTRE_SEGMENTS_M)[0][0]
    intervals = np.repeat(np.arange(12), 2)
    positions = np.tile([1, 2], 12)
    stripes = paint_geometry.positioned_segments(paint_geometry.CENTRE_SEGMENTS_M, intervals, positions)
    edges = detector.project(homography[None], stripes)[0][0]
    return {"corners": working.tolist(), "centres": centres.reshape(-1, 2, 2).tolist(),
            "edges": edges.reshape(-1, 2, 2).tolist()}


def image_for_case(case: dict) -> tuple[str, list[int]]:
    case_id = case["case_id"]
    shared = ROOT / "colour_consistency/gallery" / f"{case_id}.jpg"
    working_size = case["working_size_wh"]
    image = cv2.imread(str(shared)) if shared.is_file() else None
    if image is not None and [image.shape[1], image.shape[0]] == working_size:
        path = shared
    else:
        path = OUTPUT / f"{case_id}.jpg"
        frame = cv2.imread(str(ROOT / case["frame_path"]))
        if frame is None:
            raise FileNotFoundError(case["frame_path"])
        resized = cv2.resize(frame, tuple(working_size), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(path), resized):
            raise OSError(f"could not write {path}")
    image = cv2.imread(str(path))
    actual = [image.shape[1], image.shape[0]] if image is not None else None
    if actual != working_size:
        raise ValueError(f"{case_id}: image dimensions {actual} != {working_size}")
    return os.path.relpath(path, OUTPUT), actual


def checked_case(case: dict, saved: dict) -> tuple[dict, dict]:
    identity = (case["case_id"], case["source_label"])
    source = saved[identity]
    baseline = source["baseline"]
    trial = source["trial"]
    if case["baseline_key"] != (baseline and baseline["origin_key"]):
        raise ValueError(f"{identity}: baseline differs from saved source")
    if case["old_fixed_trial_key"] != (trial and trial["origin_key"]):
        raise ValueError(f"{identity}: old net choice differs from saved source")
    rows = {row["origin_key"]: row for row in case["candidate_rows"]}
    geometries = case["selected_geometry"]
    keys = {case["baseline_key"], case["old_fixed_trial_key"], *case["choices"].values()}
    keys.discard(None)
    if set(geometries) != keys or not keys <= rows.keys():
        raise ValueError(f"{identity}: selected geometry or candidate row missing")
    record = read(REPO / case["source_record"])
    if record["case_id"] != case["case_id"]:
        raise ValueError(f"{identity}: source record differs")
    source_candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
    views = {}
    geometry_audit = []
    for key in sorted(keys):
        selection = geometries[key]
        row = rows[key]
        candidate = source_candidates[key]
        if key != selection["origin_key"] or row["candidate_id"] != selection["candidate_id"]:
            raise ValueError(f"{identity}: selection identity differs at {key}")
        if not np.array_equal(candidate["corners_px"], selection["corners_native_px"]):
            raise ValueError(f"{identity}: source geometry differs at {key}")
        if row["paint_score"] != candidate["evidence"][case["ordering_criterion"]]:
            raise ValueError(f"{identity}: source paint score differs at {key}")
        for role, stored in (("baseline", baseline), ("old_fixed", trial)):
            if stored is not None and key == stored["origin_key"]:
                for field in ("corners_native_px", "corners_working_px", "homography_working"):
                    if not np.array_equal(selection[field], stored[field]):
                        raise ValueError(f"{identity}: {role} {field} differs at {key}")
        geometry = project_selection(selection, case)
        net = selection["net"]
        if net["state"] == "measured":
            pieces = np.asarray(net["pieces_working_px"])
            native_pieces = np.asarray(net["pieces_native_px"])
            scale = np.asarray(case["native_size_wh"]) / np.asarray(case["working_size_wh"])
            if not np.allclose(pieces * scale, native_pieces, rtol=0, atol=1e-9):
                raise ValueError(f"{identity}: net projection scale differs at {key}")
        views[key] = {"geometry": geometry, "net": net, "row": row}
        geometry_audit.append({"origin_key": key, "candidate_id": row["candidate_id"],
                               "source_record": case["source_record"], "native_corners_equal": True,
                               "working_scale_equal": True, "homography_equal": True})
    settings_audit = {}
    for name, key in case["choices"].items():
        scores = case["scores"][name]
        score_rows = {score["origin_key"]: score for score in scores}
        if set(score_rows) != {row_key for row_key, row in rows.items() if row["historical_fullcourt"]}:
            raise ValueError(f"{identity}: {name} score pool differs")
        for score in scores:
            row = rows[score["origin_key"]]
            if score["paint_score"] != row["paint_score"] or score["full_court_rank"] != row["full_court_rank"]:
                raise ValueError(f"{identity}: {name} score source differs")
        if key != (max(scores, key=lambda score: score["combined_score"])["origin_key"] if scores else None):
            raise ValueError(f"{identity}: {name} selected score differs")
        settings_audit[name] = {"origin_key": key, "selected_score_equal": True}
    if case["choices"]["zero"] != case["baseline_key"]:
        raise ValueError(f"{identity}: zero weight differs from baseline")
    image, size = image_for_case(case)
    used_segment_ids = set()
    for key in keys:
        for feature in views[key]["row"]["posts"].values():
            used_segment_ids.update(feature["covering_ids"])
    segments = case["segments_working_px"]
    if any(index < 0 or index >= len(segments) for index in used_segment_ids):
        raise ValueError(f"{identity}: covering segment ID outside source")
    selected_scores = {}
    for name, scores in case["scores"].items():
        selected_scores[name] = next((score for score in scores if score["origin_key"] == case["choices"][name]), None)
    baseline_score = next((score for score in case["scores"]["primary"]
                           if score["origin_key"] == case["baseline_key"]), None)
    row = {"case_id": case["case_id"], "source_label": case["source_label"], "cohort": case["cohort"],
           "group": case["group"], "scan_label": case["scan_label"], "view_status": case["view_status"],
           "reference_status": case["reference_status"], "image": image, "size": size,
           "baseline_key": case["baseline_key"], "old_fixed_trial_key": case["old_fixed_trial_key"],
           "choices": case["choices"], "selected_scores": selected_scores, "baseline_score": baseline_score,
           "views": views,
           "segments": {str(index): segments[index] for index in sorted(used_segment_ids)}}
    audit = {"case_id": case["case_id"], "source_label": case["source_label"],
             "source_scan": case["source_scan"], "source_record": case["source_record"],
             "frame_path": case["frame_path"], "image": image, "image_size_wh": size,
             "geometries": geometry_audit, "settings": settings_audit,
             "baseline_key": case["baseline_key"], "old_fixed_trial_key": case["old_fixed_trial_key"]}
    return row, audit


def order_rows(rows: list[dict], reviewed: set[str]) -> list[dict]:
    def priority(row: dict) -> tuple[int, str]:
        seeded = row["source_label"] == "am1_seeded_pool"
        original = row["source_label"] == "frozen71" and row["case_id"] in reviewed
        primary = row["choices"]["primary"] != row["baseline_key"]
        sensitive = any(value != row["choices"]["primary"] for value in row["choices"].values())
        group = (0 if seeded else 1 if original and primary else 2 if primary else
                 3 if original else 4 if sensitive else 5)
        return group, row["case_id"]
    return sorted(rows, key=priority)


def page(rows: list[dict]) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    template = replace_once(template, "SVD12 search depth · court gallery", "Bounded lower-post choices · court gallery")
    template = replace_once(template, "</style>",
                            ".badge{display:inline-block;border:1px solid #8195a8;border-radius:4px;"
                            "padding:3px 7px;background:#253240}\n</style>")
    template = replace_once(template, "<h1>SVD12 search-depth comparison</h1>",
                            "<h1>Bounded lower-post choices · experimental visual review</h1>")
    start = template.index("<p>Three independent automatic G0 searches")
    end = template.index("<nav id=\"nav\"></nav>", start)
    template = template[:start] + ("<p>Court markings and net tape need not share a colour; matching uses segment geometry. "
                                  "The displayed court and net overlays use separate colours. Saved court geometry "
                                  "is unchanged by this selection trial. "
                                  "Automatic stripe correction is not applied here. "
                                  "Projected tape and tops are diagnostic only and do not enter the score.</p>\n"
                                  "<p>The bounded lower-post score is experimental and pending visual review. "
                                  "Paint score and net bonus are score units, not probabilities. "
                                  "Centre and Am4 were withheld from the current net design review, "
                                  "but were used in earlier detector experiments. Their views are in this 72-case pool. "
                                  "The left panel stays on the saved paint choice. "
                                  "The right panel shows only computed choices.</p>\n"
                                  "<p>Lower-quarter post support usually does not match the actual base: "
                                  "only 4 of 26 supported posts on changed choices match the base sample. "
                                  "The paint-score cap avoids the known regressions. "
                                  "The below-base fragment veto changes no current winner. "
                                  "Weight 0.04 is provisional.</p>\n"
                                  "<p>Click a full image to focus both enlarged crops. Select a corner to jump there. "
                                  "Hold Space to hide overlays temporarily.</p>\n"
                                  '<p><a href="../polarity_gallery/">View selected-court fitting comparison</a></p>\n') + template[end:]
    start = template.index('<label>Show <select id="view">')
    end = template.index('<label>Overlay <select id="mode">', start)
    template = template[:start] + ('<label>Right choice <select id="view">'
                                   '<option value="primary">Bounded · weight 0.04 · overrun 4 px</option>'
                                   '<option value="weight_02">Bounded · weight 0.02 · overrun 4 px</option>'
                                   '<option value="weight_08">Bounded · weight 0.08 · overrun 4 px</option>'
                                   '<option value="overrun_02">Bounded · weight 0.04 · overrun 2 px</option>'
                                   '<option value="overrun_08">Bounded · weight 0.04 · overrun 8 px</option>'
                                   '<option value="old_fixed">Old unrestricted net choice</option>'
                                   '</select></label>\n'
                                   '<label>Rows <select id="filter">'
                                   '<option value="default">Primary changes + original review + seeded Am1</option>'
                                   '<option value="sensitivity">All sensitivity changes</option>'
                                   '<option value="all">All 72 cases</option>'
                                   '</select></label>\n') + template[end:]
    template = replace_once(template,
                            '<label><input id="reference" type="checkbox"> Reference annotation (dashed)</label>',
                            '<label><input id="projected" type="checkbox"> Projected net (diagnostic)</label>\n'
                            '<label><input id="base" type="checkbox"> Lower-post evidence</label>')
    start = template.index('<script>')
    end = template.index('</script>', start) + len('</script>')
    script = SCRIPT.read_text(encoding="utf-8")
    script = script.replace('__GALLERY_DATA__', json.dumps(rows, allow_nan=False, separators=(",", ":")))
    return template[:start] + '<script>\n' + script + '\n</script>' + template[end:]


def main() -> None:
    trial = read(TRIAL)
    if trial["schema"] != "bounded-post-base-scoring-replay/1" or not trial["complete"] or len(trial["cases"]) != 72:
        raise ValueError("bounded replay must be complete with 72 rows")
    expected_settings = {"primary": (0.04, 4.0), "weight_02": (0.02, 4.0),
                         "weight_08": (0.08, 4.0), "overrun_02": (0.04, 2.0),
                         "overrun_08": (0.04, 8.0), "zero": (0.0, 4.0)}
    actual_settings = {setting["name"]: (setting["weight"], setting["overrun_working_px"])
                       for setting in trial["settings"]}
    if actual_settings != expected_settings:
        raise ValueError("bounded replay settings differ")
    old_scan = read(SAVED)
    am1_scan = read(SEEDED)
    seeded = [case for case in am1_scan["cases"] if case["label"] == "am1_seeded_pool"]
    if len(old_scan["cases"]) != 71 or len(seeded) != 1:
        raise ValueError("source scans differ")
    sources = {(case["case_id"], "frozen71"): case for case in old_scan["cases"]}
    sources[(seeded[0]["case_id"], "am1_seeded_pool")] = seeded[0]
    identities = [(case["case_id"], case["source_label"]) for case in trial["cases"]]
    if len(set(identities)) != 72 or set(identities) != set(sources):
        raise ValueError("replay/source identities differ")
    reviewed = {case["case_id"] for case in old_scan["cases"] if case["changed"]}
    reviewed.update(("gxBQ_window_00_frame_5", "gxBQ_window_00_frame_0"))
    if len(reviewed) != 15:
        raise ValueError("original review set differs")
    OUTPUT.mkdir(exist_ok=True)
    rows, audits = [], []
    for case in trial["cases"]:
        row, audit = checked_case(case, sources)
        row["original_review"] = row["source_label"] == "frozen71" and row["case_id"] in reviewed
        rows.append(row)
        audits.append(audit)
    ordered = order_rows(rows, reviewed)
    (OUTPUT / "index.html").write_text(page(ordered), encoding="utf-8")
    audit_bytes = json.dumps({"schema": "bounded-gallery-source-audit/1", "source": str(TRIAL.relative_to(REPO)),
                              "settings": trial["settings"], "cases": audits}, separators=(",", ":")).encode()
    (OUTPUT / "source_audit.json.gz").write_bytes(gzip.compress(audit_bytes, mtime=0))
    print(f"Built {len(rows)} rows with {sum(row['original_review'] for row in rows)} original review rows")


if __name__ == "__main__":
    main()
