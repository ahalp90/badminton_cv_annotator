"""Build a two-choice review gallery from the completed saved-pool net scan."""

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
OUTPUT = BASE / "gallery"
TEMPLATE = ROOT / "svd_search/gallery_template.html"
SCAN = BASE / "saved_net_scan.json.gz"
PRIORITY_CASE = "letterboxed_short_frame_78"
REQUESTED_CASES = ("gxBQ_window_00_frame_5", "gxBQ_window_00_frame_0")
PIECES = ("tape_left", "tape_right", "post_left", "post_right")
CORNER_TOLERANCE_PX = 1e-6
HOMOGRAPHY_CORNER_TOLERANCE_PX = 0.02


def read_scan() -> dict:
    with gzip.open(SCAN, "rt", encoding="utf-8") as stream:
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
    return {
        "corners": working.tolist(),
        "centres": centres.reshape(-1, 2, 2).tolist(),
        "edges": edges.reshape(-1, 2, 2).tolist(),
    }


def image_for_case(case: dict) -> tuple[str, list[int]]:
    case_id = case["case_id"]
    shared = ROOT / "colour_consistency/gallery" / f"{case_id}.jpg"
    working_size = case["working_size_wh"]
    if shared.is_file():
        image = cv2.imread(str(shared))
        if image is not None and [image.shape[1], image.shape[0]] == working_size:
            path = shared
        else:
            path = OUTPUT / f"{case_id}.jpg"
    else:
        path = OUTPUT / f"{case_id}.jpg"
    if path != shared:
        frame = cv2.imread(str(ROOT / case["frame_path"]))
        if frame is None:
            raise FileNotFoundError(f"{case_id}: frozen frame {case['frame_path']}")
        resized = cv2.resize(frame, tuple(working_size), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(path), resized):
            raise OSError(f"{case_id}: could not write working JPEG")
    image = cv2.imread(str(path))
    actual_size = [image.shape[1], image.shape[0]] if image is not None else None
    if actual_size != working_size:
        raise ValueError(f"{case_id}: gallery image size {actual_size} != {working_size}")
    return os.path.relpath(path, OUTPUT), actual_size


def support_note(selection: dict) -> str:
    if selection["net"]["state"] != "measured":
        return f"Net support {selection['net']['state'].replace('_', ' ')} (neutral)"
    pieces = []
    for name in PIECES:
        coverage = selection["coverage"][name]
        visible = selection["visible_samples"][name]
        fraction = "—" if coverage is None else f"{coverage:.0%}"
        pieces.append(f"{name.replace('_', ' ')} {fraction} ({visible} visible)")
    return "Net support: " + "; ".join(pieces)


def reference_note(case: dict, role: str) -> str:
    diagnostic = case["retrospective_reference_diagnostic_only"]
    if diagnostic["status"] != "measured":
        return ""
    error = diagnostic[role]
    return f"; retrospective max corner error {error['all_max_native_px']:.1f} native px"


def make_view(case: dict, role: str) -> dict:
    selection = case[role]
    if selection is None:
        raise ValueError(f"{case['case_id']}: {role} has no selection")
    matches = [candidate for candidate in case["candidates"]
               if candidate["origin_key"] == selection["origin_key"]]
    if len(matches) != 1:
        raise ValueError(f"{case['case_id']}: {role} origin key is not unique in pool")
    candidate = matches[0]
    for field in ("candidate_id", "source", "original_rank", "full_court_rank"):
        if selection[field] != candidate[field]:
            raise ValueError(f"{case['case_id']}: {role} {field} differs from pool row")
    geometry = project_selection(selection, case)
    title = "Saved W5" if role == "baseline" else "Fixed net preference"
    detail = (f"{selection['origin_key']} · source {selection['source']} · candidate ID {selection['candidate_id']}"
              f" · full-court rank {selection['full_court_rank']}; paint-order rank {selection['original_rank']}"
              f" · {support_note(selection)}{reference_note(case, role)}")
    return {"title": title, "valid": True, "status": "selected", "geometry": geometry,
            "net_pieces_working_px": selection["net"]["pieces_working_px"],
            "origin_key": selection["origin_key"], "source": selection["source"],
            "candidate_id": selection["candidate_id"], "detail": detail}


def make_row(case: dict) -> tuple[dict, dict]:
    baseline = make_view(case, "baseline")
    trial = make_view(case, "trial")
    image, actual_size = image_for_case(case)
    baseline_selection = case["baseline"]
    trial_selection = case["trial"]
    gap = case["paint_score_gap_trial_minus_baseline"]
    gap_note = "same selection" if gap is None else f"paint score gap (fixed − saved) {gap:+.4f}"
    label = case.get("reference_status")
    control_note = f" · control {label.replace('_', ' ')}" if label else ""
    context = (f"{case['group']}{control_note} · full-court rank "
               f"{baseline_selection['full_court_rank']} → {trial_selection['full_court_rank']}"
               f" · paint-order rank {baseline_selection['original_rank']} → {trial_selection['original_rank']}"
               f" · {gap_note} · native {case['native_size_wh'][0]} × {case['native_size_wh'][1]} px"
               f"; working {case['working_size_wh'][0]} × {case['working_size_wh'][1]} px")
    if case["case_id"] == PRIORITY_CASE:
        context = "Priority concern · " + context
    row = {"gallery_mode": "decision_trials", "case_id": case["case_id"], "image": image,
           "size": case["working_size_wh"], "decision_views": {"original": baseline, "automatic": trial},
           "saved_selected": baseline["geometry"]["corners"], "context": context}
    audit = {"case_id": case["case_id"], "source_record": case["source_record"],
             "frame_path": case["frame_path"], "image": image, "image_size_wh": actual_size,
             "native_size_wh": case["native_size_wh"], "working_size_wh": case["working_size_wh"],
             "changed": case["changed"], "reference_status": label, "displayed": {}}
    for name, selection, view in (("original", baseline_selection, baseline),
                                  ("automatic", trial_selection, trial)):
        audit["displayed"][name] = {
            "origin_key": view["origin_key"], "source": view["source"],
            "candidate_id": view["candidate_id"], "corners_working_px": view["geometry"]["corners"],
            "net_pieces_working_px": view["net_pieces_working_px"],
            "source_net_pieces_working_px": selection["net"]["pieces_working_px"],
            "source_corners_native_px": selection["corners_native_px"],
            "source_corners_working_px": selection["corners_working_px"],
        }
    return row, audit


def gallery_page(rows: list[dict]) -> str:
    page = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(rows, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    page = replace_once(page, "__GALLERY_DATA__", payload)
    page = replace_once(page, "<title>SVD12 search depth · court gallery</title>",
                        "<title>Fixed net preference — wider review</title>")
    page = replace_once(page, '<label><input id="outlines" type="checkbox" checked> Show outlines</label>',
                        '<label><input id="outlines" type="checkbox" checked> Show outlines</label>\n'
                        '<label><input id="net-projection" type="checkbox"> Projected net tape and posts</label>')
    page = replace_once(page,
                        "const controls=['view','mode','width','opacity','reference','outlines','smooth']",
                        "const controls=['view','mode','width','opacity','reference','outlines','net-projection','smooth']")
    page = replace_once(page,
                        "const decisionNames={original:'Original fit',earlier:'Earlier bright-paint rule',"
                        "automatic:'Automatic stripe polarity'};",
                        "const decisionNames={original:'Saved W5',automatic:'Fixed net preference'};")
    page = replace_once(page,
                        "const decisionColours={original:'#e9eef2',earlier:'#ffbd67',automatic:'#69d9f0'};",
                        "const decisionColours={original:'#e9eef2',automatic:'#69d9f0'};")
    start = page.index("if(decisionMode){\n document.title=")
    end = page.index("if(colourMode){", start)
    page = page[:start] + """if(decisionMode){
 document.title='Fixed net preference — wider review';
 document.querySelector('h1').textContent='Fixed net preference — wider review';
 const paragraphs=document.querySelectorAll('header p');
 paragraphs[0].textContent='The fixed net preference changed 13 of 71 completed SAVED-pool selections. This gallery generated no new candidates. The separate GX5 combined pass is not shown here.';
 paragraphs[1].textContent='Some selections jump far down the saved paint order. Letterboxed78 is the priority concern. Visual review is needed before core promotion. Controls marked unlabelled have no non-court label.';
 paragraphs[2].textContent='Retrospective corner errors use native pixels only where reference corners exist. Annotation conventions differ and off-image corners limit small comparisons. Click an image to focus both crops; hold Space to hide outlines.';
 document.getElementById('view').closest('label').hidden=true;
 document.getElementById('reference').closest('label').hidden=true;
}
""" + page[end:]
    start = page.index("function colourSource(data,name,view){")
    end = page.index("for(const data of cases){", start)
    page = page[:start] + "const redraw=[];\n" + page[end:]
    start = page.index(" if(decisionMode){const context=document.createElement('p');")
    end = page.index("\n const names=decisionMode?", start)
    page = (page[:start] + " if(decisionMode){const context=document.createElement('p');"
            "context.className='small';context.textContent=data.context;section.append(context);}"
            + page[end:])
    page = replace_once(page, "const names=decisionMode?['original','earlier','automatic']:",
                        "const names=decisionMode?['original','automatic']:")
    page = replace_once(page,
                        "data.saved_selected||data.saved_w5_geometry?.corners||"
                        "data.decision_views.earlier.geometry?.corners||data.decision_views.automatic.geometry?.corners",
                        "data.saved_selected||data.decision_views.automatic.geometry?.corners")
    page = replace_once(page,
                        "Solid: original. Dashed: earlier bright-paint rule. Dash-dot: automatic stripe polarity. "
                        "Invalid or unavailable fits have no outline.",
                        "Solid: saved W5. Dash-dot: fixed net preference. Amber dots show projected net tape halves "
                        "and posts, not detected net. Both panels show actual selections.")
    page = replace_once(page, " if(decisionMode)decisionEvidence(section,data);\n", "")
    page = replace_once(page, " if(colourMode)colourEvidence(section,data,names);\n", "")
    page = replace_once(page,
                        "const dashes=decisionMode?(name==='earlier'?[12,7]:name==='automatic'?[18,6,3,6]:[]):",
                        "const dashes=decisionMode?(name==='automatic'?[18,6,3,6]:[]):")
    page = replace_once(page,
                        " if(decisionMode&&name==='original'&&data.saved_w5_geometry){",
                        " if(decisionMode&&document.getElementById('net-projection').checked){\n"
                        "  strokeLines(ctx,data.decision_views[name].net_pieces_working_px,'#ffbd67',thickness,scale,[2,6]);\n"
                        " }\n"
                        " if(decisionMode&&name==='original'&&data.saved_w5_geometry){")
    start = page.index("   if(decisionMode){\n    const fit=data.decision_views[name];title.textContent=")
    end = page.index("   }else if(colourMode){", start)
    page = page[:start] + """   if(decisionMode){
    const fit=data.decision_views[name];
    title.textContent=fit.title;
    detail.textContent=fit.detail;
""" + page[end:]
    return page


def write_audit(audit: dict) -> None:
    payload = json.dumps(audit, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with (
        (OUTPUT / "source_audit.json.gz").open("wb") as file,
        gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as stream,
    ):
        stream.write(payload)


def main() -> None:
    scan = read_scan()
    if (scan["schema"] != "saved-net-scan/1" or not scan["complete"]
            or scan["completed_case_count"] != scan["planned_case_count"] != 71
            or scan["changed_selection_count"] != 13):
        raise ValueError("expected the completed 71-case, 13-change saved scan")
    by_id = {case["case_id"]: case for case in scan["cases"]}
    if len(by_id) != 71 or by_id[PRIORITY_CASE]["changed"] is not True:
        raise ValueError("case identity or priority case changed")
    if any(by_id[case_id]["changed"] for case_id in REQUESTED_CASES):
        raise ValueError("requested GX cases are no longer unchanged")
    changed = [case for case in scan["cases"] if case["changed"]]
    if len(changed) != 13:
        raise ValueError("changed-case count differs from scan summary")
    ordered = [by_id[PRIORITY_CASE], *(by_id[case_id] for case_id in REQUESTED_CASES)]
    ordered.extend(case for case in changed if case["case_id"] != PRIORITY_CASE)
    if len(ordered) != 15 or len({case["case_id"] for case in ordered}) != 15:
        raise ValueError("gallery must have 15 distinct cases")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    audits = []
    for case in ordered:
        row, audit = make_row(case)
        rows.append(row)
        audits.append(audit)
    (OUTPUT / "index.html").write_text(gallery_page(rows), encoding="utf-8")
    write_audit({"schema": "net-gallery-source-audit/1", "source_scan": str(SCAN.relative_to(ROOT)),
                 "case_count": len(rows), "corner_tolerance_working_px": CORNER_TOLERANCE_PX,
                 "homography_corner_tolerance_working_px": HOMOGRAPHY_CORNER_TOLERANCE_PX,
                 "cases": audits})
    print(f"Built {len(rows)} cases: {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
