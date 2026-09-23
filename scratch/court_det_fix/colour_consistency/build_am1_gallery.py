"""Build a three-case review of Am1 seeding and the experimental net preference."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from experiments.annotator.independent_court import detector, paint_geometry

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "am1_gallery"
TEMPLATE = BASE.parent / "svd_search/gallery_template.html"
RECOVERY = BASE / "am1_recovery_trial.json.gz"
NET_TRIAL = BASE / "am1_net_selection_trial.json.gz"
CASE_LABELS = (
    ("am1_window_00_frame_54", "Am1", "am1_seeded_pool", "am1_saved_baseline"),
    ("gxBQ_window_00_frame_5", "GX5", "gx5_retention", None),
    ("letterboxed_short_frame_45", "Letterboxed45", "letterboxed45_retention", None),
)
CORNER_TOLERANCE_PX = 1e-6
HOMOGRAPHY_CORNER_TOLERANCE_PX = 0.02


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def replace_once(page: str, old: str, new: str) -> str:
    if page.count(old) != 1:
        raise ValueError(f"shared gallery template changed around {old[:60]!r}")
    return page.replace(old, new)


def project_selection(selection: dict | None, case: dict) -> dict | None:
    if selection is None:
        return None
    native = np.asarray(selection["corners_native_px"], dtype=float)
    working = np.asarray(selection["corners_working_px"], dtype=float)
    native_size = np.asarray(selection["native_size_wh"], dtype=float)
    working_size = np.asarray(selection["working_size_wh"], dtype=float)
    if native.shape != (4, 2) or working.shape != (4, 2):
        raise ValueError(f"{case['case_id']}: expected four x/y court corners")
    if not np.array_equal(working_size, case["working_size_wh"]):
        raise ValueError(f"{case['case_id']}: selection and image working sizes differ")
    if not np.array_equal(native_size, case["native_size_wh"]):
        raise ValueError(f"{case['case_id']}: selection and case native sizes differ")
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
    geometry = {"corners": working.tolist(), "centres": centres.reshape(-1, 2, 2).tolist(),
                "edges": edges.reshape(-1, 2, 2).tolist()}
    if not np.allclose(geometry["corners"], working, rtol=0, atol=CORNER_TOLERANCE_PX):
        raise ValueError(f"{case['case_id']}: displayed corners differ from source working corners")
    return geometry


def same_selection(first: dict | None, second: dict | None, label: str) -> None:
    if first is None or second is None:
        if first is not second:
            raise ValueError(f"{label}: one source has no selection")
        return
    if first["origin_key"] != second["origin_key"]:
        raise ValueError(f"{label}: selected origin keys differ")
    for field in ("corners_native_px", "corners_working_px"):
        if not np.allclose(first[field], second[field], rtol=0, atol=CORNER_TOLERANCE_PX):
            raise ValueError(f"{label}: {field} differs between sources")


def net_caption(selection: dict | None) -> str:
    if selection is None:
        return "Net support: no selected candidate"
    net = selection["net"]
    if net["state"] != "measured":
        return "Net support: unavailable (neutral)"
    coverage = net["coverage"]
    pieces = ("tape_left", "tape_right", "post_left", "post_right")
    values = ["missing" if coverage[name] is None else f"{coverage[name]:.0%}" for name in pieces]
    return f"Net support: tape L/R {values[0]}/{values[1]}, posts L/R {values[2]}/{values[3]}"


def view(selection: dict | None, title: str, case: dict, error: dict | None,
         caption: str = "") -> dict:
    if selection is None:
        return {"title": title, "valid": False, "status": "no_selection", "geometry": None,
                "origin_key": None, "candidate_id": None, "source": None,
                "detail": "No selected candidate; no outline"}
    ranked_rows = [row for row in case["candidates"] if row["origin_key"] == selection["origin_key"]]
    if len(ranked_rows) != 1:
        raise ValueError(f"{case['label']}: selected candidate is missing from the ranked pool")
    ranked = ranked_rows[0]
    if (selection.get("candidate_id") not in (None, ranked["candidate_id"])
            or ranked["source"] != selection["source"]):
        raise ValueError(f"{case['label']}: selected candidate identity differs from its ranking row")
    candidate_id = ranked["candidate_id"]
    eligible = ranked["camera_eligible"]
    full_court = selection.get("historical_fullcourt")
    if full_court is None:
        full_court = selection.get("historical", {}).get("historical_fullcourt")
    parts = [selection["origin_key"], f"source {selection['source']}"]
    parts.append(f"candidate ID {candidate_id}")
    parts.append(f"camera eligible: {'yes' if eligible else 'no'}")
    if full_court is not None:
        parts.append(f"full-court gate: {'yes' if full_court else 'no'}")
    if caption:
        parts.append(caption)
    if error is not None:
        parts.append(f"retrospective max corner error: {error['maximum']:.1f} native px")
    return {"title": title, "valid": True, "status": "selected", "geometry": project_selection(selection, case),
            "origin_key": selection["origin_key"], "candidate_id": candidate_id,
            "source": selection["source"], "camera_eligible": eligible,
            "historical_fullcourt": full_court, "detail": " · ".join(parts)}


def gallery_page(rows: list[dict]) -> str:
    page = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(rows, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    page = replace_once(page, "__GALLERY_DATA__", payload)
    page = replace_once(page, "<title>SVD12 search depth · court gallery</title>",
                        "<title>Am1 recovery and net preference — experimental</title>")
    page = replace_once(
        page,
        "const decisionNames={original:'Original fit',earlier:'Earlier bright-paint rule',"
        "automatic:'Automatic stripe polarity'};",
        "const decisionNames={original:'Saved W5',earlier:'Seeded W5',automatic:'Net preference'};",
    )
    start = page.index("if(decisionMode){\n document.title=")
    end = page.index("if(colourMode){", start)
    page = page[:start] + """if(decisionMode){
 document.title='Am1 recovery and net preference — experimental';
 document.querySelector('h1').textContent='Am1 recovery and net preference — experimental';
 const paragraphs=document.querySelectorAll('header p');
 paragraphs[0].textContent='Does seeded generation plus one net preference improve Am1, and does the preference damage '
  +'saved GX5 or Letterboxed45? These are experimental selections for visual review.';
 paragraphs[1].textContent='Each outline is the actual selected W5 candidate. Passing camera or full-court gates '
  +'does not establish a correct court. Missing net support is neutral.';
 paragraphs[2].textContent='Geometry and enlarged crops use working pixels; corner error is retrospective in native '
  +'pixels. Amateur annotations mix centre and outer-edge conventions, so small drift is inconclusive. '
  +'Click an image to focus all crops; Hold Space to hide outlines.';
 document.getElementById('view').closest('label').hidden=true;
 document.getElementById('reference').closest('label').hidden=true;
}
""" + page[end:]
    start = page.index("const redraw=[];\nif(decisionMode){")
    end = page.index("for(const data of cases){", start)
    page = page[:start] + "const redraw=[];\n" + page[end:]
    start = page.index(" if(decisionMode){const context=document.createElement('p');")
    end = page.index("\n const names=decisionMode?", start)
    page = (page[:start] + " if(decisionMode){const context=document.createElement('p');"
            "context.className='small';context.textContent=data.context;section.append(context);}"
            + page[end:])
    page = replace_once(
        page,
        "Solid: original. Dashed: earlier bright-paint rule. Dash-dot: automatic stripe polarity. "
        "Invalid or unavailable fits have no outline.",
        "Solid: saved W5. Dashed: seeded W5. Dash-dot: net preference. Missing selections have no outline.",
    )
    page = replace_once(page, " if(decisionMode)decisionEvidence(section,data);\n", "")
    start = page.index("   if(decisionMode){\n    const fit=data.decision_views[name];title.textContent=")
    end = page.index("   }else if(colourMode){", start)
    page = page[:start] + """   if(decisionMode){
    const fit=data.decision_views[name];
    title.textContent=fit.title+(fit.valid?'':' · no selection');
    detail.textContent=fit.detail;
""" + page[end:]
    return page


def main() -> None:
    recovery = read(RECOVERY)
    net = read(NET_TRIAL)
    if recovery["case_id"] != CASE_LABELS[0][0] or net["schema"] != "am1-net-selection-trial/1":
        raise ValueError("unexpected Am1 recovery or net trial source")
    net_cases = {case["label"]: case for case in net["cases"]}
    if len(net_cases) != len(net["cases"]):
        raise ValueError("duplicate net trial case labels")
    rows = []
    audit = {"schema": "am1-gallery-source-audit/1", "source_files": [str(RECOVERY), str(NET_TRIAL)],
             "corner_tolerance_working_px": CORNER_TOLERANCE_PX, "cases": []}
    for case_id, label, net_label, saved_label in CASE_LABELS:
        case = net_cases[net_label]
        if case["case_id"] != case_id:
            raise ValueError(f"{net_label}: unexpected case ID")
        saved_case = net_cases[saved_label] if saved_label else case
        if saved_case["case_id"] != case_id:
            raise ValueError(f"{saved_label}: unexpected saved case ID")
        saved = saved_case["baseline"]
        seeded = recovery["selected_geometry"]["full"]["gated"] if saved_label else saved
        if saved_label:
            same_selection(saved, recovery["frozen_selected_geometry"]["full"]["gated"], "Am1 saved W5")
            same_selection(seeded, case["baseline"], "Am1 seeded W5")
        else:
            same_selection(saved, case["baseline"], f"{label} saved W5")
        trial = case["trial"]
        image = BASE / "gallery" / f"{case_id}.jpg"
        if not image.is_file():
            raise FileNotFoundError(image)
        errors = case["retrospective_reference_error_diagnostic_only"]
        saved_error = saved_case["retrospective_reference_error_diagnostic_only"]["baseline"]
        seeded_error = (recovery["retrospective_selected_reference_error"]["full"]["gated"]
                        if saved_label else saved_error)
        views = {
            "original": view(saved, "Saved W5", saved_case, saved_error),
            "earlier": view(seeded, "Seeded W5" if saved_label else "Saved W5; seeding not run",
                            case, seeded_error),
            "automatic": view(trial, "Net preference", case, errors["trial"], net_caption(trial)),
        }
        rows.append({"gallery_mode": "decision_trials", "case_id": case_id,
                     "image": f"../gallery/{case_id}.jpg", "size": case["working_size_wh"],
                     "decision_views": views, "saved_selected": views["original"]["geometry"]["corners"]
                     if views["original"]["geometry"] else None, "saved_w5_geometry": None,
                     "context": f"{label} · native {case['native_size_wh'][0]} × {case['native_size_wh'][1]} px; "
                                f"working {case['working_size_wh'][0]} × {case['working_size_wh'][1]} px "
                                "· selection, not acceptance"})
        audit["cases"].append({
            "case_id": case_id, "image": rows[-1]["image"], "native_size_wh": case["native_size_wh"],
            "working_size_wh": case["working_size_wh"], "net_source_record": case["source_record"],
            "seeding_run": bool(saved_label), "trial_reason": case["trial_reason"],
            "displayed": {name: {"origin_key": value["origin_key"], "candidate_id": value["candidate_id"],
                                 "corners_working_px": value["geometry"]["corners"] if value["geometry"] else None}
                          for name, value in views.items()},
            "source": {name: None if selection is None else {
                "origin_key": selection["origin_key"], "candidate_id": selection.get("candidate_id"),
                "corners_native_px": selection["corners_native_px"],
                "corners_working_px": selection["corners_working_px"],
            } for name, selection in (("original", saved), ("earlier", seeded), ("automatic", trial))},
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "index.html").write_text(gallery_page(rows), encoding="utf-8")
    with gzip.open(OUTPUT / "source_audit.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(audit, stream, allow_nan=False, indent=2)
    print(f"Built {len(rows)} cases: {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
