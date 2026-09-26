"""Build a static gallery for automatic direction-axis matching results."""

from __future__ import annotations

import argparse
import html
import importlib
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

HELPERS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HELPERS / "marking_diagnosis"))
sys.path.insert(0, str(HELPERS / "vp_pruning"))

from diagnose_targets import read
from render_followup import case_title
from render_viewer import CASES, CHECKS, panel


def _load_template() -> str:
    template_root = CHECKS / "player_guided/20260913/seed_selection"
    sys.path.insert(0, str(template_root))
    template = importlib.import_module("render_visual_check")
    return template.document_header()


def _sources() -> tuple[dict[str, dict], dict[str, dict]]:
    packs = [
        read(CHECKS / "player_guided/20260909/gx_extension/inputs.json.gz"),
        read(CHECKS / "player_guided/20260909/broadcast_extension/inputs.json.gz"),
        read(CHECKS / "player_guided/20260908/marking_refit/marking_inputs.json.gz"),
    ]
    sources = {source["id"]: source for pack in packs for source in pack["cases"]}
    references = {case_id: reference for pack in packs for case_id, reference in pack["references"].items()}
    return sources, references


def _entry_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["candidate_id"]: entry for entry in result["entries"]}


def _shortlist_entry(result: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [entry for pair in result["pairs"] for entry in pair.get("shortlist", [])
               if entry["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"expected one saved shortlist entry for {candidate_id!r}, found {len(matches)}")
    return matches[0]


def _saved_camera_error(result: dict[str, Any], candidate_id: str) -> float | None:
    matches = [record for record in result["camera_prefilter"]
               if record["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"expected one saved camera record for {candidate_id!r}, found {len(matches)}")
    error = matches[0]["camera_error"]
    return None if error is None else float(error)


def _camera_check_description(result: dict[str, Any], error: float | None) -> str:
    limit = float(result.get("camera_error_limit", .1))
    if error is None or not np.isfinite(error):
        return "Saved existing camera check: rejected because no finite camera error was saved."
    status = "passed" if error <= limit else "rejected"
    return f"Saved existing camera check: {status} (error {error:.4f}; limit {limit:.4f})."


def _winner(
    result: dict[str, Any], diagnosis: dict[str, Any], ranking: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Join result geometry to the saved, reference-derived diagnosis only."""
    result_id = result[f"{ranking}_winner_id"]
    diagnostic = diagnosis["winners"][ranking]
    if result_id is None:
        if diagnostic is not None:
            raise ValueError(f"{ranking} diagnosis has a winner but result has none")
        return None, None
    if diagnostic is None or diagnostic["candidate_id"] != result_id:
        raise ValueError(f"{ranking} result and diagnosis winners disagree")
    try:
        return _entry_map(result)[result_id], diagnostic
    except KeyError as error:
        raise ValueError(f"{ranking} winner {result_id!r} is absent from result entries") from error


def _winner_description(
    entry: dict[str, Any], diagnostic: dict[str, Any], ranking: str, *, both: bool = False,
) -> str:
    candidate_id = entry["candidate_id"]
    line_score = entry["stripe"]["exclusive"]["score"]
    paint_score = entry["profile"]["score"]
    prefix = "Both automatic rankings" if both else ("Automatic line-score winner" if ranking == "line"
                                                       else "Automatic paint-ranked winner")
    return (f"{prefix}; candidate {candidate_id}. "
            f"Line score {line_score:.3f}; paint-profile score {paint_score:.3f}. "
            f"Saved diagnosis reference error {diagnostic['reference_display']:.2f} px at 1280×720.")


def _winner_panel(
    result: dict[str, Any], diagnosis: dict[str, Any], ranking: str,
    source: dict[str, Any], reference: dict[str, Any], output: Path,
) -> str:
    entry, diagnostic = _winner(result, diagnosis, ranking)
    title = "Automatic line-score winner" if ranking == "line" else "Automatic paint-ranked winner"
    if entry is None:
        return panel(title, "No automatic winner; none camera eligible.", None, source, reference, output)
    return panel(title, _winner_description(entry, diagnostic, ranking),
                 np.asarray(entry["corners_px"]), source, reference, output)


def _main_panels(
    result: dict[str, Any], diagnosis: dict[str, Any],
    source: dict[str, Any], reference: dict[str, Any], output: Path,
) -> str:
    line_entry, line_diagnostic = _winner(result, diagnosis, "line")
    paint_entry, _paint_diagnostic = _winner(result, diagnosis, "paint")
    if line_entry is not None and paint_entry is not None and line_entry["candidate_id"] == paint_entry["candidate_id"]:
        description = _winner_description(line_entry, line_diagnostic, "line", both=True)
        description += " Both automatic rankings selected this candidate."
        return panel("Automatic line and paint winner", description,
                     np.asarray(line_entry["corners_px"]), source, reference, output)
    return (_winner_panel(result, diagnosis, "line", source, reference, output)
            + _winner_panel(result, diagnosis, "paint", source, reference, output))


def _diagnostic_panel(
    result: dict[str, Any], diagnosis: dict[str, Any],
    source: dict[str, Any], reference: dict[str, Any], output: Path,
) -> str:
    nearest = diagnosis.get("after_global_cap")
    automatic_ids = {result["line_winner_id"], result["paint_winner_id"]}
    nearest_id = None if nearest is None else nearest["candidate_id"]
    blocks: list[str] = []
    notes: list[str] = []
    if nearest is None:
        notes.append("No final candidate was available for the inspected-control comparison.")
    elif nearest_id in automatic_ids:
        notes.append(f"Candidate {nearest_id} is already shown as an automatic winner; no duplicate panel is rendered.")
    else:
        entries = _entry_map(result)
        try:
            entry = entries[nearest_id]
        except KeyError as error:
            raise ValueError(f"diagnosis control candidate {nearest_id!r} is absent from result entries") from error
        description = ("Diagnostic/manual comparison: nearest final candidate to the inspected control. "
                       f"Saved diagnosis control error: {nearest['max_corner_px']:.2f} working px. "
                       "This candidate is not an automatic winner.")
        blocks.append(panel("Nearest final candidate to inspected control", description,
                            np.asarray(entry["corners_px"]), source, reference, output))

    before = diagnosis.get("before_global_cap")
    before_id = None if before is None else before["candidate_id"]
    if before is None:
        notes.append("No generated candidate was available before camera filtering.")
    elif before_id in automatic_ids or before_id == nearest_id:
        notes.append(f"Candidate {before_id} is already shown in the automatic or nearest-final panels; "
                     "no duplicate pre-camera panel is rendered.")
    else:
        entry = _shortlist_entry(result, before_id)
        camera_error = _saved_camera_error(result, before_id)
        description = ("Reference-selected diagnostic: closest generated court before camera filtering. "
                       f"Saved diagnosis control distance: {before['max_corner_px']:.2f} working px. "
                       f"{_camera_check_description(result, camera_error)} "
                       "This panel makes no visual or acceptance claim.")
        blocks.append(panel("Closest generated court before camera filtering", description,
                            np.asarray(entry["corners_px"]), source, reference, output))

    body = "".join(blocks) + "".join(f"<p>{html.escape(note)}</p>" for note in notes)
    return f"<details><summary>Optional reference-selected diagnostics</summary>{body}</details>"


def _header() -> str:
    heading = (
        "<header><h1>Automatic direction-axis matching</h1>"
        "<p>The main panels show the automatic line-score winner and paint-ranked winner "
        "from the full saved camera-eligible per-pair comparison. If both rankings selected one candidate, it is "
        "shown once. This records the generated comparison and makes no production-readiness claim.</p>"
        "<p>Automatic directions, axis matching and ranking use image fragments and player observations. "
        "Expandable panels are reference-selected after-generation diagnostics for the nearest final court and, "
        "when distinct, the closest generated court before camera filtering. Manual reference data never supplies "
        "directions or selects an automatic winner.</p>"
        "<p>Magenta is generated candidate geometry. Use the existing controls to show the optional blue manual "
        "reference overlay. Errors shown here come only from the saved diagnosis. "
        "This gallery makes no acceptance claim.</p>"
        "<p>Inspect first: <a href=\"#gxBQ_window_00_frame_5\">GX frame 5</a> · "
        "<a href=\"#am2_window_00_frame_150\">Amateur-2 frame 150</a> · "
        "<a href=\"#shuttleset_03_scene_0017\">ShuttleSet 03 scene 17</a>.</p></header>"
    )
    header = re.sub(r"<header>.*?</header>", heading, _load_template(), flags=re.DOTALL)
    return header.replace("Frame-5 court usability check", "Automatic direction-axis matching")


def render(results: Path, diagnosis_path: Path, output: Path, case_ids: list[str]) -> None:
    """Render the nine automatic result records without recomputing geometry or errors."""
    sources, references = _sources()
    diagnosis_records = {record["case_id"]: record for record in read(diagnosis_path)["records"]}
    sections = []
    for case_id in case_ids:
        result = read(results / f"{case_id}.json.gz")
        try:
            diagnosis = diagnosis_records[case_id]
            source, reference = sources[case_id], references[case_id]
        except KeyError as error:
            raise ValueError(f"missing diagnosis or source record for {case_id}") from error
        main = _main_panels(result, diagnosis, source, reference, output)
        diagnostic = _diagnostic_panel(result, diagnosis, source, reference, output)
        sections.append(f'<section id="{case_id}"><h2>{html.escape(case_title(case_id))}</h2>'
                        f'<div class="pair">{main}</div>{diagnostic}</section>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_header() + "".join(sections) + "</body></html>\n", encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", default=list(CASES))
    args = parser.parse_args()
    render(args.results, args.diagnosis, args.output, args.ids)


if __name__ == "__main__":
    main()
