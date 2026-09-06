"""Rebuild contact, rally and selection tables from saved predictions and source labels."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from annotator.fps_constants import ScalingKind
from scratch.contact_det.scripts.score_contact_rallies import FixedSpan, RallyReference
from scratch.contact_det_closing_pass.scripts.evaluation import (
    score_sections,
    test_labels,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.matching import match_contacts
from scratch.contact_det_closing_pass.scripts.regenerate_figures import (
    regenerate_metric_figures,
)
from scratch.contact_det_closing_pass.scripts.run_later_broader import restore_stream
from scratch.contact_det_followup.scripts.prediction_io import read_json
from scratch.contact_det_followup.scripts.score_start_model import (
    _with_alternating_sides,
    apply_whole_rally_alternation,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    HumanLabels,
)
from scratch.contact_det_full_ds_fit.scripts.score_shuttleset22_test import _player_slot

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "original": ("broader_predictions.json.gz", "baseline", "Original detector"),
    "opening": ("broader_predictions.json.gz", "opening_only", "First-contact repair"),
    "combined": ("broader_predictions.json.gz", "combined", "Whole-sequence model"),
    "later": ("later/later_broader_predictions.json.gz", None, "+ one later contact"),
    "local": ("followups/local_broader_predictions.json.gz", None, "+ local insertion score"),
    "boundaries": (
        "followups/session_start_boundary_broader_predictions_fixed_membership.json.gz", None,
        "Boundary fix only",
    ),
    "recommended": (
        "followups/local_boundary_broader_predictions_fixed_membership.json.gz", None, "Final detector",
    ),
    "early": (
        "followups/early_boundary_broader_predictions_fixed_membership.json.gz", None, "Wider early shortlist",
    ),
}
SideRows = Mapping[tuple[str, str], tuple[str | None, ...]]


def load_populations(annotations: Path) -> tuple[dict[str, HumanLabels], dict]:
    """Retain each source row's side, including conflicting sides at duplicate timestamps."""
    retained = test_labels()
    names = pd.read_csv(annotations / "set/match.csv").set_index("id")["video"].to_dict()
    rallies, sides, side_rows = {}, {}, {}
    for fixture in retained.rallies:
        references = []
        for path in sorted((annotations / "set" / names[int(fixture)]).glob("set*.csv")):
            for rally_number, group in pd.read_csv(path).groupby("rally", sort=True):
                ordered = group.sort_values(["ball_round", "frame_num"])
                frames = tuple(int(value) for value in ordered["frame_num"])
                rally_id = f"{path.stem}:{int(rally_number)}"
                targets = tuple(_player_slot(row, pd) for _, row in ordered.iterrows())
                references.append(RallyReference(fixture, len(references), rally_id, frames))
                side_rows[fixture, rally_id] = targets
                sides.update({(fixture, frame): side for frame, side in zip(frames, targets, strict=True)})
        rallies[fixture] = tuple(references)
        by_id = {rally.rally_id: rally for rally in references}
        for rally in retained.rallies[fixture]:
            assert by_id[rally.rally_id].frames == rally.frames
            assert side_rows[fixture, rally.rally_id] == tuple(
                retained.target_sides[fixture, frame] for frame in rally.frames
            )
    assert sum(map(len, rallies.values())) == 3965
    assert sum(len(rally.frames) for values in rallies.values() for rally in values) == 43159
    return {"retained": retained, "all_gt": HumanLabels(rallies, sides)}, side_rows


def load_stream(results: Path, stage: str) -> tuple[ContactStreams, dict[str, float]]:
    filename, output_key, _ = STAGES[stage]
    spans, events, fps = [], {}, {}
    for video in read_json(results / filename)["videos"]:
        output = video["output"] if output_key is None else video["outputs"][output_key]
        stream = restore_stream(output)
        spans.extend(stream.spans)
        events.update(stream.events_by_fixture)
        fps[video["fixture"]] = float(video["fps"])
    return ContactStreams(tuple(spans), events), fps


def judgement(row: Mapping[str, Any]) -> str:
    """Keep absent labels unknown; a known contradiction still makes a proposal wrong."""
    if row["overlapping_rallies"] == 0:
        return "unknown"
    if row["overlapping_rallies"] > 1 or not row["whole_rally_contained"]:
        return "wrong"
    matched = len(row["matches"])
    if matched < row["labelled_contacts"] or row["voted_correct_sides"] < row["known_label_sides"]:
        return "wrong"
    if row["events"] > matched:
        return "wrong"
    if row["known_label_sides"] < row["labelled_contacts"]:
        return "unknown"
    assert row["fully_correct"]
    return "correct"


def section_rows(
    stream: ContactStreams, labels: HumanLabels, sides: SideRows, fps: dict[str, float], tolerance: int,
) -> list[dict[str, Any]]:
    rows = score_sections(stream.spans, labels, fps, tolerance)
    for row, span in zip(rows, stream.spans, strict=True):
        targets = sides.get((span.fixture, row["rally_id"]), ())
        voted = _with_alternating_sides(span)
        # Frame-keyed side maps cannot represent contradictory duplicate source rows.
        row["voted_correct_sides"] = sum(
            targets[gt_index] is not None and targets[gt_index] == voted.events[pred_index].predicted_side
            for gt_index, pred_index, _ in row["matches"]
        )
        row["known_label_sides"] = sum(side is not None for side in targets)
        row["fully_correct"] = row["timing_complete"] and row["voted_correct_sides"] == row["events"]
        row["outcome"] = judgement(row)
    return rows


def selection_summary(rows: Sequence[Mapping[str, Any]], labelled_rallies: int) -> dict[str, Any]:
    outcomes = Counter(row["outcome"] for row in rows)
    complete = {(row["fixture"], row["rally_id"]) for row in rows if row["fully_correct"]}
    contained = {
        (row["fixture"], row["rally_id"]) for row in rows
        if row["overlapping_rallies"] == 1 and row["whole_rally_contained"]
    }
    return {
        "proposals": len(rows), "correct": outcomes["correct"], "wrong": outcomes["wrong"],
        "unknown": outcomes["unknown"], "unique_complete": len(complete),
        "unique_contained": len(contained), "labelled_rallies": labelled_rallies,
    }


def full_stream_counts(
    stream: ContactStreams, labels: HumanLabels, sides: SideRows, fps: dict[str, float], tolerance: int,
) -> dict[str, int]:
    """Match once per video; derive serve and non-serve counts from those same pairs."""
    voted = apply_whole_rally_alternation(stream)
    counts: Counter[str] = Counter(dict.fromkeys((
        "labelled", "predicted", "matched", "labelled_serves", "side_correct", "serve_matched",
        "serve_side_correct", "starts", "start_matched", "start_side_correct",
    ), 0))
    spans_by_fixture: dict[str, list[FixedSpan]] = {}
    for span in voted.spans:
        spans_by_fixture.setdefault(span.fixture, []).append(span)
    for fixture, predictions in voted.events_by_fixture.items():
        contacts = []
        for rally in labels.rallies[fixture]:
            for index, (frame, side) in enumerate(zip(rally.frames, sides[fixture, rally.rally_id], strict=True)):
                contacts.append((frame, rally.rally_id, index, side))
        contacts.sort(key=lambda contact: contact[:3])
        window = ScalingKind.FRAME_COUNT.scale(tolerance, fps[fixture])
        pairs = match_contacts([row[0] for row in contacts], [event.frame for event in predictions], window)
        counts.update(labelled=len(contacts), predicted=len(predictions), matched=len(pairs),
                      labelled_serves=len(labels.rallies[fixture]))
        matches_by_prediction = {}
        for gt_index, pred_index, _ in pairs:
            _, _, contact_index, side = contacts[gt_index]
            correct_side = side is not None and side == predictions[pred_index].predicted_side
            counts["side_correct"] += correct_side
            if contact_index == 0:
                counts["serve_matched"] += 1
                counts["serve_side_correct"] += correct_side
            matches_by_prediction[predictions[pred_index].frame] = (contact_index == 0, side)
        for span in spans_by_fixture[fixture]:
            if not span.events:
                continue
            counts["starts"] += 1
            first = span.events[0]
            matched_serve, side = matches_by_prediction.get(first.frame, (False, None))
            counts["start_matched"] += matched_serve
            counts["start_side_correct"] += matched_serve and side is not None and side == first.predicted_side
    return dict(counts)


def prf(correct: int, predicted: int, labelled: int) -> str:
    return (
        f"{100 * correct / predicted:.1f}% | {100 * correct / labelled:.1f}% | "
        f"{200 * correct / (predicted + labelled):.1f}%"
    )


def contact_table(result: Mapping[str, Any], task: str, tolerance: int = 10) -> str:
    """Show each label population on its own row with explicit metric columns."""
    keys = {
        "contacts": (("Timing only", "matched"), ("Timing + correct player", "side_correct")),
        "starts": (("Start is serve", "start_matched"), ("Start + correct server", "start_side_correct")),
    }
    prediction_key, label_key = ("predicted", "labelled") if task == "contacts" else ("starts", "labelled_serves")
    lines = ["| Task | Labels | Precision | Recall | F1 |", "|---|---|---:|---:|---:|"]
    for title, correct_key in keys[task]:
        for population, label in (("retained", "Trusted GT"), ("all_gt", "All GT")):
            counts = result["contacts"][population][str(tolerance)]
            values = prf(counts[correct_key], counts[prediction_key], counts[label_key])
            lines.append(f"| {title} | {label} | {values} |")
    return "\n".join(lines)


def recall_table(result: Mapping[str, Any], tolerance: int = 10) -> str:
    lines = ["| Contact type | Labels | Timing recall | Timing + correct-player recall |",
             "|---|---|---:|---:|"]
    for serve, title in ((False, "Non-serve"), (True, "Serve")):
        for population, label in (("retained", "Trusted GT"), ("all_gt", "All GT")):
            counts = result["contacts"][population][str(tolerance)]
            labelled = counts["labelled_serves"] if serve else counts["labelled"] - counts["labelled_serves"]
            timing = counts["serve_matched"] if serve else counts["matched"] - counts["serve_matched"]
            joint = counts["serve_side_correct"] if serve else counts["side_correct"] - counts["serve_side_correct"]
            lines.append(f"| {title} | {label} | {100 * timing / labelled:.1f}% | {100 * joint / labelled:.1f}% |")
    return "\n".join(lines)


def write_table(result: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Contact, serve and automatic-use numbers at a glance", "",
        "The final detector finds contacts; the ranking model selects 784 rally clips for review.", "",
        ("**Trusted GT:** 3,422 rallies and 38,218 contacts. "
        "**All GT:** all 3,965 source rallies and 43,159 contacts."), "",
        "The difference is 543 rallies excluded during label cleaning. Both scores use the same predictions.", "",
        "Results below use **±10 frames on a 30 fps clock**. The tighter ±5 check is at the end.", "",
        "## Individual contacts", "", contact_table(result, "contacts"), "",
        "## Serve and non-serve recall", "", recall_table(result), "",
        "## Does the proposed rally start at the serve?", "", contact_table(result, "starts"), "",
        "## Automatic use", "",
        ("Fully correct means every contact and player is right. "
        "A whole-rally clip can still contain contact errors."), "",
        ("Trusted-GT precision uses the 740 judgeable clips. "
        "All-GT precision uses all 784; unknown clips receive no credit."), "",
        "| Selection task | Labels | Precision | Recall | F1 |", "|---|---|---:|---:|---:|",
    ]
    for title, key in (("Fully correct rally", "unique_complete"), ("Contains one whole rally", "unique_contained")):
        for population, label in (("retained", "Trusted GT"), ("all_gt", "All GT")):
            selected = result["selected"][population]["10"]
            predicted = selected["proposals"] - (selected["unknown"] if population == "retained" else 0)
            values = prf(selected[key], predicted, selected["labelled_rallies"])
            lines.append(f"| {title} | {label} | {values} |")
    lines.extend(["", "### Selected clip counts", "",
                  "| Labels | Fully correct | Wrong | Unknown | Contains one whole rally |",
                  "|---|---:|---:|---:|---:|"])
    for population, label in (("retained", "Trusted GT"), ("all_gt", "All GT")):
        selected = result["selected"][population]["10"]
        lines.append(f"| {label} | {selected['correct']} | {selected['wrong']} | "
                     f"{selected['unknown']} | {selected['unique_contained']} |")
    lines.extend(["", "## Whole-rally recovery before selection", "",
                  ("Each entry counts fully correct rallies. "
                  "The same detector predictions are checked at both timing windows."), "",
                  "| Detector | Trusted ±10 | All GT ±10 | Trusted ±5 | All GT ±5 |", "|---|---:|---:|---:|---:|"])
    for stage, (_, _, title) in STAGES.items():
        values = result["stages"][stage]
        cells = [f"{values[population][str(tolerance)]['unique_complete']:,}"
                 for tolerance in (10, 5) for population in ("retained", "all_gt")]
        lines.append(f"| {title} | {' | '.join(cells)} |")
    lines.extend(["", "<details>", "<summary>Tighter timing check: ±5</summary>", "",
                  "### Individual contacts", "", contact_table(result, "contacts", 5), "",
                  "### Serve and non-serve recall", "", recall_table(result, 5), "",
                  "### Rally starts", "", contact_table(result, "starts", 5), "",
                  "### Selected clips", "",
                  "| Labels | Fully correct | Wrong | Unknown | Contains one whole rally |",
                  "|---|---:|---:|---:|---:|"])
    for population, label in (("retained", "Trusted GT"), ("all_gt", "All GT")):
        selected = result["selected"][population]["5"]
        lines.append(f"| {label} | {selected['correct']} | {selected['wrong']} | "
                     f"{selected['unknown']} | {selected['unique_contained']} |")
    lines.extend(["", "</details>", "", "## Reproduce these results", "",
                  "Run from the repository root with the original ShuttleSet22 annotation folder:", "",
                  "```bash", 'PYTHONPATH="$PWD/src:$PWD" ~/.venvs/badminton-cicd/bin/python \\',
                  "  -m scratch.contact_det_closing_pass.scripts.summarise_metrics \\",
                  "  --annotations /path/to/ShuttleSet22", "```", "",
                  ("This rebuilds the saved counts, this reference and the comparison charts from saved predictions. "
                  "It checks the trusted-GT counts against the original experiments. "
                  "No training or vision inference is needed."), "",
                  ("Saved counts: `results/metric_summary.json.gz`. "
                  "Clip review: [individual notes](results/selected_clip_review.csv)."), ""])
    path.write_text("\n".join(lines))


def run(annotations: Path, results: Path, table: Path) -> None:
    populations, sides = load_populations(annotations)
    acceptance = read_json(results / "serve_followups/chosen_acceptance_broader.json.gz")
    threshold = acceptance["frozen_policies"]["gap"]["comparison"]["threshold"]
    accepted = {(row["fixture"], row["span_id"]): row for row in acceptance["rows"] if row["gap_score"] >= threshold}
    expected = {"original": 995, "opening": 1105, "combined": 1435, "later": 1597,
                "local": 1622, "boundaries": 1732, "recommended": 1763, "early": 1767}
    result: dict[str, Any] = {"schema": "contact-metric-summary/1", "selection_threshold": threshold,
                              "stages": {}, "contacts": {}, "selected": {}, "selected_rows": {}}
    for stage in STAGES:
        stream, fps = load_stream(results, stage)
        stage_result = {}
        for population, labels in populations.items():
            stage_result[population] = {}
            labelled = sum(map(len, labels.rallies.values()))
            for tolerance in (10, 5):
                rows = section_rows(stream, labels, sides, fps, tolerance)
                stage_result[population][str(tolerance)] = selection_summary(rows, labelled)
                if stage == "recommended":
                    selected = [row for row in rows if (row["fixture"], row["span_id"]) in accepted]
                    result["selected"].setdefault(population, {})[str(tolerance)] = selection_summary(selected, labelled)
                    result["selected_rows"].setdefault(population, {})[str(tolerance)] = selected
                    if population == "retained":
                        for row in selected:
                            saved = accepted[row["fixture"], row["span_id"]]["judgements"][str(tolerance)]["outcome"]
                            assert row["outcome"] == ("unknown" if saved == "unjudgeable" else saved)
            if stage == "recommended":
                result["contacts"][population] = {
                    str(tolerance): full_stream_counts(stream, labels, sides, fps, tolerance)
                    for tolerance in (10, 5)
                }
        assert stage_result["retained"]["10"]["unique_complete"] == expected[stage]
        result["stages"][stage] = stage_result
        print(stage, {key: value["10"]["unique_complete"] for key, value in stage_result.items()}, flush=True)
    counts = result["contacts"]["retained"]["10"]
    for key, value in {"labelled": 38218, "predicted": 41605, "matched": 33716, "side_correct": 32667,
                       "serve_matched": 2781, "serve_side_correct": 2647,
                       "start_matched": 2624, "start_side_correct": 2536}.items():
        assert counts[key] == value, (key, counts[key], value)
    write_json(results / "metric_summary.json.gz", result)
    write_table(result, table)
    regenerate_metric_figures(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--table", type=Path, default=ROOT / "serve_tables.md")
    args = parser.parse_args()
    run(args.annotations, args.results, args.table)


if __name__ == "__main__":
    main()
