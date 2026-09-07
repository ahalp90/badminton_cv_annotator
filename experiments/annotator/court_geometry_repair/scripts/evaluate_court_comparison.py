"""Evaluate two saved court-dependent contact streams for one video."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scratch.contact_det.scripts.score_contact_rallies import RallyReference
from scratch.contact_det_closing_pass.scripts.evaluation import (
    score_contacts,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.run_later_broader import restore_stream
from scratch.contact_det_closing_pass.scripts.whole_rally_evaluation import (
    paired_sections,
    section_views,
    voted_contact_scores,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_followup.scripts.score_followup import load_saved_test_labels
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    HumanLabels,
)

TOLERANCES = (10, 5)
FINAL_RECORD = "final_stream_records.json.gz"
METADATA_RECORD = "run_metadata.json.gz"
CLEAN_LABEL_RECORD = "clean_labels.json.gz"


def _labels_for(fixture: str, labels_root: Path) -> HumanLabels:
    labels_path = Path(labels_root).resolve(strict=True) / CLEAN_LABEL_RECORD
    all_labels = load_saved_test_labels(labels_path).labels
    rallies = all_labels.rallies_by_fixture
    if fixture not in rallies:
        raise ValueError(f"saved labels do not contain video {fixture}")
    fixture_references = []
    sides = {}
    for index, rally in enumerate(rallies[fixture]):
        frames = tuple(contact.frame for contact in rally.contacts)
        fixture_references.append(
            RallyReference(
                fixture,
                index,
                f"{rally.set_id}:{rally.rally}",
                frames,
            )
        )
        for contact in rally.contacts:
            sides[(fixture, contact.frame)] = contact.side
    return HumanLabels({fixture: tuple(fixture_references)}, sides)


def _load_stream(root: Path, fixture: str, expected_baseline: bool) -> ContactStreams:
    root = Path(root).resolve(strict=True)
    stream_path = root / FINAL_RECORD
    metadata = prediction_io.read_json(root / METADATA_RECORD)
    if str(metadata.get("video_id")) != fixture:
        raise ValueError(f"{root}: run metadata video_id differs from {fixture}")
    if bool(metadata.get("baseline")) != expected_baseline:
        raise ValueError(f"{root}: baseline flag does not match its CLI role")
    stream = restore_stream(prediction_io.read_json(stream_path))
    if set(stream.events_by_fixture) != {fixture}:
        raise ValueError(f"{root}: final stream contains another fixture")
    if any(span.fixture != fixture for span in stream.spans):
        raise ValueError(f"{root}: final spans contain another fixture")
    return stream


def _contact_metrics(
    stream: ContactStreams, labels: HumanLabels, fixture: str,
) -> dict[str, Any]:
    voted = start.apply_whole_rally_alternation(stream)
    fps = {fixture: 30.0}
    result: dict[str, Any] = {}
    for tolerance in TOLERANCES:
        raw = score_contacts(stream.events_by_fixture, labels, fps, tolerance)
        voted_scores = voted_contact_scores(raw, voted.events_by_fixture)
        result[str(tolerance)] = {
            # Each input contains one fixture, so these totals are per-video.
            "raw": raw["total"],
            "voted_side": voted_scores["total"],
        }
    return result


def _whole_rally_metrics(
    before: ContactStreams,
    after: ContactStreams,
    labels: HumanLabels,
    fixture: str,
) -> dict[str, Any]:
    groups = {fixture: "ShuttleSet22"}
    fps = {fixture: 30.0}
    result: dict[str, Any] = {}
    for tolerance in TOLERANCES:
        before_view = section_views(before.spans, labels, fps, groups, tolerance)
        after_view = section_views(after.spans, labels, fps, groups, tolerance)
        before_rows = before_view["fixed_side"]["sections"]
        after_rows = after_view["fixed_side"]["sections"]
        paired = paired_sections(before_rows, after_rows)
        paired["improved_rallies"] = [
            rally_id for _fixture, rally_id in paired.pop("repaired")
        ]
        paired["regressed_rallies"] = [
            rally_id for _fixture, rally_id in paired.pop("lost")
        ]
        result[str(tolerance)] = {
            "outcomes_before": before_view["fixed_side"]["summary"]["by_video"][0],
            "outcomes_after": after_view["fixed_side"]["summary"]["by_video"][0],
            "paired": paired,
        }
    return result


def evaluate(arguments: argparse.Namespace) -> dict[str, Any]:
    fixture = str(arguments.video_id)
    before = _load_stream(arguments.baseline, fixture, True)
    after = _load_stream(arguments.changed, fixture, False)
    labels = _labels_for(fixture, arguments.labels_root)
    result = {
        "schema": "court-comparison-evaluation/1",
        "status": "complete",
        "labels_read": True,
        "video_id": arguments.video_id,
        "fixture": fixture,
        "contacts": {
            "before": _contact_metrics(before, labels, fixture),
            "after": _contact_metrics(after, labels, fixture),
        },
        "whole_rally": _whole_rally_metrics(before, after, labels, fixture),
    }
    write_json(arguments.output, result)
    for tolerance in TOLERANCES:
        row = result["whole_rally"][str(tolerance)]
        before_total = row["outcomes_before"]["fully_correct"]
        after_total = row["outcomes_after"]["fully_correct"]
        print(
            f"{fixture} ±{tolerance}: fully-correct {before_total} -> {after_total}; "
            f"improved {len(row['paired']['improved_rallies'])}, "
            f"regressed {len(row['paired']['regressed_rallies'])}",
            flush=True,
        )
    print(f"Saved evaluation to {arguments.output}", flush=True)
    return result


def _check_expected(result: dict[str, Any], expected_path: Path) -> None:
    expected = prediction_io.read_json(Path(expected_path))
    if result != expected:
        raise ValueError(f"recomputed evaluation differs from {expected_path}")
    print(f"Matched expected evaluation at {expected_path}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--changed", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--video-id", type=int, choices=(17, 53), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected",
        type=Path,
        help="Optional recorded evaluation to compare with the recomputed result.",
    )
    arguments = parser.parse_args(argv)
    result = evaluate(arguments)
    if arguments.expected is not None:
        _check_expected(result, arguments.expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
