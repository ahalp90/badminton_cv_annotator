"""Build the small cleanup and one-missing rally target list."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path
from typing import Any

FIELD_NAMES = [
    "category",
    "fixture",
    "span_id",
    "rally_id",
    "start_frame",
    "end_frame",
    "event_count",
    "ground_truth_contacts",
    "timing_matches",
    "correct_side_answers",
    "timing_confidence",
    "rejection_reasons",
]
CATEGORY_ORDER = {
    "remove_extra_event": 0,
    "add_one_missing_event": 1,
}
FIXTURE_ORDER = {
    "sset_01": 0,
    "sset_15": 1,
    "sset_21": 2,
}


def repair_category(span: dict[str, Any]) -> str | None:
    """Return the simple repair that would make this one-rally span exact."""
    if span["rally_id"] is None:
        return None

    event_count = int(span["event_count"])
    ground_truth_contacts = int(span["ground_truth_contacts"])
    timing_matches = int(span["timing_matches"])
    correct_side_answers = int(span["correct_side_answers"])

    all_contacts_already_correct = (
        event_count > ground_truth_contacts
        and timing_matches == ground_truth_contacts
        and correct_side_answers == ground_truth_contacts
    )
    if all_contacts_already_correct:
        return "remove_extra_event"

    exactly_one_contact_missing = (
        event_count + 1 == ground_truth_contacts
        and timing_matches == event_count
        and correct_side_answers == event_count
    )
    if exactly_one_contact_missing:
        return "add_one_missing_event"

    return None


def target_row(span: dict[str, Any], category: str) -> dict[str, Any]:
    """Copy the fields needed to inspect or replay one repair target."""
    row = {field_name: span[field_name] for field_name in FIELD_NAMES[1:-1]}
    row["category"] = category
    row["rejection_reasons"] = ", ".join(span["rejection_reasons"])
    return row


def sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    """Keep the output stable and grouped by repair job."""
    return (
        CATEGORY_ORDER[str(row["category"])],
        FIXTURE_ORDER[str(row["fixture"])],
        int(row["span_id"]),
    )


def build_targets(rally_score_path: Path) -> list[dict[str, Any]]:
    """Derive repair targets from the original HGB strict span records."""
    with gzip.open(rally_score_path, "rt") as source:
        report = json.load(source)

    rows: list[dict[str, Any]] = []
    for span in report["primary"]["spans"]:
        category = repair_category(span)
        if category is not None:
            rows.append(target_row(span, category))
    rows.sort(key=sort_key)
    return rows


def write_targets(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write the target list as the repository-standard compressed CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_text = io.StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=FIELD_NAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output_path.write_bytes(gzip.compress(csv_text.getvalue().encode(), mtime=0))


def main() -> None:
    contact_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rally-score",
        type=Path,
        default=contact_dir / "raw/followups/phase1/run_a/contact_rally_score.json.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            contact_dir
            / "raw/followups/rally_cleanup_targets/contact_followup_rally_targets.csv.gz"
        ),
    )
    args = parser.parse_args()

    rows = build_targets(args.rally_score)
    write_targets(rows, args.output)
    print(f"Wrote {len(rows)} rally repair targets to {args.output}")


if __name__ == "__main__":
    main()
