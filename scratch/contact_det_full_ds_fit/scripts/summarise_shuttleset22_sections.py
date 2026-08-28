"""Recount ShuttleSet22 rally sections from the saved test files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PREDICTIONS = (
    EXPERIMENT_DIR / "raw/shuttleset22-test-predictions/combined_predictions.json.gz"
)
DEFAULT_LABELS = EXPERIMENT_DIR / "raw/shuttleset22-test-result/clean_labels.json.gz"
DEFAULT_RESULT = EXPERIMENT_DIR / "raw/shuttleset22-test-result/result.json"

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class Rally:
    """One labelled rally, represented by its contact frames."""

    rally_id: str
    frames: tuple[int, ...]


@dataclass(frozen=True)
class Section:
    """One predicted half-open video section."""

    fixture: str
    span_id: int
    start_frame: int
    end_frame: int


def load_json(path: Path) -> JsonObject:
    """Load a JSON or gzipped JSON object."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            value = json.load(file)
    else:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one saved input."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_list(value: Any, field_name: str) -> list[Any]:
    """Return a required list field."""
    if not isinstance(value, list):
        raise TypeError(f"Expected {field_name} to be a list")
    return value


def load_rallies(labels: JsonObject) -> dict[str, tuple[Rally, ...]]:
    """Read labelled contact frames by fixture."""
    rallies_by_fixture: dict[str, tuple[Rally, ...]] = {}
    for raw_video in require_list(labels.get("videos"), "labels.videos"):
        if not isinstance(raw_video, dict):
            raise TypeError("Expected each labels.videos item to be an object")
        fixture = str(raw_video["fixture"])
        rallies: list[Rally] = []
        for raw_rally in require_list(raw_video.get("rallies"), "video.rallies"):
            if not isinstance(raw_rally, dict):
                raise TypeError("Expected each rally to be an object")
            contacts = require_list(raw_rally.get("contacts"), "rally.contacts")
            frames = tuple(int(contact["frame"]) for contact in contacts)
            if not frames:
                raise ValueError(
                    f"Rally {fixture}:{raw_rally['rally']} has no contacts"
                )
            rally_id = f"{raw_rally['set_id']}:{raw_rally['rally']}"
            rallies.append(Rally(rally_id, frames))
        rallies_by_fixture[fixture] = tuple(rallies)
    return rallies_by_fixture


def load_sections(predictions: JsonObject) -> dict[str, tuple[Section, ...]]:
    """Read predicted sections by fixture and require non-overlap."""
    sections_by_fixture: dict[str, tuple[Section, ...]] = {}
    for raw_video in require_list(predictions.get("videos"), "predictions.videos"):
        if not isinstance(raw_video, dict):
            raise TypeError("Expected each predictions.videos item to be an object")
        fixture = str(raw_video["fixture"])
        sections = tuple(
            Section(
                fixture=fixture,
                span_id=int(raw_section["span_id"]),
                start_frame=int(raw_section["start_frame"]),
                end_frame=int(raw_section["end_frame"]),
            )
            for raw_section in require_list(raw_video.get("spans"), "video.spans")
        )
        previous_end = -1
        for section in sections:
            if section.start_frame < previous_end:
                raise ValueError(f"Overlapping sections in fixture {fixture}")
            if section.end_frame <= section.start_frame:
                raise ValueError(f"Empty section {fixture}:{section.span_id}")
            previous_end = section.end_frame
        sections_by_fixture[fixture] = sections
    return sections_by_fixture


def frames_inside(section: Section, rally: Rally) -> int:
    """Count labelled contacts inside one half-open section."""
    return sum(
        section.start_frame <= frame < section.end_frame for frame in rally.frames
    )


def nearest_rank(values: list[int], share: float) -> int:
    """Return a simple nearest-rank percentile."""
    ordered = sorted(values)
    index = math.ceil(share * len(ordered)) - 1
    return ordered[index]


def context_summary(values: list[int]) -> dict[str, int | float]:
    """Summarise the frames before or after labelled contacts."""
    return {
        "minimum": min(values),
        "p10": nearest_rank(values, 0.10),
        "median": statistics.median(values),
        "p90": nearest_rank(values, 0.90),
        "maximum": max(values),
    }


def section_and_rally_counts(
    sections_by_fixture: dict[str, tuple[Section, ...]],
    rallies_by_fixture: dict[str, tuple[Rally, ...]],
) -> tuple[dict[str, int], dict[str, int], set[tuple[str, int]], list[int], list[int]]:
    """Count clean, partial, merged and missed rally sections."""
    if sections_by_fixture.keys() != rallies_by_fixture.keys():
        raise ValueError("Prediction and label fixtures differ")

    section_counts = {
        "one_complete_rally": 0,
        "one_partial_rally": 0,
        "no_labelled_rally": 0,
        "several_labelled_rallies": 0,
    }
    rally_counts = {
        "clean_one_to_one": 0,
        "complete_but_merged": 0,
        "partial_or_split": 0,
        "missed": 0,
    }
    clean_section_ids: set[tuple[str, int]] = set()
    frames_before_first_contact: list[int] = []
    frames_after_last_contact: list[int] = []

    for fixture, sections in sections_by_fixture.items():
        rallies = rallies_by_fixture[fixture]
        for section in sections:
            overlaps: list[tuple[Rally, int]] = []
            for rally in rallies:
                contact_count = frames_inside(section, rally)
                if contact_count:
                    overlaps.append((rally, contact_count))
            if not overlaps:
                section_counts["no_labelled_rally"] += 1
                continue
            if len(overlaps) > 1:
                section_counts["several_labelled_rallies"] += 1
                continue
            rally, contact_count = overlaps[0]
            if contact_count != len(rally.frames):
                section_counts["one_partial_rally"] += 1
                continue
            section_counts["one_complete_rally"] += 1
            clean_section_ids.add((fixture, section.span_id))
            frames_before_first_contact.append(rally.frames[0] - section.start_frame)
            frames_after_last_contact.append(section.end_frame - 1 - rally.frames[-1])

        for rally in rallies:
            overlaps: list[tuple[Section, int]] = []
            for section in sections:
                contact_count = frames_inside(section, rally)
                if contact_count:
                    overlaps.append((section, contact_count))
            if not overlaps:
                rally_counts["missed"] += 1
                continue
            complete_sections = [
                section
                for section, contact_count in overlaps
                if contact_count == len(rally.frames)
            ]
            if len(complete_sections) != 1:
                rally_counts["partial_or_split"] += 1
                continue
            section = complete_sections[0]
            other_rally_present = any(
                other_rally != rally and frames_inside(section, other_rally)
                for other_rally in rallies
            )
            outcome = (
                "complete_but_merged" if other_rally_present else "clean_one_to_one"
            )
            rally_counts[outcome] += 1

    return (
        section_counts,
        rally_counts,
        clean_section_ids,
        frames_before_first_contact,
        frames_after_last_contact,
    )


def fully_correct_counts(
    result: JsonObject,
    clean_section_ids: set[tuple[str, int]],
    one_rally_sections: int,
    predicted_sections: int,
) -> dict[str, dict[str, int | float]]:
    """Compare the old whole-rally score with clean one-rally sections."""
    whole_rallies = result.get("whole_rallies")
    if not isinstance(whole_rallies, dict):
        raise TypeError("Expected result.whole_rallies to be an object")
    by_tolerance = whole_rallies.get("by_tolerance")
    if not isinstance(by_tolerance, dict):
        raise TypeError("Expected result.whole_rallies.by_tolerance to be an object")

    counts: dict[str, dict[str, int | float]] = {}
    for tolerance in ("5", "10"):
        tolerance_result = by_tolerance.get(tolerance)
        if not isinstance(tolerance_result, dict):
            raise TypeError(f"Expected whole-rally result for {tolerance} frames")
        rows = require_list(
            tolerance_result.get("sections"), f"sections at {tolerance}"
        )
        fully_correct = {
            (str(row["fixture"]), int(row["span_id"]))
            for row in rows
            if row["outcome"] == "fully_correct"
        }
        clean_and_correct = len(fully_correct & clean_section_ids)
        counts[tolerance] = {
            "old_scorer_fully_correct": len(fully_correct),
            "clean_span_and_fully_correct": clean_and_correct,
            "contact_tolerance_crossed_section_edge": len(
                fully_correct - clean_section_ids
            ),
            "share_of_one_rally_sections": len(fully_correct) / one_rally_sections,
            "share_of_all_predicted_sections": len(fully_correct) / predicted_sections,
            "clean_span_share_of_all_predicted_sections": clean_and_correct
            / predicted_sections,
        }
    return counts


def build_summary(
    predictions_path: Path,
    labels_path: Path,
    result_path: Path,
) -> JsonObject:
    """Build the post-test section summary."""
    predictions = load_json(predictions_path)
    labels = load_json(labels_path)
    result = load_json(result_path)
    sections_by_fixture = load_sections(predictions)
    rallies_by_fixture = load_rallies(labels)
    (
        section_counts,
        rally_counts,
        clean_section_ids,
        frames_before_first_contact,
        frames_after_last_contact,
    ) = section_and_rally_counts(sections_by_fixture, rallies_by_fixture)

    predicted_sections = sum(len(value) for value in sections_by_fixture.values())
    labelled_rallies = sum(len(value) for value in rallies_by_fixture.values())
    clean_sections = section_counts["one_complete_rally"]
    clean_rallies = rally_counts["clean_one_to_one"]
    if clean_sections != clean_rallies:
        raise ValueError("Clean section and rally counts differ")

    precision = clean_sections / predicted_sections
    recall = clean_rallies / labelled_rallies
    return {
        "schema": "shuttleset22-rally-section-summary/1",
        "inputs": {
            "combined_predictions_sha256": sha256(predictions_path),
            "clean_labels_sha256": sha256(labels_path),
            "test_result_sha256": sha256(result_path),
        },
        "rule": (
            "A correct rally section contains every labelled contact from one rally "
            "and no labelled contact from another rally."
        ),
        "frame_rate": 30,
        "predicted_sections": predicted_sections,
        "labelled_rallies": labelled_rallies,
        "correct_rally_sections": clean_sections,
        "false_positive_sections": predicted_sections - clean_sections,
        "missed_or_unclean_rallies": labelled_rallies - clean_rallies,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "section_counts": section_counts,
        "rally_counts": rally_counts,
        "clean_section_context_frames": {
            "before_first_labelled_contact": context_summary(
                frames_before_first_contact
            ),
            "after_last_labelled_contact": context_summary(frames_after_last_contact),
        },
        "whole_rally_contact_score": fully_correct_counts(
            result,
            clean_section_ids,
            section_counts["one_complete_rally"] + section_counts["one_partial_rally"],
            predicted_sections,
        ),
    }


def parse_args() -> argparse.Namespace:
    """Read the three saved inputs and an optional output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Recount the saved test and print or save the result."""
    args = parse_args()
    summary = build_summary(args.predictions, args.labels, args.result)
    encoded = json.dumps(summary, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
        return
    args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
