"""Classify buffered first/last-stroke candidates after an e2e measurement.

The analysis reads only closed runner outputs. It does not rerun inference or
change the measurement tree. Each output row describes one ground-truth first
or last stroke, the candidate selected by the existing wide search, its strict
ground-truth match, and its position relative to the associated predicted span.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from annotator.artifact_io import open_text_artifact, read_json_object
from annotator.calibration.scoring import RallyBoundary, classify_rally_boundary


class CandidateGtRelation(StrEnum):
    """How the selected buffered candidate relates to labelled strokes."""

    NO_CANDIDATE = "no_candidate"
    FIRST_LAST_STROKE_MATCH = "first_last_stroke_match"
    OTHER_GT_STROKE_MATCH = "other_gt_stroke_match"
    NO_GT_STROKE_MATCH = "no_gt_stroke_match"


class CandidateSpanRelation(StrEnum):
    """Where the selected candidate falls relative to the associated span."""

    NO_CANDIDATE = "no_candidate"
    NO_ASSOCIATED_SPAN = "no_associated_span"
    INSIDE_ASSOCIATED_SPAN = "inside_associated_span"
    BEFORE_ASSOCIATED_SPAN = "before_associated_span"
    AFTER_ASSOCIATED_SPAN = "after_associated_span"


@dataclass(frozen=True)
class BufferedSearchRow:
    """One ground-truth first/last-stroke buffer and its post-hoc result."""

    configuration_id: str
    window_id: int
    gt_rally_id: int
    first_or_last: str
    gt_frame: int
    buffer_start: int
    buffer_end: int
    buffer_has_candidate: bool
    selected_candidate_frame: int | None
    additional_candidate_count: int
    strict_tolerance_base30: int
    strict_tolerance_frames: int
    candidate_gt_relation: str
    matched_gt_rally_id: int | None
    matched_gt_frame: int | None
    predicted_boundary_classification: str
    associated_predicted_span_id: int | None
    associated_predicted_span_start: int | None
    associated_predicted_span_end: int | None
    candidate_predicted_rally_id: int | None
    candidate_assigned_to_associated_span: bool | None
    candidate_span_relation: str
    required_boundary_extension_frames: int | None


def _parse_required_int(row: dict[str, str], field: str, path: Path) -> int:
    value = row[field]
    if not value:
        raise ValueError(f"{path}: {field} is empty on a required row")
    return int(value)


def _load_gt_frames(
    path: Path, tolerance_base30: int,
) -> tuple[dict[int, tuple[int, ...]], int]:
    frames_by_rally: dict[int, set[int]] = defaultdict(set)
    tolerance_frames: set[int] = set()
    with open_text_artifact(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["tolerance_base30"]) != tolerance_base30:
                continue
            tolerance_frames.add(int(row["tolerance_frames"]))
            if row["gt_frame"]:
                frames_by_rally[int(row["rally_id"])].add(int(row["gt_frame"]))

    if not frames_by_rally:
        raise ValueError(f"{path}: no GT frames for base-30 tolerance {tolerance_base30}")
    if len(tolerance_frames) != 1:
        raise ValueError(
            f"{path}: expected one scaled tolerance for base-30 {tolerance_base30}, "
            f"found {sorted(tolerance_frames)}"
        )
    ordered = {
        rally_id: tuple(sorted(frames))
        for rally_id, frames in sorted(frames_by_rally.items())
    }
    return ordered, tolerance_frames.pop()


def _load_annotations(path: Path) -> tuple[list[tuple[int, int]], dict[int, int]]:
    payload = read_json_object(path)
    spans: list[tuple[int, int]] = []
    for raw_span in payload["spans"]:
        if len(raw_span) != 2:
            raise ValueError(f"{path}: every predicted span must have two frame bounds")
        start, end = raw_span
        spans.append((int(start), int(end)))

    candidate_rally_ids: dict[int, int] = {}
    for contact in payload["filtered_contacts"]:
        candidate_frame = int(contact["contact_frame"])
        rally_id = int(contact["rally_id"])
        previous = candidate_rally_ids.setdefault(candidate_frame, rally_id)
        if previous != rally_id:
            raise ValueError(
                f"{path}: candidate frame {candidate_frame} belongs to predicted rallies "
                f"{previous} and {rally_id}"
            )
    return spans, candidate_rally_ids


def _load_buffer_rows(
    path: Path,
) -> tuple[list[dict[str, str]], dict[int, int]]:
    targets: dict[int, dict[str, str]] = {}
    additional_counts: dict[int, int] = defaultdict(int)
    with open_text_artifact(path, newline="") as handle:
        for row in csv.DictReader(handle):
            window_id = int(row["window_id"])
            if row["row_kind"] == "unmatched_candidate":
                additional_counts[window_id] += 1
                continue
            if row["row_kind"] not in {"matched", "unmatched_gt"}:
                raise ValueError(f"{path}: unknown row_kind {row['row_kind']!r}")
            if window_id in targets:
                raise ValueError(f"{path}: window {window_id} has more than one target row")
            targets[window_id] = row
    if not targets:
        raise ValueError(f"{path}: no first/last-stroke target rows")
    return [targets[window_id] for window_id in sorted(targets)], additional_counts


def _closest_other_gt_stroke(
    candidate_frame: int,
    target_rally_id: int,
    target_frame: int,
    gt_frames_by_rally: dict[int, tuple[int, ...]],
    tolerance_frames: int,
) -> tuple[int, int] | None:
    ranked: list[tuple[int, int, int]] = []
    for rally_id, gt_frames in gt_frames_by_rally.items():
        for gt_frame in gt_frames:
            if rally_id == target_rally_id and gt_frame == target_frame:
                continue
            distance = abs(candidate_frame - gt_frame)
            if distance <= tolerance_frames:
                ranked.append((distance, rally_id, gt_frame))
    if not ranked:
        return None
    _distance, rally_id, gt_frame = min(ranked)
    return rally_id, gt_frame


def _candidate_gt_match(
    candidate_frame: int | None,
    target_rally_id: int,
    target_frame: int,
    gt_frames_by_rally: dict[int, tuple[int, ...]],
    tolerance_frames: int,
) -> tuple[CandidateGtRelation, int | None, int | None]:
    if candidate_frame is None:
        return CandidateGtRelation.NO_CANDIDATE, None, None
    if abs(candidate_frame - target_frame) <= tolerance_frames:
        return CandidateGtRelation.FIRST_LAST_STROKE_MATCH, target_rally_id, target_frame
    other = _closest_other_gt_stroke(
        candidate_frame,
        target_rally_id,
        target_frame,
        gt_frames_by_rally,
        tolerance_frames,
    )
    if other is None:
        return CandidateGtRelation.NO_GT_STROKE_MATCH, None, None
    return CandidateGtRelation.OTHER_GT_STROKE_MATCH, other[0], other[1]


def _candidate_span_position(
    candidate_frame: int | None,
    boundary: RallyBoundary,
    span_id: int | None,
    spans: Sequence[tuple[int, int]],
) -> tuple[CandidateSpanRelation, int | None, int | None, int | None]:
    if boundary is not RallyBoundary.COVERED or span_id is None:
        if candidate_frame is None:
            return CandidateSpanRelation.NO_CANDIDATE, None, None, None
        return CandidateSpanRelation.NO_ASSOCIATED_SPAN, None, None, None

    start, end = spans[span_id]
    if candidate_frame is None:
        return CandidateSpanRelation.NO_CANDIDATE, start, end, None
    if candidate_frame < start:
        return CandidateSpanRelation.BEFORE_ASSOCIATED_SPAN, start, end, start - candidate_frame
    if candidate_frame >= end:
        return CandidateSpanRelation.AFTER_ASSOCIATED_SPAN, start, end, candidate_frame - end + 1
    return CandidateSpanRelation.INSIDE_ASSOCIATED_SPAN, start, end, 0


def analyse_leaf(
    leaf_dir: Path,
    configuration_id: str,
    tolerance_base30: int,
) -> list[BufferedSearchRow]:
    """Analyse one successful e2e configuration leaf.

    :param leaf_dir: Directory containing annotations and scoring CSVs.
    :param configuration_id: Stable parent/case identifier from the root manifest.
    :param tolerance_base30: Strict base-30 frame tolerance used to verify candidates.
    :return: One row per first/last-stroke buffer.
    """
    gt_frames_by_rally, tolerance_frames = _load_gt_frames(
        leaf_dir / "strict_contacts.csv.gz", tolerance_base30
    )
    spans, candidate_rally_ids = _load_annotations(leaf_dir / "annotations.json.gz")
    targets, additional_counts = _load_buffer_rows(leaf_dir / "wide_edge_contacts.csv.gz")

    rows: list[BufferedSearchRow] = []
    for target in targets:
        window_id = int(target["window_id"])
        gt_rally_id = int(target["rally_id"])
        first_or_last = target["edge"]
        gt_frame = _parse_required_int(target, "gt_frame", leaf_dir / "wide_edge_contacts.csv.gz")
        gt_frames = gt_frames_by_rally[gt_rally_id]
        expected_gt_frame = gt_frames[0] if first_or_last == "first" else gt_frames[-1]
        if gt_frame != expected_gt_frame:
            raise ValueError(
                f"{leaf_dir}: {first_or_last} target {gt_frame} does not match "
                f"rally {gt_rally_id} GT extent {gt_frames[0]}..{gt_frames[-1]}"
            )

        candidate_frame = int(target["candidate_frame"]) if target["candidate_frame"] else None
        candidate_gt_relation, matched_gt_rally_id, matched_gt_frame = _candidate_gt_match(
            candidate_frame,
            gt_rally_id,
            gt_frame,
            gt_frames_by_rally,
            tolerance_frames,
        )
        boundary, span_id = classify_rally_boundary(gt_frames, spans)
        span_relation, span_start, span_end, required_extension = _candidate_span_position(
            candidate_frame, boundary, span_id, spans
        )
        candidate_predicted_rally_id = (
            candidate_rally_ids[candidate_frame] if candidate_frame is not None else None
        )
        assigned_to_associated_span = (
            candidate_predicted_rally_id == span_id
            if candidate_predicted_rally_id is not None and span_id is not None
            else None
        )

        rows.append(BufferedSearchRow(
            configuration_id=configuration_id,
            window_id=window_id,
            gt_rally_id=gt_rally_id,
            first_or_last=first_or_last,
            gt_frame=gt_frame,
            buffer_start=int(target["window_start"]),
            buffer_end=int(target["window_end"]),
            buffer_has_candidate=candidate_frame is not None,
            selected_candidate_frame=candidate_frame,
            additional_candidate_count=additional_counts.get(window_id, 0),
            strict_tolerance_base30=tolerance_base30,
            strict_tolerance_frames=tolerance_frames,
            candidate_gt_relation=candidate_gt_relation.value,
            matched_gt_rally_id=matched_gt_rally_id,
            matched_gt_frame=matched_gt_frame,
            predicted_boundary_classification=boundary.value,
            associated_predicted_span_id=span_id,
            associated_predicted_span_start=span_start,
            associated_predicted_span_end=span_end,
            candidate_predicted_rally_id=candidate_predicted_rally_id,
            candidate_assigned_to_associated_span=assigned_to_associated_span,
            candidate_span_relation=span_relation.value,
            required_boundary_extension_frames=required_extension,
        ))
    return rows


def analyse_measurement(
    measurement_root: Path,
    tolerance_base30: int = 5,
) -> list[BufferedSearchRow]:
    """Analyse every successful configuration declared by a root manifest."""
    root_manifest = read_json_object(measurement_root / "manifest.json.gz")
    if root_manifest.get("status") != "succeeded":
        raise ValueError(f"{measurement_root}: root manifest is not successful")

    rows: list[BufferedSearchRow] = []
    for configuration in root_manifest["configurations"]:
        if configuration["status"] != "succeeded":
            raise ValueError(
                f"{measurement_root}: configuration {configuration['configuration_id']} "
                f"has status {configuration['status']!r}"
            )
        configuration_id = str(configuration["configuration_id"])
        rows.extend(analyse_leaf(
            measurement_root / configuration_id,
            configuration_id,
            tolerance_base30,
        ))
    return rows


def write_rows(rows: Sequence[BufferedSearchRow], output_path: Path) -> None:
    """Write the post-hoc rows to a CSV outside the closed measurement tree."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[field.name for field in fields(BufferedSearchRow)],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def timestamped_output_path(output_directory: Path) -> Path:
    """Return the required local-time output filename in ``output_directory``."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return output_directory / f"first_last_stroke_buffered_search_{timestamp}.csv"


def _print_summary(rows: Sequence[BufferedSearchRow]) -> None:
    configuration_ids = dict.fromkeys(row.configuration_id for row in rows)
    for configuration_id in configuration_ids:
        configuration_rows = [row for row in rows if row.configuration_id == configuration_id]
        with_candidate = sum(row.buffer_has_candidate for row in configuration_rows)
        without_candidate = len(configuration_rows) - with_candidate
        first_last_matches = sum(
            row.candidate_gt_relation == CandidateGtRelation.FIRST_LAST_STROKE_MATCH
            for row in configuration_rows
        )
        other_gt_matches = sum(
            row.candidate_gt_relation == CandidateGtRelation.OTHER_GT_STROKE_MATCH
            for row in configuration_rows
        )
        without_gt_match = sum(
            row.candidate_gt_relation == CandidateGtRelation.NO_GT_STROKE_MATCH
            for row in configuration_rows
        )
        first_last_matches_outside_span = sum(
            row.candidate_gt_relation == CandidateGtRelation.FIRST_LAST_STROKE_MATCH
            and row.candidate_span_relation
            in {
                CandidateSpanRelation.BEFORE_ASSOCIATED_SPAN,
                CandidateSpanRelation.AFTER_ASSOCIATED_SPAN,
            }
            for row in configuration_rows
        )
        additional_candidates = sum(row.additional_candidate_count for row in configuration_rows)
        print(
            f"{configuration_id}: "
            f"first_last_buffers_with_candidate={with_candidate} "
            f"first_last_buffers_without_candidate={without_candidate} "
            f"selected_candidates_matching_first_last_stroke={first_last_matches} "
            f"selected_candidates_matching_other_gt_stroke={other_gt_matches} "
            f"selected_candidates_without_gt_match={without_gt_match} "
            f"first_last_matches_outside_associated_span={first_last_matches_outside_span} "
            f"additional_candidates_in_buffers={additional_candidates}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "measurement_root",
        type=Path,
        help="Completed e2e output root containing the terminal manifest.json.gz",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="Directory for first_last_stroke_buffered_search_<DATETIME>.csv",
    )
    parser.add_argument(
        "--tolerance-base30",
        type=int,
        default=5,
        help="Strict base-30 tolerance used to verify the selected candidate (default: 5)",
    )
    args = parser.parse_args(argv)

    measurement_root = args.measurement_root.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory == measurement_root or measurement_root in output_directory.parents:
        parser.error("output directory must be outside the closed measurement root")
    output_path = timestamped_output_path(output_directory)
    if output_path.exists():
        parser.error(f"timestamped output already exists: {output_path}")

    rows = analyse_measurement(measurement_root, args.tolerance_base30)
    write_rows(rows, output_path)
    _print_summary(rows)
    print(f"wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
