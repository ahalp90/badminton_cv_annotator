"""Build and measure the fixed rally-start candidate list."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scratch.contact_det_full_ds_fit.scripts import (
    check_missed_contacts as missed_checker,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    RESULT_SCHEMA,
    _scaled_frames,
)
from scratch.contact_det_full_ds_fit.scripts.score_validation_rallies import (
    RALLY_RESULT_SCHEMA,
    SavedRunPredictions,
    VerifiedRallyPredictions,
    _read_json,
    _sha256,
    _write_json,
    load_validation_rally_predictions,
)

CONSTRUCTION_SCHEMA = "full-dataset-rally-start-candidates/1"
RESULT_SCHEMA_NAME = "full-dataset-rally-start-candidate-check/1"
MAX_CANDIDATES_PER_SECTION = 3
MAX_TOTAL_CANDIDATES = 1_845
MIN_COVERED_TARGET_CONTACTS = 50
MAX_ADDED_PER_COVERED_CONTACT = 25.0
TARGET_CONTACT_COUNT = 81
DUPLICATE_DISTANCE_AT_30_FPS = 6


@dataclass(frozen=True)
class CandidateInputs:
    """Checked model outputs that do not expose contact-label rows."""

    summary: Mapping[str, Any]
    run_result: Mapping[str, Any]
    score_rows: np.ndarray
    verified_predictions: VerifiedRallyPredictions
    saved_run: SavedRunPredictions
    intervals_by_fixture: Mapping[str, tuple[tuple[int, int], ...]]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    return missed_checker._mapping(value, label)


def _json_bytes(value: Mapping[str, object]) -> bytes:
    """Return the fixed compact JSON form used for the construction check."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_gzip_bytes(path: Path, value: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as zipped,
    ):
        zipped.write(value)
    os.replace(temporary, destination)


def _load_search_intervals(
    path: Path,
    verified: VerifiedRallyPredictions,
    score_rows: np.ndarray,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Read the checked search ranges used to assign score interval IDs."""
    raw_record = _mapping(
        json.loads(Path(path).read_text(encoding="utf-8")),
        "raw feature record",
    )
    raw_videos = raw_record.get("videos")
    if not isinstance(raw_videos, list):
        raise TypeError("raw feature record videos must be a list")
    by_fixture: dict[str, Mapping[str, Any]] = {}
    for raw_video in raw_videos:
        record = _mapping(raw_video, "raw feature video")
        identity = _mapping(record.get("video"), "raw feature video identity")
        fixture = identity.get("name")
        if not isinstance(fixture, str) or fixture in by_fixture:
            raise ValueError("raw feature video identities differ")
        by_fixture[fixture] = record

    score_names = np.char.decode(score_rows["fixture"], "ascii")
    output: dict[str, tuple[tuple[int, int], ...]] = {}
    for video in verified.split.validation_videos:
        record = by_fixture.get(video.fixture)
        if record is None:
            raise ValueError(f"{video.fixture}: raw feature record is missing")
        summary = _mapping(
            record.get("feature_summary"), f"{video.fixture}: feature summary"
        )
        raw_intervals = summary.get("search_intervals")
        if not isinstance(raw_intervals, list):
            raise TypeError(f"{video.fixture}: search intervals must be a list")
        intervals: list[tuple[int, int]] = []
        for raw_interval in raw_intervals:
            if (
                not isinstance(raw_interval, list)
                or len(raw_interval) != 2
                or type(raw_interval[0]) is not int
                or type(raw_interval[1]) is not int
            ):
                raise ValueError(f"{video.fixture}: search interval differs")
            start, end = raw_interval
            if start < 0 or end <= start:
                raise ValueError(f"{video.fixture}: search interval differs")
            intervals.append((start, end))

        video_rows = score_rows[score_names == video.fixture]
        if not np.array_equal(
            np.unique(video_rows["interval_id"]),
            np.arange(len(intervals), dtype=video_rows["interval_id"].dtype),
        ):
            raise ValueError(f"{video.fixture}: score interval IDs differ")
        for interval_id, (start, end) in enumerate(intervals):
            frames = video_rows["frame"][video_rows["interval_id"] == interval_id]
            if len(frames) == 0 or np.any((frames < start) | (frames >= end)):
                raise ValueError(f"{video.fixture}: score interval frames differ")
        output[video.fixture] = tuple(intervals)
    return output


def load_candidate_inputs(
    summary_path: Path,
    rally_result_path: Path,
    prediction_path: Path,
    menu_result_path: Path,
    run_result_path: Path,
    score_path: Path,
    split_path: Path,
    raw_feature_record_path: Path,
    shots_master_path: Path,
) -> CandidateInputs:
    """Check the model outputs without parsing saved label-derived results."""
    summary = _mapping(_read_json(summary_path), "baseline summary")
    missed_checker._check_named_hash(
        summary,
        rally_result_path,
        "result_file",
        "result_sha256",
        "rally result",
    )
    missed_checker._check_named_hash(
        summary,
        prediction_path,
        "prediction_file",
        "prediction_sha256",
        "rally prediction file",
    )
    selected_summary = missed_checker._selected_summary_run(summary)
    run_id = summary.get("chosen_run_id")
    if not isinstance(run_id, str):
        raise TypeError("chosen run ID must be text")
    chosen_files = _mapping(summary.get("chosen_run_files"), "chosen run files")
    missed_checker._check_named_hash(
        chosen_files,
        run_result_path,
        "run_result_file",
        "run_result_sha256",
        "chosen run result",
    )
    missed_checker._check_named_hash(
        chosen_files,
        score_path,
        "score_file",
        "score_sha256",
        "chosen score file",
    )

    verified = load_validation_rally_predictions(
        prediction_path,
        menu_result_path,
        split_path,
        raw_feature_record_path,
        shots_master_path,
    )
    matching_runs = [
        saved_run for saved_run in verified.runs if saved_run.run_id == run_id
    ]
    if len(matching_runs) != 1:
        raise ValueError("saved prediction chosen run differs")
    saved_run = matching_runs[0]

    run_result = _mapping(_read_json(run_result_path), "chosen baseline run result")
    if (
        run_result.get("schema") != RESULT_SCHEMA
        or run_result.get("status") != "complete"
        or run_result.get("run_id") != run_id
        or run_result.get("contact_label_sha256")
        != verified.payload["contact_label_sha256"]
        or run_result.get("split_sha256") != verified.payload["split_sha256"]
        or run_result.get("validation_videos")
        != [video.fixture for video in verified.split.validation_videos]
        or run_result.get("selected_score_cutoff")
        != selected_summary.get("selected_score_cutoff")
        or run_result.get("selected_duplicate_distance_at_30_fps")
        != selected_summary.get("selected_duplicate_distance_at_30_fps")
    ):
        raise ValueError("chosen baseline run result differs")
    score_rows = missed_checker._load_score_rows(score_path, run_result)
    missed_checker._check_score_identities(score_rows, verified, saved_run)
    intervals = _load_search_intervals(raw_feature_record_path, verified, score_rows)
    return CandidateInputs(
        summary, run_result, score_rows, verified, saved_run, intervals
    )


def _candidate(
    row: np.void,
    *,
    is_fixed_contact: bool,
) -> dict[str, object]:
    return {
        "frame": int(row["frame"]),
        "contact_score": float(row["contact_score"]),
        "is_fixed_contact": is_fixed_contact,
    }


def build_video_candidate_lists(
    fixture: str,
    fps: float,
    video_rows: np.ndarray,
    kept_frames: Sequence[int],
    spans: Sequence[Mapping[str, int]],
    intervals: Sequence[tuple[int, int]],
    duplicate_distance_at_30_fps: int,
) -> tuple[list[dict[str, object]], int]:
    """Build the fixed candidate lists for one video's detected sections."""
    if duplicate_distance_at_30_fps != DUPLICATE_DISTANCE_AT_30_FPS:
        raise ValueError("selected duplicate distance must be six frames at 30 fps")
    distance = _scaled_frames(duplicate_distance_at_30_fps, fps)
    candidate_lists: list[dict[str, object]] = []
    sections_without_kept_contact = 0
    previous_end: int | None = None

    for span in spans:
        span_id = int(span["span_id"])
        span_start = int(span["start_frame"])
        span_end = int(span["end_frame"])
        section_frames = [
            frame for frame in kept_frames if span_start <= frame < span_end
        ]
        if not section_frames:
            sections_without_kept_contact += 1
            previous_end = span_end
            continue

        fixed_frame = section_frames[0]
        fixed_rows = video_rows[video_rows["frame"] == fixed_frame]
        if len(fixed_rows) != 1 or not bool(fixed_rows[0]["kept"]):
            raise ValueError(f"{fixture}/{span_id}: fixed contact score row differs")
        fixed_row = fixed_rows[0]
        interval_id = int(fixed_row["interval_id"])
        interval_start, interval_end = intervals[interval_id]
        prefix_start = interval_start
        if previous_end is not None and interval_start <= previous_end < interval_end:
            prefix_start = max(prefix_start, previous_end)

        possible = video_rows[
            (video_rows["interval_id"] == interval_id)
            & (video_rows["frame"] >= prefix_start)
            & (video_rows["frame"] < fixed_frame)
        ]
        ordered = sorted(
            possible,
            key=lambda row: (-float(row["contact_score"]), int(row["frame"])),
        )
        earlier_rows: list[np.void] = []
        for row in ordered:
            frame = int(row["frame"])
            chosen_frames = [
                fixed_frame,
                *(int(chosen["frame"]) for chosen in earlier_rows),
            ]
            if all(
                abs(frame - chosen_frame) > distance for chosen_frame in chosen_frames
            ):
                earlier_rows.append(row)
            if len(earlier_rows) == MAX_CANDIDATES_PER_SECTION - 1:
                break

        candidates = [
            _candidate(fixed_row, is_fixed_contact=True),
            *(_candidate(row, is_fixed_contact=False) for row in earlier_rows),
        ]
        candidate_lists.append(
            {
                "fixture": fixture,
                "span_id": span_id,
                "section_start_frame": span_start,
                "section_end_frame": span_end,
                "interval_id": interval_id,
                "prefix_start_frame": prefix_start,
                "fixed_contact_frame": fixed_frame,
                "duplicate_distance_frames": distance,
                "candidates": candidates,
            }
        )
        previous_end = span_end

    return candidate_lists, sections_without_kept_contact


def build_candidate_construction(
    inputs: CandidateInputs,
    source_commit: str,
) -> dict[str, object]:
    """Build the fixed candidate lists without saved contact-label detail."""
    score_names = np.char.decode(inputs.score_rows["fixture"], "ascii")
    raw_duplicate_distance = inputs.run_result.get(
        "selected_duplicate_distance_at_30_fps"
    )
    if (
        type(raw_duplicate_distance) is not int
        or raw_duplicate_distance != DUPLICATE_DISTANCE_AT_30_FPS
    ):
        raise ValueError("selected duplicate distance must be six frames at 30 fps")
    duplicate_at_30 = raw_duplicate_distance
    candidate_lists: list[dict[str, object]] = []
    sections_without_kept_contact = 0
    video_counts: dict[str, dict[str, int]] = {}

    for video in inputs.verified_predictions.split.validation_videos:
        fixture = video.fixture
        video_rows = inputs.score_rows[score_names == fixture]
        events = inputs.saved_run.events_by_fixture[fixture]
        spans = inputs.verified_predictions.spans_by_fixture[fixture]
        intervals = inputs.intervals_by_fixture[fixture]
        video_lists, skipped = build_video_candidate_lists(
            fixture,
            video.fps,
            video_rows,
            [event.frame for event in events],
            spans,
            intervals,
            duplicate_at_30,
        )
        candidate_lists.extend(video_lists)
        sections_without_kept_contact += skipped
        video_candidate_count = sum(len(row["candidates"]) for row in video_lists)
        video_added_count = sum(len(row["candidates"]) - 1 for row in video_lists)
        video_counts[fixture] = {
            "detected_sections": len(spans),
            "candidate_lists": len(video_lists),
            "candidate_entries": video_candidate_count,
            "earlier_candidate_entries": video_added_count,
        }

    total_candidates = sum(len(row["candidates"]) for row in candidate_lists)
    added_candidates = sum(
        not bool(candidate["is_fixed_contact"])
        for row in candidate_lists
        for candidate in row["candidates"]
    )
    return {
        "schema": CONSTRUCTION_SCHEMA,
        "status": "complete",
        "source_commit": source_commit,
        "run_id": inputs.saved_run.run_id,
        "selected_duplicate_distance_at_30_fps": duplicate_at_30,
        "limits": {
            "maximum_candidates_per_section": MAX_CANDIDATES_PER_SECTION,
            "maximum_total_candidates": MAX_TOTAL_CANDIDATES,
        },
        "counts": {
            "detected_sections": sum(
                len(spans)
                for spans in inputs.verified_predictions.spans_by_fixture.values()
            ),
            "candidate_lists": len(candidate_lists),
            "sections_without_kept_contact": sections_without_kept_contact,
            "candidate_entries": total_candidates,
            "fixed_contact_entries": len(candidate_lists),
            "earlier_candidate_entries": added_candidates,
        },
        "video_counts": video_counts,
        "candidate_lists": candidate_lists,
    }


def freeze_candidate_construction(
    inputs: CandidateInputs,
    source_commit: str,
    output_path: Path,
) -> tuple[dict[str, object], str]:
    """Build the construction twice, then save its fixed byte form."""
    first = build_candidate_construction(inputs, source_commit)
    second = build_candidate_construction(inputs, source_commit)
    first_bytes = _json_bytes(first)
    if first_bytes != _json_bytes(second):
        raise ValueError("candidate construction differs between repeated builds")
    _write_gzip_bytes(output_path, first_bytes)
    with gzip.open(output_path, "rb") as saved:
        if saved.read() != first_bytes:
            raise ValueError("saved candidate construction differs")
    return first, _sha256(output_path)


def load_missed_contact_result(
    summary_path: Path,
    result_path: Path,
    inputs: CandidateInputs,
    baseline_summary_path: Path,
    rally_result_path: Path,
    prediction_path: Path,
    menu_result_path: Path,
    run_result_path: Path,
    score_path: Path,
    split_path: Path,
    raw_feature_record_path: Path,
    shots_master_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Verify the label-derived result bytes before parsing its details."""
    summary = _mapping(_read_json(summary_path), "missed-contact summary")
    missed_checker._check_named_hash(
        summary,
        result_path,
        "result_file",
        "result_sha256",
        "missed-contact result",
    )
    result = _mapping(_read_json(result_path), "missed-contact result")
    saved_inputs = _mapping(result.get("inputs"), "missed-contact inputs")
    if (
        result.get("schema") != missed_checker.CHECK_SCHEMA
        or result.get("status") != "complete"
        or result.get("run_id") != inputs.saved_run.run_id
        or summary.get("run_id") != inputs.saved_run.run_id
        or saved_inputs.get("baseline_summary_sha256") != _sha256(baseline_summary_path)
        or saved_inputs.get("rally_result_sha256") != _sha256(rally_result_path)
        or saved_inputs.get("rally_prediction_sha256") != _sha256(prediction_path)
        or saved_inputs.get("menu_result_sha256") != _sha256(menu_result_path)
        or saved_inputs.get("run_result_sha256") != _sha256(run_result_path)
        or saved_inputs.get("score_sha256") != _sha256(score_path)
        or saved_inputs.get("split_sha256") != _sha256(split_path)
        or saved_inputs.get("raw_feature_record_sha256")
        != _sha256(raw_feature_record_path)
        or saved_inputs.get("contact_label_sha256") != _sha256(shots_master_path)
    ):
        raise ValueError("missed-contact result differs from the fixed inputs")
    rally_result = _mapping(_read_json(rally_result_path), "rally result")
    raw_runs = rally_result.get("runs")
    if (
        rally_result.get("schema") != RALLY_RESULT_SCHEMA
        or rally_result.get("status") != "complete"
        or not isinstance(raw_runs, list)
        or sum(
            isinstance(raw_run, Mapping)
            and raw_run.get("run_id") == inputs.saved_run.run_id
            for raw_run in raw_runs
        )
        != 1
    ):
        raise ValueError("rally result differs from the fixed inputs")
    return result, rally_result


def _number_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def _added_candidates(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_candidates = row.get("candidates")
    if not isinstance(raw_candidates, list):
        raise TypeError("candidate list entries must be a list")
    return [
        _mapping(candidate, "candidate")
        for candidate in raw_candidates
        if isinstance(candidate, Mapping) and candidate.get("is_fixed_contact") is False
    ]


def _coverage_at_tolerance(
    target_details: Sequence[Mapping[str, Any]],
    lists_by_section: Mapping[tuple[str, int], Mapping[str, Any]],
    fps_by_fixture: Mapping[str, float],
    tolerance_at_30_fps: int,
) -> dict[str, object]:
    covered = 0
    before_start = 0
    only_before_start = 0
    matching_scores: list[float] = []
    matching_distances: list[float] = []
    other_scores: list[float] = []
    other_distances: list[float] = []
    for detail in target_details:
        fixture = str(detail["fixture"])
        span_id = int(detail["span_id"])
        missing_frame = int(detail["missing_frame"])
        section_start = int(detail["section_start_frame"])
        candidate_row = lists_by_section.get((fixture, span_id))
        candidates = [] if candidate_row is None else _added_candidates(candidate_row)
        tolerance = _scaled_frames(tolerance_at_30_fps, fps_by_fixture[fixture])
        matches: list[Mapping[str, Any]] = []
        for candidate in candidates:
            frame = int(candidate["frame"])
            distance = abs(frame - missing_frame)
            if distance <= tolerance:
                matches.append(candidate)
                matching_scores.append(float(candidate["contact_score"]))
                matching_distances.append(float(distance))
            else:
                other_scores.append(float(candidate["contact_score"]))
                other_distances.append(float(distance))
        if matches:
            covered += 1
            has_before = any(
                int(candidate["frame"]) < section_start for candidate in matches
            )
            has_inside = any(
                int(candidate["frame"]) >= section_start for candidate in matches
            )
            before_start += has_before
            only_before_start += has_before and not has_inside
    return {
        "tolerance_at_30_fps": tolerance_at_30_fps,
        "target_contacts": len(target_details),
        "covered_contacts": covered,
        "covered_share": covered / len(target_details) if target_details else None,
        "covered_with_a_candidate_before_section": before_start,
        "covered_only_by_candidates_before_section": only_before_start,
        "matching_candidate_scores": _number_summary(matching_scores),
        "matching_candidate_absolute_frame_distances": _number_summary(
            matching_distances
        ),
        "other_candidate_scores": _number_summary(other_scores),
        "other_candidate_absolute_frame_distances": _number_summary(other_distances),
    }


def _all_missed_first_coverage(
    details: Sequence[Mapping[str, Any]],
    construction: Mapping[str, Any],
    section_by_rally: Mapping[tuple[str, str], tuple[str, int]],
) -> dict[str, int]:
    raw_lists = construction.get("candidate_lists")
    if not isinstance(raw_lists, list):
        raise TypeError("candidate lists must be a list")
    candidates_by_section: dict[tuple[str, int], list[int]] = {}
    for raw_row in raw_lists:
        row = _mapping(raw_row, "candidate list")
        identity = (str(row["fixture"]), int(row["span_id"]))
        candidates_by_section.setdefault(identity, []).extend(
            int(candidate["frame"]) for candidate in _added_candidates(row)
        )
    first = [detail for detail in details if detail.get("contact_type") == "first"]
    assigned = [
        (
            detail,
            section_by_rally.get((str(detail["fixture"]), str(detail["rally_id"]))),
        )
        for detail in first
    ]
    covered = sum(
        section is not None
        and any(
            abs(candidate_frame - int(detail["frame"]))
            <= int(detail["tolerance_frames"])
            for candidate_frame in candidates_by_section.get(section, ())
        )
        for detail, section in assigned
    )
    assigned_count = sum(section is not None for _detail, section in assigned)
    return {
        "missed_first_contacts": len(first),
        "assigned_to_one_detected_section": assigned_count,
        "not_assigned_to_one_detected_section": len(first) - assigned_count,
        "covered_first_contacts": covered,
    }


def _section_by_rally(
    rally_result: Mapping[str, Any],
    run_id: str,
) -> dict[tuple[str, str], tuple[str, int]]:
    raw_runs = rally_result.get("runs")
    if not isinstance(raw_runs, list):
        raise TypeError("rally result runs must be a list")
    chosen = next(
        _mapping(raw_run, "chosen rally result")
        for raw_run in raw_runs
        if isinstance(raw_run, Mapping) and raw_run.get("run_id") == run_id
    )
    primary = _mapping(chosen.get("primary"), "chosen primary rally result")
    raw_spans = primary.get("spans")
    if not isinstance(raw_spans, list):
        raise TypeError("chosen rally sections must be a list")
    output: dict[tuple[str, str], tuple[str, int]] = {}
    for raw_span in raw_spans:
        span = _mapping(raw_span, "chosen rally section")
        rally_id = span.get("rally_id")
        if rally_id is None:
            continue
        if not isinstance(rally_id, str):
            raise TypeError("chosen rally ID must be text")
        fixture = str(span["fixture"])
        identity = (fixture, rally_id)
        if identity in output:
            raise ValueError(
                "a labelled rally is assigned to more than one detected section"
            )
        output[identity] = (fixture, int(span["span_id"]))
    return output


def measure_candidate_construction(
    construction: Mapping[str, Any],
    missed_result: Mapping[str, Any],
    rally_result: Mapping[str, Any],
    inputs: CandidateInputs,
) -> dict[str, object]:
    """Measure the fixed list against the already checked missed contacts."""
    raw_lists = construction.get("candidate_lists")
    if not isinstance(raw_lists, list):
        raise TypeError("candidate lists must be a list")
    candidate_lists = [_mapping(row, "candidate list") for row in raw_lists]
    lists_by_section = {
        (str(row["fixture"]), int(row["span_id"])): row for row in candidate_lists
    }
    if len(lists_by_section) != len(candidate_lists):
        raise ValueError("candidate section identities are repeated")

    details = _mapping(missed_result.get("details"), "missed-contact details")
    raw_targets = details.get("otherwise_correct_one_short_sections")
    if not isinstance(raw_targets, list):
        raise TypeError("one-short section details must be a list")
    target_details = [
        _mapping(row, "one-short section")
        for row in raw_targets
        if isinstance(row, Mapping) and row.get("missing_contact_type") == "first"
    ]
    if len(target_details) != TARGET_CONTACT_COUNT:
        raise ValueError("target first-contact count differs")
    if len({(row["fixture"], row["span_id"]) for row in target_details}) != len(
        target_details
    ):
        raise ValueError("target section identities are repeated")

    fps_by_fixture = {
        video.fixture: video.fps
        for video in inputs.verified_predictions.split.validation_videos
    }
    target_coverage = {
        str(tolerance): _coverage_at_tolerance(
            target_details,
            lists_by_section,
            fps_by_fixture,
            tolerance,
        )
        for tolerance in (5, 10)
    }

    raw_missed = details.get("missed_contacts")
    if not isinstance(raw_missed, Mapping):
        raise TypeError("missed-contact detail groups must be an object")
    all_first_coverage: dict[str, dict[str, int]] = {}
    section_by_rally = _section_by_rally(rally_result, inputs.saved_run.run_id)
    for tolerance in (5, 10):
        tolerance_details = raw_missed.get(str(tolerance))
        if not isinstance(tolerance_details, list):
            raise TypeError("missed-contact details must be a list")
        all_first_coverage[str(tolerance)] = _all_missed_first_coverage(
            [_mapping(row, "missed contact") for row in tolerance_details],
            construction,
            section_by_rally,
        )

    counts = _mapping(construction.get("counts"), "construction counts")
    total_candidates = int(counts["candidate_entries"])
    added_candidates = int(counts["earlier_candidate_entries"])
    maximum_section_size = max(
        (len(_mapping(row, "candidate list")["candidates"]) for row in candidate_lists),
        default=0,
    )
    covered_at_10 = int(target_coverage["10"]["covered_contacts"])
    added_per_covered = added_candidates / covered_at_10 if covered_at_10 else None
    checks = {
        "maximum_candidates_per_section": maximum_section_size
        <= MAX_CANDIDATES_PER_SECTION,
        "maximum_total_candidates": total_candidates <= MAX_TOTAL_CANDIDATES,
        "minimum_covered_target_contacts": covered_at_10 >= MIN_COVERED_TARGET_CONTACTS,
        "maximum_added_per_covered_contact": (
            added_per_covered is not None
            and added_per_covered <= MAX_ADDED_PER_COVERED_CONTACT
        ),
    }
    return {
        "construction_counts": counts,
        "maximum_candidates_in_one_section": maximum_section_size,
        "target_first_contacts": target_coverage,
        "all_missed_first_contacts": all_first_coverage,
        "added_candidates_per_covered_target_at_10_frames": added_per_covered,
        "fixed_limits": {
            "maximum_candidates_per_section": MAX_CANDIDATES_PER_SECTION,
            "maximum_total_candidates": MAX_TOTAL_CANDIDATES,
            "minimum_covered_target_contacts": MIN_COVERED_TARGET_CONTACTS,
            "maximum_added_per_covered_contact": MAX_ADDED_PER_COVERED_CONTACT,
        },
        "limit_checks": checks,
        "all_limits_pass": all(checks.values()),
    }


def check_rally_start_candidates(
    summary_path: Path,
    missed_summary_path: Path,
    rally_result_path: Path,
    prediction_path: Path,
    menu_result_path: Path,
    run_result_path: Path,
    score_path: Path,
    split_path: Path,
    raw_feature_record_path: Path,
    shots_master_path: Path,
    missed_result_path: Path,
    construction_output_path: Path,
    output_path: Path,
    source_commit: str,
) -> Path:
    """Freeze the candidate list, then open and measure saved label detail."""
    _write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA_NAME,
            "status": "running",
            "source_commit": source_commit,
        },
    )
    if missed_checker.SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a short or full Git commit")
    inputs = load_candidate_inputs(
        summary_path,
        rally_result_path,
        prediction_path,
        menu_result_path,
        run_result_path,
        score_path,
        split_path,
        raw_feature_record_path,
        shots_master_path,
    )
    construction, construction_hash = freeze_candidate_construction(
        inputs,
        source_commit,
        construction_output_path,
    )
    missed_result, rally_result = load_missed_contact_result(
        missed_summary_path,
        missed_result_path,
        inputs,
        summary_path,
        rally_result_path,
        prediction_path,
        menu_result_path,
        run_result_path,
        score_path,
        split_path,
        raw_feature_record_path,
        shots_master_path,
    )
    measurement = measure_candidate_construction(
        construction,
        missed_result,
        rally_result,
        inputs,
    )
    result = {
        "schema": RESULT_SCHEMA_NAME,
        "status": "complete",
        "source_commit": source_commit,
        "run_id": inputs.saved_run.run_id,
        "saved_missed_contacts_opened_after_candidate_list_fixed": True,
        "inputs": {
            "baseline_summary_file": Path(summary_path).name,
            "baseline_summary_sha256": _sha256(summary_path),
            "candidate_construction_file": Path(construction_output_path).name,
            "candidate_construction_sha256": construction_hash,
            "missed_contact_summary_file": Path(missed_summary_path).name,
            "missed_contact_summary_sha256": _sha256(missed_summary_path),
            "missed_contact_result_file": Path(missed_result_path).name,
            "missed_contact_result_sha256": _sha256(missed_result_path),
            "rally_result_file": Path(rally_result_path).name,
            "rally_result_sha256": _sha256(rally_result_path),
            "rally_prediction_file": Path(prediction_path).name,
            "rally_prediction_sha256": _sha256(prediction_path),
            "run_result_file": Path(run_result_path).name,
            "run_result_sha256": _sha256(run_result_path),
            "score_file": Path(score_path).name,
            "score_sha256": _sha256(score_path),
            "split_file": Path(split_path).name,
            "split_sha256": _sha256(split_path),
            "raw_feature_record_file": Path(raw_feature_record_path).name,
            "raw_feature_record_sha256": _sha256(raw_feature_record_path),
            "contact_label_file": Path(shots_master_path).name,
            "contact_label_sha256": _sha256(shots_master_path),
        },
        "measurement": measurement,
    }
    _write_json(output_path, result)
    return Path(output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--missed-summary", type=Path, required=True)
    parser.add_argument("--rally-result", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--menu-result", type=Path, required=True)
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--raw-feature-record", type=Path, required=True)
    parser.add_argument("--shots-master", type=Path, required=True)
    parser.add_argument("--missed-result", type=Path, required=True)
    parser.add_argument("--construction-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    check_rally_start_candidates(
        arguments.summary,
        arguments.missed_summary,
        arguments.rally_result,
        arguments.predictions,
        arguments.menu_result,
        arguments.run_result,
        arguments.scores,
        arguments.split,
        arguments.raw_feature_record,
        arguments.shots_master,
        arguments.missed_result,
        arguments.construction_output,
        arguments.output,
        arguments.source_commit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
