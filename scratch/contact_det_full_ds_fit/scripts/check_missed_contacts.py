"""Explain where the chosen validation contact stream misses labelled contacts."""

from __future__ import annotations

import argparse
import lzma
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    RESULT_SCHEMA,
    SCORE_DTYPE,
    _match_contacts,
    _scaled_frames,
)
from scratch.contact_det_full_ds_fit.scripts.score_validation_rallies import (
    SavedRunPredictions,
    TimingLabels,
    VerifiedRallyPredictions,
    _read_json,
    _sha256,
    _write_json,
    load_timing_labels,
    load_validation_rally_predictions,
)

CHECK_SCHEMA = "full-dataset-missed-contact-check/1"
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
TOLERANCES_AT_30_FPS = (5, 10)
KEPT_NEARBY = "kept prediction nearby"
REMOVED_NEARBY = "score reached cutoff but frame was removed near a kept prediction"
BELOW_CUTOFF = "candidate scores below cutoff"
NO_CANDIDATE = "no saved candidate nearby"
KEPT_OUTSIDE_SECTION = "kept prediction outside detected section"
EXPLANATION_ORDER = (KEPT_NEARBY, REMOVED_NEARBY, BELOW_CUTOFF, NO_CANDIDATE)
ONE_SHORT_EXPLANATION_ORDER = (
    KEPT_NEARBY,
    KEPT_OUTSIDE_SECTION,
    REMOVED_NEARBY,
    BELOW_CUTOFF,
    NO_CANDIDATE,
)


@dataclass(frozen=True)
class CheckedInputs:
    """Saved predictions and scores checked before contact labels are read."""

    summary: Mapping[str, Any]
    rally_result: Mapping[str, Any]
    run_result: Mapping[str, Any]
    score_rows: np.ndarray
    verified_predictions: VerifiedRallyPredictions
    saved_run: SavedRunPredictions


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be an object with string keys")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _selected_summary_run(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    run_id = summary.get("chosen_run_id")
    raw_runs = summary.get("runs")
    if not isinstance(run_id, str):
        raise TypeError("baseline summary chosen run must be text")
    if not isinstance(raw_runs, list):
        raise TypeError("baseline summary runs must be a list")
    matching = [
        _mapping(raw_run, "baseline summary run")
        for raw_run in raw_runs
        if isinstance(raw_run, Mapping) and raw_run.get("run_id") == run_id
    ]
    if len(matching) != 1:
        raise ValueError("baseline summary chosen run differs")
    return matching[0]


def _check_named_hash(
    record: Mapping[str, Any],
    path: Path,
    filename_field: str,
    hash_field: str,
    label: str,
) -> None:
    if record.get(filename_field) != Path(path).name:
        raise ValueError(f"{label} filename differs")
    expected_hash = record.get(hash_field)
    if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
        raise ValueError(f"{label} hash differs")


def _load_score_rows(path: Path, run_result: Mapping[str, Any]) -> np.ndarray:
    _check_named_hash(
        run_result,
        path,
        "validation_score_file",
        "validation_score_sha256",
        "validation score file",
    )
    with lzma.open(path, "rb") as score_file:
        rows = np.load(score_file, allow_pickle=False)
    if rows.ndim != 1 or rows.dtype != SCORE_DTYPE:
        raise ValueError("validation score rows differ")
    if len(rows) != _integer(
        run_result.get("validation_score_row_count"), "validation score row count"
    ):
        raise ValueError("validation score row count differs")
    if not np.isfinite(rows["contact_score"]).all() or np.any(
        (rows["contact_score"] < 0.0) | (rows["contact_score"] > 1.0)
    ):
        raise ValueError("validation contact scores differ")
    return rows


def _check_score_identities(
    rows: np.ndarray,
    verified: VerifiedRallyPredictions,
    saved_run: SavedRunPredictions,
) -> None:
    names = np.char.decode(rows["fixture"], "ascii")
    expected_names = [video.fixture for video in verified.split.validation_videos]
    if set(names.tolist()) != set(expected_names):
        raise ValueError("validation score videos differ")
    for video in verified.split.validation_videos:
        video_rows = rows[names == video.fixture]
        frames = video_rows["frame"]
        if (
            len(video_rows) == 0
            or np.any(np.diff(frames.astype(np.int64, copy=False)) <= 0)
            or np.any(video_rows["interval_id"] < 0)
            or not np.all(video_rows["fps"] == video.fps)
        ):
            raise ValueError(f"{video.fixture}: validation score identities differ")
        kept_rows = video_rows[video_rows["kept"]]
        events = saved_run.events_by_fixture[video.fixture]
        event_frames = np.asarray([event.frame for event in events], dtype=np.int32)
        event_scores = np.asarray([event.timing_score for event in events], dtype=np.float64)
        if not np.array_equal(kept_rows["frame"], event_frames) or not np.array_equal(
            kept_rows["contact_score"], event_scores
        ):
            raise ValueError(f"{video.fixture}: kept score rows differ from saved predictions")


def load_checked_inputs(
    summary_path: Path,
    rally_result_path: Path,
    prediction_path: Path,
    menu_result_path: Path,
    run_result_path: Path,
    score_path: Path,
    split_path: Path,
    raw_feature_record_path: Path,
    shots_master_path: Path,
) -> CheckedInputs:
    """Check all saved prediction inputs without parsing a contact-label row."""
    summary = _mapping(_read_json(summary_path), "baseline summary")
    _check_named_hash(summary, rally_result_path, "result_file", "result_sha256", "rally result")
    _check_named_hash(
        summary,
        prediction_path,
        "prediction_file",
        "prediction_sha256",
        "rally prediction file",
    )
    selected_summary = _selected_summary_run(summary)
    run_id = str(summary["chosen_run_id"])
    chosen_files = _mapping(summary.get("chosen_run_files"), "chosen run files")
    _check_named_hash(
        chosen_files,
        run_result_path,
        "run_result_file",
        "run_result_sha256",
        "chosen run result",
    )
    _check_named_hash(
        chosen_files,
        score_path,
        "score_file",
        "score_sha256",
        "chosen score file",
    )

    rally_result = _mapping(_read_json(rally_result_path), "rally result")
    rally_inputs = _mapping(rally_result.get("inputs"), "rally result inputs")
    if (
        rally_result.get("status") != "complete"
        or rally_result.get("source_commit") != summary.get("source_commit")
        or rally_inputs.get("rally_prediction_sha256") != summary.get("prediction_sha256")
    ):
        raise ValueError("rally result differs from the baseline summary")
    raw_rally_runs = rally_result.get("runs")
    if not isinstance(raw_rally_runs, list):
        raise TypeError("rally result runs must be a list")
    if sum(
        isinstance(raw_run, Mapping) and raw_run.get("run_id") == run_id
        for raw_run in raw_rally_runs
    ) != 1:
        raise ValueError("rally result chosen run differs")

    verified = load_validation_rally_predictions(
        prediction_path,
        menu_result_path,
        split_path,
        raw_feature_record_path,
        shots_master_path,
    )
    matching_runs = [saved_run for saved_run in verified.runs if saved_run.run_id == run_id]
    if len(matching_runs) != 1:
        raise ValueError("saved prediction chosen run differs")
    saved_run = matching_runs[0]

    run_result = _mapping(_read_json(run_result_path), "chosen baseline run result")
    if (
        run_result.get("schema") != RESULT_SCHEMA
        or run_result.get("status") != "complete"
        or run_result.get("run_id") != run_id
        or run_result.get("contact_label_sha256") != verified.payload["contact_label_sha256"]
        or run_result.get("split_sha256") != verified.payload["split_sha256"]
        or run_result.get("validation_videos")
        != [video.fixture for video in verified.split.validation_videos]
        or run_result.get("selected_score_cutoff")
        != selected_summary.get("selected_score_cutoff")
        or run_result.get("selected_duplicate_distance_at_30_fps")
        != selected_summary.get("selected_duplicate_distance_at_30_fps")
    ):
        raise ValueError("chosen baseline run result differs")
    rows = _load_score_rows(score_path, run_result)
    _check_score_identities(rows, verified, saved_run)
    return CheckedInputs(summary, rally_result, run_result, rows, verified, saved_run)


def nearby_score_summary(
    rows: np.ndarray,
    frame: int,
    tolerance_frames: int,
    cutoff: float,
    *,
    section_bounds: tuple[int, int] | None = None,
    matched_kept_frames: frozenset[int] = frozenset(),
) -> dict[str, object]:
    """Describe saved candidate rows near one missed labelled contact."""
    nearby = rows[
        np.abs(rows["frame"].astype(np.int64, copy=False) - int(frame)) <= tolerance_frames
    ]
    kept_count = int(np.count_nonzero(nearby["kept"]))
    reached_cutoff_count = int(np.count_nonzero(nearby["contact_score"] >= cutoff))
    inside_count: int | None = None
    outside_count: int | None = None
    kept_inside_count: int | None = None
    kept_outside_count: int | None = None
    if section_bounds is not None:
        start, end = section_bounds
        inside = (nearby["frame"] >= start) & (nearby["frame"] < end)
        inside_count = int(np.count_nonzero(inside))
        outside_count = len(nearby) - inside_count
        kept_inside_count = int(np.count_nonzero(nearby["kept"] & inside))
        kept_outside_count = kept_count - kept_inside_count
        if kept_inside_count:
            kept_inside_frames = {
                int(frame_value) for frame_value in nearby["frame"][nearby["kept"] & inside]
            }
            if not kept_inside_frames.issubset(matched_kept_frames):
                raise ValueError("kept prediction near a missed contact was not matched elsewhere")
            explanation = KEPT_NEARBY
        elif kept_outside_count:
            explanation = KEPT_OUTSIDE_SECTION
        elif reached_cutoff_count:
            explanation = REMOVED_NEARBY
        elif len(nearby):
            explanation = BELOW_CUTOFF
        else:
            explanation = NO_CANDIDATE
    elif kept_count:
        kept_frames = {int(frame_value) for frame_value in nearby["frame"][nearby["kept"]]}
        if not kept_frames.issubset(matched_kept_frames):
            raise ValueError("kept prediction near a missed contact was not matched elsewhere")
        explanation = KEPT_NEARBY
    elif reached_cutoff_count:
        explanation = REMOVED_NEARBY
    elif len(nearby):
        explanation = BELOW_CUTOFF
    else:
        explanation = NO_CANDIDATE

    best: np.void | None = None
    if len(nearby):
        best_index = min(
            range(len(nearby)),
            key=lambda index: (
                -float(nearby[index]["contact_score"]),
                int(nearby[index]["frame"]),
            ),
        )
        best = nearby[best_index]
    return {
        "explanation": explanation,
        "nearby_candidate_count": len(nearby),
        "nearby_kept_count": kept_count,
        "nearby_at_or_above_cutoff_count": reached_cutoff_count,
        "nearby_inside_section_count": inside_count,
        "nearby_outside_section_count": outside_count,
        "nearby_kept_inside_section_count": kept_inside_count,
        "nearby_kept_outside_section_count": kept_outside_count,
        "best_candidate_frame": None if best is None else int(best["frame"]),
        "best_candidate_score": None if best is None else float(best["contact_score"]),
        "best_candidate_frame_offset": (
            None if best is None else int(best["frame"]) - int(frame)
        ),
        "best_candidate_absolute_frame_distance": (
            None if best is None else abs(int(best["frame"]) - int(frame))
        ),
        "best_candidate_kept": None if best is None else bool(best["kept"]),
    }


def _contact_identities(timing: TimingLabels) -> dict[tuple[str, int], tuple[str, int]]:
    identities: dict[tuple[str, int], tuple[str, int]] = {}
    for fixture, rallies in timing.rallies.items():
        for rally in rallies:
            for contact_index, frame in enumerate(rally.frames):
                identity = (fixture, int(frame))
                if identity in identities:
                    raise ValueError(f"{fixture}/{frame}: contact frame is repeated")
                identities[identity] = (rally.rally_id, contact_index)
    return identities


def _explanation_counts(
    details: Sequence[Mapping[str, object]], order: Sequence[str]
) -> dict[str, int]:
    counts = Counter(str(detail["explanation"]) for detail in details)
    if set(counts) - set(order):
        raise ValueError("missed-contact explanation differs")
    return {explanation: counts[explanation] for explanation in order}


def _contact_type_counts(
    details: Sequence[Mapping[str, object]],
    *,
    contact_type: str,
    labelled: int,
) -> dict[str, object]:
    selected = [detail for detail in details if detail["contact_type"] == contact_type]
    return {
        "labelled_contacts": labelled,
        "matched_contacts": labelled - len(selected),
        "missed_contacts": len(selected),
        "explanations": _explanation_counts(selected, EXPLANATION_ORDER),
    }


def missed_contact_details(
    checked: CheckedInputs,
    timing: TimingLabels,
    tolerance_at_30_fps: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return every globally unmatched contact and the checked totals."""
    score_names = np.char.decode(checked.score_rows["fixture"], "ascii")
    identities = _contact_identities(timing)
    cutoff = float(checked.run_result["selected_score_cutoff"])
    details: list[dict[str, object]] = []
    matched_total = 0
    matched_first = 0
    for video in checked.verified_predictions.split.validation_videos:
        fixture = video.fixture
        expected = timing.frames[fixture]
        events = checked.saved_run.events_by_fixture[fixture]
        predicted = np.asarray([event.frame for event in events], dtype=np.int32)
        tolerance = _scaled_frames(tolerance_at_30_fps, video.fps)
        matches = _match_contacts(expected, predicted, tolerance)
        matched_indexes = {contact_index for contact_index, _, _ in matches}
        matched_prediction_frames = frozenset(
            int(predicted[prediction_index]) for _, prediction_index, _ in matches
        )
        matched_total += len(matches)
        matched_first += sum(
            int(expected[contact_index]) in timing.first_contacts[fixture]
            for contact_index in matched_indexes
        )
        video_scores = checked.score_rows[score_names == fixture]
        for contact_index, raw_frame in enumerate(expected):
            if contact_index in matched_indexes:
                continue
            frame = int(raw_frame)
            rally_id, rally_contact_index = identities[(fixture, frame)]
            contact_type = "first" if rally_contact_index == 0 else "later"
            nearby = nearby_score_summary(
                video_scores,
                frame,
                tolerance,
                cutoff,
                matched_kept_frames=matched_prediction_frames,
            )
            details.append(
                {
                    "fixture": fixture,
                    "rally_id": rally_id,
                    "rally_contact_index": rally_contact_index,
                    "frame": frame,
                    "contact_type": contact_type,
                    "tolerance_frames": tolerance,
                    **nearby,
                }
            )
    saved_run_result = next(
        _mapping(raw_run, "rally result run")
        for raw_run in checked.rally_result["runs"]
        if isinstance(raw_run, Mapping) and raw_run.get("run_id") == checked.saved_run.run_id
    )
    saved_timing = _mapping(
        _mapping(saved_run_result.get("contact_timing"), "saved contact timing").get(
            str(tolerance_at_30_fps)
        ),
        "saved timing tolerance",
    )
    if matched_total != saved_timing.get("matched") or matched_first != saved_timing.get(
        "first_contact_matched"
    ):
        raise ValueError("recalculated timing matches differ from the saved result")
    first_count = sum(len(first) for first in timing.first_contacts.values())
    all_count = sum(len(frames) for frames in timing.frames.values())
    summary = {
        "tolerance_at_30_fps": tolerance_at_30_fps,
        "first_contacts": _contact_type_counts(
            details,
            contact_type="first",
            labelled=first_count,
        ),
        "later_contacts": _contact_type_counts(
            details,
            contact_type="later",
            labelled=all_count - first_count,
        ),
    }
    return details, summary


def _chosen_rally_result(checked: CheckedInputs) -> Mapping[str, Any]:
    return next(
        _mapping(raw_run, "chosen rally result")
        for raw_run in checked.rally_result["runs"]
        if isinstance(raw_run, Mapping) and raw_run.get("run_id") == checked.saved_run.run_id
    )


def _is_otherwise_correct_one_short(span: Mapping[str, Any]) -> bool:
    return bool(
        span.get("rally_id") is not None
        and span.get("event_count", 0) > 0
        and span.get("ground_truth_contacts") == span.get("timing_matches", -1) + 1
        and span.get("event_count") == span.get("timing_matches")
        and span.get("correct_side_answers") == span.get("timing_matches")
        and span.get("side_answerable") is True
        and "extra_event" not in span.get("rejection_reasons", ())
    )


def one_short_section_details(
    checked: CheckedInputs,
    timing: TimingLabels,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Find the one missing label in every otherwise-correct one-short section."""
    chosen_result = _chosen_rally_result(checked)
    primary = _mapping(chosen_result.get("primary"), "chosen primary rally result")
    raw_spans = primary.get("spans")
    if not isinstance(raw_spans, list):
        raise TypeError("chosen primary rally spans must be a list")
    selected_spans = [
        _mapping(raw_span, "chosen primary rally span")
        for raw_span in raw_spans
        if isinstance(raw_span, Mapping) and _is_otherwise_correct_one_short(raw_span)
    ]
    expected_count = _mapping(
        checked.summary.get("chosen_run_failed_single_rally_sections"),
        "chosen run failure summary",
    ).get("exactly_one_contact_missing_with_remaining_times_and_sides_correct")
    if len(selected_spans) != expected_count:
        raise ValueError("otherwise-correct one-short section count differs")

    rallies = {
        (fixture, rally.rally_id): rally
        for fixture, fixture_rallies in timing.rallies.items()
        for rally in fixture_rallies
    }
    videos = {
        video.fixture: video for video in checked.verified_predictions.split.validation_videos
    }
    score_names = np.char.decode(checked.score_rows["fixture"], "ascii")
    cutoff = float(checked.run_result["selected_score_cutoff"])
    details: list[dict[str, object]] = []
    for span in selected_spans:
        fixture = str(span["fixture"])
        start = int(span["start_frame"])
        end = int(span["end_frame"])
        rally = rallies[(fixture, str(span["rally_id"]))]
        events = [
            event
            for event in checked.saved_run.events_by_fixture[fixture]
            if start <= event.frame < end
        ]
        event_frames = np.asarray([event.frame for event in events], dtype=np.int32)
        tolerance = _scaled_frames(10, videos[fixture].fps)
        matches = _match_contacts(np.asarray(rally.frames, dtype=np.int32), event_frames, tolerance)
        matched_indexes = {contact_index for contact_index, _, _ in matches}
        matched_event_frames = frozenset(
            int(event_frames[event_index]) for _, event_index, _ in matches
        )
        missing_indexes = [
            contact_index
            for contact_index in range(len(rally.frames))
            if contact_index not in matched_indexes
        ]
        if (
            len(matches) != span["timing_matches"]
            or len(events) != span["event_count"]
            or len(missing_indexes) != 1
        ):
            raise ValueError(f"{fixture} span {span['span_id']}: one-short timing differs")
        missing_index = missing_indexes[0]
        frame = int(rally.frames[missing_index])
        video_scores = checked.score_rows[score_names == fixture]
        nearby = nearby_score_summary(
            video_scores,
            frame,
            tolerance,
            cutoff,
            section_bounds=(start, end),
            matched_kept_frames=matched_event_frames,
        )
        details.append(
            {
                "fixture": fixture,
                "span_id": int(span["span_id"]),
                "rally_id": rally.rally_id,
                "section_start_frame": start,
                "section_end_frame": end,
                "missing_frame": frame,
                "missing_contact_type": "first" if missing_index == 0 else "later",
                "missing_rally_contact_index": missing_index,
                "tolerance_frames": tolerance,
                **nearby,
            }
        )

    first = [detail for detail in details if detail["missing_contact_type"] == "first"]
    later = [detail for detail in details if detail["missing_contact_type"] == "later"]

    def counts(selected: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return {
            "section_count": len(selected),
            "explanations": _explanation_counts(selected, ONE_SHORT_EXPLANATION_ORDER),
            "candidate_inside_section": sum(
                int(detail["nearby_inside_section_count"]) > 0 for detail in selected
            ),
            "candidate_only_outside_section": sum(
                int(detail["nearby_candidate_count"]) > 0
                and int(detail["nearby_inside_section_count"]) == 0
                for detail in selected
            ),
            "no_nearby_candidate": sum(
                int(detail["nearby_candidate_count"]) == 0 for detail in selected
            ),
        }

    return details, {
        "tolerance_at_30_fps": 10,
        "section_count": len(details),
        "missing_first_contact": counts(first),
        "missing_later_contact": counts(later),
    }


def check_missed_contacts(
    summary_path: Path,
    rally_result_path: Path,
    prediction_path: Path,
    menu_result_path: Path,
    run_result_path: Path,
    score_path: Path,
    split_path: Path,
    raw_feature_record_path: Path,
    shots_master_path: Path,
    output_path: Path,
    source_commit: str,
) -> Path:
    """Check saved inputs, then load timing labels and explain every miss."""
    _write_json(
        output_path,
        {"schema": CHECK_SCHEMA, "status": "running", "source_commit": source_commit},
    )
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a short or full Git commit")
    checked = load_checked_inputs(
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
    import pandas as pd

    timing = load_timing_labels(shots_master_path, checked.verified_predictions.split, pd.read_csv)
    if _sha256(shots_master_path) != checked.verified_predictions.payload["contact_label_sha256"]:
        raise ValueError("contact label file changed during the timing-label read")

    details_by_tolerance: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, dict[str, object]] = {}
    for tolerance in TOLERANCES_AT_30_FPS:
        details, summary = missed_contact_details(checked, timing, tolerance)
        details_by_tolerance[str(tolerance)] = details
        summaries[str(tolerance)] = summary
    one_short_details, one_short_summary = one_short_section_details(checked, timing)
    result = {
        "schema": CHECK_SCHEMA,
        "status": "complete",
        "source_commit": source_commit,
        "labels_read_after_predictions_checked": True,
        "run_id": checked.saved_run.run_id,
        "selected_score_cutoff": checked.run_result["selected_score_cutoff"],
        "selected_duplicate_distance_at_30_fps": checked.run_result[
            "selected_duplicate_distance_at_30_fps"
        ],
        "inputs": {
            "baseline_summary_file": Path(summary_path).name,
            "baseline_summary_sha256": _sha256(summary_path),
            "rally_result_file": Path(rally_result_path).name,
            "rally_result_sha256": _sha256(rally_result_path),
            "rally_prediction_file": Path(prediction_path).name,
            "rally_prediction_sha256": _sha256(prediction_path),
            "menu_result_file": Path(menu_result_path).name,
            "menu_result_sha256": _sha256(menu_result_path),
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
        "missed_contacts": summaries,
        "otherwise_correct_one_short_sections": one_short_summary,
        "details": {
            "missed_contacts": details_by_tolerance,
            "otherwise_correct_one_short_sections": one_short_details,
        },
    }
    _write_json(output_path, result)
    return Path(output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--rally-result", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--menu-result", type=Path, required=True)
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--raw-feature-record", type=Path, required=True)
    parser.add_argument("--shots-master", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    output = check_missed_contacts(
        arguments.summary,
        arguments.rally_result,
        arguments.predictions,
        arguments.menu_result,
        arguments.run_result,
        arguments.scores,
        arguments.split,
        arguments.raw_feature_record,
        arguments.shots_master,
        arguments.output,
        arguments.source_commit,
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
