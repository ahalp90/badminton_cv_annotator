"""Scores shared by the multiscale VLM experiment and its final CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Any

from annotator.calibration.gt_scoring import RallyRow, VideoScoring

from .multiscale_schema import (
    BroadContent,
    MultiscaleCase,
    load_manifest,
    parse_broad_reply,
    target_route,
)

SCORE_SCHEMA = "vlm-multiscale-score-v1"
BROAD_SCORE_SCHEMA = "vlm-multiscale-broad-score-v1"
ATTEMPT_SCHEMA = "vlm-multiscale-attempt-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected an object in {path}")
    return payload


def _truth_content(value: str) -> BroadContent:
    if value in {"live", "live-non-standard"}:
        return BroadContent.LIVE
    return BroadContent(value)


def _deterministic_content(case: MultiscaleCase) -> BroadContent:
    excluded_fraction = float(case.pipeline_priors["definitive_mask_fraction"])
    return BroadContent.OTHER if excluded_fraction > 0 else BroadContent.LIVE


def _overlap_length(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def _case_scene_score(
    case: MultiscaleCase,
    truth: dict[str, Any],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    target_frames = case.target_end_frame - case.target_start_frame
    parsed = None
    if attempt["parsed_response"] is not None:
        raw_reply = json.dumps({"segments": attempt["parsed_response"]})
        parsed = parse_broad_reply(case, raw_reply)
    by_id = {} if parsed is None else {reply.segment_id: reply for reply in parsed}
    exact_correct = 0
    binary_correct = 0
    deterministic_correct = 0
    truth_is_live_only = True
    deterministic = _deterministic_content(case)
    for truth_interval in truth["truth_intervals"]:
        truth_content = _truth_content(str(truth_interval["truth"]))
        if truth_content is not BroadContent.LIVE:
            truth_is_live_only = False
        interval_start = int(truth_interval["source_start_frame"])
        interval_end = int(truth_interval["source_end_frame"])
        truth_frames = interval_end - interval_start
        if (deterministic is BroadContent.LIVE) == (truth_content is BroadContent.LIVE):
            deterministic_correct += truth_frames
        for segment in case.segments:
            overlap = _overlap_length(
                interval_start,
                interval_end,
                segment.source_start_frame,
                segment.source_end_frame,
            )
            if not overlap or segment.segment_id not in case.target_segment_ids:
                continue
            predicted = None if parsed is None else by_id[segment.segment_id].content
            if predicted == truth_content:
                exact_correct += overlap
            if (
                predicted is not None
                and predicted is not BroadContent.UNCLEAR
                and (predicted is BroadContent.LIVE) == (truth_content is BroadContent.LIVE)
            ):
                binary_correct += overlap
    predicted_route = target_route(case, parsed).value
    truth_route = "routine_live" if truth_is_live_only else "close_check"
    deterministic_route = (
        "routine_live" if deterministic is BroadContent.LIVE else "close_check"
    )
    return {
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "video_id": case.video_id,
        "context_seconds": case.context_seconds,
        "valid_reply": parsed is not None,
        "target_frames": target_frames,
        "exact_scene_correct_frames": exact_correct,
        "exact_scene_accuracy": exact_correct / target_frames,
        "binary_scene_correct_frames": binary_correct,
        "binary_scene_accuracy": binary_correct / target_frames,
        "deterministic_binary_correct_frames": deterministic_correct,
        "deterministic_binary_accuracy": deterministic_correct / target_frames,
        "predicted_route": predicted_route,
        "deterministic_route": deterministic_route,
        "truth_route": truth_route,
        "route_correct": predicted_route == truth_route,
        "parser_error": attempt["parser_error"],
        "generation_error": attempt["generation_error"],
    }


def _aggregate_case_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_frames = sum(int(row["target_frames"]) for row in rows)
    truth_close = [row for row in rows if row["truth_route"] == "close_check"]
    truth_routine = [row for row in rows if row["truth_route"] == "routine_live"]
    predicted_close = [row for row in rows if row["predicted_route"] == "close_check"]
    predicted_routine = [row for row in rows if row["predicted_route"] == "routine_live"]
    return {
        "cases": len(rows),
        "valid_replies": sum(bool(row["valid_reply"]) for row in rows),
        "mean_case_exact_scene_accuracy": mean(float(row["exact_scene_accuracy"]) for row in rows),
        "mean_case_binary_scene_accuracy": mean(float(row["binary_scene_accuracy"]) for row in rows),
        "mean_case_deterministic_binary_accuracy": mean(
            float(row["deterministic_binary_accuracy"]) for row in rows
        ),
        "target_frame_exact_scene_accuracy": sum(
            int(row["exact_scene_correct_frames"]) for row in rows
        )
        / target_frames,
        "target_frame_binary_scene_accuracy": sum(
            int(row["binary_scene_correct_frames"]) for row in rows
        )
        / target_frames,
        "target_frame_deterministic_binary_accuracy": sum(
            int(row["deterministic_binary_correct_frames"]) for row in rows
        )
        / target_frames,
        "route_accuracy": sum(bool(row["route_correct"]) for row in rows) / len(rows),
        "close_check_recall": (
            sum(row["predicted_route"] == "close_check" for row in truth_close)
            / len(truth_close)
            if truth_close
            else None
        ),
        "routine_live_recall": (
            sum(row["predicted_route"] == "routine_live" for row in truth_routine)
            / len(truth_routine)
            if truth_routine
            else None
        ),
        "routine_live_precision": (
            sum(row["truth_route"] == "routine_live" for row in predicted_routine)
            / len(predicted_routine)
            if predicted_routine
            else None
        ),
        "close_check_precision": (
            sum(row["truth_route"] == "close_check" for row in predicted_close)
            / len(predicted_close)
            if predicted_close
            else None
        ),
        "deterministic_close_check_recall": (
            sum(row["deterministic_route"] == "close_check" for row in truth_close)
            / len(truth_close)
            if truth_close
            else None
        ),
    }


def _load_broad_attempt(
    attempts_root: Path,
    backend: str,
    case: MultiscaleCase,
    manifest_sha256: str,
) -> dict[str, Any]:
    path = attempts_root / backend / f"{case.case_id}.json"
    attempt = _load_json(path)
    if attempt.get("schema") != ATTEMPT_SCHEMA:
        raise ValueError(f"{path}: unexpected attempt schema")
    if attempt.get("backend") != backend or attempt.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"{path}: backend or manifest identity differs")
    if attempt.get("case", {}).get("case_id") != case.case_id:
        raise ValueError(f"{path}: case identity differs")
    parsed = attempt.get("parsed_response")
    parser_error = attempt.get("parser_error")
    if (parsed is None) == (parser_error is None) and attempt.get("generation_error") is None:
        raise ValueError(f"{path}: parsed response and parser error are inconsistent")
    return attempt


def score_broad_attempts(
    manifest_path: Path,
    truth_path: Path,
    attempts_root: Path,
    backend: str,
) -> dict[str, Any]:
    """Score paired broad replies only inside each proposed rally span."""
    cases = load_manifest(manifest_path)
    truth_payload = _load_json(truth_path)
    if truth_payload.get("schema") != "vlm-multiscale-truth-v1":
        raise ValueError("unexpected multiscale truth schema")
    truth_by_pair = {str(row["pair_id"]): row for row in truth_payload["cases"]}
    if len(truth_by_pair) * 2 != len(cases):
        raise ValueError("truth sidecar and paired manifest have different case counts")
    manifest_sha256 = _sha256(manifest_path)
    rows = []
    for case in cases:
        truth = truth_by_pair.get(case.pair_id)
        if truth is None:
            raise ValueError(f"missing truth for {case.pair_id}")
        attempt = _load_broad_attempt(attempts_root, backend, case, manifest_sha256)
        rows.append(_case_scene_score(case, truth, attempt))

    by_duration = {
        str(seconds): _aggregate_case_scores(
            [row for row in rows if row["context_seconds"] == seconds]
        )
        for seconds in (90, 120)
    }
    by_video_and_duration = {}
    for video_id in sorted({str(row["video_id"]) for row in rows}):
        by_video_and_duration[video_id] = {
            str(seconds): _aggregate_case_scores(
                [
                    row
                    for row in rows
                    if row["video_id"] == video_id and row["context_seconds"] == seconds
                ]
            )
            for seconds in (90, 120)
        }
    accuracy_90 = float(by_duration["90"]["mean_case_exact_scene_accuracy"])
    accuracy_120 = float(by_duration["120"]["mean_case_exact_scene_accuracy"])
    selected_duration = 120 if accuracy_120 > accuracy_90 else 90
    selected = by_duration[str(selected_duration)]
    broad_gate = (
        int(selected["valid_replies"]) >= len(truth_by_pair) - 1
        and float(selected["mean_case_binary_scene_accuracy"])
        > float(selected["mean_case_deterministic_binary_accuracy"])
    )
    close_check_recall = selected["close_check_recall"]
    routine_live_recall = selected["routine_live_recall"]
    safe_bypass_gate = (
        broad_gate
        and close_check_recall is not None
        and float(close_check_recall) >= 0.8
        and routine_live_recall is not None
        and float(routine_live_recall) >= 0.5
    )
    return {
        "schema": BROAD_SCORE_SCHEMA,
        "backend": backend,
        "manifest_sha256": manifest_sha256,
        "truth_sha256": _sha256(truth_path),
        "selection_metric": "mean_case_exact_scene_accuracy; 90 seconds wins a tie",
        "selected_duration_seconds": selected_duration,
        "broad_gate_passed": broad_gate,
        "safe_bypass_gate_passed": safe_bypass_gate,
        "by_duration": by_duration,
        "by_video_and_duration": by_video_and_duration,
        "cases": rows,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _row_is_complete(row: RallyRow) -> bool:
    """Return whether one mapped truth rally has a usable complete record."""
    # The annotator fits one of two strictly alternating phases. Under that
    # invariant, an exact count and correct server fix every contact's actor.
    return (
        row.classification == "covered"
        and row.ball_round_correct
        and row.timing_matched_n == row.n_gt_strokes
        and row.server_correct
        and row.player_correct
        and row.getpoint_eligible
        and row.getpoint_correct is True
    )


def score_prediction_side(
    scoring: VideoScoring,
    *,
    predicted_span_count: int,
    retained_span_ids: Iterable[int],
    baseline_correct_span_ids: Iterable[int] = (),
) -> dict[str, Any]:
    """Score every retained predicted span, including false and merged spans."""
    retained_list = list(retained_span_ids)
    baseline_correct_list = list(baseline_correct_span_ids)
    if len(set(retained_list)) != len(retained_list):
        raise ValueError("retained span IDs must be unique")
    if len(set(baseline_correct_list)) != len(baseline_correct_list):
        raise ValueError("baseline-correct span IDs must be unique")
    retained = set(retained_list)
    baseline_correct = set(baseline_correct_list)
    valid_ids = set(range(predicted_span_count))
    unknown_retained = retained - valid_ids
    unknown_baseline = baseline_correct - valid_ids
    if unknown_retained:
        raise ValueError(f"retained span IDs are out of range: {sorted(unknown_retained)}")
    if unknown_baseline:
        raise ValueError(f"baseline-correct span IDs are out of range: {sorted(unknown_baseline)}")

    rows_by_span: dict[int, list[RallyRow]] = defaultdict(list)
    for row in scoring.rows:
        if row.mapped_span is not None:
            rows_by_span[row.mapped_span].append(row)

    records: list[dict[str, Any]] = []
    correct_span_ids: set[int] = set()
    for span_id in sorted(retained):
        rows = rows_by_span[span_id]
        if not rows:
            status = "spurious_or_partial"
            truth_rally = None
        elif len(rows) > 1:
            status = "merged"
            truth_rally = None
        else:
            row = rows[0]
            truth_rally = {"set_id": row.set_id, "rally": row.rally}
            if _row_is_complete(row):
                status = "correct"
                correct_span_ids.add(span_id)
            else:
                status = "incorrect_record"
        records.append(
            {
                "span_id": span_id,
                "status": status,
                "truth_rally": truth_rally,
            }
        )

    correct = len(correct_span_ids)
    retained_total = len(retained)
    baseline_total = len(baseline_correct)
    baseline_still_usable = len(baseline_correct & retained & correct_span_ids)
    return {
        "schema": SCORE_SCHEMA,
        "video_id": scoring.name,
        "retained_records": retained_total,
        "correct_complete_records": correct,
        "complete_record_precision": correct / retained_total if retained_total else None,
        "baseline_correct_records": baseline_total,
        "baseline_correct_still_usable": baseline_still_usable,
        "baseline_correct_coverage": (
            baseline_still_usable / baseline_total if baseline_total else None
        ),
        "passes_half_baseline_gate": (
            baseline_still_usable * 2 >= baseline_total if baseline_total else None
        ),
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--backend", choices=("qwen3-vl", "internvideo3"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    score = score_broad_attempts(
        args.manifest,
        args.truth,
        args.attempts,
        args.backend,
    )
    _write_json(args.out, score)


if __name__ == "__main__":
    main()
