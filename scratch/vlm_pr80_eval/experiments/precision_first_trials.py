"""Evaluate a frozen precision-first rally filter without label leakage."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from annotator.calibration.fixtures import FIXTURES, Fixture
from annotator.calibration.gt_scoring import RallyRow, load_gt_tables, score_video
from annotator.fps_constants import ScalingKind
from annotator.inpaint_guard import NO_FLAG
from annotator.run_video import AnnotatorResult
from dataset_builder._runtime_support import _annotation_result
from dataset_builder.vision import load_npy_xz

from .score_multiscale_trials import score_prediction_side

FEATURE_SCHEMA = "vlm-precision-first-features/0.1"
SCORE_SCHEMA = "vlm-precision-first-score/0.1"
VIDEO_SCORE_SCHEMA = "vlm-precision-first-video-score/0.1"
VIDEO_IDS = ("sset_01", "sset_15", "sset_21")
RESULT_HASHES = {
    "sset_01": "b1656fcd60ba354b1003ee6f70e8192395d4f4b479b5f8245e3d98518ae1d52e",
    "sset_15": "4c4b51b216363444b0117f07e7646054c9ba20fb2f9639b78430103285e8ec04",
    "sset_21": "b902c8766ee8cff20768c0eea4da4371d7aef8982e0dcf0b4f871522611b9f51",
}
RULE_LADDER = (
    "automatic-completeness",
    "scene-support",
    "track-support",
    "outcome-corroboration",
)
PRIMARY_TOLERANCE_BASE30 = 5
SENSITIVITY_TOLERANCES_BASE30 = (10, 15)
SUPPORT_FRACTION = 0.8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _load_annotation(path: Path, video_id: str) -> AnnotatorResult:
    observed_hash = _sha256(path)
    if observed_hash != RESULT_HASHES[video_id]:
        raise ValueError(f"{video_id}: annotator result hash differs")
    payload = _load_json_gz(path)
    if set(payload) != {"schema", "video_id", "result"}:
        raise ValueError(f"{video_id}: annotator result fields differ")
    if payload["schema"] != "annotator-result/0.1" or payload["video_id"] != video_id:
        raise ValueError(f"{video_id}: annotator result identity differs")
    return _annotation_result(payload["result"])


def _load_vector(path: Path, frame_count: int, dtype: np.dtype[Any]) -> np.ndarray:
    values = load_npy_xz(path)
    if values.shape != (frame_count,) or values.dtype != dtype:
        raise ValueError(
            f"{path.name}: expected shape ({frame_count},) and dtype {dtype}, "
            f"got {values.shape} and {values.dtype}"
        )
    return values


def _fraction(values: np.ndarray, start: int, end: int) -> float:
    if not 0 <= start < end <= len(values):
        raise ValueError(f"invalid span [{start}, {end}) for {len(values)} frames")
    return float(np.mean(values[start:end]))


def _contact_evidence_complete(
    result: AnnotatorResult,
    span_id: int,
    track_visible: np.ndarray,
    guard_codes: np.ndarray,
    definitive_exclusion: np.ndarray,
) -> bool:
    contacts = [row for row in result.filtered_contacts if row.rally_id == span_id]
    return bool(contacts) and all(
        row.wrist_near is True
        and row.suppressed is False
        and bool(track_visible[row.contact_frame])
        and int(guard_codes[row.contact_frame]) == NO_FLAG
        and not bool(definitive_exclusion[row.contact_frame])
        for row in contacts
    )


def _record_features(
    result: AnnotatorResult,
    span_id: int,
    *,
    court_present: np.ndarray,
    keep_vote: np.ndarray,
    track_visible: np.ndarray,
    guard_codes: np.ndarray,
    definitive_exclusion: np.ndarray,
) -> dict[str, object]:
    start, end = result.spans[span_id]
    verdict = result.verdict_rows.get(span_id)
    geometric = result.geometric_verdict_rows.get(span_id)
    accepted_contact_count = len(result.filtered_by_rally.get(span_id, []))
    automatic_record_complete = (
        result.fitted_first_all[span_id] is not None
        and result.striker_halves[span_id] is not None
        and verdict is not None
        and verdict.verdict is not None
        and _contact_evidence_complete(
            result,
            span_id,
            track_visible,
            guard_codes,
            definitive_exclusion,
        )
    )
    return {
        "span_id": span_id,
        "start_frame": start,
        "end_frame": end,
        "accepted_contact_count": accepted_contact_count,
        "automatic_record_complete": automatic_record_complete,
        "court_present_fraction": _fraction(court_present, start, end),
        "court_keep_fraction": _fraction(keep_vote, start, end),
        "track_visible_fraction": _fraction(track_visible, start, end),
        "outcome_corroborated": bool(
            verdict is not None
            and verdict.verdict_source is not None
            and verdict.verdict_source.value == "next_server"
            and geometric is not None
            and geometric.agreement is True
        ),
    }


def build_features(artifacts_root: Path) -> dict[str, object]:
    """Build the label-free table consumed by the held-out scorer."""
    videos: dict[str, object] = {}
    stages = artifacts_root / "stages"
    for video_id in VIDEO_IDS:
        annotation_root = stages / "annotation" / video_id
        result_path = annotation_root / "annotator_result.json.gz"
        result = _load_annotation(result_path, video_id)
        shuttle_root = stages / "shuttle" / video_id
        track_path = shuttle_root / "shuttle_track.npy.xz"
        guard_path = shuttle_root / "shuttle_guard_codes.npy.xz"
        track = load_npy_xz(track_path)
        if track.ndim != 2 or track.shape[1] != 3:
            raise ValueError(f"{video_id}: shuttle track must have shape (frames, 3)")
        frame_count = len(track)
        guard_codes = _load_vector(guard_path, frame_count, np.dtype(np.uint8))
        court_root = stages / "court" / video_id
        court_path = court_root / "court_present.npy.xz"
        keep_path = court_root / "court_keep_vote.npy.xz"
        definitive_path = annotation_root / "definitive_exclusion_mask.npy.xz"
        court_present = _load_vector(court_path, frame_count, np.dtype(np.bool_))
        keep_vote = _load_vector(keep_path, frame_count, np.dtype(np.bool_))
        definitive_exclusion = _load_vector(
            definitive_path, frame_count, np.dtype(np.bool_)
        )
        track_visible = track[:, 2] > 0
        records = [
            _record_features(
                result,
                span_id,
                court_present=court_present,
                keep_vote=keep_vote,
                track_visible=track_visible,
                guard_codes=guard_codes,
                definitive_exclusion=definitive_exclusion,
            )
            for span_id in range(len(result.spans))
        ]
        videos[video_id] = {
            "predicted_span_count": len(result.spans),
            "sources": {
                "annotator_result": _sha256(result_path),
                "court_present": _sha256(court_path),
                "court_keep_vote": _sha256(keep_path),
                "shuttle_track": _sha256(track_path),
                "shuttle_guard_codes": _sha256(guard_path),
                "definitive_exclusion_mask": _sha256(definitive_path),
            },
            "records": records,
        }
    return {
        "schema": FEATURE_SCHEMA,
        "contains_ground_truth": False,
        "support_fraction": SUPPORT_FRACTION,
        "rule_ladder": list(RULE_LADDER),
        "videos": videos,
    }


def retained_span_ids(records: Sequence[Mapping[str, object]], rung: str) -> list[int]:
    """Apply one rung of the frozen monotone ladder."""
    if rung not in RULE_LADDER:
        raise ValueError(f"unknown rule rung {rung!r}")
    rung_index = RULE_LADDER.index(rung)
    retained: list[int] = []
    for record in records:
        keep = bool(record["automatic_record_complete"])
        if rung_index >= 1:
            keep = (
                keep
                and float(record["court_present_fraction"]) >= SUPPORT_FRACTION
                and float(record["court_keep_fraction"]) >= SUPPORT_FRACTION
            )
        if rung_index >= 2:
            keep = keep and float(record["track_visible_fraction"]) >= SUPPORT_FRACTION
        if rung_index >= 3:
            keep = keep and bool(record["outcome_corroborated"])
        if keep:
            retained.append(int(record["span_id"]))
    return retained


def _failed_fields(row: RallyRow) -> list[str]:
    failures: list[str] = []
    if row.classification != "covered":
        failures.append("rally_boundary")
    if not row.ball_round_correct:
        failures.append("exact_contact_count")
    if row.timing_matched_n != row.n_gt_strokes:
        failures.append("contact_timing")
    if not row.player_correct:
        failures.append("player_attribution")
    if not row.server_correct:
        failures.append("server")
    if not (row.getpoint_eligible and row.getpoint_correct is True):
        failures.append("point_outcome")
    return failures


def _score_spans(
    fixture: Fixture,
    result: AnnotatorResult,
    retained: Iterable[int],
    tolerance_base30: int,
) -> dict[str, object]:
    master, _homography, courts, _resolution = load_gt_tables()
    native_tolerance = int(
        ScalingKind.FRAME_COUNT.scale(tolerance_base30, fixture.fps)
    )
    scoring = score_video(fixture, result, master, courts, native_tolerance)
    rows_by_span: dict[int, list[RallyRow]] = defaultdict(list)
    for row in scoring.rows:
        if row.mapped_span is not None:
            rows_by_span[row.mapped_span].append(row)
    baseline_correct = [
        span_id
        for span_id, rows in rows_by_span.items()
        if len(rows) == 1 and not _failed_fields(rows[0])
    ]
    summary = score_prediction_side(
        scoring,
        predicted_span_count=len(result.spans),
        retained_span_ids=retained,
        baseline_correct_span_ids=baseline_correct,
    )
    summary["schema"] = VIDEO_SCORE_SCHEMA
    for record in summary["records"]:
        rows = rows_by_span[record["span_id"]]
        if record["status"] == "incorrect_record":
            record["failed_fields"] = _failed_fields(rows[0])
        elif record["status"] == "merged":
            record["failed_fields"] = ["rally_boundary_merged"]
        elif record["status"] == "spurious_or_partial":
            record["failed_fields"] = ["rally_boundary_spurious_or_partial"]
        else:
            record["failed_fields"] = []
    summary["tolerance_base30_frames"] = tolerance_base30
    summary["native_frame_tolerance"] = native_tolerance
    return summary


def _development_counts(
    scored: Mapping[str, Mapping[str, Mapping[str, object]]],
    development_videos: Sequence[str],
    rung: str,
) -> tuple[int, int]:
    retained = sum(
        int(scored[video_id][rung]["retained_records"])
        for video_id in development_videos
    )
    correct = sum(
        int(scored[video_id][rung]["correct_complete_records"])
        for video_id in development_videos
    )
    return retained, retained - correct


def choose_rung(
    scored: Mapping[str, Mapping[str, Mapping[str, object]]],
    development_videos: Sequence[str],
) -> str | None:
    """Choose the highest-coverage zero-error rung, preferring stricter ties."""
    candidates: list[tuple[int, int, str]] = []
    for rung_index, rung in enumerate(RULE_LADDER):
        retained, errors = _development_counts(scored, development_videos, rung)
        if retained and errors == 0:
            candidates.append((retained, rung_index, rung))
    if not candidates:
        return None
    return max(candidates)[2]


def score_features(features_path: Path, artifacts_root: Path) -> dict[str, object]:
    """Open ground truth only after the automatic feature table is frozen."""
    with features_path.open(encoding="utf-8") as stream:
        features = json.load(stream)
    if not isinstance(features, dict) or features.get("schema") != FEATURE_SCHEMA:
        raise ValueError("feature table schema differs")
    if features.get("contains_ground_truth") is not False:
        raise ValueError("feature table must be explicitly truth-free")
    fixture_by_name = {fixture.name: fixture for fixture in FIXTURES}
    results = {
        video_id: _load_annotation(
            artifacts_root
            / "stages"
            / "annotation"
            / video_id
            / "annotator_result.json.gz",
            video_id,
        )
        for video_id in VIDEO_IDS
    }
    primary_by_video: dict[str, dict[str, dict[str, object]]] = {}
    for video_id in VIDEO_IDS:
        records = features["videos"][video_id]["records"]
        primary_by_video[video_id] = {
            rung: _score_spans(
                fixture_by_name[video_id],
                results[video_id],
                retained_span_ids(records, rung),
                PRIMARY_TOLERANCE_BASE30,
            )
            for rung in RULE_LADDER
        }

    folds: list[dict[str, object]] = []
    for held_out in VIDEO_IDS:
        development = [video_id for video_id in VIDEO_IDS if video_id != held_out]
        chosen = choose_rung(primary_by_video, development)
        development_summary = {
            rung: {
                "retained_records": _development_counts(
                    primary_by_video, development, rung
                )[0],
                "errors": _development_counts(primary_by_video, development, rung)[1],
            }
            for rung in RULE_LADDER
        }
        by_tolerance: dict[str, object] = {}
        for tolerance in (
            PRIMARY_TOLERANCE_BASE30,
            *SENSITIVITY_TOLERANCES_BASE30,
        ):
            retained = (
                []
                if chosen is None
                else retained_span_ids(features["videos"][held_out]["records"], chosen)
            )
            by_tolerance[str(tolerance)] = _score_spans(
                fixture_by_name[held_out],
                results[held_out],
                retained,
                tolerance,
            )
        folds.append(
            {
                "held_out_video": held_out,
                "development_videos": development,
                "chosen_rung": chosen,
                "development_primary": development_summary,
                "held_out_by_tolerance": by_tolerance,
            }
        )

    aggregate: dict[str, object] = {}
    for tolerance in (
        PRIMARY_TOLERANCE_BASE30,
        *SENSITIVITY_TOLERANCES_BASE30,
    ):
        fold_scores = [fold["held_out_by_tolerance"][str(tolerance)] for fold in folds]
        retained_total = sum(int(score["retained_records"]) for score in fold_scores)
        correct_total = sum(
            int(score["correct_complete_records"]) for score in fold_scores
        )
        baseline_total = sum(int(score["baseline_correct_records"]) for score in fold_scores)
        baseline_kept = sum(
            int(score["baseline_correct_still_usable"]) for score in fold_scores
        )
        aggregate[str(tolerance)] = {
            "retained_records": retained_total,
            "rejected_records": sum(
                int(features["videos"][video_id]["predicted_span_count"])
                for video_id in VIDEO_IDS
            )
            - retained_total,
            "correct_complete_records": correct_total,
            "records_with_errors": retained_total - correct_total,
            "complete_record_precision": (
                correct_total / retained_total if retained_total else None
            ),
            "baseline_correct_records": baseline_total,
            "baseline_correct_still_usable": baseline_kept,
            "baseline_correct_coverage": (
                baseline_kept / baseline_total if baseline_total else None
            ),
        }
    return {
        "schema": SCORE_SCHEMA,
        "feature_table_sha256": _sha256(features_path),
        "primary_tolerance_base30_frames": PRIMARY_TOLERANCE_BASE30,
        "sensitivity_tolerances_base30_frames": list(
            SENSITIVITY_TOLERANCES_BASE30
        ),
        "selection": (
            "highest retained development count among non-empty zero-error rungs; "
            "prefer the stricter rung on a tie"
        ),
        "completeness_fields": [
            "exact_contact_count",
            "contact_timing",
            "player_attribution",
            "server",
            "point_outcome",
        ],
        "primary_rule_scores_by_video": primary_by_video,
        "folds": folds,
        "aggregate_by_tolerance": aggregate,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-features")
    build.add_argument("--artifacts-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--features", type=Path, required=True)
    score.add_argument("--artifacts-root", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build-features":
        payload = build_features(args.artifacts_root)
    else:
        payload = score_features(args.features, args.artifacts_root)
    _write_json(args.output, payload)


if __name__ == "__main__":
    main()
