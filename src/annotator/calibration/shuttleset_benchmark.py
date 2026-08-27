"""Score pinned production rally records against ShuttleSet ground truth."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from annotator import point_winner
from annotator.calibration.gt_scoring import (
    VideoScoring,
    canonical_tolerance,
    flatten_metrics,
    load_set_tables,
    score_video,
)
from annotator.calibration.shuttleset_features import (
    court_corner_error_rows,
    derive_player_feature_inputs,
    derive_shuttle_feature_inputs,
    evaluate_rally_features,
    feature_population,
    score_contact_coordinates,
)
from annotator.config import BaseAnnotatorConfig
from annotator.calibration.scoring import (
    CONTACT_TOLERANCES_BASE30,
    GtRally,
    load_gt_rallies,
    score_contacts,
)
from annotator.fps_constants import ScalingKind
from annotator.point_winner import (
    GeometricVerdictRow,
    Half,
    Landing,
    Verdict,
    VerdictRow,
    VerdictSource,
)
from annotator.run_video import AnnotatorResult
from annotator.types import ContactCandidate
from annotator.video_metadata import VideoMetadata
from dataset_builder.records import load_rally_records
from dataset_builder.vision import load_json_gz, save_json_gz
from shared.court import load_all_court_info


BENCHMARK_SCHEMA = "shuttleset-production-benchmark/1"
EXPECTED_RUN_ID = "a5d37677def443469f6b83d8ee838e7b"
EXPECTED_SOURCE_COMMIT = "ad8da4f297e9278a9cc39bf216026545a7bbab05"
EXPECTED_RUN_MANIFEST_SHA256 = (
    "84f91c139decdc4fe29957b8dd56cdd400491ba2b5aa190684fd3aa0e84a55db"
)
EXPECTED_CONFIGURATION_SHA256 = (
    "6e2a15ea3c44c4bc3cf8b38c461cdfd55c359178b49854080521949c07e93b20"
)
EXPECTED_RALLY_RECORDS_SHA256 = (
    "71c54a7a7521871c152acedd46b399c86e78969b24949b35f6f4bda59567409c"
)
EVALUATOR_BASE_COMMIT = "f7571e60e439230346e4ed3449d56dd3929e7eb6"
VIDEO_ID_PATTERN = re.compile(r"sset_(\d{2})\Z")
LANDING_COORDINATE_UNITS = "normalized doubles-court Euclidean distance"
LANDING_COORDINATE_MATCHING = (
    "GT rally paired through one unique, unmerged covered predicted span; "
    "landing frames are not matched"
)

_RATIO_METRICS = ("ball_round", "player", "server", "hit_height", "landing", "getpoint")
_BOUNDARY_COUNTS = (
    "n_gt_rallies",
    "covered",
    "split",
    "missed",
    "merged_spans",
    "spurious_spans",
)


@dataclass(frozen=True)
class BenchmarkVideo:
    """One exact production source and its ShuttleSet label directory."""

    name: str
    video_id: int
    fps: float
    frame_count: int
    gt_set_dir: Path


def projection_to_annotator_result(
    records: Sequence[Mapping[str, object]],
) -> AnnotatorResult:
    """Rebuild the scored annotator fields from validated rally records."""
    spans: list[tuple[int, int]] = []
    contacts: list[ContactCandidate] = []
    filtered_contacts: list[ContactCandidate] = []
    filtered_by_rally: dict[int, list[int]] = {}
    striker_halves: list[Half | None] = []
    stroke_counts: list[int] = []
    next_servers: list[Half | None] = []
    servers: list[Half | None] = []
    verdict_rows: dict[int, VerdictRow] = {}
    landings: dict[int, Landing | None] = {}
    geometric_rows: dict[int, GeometricVerdictRow] = {}
    hit_heights: dict[int, int] = {}
    hit_height_failures: list[tuple[int, int, int, str]] = []

    for expected_rally_id, raw_record in enumerate(records):
        record = _mapping(raw_record, "rally record")
        key = _mapping(record["key"], "rally key")
        rally_id = _integer(key["rally_id"], "rally_id")
        if rally_id != expected_rally_id:
            raise ValueError("rally records must be contiguous and ordered from zero")
        rally = _mapping(record["rally"], "rally fields")
        spans.append(
            (
                _integer(rally["start_frame"], "start_frame"),
                _integer(rally["end_frame"], "end_frame"),
            )
        )

        contact_payload = _mapping(record["contacts"], "contact fields")
        by_frame: dict[int, ContactCandidate] = {}
        for raw_candidate in _sequence(
            contact_payload["raw_candidates"], "raw candidates"
        ):
            candidate = _mapping(raw_candidate, "raw candidate")
            frame = _integer(candidate["contact_frame"], "raw contact frame")
            row = ContactCandidate(
                rally_id,
                frame,
                _optional_boolean(candidate["proximity_ok"], "proximity_ok"),
                _optional_boolean(candidate["wrist_near"], "wrist_near"),
                _optional_boolean(candidate["suppressed"], "suppressed"),
            )
            contacts.append(row)
            by_frame[frame] = row

        accepted_frames: list[int] = []
        for stroke_idx, raw_accepted in enumerate(
            _sequence(contact_payload["accepted"], "accepted contacts")
        ):
            accepted = _mapping(raw_accepted, "accepted contact")
            if _integer(accepted["stroke_idx"], "stroke_idx") != stroke_idx:
                raise ValueError("accepted stroke indexes must be contiguous")
            frame = _integer(accepted["contact_frame"], "accepted contact frame")
            try:
                filtered_contacts.append(by_frame[frame])
            except KeyError as error:
                raise ValueError("accepted contact has no raw candidate") from error
            accepted_frames.append(frame)
            hit_height = accepted["hit_height_code"]
            if hit_height is not None:
                hit_heights[frame] = _integer(hit_height, "hit_height_code")
        filtered_by_rally[rally_id] = accepted_frames
        stroke_count = _integer(contact_payload["stroke_count"], "stroke_count")
        if stroke_count != len(accepted_frames):
            raise ValueError("stroke_count differs from accepted contacts")
        stroke_counts.append(stroke_count)
        for raw_failure in _sequence(
            contact_payload["hit_height_failures"], "hit-height failures"
        ):
            failure = _mapping(raw_failure, "hit-height failure")
            reason = failure["reason"]
            if not isinstance(reason, str) or not reason:
                raise ValueError("hit-height failure reason must be non-empty")
            hit_height_failures.append(
                (
                    rally_id,
                    _integer(failure["stroke_idx"], "failure stroke_idx"),
                    _integer(failure["contact_frame"], "failure contact_frame"),
                    reason,
                )
            )

        outcomes = _mapping(record["outcomes"], "outcome fields")
        striker = _optional_half(outcomes["striker_half"], "striker_half")
        striker_halves.append(striker)
        servers.append(
            _optional_half(outcomes["server_prediction"], "server_prediction")
        )
        next_servers.append(_optional_half(outcomes["next_server"], "next_server"))
        if striker is None:
            continue
        verdict = _mapping(outcomes["verdict"], "verdict")
        verdict_rows[rally_id] = VerdictRow(
            rally_id,
            striker,
            _optional_enum(verdict["value"], Verdict, "verdict value"),
            _optional_enum(verdict["source"], VerdictSource, "verdict source"),
            _optional_float(verdict["landing_margin_m"], "landing margin"),
            _boolean(verdict["within_line_margin"], "within_line_margin"),
            _boolean(verdict["within_net_margin"], "within_net_margin"),
        )
        landing_payload = outcomes["landing"]
        landings[rally_id] = (
            None if landing_payload is None else _landing(landing_payload)
        )
        geometric = _mapping(outcomes["geometric_verdict"], "geometric verdict")
        geometric_rows[rally_id] = GeometricVerdictRow(
            rally_id,
            _optional_enum(geometric["value"], Verdict, "geometric value"),
            _optional_half(geometric["winner"], "geometric winner"),
            _optional_boolean(geometric["agreement"], "geometric agreement"),
            _boolean(geometric["window_closed_by_mask"], "window_closed_by_mask"),
        )

    return AnnotatorResult(
        spans,
        contacts,
        filtered_contacts,
        filtered_by_rally,
        striker_halves,
        stroke_counts,
        next_servers,
        servers,
        verdict_rows,
        landings,
        geometric_rows,
        hit_heights,
        hit_height_failures,
    )


def benchmark_records(
    *,
    records_path: Path,
    configuration_path: Path,
    ground_truth_root: Path,
) -> dict[str, object]:
    """Validate and score one exact issue #103 record collection."""
    _validate_pinned_file(
        records_path, EXPECTED_RALLY_RECORDS_SHA256, "rally records"
    )
    _validate_pinned_file(
        records_path.parent / "run_manifest.json.gz",
        EXPECTED_RUN_MANIFEST_SHA256,
        "run manifest",
    )
    _validate_pinned_file(
        configuration_path, EXPECTED_CONFIGURATION_SHA256, "production configuration"
    )
    records = load_rally_records(records_path)
    payload = load_json_gz(records_path)
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("rally-record run_id differs from the pinned issue #103 run")
    if payload.get("code_version") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("rally-record source commit differs from issue #103")
    sources = _sequence(payload.get("sources"), "record sources")
    source_rows = [_mapping(source, "record source") for source in sources]
    grouped = _group_records(records)
    source_ids = [
        _string(source["video_id"], "source video_id") for source in source_rows
    ]
    if source_ids != list(grouped):
        raise ValueError("record video order differs from source order")

    master_path = ground_truth_root / "shots_master.csv"
    homography_path = ground_truth_root / "set" / "homography.csv"
    master = pd.read_csv(master_path)
    court_info = load_all_court_info(homography_path)
    per_video: dict[str, dict[str, object]] = {}
    all_scoring: dict[str, VideoScoring] = {}
    contact_curves: dict[str, dict[str, object]] = {}
    landing_coordinate_rows: dict[str, list[dict[str, object]]] = {}
    for source in source_rows:
        video = _benchmark_video(source, ground_truth_root)
        video_records = grouped[video.name]
        result = projection_to_annotator_result(video_records)
        gt_rallies = load_gt_rallies(master, video.video_id)
        last_gt_frame = max(
            frame for rally in gt_rallies for frame in rally.stroke_frames
        )
        if last_gt_frame >= video.frame_count:
            raise ValueError(
                f"{video.name}: GT frame {last_gt_frame} exceeds frame count {video.frame_count}"
            )
        scoring = score_video(
            video,
            result,
            master,
            court_info,
            canonical_tolerance(video.fps),
        )
        all_scoring[video.name] = scoring
        curve = _contact_curve(result, gt_rallies, video.fps)
        landing_rows = _landing_coordinate_rows(
            scoring, result, court_info[video.video_id]
        )
        contact_curves[video.name] = curve
        landing_coordinate_rows[video.name] = landing_rows
        per_video[video.name] = {
            "video_id": video.video_id,
            "fps": video.fps,
            "frame_count": video.frame_count,
            "last_gt_frame": last_gt_frame,
            "detected_rallies": len(result.spans),
            "metrics": _benchmark_metrics(scoring),
            "contact_curve": curve,
            "landing_coordinates": _landing_coordinate_output(
                landing_rows, include_rows=True
            ),
            "ground_truth_reconciliation": _reconciliation_counts(scoring),
            "rallies": _rally_output_rows(scoring),
        }

    return {
        "schema": BENCHMARK_SCHEMA,
        "provenance": {
            "run_id": EXPECTED_RUN_ID,
            "production_source_commit": EXPECTED_SOURCE_COMMIT,
            "production_configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
            "run_manifest_sha256": EXPECTED_RUN_MANIFEST_SHA256,
            "rally_records_sha256": EXPECTED_RALLY_RECORDS_SHA256,
            "ground_truth_content_sha256": _tree_digest(ground_truth_root),
            "shots_master_sha256": _sha256(master_path),
            "homography_sha256": _sha256(homography_path),
        },
        "population": {
            "videos": len(per_video),
            "predicted_rallies": len(records),
            "gt_rallies": sum(len(scoring.rows) for scoring in all_scoring.values()),
        },
        "matching": {
            "rally": "all GT contacts must fall in exactly one half-open predicted span",
            "contact": "deterministic greedy one-to-one nearest-frame matching",
            "landing_coordinates": LANDING_COORDINATE_MATCHING,
            "contact_tolerances_base30": list(CONTACT_TOLERANCES_BASE30),
        },
        "aggregate": _aggregate(
            all_scoring, contact_curves, landing_coordinate_rows
        ),
        "per_video": per_video,
    }


def benchmark_features(
    *,
    records_path: Path,
    configuration_path: Path,
    ground_truth_root: Path,
    run_dir: Path,
    pinned_repo_dir: Path,
    source_root: Path,
) -> dict[str, object]:
    """Evaluate issue #22 features from replay-validated issue #103 primitives."""
    destination = Path(run_dir).resolve(strict=True)
    pinned_repo = Path(pinned_repo_dir).resolve(strict=True)
    source_root = Path(source_root).resolve(strict=True)
    workspace = destination / "workspace"
    os.environ["BADMINTON_SCRAPE_DIR"] = os.fspath(workspace)

    # Import after binding the workspace because scraper configuration is set
    # at import time. This is the same supported restore path used by replay.
    from dataset_builder._pipeline_runtime import DefaultPipelineRuntime
    from dataset_builder.artifact_index import artifact_index_path
    from dataset_builder.cli import load_builder_config
    from dataset_builder.manifest import load_run_manifest

    _validate_pinned_file(
        records_path, EXPECTED_RALLY_RECORDS_SHA256, "rally records"
    )
    _validate_pinned_file(
        configuration_path,
        EXPECTED_CONFIGURATION_SHA256,
        "production configuration",
    )
    _validate_pinned_file(
        destination / "run_manifest.json.gz",
        EXPECTED_RUN_MANIFEST_SHA256,
        "run manifest",
    )
    config = load_builder_config(configuration_path, repo_root=pinned_repo)
    if config.fixed_sources is None:
        raise ValueError("feature evaluation requires fixed ShuttleSet sources")
    os.environ[config.fixed_sources.source_root_environment] = os.fspath(source_root)
    runtime = DefaultPipelineRuntime(config, destination, EXPECTED_SOURCE_COMMIT)
    runtime.preflight_replay()
    selected_ids = runtime.prepare_annotation_replay(load_run_manifest(destination))
    if selected_ids != config.fixed_sources.video_ids:
        raise ValueError("restored video IDs differ from the pinned configuration")

    payload = load_json_gz(records_path)
    records = load_rally_records(records_path)
    grouped = _group_records(records)
    sources = {
        _string(_mapping(row, "record source")["video_id"], "source video_id"):
        _mapping(row, "record source")
        for row in _sequence(payload.get("sources"), "record sources")
    }
    if tuple(grouped) != selected_ids or set(sources) != set(selected_ids):
        raise ValueError("record, artifact, and configured video populations differ")

    master = pd.read_csv(ground_truth_root / "shots_master.csv")
    homography = pd.read_csv(ground_truth_root / "set" / "homography.csv").set_index("id")
    static_courts = load_all_court_info(ground_truth_root / "set" / "homography.csv")
    base = BaseAnnotatorConfig()
    per_video: dict[str, dict[str, object]] = {}
    for video_id in selected_ids:
        video = _benchmark_video(sources[video_id], ground_truth_root)
        metadata = runtime.state.metadata[video_id]
        if metadata.frame_count != video.frame_count or float(metadata.fps) != video.fps:
            raise ValueError(f"{video_id}: restored metadata differs from rally records")
        shuttle = runtime.state.shuttles[video_id]
        pose = runtime.state.poses[video_id]
        court = runtime.state.courts[video_id]
        court_inputs = court.evidence.inputs
        if court_inputs is None:
            raise ValueError(f"{video_id}: court evidence has no operational inputs")
        player_inputs = derive_player_feature_inputs(
            shuttle.track, pose, court, video_id
        )
        shuttle_inputs = derive_shuttle_feature_inputs(
            shuttle.track,
            shuttle.guard_codes,
            base.rejected_grades,
            court_inputs.homography_rows,
            court_inputs.resolution,
        )
        feature_rows = [
            evaluate_rally_features(
                record,
                player_inputs.posture,
                player_inputs.court_positions,
                player_inputs.posture_interpolation,
                player_inputs.position_interpolation,
                video.fps,
            )
            for record in grouped[video_id]
        ]
        gt_rallies = load_gt_rallies(master, video.video_id)
        tables = load_set_tables(video, gt_rallies)
        contacts, annotation_population = _contact_ground_truth(
            tables, master, video.video_id
        )
        contact_coordinates = score_contact_coordinates(
            contacts,
            shuttle_inputs,
            player_inputs.court_positions,
            static_courts[video.video_id],
        )
        corner_rows = court_corner_error_rows(
            court_inputs.homography_rows,
            _ground_truth_corners(homography.loc[video.video_id]),
            court_inputs.resolution,
        )
        per_video[video_id] = {
            "video_id": video.video_id,
            "fps": video.fps,
            "frame_count": video.frame_count,
            "artifact_index_sha256": _sha256(
                artifact_index_path(destination, video_id)
            ),
            "population": feature_population(feature_rows),
            "annotation_population": annotation_population,
            "contact_coordinates": contact_coordinates,
            "court_corners": {
                "units": "pixels at 1280x720 ShuttleSet reference resolution",
                "matching": "each accepted production scene against the static ShuttleSet quad",
                "summary": _court_corner_summary(corner_rows),
                "rows": corner_rows,
            },
            "rallies": feature_rows,
        }

    return {
        "schema": "shuttleset-trial-feature-benchmark/1",
        "provenance": {
            "artifact_run_id": EXPECTED_RUN_ID,
            "artifact_source_commit": EXPECTED_SOURCE_COMMIT,
            "artifact_run_manifest_sha256": EXPECTED_RUN_MANIFEST_SHA256,
            "production_configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
            "rally_records_sha256": EXPECTED_RALLY_RECORDS_SHA256,
            "evaluator_base_commit": EVALUATOR_BASE_COMMIT,
            "evaluator_files_sha256": {
                "shuttleset_benchmark.py": _sha256(Path(__file__)),
                "shuttleset_features.py": _sha256(
                    Path(__file__).with_name("shuttleset_features.py")
                ),
                "gt_scoring.py": _sha256(Path(__file__).with_name("gt_scoring.py")),
            },
        },
        "policy": {
            "rally_duration_end_offset": "unresolved",
            "serve_endpoint_and_static_tolerance": "unresolved",
            "degradation_temperature": "unresolved",
            "backward_extrapolation": "unresolved",
            "commentary": "unavailable because issue #103 disabled commentary",
        },
        "population": feature_population(
            [row for video in per_video.values() for row in video["rallies"]]
        ),
        "aggregate": _aggregate_feature_outputs(per_video),
        "per_video": per_video,
    }


def _benchmark_video(
    source: Mapping[str, object], ground_truth_root: Path
) -> BenchmarkVideo:
    video_id = _string(source["video_id"], "source video_id")
    match = VIDEO_ID_PATTERN.fullmatch(video_id)
    if match is None:
        raise ValueError(f"unsupported ShuttleSet video ID: {video_id!r}")
    reference = _mapping(source["source_reference"], "source reference")
    if reference.get("video_id") != video_id:
        raise ValueError(f"{video_id}: source-reference identity differs")
    metadata = VideoMetadata.from_dict(source["video_metadata"])
    title = _string(reference["title"], "source title")
    gt_set_dir = ground_truth_root / 'set' / title
    if not gt_set_dir.is_dir():
        raise FileNotFoundError(
            f"{video_id}: ground-truth directory is missing: {gt_set_dir}"
        )
    return BenchmarkVideo(
        name=video_id,
        video_id=int(match.group(1)),
        fps=float(metadata.fps),
        frame_count=metadata.frame_count,
        gt_set_dir=gt_set_dir,
    )


def _contact_curve(
    result: AnnotatorResult,
    gt_rallies: Sequence[GtRally],
    fps: float,
) -> dict[str, object]:
    tolerances = tuple(
        int(ScalingKind.FRAME_COUNT.scale(base_frames, fps))
        for base_frames in CONTACT_TOLERANCES_BASE30
    )
    scored = score_contacts(
        result.spans,
        result.filtered_contacts,
        gt_rallies,
        tolerances=tolerances,
    )
    return {
        str(base): {
            "frames": frames,
            "matched": scored["tolerances"][str(frames)]["matched"],
            "gt": scored["tolerances"][str(frames)]["gt"],
            "candidates": scored["tolerances"][str(frames)]["candidates"],
            "raw_matched": scored["precision_raw"][str(frames)]["matched"],
            "raw_candidates": scored["precision_raw"][str(frames)]["candidates"],
        }
        for base, frames in zip(CONTACT_TOLERANCES_BASE30, tolerances, strict=True)
    }


def _aggregate(
    scorings: Mapping[str, VideoScoring],
    contact_curves: Mapping[str, Mapping[str, object]],
    landing_coordinate_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    metrics = {name: _benchmark_metrics(scoring) for name, scoring in scorings.items()}
    result: dict[str, object] = {
        key: sum(int(row[key]) for row in metrics.values()) for key in _BOUNDARY_COUNTS
    }
    gt_rallies = int(result["n_gt_rallies"])
    result["covered_fraction"] = _ratio(int(result["covered"]), gt_rallies)
    for name in _RATIO_METRICS:
        for view in ("primary", "covered"):
            correct_key = f"{name}_{view}_correct"
            total_key = f"{name}_{view}_total"
            correct = sum(int(row[correct_key]) for row in metrics.values())
            total = sum(int(row[total_key]) for row in metrics.values())
            result[correct_key] = correct
            result[total_key] = total
            result[f"{name}_{view}"] = _ratio(correct, total)
    for view in ("timing_primary", "timing_covered"):
        matched_key = f"{view}_matched"
        total_key = f"{view}_total"
        matched = sum(int(row[matched_key]) for row in metrics.values())
        total = sum(int(row[total_key]) for row in metrics.values())
        result[matched_key] = matched
        result[total_key] = total
        result[f"{view}_recall"] = _ratio(matched, total)
    contact_matches = sum(int(row["contact_matches"]) for row in metrics.values())
    contact_candidates = sum(
        int(row["contact_filtered_total"]) for row in metrics.values()
    )
    contact_gt = sum(int(row["contact_gt_total"]) for row in metrics.values())
    precision = _ratio(contact_matches, contact_candidates)
    recall = _ratio(contact_matches, contact_gt)
    result.update(
        {
            "contact_matches": contact_matches,
            "contact_filtered_total": contact_candidates,
            "contact_gt_total": contact_gt,
            "contact_precision": precision,
            "contact_recall": recall,
            "contact_f1": _f1(precision, recall),
            "n_raw_contacts": sum(
                int(row["n_raw_contacts"]) for row in metrics.values()
            ),
            "n_filtered_contacts": sum(
                int(row["n_filtered_contacts"]) for row in metrics.values()
            ),
            "hit_height_failures": sum(
                int(row["hit_height_failures"]) for row in metrics.values()
            ),
        }
    )
    result["contact_curve"] = _aggregate_contact_curves(contact_curves)
    result["landing_coordinates"] = _landing_coordinate_output(
        [row for rows in landing_coordinate_rows.values() for row in rows],
        include_rows=False,
    )
    result["ground_truth_reconciliation"] = {
        key: sum(_reconciliation_counts(scoring)[key] for scoring in scorings.values())
        for key in ("exact_rallies", "deduplicated_rallies", "mismatched_rallies")
    }
    result["outcome_mapping_excluded_merged_rallies"] = sum(
        int(row["outcome_mapping_excluded_merged_rallies"])
        for row in metrics.values()
    )
    return result


def _merged_outcome_rows(scoring: VideoScoring) -> set[int]:
    """Return GT row indexes whose outcome mapping reuses a merged span."""
    span_counts: dict[int, int] = defaultdict(int)
    for row in scoring.rows:
        if row.classification == "covered" and row.mapped_span is not None:
            span_counts[row.mapped_span] += 1
    return {
        row.gt_index
        for row in scoring.rows
        if row.mapped_span is not None and span_counts[row.mapped_span] > 1
    }


def _benchmark_metrics(scoring: VideoScoring) -> dict[str, int | float | None]:
    """Return benchmark metrics without reusing one merged-span outcome."""
    metrics = flatten_metrics(scoring)
    excluded = _merged_outcome_rows(scoring)
    fields = {
        "ball_round": ("ball_round_correct", None),
        "player": ("player_correct", None),
        "server": ("server_correct", None),
        "landing": ("landing_correct", "landing_eligible"),
        "getpoint": ("getpoint_correct", "getpoint_eligible"),
    }
    for name, (correct_field, eligibility_field) in fields.items():
        eligible_rows = [
            row
            for row in scoring.rows
            if row.gt_index not in excluded
            and (
                eligibility_field is None
                or bool(getattr(row, eligibility_field))
            )
        ]
        covered_rows = [
            row for row in eligible_rows if row.classification == "covered"
        ]
        for view, rows in (("primary", eligible_rows), ("covered", covered_rows)):
            correct = sum(bool(getattr(row, correct_field)) for row in rows)
            total = len(rows)
            metrics[f"{name}_{view}_correct"] = correct
            metrics[f"{name}_{view}_total"] = total
            metrics[f"{name}_{view}"] = _ratio(correct, total)
    metrics["outcome_mapping_excluded_merged_rallies"] = len(excluded)
    return metrics


def _rally_output_rows(scoring: VideoScoring) -> list[dict[str, object]]:
    """Serialise rallies while marking merged-span outcomes unusable."""
    excluded = _merged_outcome_rows(scoring)
    rows = []
    for row in scoring.rows:
        payload = dict(row._asdict())
        eligible = row.gt_index not in excluded
        payload["outcome_mapping_eligible"] = eligible
        if not eligible:
            for field in (
                "ball_round_pred",
                "ball_round_correct",
                "player_pred",
                "player_correct",
                "server_pred",
                "server_correct",
                "getpoint_pred",
                "getpoint_correct",
                "landing_pred",
                "landing_correct",
            ):
                payload[field] = None
        rows.append(payload)
    return rows


def _landing_coordinate_rows(
    scoring: VideoScoring,
    result: AnnotatorResult,
    court_info: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    merged_outcomes = _merged_outcome_rows(scoring)
    for scored, winner in zip(
        scoring.rows, scoring.reconciliation.winners, strict=True
    ):
        gt_position = None
        if winner.landing_px is not None:
            projected = point_winner.project_pixels_to_court(
                np.asarray(winner.landing_px, dtype=float).reshape(2, 1),
                point_winner.HOMOGRAPHY_RESOLUTION,
                court_info,
            )
            gt_position = (float(projected[0, 0]), float(projected[1, 0]))
        mapping_eligible = scored.gt_index not in merged_outcomes
        prediction = (
            result.landings.get(scored.mapped_span)
            if mapping_eligible and scored.mapped_span is not None
            else None
        )
        predicted_position = None if prediction is None else prediction.norm
        error = (
            None
            if gt_position is None or predicted_position is None
            else float(
                np.linalg.norm(
                    np.asarray(predicted_position) - np.asarray(gt_position)
                )
            )
        )
        rows.append(
            {
                "gt_index": scored.gt_index,
                "set_id": scored.set_id,
                "rally": scored.rally,
                "mapping_eligible": mapping_eligible,
                "ground_truth_available": gt_position is not None,
                "prediction_available": predicted_position is not None,
                "error": error,
            }
        )
    return rows


def _coordinate_rows_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int | float | None]:
    mapping_eligible = [bool(row.get("mapping_eligible", True)) for row in rows]
    errors = [
        float(row["error"])
        for eligible, row in zip(mapping_eligible, rows, strict=True)
        if eligible and row["error"] is not None
    ]
    gt_available = sum(
        eligible and bool(row["ground_truth_available"])
        for eligible, row in zip(mapping_eligible, rows, strict=True)
    )
    pred_available = sum(
        eligible
        and bool(row["ground_truth_available"])
        and bool(row["prediction_available"])
        for eligible, row in zip(mapping_eligible, rows, strict=True)
    )
    excluded_mapping = len(mapping_eligible) - sum(mapping_eligible)
    eligible_population = len(rows) - excluded_mapping
    return {
        "population": len(rows),
        "mapping_eligible": eligible_population,
        "excluded_merged_mapping": excluded_mapping,
        "ground_truth_available": gt_available,
        "prediction_available": pred_available,
        "eligible": len(errors),
        "excluded_ground_truth": eligible_population - gt_available,
        "excluded_prediction": gt_available - pred_available,
        "mean_error": None if not errors else float(np.mean(errors)),
        "median_error": None if not errors else float(np.median(errors)),
        "p90_error": None if not errors else float(np.percentile(errors, 90)),
    }


def _landing_coordinate_output(
    rows: Sequence[Mapping[str, object]], *, include_rows: bool
) -> dict[str, object]:
    output: dict[str, object] = {
        "units": LANDING_COORDINATE_UNITS,
        "matching": LANDING_COORDINATE_MATCHING,
        "summary": _coordinate_rows_summary(rows),
    }
    if include_rows:
        output["rows"] = list(rows)
    return output


def _ground_truth_corners(row: pd.Series) -> np.ndarray:
    """Return ShuttleSet's static quad in production scene-corner order."""
    return np.asarray(
        (
            (row["upleft_x"], row["upleft_y"]),
            (row["upright_x"], row["upright_y"]),
            (row["downright_x"], row["downright_y"]),
            (row["downleft_x"], row["downleft_y"]),
        ),
        dtype=float,
    )


def _contact_ground_truth(
    tables: Mapping[str, pd.DataFrame],
    master: pd.DataFrame,
    video_id: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return detailed contacts with an exact authoritative master-row match."""
    contacts = pd.concat(
        tuple(table.assign(set_id=set_id) for set_id, table in tables.items()),
        ignore_index=True,
    ).copy()
    keys = ["set_id", "rally", "ball_round", "frame_num"]
    per_video = master.loc[
        master["vid"] == video_id, [*keys, "player_side"]
    ].copy()
    if per_video.duplicated(keys).any():
        raise ValueError(f"video {video_id}: shots_master contact keys are not unique")
    joined = contacts.merge(
        per_video,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    aligned = joined["_merge"] == "both"
    if int(aligned.sum()) != len(per_video):
        raise ValueError(f"video {video_id}: not every shots_master row was recovered")
    frame_group_sizes = contacts.groupby(["set_id", "rally", "frame_num"]).size()
    duplicate_groups = frame_group_sizes[frame_group_sizes > 1]
    flaw_marked = contacts["flaw"].notna()
    population = {
        "source_rows": len(contacts),
        "master_aligned_rows": int(aligned.sum()),
        "unmatched_rows": int((~aligned).sum()),
        "flaw_marked_rows": int(flaw_marked.sum()),
        "flaw_marked_aligned_rows": int((flaw_marked & aligned).sum()),
        "flaw_marked_unmatched_rows": int((flaw_marked & ~aligned).sum()),
        "duplicate_frame_groups": len(duplicate_groups),
        "duplicate_frame_extra_rows": int((duplicate_groups - 1).sum()),
    }
    return joined.loc[aligned].drop(columns="_merge").reset_index(drop=True), population


def _court_corner_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int | float | None]:
    errors = [
        float(error)
        for row in rows
        for error in _sequence(row["corner_errors_px"], "corner errors")
    ]
    return {
        "scenes": len(rows),
        "corners": len(errors),
        **_error_statistics(errors),
    }


def _aggregate_feature_outputs(
    per_video: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    contact: dict[str, object] = {}
    for name in ("shuttle", "striker", "opponent"):
        population = eligible = excluded_prediction = excluded_ground_truth = 0
        errors: list[float] = []
        for video in per_video.values():
            output = _mapping(video["contact_coordinates"], "contact coordinates")
            summary = _mapping(output["summary"], "contact summary")
            row = _mapping(summary[name], f"{name} summary")
            population += _integer(row["population"], "coordinate population")
            eligible += _integer(row["eligible"], "coordinate eligible")
            excluded_prediction += _integer(
                row["excluded_prediction"], "coordinate prediction exclusions"
            )
            excluded_ground_truth += _integer(
                row["excluded_ground_truth"], "coordinate ground-truth exclusions"
            )
            for detail in _sequence(output["rows"], "coordinate rows"):
                error = _mapping(detail, "coordinate row")[f"{name}_error"]
                if error is not None:
                    errors.append(_float(error, "coordinate error"))
        contact[name] = {
            "population": population,
            "eligible": eligible,
            "excluded_prediction": excluded_prediction,
            "excluded_ground_truth": excluded_ground_truth,
            **_error_statistics(errors),
        }

    corner_rows = [
        _mapping(row, "court corner row")
        for video in per_video.values()
        for row in _sequence(
            _mapping(video["court_corners"], "court corners")["rows"],
            "court corner rows",
        )
    ]
    fps_groups: dict[str, dict[str, int]] = {}
    for video in per_video.values():
        fps = _float(video["fps"], "video fps")
        key = f"{fps:g}"
        group = fps_groups.setdefault(key, {"videos": 0, "rallies": 0})
        group["videos"] += 1
        group["rallies"] += _integer(
            _mapping(video["population"], "feature population")["rallies"],
            "feature rallies",
        )
    return {
        "fps_groups": fps_groups,
        "contact_coordinates": {
            "units": "normalized doubles-court Euclidean distance",
            "matching": "exact authoritative GT contact frame",
            "summary": contact,
        },
        "court_corners": {
            "units": "pixels at 1280x720 ShuttleSet reference resolution",
            "matching": "each accepted production scene against the static ShuttleSet quad",
            "summary": _court_corner_summary(corner_rows),
        },
    }


def _error_statistics(
    errors: Sequence[float],
) -> dict[str, int | float | None]:
    return {
        "eligible": len(errors),
        "mean_error": None if not errors else float(np.mean(errors)),
        "median_error": None if not errors else float(np.median(errors)),
        "p90_error": None if not errors else float(np.percentile(errors, 90)),
    }


def _reconciliation_counts(scoring: VideoScoring) -> dict[str, int]:
    reconciliation = scoring.reconciliation
    return {
        "exact_rallies": reconciliation.n_exact,
        "deduplicated_rallies": reconciliation.n_dedup,
        "mismatched_rallies": reconciliation.n_mismatch,
        "frame_offset": reconciliation.offset,
    }


def _aggregate_contact_curves(
    curves: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, int | float | None]]:
    aggregate: dict[str, dict[str, int | float | None]] = {}
    for base in CONTACT_TOLERANCES_BASE30:
        key = str(base)
        rows = [
            _mapping(curve[key], f"contact curve {key}") for curve in curves.values()
        ]
        matched = sum(_integer(row["matched"], "matched") for row in rows)
        gt = sum(_integer(row["gt"], "gt") for row in rows)
        candidates = sum(_integer(row["candidates"], "candidates") for row in rows)
        raw_matched = sum(_integer(row["raw_matched"], "raw matched") for row in rows)
        raw_candidates = sum(
            _integer(row["raw_candidates"], "raw candidates") for row in rows
        )
        precision = _ratio(matched, candidates)
        raw_precision = _ratio(raw_matched, raw_candidates)
        recall = _ratio(matched, gt)
        aggregate[key] = {
            "matched": matched,
            "gt": gt,
            "candidates": candidates,
            "raw_matched": raw_matched,
            "raw_candidates": raw_candidates,
            "precision": precision,
            "raw_precision": raw_precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "raw_f1": _f1(raw_precision, recall),
        }
    return aggregate


def _group_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        key = _mapping(record["key"], "record key")
        grouped[_string(key["video_id"], "record video_id")].append(record)
    return dict(grouped)


def _landing(payload: object) -> Landing:
    row = _mapping(payload, "landing")
    position = _sequence(row["normalized_court_position"], "landing position")
    if len(position) != 2:
        raise ValueError("landing position must contain two coordinates")
    return Landing(
        _integer(row["frame"], "landing frame"),
        (_float(position[0], "landing x"), _float(position[1], "landing y")),
        Half(_string(row["court_half"], "landing half")),
        _boolean(row["at_image_border"], "landing border flag"),
        _boolean(row["net_ender"], "landing net flag"),
    )


def _validate_pinned_file(path: Path, expected_sha256: str, name: str) -> None:
    if _sha256(path) != expected_sha256:
        raise ValueError(f"{name} SHA-256 differs from the pinned issue #103 input")


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _optional_boolean(value: object, name: str) -> bool | None:
    return None if value is None else _boolean(value, name)


def _optional_float(value: object, name: str) -> float | None:
    return None if value is None else _float(value, name)


def _optional_half(value: object, name: str) -> Half | None:
    return None if value is None else Half(_string(value, name))


def _optional_enum(value: object, enum_type: type[Any], name: str) -> Any:
    return None if value is None else enum_type(_string(value, name))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--pinned-repo-dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = benchmark_records(
        records_path=args.records.resolve(strict=True),
        configuration_path=args.configuration.resolve(strict=True),
        ground_truth_root=args.ground_truth_root.resolve(strict=True),
    )
    feature_arguments = (args.run_dir, args.pinned_repo_dir, args.source_root)
    if any(value is not None for value in feature_arguments):
        if any(value is None for value in feature_arguments):
            raise ValueError(
                "--run-dir, --pinned-repo-dir, and --source-root must be supplied together"
            )
        document["feature_evaluation"] = benchmark_features(
            records_path=args.records.resolve(strict=True),
            configuration_path=args.configuration.resolve(strict=True),
            ground_truth_root=args.ground_truth_root.resolve(strict=True),
            run_dir=args.run_dir,
            pinned_repo_dir=args.pinned_repo_dir,
            source_root=args.source_root,
        )
    save_json_gz(args.out, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
