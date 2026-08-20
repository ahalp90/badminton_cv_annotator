"""Frame and boundary scoring for validated VLM scene benchmark records."""

from __future__ import annotations

from bisect import bisect_left
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np

from annotator.broadcast_timeline_labels import LabelInterval, SceneTruth, VideoMetadata, validate_partition

from .contracts import BenchmarkRunRecord, RunOutcome


SCORE_SCHEMA_VERSION = 1
LABEL_ORDER = tuple(SceneTruth)


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _sampling_coverage_failures(record: BenchmarkRunRecord) -> list[str]:
    observed = record.observed_sampling
    if observed is None:
        return ["observed sampling is unavailable"]
    frames = observed.sampled_source_frames
    source_fps = record.shard.fps
    requested_fps = record.requested_sampling.fps
    if requested_fps > source_fps:
        return ["requested sampling FPS exceeds the source FPS"]

    duration_frames = record.shard.end_frame - record.shard.start_frame
    expected_count = math.ceil(duration_frames * requested_fps / source_fps)
    count_tolerance = max(1, math.ceil(expected_count * 0.001))
    failures: list[str] = []
    if abs(len(frames) - expected_count) > count_tolerance:
        failures.append(
            f"observed sample count {len(frames)} differs from requested cadence "
            f"(expected about {expected_count})"
        )
    if len(frames) == 1:
        if duration_frames > math.ceil(source_fps / requested_fps):
            failures.append("one sampled frame cannot establish complete shard coverage")
        return failures

    gaps = [right - left for left, right in zip(frames, frames[1:])]
    nominal_gap = source_fps / requested_fps
    edge_tolerance = math.ceil(nominal_gap)
    if frames[0] - record.shard.start_frame > edge_tolerance:
        failures.append("observed sampling starts too late to cover the shard")
    if record.shard.end_frame - 1 - frames[-1] > edge_tolerance:
        failures.append("observed sampling ends too early to cover the shard")

    if expected_count >= 3:
        minimum_gap = max(1, math.floor(nominal_gap) - 1)
        maximum_gap = math.ceil(nominal_gap) + 1
        if any(gap < minimum_gap or gap > maximum_gap for gap in gaps):
            failures.append("observed frame gaps do not match the requested sampling cadence")
    if len(gaps) >= 2 and max(gaps) - min(gaps) > 1:
        failures.append("observed frame gaps contradict a uniform frame grid")
    return failures


def deployment_failures(record: BenchmarkRunRecord) -> list[str]:
    """Return every reason a run cannot enter the accuracy comparison."""
    failures: list[str] = []
    if record.outcome is RunOutcome.FAILED:
        failures.append(f"backend failed: {record.failure_reason}")
        return failures

    observed = record.observed_sampling
    if observed is None:
        failures.append("observed sampling is unavailable")
        return failures
    if not observed.complete_source_coverage:
        failures.append("backend did not attest complete source coverage")
    if not observed.uniform_frame_grid:
        failures.append("observed frame grid is not uniform")
    if (observed.width, observed.height) != (
        record.requested_sampling.width,
        record.requested_sampling.height,
    ):
        failures.append(
            f"observed resolution {observed.width}x{observed.height} differs from requested "
            f"{record.requested_sampling.width}x{record.requested_sampling.height}"
        )
    failures.extend(_sampling_coverage_failures(record))
    if observed.visual_tokens is None:
        failures.append("visual token count is unavailable")
    if observed.total_input_tokens is None:
        failures.append("total input token count is unavailable")
    if record.runtime.peak_vram_mib is None:
        failures.append("peak VRAM is unavailable")
    if record.runtime.cache_dtype is None:
        failures.append("cache dtype is unavailable")
    if record.runtime.cpu_offload:
        failures.append("CPU offload is prohibited by the benchmark")
    return failures


def _truth_for_shard(intervals: Sequence[LabelInterval], record: BenchmarkRunRecord) -> np.ndarray:
    shard = record.shard
    expected = VideoMetadata(shard.video_id, shard.fps, shard.frame_count)
    validate_partition(intervals, expected_metadata=expected)
    truth = np.empty(shard.end_frame - shard.start_frame, dtype=np.int8)
    label_indices = {label: index for index, label in enumerate(LABEL_ORDER)}
    for interval in intervals:
        start = max(interval.start_frame, shard.start_frame)
        end = min(interval.end_frame, shard.end_frame)
        if start < end:
            truth[start - shard.start_frame:end - shard.start_frame] = label_indices[interval.truth]
    return truth


def _predictions_for_shard(record: BenchmarkRunRecord) -> np.ndarray:
    shard = record.shard
    prediction = np.empty(shard.end_frame - shard.start_frame, dtype=np.int8)
    label_indices = {label: index for index, label in enumerate(LABEL_ORDER)}
    for segment in record.segments:
        prediction[segment.start_frame - shard.start_frame:segment.end_frame - shard.start_frame] = label_indices[
            segment.scene_label
        ]
    return prediction


def _confusion(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    n_labels = len(LABEL_ORDER)
    flat = np.bincount(truth * n_labels + prediction, minlength=n_labels * n_labels)
    return flat.reshape(n_labels, n_labels)


def _class_metrics(confusion: np.ndarray) -> dict[str, dict[str, int | float]]:
    metrics: dict[str, dict[str, int | float]] = {}
    for index, label in enumerate(LABEL_ORDER):
        true_positive = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted = int(confusion[:, index].sum())
        precision = _ratio(true_positive, predicted)
        recall = _ratio(true_positive, support)
        metrics[label.value] = {
            "support_frames": support,
            "predicted_frames": predicted,
            "true_positive_frames": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
    return metrics


def _internal_truth_boundaries(intervals: Sequence[LabelInterval], record: BenchmarkRunRecord) -> list[int]:
    return sorted(
        interval.start_frame
        for interval in intervals
        if record.shard.start_frame < interval.start_frame < record.shard.end_frame
    )


def _internal_prediction_boundaries(record: BenchmarkRunRecord) -> list[int]:
    return [segment.start_frame for segment in record.segments[1:]]


def _nearest_errors(reference: Sequence[int], candidates: Sequence[int]) -> list[int]:
    if not candidates:
        return []
    ordered = list(candidates)
    errors: list[int] = []
    for frame in reference:
        index = bisect_left(ordered, frame)
        distances: list[int] = []
        if index < len(ordered):
            distances.append(abs(ordered[index] - frame))
        if index > 0:
            distances.append(abs(ordered[index - 1] - frame))
        errors.append(min(distances))
    return errors


def _error_summary(errors: Sequence[int], reference_count: int) -> dict[str, int | float | None]:
    if not errors:
        return {
            "reference_boundaries": reference_count,
            "median_frames": None,
            "p95_frames": None,
            "max_frames": None,
        }
    ordered = sorted(errors)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "reference_boundaries": reference_count,
        "median_frames": float(median(ordered)),
        "p95_frames": ordered[p95_index],
        "max_frames": ordered[-1],
    }


def _match_with_tolerance(truth: Sequence[int], prediction: Sequence[int], tolerance: int) -> int:
    truth_index = 0
    prediction_index = 0
    matches = 0
    while truth_index < len(truth) and prediction_index < len(prediction):
        truth_frame = truth[truth_index]
        prediction_frame = prediction[prediction_index]
        if abs(truth_frame - prediction_frame) <= tolerance:
            matches += 1
            truth_index += 1
            prediction_index += 1
        elif prediction_frame < truth_frame - tolerance:
            prediction_index += 1
        else:
            truth_index += 1
    return matches


def boundary_metrics(
    truth_boundaries: Sequence[int],
    prediction_boundaries: Sequence[int],
    fps: float,
) -> dict[str, Any]:
    """Score ordered boundary sets without allowing one prediction to match twice."""
    tolerances = {
        "5_frames": 5,
        "10_frames": 10,
        "25_frames": 25,
        "one_second": max(1, round(fps)),
    }
    matches: dict[str, dict[str, int | float]] = {}
    for name, tolerance in tolerances.items():
        matched = _match_with_tolerance(truth_boundaries, prediction_boundaries, tolerance)
        precision = _ratio(matched, len(prediction_boundaries))
        recall = _ratio(matched, len(truth_boundaries))
        matches[name] = {
            "tolerance_frames": tolerance,
            "matched": matched,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
    truth_errors = _nearest_errors(truth_boundaries, prediction_boundaries)
    prediction_errors = _nearest_errors(prediction_boundaries, truth_boundaries)
    return {
        "truth_boundary_count": len(truth_boundaries),
        "prediction_boundary_count": len(prediction_boundaries),
        "matches": matches,
        "truth_to_prediction_error": _error_summary(truth_errors, len(truth_boundaries)),
        "prediction_to_truth_error": _error_summary(prediction_errors, len(prediction_boundaries)),
    }


def score_run_record(record: BenchmarkRunRecord, intervals: Sequence[LabelInterval]) -> dict[str, Any]:
    """Build a JSON-ready deployment and accuracy summary for one run."""
    failures = deployment_failures(record)
    observed = record.observed_sampling
    summary: dict[str, Any] = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "run_id": record.run_id,
        "model": record.model.to_json(),
        "shard": record.shard.to_json(),
        "deployment_gate": {"passed": not failures, "failures": failures},
        "output_validity": {
            "outcome": record.outcome.value,
            "attempt_count": record.attempt_count,
            "first_attempt_valid_json": record.first_attempt_valid_json,
            "first_attempt_valid_prediction": record.first_attempt_valid_prediction,
        },
        "runtime": record.runtime.to_json(),
        "requested_sampling": record.requested_sampling.to_json(),
        "observed_sampling": None if observed is None else observed.to_json(),
        "accuracy": None,
        "boundaries": None,
    }
    if failures:
        return summary

    truth = _truth_for_shard(intervals, record)
    prediction = _predictions_for_shard(record)
    confusion = _confusion(truth, prediction)
    per_class = _class_metrics(confusion)
    macro_f1 = float(np.mean([metrics["f1"] for metrics in per_class.values()]))
    live_index = LABEL_ORDER.index(SceneTruth.LIVE)
    non_standard_index = LABEL_ORDER.index(SceneTruth.LIVE_NON_STANDARD)
    summary["accuracy"] = {
        "frames": len(truth),
        "correct_frames": int(np.trace(confusion)),
        "accuracy": float(np.trace(confusion) / len(truth)),
        "macro_f1": macro_f1,
        "labels": [label.value for label in LABEL_ORDER],
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "live_non_standard_confusion": {
            "truth_live_predicted_live_non_standard": int(confusion[live_index, non_standard_index]),
            "truth_live_non_standard_predicted_live": int(confusion[non_standard_index, live_index]),
        },
    }
    summary["boundaries"] = boundary_metrics(
        _internal_truth_boundaries(intervals, record),
        _internal_prediction_boundaries(record),
        record.shard.fps,
    )
    return summary


def write_score_summary(path: Path, summary: dict[str, Any]) -> None:
    """Atomically write deterministic score JSON and verify direct reload equality."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        reloaded = json.loads(temporary.read_text(encoding="utf-8"))
        if reloaded != summary:
            raise RuntimeError(f"score summary round trip changed values: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
