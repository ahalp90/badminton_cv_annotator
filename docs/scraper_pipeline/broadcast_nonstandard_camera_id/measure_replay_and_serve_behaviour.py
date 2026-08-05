"""Measure current replay and serve behaviour against the reviewed sset_01 timeline.

Run from the repository root with the digest-pinned external fixture available::

    PYTHONPATH=src python \
      docs/scraper_pipeline/broadcast_nonstandard_camera_id/measure_replay_and_serve_behaviour.py \
      --fixture-root /path/to/autograder_architecture \
      --fixture-profile historical-calibration \
      --out /path/to/issue29-results

The command does not change production output. It writes reload-checked gzip
tables, a gzip summary, and a concise Markdown report. The replay duplicate
study records the reviewer's interval-level source adjudication while leaving
exact frame-pair margin measurement to follow-up work.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, NamedTuple, Sequence

import numpy as np

from annotator.calibration.fixtures import (
    SSET_01,
    Fixture,
    FixtureDigests,
    fixtures_root,
)
from annotator.calibration.gt_scoring import build_run_video_inputs, canonical_tolerance
from annotator.calibration.scoring import GtRally, classify_all, greedy_match, load_gt_rallies
from annotator.config import BaseAnnotatorConfig, SLOWMO_SPEED_FRAC
from annotator.inpaint_guard import CODE_NAMES, NO_FLAG
from annotator.broadcast_timeline_labels import (
    LabelInterval,
    SceneTruth,
    VideoMetadata,
    read_label_csv,
    validate_partition,
)
from annotator.rally_segmentation import (
    BODY_UNIT_WRIST_THRESHOLD,
    build_sticky_result,
    compute_speed,
    detect_contact_flags,
    find_rally_spans,
    rolling_nanmedian,
    tracker_segments,
)
from annotator.replay_mask import (
    court_absence_signal,
    filter_short_exclusion_runs,
    perspective_shift_signal,
    velocity_drop_signal,
)
from annotator.resolve import resolve
from annotator.run_video import RunCapture, run_video


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS = SCRIPT_DIR / "data/sset_01_broadcast_timeline_labels.csv.gz"
TRUTH_ORDER = tuple(SceneTruth)
POSITIVE_TRUTHS = (SceneTruth.REPLAY.value, SceneTruth.CUTAWAY.value)
DUPLICATE_EXCLUDED_LONG_MONTAGE = (147049, 148312)
DUPLICATE_REVIEWER_ADJUDICATION = (
    "The human reviewer confirmed that each short replay shows live footage from immediately "
    "before it. The long end-of-match replay montage is excluded from this source relation."
)

MASK_COLUMNS = (
    "detector", "stratum", "frames", "flagged_frames", "flag_rate",
    "tp", "fp", "tn", "fn", "precision", "recall",
)
SLOW_COLUMNS = (
    "truth", "inpaint_code", "inpaint_name", "frames", "visible_frames",
    "measured_speed_frames", "slow_signal_frames", "slow_signal_rate",
    "rolling_speed_median",
)
CANDIDATE_COLUMNS = (
    "span_id", "span_start", "span_end", "anchor_frame", "lookback_start",
    "candidate_frame", "impulse", "manual_truth", "inpaint_code", "inpaint_name",
    "track_visible", "sticky_analysed", "sticky_distance_bh", "sticky_top_pick",
    "sticky_bottom_pick", "wrist_gate_pass", "evidence_pass", "definitive_mask",
    "raw_mask", "policy_pass", "selected_evidence", "selected_policy", "reject_reasons",
    "nearest_replay_or_cutaway_distance", "nearest_gt_rally_index", "nearest_gt_serve_frame",
    "nearest_gt_serve_delta", "tolerance_frames", "gt_serve_match",
    "nearest_gt_status", "natural_raw_contact",
    "natural_accepted_contact",
)
OPPORTUNITY_COLUMNS = (
    "span_id", "span_start", "span_end", "anchor_frame", "lookback_start",
    "candidate_count", "evidence_pass_count", "policy_pass_count",
    "selected_evidence_frame", "selected_policy_frame",
)


class SlowMotionDetails(NamedTuple):
    signal: np.ndarray
    speed: np.ndarray
    rolling_speed: np.ndarray
    rally_median: float
    slow_threshold: float


class FixtureProfile(NamedTuple):
    fixture: Fixture
    source: str
    source_commit: str | None


UNE_189C5AF_STATIC_STRIDE8 = replace(
    SSET_01,
    digests=FixtureDigests(
        track="08c5afced66b561517a43571df567b2f",
        bboxes="4c9525949d1c79f0161f81b2bb63d5ef",
        scores="03e655b3429f9482c5a3f4df766a3534",
        kps="621427713fc617d81d4081db15613b06",
        kp_scores="deb1ab46efcbe34a19bd4590b2f1b384",
        ndet="5cc366f2cd459ea9be44876bc07e74ea",
        dead_mask="70a2a4e9cbd7c6c02b497b468682c462",
        court_present="65f4e28d0556c0e5422f569ad4b69fac",
        scene_rows="7d781f33e29804ef8363bbbd1b60d772",
    ),
)
FIXTURE_PROFILES = {
    "historical-calibration": FixtureProfile(
        SSET_01,
        "maintained calibration pins",
        None,
    ),
    "une-189c5af-static-stride8": FixtureProfile(
        UNE_189C5AF_STATIC_STRIDE8,
        "UNE sset_measure_189c5af static_shuttleset_homography/sset_01/tracknet-stride-8",
        "189c5af58e45d23ae827dde516924194eb238e18",
    ),
}


def expand_truth(intervals: Sequence[LabelInterval], metadata: VideoMetadata) -> np.ndarray:
    """Expand a validated interval partition to one in-memory class per frame."""
    validate_partition(intervals, expected_metadata=metadata)
    truth = np.empty(metadata.frame_count, dtype="U18")
    for interval in intervals:
        truth[interval.start_frame:interval.end_frame] = interval.truth.value
    return truth


def binary_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, int | float | None]:
    """Score one Boolean prediction while excluding manual ``other`` frames."""
    if prediction.shape != truth.shape or prediction.dtype != np.bool_:
        raise ValueError("prediction must be a bool array with the truth shape")
    known = truth != SceneTruth.OTHER.value
    positive = np.isin(truth, POSITIVE_TRUTHS)
    tp = int(np.count_nonzero(prediction & positive & known))
    fp = int(np.count_nonzero(prediction & ~positive & known))
    tn = int(np.count_nonzero(~prediction & ~positive & known))
    fn = int(np.count_nonzero(~prediction & positive & known))
    return {
        "frames": int(np.count_nonzero(known)),
        "flagged_frames": int(np.count_nonzero(prediction & known)),
        "flag_rate": _ratio(tp + fp, tp + fp + tn + fn),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
    }


def mask_metric_rows(detectors: dict[str, np.ndarray], truth: np.ndarray) -> list[dict[str, Any]]:
    """Return binary and class-specific frame counts for each detector."""
    rows: list[dict[str, Any]] = []
    for detector, prediction in detectors.items():
        rows.append({"detector": detector, "stratum": "binary", **binary_metrics(prediction, truth)})
        for scene_truth in TRUTH_ORDER:
            selected = truth == scene_truth.value
            frames = int(selected.sum())
            flagged = int(np.count_nonzero(prediction & selected))
            rows.append({
                "detector": detector,
                "stratum": scene_truth.value,
                "frames": frames,
                "flagged_frames": flagged,
                "flag_rate": _ratio(flagged, frames),
                "tp": None,
                "fp": None,
                "tn": None,
                "fn": None,
                "precision": None,
                "recall": None,
            })
    return rows


def nearest_replay_or_cutaway_distance(truth: np.ndarray) -> np.ndarray:
    """Return distance in frames to the nearest replay or cutaway frame."""
    positive = np.isin(truth, POSITIVE_TRUTHS)
    n_frames = len(truth)
    sentinel = n_frames + 1
    distance = np.full(n_frames, sentinel, dtype=np.int64)
    previous = -sentinel
    for frame in range(n_frames):
        if positive[frame]:
            previous = frame
        distance[frame] = frame - previous
    following = n_frames + sentinel
    for frame in range(n_frames - 1, -1, -1):
        if positive[frame]:
            following = frame
        distance[frame] = min(distance[frame], following - frame)
    return distance


def replay_duplicate_feasibility(
    intervals: Sequence[LabelInterval],
    gt_rally_count: int,
    *,
    excluded_long_montage: tuple[int, int] = DUPLICATE_EXCLUDED_LONG_MONTAGE,
) -> dict[str, Any]:
    """Record the human-adjudicated replay-pair feasibility result."""
    replays = [interval for interval in intervals if interval.truth is SceneTruth.REPLAY]
    excluded = [
        interval
        for interval in replays
        if (interval.start_frame, interval.end_frame) == excluded_long_montage
    ]
    if len(excluded) != 1:
        raise ValueError(
            f"expected one excluded long replay montage {excluded_long_montage}, "
            f"found {len(excluded)}"
        )
    eligible_count = len(replays) - 1
    if eligible_count < 1 or gt_rally_count < 2:
        raise ValueError("duplicate feasibility needs a replay pair and a different-rally negative")
    return {
        "status": "supported-for-follow-up",
        "reviewer_adjudication": DUPLICATE_REVIEWER_ADJUDICATION,
        "replay_intervals": len(replays),
        "eligible_preceding_live_source_relations": eligible_count,
        "excluded_long_replay_montage": {
            "start_frame": excluded_long_montage[0],
            "end_frame": excluded_long_montage[1],
        },
        "same_video_gt_rallies": gt_rally_count,
        "retrieval_margin_status": "unmeasured",
        "retrieval_margin_limit": (
            "The review establishes interval-level source relations but does not annotate exact "
            "live-source frame pairs. A retrieval margin remains follow-up work."
        ),
    }


def slow_motion_details(
    track: np.ndarray,
    rally_spans: Sequence[tuple[int, int]],
    fps: float,
    *,
    non_evidence: np.ndarray,
    baseline_exclude: np.ndarray,
) -> SlowMotionDetails:
    """Expose the current velocity-drop baseline without changing its rule."""
    speed = compute_speed(track)
    unmeasured_steps = non_evidence[1:] | non_evidence[:-1]
    speed[1:][unmeasured_steps] = np.nan
    in_rally = np.zeros(len(track), dtype=bool)
    for start, end in rally_spans:
        in_rally[start:end] = True
    in_rally &= ~baseline_exclude
    measured = speed[in_rally]
    if not np.isfinite(measured).any():
        raise ValueError("slow-motion baseline has no measured rally speed")
    rally_median = float(np.nanmedian(measured))
    if rally_median <= 0:
        raise ValueError("slow-motion rally median must be positive")
    resolved = resolve(BaseAnnotatorConfig(), fps)
    rolling_speed = rolling_nanmedian(speed, resolved.constants.rest_window)
    threshold = SLOWMO_SPEED_FRAC * rally_median
    signal = (
        (track[:, 2] == 1)
        & (rolling_speed >= resolved.constants.rest_speed)
        & (rolling_speed < threshold)
    )
    return SlowMotionDetails(signal, speed, rolling_speed, rally_median, threshold)


def slow_motion_rows(
    details: SlowMotionDetails,
    truth: np.ndarray,
    inpaint_codes: np.ndarray,
    track: np.ndarray,
) -> list[dict[str, Any]]:
    """Stratify the unchanged slow-motion signal by truth and inpaint grade."""
    rows: list[dict[str, Any]] = []
    for scene_truth in TRUTH_ORDER:
        for code, name in sorted(CODE_NAMES.items()):
            selected = (truth == scene_truth.value) & (inpaint_codes == code)
            frames = int(selected.sum())
            visible = int(np.count_nonzero(selected & (track[:, 2] == 1)))
            measured_speed = int(np.count_nonzero(selected & np.isfinite(details.speed)))
            slow = int(np.count_nonzero(selected & details.signal))
            rolling_values = details.rolling_speed[selected & np.isfinite(details.rolling_speed)]
            rows.append({
                "truth": scene_truth.value,
                "inpaint_code": code,
                "inpaint_name": name,
                "frames": frames,
                "visible_frames": visible,
                "measured_speed_frames": measured_speed,
                "slow_signal_frames": slow,
                "slow_signal_rate": _ratio(slow, frames),
                "rolling_speed_median": (
                    float(np.median(rolling_values)) if len(rolling_values) else None
                ),
            })
    return rows


def bootstrap_spans(track: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """Reproduce the current unmasked span pass used to build the replay mask."""
    resolved = resolve(BaseAnnotatorConfig(), fps)
    return find_rally_spans(
        track,
        resolved.thresholds,
        span_open=resolved.span_open,
        constants=resolved.constants,
        gap_state_demotion_bound=resolved.gap_state_demotion_bound,
        reentry_guard_variant=resolved.reentry_guard_variant,
        reentry_guard_buffer=resolved.reentry_guard_buffer,
        quiet_start_window=resolved.quiet_start_window,
    )


def gt_statuses(
    gt_rallies: Sequence[GtRally],
    spans: Sequence[tuple[int, int]],
    accepted_by_span: dict[int, list[int]],
    tolerance: int,
) -> list[dict[str, Any]]:
    """Describe serve coverage for every GT rally under the natural run."""
    classifications = classify_all(spans, gt_rallies)
    statuses: list[dict[str, Any]] = []
    for rally_index, (rally, (boundary, mapped_span)) in enumerate(
        zip(gt_rallies, classifications, strict=True)
    ):
        first, last = rally.extent
        overlapping = [
            span_id for span_id, (start, end) in enumerate(spans)
            if start <= last and first < end
        ]
        accepted = sorted(
            frame for span_id in overlapping for frame in accepted_by_span.get(span_id, [])
        )
        matches = greedy_match(rally.stroke_frames, accepted, tolerance)
        matched_gt = {gt_index for gt_index, _candidate_index in matches}
        serve_matched = 0 in matched_gt
        later_matched = any(index > 0 for index in matched_gt)
        statuses.append({
            "rally_index": rally_index,
            "set_id": rally.set_id,
            "rally": rally.rally,
            "serve_frame": rally.stroke_frames[0],
            "boundary": boundary.value,
            "mapped_span": mapped_span,
            "serve_matched": serve_matched,
            "later_strokes_matched": later_matched,
            "target_miss": not serve_matched and later_matched,
        })
    return statuses


def evaluate_serve_candidates(
    *,
    track: np.ndarray,
    truth: np.ndarray,
    inpaint_codes: np.ndarray,
    raw_mask: np.ndarray,
    definitive_mask: np.ndarray,
    sticky: Any,
    spans: Sequence[tuple[int, int]],
    natural_raw_frames: set[int],
    accepted_by_span: dict[int, list[int]],
    gt_rallies: Sequence[GtRally],
    statuses: Sequence[dict[str, Any]],
    fps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Measure one fixed recording-only serve-lookback candidate rule."""
    resolved = resolve(BaseAnnotatorConfig(), fps)
    tolerance = canonical_tolerance(fps)
    distance = nearest_replay_or_cutaway_distance(truth)
    gt_serve_frames = [rally.stroke_frames[0] for rally in gt_rallies]
    target_indexes = [int(row["rally_index"]) for row in statuses if row["target_miss"]]
    target_frames = [gt_serve_frames[index] for index in target_indexes]
    candidate_rows: list[dict[str, Any]] = []
    opportunity_rows: list[dict[str, Any]] = []

    for span_id, (span_start, span_end) in enumerate(spans):
        accepted = sorted(accepted_by_span.get(span_id, []))
        if not accepted:
            continue
        anchor = accepted[0]
        lookback_start = max(0, anchor - resolved.constants.serve_start_lookback_frames)
        flags = detect_contact_flags(
            track,
            lookback_start,
            anchor,
            resolved.thresholds,
            smoothing_mode=resolved.smoothing_mode,
        )
        evaluated: list[dict[str, Any]] = []
        for frame, impulse in flags:
            nearest_gt = min(
                range(len(gt_serve_frames)),
                key=lambda index: (abs(gt_serve_frames[index] - frame), index),
            )
            nearest_delta = abs(gt_serve_frames[nearest_gt] - frame)
            sticky_distance = float(sticky.distances[frame])
            visible = bool(track[frame, 2] == 1)
            code = int(inpaint_codes[frame])
            wrist_pass = bool(
                np.isfinite(sticky_distance)
                and sticky_distance <= BODY_UNIT_WRIST_THRESHOLD
            )
            evidence_pass = visible and code == NO_FLAG and wrist_pass
            policy_pass = evidence_pass and not bool(definitive_mask[frame])
            reasons = []
            if not visible:
                reasons.append("track-not-visible")
            if code != NO_FLAG:
                reasons.append(f"inpaint-code-{code}")
            if not wrist_pass:
                reasons.append("sticky-wrist-gate")
            if evidence_pass and definitive_mask[frame]:
                reasons.append("definitive-mask")
            row = {
                "span_id": span_id,
                "span_start": span_start,
                "span_end": span_end,
                "anchor_frame": anchor,
                "lookback_start": lookback_start,
                "candidate_frame": frame,
                "impulse": impulse,
                "manual_truth": str(truth[frame]),
                "inpaint_code": code,
                "inpaint_name": CODE_NAMES[code],
                "track_visible": visible,
                "sticky_analysed": bool(sticky.analysed[frame]),
                "sticky_distance_bh": sticky_distance if np.isfinite(sticky_distance) else None,
                "sticky_top_pick": int(sticky.picks[frame, 0]),
                "sticky_bottom_pick": int(sticky.picks[frame, 1]),
                "wrist_gate_pass": wrist_pass,
                "evidence_pass": evidence_pass,
                "definitive_mask": bool(definitive_mask[frame]),
                "policy_pass": policy_pass,
                "selected_evidence": False,
                "selected_policy": False,
                "reject_reasons": ";".join(reasons),
                "nearest_replay_or_cutaway_distance": int(distance[frame]),
                "nearest_gt_rally_index": nearest_gt,
                "nearest_gt_serve_frame": gt_serve_frames[nearest_gt],
                "nearest_gt_serve_delta": nearest_delta,
                "tolerance_frames": tolerance,
                "gt_serve_match": nearest_delta <= tolerance,
                "nearest_gt_status": (
                    "target-miss" if statuses[nearest_gt]["target_miss"]
                    else "serve-covered" if statuses[nearest_gt]["serve_matched"]
                    else "whole-rally-or-unresolved-miss"
                ),
                "natural_raw_contact": frame in natural_raw_frames,
                "natural_accepted_contact": frame in accepted,
                "raw_mask": bool(raw_mask[frame]),
            }
            evaluated.append(row)

        evidence_rows = [row for row in evaluated if row["evidence_pass"]]
        policy_rows = [row for row in evaluated if row["policy_pass"]]
        selected_evidence = _select_candidate(evidence_rows)
        selected_policy = _select_candidate(policy_rows)
        if selected_evidence is not None:
            selected_evidence["selected_evidence"] = True
        if selected_policy is not None:
            selected_policy["selected_policy"] = True
        candidate_rows.extend(evaluated)
        opportunity_rows.append({
            "span_id": span_id,
            "span_start": span_start,
            "span_end": span_end,
            "anchor_frame": anchor,
            "lookback_start": lookback_start,
            "candidate_count": len(evaluated),
            "evidence_pass_count": len(evidence_rows),
            "policy_pass_count": len(policy_rows),
            "selected_evidence_frame": (
                selected_evidence["candidate_frame"] if selected_evidence is not None else None
            ),
            "selected_policy_frame": (
                selected_policy["candidate_frame"] if selected_policy is not None else None
            ),
        })

    evidence_selected = [int(row["candidate_frame"]) for row in candidate_rows if row["selected_evidence"]]
    policy_selected = [int(row["candidate_frame"]) for row in candidate_rows if row["selected_policy"]]
    summary = {
        "candidate_rule": {
            "lookback_frames": resolved.constants.serve_start_lookback_frames,
            "gt_serve_tolerance_frames": tolerance,
            "wrist_threshold_body_heights": BODY_UNIT_WRIST_THRESHOLD,
            "requires_visible_track": True,
            "requires_inpaint_code": NO_FLAG,
            "definitive_mask_policy": "blocked",
            "selection": "largest impulse, then earliest frame",
        },
        "n_opportunities": len(opportunity_rows),
        "n_candidate_rows": len(candidate_rows),
        "target_serve_misses": len(target_frames),
        "evidence_only": _trigger_metrics(evidence_selected, target_frames, tolerance, truth),
        "current_mask_policy": _trigger_metrics(policy_selected, target_frames, tolerance, truth),
    }
    return candidate_rows, opportunity_rows, summary


def _select_candidate(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    return min(rows, key=lambda row: (-float(row["impulse"]), int(row["candidate_frame"])), default=None)


def _trigger_metrics(
    selected_frames: Sequence[int],
    target_frames: Sequence[int],
    tolerance: int,
    truth: np.ndarray,
) -> dict[str, Any]:
    matches = greedy_match(target_frames, selected_frames, tolerance)
    matched_candidates = {candidate_index for _gt_index, candidate_index in matches}
    by_truth = {
        scene_truth.value: sum(truth[frame] == scene_truth.value for frame in selected_frames)
        for scene_truth in TRUTH_ORDER
    }
    tp = len(matches)
    fp = len(selected_frames) - len(matched_candidates)
    fn = len(target_frames) - tp
    return {
        "selected_triggers": len(selected_frames),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "selected_by_manual_truth": by_truth,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _gt_mask_loss(mask: np.ndarray, rallies: Sequence[GtRally]) -> dict[str, int]:
    gt_frames = np.zeros(len(mask), dtype=bool)
    for rally in rallies:
        first, last = rally.extent
        gt_frames[first:last + 1] = True
    return {
        "gt_extent_frames": int(gt_frames.sum()),
        "masked_gt_extent_frames": int(np.count_nonzero(mask & gt_frames)),
    }


def _git_state() -> dict[str, str | bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True,
    ).stdout
    return {"sha": sha, "dirty": bool(status)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(path)


def _write_csv_gz(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    expected_keys = set(columns)
    serialised: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if set(row) != expected_keys:
            missing = sorted(expected_keys - set(row))
            extra = sorted(set(row) - expected_keys)
            raise ValueError(f"CSV row {index} keys differ: missing={missing}, extra={extra}")
        serialised.append({column: str(_csv_value(row[column])) for column in columns})
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_handle:
                writer = csv.DictWriter(
                    text_handle,
                    fieldnames=list(columns),
                    extrasaction="raise",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(serialised)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        reloaded = list(reader)
    if tuple(reader.fieldnames or ()) != tuple(columns) or reloaded != serialised:
        raise RuntimeError(f"gzip CSV reload changed schema or values: {path}")


def _write_json_gz(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            compressed.write(payload)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        if json.load(handle) != value:
            raise RuntimeError(f"gzip JSON reload changed value: {path}")


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _build_report(summary: dict[str, Any]) -> str:
    mask = summary["replay_mask"]["detectors"]["duration_filtered"]
    components = summary["replay_mask"]["detectors"]
    raw_union = components["raw_union"]
    e2e_definitive = components["e2e_definitive"]
    gt_loss = summary["replay_mask"]["gt_extent_loss"]["e2e_definitive"]
    profile = summary["fixture_profile"]
    duplicate = summary["replay_duplicate_margin"]
    serve = summary["serve_lookback"]["current_mask_policy"]
    serve_truth = serve["selected_by_manual_truth"]
    return "\n".join([
        "# sset_01 replay and serve behaviour measurement",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        f"Fixture profile: `{profile['name']}` from source commit "
        f"`{profile['source_commit'] or 'unrecorded'}`.",
        "",
        "## Replay mask",
        "",
        f"The fresh current union differs from the pinned replacement mask on "
        f"{summary['replay_mask']['fresh_union_pinned_raw_diff_frames']:,} frames. The raw union "
        f"flags {raw_union['flagged_frames']:,} of {raw_union['frames']:,} scored frames. "
        f"Duration filtering leaves {mask['flagged_frames']:,} flagged frames, with precision "
        f"{_fmt(mask['precision'])} and recall {_fmt(mask['recall'])}. The e2e court-invalid "
        f"union flags {e2e_definitive['flagged_frames']:,} scored frames.",
        "",
        f"Court absence contributes {components['court_absence']['flagged_frames']:,} flagged "
        f"frames. Perspective shift contributes {components['perspective_shift']['flagged_frames']:,}, "
        f"and velocity drop contributes {components['velocity_drop']['flagged_frames']:,}.",
        "",
        f"The e2e mask covers {gt_loss['masked_gt_extent_frames']:,} of "
        f"{gt_loss['gt_extent_frames']:,} GT-rally extent frames.",
        "",
        "## Slow motion",
        "",
        f"The unchanged velocity signal uses a rally-speed median of "
        f"{summary['slow_motion']['rally_speed_median']:.8f} and threshold "
        f"{summary['slow_motion']['slow_threshold']:.8f}. It flags "
        f"{summary['slow_motion']['signal_frames']:,} frames.",
        "",
        "## Replay duplicate margin",
        "",
        f"Supported for follow-up: {duplicate['eligible_preceding_live_source_relations']} "
        f"short replay intervals have a human-adjudicated immediately preceding live source. "
        f"The long replay montage `[{duplicate['excluded_long_replay_montage']['start_frame']}, "
        f"{duplicate['excluded_long_replay_montage']['end_frame']})` is excluded by human "
        f"adjudication. "
        f"{duplicate['same_video_gt_rallies']} GT rallies provide different-rally negatives.",
        "",
        f"Retrieval margin unmeasured: {duplicate['retrieval_margin_limit']}",
        "",
        "## Serve lookback",
        "",
        f"The current mask-policy candidate records {serve['true_positives']} true positives, "
        f"{serve['false_positives']} false positives, and {serve['false_negatives']} false negatives "
        f"across {summary['serve_lookback']['target_serve_misses']} target serve misses. Its precision "
        f"is {_fmt(serve['precision'])} and recall is {_fmt(serve['recall'])}.",
        "",
        f"The selected trigger frames contain {serve_truth['live']} `live`, "
        f"{serve_truth['live-non-standard']} `live-non-standard`, "
        f"{serve_truth['replay']} `replay`, {serve_truth['cutaway']} `cutaway`, and "
        f"{serve_truth['other']} `other` labels. The evidence-only and current mask-policy "
        f"counts are {'the same' if serve == summary['serve_lookback']['evidence_only'] else 'different'}.",
        "",
        "These results describe one labelled video. They do not authorise a production change.",
        "",
    ])


def _fmt(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def run_measurement(
    labels_path: Path,
    output_root: Path,
    profile_name: str,
) -> dict[str, Any]:
    """Run the four bounded studies and write their reload-checked outputs."""
    profile = FIXTURE_PROFILES[profile_name]
    fixture = profile.fixture
    inputs = build_run_video_inputs(fixture)
    track, bboxes, scores, kps, ndet = inputs.positional
    keyword = dict(inputs.keyword)
    metadata = VideoMetadata(fixture.name, fixture.fps, len(track))
    intervals = read_label_csv(labels_path)
    validate_partition(intervals, expected_metadata=metadata)
    truth = expand_truth(intervals, metadata)

    inpaint_codes = np.asarray(keyword["inpaint_codes"])
    court_present = np.asarray(keyword["court_present"])
    homography_rows = keyword["homography_rows"]
    row_records = homography_rows.to_dict("records") if hasattr(homography_rows, "to_dict") else homography_rows
    bootstrap = bootstrap_spans(track, fixture.fps)
    non_evidence = np.isin(inpaint_codes, tuple(sorted(BaseAnnotatorConfig().rejected_grades)))
    court = court_absence_signal(court_present, len(track), fixture.fps)
    perspective = perspective_shift_signal(row_records, len(track))
    velocity = velocity_drop_signal(
        track,
        bootstrap,
        len(track),
        fixture.fps,
        non_evidence=non_evidence,
        baseline_exclude=court | perspective,
    )
    raw_union = court | perspective | velocity
    committed_mask = np.asarray(keyword["raw_exclusion_mask"])
    pinned_mask_diff_frames = int(np.count_nonzero(raw_union != committed_mask))
    resolved = resolve(BaseAnnotatorConfig(), fixture.fps)
    duration_filtered = filter_short_exclusion_runs(
        raw_union, resolved.constants.replay_mask_min_frames,
    )
    e2e_definitive = duration_filtered | ~court_present

    capture = RunCapture()
    keyword["raw_exclusion_mask"] = raw_union
    keyword["court_invalid_is_excluded"] = True
    keyword["capture"] = capture
    result = run_video(*inputs.positional, **keyword)
    if capture.raw_exclusion_mask is None or capture.definitive_exclusion_mask is None:
        raise RuntimeError("run_video did not capture both replay masks")
    if not np.array_equal(capture.raw_exclusion_mask, raw_union):
        raise RuntimeError("run_video raw mask differs from the fresh replay union")
    if not np.array_equal(capture.definitive_exclusion_mask, e2e_definitive):
        raise RuntimeError("run_video definitive mask differs from the e2e court-invalid union")

    details = slow_motion_details(
        track,
        bootstrap,
        fixture.fps,
        non_evidence=non_evidence,
        baseline_exclude=court | perspective,
    )
    if not np.array_equal(details.signal, velocity):
        raise RuntimeError("slow-motion diagnostic reconstruction differs from velocity_drop_signal")

    detectors = {
        "court_absence": court,
        "perspective_shift": perspective,
        "velocity_drop": velocity,
        "raw_union": raw_union,
        "duration_filtered": duration_filtered,
        "e2e_definitive": e2e_definitive,
    }
    mask_rows = mask_metric_rows(detectors, truth)
    slow_rows = slow_motion_rows(details, truth, inpaint_codes, track)

    homography_for_sticky = row_records
    segments = tracker_segments(homography_for_sticky, court_present, len(track))
    sticky = build_sticky_result(
        track,
        segments,
        bboxes,
        scores,
        kps,
        ndet,
        str(fixture.video_id),
        keyword["gate_court_info"],
        keyword["gate_resolution_table"],
        fixture.resolution,
        resolved.constants.body_unit_half_window,
    )
    accepted_by_span = {
        int(span_id): sorted(map(int, frames))
        for span_id, frames in result.filtered_by_rally.items()
    }
    raw_frames = {int(contact.contact_frame) for contact in result.contacts}
    gt_rallies = load_gt_rallies(inputs.master, fixture.video_id)
    statuses = gt_statuses(
        gt_rallies,
        result.spans,
        accepted_by_span,
        canonical_tolerance(fixture.fps),
    )
    candidate_rows, opportunity_rows, serve_summary = evaluate_serve_candidates(
        track=track,
        truth=truth,
        inpaint_codes=inpaint_codes,
        raw_mask=raw_union,
        definitive_mask=e2e_definitive,
        sticky=sticky,
        spans=result.spans,
        natural_raw_frames=raw_frames,
        accepted_by_span=accepted_by_span,
        gt_rallies=gt_rallies,
        statuses=statuses,
        fps=fixture.fps,
    )

    binary_by_detector = {
        name: binary_metrics(values, truth) for name, values in detectors.items()
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_code": {
            "path": _display_path(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
            "git": _git_state(),
        },
        "fixture": fixture.name,
        "fixture_profile": {
            "name": profile_name,
            "source": profile.source,
            "source_commit": profile.source_commit,
        },
        "fps": fixture.fps,
        "frame_count": len(track),
        "labels": {
            "path": _display_path(labels_path),
            "sha256": _sha256(labels_path),
            "interval_count": len(intervals),
        },
        "fixture_inputs": [
            {"path": pin.path.as_posix(), "md5": pin.md5} for pin in fixture.files
        ],
        "replay_mask": {
            "replay_mask_min_frames": resolved.constants.replay_mask_min_frames,
            "fresh_union_matches_pinned_raw_mask": pinned_mask_diff_frames == 0,
            "fresh_union_pinned_raw_diff_frames": pinned_mask_diff_frames,
            "detectors": binary_by_detector,
            "gt_extent_loss": {
                name: _gt_mask_loss(values, gt_rallies) for name, values in detectors.items()
            },
        },
        "slow_motion": {
            "slowmo_speed_fraction": SLOWMO_SPEED_FRAC,
            "rally_speed_median": details.rally_median,
            "slow_threshold": details.slow_threshold,
            "signal_frames": int(details.signal.sum()),
        },
        "replay_duplicate_margin": replay_duplicate_feasibility(intervals, len(gt_rallies)),
        "serve_lookback": serve_summary,
        "natural_run": {
            "spans": len(result.spans),
            "raw_contacts": len(result.contacts),
            "accepted_contacts": len(result.filtered_contacts),
            "gt_rallies": len(gt_rallies),
        },
        "limits": [
            "This is one labelled ShuttleSet video and does not establish broadcast-wide performance.",
            "The serve candidate is recording-only and does not modify spans or contacts.",
            "The natural run uses the e2e court-invalid mask policy; the duration-only mask is also reported.",
        ],
    }

    output_root.mkdir(parents=True, exist_ok=False)
    _write_csv_gz(output_root / "mask_metrics.csv.gz", MASK_COLUMNS, mask_rows)
    _write_csv_gz(output_root / "slow_motion_strata.csv.gz", SLOW_COLUMNS, slow_rows)
    _write_csv_gz(output_root / "serve_candidates.csv.gz", CANDIDATE_COLUMNS, candidate_rows)
    _write_csv_gz(output_root / "serve_opportunities.csv.gz", OPPORTUNITY_COLUMNS, opportunity_rows)
    _write_json_gz(output_root / "summary.json.gz", summary)
    report = _build_report(summary)
    (output_root / "report.md").write_text(report, encoding="utf-8")
    if (output_root / "report.md").read_text(encoding="utf-8") != report:
        raise RuntimeError("report reload changed text")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument(
        "--fixture-profile",
        choices=tuple(FIXTURE_PROFILES),
        required=True,
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    fixture_root = args.fixture_root.expanduser().resolve()
    if not fixture_root.is_dir():
        raise FileNotFoundError(f"fixture root not found: {fixture_root}")
    os.environ["ANNOTATOR_FIXTURES_ROOT"] = str(fixture_root)
    if fixtures_root() != fixture_root:
        raise RuntimeError("configured fixture root did not resolve to the requested path")
    labels_path = args.labels.expanduser().resolve()
    output_root = args.out.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output path already exists: {output_root}")
    summary = run_measurement(labels_path, output_root, args.fixture_profile)
    print(json.dumps({
        "output": str(output_root),
        "replay_mask": summary["replay_mask"]["detectors"]["duration_filtered"],
        "serve_lookback": summary["serve_lookback"]["current_mask_policy"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
