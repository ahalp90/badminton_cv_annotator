"""Measure current serve-prepend evidence without changing annotator output.

Run from the repository root with the external fixture root configured::

    ANNOTATOR_FIXTURES_ROOT=/path/to/fixture/root PYTHONPATH=src \
        python docs/scraper_pipeline/serve_prepend_lookback/measure_serve_prepend_lookback.py \
        --out docs/scraper_pipeline/serve_prepend_lookback/data/run-YYYYMMDD-HHMMSS

The script uses the maintained ``FIXTURES`` and ``build_run_video_inputs`` seams. It runs the
normal committed-mask chain and, by default, the existing ``no_replay`` variant. That variant passes a
per-frame ``raw_exclusion_mask`` vector of false values; other processing and downstream filters remain
active. It is a measurement variant; no contacts or spans are fed back into production.

CSV and JSON files are gzip-compressed. Compact per-rally NumPy evidence tables are stored by
the native Python ``lzma`` module with XZ/LZMA compression preset 9. Those ``.npy.xz`` files are
normal NumPy ``.npy`` streams wrapped in native XZ compression and must be reloaded through
``lzma.open`` and ``np.load``.
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
import lzma
import math
from pathlib import Path
import subprocess
from typing import Any, NamedTuple, Sequence

import numpy as np

from annotator.broadcast_timeline_labels import VideoMetadata
from annotator.calibration import serve_prepend_measurement as candidate_measurement
from annotator.calibration.fixtures import (
    FIXTURES,
    SSET_01,
    SSET_15,
    SSET_21,
    Fixture,
    fixtures_root,
)
from annotator.calibration.gt_scoring import build_run_video_inputs, canonical_tolerance
from annotator.calibration.scoring import (
    RallyBoundary,
    classify_all,
    greedy_match,
    load_gt_rallies,
)
from annotator.fps_constants import scale_for_fps
from annotator.inpaint_guard import DEGRADED, FABRICATED, NO_FLAG, SUSPECT_FLAT
from annotator.run_video import RunCapture, run_video


WINDOW_SECONDS = 1.0
CLEAN_RUN_BASE30_FRAMES = 5
MASK_MODES = ("committed", "no_replay")
PROFILE_SOURCE_COMMIT = "189c5af58e45d23ae827dde516924194eb238e18"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS_DIR = SCRIPT_DIR.parent / "broadcast_nonstandard_camera_id/data"
POSE_BANDS = candidate_measurement.POSE_BANDS
CANDIDATE_COLUMNS = candidate_measurement.CANDIDATE_COLUMNS
OPPORTUNITY_COLUMNS = candidate_measurement.OPPORTUNITY_COLUMNS
expand_truth = candidate_measurement.expand_truth
central_pose_evidence = candidate_measurement.central_pose_evidence
evaluate_serve_candidates = candidate_measurement.evaluate_serve_candidates
run_contact_injection_counterfactual = (
    candidate_measurement.run_contact_injection_counterfactual
)


class FixtureSetProfile(NamedTuple):
    fixtures: tuple[Fixture, ...]
    source: str
    source_commit: str | None


def _release_fixture(
    fixture: Fixture,
    *,
    dead_mask: str,
    court_present: str,
    scene_rows: str,
) -> Fixture:
    return replace(
        fixture,
        digests=replace(
            fixture.digests,
            dead_mask=dead_mask,
            court_present=court_present,
            scene_rows=scene_rows,
        ),
    )


UNE_189C5AF_STATIC_STRIDE8 = (
    _release_fixture(
        SSET_01,
        dead_mask="70a2a4e9cbd7c6c02b497b468682c462",
        court_present="65f4e28d0556c0e5422f569ad4b69fac",
        scene_rows="7d781f33e29804ef8363bbbd1b60d772",
    ),
    _release_fixture(
        SSET_15,
        dead_mask="281562f7933f1fd24301bdba48bb26b9",
        court_present="9b3ab966ef6d357a70ec4541410046a5",
        scene_rows="15f3d6751e75f3c68bab520186096c25",
    ),
    _release_fixture(
        SSET_21,
        dead_mask="4d2dfde901ccb5253a54542e60585d71",
        court_present="0c51f0e894c3addfc576c140805fd96f",
        scene_rows="06386f0b6d604819c18b3dd1c3097bef",
    ),
)

FIXTURE_PROFILES = {
    "historical-calibration": FixtureSetProfile(
        FIXTURES,
        "maintained calibration pins",
        None,
    ),
    "une-189c5af-static-stride8": FixtureSetProfile(
        UNE_189C5AF_STATIC_STRIDE8,
        "UNE sset_measure_189c5af static_shuttleset_homography stride-8",
        PROFILE_SOURCE_COMMIT,
    ),
}

ARRAY_DTYPE = np.dtype([
    ("rally_index", "<i4"),
    ("gt_serve_frame", "<i8"),
    ("span_category_code", "<i1"),
    ("n_gt_strokes", "<i4"),
    ("n_matched_strokes", "<i4"),
    ("serve_matched", "<?"),
    ("later_strokes_matched", "<i4"),
    ("n_raw_candidates_in_window", "<i4"),
    ("n_accepted_candidates_in_window", "<i4"),
    ("nearest_raw_delta", "<i4"),
    ("nearest_accepted_delta", "<i4"),
    ("window_visible_fraction", "<f4"),
    ("window_clean_fraction", "<f4"),
    ("window_fabricated_fraction", "<f4"),
    ("window_suspect_flat_fraction", "<f4"),
    ("window_degraded_fraction", "<f4"),
    ("window_longest_clean_run", "<i4"),
    ("window_raw_mask_fraction", "<f4"),
    ("window_believed_mask_fraction", "<f4"),
    ("serve_on_believed_mask", "<?"),
    ("court_present_at_serve", "<?"),
    ("n_pose_detections", "<i2"),
    ("n_valid_bbox_detections", "<i2"),
    ("n_valid_wrist_detections", "<i2"),
    ("max_bbox_area", "<f4"),
])


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _fraction(values: np.ndarray) -> float:
    return round(float(values.mean()), 6) if len(values) else 0.0


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _segment_stats(
    track: np.ndarray,
    codes: np.ndarray,
    raw_mask: np.ndarray,
    believed_mask: np.ndarray,
    court_present: np.ndarray,
    start: int,
    end: int,
) -> dict[str, Any]:
    """Return compact evidence for the half-open frame interval ``[start, end)``."""
    start = max(0, start)
    end = min(len(track), end)
    if end <= start:
        return {
            "frame_count": 0,
            "visible_frames": 0,
            "clean_visible_frames": 0,
            "visible_fraction": 0.0,
            "clean_fraction": 0.0,
            "fabricated_fraction": 0.0,
            "suspect_flat_fraction": 0.0,
            "degraded_fraction": 0.0,
            "longest_clean_run": 0,
            "raw_mask_fraction": 0.0,
            "believed_mask_fraction": 0.0,
            "believed_mask_frames": 0,
            "court_present_fraction": 0.0,
        }

    track_slice = track[start:end]
    code_slice = codes[start:end]
    raw_slice = raw_mask[start:end]
    believed_slice = believed_mask[start:end]
    court_slice = court_present[start:end]
    visible = track_slice[:, 2] == 1
    clean_visible = visible & (code_slice == NO_FLAG)
    return {
        "frame_count": end - start,
        "visible_frames": int(visible.sum()),
        "clean_visible_frames": int(clean_visible.sum()),
        "visible_fraction": _fraction(visible),
        "clean_fraction": _fraction(clean_visible),
        "fabricated_fraction": _fraction(code_slice == FABRICATED),
        "suspect_flat_fraction": _fraction(code_slice == SUSPECT_FLAT),
        "degraded_fraction": _fraction(code_slice == DEGRADED),
        "longest_clean_run": _longest_true_run(clean_visible),
        "raw_mask_fraction": _fraction(raw_slice),
        "believed_mask_fraction": _fraction(believed_slice),
        "believed_mask_frames": int(believed_slice.sum()),
        "court_present_fraction": _fraction(court_slice),
    }


def _pose_stats(
    bboxes: np.ndarray, kps: np.ndarray, ndet: np.ndarray, frame: int,
) -> dict[str, Any]:
    """Summarise raw pose availability at one frame without choosing a person."""
    n_pose = max(0, min(int(ndet[frame]), bboxes.shape[1]))
    frame_boxes = bboxes[frame, :n_pose]
    finite_boxes = np.isfinite(frame_boxes).all(axis=1)
    widths = frame_boxes[:, 2] - frame_boxes[:, 0]
    heights = frame_boxes[:, 3] - frame_boxes[:, 1]
    valid_boxes = finite_boxes & (widths > 0) & (heights > 0)
    areas = widths * heights
    max_area = float(areas[valid_boxes].max()) if valid_boxes.any() else -1.0
    if kps.shape[2] >= 11:
        frame_kps = kps[frame, :n_pose, [9, 10], :]
        valid_wrists = np.isfinite(frame_kps).all(axis=(1, 2))
    else:
        valid_wrists = np.zeros(n_pose, dtype=bool)
    return {
        "n_pose_detections": n_pose,
        "n_valid_bbox_detections": int(valid_boxes.sum()),
        "n_valid_wrist_detections": int(valid_wrists.sum()),
        "max_bbox_area": round(max_area, 3),
    }


def _nearest(frames: list[int], target: int) -> tuple[int | None, int | None]:
    if not frames:
        return None, None
    frame = min(frames, key=lambda value: (abs(value - target), value))
    return frame, abs(frame - target)


def _overlapping_span_ids(extent: tuple[int, int], spans: list[tuple[int, int]]) -> list[int]:
    first, last = extent
    return [span_id for span_id, (start, end) in enumerate(spans) if start <= last and first < end]


def _span_code(category: RallyBoundary) -> int:
    return {
        RallyBoundary.COVERED: 0,
        RallyBoundary.SPLIT: 1,
        RallyBoundary.MISSED: 2,
    }[category]


def _status(serve_matched: bool, n_matched: int) -> str:
    if serve_matched:
        return "serve_matched"
    if n_matched:
        return "serve_missed_later_strokes_matched"
    return "whole_rally_missed"


def _measure_variant(fixture: Fixture, mask_mode: str) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    inputs = build_run_video_inputs(fixture)
    track, bboxes, _scores, kps, ndet = inputs.positional
    keyword = dict(inputs.keyword)
    if mask_mode == "no_replay":
        keyword["raw_exclusion_mask"] = np.zeros(len(track), dtype=bool)
    capture = RunCapture()
    keyword["capture"] = capture
    keyword["rejection_diagnostics"] = []
    result = run_video(*inputs.positional, **keyword)

    if capture.raw_exclusion_mask is None or capture.definitive_exclusion_mask is None:
        raise RuntimeError(f"{fixture.name}/{mask_mode}: run did not capture replay masks")
    raw_mask = capture.raw_exclusion_mask
    believed_mask = capture.definitive_exclusion_mask
    codes = np.asarray(inputs.keyword["inpaint_codes"])
    court_present = np.asarray(inputs.keyword["court_present"])
    gt_rallies = load_gt_rallies(inputs.master, fixture.video_id)
    classifications = classify_all(result.spans, gt_rallies)
    tolerance = canonical_tolerance(fixture.fps)
    constants = scale_for_fps(fixture.fps)
    clean_run_min_frames = max(
        1, math.floor(CLEAN_RUN_BASE30_FRAMES * fixture.fps / 30.0 + 0.5),
    )
    raw_frames = sorted(int(contact.contact_frame) for contact in result.contacts)
    accepted_frames = sorted(int(contact.contact_frame) for contact in result.filtered_contacts)
    rows: list[dict[str, Any]] = []

    for rally_index, (rally, (category, span_id)) in enumerate(zip(gt_rallies, classifications)):
        overlapping = _overlapping_span_ids(rally.extent, result.spans)
        accepted_for_rally = [
            int(contact.contact_frame)
            for contact in result.filtered_contacts
            if contact.rally_id in overlapping
        ]
        matched = greedy_match(rally.stroke_frames, accepted_for_rally, tolerance)
        matched_gt = {gt_index for gt_index, _ in matched}
        serve_matched = 0 in matched_gt
        later_strokes_matched = sum(gt_index > 0 for gt_index in matched_gt)
        serve_frame = rally.stroke_frames[0]
        first_accepted_frame, _ = _nearest(accepted_for_rally, serve_frame)
        lookback_anchor = first_accepted_frame if first_accepted_frame is not None else serve_frame
        anchor_source = (
            "first_assigned_accepted_contact"
            if first_accepted_frame is not None else "gt_serve_fallback"
        )
        lookback_start = max(0, lookback_anchor - constants.serve_start_lookback_frames)
        window_half = int(round(WINDOW_SECONDS * fixture.fps))
        window_start = max(0, serve_frame - window_half)
        window_end = min(len(track), serve_frame + window_half + 1)
        window = _segment_stats(
            track, codes, raw_mask, believed_mask, court_present, window_start, window_end,
        )
        lookback = _segment_stats(
            track, codes, raw_mask, believed_mask, court_present, lookback_start, lookback_anchor,
        )
        raw_frame, raw_delta = _nearest(raw_frames, serve_frame)
        accepted_frame, accepted_delta = _nearest(accepted_frames, serve_frame)
        raw_in_window = [frame for frame in raw_frames if window_start <= frame < window_end]
        accepted_in_window = [frame for frame in accepted_frames if window_start <= frame < window_end]
        pose = _pose_stats(bboxes, kps, ndet, serve_frame)
        span_start = result.spans[span_id][0] if span_id is not None else None
        span_end = result.spans[span_id][1] if span_id is not None else None
        row = {
            "mask_mode": mask_mode,
            "rally_index": rally_index,
            "set_id": rally.set_id,
            "rally_number": rally.rally,
            "status": _status(serve_matched, len(matched)),
            "span_category": category.value,
            "span_id": span_id if span_id is not None else "",
            "span_start": span_start if span_start is not None else "",
            "span_end": span_end if span_end is not None else "",
            "gt_first_contact": rally.extent[0],
            "gt_last_contact": rally.extent[1],
            "gt_serve_frame": serve_frame,
            "n_gt_strokes": rally.n_strokes,
            "n_matched_strokes": len(matched),
            "serve_matched": serve_matched,
            "later_strokes_matched": later_strokes_matched,
            "tolerance_frames": tolerance,
            "nearest_raw_frame": raw_frame if raw_frame is not None else "",
            "nearest_raw_delta": raw_delta if raw_delta is not None else "",
            "nearest_accepted_frame": accepted_frame if accepted_frame is not None else "",
            "nearest_accepted_delta": accepted_delta if accepted_delta is not None else "",
            "n_raw_candidates_in_window": len(raw_in_window),
            "n_accepted_candidates_in_window": len(accepted_in_window),
            "window_start": window_start,
            "window_end_exclusive": window_end,
            "window_frame_count": window["frame_count"],
            "window_visible_frames": window["visible_frames"],
            "window_clean_visible_frames": window["clean_visible_frames"],
            "window_visible_fraction": window["visible_fraction"],
            "window_clean_fraction": window["clean_fraction"],
            "window_fabricated_fraction": window["fabricated_fraction"],
            "window_suspect_flat_fraction": window["suspect_flat_fraction"],
            "window_degraded_fraction": window["degraded_fraction"],
            "window_longest_clean_run": window["longest_clean_run"],
            "window_raw_mask_fraction": window["raw_mask_fraction"],
            "window_believed_mask_fraction": window["believed_mask_fraction"],
            "window_believed_mask_frames": window["believed_mask_frames"],
            "window_court_present_fraction": window["court_present_fraction"],
            "lookback_anchor_frame": lookback_anchor,
            "lookback_anchor_source": anchor_source,
            "lookback_start": lookback_start,
            "lookback_end_exclusive": lookback_anchor,
            "lookback_frame_count": lookback["frame_count"],
            "lookback_visible_frames": lookback["visible_frames"],
            "lookback_clean_visible_frames": lookback["clean_visible_frames"],
            "lookback_longest_clean_run": lookback["longest_clean_run"],
            "lookback_believed_mask_frames": lookback["believed_mask_frames"],
            "serve_on_raw_mask": bool(raw_mask[serve_frame]),
            "serve_on_believed_mask": bool(believed_mask[serve_frame]),
            "court_present_at_serve": bool(court_present[serve_frame]),
            "replay_mask_min_frames": constants.replay_mask_min_frames,
            "serve_start_lookback_frames": constants.serve_start_lookback_frames,
            "clean_run_min_frames": clean_run_min_frames,
            **pose,
        }
        rows.append(row)

    metadata = {
        "fixture": fixture.name,
        "video_id": fixture.video_id,
        "fps": fixture.fps,
        "n_frames": len(track),
        "mask_mode": mask_mode,
        "raw_mask_true_frames": int(raw_mask.sum()),
        "believed_mask_true_frames": int(believed_mask.sum()),
        "n_spans": len(result.spans),
        "n_raw_contacts": len(result.contacts),
        "n_accepted_contacts": len(result.filtered_contacts),
        "replay_mask_min_frames": constants.replay_mask_min_frames,
        "serve_start_lookback_frames": constants.serve_start_lookback_frames,
        "clean_run_min_frames": clean_run_min_frames,
    }
    array = np.empty(len(rows), dtype=ARRAY_DTYPE)
    for index, row in enumerate(rows):
        array[index] = (
            row["rally_index"], row["gt_serve_frame"], _span_code(RallyBoundary(row["span_category"])),
            row["n_gt_strokes"], row["n_matched_strokes"], row["serve_matched"],
            row["later_strokes_matched"], row["n_raw_candidates_in_window"],
            row["n_accepted_candidates_in_window"],
            -1 if row["nearest_raw_delta"] == "" else row["nearest_raw_delta"],
            -1 if row["nearest_accepted_delta"] == "" else row["nearest_accepted_delta"],
            row["window_visible_fraction"],
            row["window_clean_fraction"], row["window_fabricated_fraction"],
            row["window_suspect_flat_fraction"], row["window_degraded_fraction"],
            row["window_longest_clean_run"], row["window_raw_mask_fraction"],
            row["window_believed_mask_fraction"], row["serve_on_believed_mask"],
            row["court_present_at_serve"], row["n_pose_detections"],
            row["n_valid_bbox_detections"], row["n_valid_wrist_detections"],
            row["max_bbox_area"],
        )
    return rows, array, metadata


def measure_candidate_fixture(
    fixture: Fixture,
    labels_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the committed baseline and both recording-only candidate variants."""
    inputs = build_run_video_inputs(fixture)
    track, _bboxes, _scores, _kps, _ndet = inputs.positional
    metadata = VideoMetadata(fixture.name, fixture.fps, len(track))
    truth, interval_count = expand_truth(labels_path, metadata)
    keyword = dict(inputs.keyword)
    capture = RunCapture()
    keyword["capture"] = capture
    natural_result = run_video(*inputs.positional, **keyword)
    if capture.raw_exclusion_mask is None or capture.definitive_exclusion_mask is None:
        raise RuntimeError(f"{fixture.name}: committed run did not capture replay masks")

    gt_rallies = load_gt_rallies(inputs.master, fixture.video_id)
    tolerance = canonical_tolerance(fixture.fps)
    raw_by_span = candidate_measurement.contacts_by_span(natural_result.contacts)
    accepted_by_span = {
        int(span_id): sorted(map(int, frames))
        for span_id, frames in natural_result.filtered_by_rally.items()
    }
    statuses = candidate_measurement.gt_statuses(
        gt_rallies,
        natural_result.spans,
        accepted_by_span,
        tolerance,
    )
    inpaint_codes = np.asarray(inputs.keyword["inpaint_codes"])
    court_present = np.asarray(inputs.keyword["court_present"])
    all_candidates: list[dict[str, Any]] = []
    all_opportunities: list[dict[str, Any]] = []
    band_summaries: dict[str, Any] = {}
    for band, fraction in POSE_BANDS.items():
        candidates, opportunities, summary = evaluate_serve_candidates(
            fixture=fixture,
            band=band,
            band_fraction=fraction,
            arrays=inputs.positional,
            truth=truth,
            inpaint_codes=inpaint_codes,
            court_present=court_present,
            raw_mask=capture.raw_exclusion_mask,
            definitive_mask=capture.definitive_exclusion_mask,
            spans=natural_result.spans,
            raw_by_span=raw_by_span,
            accepted_by_span=accepted_by_span,
            gt_rallies=gt_rallies,
            statuses=statuses,
        )
        summary["contact_injection"] = {
            "evidence_only_mask_exemption": run_contact_injection_counterfactual(
                inputs=inputs,
                natural_result=natural_result,
                gt_rallies=gt_rallies,
                natural_statuses=statuses,
                candidate_rows=candidates,
                tolerance=tolerance,
                selection_field="selected_evidence",
                exempt_selected_from_raw_mask=True,
            ),
            "current_mask_policy": run_contact_injection_counterfactual(
                inputs=inputs,
                natural_result=natural_result,
                gt_rallies=gt_rallies,
                natural_statuses=statuses,
                candidate_rows=candidates,
                tolerance=tolerance,
                selection_field="selected_policy",
                exempt_selected_from_raw_mask=False,
            ),
        }
        all_candidates.extend(candidates)
        all_opportunities.extend(opportunities)
        band_summaries[band] = summary

    summary = {
        "fixture": fixture.name,
        "video_id": fixture.video_id,
        "fps": fixture.fps,
        "frame_count": len(track),
        "labels": {
            "path": str(labels_path),
            "sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            "interval_count": interval_count,
        },
        "fixture_inputs": [
            {"path": pin.path.as_posix(), "md5": pin.md5}
            for pin in fixture.run_video_files
        ],
        "natural_run": {
            "spans": len(natural_result.spans),
            "raw_contacts": len(natural_result.contacts),
            "accepted_contacts": len(natural_result.filtered_contacts),
            "gt_rallies": len(gt_rallies),
            "target_serve_misses": sum(row["target_miss"] for row in statuses),
            "serve_matched": sum(row["serve_matched"] for row in statuses),
            "whole_rally_or_unresolved_misses": sum(
                not row["serve_matched"] and not row["later_strokes_matched"]
                for row in statuses
            ),
        },
        "bands": band_summaries,
    }
    return all_candidates, all_opportunities, summary


def _sum_named_counts(values: Sequence[dict[str, int]]) -> dict[str, int]:
    keys = values[0] if values else {}
    return {key: sum(value[key] for value in values) for key in keys}


def _pool_trigger(summaries: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    triggers = [value["trigger_metrics"][arm] for value in summaries]
    true_positives = sum(value["true_positives"] for value in triggers)
    false_positives = sum(value["false_positives"] for value in triggers)
    false_negatives = sum(value["false_negatives"] for value in triggers)
    return {
        "selected_triggers": sum(value["selected_triggers"] for value in triggers),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives else None
        ),
        "recall": (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives else None
        ),
        "selected_by_manual_truth": _sum_named_counts([
            value["selected_by_manual_truth"] for value in triggers
        ]),
        "selected_by_nearest_gt_status": _sum_named_counts([
            value["selected_by_nearest_gt_status"] for value in triggers
        ]),
    }


def _pool_injection(summaries: Sequence[dict[str, Any]], arm: str) -> dict[str, int]:
    injections = [value["contact_injection"][arm] for value in summaries]
    fields = (
        "selected_candidates",
        "new_selected_candidates",
        "accepted_selected_candidates",
        "exempted_raw_mask_frames",
        "target_serves_recovered",
        "target_serves_still_missed",
        "all_unmatched_serves_recovered",
        "all_unmatched_serves_still_missed",
        "accepted_contact_delta",
        "span_delta",
        "stroke_count_rows_changed",
        "next_server_rows_changed",
    )
    return {field: sum(value[field] for value in injections) for field in fields}


def _pool_candidate_summaries(per_fixture: dict[str, Any]) -> dict[str, Any]:
    pooled: dict[str, Any] = {}
    for band in POSE_BANDS:
        summaries = [value["bands"][band] for value in per_fixture.values()]
        pooled[band] = {
            "n_opportunities": sum(value["n_opportunities"] for value in summaries),
            "n_candidate_rows": sum(value["n_candidate_rows"] for value in summaries),
            "n_evidence_pass": sum(value["n_evidence_pass"] for value in summaries),
            "n_candidate_pass": sum(value["n_candidate_pass"] for value in summaries),
            "n_policy_pass": sum(value["n_policy_pass"] for value in summaries),
            "target_serve_misses": sum(value["target_serve_misses"] for value in summaries),
            "trigger_metrics": {
                arm: _pool_trigger(summaries, arm)
                for arm in ("evidence_only", "current_mask_policy")
            },
            "target_outcomes": {
                arm: _sum_named_counts([
                    value["target_outcomes"][arm] for value in summaries
                ])
                for arm in ("evidence_only", "current_mask_policy")
            },
            "contact_injection": {
                arm: _pool_injection(summaries, arm)
                for arm in ("evidence_only_mask_exemption", "current_mask_policy")
            },
        }
    return pooled


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    misses = [row for row in rows if row["status"] == "serve_missed_later_strokes_matched"]
    serve_misses = [row for row in rows if not row["serve_matched"]]
    return {
        "n_rallies": len(rows),
        "n_serve_matched": sum(bool(row["serve_matched"]) for row in rows),
        "n_serve_missed": len(serve_misses),
        "n_serve_missed_later_strokes_matched": len(misses),
        "n_whole_rally_missed": sum(row["status"] == "whole_rally_missed" for row in rows),
        "n_clean_serve_window_runs": sum(
            row["window_longest_clean_run"] >= row["clean_run_min_frames"] for row in misses
        ),
        "n_clean_lookback_runs": sum(
            row["lookback_longest_clean_run"] >= row["clean_run_min_frames"] for row in misses
        ),
        "n_serve_frames_on_believed_mask": sum(row["serve_on_believed_mask"] for row in misses),
        "n_raw_candidates_near_serve": sum(
            row["nearest_raw_delta"] != "" and row["nearest_raw_delta"] <= row["tolerance_frames"]
            for row in misses
        ),
        "n_accepted_candidates_near_serve": sum(
            row["nearest_accepted_delta"] != "" and row["nearest_accepted_delta"] <= row["tolerance_frames"]
            for row in misses
        ),
        "span_categories": {
            category: sum(row["span_category"] == category for row in rows)
            for category in ("covered", "split", "missed")
        },
    }


def _write_csv_gz(rows: list[dict[str, Any]], path: Path) -> None:
    _write_schema_csv_gz(rows, tuple(rows[0]), path)


def _write_schema_csv_gz(
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
    path: Path,
) -> None:
    expected = set(fieldnames)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"CSV row {index} keys differ: "
                f"missing={sorted(expected - set(row))}, extra={sorted(set(row) - expected)}"
            )
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(fieldnames),
                    extrasaction="raise",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(fieldnames) or sum(1 for _ in reader) != len(rows):
            raise RuntimeError(f"gzip CSV reload changed schema or row count for {path}")


def _write_json_gz(value: object, path: Path) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            compressed.write(payload)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        if json.load(handle) != value:
            raise RuntimeError(f"gzip JSON reload changed value for {path}")


def _write_array_lzma(array: np.ndarray, path: Path) -> None:
    with lzma.open(path, "wb", preset=9) as handle:
        np.save(handle, array, allow_pickle=False)
    with lzma.open(path, "rb") as handle:
        reloaded = np.load(handle, allow_pickle=False)
    if not isinstance(reloaded, np.ndarray) or reloaded.dtype != array.dtype:
        raise RuntimeError(f"native lzma reload changed array type or dtype for {path}")
    if reloaded.tobytes() != array.tobytes():
        raise RuntimeError(f"native lzma reload changed array values for {path}")


def _default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path(__file__).resolve().parent / "data" / f"serve_prepend_lookback_{stamp}"


def _git_state() -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"sha": _git_sha(), "dirty": bool(status)}


def _fmt_ratio(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def _build_candidate_report(summary: dict[str, Any]) -> str:
    analysis = summary["candidate_analysis"]
    lines = [
        "# Serve-lookback candidate measurement",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        f"Fixture profile: `{summary['fixture_profile']['name']}` from source commit "
        f"`{summary['fixture_profile']['source_commit'] or 'unrecorded'}`.",
        "",
        "This is a recording-only measurement. It does not change production output.",
        "",
        "## Results",
        "",
        "| Pose band | Arm | Target misses | Selected | Target recovered | False positives | Precision | Recall | All unmatched recovered |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for band in POSE_BANDS:
        pooled = analysis["pooled"][band]
        for arm, injection_arm in (
            ("evidence_only", "evidence_only_mask_exemption"),
            ("current_mask_policy", "current_mask_policy"),
        ):
            trigger = pooled["trigger_metrics"][arm]
            injection = pooled["contact_injection"][injection_arm]
            lines.append(
                f"| {band} | {arm} | {pooled['target_serve_misses']} | "
                f"{trigger['selected_triggers']} | {trigger['true_positives']} | "
                f"{trigger['false_positives']} | {_fmt_ratio(trigger['precision'])} | "
                f"{_fmt_ratio(trigger['recall'])} | "
                f"{injection['all_unmatched_serves_recovered']} |"
            )

    lines.extend([
        "",
        "## Per-video results",
        "",
        "| Video | Pose band | Arm | Target misses | Target recovered | False positives | All unmatched recovered |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for fixture, fixture_summary in analysis["per_fixture"].items():
        for band in POSE_BANDS:
            band_summary = fixture_summary["bands"][band]
            for arm, injection_arm in (
                ("evidence_only", "evidence_only_mask_exemption"),
                ("current_mask_policy", "current_mask_policy"),
            ):
                trigger = band_summary["trigger_metrics"][arm]
                injection = band_summary["contact_injection"][injection_arm]
                lines.append(
                    f"| {fixture} | {band} | {arm} | "
                    f"{band_summary['target_serve_misses']} | "
                    f"{trigger['true_positives']} | {trigger['false_positives']} | "
                    f"{injection['all_unmatched_serves_recovered']} |"
                )

    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- A true positive is a selected candidate within the canonical tolerance of a target missed serve.",
        "- A false positive is a selected candidate that does not match a target missed serve.",
        "- Contact injection copies accepted contacts, adds selected candidates, and keeps spans fixed.",
        "- The evidence-only injection clears the raw mask only at selected candidate frames.",
        "- `live-non-standard` labels identify unusual live views. They do not by themselves prove a serve contact.",
        "- The two pose bands are sensitivity variants. Neither is a production configuration.",
        "",
    ])
    return "\n".join(lines)


def _write_text(path: Path, value: str) -> None:
    payload = value if value.endswith("\n") else value + "\n"
    path.write_text(payload, encoding="utf-8")
    if path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"text reload changed value for {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="dated output directory")
    parser.add_argument(
        "--mask-mode", choices=("committed", "no_replay", "both"), default="both",
        help="run the committed mask, the no_replay sensitivity control, or both",
    )
    parser.add_argument(
        "--fixture-profile",
        choices=tuple(FIXTURE_PROFILES),
        default="historical-calibration",
        help="digest-pinned three-video input profile",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help="directory containing reviewed <fixture>_broadcast_timeline_labels.csv.gz files",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile = FIXTURE_PROFILES[args.fixture_profile]
    output_root = (args.out or _default_output_root()).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    selected_modes = MASK_MODES if args.mask_mode == "both" else (args.mask_mode,)
    all_summaries: dict[str, Any] = {}
    files: list[str] = []
    for mask_mode in selected_modes:
        mode_summaries: dict[str, Any] = {}
        pooled_rows: list[dict[str, Any]] = []
        for fixture in profile.fixtures:
            rows, array, run_metadata = _measure_variant(fixture, mask_mode)
            csv_path = output_root / f"{fixture.name}_{mask_mode}_rallies.csv.gz"
            array_path = output_root / f"{fixture.name}_{mask_mode}_evidence.npy.xz"
            _write_csv_gz(rows, csv_path)
            _write_array_lzma(array, array_path)
            files.extend([str(csv_path.relative_to(output_root)), str(array_path.relative_to(output_root))])
            mode_summaries[fixture.name] = {**run_metadata, **_summary(rows)}
            pooled_rows.extend(rows)
        mode_summaries["pooled"] = _summary(pooled_rows)
        all_summaries[mask_mode] = mode_summaries

    candidate_summaries: dict[str, Any] = {}
    labels_dir = args.labels_dir.resolve()
    for fixture in profile.fixtures:
        labels_path = labels_dir / f"{fixture.name}_broadcast_timeline_labels.csv.gz"
        candidates, opportunities, candidate_summary = measure_candidate_fixture(
            fixture,
            labels_path,
        )
        candidate_path = output_root / f"{fixture.name}_serve_candidates.csv.gz"
        opportunity_path = output_root / f"{fixture.name}_serve_opportunities.csv.gz"
        _write_schema_csv_gz(candidates, CANDIDATE_COLUMNS, candidate_path)
        _write_schema_csv_gz(opportunities, OPPORTUNITY_COLUMNS, opportunity_path)
        files.extend([
            str(candidate_path.relative_to(output_root)),
            str(opportunity_path.relative_to(output_root)),
        ])
        candidate_summaries[fixture.name] = candidate_summary

    summary = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "candidate_module": {
                "path": str(Path(candidate_measurement.__file__).resolve()),
                "sha256": hashlib.sha256(
                    Path(candidate_measurement.__file__).read_bytes()
                ).hexdigest(),
            },
            "git": _git_state(),
        },
        "fixture_root": str(fixtures_root()),
        "fixture_profile": {
            "name": args.fixture_profile,
            "source": profile.source,
            "source_commit": profile.source_commit,
        },
        "fixtures": [fixture.name for fixture in profile.fixtures],
        "mask_modes": list(selected_modes),
        "window_seconds_each_side": WINDOW_SECONDS,
        "notes": [
            "Rally rows describe the current baseline and evidence around GT serves.",
            "The no_replay variant passes raw_exclusion_mask=False for every frame through the existing calibration precedent.",
            "Candidate and injection results are recording-only and do not change production output.",
            "Contact injection copies accepted contacts, adds selected frames, and keeps spans fixed.",
            "The array files are native NumPy .npy streams wrapped with lzma XZ preset 9.",
        ],
        "variants": all_summaries,
        "candidate_analysis": {
            "per_fixture": candidate_summaries,
            "pooled": _pool_candidate_summaries(candidate_summaries),
        },
        "files": [],
    }
    summary_path = output_root / "summary.json.gz"
    report_path = output_root / "report.md"
    files.extend([
        str(report_path.relative_to(output_root)),
        str(summary_path.relative_to(output_root)),
    ])
    summary["files"] = sorted(files)
    _write_json_gz(summary, summary_path)
    _write_text(report_path, _build_candidate_report(summary))
    print(json.dumps(summary["candidate_analysis"]["pooled"], indent=2, sort_keys=True))
    print(f"\nWrote current serve-prepend evidence to {output_root}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
