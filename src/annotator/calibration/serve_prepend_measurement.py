"""Recording-only candidate rules for the serve-prepend measurement."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np

from annotator.broadcast_timeline_labels import (
    SceneTruth,
    VideoMetadata,
    read_label_csv,
    validate_partition,
)
from annotator.calibration.fixtures import Fixture
from annotator.calibration.gt_scoring import canonical_tolerance
from annotator.calibration.scoring import GtRally, classify_all, greedy_match
from annotator.config import BaseAnnotatorConfig
from annotator.inpaint_guard import CODE_NAMES, NO_FLAG
from annotator.rally_segmentation import BODY_UNIT_WRIST_THRESHOLD, detect_contact_flags
from annotator.resolve import resolve
from annotator.run_video import run_video


POSE_BANDS = {
    "middle-half": 0.5,
    "middle-two-thirds": 2.0 / 3.0,
}
TRUTH_ORDER = tuple(SceneTruth)

CANDIDATE_COLUMNS = (
    "fixture", "band", "span_id", "span_start", "span_end", "anchor_frame",
    "anchor_source", "lookback_start", "candidate_frame", "impulse", "manual_truth",
    "inpaint_code", "inpaint_name", "track_visible", "court_present", "court_absent",
    "pose_slot", "pose_score", "pose_bbox_height", "pose_wrist_distance_bh",
    "pose_detections", "valid_pose_detections", "central_pose_detections",
    "wrist_gate_pass", "evidence_pass", "anchor_distance_frames",
    "suppression_radius_frames", "anchor_suppression_pass", "candidate_pass",
    "raw_mask", "definitive_mask", "policy_pass",
    "selected_evidence", "selected_policy", "reject_reasons", "nearest_gt_rally_index",
    "nearest_gt_serve_frame", "nearest_gt_serve_delta", "nearest_gt_status",
    "tolerance_frames", "gt_serve_match", "natural_raw_contact",
    "natural_accepted_contact",
)

OPPORTUNITY_COLUMNS = (
    "fixture", "band", "span_id", "span_start", "span_end", "anchor_frame",
    "anchor_source", "lookback_start", "candidate_count", "evidence_pass_count",
    "candidate_pass_count", "policy_pass_count", "selected_evidence_frame",
    "selected_policy_frame",
)


class CentralPoseEvidence(NamedTuple):
    slot: int | None
    score: float | None
    bbox_height: float | None
    wrist_distance_bh: float | None
    n_detections: int
    n_valid_detections: int
    n_central_detections: int


def expand_truth(labels_path: Path, metadata: VideoMetadata) -> tuple[np.ndarray, int]:
    """Load a reviewed interval partition as one class value per frame."""
    intervals = read_label_csv(labels_path)
    validate_partition(intervals, expected_metadata=metadata)
    truth = np.empty(metadata.frame_count, dtype="U18")
    for interval in intervals:
        truth[interval.start_frame:interval.end_frame] = interval.truth.value
    return truth, len(intervals)


def central_pose_evidence(
    *,
    track: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray,
    kps: np.ndarray,
    ndet: np.ndarray,
    frame: int,
    resolution: tuple[float, float],
    band_fraction: float,
) -> CentralPoseEvidence:
    """Measure the largest valid pose whose box centre is in a central vertical band."""
    n_detections = max(
        0,
        min(int(ndet[frame]), bboxes.shape[1], scores.shape[1], kps.shape[1]),
    )
    if n_detections == 0 or kps.shape[2] <= 10:
        return CentralPoseEvidence(None, None, None, None, n_detections, 0, 0)

    frame_boxes = bboxes[frame, :n_detections]
    frame_scores = scores[frame, :n_detections]
    frame_wrists = kps[frame, :n_detections][:, [9, 10], :]
    widths = frame_boxes[:, 2] - frame_boxes[:, 0]
    heights = frame_boxes[:, 3] - frame_boxes[:, 1]
    valid = (
        np.isfinite(frame_boxes).all(axis=1)
        & np.isfinite(frame_scores)
        & np.isfinite(frame_wrists).all(axis=(1, 2))
        & (widths > 0)
        & (heights > 0)
    )
    valid_count = int(valid.sum())
    frame_width = float(resolution[0])
    margin = frame_width * (1.0 - band_fraction) / 2.0
    centres = (frame_boxes[:, 0] + frame_boxes[:, 2]) / 2.0
    central = valid & (centres >= margin) & (centres <= frame_width - margin)
    central_slots = np.flatnonzero(central)
    if not len(central_slots):
        return CentralPoseEvidence(
            None, None, None, None, n_detections, valid_count, 0,
        )

    areas = widths * heights
    slot = max(
        map(int, central_slots),
        key=lambda index: (float(areas[index]), float(frame_scores[index]), -index),
    )
    shuttle = track[frame, :2] * np.asarray(resolution)
    wrist_distance = float(np.linalg.norm(frame_wrists[slot] - shuttle, axis=1).min())
    bbox_height = float(heights[slot])
    return CentralPoseEvidence(
        slot=slot,
        score=float(frame_scores[slot]),
        bbox_height=bbox_height,
        wrist_distance_bh=wrist_distance / bbox_height,
        n_detections=n_detections,
        n_valid_detections=valid_count,
        n_central_detections=len(central_slots),
    )


def gt_statuses(
    gt_rallies: Sequence[GtRally],
    spans: Sequence[tuple[int, int]],
    accepted_by_span: dict[int, list[int]],
    tolerance: int,
) -> list[dict[str, Any]]:
    """Describe serve coverage for every ground-truth rally."""
    classifications = classify_all(list(spans), list(gt_rallies))
    statuses: list[dict[str, Any]] = []
    for rally_index, (rally, (boundary, mapped_span)) in enumerate(
        zip(gt_rallies, classifications, strict=True)
    ):
        first, last = rally.extent
        overlapping = [
            span_id
            for span_id, (start, end) in enumerate(spans)
            if start <= last and first < end
        ]
        accepted = sorted(
            frame
            for span_id in overlapping
            for frame in accepted_by_span.get(span_id, [])
        )
        matches = greedy_match(rally.stroke_frames, accepted, tolerance)
        matched_gt = {gt_index for gt_index, _candidate_index in matches}
        statuses.append({
            "rally_index": rally_index,
            "serve_frame": rally.stroke_frames[0],
            "boundary": boundary.value,
            "mapped_span": mapped_span,
            "serve_matched": 0 in matched_gt,
            "later_strokes_matched": any(index > 0 for index in matched_gt),
            "target_miss": 0 not in matched_gt and any(index > 0 for index in matched_gt),
        })
    return statuses


def _candidate_reasons(
    *,
    visible: bool,
    inpaint_code: int,
    court_absent: bool,
    wrist_pass: bool,
    evidence_pass: bool,
    anchor_suppression_pass: bool,
    definitive_mask: bool,
) -> str:
    reasons = []
    if not visible:
        reasons.append("track-not-visible")
    if inpaint_code != NO_FLAG:
        reasons.append(f"inpaint-code-{inpaint_code}")
    if not court_absent:
        reasons.append("court-present")
    if not wrist_pass:
        reasons.append("central-pose-wrist-gate")
    if not anchor_suppression_pass:
        reasons.append("contact-suppression-with-anchor")
    if evidence_pass and anchor_suppression_pass and definitive_mask:
        reasons.append("definitive-mask")
    return ";".join(reasons)


def _nearest_gt_status(status: dict[str, Any]) -> str:
    if status["target_miss"]:
        return "target-miss"
    if status["serve_matched"]:
        return "serve-covered"
    return "whole-rally-or-unresolved-miss"


def _select_candidate(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    return min(
        rows,
        key=lambda row: (-float(row["impulse"]), int(row["candidate_frame"])),
        default=None,
    )


def _trigger_metrics(
    selected: Sequence[dict[str, Any]],
    statuses: Sequence[dict[str, Any]],
    tolerance: int,
) -> dict[str, Any]:
    target_frames = [int(row["serve_frame"]) for row in statuses if row["target_miss"]]
    selected_frames = [int(row["candidate_frame"]) for row in selected]
    matches = greedy_match(target_frames, selected_frames, tolerance)
    matched_candidates = {candidate_index for _gt_index, candidate_index in matches}
    true_positives = len(matches)
    false_positives = len(selected) - len(matched_candidates)
    false_negatives = len(target_frames) - true_positives
    return {
        "selected_triggers": len(selected),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives else None
        ),
        "recall": true_positives / len(target_frames) if target_frames else None,
        "selected_by_manual_truth": {
            truth.value: sum(row["manual_truth"] == truth.value for row in selected)
            for truth in TRUTH_ORDER
        },
        "selected_by_nearest_gt_status": {
            status: sum(row["nearest_gt_status"] == status for row in selected)
            for status in (
                "target-miss",
                "serve-covered",
                "whole-rally-or-unresolved-miss",
            )
        },
    }


def _target_outcomes(
    candidate_rows: Sequence[dict[str, Any]],
    statuses: Sequence[dict[str, Any]],
    selection_field: str,
) -> dict[str, int]:
    outcomes = {
        "selected_match": 0,
        "no_raw_candidate_within_tolerance": 0,
        "no_clean_shuttle_evidence": 0,
        "court_present_not_prototype_target": 0,
        "no_usable_central_pose_wrist_evidence": 0,
        "blocked_by_contact_suppression": 0,
        "blocked_by_definitive_mask": 0,
        "eligible_but_not_selected": 0,
    }
    for status in statuses:
        if not status["target_miss"]:
            continue
        rally_index = int(status["rally_index"])
        near = [
            row
            for row in candidate_rows
            if row["nearest_gt_rally_index"] == rally_index and row["gt_serve_match"]
        ]
        if any(row[selection_field] for row in near):
            outcome = "selected_match"
        elif not near:
            outcome = "no_raw_candidate_within_tolerance"
        elif not any(row["track_visible"] and row["inpaint_code"] == NO_FLAG for row in near):
            outcome = "no_clean_shuttle_evidence"
        elif not any(row["court_absent"] for row in near):
            outcome = "court_present_not_prototype_target"
        elif not any(row["wrist_gate_pass"] for row in near):
            outcome = "no_usable_central_pose_wrist_evidence"
        elif not any(row["anchor_suppression_pass"] for row in near):
            outcome = "blocked_by_contact_suppression"
        elif any(row["candidate_pass"] and row["definitive_mask"] for row in near):
            outcome = "blocked_by_definitive_mask"
        else:
            outcome = "eligible_but_not_selected"
        outcomes[outcome] += 1
    return outcomes


def _build_candidate_row(
    *,
    fixture: Fixture,
    band: str,
    band_fraction: float,
    span_id: int,
    span: tuple[int, int],
    anchor: int,
    anchor_source: str,
    lookback_start: int,
    frame: int,
    impulse: float,
    arrays: tuple[np.ndarray, ...],
    truth: np.ndarray,
    inpaint_codes: np.ndarray,
    court_present: np.ndarray,
    raw_mask: np.ndarray,
    definitive_mask: np.ndarray,
    gt_rallies: Sequence[GtRally],
    statuses: Sequence[dict[str, Any]],
    tolerance: int,
    suppression_radius: int,
    natural_raw: set[tuple[int, int]],
    natural_accepted: set[tuple[int, int]],
) -> dict[str, Any]:
    track, bboxes, scores, kps, ndet = arrays
    pose = central_pose_evidence(
        track=track,
        bboxes=bboxes,
        scores=scores,
        kps=kps,
        ndet=ndet,
        frame=frame,
        resolution=fixture.resolution,
        band_fraction=band_fraction,
    )
    visible = bool(track[frame, 2] == 1)
    code = int(inpaint_codes[frame])
    court_absent = not bool(court_present[frame])
    wrist_pass = bool(
        pose.wrist_distance_bh is not None
        and np.isfinite(pose.wrist_distance_bh)
        and pose.wrist_distance_bh <= BODY_UNIT_WRIST_THRESHOLD
    )
    evidence_pass = visible and code == NO_FLAG and court_absent and wrist_pass
    anchor_distance = anchor - frame
    anchor_suppression_pass = bool(
        anchor_source != "first-accepted-contact"
        or anchor_distance >= suppression_radius
    )
    candidate_pass = evidence_pass and anchor_suppression_pass
    policy_pass = candidate_pass and not bool(definitive_mask[frame])
    nearest_gt = min(
        range(len(gt_rallies)),
        key=lambda index: (abs(gt_rallies[index].stroke_frames[0] - frame), index),
    )
    nearest_serve = int(gt_rallies[nearest_gt].stroke_frames[0])
    nearest_delta = abs(nearest_serve - frame)
    return {
        "fixture": fixture.name,
        "band": band,
        "span_id": span_id,
        "span_start": span[0],
        "span_end": span[1],
        "anchor_frame": anchor,
        "anchor_source": anchor_source,
        "lookback_start": lookback_start,
        "candidate_frame": frame,
        "impulse": impulse,
        "manual_truth": str(truth[frame]),
        "inpaint_code": code,
        "inpaint_name": CODE_NAMES[code],
        "track_visible": visible,
        "court_present": not court_absent,
        "court_absent": court_absent,
        "pose_slot": pose.slot,
        "pose_score": pose.score,
        "pose_bbox_height": pose.bbox_height,
        "pose_wrist_distance_bh": pose.wrist_distance_bh,
        "pose_detections": pose.n_detections,
        "valid_pose_detections": pose.n_valid_detections,
        "central_pose_detections": pose.n_central_detections,
        "wrist_gate_pass": wrist_pass,
        "evidence_pass": evidence_pass,
        "anchor_distance_frames": anchor_distance,
        "suppression_radius_frames": suppression_radius,
        "anchor_suppression_pass": anchor_suppression_pass,
        "candidate_pass": candidate_pass,
        "raw_mask": bool(raw_mask[frame]),
        "definitive_mask": bool(definitive_mask[frame]),
        "policy_pass": policy_pass,
        "selected_evidence": False,
        "selected_policy": False,
        "reject_reasons": _candidate_reasons(
            visible=visible,
            inpaint_code=code,
            court_absent=court_absent,
            wrist_pass=wrist_pass,
            evidence_pass=evidence_pass,
            anchor_suppression_pass=anchor_suppression_pass,
            definitive_mask=bool(definitive_mask[frame]),
        ),
        "nearest_gt_rally_index": nearest_gt,
        "nearest_gt_serve_frame": nearest_serve,
        "nearest_gt_serve_delta": nearest_delta,
        "nearest_gt_status": _nearest_gt_status(statuses[nearest_gt]),
        "tolerance_frames": tolerance,
        "gt_serve_match": nearest_delta <= tolerance,
        "natural_raw_contact": (span_id, frame) in natural_raw,
        "natural_accepted_contact": (span_id, frame) in natural_accepted,
    }


def evaluate_serve_candidates(
    *,
    fixture: Fixture,
    band: str,
    band_fraction: float,
    arrays: tuple[np.ndarray, ...],
    truth: np.ndarray,
    inpaint_codes: np.ndarray,
    court_present: np.ndarray,
    raw_mask: np.ndarray,
    definitive_mask: np.ndarray,
    spans: Sequence[tuple[int, int]],
    raw_by_span: dict[int, list[int]],
    accepted_by_span: dict[int, list[int]],
    gt_rallies: Sequence[GtRally],
    statuses: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Score every bounded raw candidate before each claimed rally start."""
    resolved = resolve(BaseAnnotatorConfig(), fixture.fps)
    tolerance = canonical_tolerance(fixture.fps)
    natural_raw = {
        (span_id, frame) for span_id, frames in raw_by_span.items() for frame in frames
    }
    natural_accepted = {
        (span_id, frame) for span_id, frames in accepted_by_span.items() for frame in frames
    }
    candidate_rows: list[dict[str, Any]] = []
    opportunity_rows: list[dict[str, Any]] = []

    for span_id, span in enumerate(spans):
        accepted = accepted_by_span.get(span_id, [])
        anchor = min(accepted) if accepted else span[0]
        anchor_source = "first-accepted-contact" if accepted else "span-start-fallback"
        lookback_start = max(0, anchor - resolved.constants.serve_start_lookback_frames)
        flags = detect_contact_flags(
            arrays[0],
            lookback_start,
            anchor,
            resolved.thresholds,
            smoothing_mode=resolved.smoothing_mode,
        )
        evaluated = [
            _build_candidate_row(
                fixture=fixture,
                band=band,
                band_fraction=band_fraction,
                span_id=span_id,
                span=span,
                anchor=anchor,
                anchor_source=anchor_source,
                lookback_start=lookback_start,
                frame=frame,
                impulse=impulse,
                arrays=arrays,
                truth=truth,
                inpaint_codes=inpaint_codes,
                court_present=court_present,
                raw_mask=raw_mask,
                definitive_mask=definitive_mask,
                gt_rallies=gt_rallies,
                statuses=statuses,
                tolerance=tolerance,
                suppression_radius=resolved.thresholds.contact_suppression_radius_frames,
                natural_raw=natural_raw,
                natural_accepted=natural_accepted,
            )
            for frame, impulse in flags
        ]
        evidence_rows = [row for row in evaluated if row["candidate_pass"]]
        policy_rows = [row for row in evaluated if row["policy_pass"]]
        selected_evidence = _select_candidate(evidence_rows)
        selected_policy = _select_candidate(policy_rows)
        if selected_evidence is not None:
            selected_evidence["selected_evidence"] = True
        if selected_policy is not None:
            selected_policy["selected_policy"] = True
        candidate_rows.extend(evaluated)
        opportunity_rows.append({
            "fixture": fixture.name,
            "band": band,
            "span_id": span_id,
            "span_start": span[0],
            "span_end": span[1],
            "anchor_frame": anchor,
            "anchor_source": anchor_source,
            "lookback_start": lookback_start,
            "candidate_count": len(evaluated),
            "evidence_pass_count": sum(row["evidence_pass"] for row in evaluated),
            "candidate_pass_count": sum(row["candidate_pass"] for row in evaluated),
            "policy_pass_count": len(policy_rows),
            "selected_evidence_frame": (
                selected_evidence["candidate_frame"]
                if selected_evidence is not None else None
            ),
            "selected_policy_frame": (
                selected_policy["candidate_frame"] if selected_policy is not None else None
            ),
        })

    evidence_selected = [row for row in candidate_rows if row["selected_evidence"]]
    policy_selected = [row for row in candidate_rows if row["selected_policy"]]
    summary = {
        "candidate_rule": {
            "lookback_frames": resolved.constants.serve_start_lookback_frames,
            "gt_serve_tolerance_frames": tolerance,
            "central_band_fraction": band_fraction,
            "pose_selection": "largest valid bbox area, then score, then earliest slot",
            "wrist_threshold_body_heights": BODY_UNIT_WRIST_THRESHOLD,
            "contact_suppression_radius_frames": (
                resolved.thresholds.contact_suppression_radius_frames
            ),
            "requires_visible_clean_track": True,
            "requires_court_absent": True,
            "definitive_mask_policy": "blocked",
            "opportunity_selection": "largest impulse, then earliest frame",
        },
        "n_opportunities": len(opportunity_rows),
        "n_candidate_rows": len(candidate_rows),
        "n_evidence_pass": sum(row["evidence_pass"] for row in candidate_rows),
        "n_candidate_pass": sum(row["candidate_pass"] for row in candidate_rows),
        "n_policy_pass": sum(row["policy_pass"] for row in candidate_rows),
        "target_serve_misses": sum(row["target_miss"] for row in statuses),
        "trigger_metrics": {
            "evidence_only": _trigger_metrics(evidence_selected, statuses, tolerance),
            "current_mask_policy": _trigger_metrics(policy_selected, statuses, tolerance),
        },
        "target_outcomes": {
            "evidence_only": _target_outcomes(
                candidate_rows,
                statuses,
                "selected_evidence",
            ),
            "current_mask_policy": _target_outcomes(
                candidate_rows,
                statuses,
                "selected_policy",
            ),
        },
    }
    return candidate_rows, opportunity_rows, summary


def contacts_by_span(contacts: Sequence[Any]) -> dict[int, list[int]]:
    """Group contact rows by their attributed span."""
    by_span: dict[int, list[int]] = {}
    for contact in contacts:
        by_span.setdefault(int(contact.rally_id), []).append(int(contact.contact_frame))
    return {span_id: sorted(set(frames)) for span_id, frames in by_span.items()}


def _changed_positions(before: Sequence[Any], after: Sequence[Any]) -> int:
    shared = sum(left != right for left, right in zip(before, after))
    return shared + abs(len(before) - len(after))


def run_contact_injection_counterfactual(
    *,
    inputs: Any,
    natural_result: Any,
    gt_rallies: Sequence[GtRally],
    natural_statuses: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    tolerance: int,
    selection_field: str,
    exempt_selected_from_raw_mask: bool,
) -> dict[str, Any]:
    """Inject all selected candidates into a copied accepted-contact map."""
    selected = [row for row in candidate_rows if row[selection_field]]
    contacts = {
        int(span_id): sorted(map(int, frames))
        for span_id, frames in natural_result.filtered_by_rally.items()
    }
    natural_pairs = {
        (span_id, frame) for span_id, frames in contacts.items() for frame in frames
    }
    for row in selected:
        contacts.setdefault(int(row["span_id"]), []).append(int(row["candidate_frame"]))
    contacts = {span_id: sorted(set(frames)) for span_id, frames in contacts.items()}
    keyword = dict(inputs.keyword)
    if exempt_selected_from_raw_mask:
        raw_mask = np.asarray(keyword["raw_exclusion_mask"]).copy()
        for row in selected:
            raw_mask[int(row["candidate_frame"])] = False
        keyword["raw_exclusion_mask"] = raw_mask
    injected_result = run_video(
        *inputs.positional,
        **keyword,
        spans=list(natural_result.spans),
        contacts=contacts,
    )
    injected_accepted = {
        (int(contact.rally_id), int(contact.contact_frame))
        for contact in injected_result.filtered_contacts
    }
    selected_pairs = {
        (int(row["span_id"]), int(row["candidate_frame"])) for row in selected
    }
    injected_statuses = gt_statuses(
        gt_rallies,
        injected_result.spans,
        {
            int(span_id): sorted(map(int, frames))
            for span_id, frames in injected_result.filtered_by_rally.items()
        },
        tolerance,
    )
    target_indexes = [
        int(row["rally_index"]) for row in natural_statuses if row["target_miss"]
    ]
    unmatched_indexes = [
        int(row["rally_index"]) for row in natural_statuses if not row["serve_matched"]
    ]
    recovered = sum(injected_statuses[index]["serve_matched"] for index in target_indexes)
    all_recovered = sum(
        injected_statuses[index]["serve_matched"] for index in unmatched_indexes
    )
    return {
        "method": "copy accepted contacts, inject selected frames, keep spans fixed",
        "selection_field": selection_field,
        "selected_raw_mask_exemption": exempt_selected_from_raw_mask,
        "exempted_raw_mask_frames": sum(
            bool(np.asarray(inputs.keyword["raw_exclusion_mask"])[int(row["candidate_frame"])])
            for row in selected
        ) if exempt_selected_from_raw_mask else 0,
        "selected_candidates": len(selected_pairs),
        "new_selected_candidates": len(selected_pairs - natural_pairs),
        "accepted_selected_candidates": len(selected_pairs & injected_accepted),
        "target_serves_recovered": recovered,
        "target_serves_still_missed": len(target_indexes) - recovered,
        "all_unmatched_serves_recovered": all_recovered,
        "all_unmatched_serves_still_missed": len(unmatched_indexes) - all_recovered,
        "accepted_contact_delta": (
            len(injected_result.filtered_contacts) - len(natural_result.filtered_contacts)
        ),
        "span_delta": len(injected_result.spans) - len(natural_result.spans),
        "stroke_count_rows_changed": _changed_positions(
            natural_result.n_strokes_list,
            injected_result.n_strokes_list,
        ),
        "next_server_rows_changed": _changed_positions(
            natural_result.next_servers,
            injected_result.next_servers,
        ),
    }
