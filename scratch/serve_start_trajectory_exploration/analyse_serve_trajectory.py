"""Test incoming shuttle motion at the first accepted contact, then prepend a shot."""

from __future__ import annotations

import gc
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-corrected-serve-trajectory")

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

matplotlib.use("Agg")
from experiment_data import VideoData, load_video_data, normalise_half, other_half
from report_outputs import write_final_outputs
from trajectory_features import (
    HISTORICAL_MIN_CLOSING_FRACTION,
    MAX_LARGEST_STEP_RATIO,
    MIN_PATH_FRAMES,
    MIN_TOTAL_MOVEMENT_BH,
    PRIMARY_MIN_NET_CLOSURE_BH,
    ROBUST_TREND_MIN_DECREASE_BH,
    align_anchor_to_gt,
    closest_pre_contact_run,
    decide_fixed_motion_rules,
    first_player_from_final_half,
    fit_path,
    fit_robust_distance_trend,
    measure_incoming_motion,
    summarise_unmatched_anchor_sequence,
)

from annotator import point_winner
from annotator.calibration.fixtures import FIXTURES
from annotator.calibration.gt_scoring import load_gt_tables
from annotator.calibration.scoring import RallyBoundary
from annotator.fps_constants import ScalingKind
from annotator.inpaint_guard import NO_FLAG
from annotator.types import Slot

RUN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RUN_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

LOOKBACK_BASE30_FRAMES = 30
MAX_FRAMES_TO_CONTACT_BASE30 = 2
CONTACT_TOLERANCES_BASE30 = (5, 10, 30)
PATH_VARIANTS = ("recurrence_clean", "producer_original")
PRIMARY_PATH_VARIANT = "recurrence_clean"


def _half_text(half: point_winner.Half | None) -> str | None:
    """Return the CSV spelling for a court half."""
    return half.value if half is not None else None


def _phase_scores(guesses: list[point_winner.Half | None]) -> tuple[int, int]:
    """Return Top-final and Bot-final match counts for the existing alternating fit."""
    scores = []
    final_index = len(guesses) - 1
    for final_half in (point_winner.Half.TOP, point_winner.Half.BOT):
        score = 0
        for contact_index, guess in enumerate(guesses):
            assigned = final_half if (final_index - contact_index) % 2 == 0 else other_half(final_half)
            if guess is not None and guess == assigned:
                score += 1
        scores.append(score)
    return scores[0], scores[1]


def _fit_first_player(guesses: list[point_winner.Half | None]) -> point_winner.Half | None:
    """Run the existing alternating fit and return its implied first player."""
    if not guesses:
        return None
    final_half = point_winner.fit_alternation(guesses)
    return first_player_from_final_half(final_half, len(guesses))


def _empty_row(data: VideoData, rally: Any, boundary: RallyBoundary, span_id: int | None) -> dict[str, object]:
    """Make one explicit result row before adding covered-rally evidence."""
    truth = data.truth_first_second[(rally.set_id, rally.rally)]
    span_start, span_end = data.spans[span_id] if span_id is not None else (None, None)
    row: dict[str, object] = {
        "fixture": data.fixture.name,
        "video_id": data.fixture.video_id,
        "fps": data.fixture.fps,
        "set_id": rally.set_id,
        "rally": rally.rally,
        "boundary": boundary.value,
        "span_id": span_id,
        "predicted_span_key": f"{data.fixture.name}:{span_id}" if span_id is not None else None,
        "predicted_span_start": span_start,
        "predicted_span_end": span_end,
        "span_multiplicity": 0,
        "primary_one_to_one": False,
        "population_detail": "end_to_end_segmentation_failure",
        "gt_stroke_frames_json": json.dumps(list(rally.stroke_frames), separators=(",", ":")),
        "gt_stroke_count": len(rally.stroke_frames),
        "gt_server": _half_text(truth["gt_server"]),
        "gt_receiver": _half_text(truth["gt_receiver"]),
        "gt_first_frame": truth["gt_first_frame"],
        "gt_second_frame": truth["gt_second_frame"],
        "baseline_server": None,
        "baseline_correct": False,
        "baseline_missing": True,
        "baseline_wrong": False,
        "frozen_server_failure": False,
        "accepted_contact_count": 0,
        "accepted_contact_frames_json": "[]",
        "n_strokes_list": None,
        "raw_candidate_count": 0,
        "anchor_frame": None,
        "anchor_player": None,
        "anchor_equal_distance_tie": False,
        "anchor_gt_match": "no_anchor",
        "earlier_raw_candidates": 0,
        "earlier_wrist_rejections": 0,
        "earlier_suppressed_candidates": 0,
        "earlier_definitive_exclusions": 0,
        "court_scene_start": None,
        "court_scene_end": None,
        "path_reaches_scene_start": False,
        "direct_contact_guesses": "",
        "direct_fit_final": None,
        "direct_fit_first": None,
        "direct_fit_top_score": 0,
        "direct_fit_bot_score": 0,
        "direct_fit_margin": 0,
        "unmatched_sequence_checked": False,
        "later_contacts_checked": 0,
        "later_serve_within_tolerance": False,
        "later_first_return_within_tolerance": False,
        "first_gt_match_rank": None,
        "first_gt_match_ordinal": None,
        "first_gt_match_multiple": False,
        "reused_gt_ordinal": False,
    }
    for tolerance in CONTACT_TOLERANCES_BASE30:
        prefix = f"anchor_tolerance_{tolerance}"
        row.update(
            {
                f"{prefix}_nearest_gt_ordinal": None,
                f"{prefix}_signed_offset_base30": math.nan,
                f"{prefix}_absolute_offset_base30": math.nan,
                f"{prefix}_in_window_count": 0,
                f"{prefix}_multiple": False,
                f"{prefix}_label": "no_anchor",
            }
        )
    for variant in PATH_VARIANTS:
        row.update(
            {
                f"{variant}_path_start": None,
                f"{variant}_path_end": None,
                f"{variant}_path_frames": 0,
                f"{variant}_frames_to_contact": None,
                f"{variant}_selected_path": False,
                f"{variant}_start_distance_bh": math.nan,
                f"{variant}_end_distance_bh": math.nan,
                f"{variant}_net_closure_bh": math.nan,
                f"{variant}_movements_towards_player": math.nan,
                f"{variant}_total_movement_bh": math.nan,
                f"{variant}_largest_step_ratio": math.nan,
                f"{variant}_linear_rmse": math.nan,
                f"{variant}_quadratic_rmse": math.nan,
                f"{variant}_quadratic_improvement": math.nan,
                f"{variant}_path_available": False,
                f"{variant}_path_quality_pass": False,
                f"{variant}_common_path_eligible": False,
                f"{variant}_historical_path_eligible": False,
                f"{variant}_robust_slope_bh_per_path": math.nan,
                f"{variant}_robust_intercept_bh": math.nan,
                f"{variant}_fitted_decrease_bh": math.nan,
                f"{variant}_residual_rms_bh": math.nan,
                f"{variant}_trend_to_jitter": math.nan,
                f"{variant}_historical_incoming": False,
                f"{variant}_robust_trend_incoming": False,
            }
        )
    return row


def _measure_path(
    row: dict[str, object],
    data: VideoData,
    anchor: int,
    anchor_player: point_winner.Half,
    variant: str,
    usable: np.ndarray,
    same_scene: np.ndarray,
    lookback_frames: int,
    maximum_frames_to_contact: int,
    point_rows: list[dict[str, object]],
    identity: tuple[str, int, str, int],
) -> dict[str, np.ndarray] | None:
    """Measure the closest usable path for one source-quality definition."""
    run = closest_pre_contact_run(usable, anchor, lookback_frames, same_scene)
    if run is None:
        return None

    slot = Slot.TOP if anchor_player is point_winner.Half.TOP else Slot.BOTTOM
    run_slice = slice(run.start, run.end)
    row[f"{variant}_path_start"] = run.start
    row[f"{variant}_path_end"] = run.end
    row[f"{variant}_path_frames"] = run.end - run.start
    row[f"{variant}_frames_to_contact"] = run.frames_to_contact
    row[f"{variant}_selected_path"] = True
    distances_bh = data.sticky.distances_per_slot[run_slice, slot]
    shuttle_xy = data.track[run_slice, :2]
    bbox_heights_px = data.sticky.bbox_height[run_slice, slot]
    fixture, video_id, set_id, rally_number = identity
    for sample_index, source_frame in enumerate(range(run.start, run.end)):
        point_rows.append(
            {
                "fixture": fixture,
                "video_id": video_id,
                "set_id": set_id,
                "rally": rally_number,
                "path_definition": variant,
                "source_frame": source_frame,
                "sample_index": sample_index,
                "distance_bh": float(distances_bh[sample_index]),
                "shuttle_x": float(shuttle_xy[sample_index, 0]),
                "shuttle_y": float(shuttle_xy[sample_index, 1]),
                "bbox_height_px": float(bbox_heights_px[sample_index]),
            }
        )
    if run.end - run.start < 2:
        return None

    motion = measure_incoming_motion(
        distances_bh,
        shuttle_xy,
        bbox_heights_px,
        data.fixture.resolution,
    )
    robust_trend = fit_robust_distance_trend(distances_bh)
    fit = fit_path(shuttle_xy)
    row[f"{variant}_start_distance_bh"] = motion.start_distance_bh
    row[f"{variant}_end_distance_bh"] = motion.end_distance_bh
    row[f"{variant}_net_closure_bh"] = motion.net_closure_bh
    row[f"{variant}_movements_towards_player"] = motion.closing_fraction
    row[f"{variant}_total_movement_bh"] = motion.total_movement_bh
    row[f"{variant}_largest_step_ratio"] = motion.largest_step_ratio
    row[f"{variant}_linear_rmse"] = fit.linear_rmse
    row[f"{variant}_quadratic_rmse"] = fit.quadratic_rmse
    row[f"{variant}_quadratic_improvement"] = fit.quadratic_improvement
    row[f"{variant}_robust_slope_bh_per_path"] = robust_trend.slope_bh_per_path
    row[f"{variant}_robust_intercept_bh"] = robust_trend.intercept_bh
    row[f"{variant}_fitted_decrease_bh"] = robust_trend.fitted_decrease_bh
    row[f"{variant}_residual_rms_bh"] = robust_trend.residual_rms_bh
    row[f"{variant}_trend_to_jitter"] = robust_trend.trend_to_jitter
    path_available = (
        motion.n_frames >= MIN_PATH_FRAMES and run.frames_to_contact <= maximum_frames_to_contact
    )
    decisions = decide_fixed_motion_rules(
        motion,
        robust_trend,
        run.frames_to_contact,
        maximum_frames_to_contact,
    )
    row[f"{variant}_path_available"] = path_available
    row[f"{variant}_path_quality_pass"] = decisions.historical_path_eligible
    row[f"{variant}_common_path_eligible"] = decisions.common_path_eligible
    row[f"{variant}_historical_path_eligible"] = decisions.historical_path_eligible
    row[f"{variant}_historical_incoming"] = decisions.historical_incoming
    row[f"{variant}_robust_trend_incoming"] = decisions.robust_trend_incoming
    return {
        "path": data.track[run_slice, :2].copy(),
        "anchor_ankles": data.sticky.ankle_pos[run_slice, slot].copy(),
    }


def _populate_covered_row(
    row: dict[str, object],
    data: VideoData,
    rally: Any,
    span_id: int,
    case_paths: dict[tuple[int, str, int], dict[str, np.ndarray]],
    point_rows: list[dict[str, object]],
) -> None:
    """Add direct contact attribution and pre-contact motion for one covered rally."""
    gt_server = normalise_half(row["gt_server"])
    baseline_server = normalise_half(data.annotations["fitted_first_all"][span_id])
    row["baseline_server"] = _half_text(baseline_server)
    row["baseline_correct"] = baseline_server == gt_server
    row["baseline_missing"] = baseline_server is None
    row["baseline_wrong"] = baseline_server is not None and baseline_server != gt_server
    row["frozen_server_failure"] = baseline_server != gt_server

    accepted = sorted(data.accepted_by_span.get(span_id, []))
    if len(accepted) != len(set(accepted)):
        raise ValueError(f"{data.fixture.name} span {span_id}: accepted contacts must be unique")
    raw_contacts = data.raw_contacts_by_span.get(span_id, [])
    row["accepted_contact_count"] = len(accepted)
    row["accepted_contact_frames_json"] = json.dumps(accepted, separators=(",", ":"))
    row["n_strokes_list"] = data.annotations["n_strokes_list"][span_id]
    row["raw_candidate_count"] = len(raw_contacts)
    if not accepted:
        return

    anchor = min(accepted)
    anchor_player = point_winner.attribute_half(
        anchor,
        data.track,
        data.sticky,
        data.bboxes,
        data.fixture.net_band,
    )
    row["anchor_frame"] = anchor
    row["anchor_player"] = _half_text(anchor_player)
    for tolerance in CONTACT_TOLERANCES_BASE30:
        alignment = align_anchor_to_gt(anchor, rally.stroke_frames, data.fixture.fps, tolerance)
        prefix = f"anchor_tolerance_{tolerance}"
        row[f"{prefix}_nearest_gt_ordinal"] = alignment.nearest_gt_ordinal
        row[f"{prefix}_signed_offset_base30"] = alignment.signed_offset_base30
        row[f"{prefix}_absolute_offset_base30"] = alignment.absolute_offset_base30
        row[f"{prefix}_in_window_count"] = alignment.in_window_count
        row[f"{prefix}_multiple"] = alignment.multiple_within_tolerance
        row[f"{prefix}_label"] = alignment.label
    row["anchor_gt_match"] = row["anchor_tolerance_10_label"]
    if row["anchor_tolerance_10_label"] == "unmatched":
        sequence = summarise_unmatched_anchor_sequence(
            accepted,
            rally.stroke_frames,
            data.fixture.fps,
            tolerance_base30=10,
        )
        row["unmatched_sequence_checked"] = True
        row["later_contacts_checked"] = sequence.later_contacts_checked
        row["later_serve_within_tolerance"] = sequence.later_serve_within_tolerance
        row["later_first_return_within_tolerance"] = (
            sequence.later_first_return_within_tolerance
        )
        row["first_gt_match_rank"] = sequence.first_gt_match_rank
        row["first_gt_match_ordinal"] = sequence.first_gt_match_ordinal
        row["first_gt_match_multiple"] = sequence.first_gt_match_multiple
        row["reused_gt_ordinal"] = sequence.reused_gt_ordinal
    distances = data.sticky.distances_per_slot[anchor]
    row["anchor_equal_distance_tie"] = bool(
        np.isfinite(distances).all() and distances[Slot.TOP] == distances[Slot.BOTTOM]
    )

    earlier_raw = [contact for contact in raw_contacts if contact.contact_frame < anchor]
    row["earlier_raw_candidates"] = len(earlier_raw)
    row["earlier_wrist_rejections"] = sum(contact.wrist_near is False for contact in earlier_raw)
    row["earlier_suppressed_candidates"] = sum(contact.suppressed is True for contact in earlier_raw)
    row["earlier_definitive_exclusions"] = sum(contact.definitive_exclusion for contact in earlier_raw)

    guesses = [
        point_winner.attribute_half(frame, data.track, data.sticky, data.bboxes, data.fixture.net_band)
        for frame in accepted
    ]
    fitted_final = point_winner.fit_alternation(guesses)
    fitted_first = first_player_from_final_half(fitted_final, len(guesses)) if fitted_final is not None else None
    frozen_final = normalise_half(data.annotations["striker_halves"][span_id])
    if fitted_final != frozen_final or fitted_first != baseline_server:
        raise ValueError(
            f"{data.fixture.name} {rally.set_id} rally {rally.rally}: direct contact refit "
            "does not reproduce the frozen release"
        )
    top_score, bot_score = _phase_scores(guesses)
    row["direct_contact_guesses"] = "|".join(_half_text(guess) or "Unknown" for guess in guesses)
    row["direct_fit_final"] = _half_text(fitted_final)
    row["direct_fit_first"] = _half_text(fitted_first)
    row["direct_fit_top_score"] = top_score
    row["direct_fit_bot_score"] = bot_score
    row["direct_fit_margin"] = abs(top_score - bot_score)

    segment = data.segment_for_frame(anchor)
    if anchor_player is None or segment is None:
        return
    segment_start, segment_end = segment
    row["court_scene_start"] = segment_start
    row["court_scene_end"] = segment_end
    same_scene = np.zeros(len(data.track), dtype=bool)
    same_scene[segment_start:segment_end] = True
    coordinate_valid = np.isfinite(data.track[:, :2]).all(axis=1)
    coordinate_valid &= ~((data.track[:, 0] == 0) & (data.track[:, 1] == 0))
    slot = Slot.TOP if anchor_player is point_winner.Half.TOP else Slot.BOTTOM
    common = (
        (data.track[:, 2] == 1)
        & coordinate_valid
        & data.court_present
        & np.isfinite(data.sticky.distances_per_slot[:, slot])
        & np.isfinite(data.sticky.bbox_height[:, slot])
        & (data.sticky.bbox_height[:, slot] > 0)
    )
    masks = {
        "recurrence_clean": common & (data.guard_codes == NO_FLAG),
        "producer_original": common & (data.guard_codes == NO_FLAG) & ~data.producer_inpaint,
    }
    lookback_frames = int(ScalingKind.FRAME_COUNT.scale(LOOKBACK_BASE30_FRAMES, data.fixture.fps))
    maximum_frames_to_contact = int(
        ScalingKind.FRAME_COUNT.scale(MAX_FRAMES_TO_CONTACT_BASE30, data.fixture.fps)
    )
    key = (data.fixture.video_id, rally.set_id, rally.rally)
    identity = (data.fixture.name, data.fixture.video_id, rally.set_id, rally.rally)
    case_paths[key] = {}
    for variant, usable in masks.items():
        evidence = _measure_path(
            row,
            data,
            anchor,
            anchor_player,
            variant,
            usable,
            same_scene,
            lookback_frames,
            maximum_frames_to_contact,
            point_rows,
            identity,
        )
        if evidence is not None:
            for name, values in evidence.items():
                case_paths[key][f"{variant}_{name}"] = values
    primary_start = row[f"{PRIMARY_PATH_VARIANT}_path_start"]
    row["path_reaches_scene_start"] = primary_start is not None and int(primary_start) == segment_start


def build_feature_rows() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[int, str, int], dict[str, np.ndarray]],
]:
    """Build checked rally, span and selected-path-point tables."""
    shared_gt_tables = load_gt_tables()
    rows: list[dict[str, object]] = []
    span_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    case_paths: dict[tuple[int, str, int], dict[str, np.ndarray]] = {}
    for fixture in FIXTURES:
        print(f"{fixture.name}: loading frozen data and rebuilding player geometry")
        data = load_video_data(fixture, shared_gt_tables)
        span_rows.extend(
            {
                "fixture": fixture.name,
                "video_id": fixture.video_id,
                "span_id": span_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
            }
            for span_id, (start_frame, end_frame) in enumerate(data.spans)
        )
        for rally, (boundary, span_id) in zip(data.gt_rallies, data.boundaries, strict=True):
            row = _empty_row(data, rally, boundary, span_id)
            if boundary is RallyBoundary.COVERED and span_id is not None:
                _populate_covered_row(row, data, rally, span_id, case_paths, point_rows)
            rows.append(row)
        print(f"{fixture.name}: measured {len(data.gt_rallies)} ShuttleSet rallies")
        del data
        gc.collect()

    features = pd.DataFrame(rows)
    if len(features) != 292 or features[["video_id", "set_id", "rally"]].duplicated().any():
        raise ValueError("feature table must contain 292 unique ShuttleSet rallies")

    covered_mask = features["boundary"].eq(RallyBoundary.COVERED.value)
    covered = features.loc[covered_mask]
    multiplicity = covered.groupby(["fixture", "span_id"])["rally"].transform("size").astype(int)
    features.loc[covered_mask, "span_multiplicity"] = multiplicity.to_numpy()
    primary_mask = covered_mask & features["span_multiplicity"].eq(1)
    features.loc[primary_mask, "primary_one_to_one"] = True
    features.loc[primary_mask, "population_detail"] = "primary_239"
    features.loc[covered_mask & ~primary_mask, "population_detail"] = "covered_merged_sensitivity"
    features.loc[features["boundary"].eq(RallyBoundary.SPLIT.value), "population_detail"] = (
        "end_to_end_split"
    )
    features.loc[features["boundary"].eq(RallyBoundary.MISSED.value), "population_detail"] = (
        "end_to_end_missed"
    )

    span_multiplicities = covered.groupby(["fixture", "span_id"]).size()
    primary_by_fixture = features.loc[primary_mask].groupby("fixture").size().to_dict()
    expected_primary_by_fixture = {"sset_01": 104, "sset_15": 84, "sset_21": 51}
    if (
        len(covered) != 249
        or len(span_multiplicities) != 244
        or int((span_multiplicities == 1).sum()) != 239
        or int((span_multiplicities == 2).sum()) != 5
        or int(span_multiplicities[span_multiplicities == 2].sum()) != 10
        or bool((span_multiplicities > 2).any())
        or primary_by_fixture != expected_primary_by_fixture
    ):
        raise ValueError("rebuilt rally mapping differs from the approved 292/249/244/239 contract")

    spans = pd.DataFrame(span_rows)
    if spans[["fixture", "span_id"]].duplicated().any():
        raise ValueError("predicted span keys must be unique within each fixture")
    path_points = pd.DataFrame(point_rows)
    return features, spans, path_points, case_paths


def _parse_guesses(value: object) -> list[point_winner.Half | None]:
    """Parse the compact direct-contact sequence stored in the feature table."""
    if not isinstance(value, str) or not value:
        return []
    guesses: list[point_winner.Half | None] = []
    for item in value.split("|"):
        guesses.append(None if item == "Unknown" else normalise_half(item))
    return guesses


def apply_fixed_rules(features: pd.DataFrame) -> pd.DataFrame:
    """Apply both predeclared rules unchanged to both path masks."""
    results = features.copy()
    server_predictions: dict[str, list[str | None]] = {
        f"{variant}_{rule}_server": []
        for variant in PATH_VARIANTS
        for rule in ("historical", "robust_trend")
    }
    evidence_servers: list[str | None] = []
    parity_servers: list[str | None] = []
    labelled_servers: list[str | None] = []
    anchor_fallback_refit_servers: list[str | None] = []
    labelled_final_changed: list[bool] = []

    for _, row in results.iterrows():
        anchor_player = normalise_half(row["anchor_player"])
        for variant in PATH_VARIANTS:
            for rule in ("historical", "robust_trend"):
                detected = bool(row[f"{variant}_{rule}_incoming"])
                inferred_server = (
                    other_half(anchor_player)
                    if detected and anchor_player is not None
                    else anchor_player
                )
                server_predictions[f"{variant}_{rule}_server"].append(
                    _half_text(inferred_server)
                )

        main_detected = bool(row[f"{PRIMARY_PATH_VARIANT}_robust_trend_incoming"])
        main_path_eligible = bool(row[f"{PRIMARY_PATH_VARIANT}_common_path_eligible"])
        main_server = (
            other_half(anchor_player)
            if main_detected and anchor_player is not None
            else anchor_player
        )
        evidence_servers.append(_half_text(main_server) if main_path_eligible else None)

        guesses = _parse_guesses(row["direct_contact_guesses"])
        natural_final = point_winner.fit_alternation(guesses)
        if not main_detected or anchor_player is None:
            natural_first = _fit_first_player(guesses)
            parity_servers.append(_half_text(natural_first))
            labelled_servers.append(_half_text(natural_first))
            anchor_fallback_refit_servers.append(_half_text(anchor_player))
            labelled_final_changed.append(False)
            continue

        parity_guesses = [None, *guesses]
        labelled_guesses = [other_half(anchor_player), *guesses]
        parity_servers.append(_half_text(_fit_first_player(parity_guesses)))
        labelled_final = point_winner.fit_alternation(labelled_guesses)
        labelled_first = _fit_first_player(labelled_guesses)
        labelled_servers.append(_half_text(labelled_first))
        anchor_fallback_refit_servers.append(
            _half_text(labelled_first if labelled_first is not None else anchor_player)
        )
        labelled_final_changed.append(labelled_final != natural_final)

    for column, values in server_predictions.items():
        results[column] = values
    results["incoming_motion_found"] = results[
        f"{PRIMARY_PATH_VARIANT}_robust_trend_incoming"
    ]
    results["assume_first_contact_is_serve"] = results["anchor_player"]
    results["motion_rule_server"] = results[f"{PRIMARY_PATH_VARIANT}_robust_trend_server"]
    results["evidence_only_server"] = evidence_servers
    results["missing_contact_refit_server"] = parity_servers
    results["inferred_player_refit_server"] = labelled_servers
    results["anchor_fallback_refit_server"] = anchor_fallback_refit_servers
    results["inferred_player_vote_changed_final_fit"] = labelled_final_changed
    prediction_columns = [
        "assume_first_contact_is_serve",
        "motion_rule_server",
        "evidence_only_server",
        "missing_contact_refit_server",
        "inferred_player_refit_server",
        "anchor_fallback_refit_server",
        *server_predictions,
    ]
    for column in prediction_columns:
        results[f"{column}_correct"] = results[column] == results["gt_server"]
    return results


def _binary_rule_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, int | float]:
    """Return explicit confusion counts for one fixed incoming rule."""
    true_positive = int(np.count_nonzero(predicted & truth))
    false_positive = int(np.count_nonzero(predicted & ~truth))
    false_negative = int(np.count_nonzero(~predicted & truth))
    true_negative = int(np.count_nonzero(~predicted & ~truth))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0
    )
    recall = true_positive / (true_positive + false_negative)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_rule_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Score the four fixed rule/mask arms on unique primary ±10 truth."""
    unique_truth = results[
        results["primary_one_to_one"].astype(bool)
        & results["anchor_tolerance_10_label"].isin(["contact_1", "contact_2"])
        & results["anchor_tolerance_10_in_window_count"].eq(1)
    ]
    if unique_truth.empty:
        raise ValueError("the primary set has no unique ±10 contact-1/contact-2 truth")

    scopes: list[tuple[str, pd.DataFrame]] = [("global", unique_truth)]
    scopes.extend((str(fixture), group) for fixture, group in unique_truth.groupby("fixture"))
    rows: list[dict[str, object]] = []
    for scope, frame in scopes:
        truth = frame["anchor_tolerance_10_label"].eq("contact_2").to_numpy()
        for variant in PATH_VARIANTS:
            for rule in ("historical", "robust_trend"):
                predicted = frame[f"{variant}_{rule}_incoming"].astype(bool).to_numpy()
                eligibility_column = (
                    f"{variant}_historical_path_eligible"
                    if rule == "historical"
                    else f"{variant}_common_path_eligible"
                )
                rows.append(
                    {
                        "scope": scope,
                        "population": "primary_239_unique_tolerance_10_truth",
                        "path_definition": variant,
                        "rule": rule,
                        "n_truth": len(frame),
                        "gt_serves": int(np.count_nonzero(~truth)),
                        "gt_first_returns": int(np.count_nonzero(truth)),
                        "common_paths_eligible": int(
                            frame[f"{variant}_common_path_eligible"].astype(bool).sum()
                        ),
                        "rule_paths_eligible": int(frame[eligibility_column].astype(bool).sum()),
                        "incoming_calls": int(np.count_nonzero(predicted)),
                        **_binary_rule_metrics(truth, predicted),
                    }
                )
    return pd.DataFrame(rows)


def classification_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, object]:
    """Score Top/Bot server labels while keeping abstentions in the denominator."""
    truth = frame["gt_server"].astype(str).to_numpy()
    predictions = frame[prediction_column].fillna("Unknown").astype(str).to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predictions,
        labels=[point_winner.Half.TOP.value, point_winner.Half.BOT.value],
        zero_division=0,
    )
    return {
        "n": len(frame),
        "known": int(np.count_nonzero(predictions != "Unknown")),
        "correct": int(np.count_nonzero(predictions == truth)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(np.mean(f1)),
        "top": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "bot": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }


def _score_methods(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Score every server answer under plain report labels."""
    return {
        "old alternating fit": classification_metrics(frame, "baseline_server"),
        "anchor player": classification_metrics(frame, "assume_first_contact_is_serve"),
        "historical rule, recurrence mask": classification_metrics(
            frame, "recurrence_clean_historical_server"
        ),
        "0.05-BH trend rule, recurrence mask": classification_metrics(
            frame, "recurrence_clean_robust_trend_server"
        ),
        "historical rule, recurrence plus producer mask": classification_metrics(
            frame, "producer_original_historical_server"
        ),
        "0.05-BH trend rule, recurrence plus producer mask": classification_metrics(
            frame, "producer_original_robust_trend_server"
        ),
        "0.05-BH trend evidence only": classification_metrics(frame, "evidence_only_server"),
        "0.05-BH trend then prepend unknown player": classification_metrics(
            frame, "missing_contact_refit_server"
        ),
        "0.05-BH trend then prepend other player": classification_metrics(
            frame, "inferred_player_refit_server"
        ),
        "0.05-BH trend then like-for-like refit": classification_metrics(
            frame, "anchor_fallback_refit_server"
        ),
    }


def _global_and_by_fixture(
    frame: pd.DataFrame,
    summarise: Any,
) -> dict[str, object]:
    """Apply one count summary globally and to each requested fixture."""
    by_fixture = {
        str(fixture): summarise(group)
        for fixture, group in frame.groupby("fixture", sort=True)
    }
    return {"global": summarise(frame), "by_fixture": by_fixture}


def _alignment_counts(frame: pd.DataFrame) -> dict[str, object]:
    """Count nearest-stroke labels and ambiguity at every tolerance."""
    rows: dict[str, object] = {}
    for tolerance in CONTACT_TOLERANCES_BASE30:
        prefix = f"anchor_tolerance_{tolerance}"
        labels = frame[f"{prefix}_label"].value_counts(dropna=False)
        rows[str(tolerance)] = {
            "n": len(frame),
            "labels": {str(label): int(count) for label, count in labels.items()},
            "multiple": int(frame[f"{prefix}_multiple"].astype(bool).sum()),
            "unique_contact_1": int(
                (
                    frame[f"{prefix}_label"].eq("contact_1")
                    & frame[f"{prefix}_in_window_count"].eq(1)
                ).sum()
            ),
            "unique_contact_2": int(
                (
                    frame[f"{prefix}_label"].eq("contact_2")
                    & frame[f"{prefix}_in_window_count"].eq(1)
                ).sum()
            ),
        }
    return rows


def _path_counts(frame: pd.DataFrame) -> dict[str, object]:
    """Count evidence states and fixed calls for both masks."""
    rows: dict[str, object] = {}
    for variant in PATH_VARIANTS:
        rows[variant] = {
            "n": len(frame),
            "anchors": int(frame["anchor_frame"].notna().sum()),
            "anchors_with_player": int(frame["anchor_player"].notna().sum()),
            "selected_paths": int(frame[f"{variant}_selected_path"].astype(bool).sum()),
            "path_available": int(frame[f"{variant}_path_available"].astype(bool).sum()),
            "common_path_eligible": int(
                frame[f"{variant}_common_path_eligible"].astype(bool).sum()
            ),
            "historical_path_eligible": int(
                frame[f"{variant}_historical_path_eligible"].astype(bool).sum()
            ),
            "historical_incoming": int(
                frame[f"{variant}_historical_incoming"].astype(bool).sum()
            ),
            "robust_trend_incoming": int(
                frame[f"{variant}_robust_trend_incoming"].astype(bool).sum()
            ),
        }
    return rows


def _sequence_counts(frame: pd.DataFrame) -> dict[str, object]:
    """Count later-contact outcomes for primary ±10-unmatched anchors."""
    unmatched = frame[
        frame["anchor_frame"].notna() & frame["anchor_tolerance_10_label"].eq("unmatched")
    ]
    serve = unmatched["later_serve_within_tolerance"].astype(bool)
    first_return = unmatched["later_first_return_within_tolerance"].astype(bool)
    any_match = unmatched["first_gt_match_rank"].notna()
    rank_counts = unmatched["first_gt_match_rank"].dropna().astype(int).value_counts().sort_index()
    return {
        "anchors_unmatched_at_tolerance_10": len(unmatched),
        "sequence_checked": int(unmatched["unmatched_sequence_checked"].astype(bool).sum()),
        "later_serve_match": int(serve.sum()),
        "no_later_serve_but_first_return_match": int((~serve & first_return).sum()),
        "other_later_gt_match": int((~serve & ~first_return & any_match).sum()),
        "no_later_gt_match": int((~any_match).sum()),
        "first_gt_match_rank": {str(rank): int(count) for rank, count in rank_counts.items()},
        "first_match_multiple": int(unmatched["first_gt_match_multiple"].astype(bool).sum()),
        "reused_gt_ordinal": int(unmatched["reused_gt_ordinal"].astype(bool).sum()),
    }


def _population_server_scores(frame: pd.DataFrame) -> dict[str, object]:
    """Score server methods globally and by fixture for one population."""
    return _global_and_by_fixture(frame, _score_methods)


def build_metrics(results: pd.DataFrame, rule_rows: pd.DataFrame) -> dict[str, object]:
    """Collect the corrected denominators, funnels and fixed-rule results."""
    populations = {
        "all_292_end_to_end": results,
        "covered_249_merge_sensitivity": results[
            results["boundary"].eq(RallyBoundary.COVERED.value)
        ],
        "primary_239_one_to_one": results[results["primary_one_to_one"].astype(bool)],
    }
    population_counts = {
        name: {
            "global": len(frame),
            "by_fixture": {
                str(fixture): len(group)
                for fixture, group in frame.groupby("fixture", sort=True)
            },
        }
        for name, frame in populations.items()
    }
    primary = populations["primary_239_one_to_one"]
    return {
        "question": (
            "Does the shuttle show a clear approach towards the contact player beyond ordinary "
            "track wobble, and what does that imply for anchor and server attribution?"
        ),
        "population_counts": population_counts,
        "rules": {
            "historical": {
                "minimum_path_frames": MIN_PATH_FRAMES,
                "maximum_frames_to_contact_base30": MAX_FRAMES_TO_CONTACT_BASE30,
                "maximum_largest_step_ratio": MAX_LARGEST_STEP_RATIO,
                "minimum_total_movement_bh": MIN_TOTAL_MOVEMENT_BH,
                "minimum_net_closure_bh": PRIMARY_MIN_NET_CLOSURE_BH,
                "minimum_closing_fraction": HISTORICAL_MIN_CLOSING_FRACTION,
                "provenance": "introduced and selected within the historical analysis",
            },
            "robust_trend": {
                "minimum_path_frames": MIN_PATH_FRAMES,
                "maximum_frames_to_contact_base30": MAX_FRAMES_TO_CONTACT_BASE30,
                "maximum_largest_step_ratio": MAX_LARGEST_STEP_RATIO,
                "minimum_fitted_decrease_bh": ROBUST_TREND_MIN_DECREASE_BH,
                "provenance": "engineering judgement fixed before corrected scoring",
                "residual_rms_and_trend_to_jitter_are_diagnostic_only": True,
            },
        },
        "alignment": {
            name: _global_and_by_fixture(frame, _alignment_counts)
            for name, frame in populations.items()
        },
        "path_funnel": {
            name: _global_and_by_fixture(frame, _path_counts)
            for name, frame in populations.items()
        },
        "unmatched_anchor_sequences": _global_and_by_fixture(primary, _sequence_counts),
        "fixed_rule_results": json.loads(rule_rows.to_json(orient="records")),
        "server_scores": {
            name: _population_server_scores(frame) for name, frame in populations.items()
        },
    }


def write_json_gz(path: Path, payload: dict[str, object]) -> None:
    """Write compressed JSON using the repository's required format."""
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def build_trend_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    """Keep continuous trend evidence for every primary unique ±10 truth row."""
    truth = results[
        results["primary_one_to_one"].astype(bool)
        & results["anchor_tolerance_10_label"].isin(["contact_1", "contact_2"])
        & results["anchor_tolerance_10_in_window_count"].eq(1)
    ]
    rows: list[dict[str, object]] = []
    for _, rally in truth.iterrows():
        is_first_return = rally["anchor_tolerance_10_label"] == "contact_2"
        for variant in PATH_VARIANTS:
            incoming = bool(rally[f"{variant}_robust_trend_incoming"])
            rows.append(
                {
                    "fixture": rally["fixture"],
                    "video_id": int(rally["video_id"]),
                    "set_id": rally["set_id"],
                    "rally": int(rally["rally"]),
                    "path_definition": variant,
                    "gt_anchor_identity": "first_return" if is_first_return else "serve",
                    "selected_path": bool(rally[f"{variant}_selected_path"]),
                    "common_path_eligible": bool(rally[f"{variant}_common_path_eligible"]),
                    "path_frames": int(rally[f"{variant}_path_frames"]),
                    "fitted_decrease_bh": rally[f"{variant}_fitted_decrease_bh"],
                    "residual_rms_bh": rally[f"{variant}_residual_rms_bh"],
                    "trend_to_jitter": rally[f"{variant}_trend_to_jitter"],
                    "incoming_call": incoming,
                    "call_correct": incoming == is_first_return,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Build the corrected row tables and fixed-rule summaries."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, spans, path_points, _case_paths = build_feature_rows()
    results = apply_fixed_rules(features)
    rule_rows = build_rule_rows(results)
    diagnostics = build_trend_diagnostics(results)
    metrics = build_metrics(results, rule_rows)
    results.to_csv(OUTPUT_DIR / "rallies.csv.gz", index=False, compression="gzip")
    spans.to_csv(OUTPUT_DIR / "spans.csv.gz", index=False, compression="gzip")
    path_points.to_csv(OUTPUT_DIR / "path_points.csv.gz", index=False, compression="gzip")
    rule_rows.to_csv(OUTPUT_DIR / "fixed_rules.csv.gz", index=False, compression="gzip")
    diagnostics.to_csv(OUTPUT_DIR / "trend_diagnostics.csv.gz", index=False, compression="gzip")
    write_json_gz(OUTPUT_DIR / "metrics.json.gz", metrics)
    write_final_outputs(
        results,
        path_points,
        rule_rows,
        diagnostics,
        metrics,
        PLOT_DIR,
        RUN_DIR / "report.md",
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
