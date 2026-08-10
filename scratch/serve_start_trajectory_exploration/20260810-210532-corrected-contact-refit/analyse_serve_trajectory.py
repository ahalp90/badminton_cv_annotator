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
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
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
CASE_PLOT_DIR = PLOT_DIR / "cases"

LOOKBACK_BASE30_FRAMES = 30
MAX_FRAMES_TO_CONTACT_BASE30 = 2
CONTACT_TOLERANCES_BASE30 = (5, 10, 30)
CLOSURE_COMPARISONS_BH = (0.0, 0.25, 0.5, 1.0)
INCOMING_PERCENTAGES = np.arange(0, 101, 5, dtype=int)
PATH_VARIANTS = ("recurrence_clean", "producer_original")
PRIMARY_PATH_VARIANT = "recurrence_clean"

COLOURS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "purple": "#6A3D9A",
    "sky": "#56B4E9",
    "pink": "#CC79A7",
    "grey": "#777777",
}


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
            labelled_final_changed.append(False)
            continue

        parity_guesses = [None, *guesses]
        labelled_guesses = [other_half(anchor_player), *guesses]
        parity_servers.append(_half_text(_fit_first_player(parity_guesses)))
        labelled_final = point_winner.fit_alternation(labelled_guesses)
        labelled_servers.append(_half_text(_fit_first_player(labelled_guesses)))
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
    results["inferred_player_vote_changed_final_fit"] = labelled_final_changed
    prediction_columns = [
        "assume_first_contact_is_serve",
        "motion_rule_server",
        "evidence_only_server",
        "missing_contact_refit_server",
        "inferred_player_refit_server",
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
    }


def _trigger_summary(results: pd.DataFrame) -> dict[str, int]:
    """Count outcomes only where one path says the anchor was a return."""
    covered = results[results["boundary"] == RallyBoundary.COVERED.value]
    triggers = covered[covered["incoming_motion_found"].astype(bool)]
    return {
        "n": len(triggers),
        "released_correct": int(triggers["baseline_correct"].astype(bool).sum()),
        "direct_inference_correct": int(triggers["motion_rule_server_correct"].astype(bool).sum()),
        "missing_contact_refit_correct": int(
            triggers["missing_contact_refit_server_correct"].astype(bool).sum()
        ),
        "inferred_player_refit_correct": int(
            triggers["inferred_player_refit_server_correct"].astype(bool).sum()
        ),
        "extra_vote_changed_final_fit": int(
            triggers["inferred_player_vote_changed_final_fit"].astype(bool).sum()
        ),
    }


def _earlier_raw_comparison(results: pd.DataFrame) -> dict[str, int | float]:
    """Show what an automatic earlier-raw-candidate veto would do."""
    clear = results[results["anchor_gt_match"].isin(["contact_1", "contact_2"])]
    truth = clear["anchor_gt_match"].eq("contact_2").to_numpy()
    predicted = clear["incoming_motion_found"].astype(bool).to_numpy()
    no_earlier_raw = clear["earlier_raw_candidates"].eq(0).to_numpy()
    vetoed_prediction = predicted & no_earlier_raw
    true_positive = int(np.count_nonzero(vetoed_prediction & truth))
    false_positive = int(np.count_nonzero(vetoed_prediction & ~truth))
    false_negative = int(np.count_nonzero(~vetoed_prediction & truth))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
    }


def _per_video_detection(results: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return clear first-return counts separately for each video."""
    clear = results[results["anchor_gt_match"].isin(["contact_1", "contact_2"])]
    rows: dict[str, dict[str, int]] = {}
    for fixture, group in clear.groupby("fixture"):
        truth = group["anchor_gt_match"].eq("contact_2")
        predicted = group["incoming_motion_found"].astype(bool)
        rows[str(fixture)] = {
            "contact_1": int((~truth).sum()),
            "contact_2": int(truth.sum()),
            "quality_paths": int(group[f"{PRIMARY_PATH_VARIANT}_path_quality_pass"].astype(bool).sum()),
            "tp": int((truth & predicted).sum()),
            "fp": int((~truth & predicted).sum()),
            "fn": int((truth & ~predicted).sum()),
        }
    return rows


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


def plot_threshold_curve(thresholds: pd.DataFrame, selected: pd.Series) -> None:
    """Plot the one headline first-return threshold in percentages and counts."""
    curve = thresholds[
        (thresholds["path_definition"] == PRIMARY_PATH_VARIANT)
        & np.isclose(thresholds["minimum_net_closure_bh"], PRIMARY_MIN_NET_CLOSURE_BH)
    ]
    figure, axis = plt.subplots(figsize=(9.5, 6), constrained_layout=True)
    labels = {
        "precision": "Of the contacts called returns, how many were returns?",
        "recall": "Of the known returns, how many were found?",
        "f1": "Balance of precision and recall",
    }
    colours = {"precision": COLOURS["blue"], "recall": COLOURS["orange"], "f1": COLOURS["purple"]}
    for metric, label in labels.items():
        axis.plot(
            curve["minimum_movements_towards_player_percent"],
            curve[metric],
            label=label,
            color=colours[metric],
            linewidth=2.4,
        )
    selected_percentage = int(selected["minimum_movements_towards_player_percent"])
    axis.axvline(selected_percentage, color="#333333", linestyle="--", linewidth=1.5)
    axis.text(
        0.98,
        0.97,
        f"Displayed setting: {selected_percentage}%\n"
        f"TP {int(selected['tp'])}, FP {int(selected['fp'])}, FN {int(selected['fn'])}",
        ha="right",
        va="top",
        transform=axis.transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.9},
    )
    axis.set(
        xlim=(0, 100),
        ylim=(0, 1.02),
        xlabel="Minimum share of shuttle movements that reduce distance",
        ylabel="Precision, recall or F1",
        title=(
            "Does incoming motion identify a first return among 103 clear anchors?\n"
            f"All paths must finish at least {PRIMARY_MIN_NET_CLOSURE_BH:g} body heights closer"
        ),
    )
    axis.xaxis.set_major_formatter(PercentFormatter(100))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left")
    figure.savefig(PLOT_DIR / "first_return_threshold.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_motion_measurements(results: pd.DataFrame, selected: pd.Series) -> None:
    """Show the two plain incoming measurements for clear first and second contacts."""
    clear = results[results["anchor_gt_match"].isin(["contact_1", "contact_2"])]
    measured = clear[clear[f"{PRIMARY_PATH_VARIANT}_path_quality_pass"].astype(bool)]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), constrained_layout=True)
    groups = (
        ("contact_1", "Anchor was the serve", COLOURS["blue"]),
        ("contact_2", "Anchor was the first return", COLOURS["orange"]),
    )
    bins = np.linspace(0, 1, 11)
    for group, label, colour in groups:
        values = measured.loc[
            measured["anchor_gt_match"] == group,
            f"{PRIMARY_PATH_VARIANT}_movements_towards_player",
        ]
        axes[0].hist(values, bins=bins, alpha=0.6, color=colour, label=f"{label} (n={len(values)})")
    axes[0].axvline(
        float(selected["minimum_movements_towards_player_percent"]) / 100.0,
        color="#333333",
        linestyle="--",
        label="Rule requires at least this share",
    )
    axes[0].set(
        xlabel="Movements towards the contact player",
        ylabel="Rallies with a usable path",
        title="How consistently did the shuttle move closer?",
    )
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    for group, label, colour in groups:
        subset = measured[measured["anchor_gt_match"] == group]
        axes[1].scatter(
            subset[f"{PRIMARY_PATH_VARIANT}_movements_towards_player"],
            subset[f"{PRIMARY_PATH_VARIANT}_net_closure_bh"],
            color=colour,
            alpha=0.75,
            label=f"{label} (n={len(subset)})",
        )
    axes[1].axvline(
        float(selected["minimum_movements_towards_player_percent"]) / 100.0,
        color="#333333",
        linestyle="--",
        label="Required share moving closer",
    )
    axes[1].axhline(
        PRIMARY_MIN_NET_CLOSURE_BH,
        color="#333333",
        linestyle=":",
        label="Required net closure",
    )
    axes[1].set(
        xlabel="Movements towards the contact player",
        ylabel="How much closer the path finished (body heights)",
        title="Direction and total closing distance",
    )
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.suptitle("The 19 usable paths among 103 anchors with clear first/second-contact truth")
    figure.savefig(PLOT_DIR / "incoming_motion_measurements.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_server_accuracy(metrics: dict[str, object]) -> None:
    """Plot correct server counts for the main methods without hiding denominators."""
    scores = metrics["scores"]
    methods = (
        "old alternating fit",
        "use the anchor player",
        "use the other player when motion is incoming",
        "prepend a contact with unknown player",
        "prepend a contact by the other player",
    )
    short_labels = (
        "Old alternating\nfit",
        "Use anchor\nplayer",
        "Use other player\nwhen motion is incoming",
        "Prepend contact\nwith unknown player",
        "Prepend contact\nby other player",
    )
    datasets = (("all_292", "All GT rallies"), ("covered", "Covered rallies"))
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    x_positions = np.arange(len(methods), dtype=float)
    width = 0.36
    for data_index, (dataset_key, dataset_label) in enumerate(datasets):
        values = [float(scores[dataset_key][method]["accuracy"]) for method in methods]
        counts = [
            (int(scores[dataset_key][method]["correct"]), int(scores[dataset_key][method]["n"]))
            for method in methods
        ]
        positions = x_positions + (data_index - 0.5) * width
        bars = axis.bar(
            positions,
            values,
            width,
            label=dataset_label,
            color=(COLOURS["blue"] if data_index == 0 else COLOURS["orange"]),
        )
        for bar, (correct, total) in zip(bars, counts, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{correct}/{total}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axis.set(
        xticks=x_positions,
        xticklabels=short_labels,
        ylabel="Correct server attribution",
        ylim=(0, 1.02),
        title="Did adding the missing shot make the server attribution correct?",
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(PLOT_DIR / "server_accuracy.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_source_quality_comparison(metrics: dict[str, object]) -> None:
    """Compare the main and producer-original first-return counts directly."""
    main = metrics["first_return_test"]
    producer = metrics["producer_original_comparison"]
    labels = (
        "Exclude repeated-position\nwarnings",
        "Also exclude filled or\ninterpolated points",
    )
    count_names = ("tp", "fp", "fn")
    count_labels = ("Returns found", "Serves wrongly called returns", "Returns missed")
    colours = (COLOURS["blue"], COLOURS["pink"], COLOURS["orange"])
    x_positions = np.arange(len(labels), dtype=float)
    width = 0.23
    figure, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    for index, (name, label, colour) in enumerate(zip(count_names, count_labels, colours, strict=True)):
        values = (int(main[name]), int(producer[name]))
        bars = axis.bar(x_positions + (index - 1) * width, values, width, label=label, color=colour)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(value),
                ha="center",
                va="bottom",
            )
    axis.set(
        xticks=x_positions,
        xticklabels=labels,
        ylabel="Rallies among 103 anchors with clear truth",
        ylim=(0, max(int(main["tp"]), int(producer["tp"]), int(main["fn"]), int(producer["fn"])) + 5),
        title="Does excluding TrackNet's filled or interpolated points help?",
    )
    for x_position, values in zip(x_positions, (main, producer), strict=True):
        axis.text(
            x_position,
            axis.get_ylim()[1] - 0.4,
            f"Precision {float(values['precision']):.1%}\nRecall {float(values['recall']):.1%}",
            ha="center",
            va="top",
        )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(PLOT_DIR / "tracknet_source_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def _finite_median(points: np.ndarray) -> np.ndarray | None:
    """Return the median finite point in a path."""
    finite = points[np.isfinite(points).all(axis=1)]
    return np.median(finite, axis=0) if len(finite) else None


def plot_selected_cases(
    results: pd.DataFrame,
    case_paths: dict[tuple[int, str, int], dict[str, np.ndarray]],
) -> int:
    """Plot all clear false positives and a small sample of the other outcomes."""
    clear = results[results["anchor_gt_match"].isin(["contact_1", "contact_2"])]
    false_positive = clear[
        clear["incoming_motion_found"].astype(bool) & clear["anchor_gt_match"].eq("contact_1")
    ]
    true_positive = clear[
        clear["incoming_motion_found"].astype(bool) & clear["anchor_gt_match"].eq("contact_2")
    ].head(4)
    false_negative = clear[
        ~clear["incoming_motion_found"].astype(bool) & clear["anchor_gt_match"].eq("contact_2")
    ].head(4)
    cases = pd.concat([false_positive, true_positive, false_negative]).drop_duplicates(
        ["video_id", "set_id", "rally"]
    )
    cases = cases.head(12)
    CASE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in cases.iterrows():
        key = (int(row["video_id"]), str(row["set_id"]), int(row["rally"]))
        evidence = case_paths.get(key, {})
        path = evidence.get(f"{PRIMARY_PATH_VARIANT}_path")
        ankles = evidence.get(f"{PRIMARY_PATH_VARIANT}_anchor_ankles")
        figure, axis = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
        scatter = None
        if path is not None:
            frame_order = np.arange(len(path))
            scatter = axis.scatter(path[:, 0], path[:, 1], c=frame_order, cmap="viridis", s=45, zorder=3)
            axis.plot(path[:, 0], path[:, 1], color=COLOURS["grey"], linewidth=1, alpha=0.7)
            if ankles is not None:
                anchor_point = _finite_median(ankles)
                if anchor_point is not None:
                    axis.scatter(
                        anchor_point[0],
                        anchor_point[1],
                        marker="X",
                        s=150,
                        color=COLOURS["pink"],
                        label="Contact player's feet",
                    )
        else:
            axis.text(0.5, 0.5, "No usable pre-contact path", ha="center", va="center", transform=axis.transAxes)
        actual = "first return" if row["anchor_gt_match"] == "contact_2" else "serve"
        decision = "first return" if bool(row["incoming_motion_found"]) else "serve / no evidence"
        axis.set(
            xlim=(0, 1),
            ylim=(1, 0),
            aspect="equal",
            xlabel="Normalised image x",
            ylabel="Normalised image y",
            title=(
                f"{row['fixture']} {row['set_id']} rally {int(row['rally'])}\n"
                f"Annotated anchor: {actual}; motion decision: {decision}"
            ),
        )
        if ankles is not None:
            axis.legend(loc="best")
        if scatter is not None:
            figure.colorbar(scatter, ax=axis, label="Frame order before contact")
        filename = f"{row['fixture']}_{row['set_id']}_r{int(row['rally']):02d}.png"
        figure.savefig(CASE_PLOT_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return len(cases)


def _score_table(scores: dict[str, dict[str, object]]) -> str:
    """Return a readable Markdown table for one rally group."""
    lines = [
        "| Method | Correct | Predictions made | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in scores.items():
        lines.append(
            f"| {method} | {values['correct']}/{values['n']} | {values['known']}/{values['n']} | "
            f"{values['accuracy']:.1%} | {values['macro_f1']:.3f} |"
        )
    return "\n".join(lines)


def write_report(metrics: dict[str, object], case_plot_count: int) -> None:
    """Write the result in plain language, with the direct answer first."""
    first_return = metrics["first_return_test"]
    counts = metrics["counts"]
    triggered = metrics["what_happens_when_motion_is_found"]
    rule = metrics["selected_rule"]
    scores = metrics["scores"]
    producer = metrics["producer_original_comparison"]
    raw_veto = metrics["earlier_raw_candidate_veto_comparison"]
    video_rows = metrics["first_return_test_by_video"]
    evidence_covered = scores["covered"]["motion answer only; abstain without a usable path"]
    video_table_lines = [
        "| Video | Anchor was serve | Anchor was first return | Usable paths | Returns found | False calls | Returns missed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fixture, values in video_rows.items():
        video_table_lines.append(
            f"| {fixture} | {values['contact_1']} | {values['contact_2']} | {values['quality_paths']} | "
            f"{values['tp']} | {values['fp']} | {values['fn']} |"
        )
    video_table = "\n".join(video_table_lines)
    report = f"""# Incoming shuttle before the first accepted contact

## Answer

This experiment asks: did the shuttle travel towards the player at the first accepted contact, and if so, did adding one earlier shot by the other player make the server attribution correct?

ShuttleSet gives clear first-or-second-contact truth for {first_return['clear_contact_1'] + first_return['clear_contact_2']} anchors: {first_return['clear_contact_1']} were serves and {first_return['clear_contact_2']} were first returns. Only {first_return['paths_passing_quality_checks']} of those anchors had a path that passed the fixed quality checks. At the selected cut-off, the rule called {first_return['tp'] + first_return['fp']} anchors returns. {first_return['tp']} calls were right, {first_return['fp']} were false calls, and {first_return['fn']} returns were missed. This is {first_return['precision']:.1%} precision, {first_return['recall']:.1%} recall and {first_return['f1']:.3f} F1.

Across all covered rallies, the rule fired {triggered['n']} times: the {first_return['tp'] + first_return['fp']} clear cases above and {triggered['n'] - first_return['tp'] - first_return['fp']} anchors that did not match a unique first or second ShuttleSet contact. Directly naming the other player as server was right in {triggered['direct_inference_correct']}/{triggered['n']}. The released alternating fit was right in {triggered['released_correct']}/{triggered['n']}. Adding one missing contact with an unknown player was right in {triggered['missing_contact_refit_correct']}/{triggered['n']}. Adding the contact and assigning it to the other player was right in {triggered['inferred_player_refit_correct']}/{triggered['n']}.

The extra player label changed the fitted final player in {triggered['extra_vote_changed_final_fit']}/{triggered['n']} triggered rallies. Usually the change comes from correcting the contact count, not from the added player vote overpowering the later contacts.

The stricter path check excludes TrackNet points that the data producer filled or interpolated. It found {producer['tp']}/{first_return['clear_contact_2']} returns with {producer['fp']} false calls. That is {producer['precision']:.1%} precision, {producer['recall']:.1%} recall and {producer['f1']:.3f} F1, compared with {first_return['f1']:.3f} F1 for the main rule.

## What was measured

The anchor is the earliest accepted geometry/impulse contact in each predicted rally. Its player comes directly from `attribute_half` at that frame. The old fitted server label is never used to select the player or measure the incoming path.

The script searches at most {rule['maximum_lookback_base30_frames']} base-30 frames before the anchor. It uses the closest continuous path in the same court scene. A path needs at least {rule['minimum_path_frames']} frames, at least {rule['minimum_total_movement_bh']} body heights of total movement, and no one-frame jump more than {rule['maximum_largest_step_ratio']} times its typical movement. The displayed rule also requires the path to finish {rule['minimum_net_closure_bh']} body heights closer and at least {rule['minimum_movements_towards_player_percent']}% of its movements to reduce distance to the contact player.

The displayed setting was chosen by first-return F1 on these same three videos. It is exploratory, not a held-out estimate.

No usable path was available in {counts['no_quality_path']}/{counts['covered_rallies']} covered rallies. The forced motion rule names the anchor player when the incoming rule does not fire. The evidence-only version abstains when no usable path exists; with a usable path below the cut-off, it names the anchor player. It answered {evidence_covered['known']}/{evidence_covered['n']} covered rallies and was right in {evidence_covered['correct']}/{evidence_covered['known']} of those answers.

The two prepend rows change the alternating fit only when incoming motion is found. `Prepend unknown player` adds one place at the start but no player vote. `Prepend other player` adds the same place and supplies the inferred server as one vote. Otherwise both rows keep the ordinary alternating fit over the measured contacts.

## Server attribution

### All {counts['all_gt_rallies']} ShuttleSet rallies

{_score_table(scores['all_292'])}

### {counts['covered_rallies']} rallies covered by one predicted span

{_score_table(scores['covered'])}

### {counts['frozen_server_failures']} rallies where the released fit was wrong or missing

{_score_table(scores['frozen_failures'])}

`Correct` includes the full table denominator. An abstention therefore counts as incorrect. `Predictions made` shows how often each method supplied either Top or Bottom, which keeps the low-coverage evidence-only result from looking like a complete server rule.

The simplest complete rule is “use the player at the first accepted contact, unless incoming motion says the other player served”. Its row is labelled `use the other player when motion is incoming`.

## First-return result by video

{video_table}

## Checks and useful subsets

- Clear anchor truth: {first_return['clear_contact_1']} first contacts and {first_return['clear_contact_2']} second contacts, {first_return['clear_contact_1'] + first_return['clear_contact_2']} total.
- Clear anchors with a path passing the fixed quality checks: {first_return['paths_passing_quality_checks']}/{first_return['clear_contact_1'] + first_return['clear_contact_2']}.
- Covered rallies with an earlier rejected raw impulse: {counts['earlier_raw_candidate_rallies']}.
- Incoming-motion triggers with an earlier rejected raw impulse: {counts['triggers_with_earlier_raw_candidate']}.
- If every earlier rejected impulse were used as a veto, the result would become {raw_veto['true_positive']} returns found, {raw_veto['false_positive']} false calls and {raw_veto['false_negative']} returns missed. This is {raw_veto['precision']:.1%} precision and {raw_veto['recall']:.1%} recall. The veto is reported only as a comparison.
- Paths that begin exactly when the court scene begins: {counts['paths_reaching_court_scene_start']}.
- Exact equal-distance anchor ties that would favour Top: {counts['anchor_equal_distance_ties']}.
- Case plots written: {case_plot_count}; all clear false positives plus a small sample of true positives and misses.

## Plots

- `outputs/plots/first_return_threshold.png`: precision, recall and F1 as the required share of movements towards the contact player changes.
- `outputs/plots/incoming_motion_measurements.png`: percentage of movements towards the contact player and net closing distance.
- `outputs/plots/server_accuracy.png`: correct server counts before and after adding a shot.
- `outputs/plots/tracknet_source_comparison.png`: return counts with and without TrackNet's filled or interpolated points.
- `outputs/plots/cases/`: at most twelve labelled shuttle paths.

## Limits

Only {first_return['clear_contact_2']} clear first-return anchors are available, so the threshold can move with a few rallies. Excluding repeated-position warnings does not guarantee that a TrackNet point is real. The stricter source comparison remains in the compressed tables. The experiment infers the player and order of a missing shot; it does not recover the serve frame or prove that the serve itself was visible.
"""
    (RUN_DIR / "report.md").write_text(report, encoding="utf-8")


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
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
