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
    classify_anchor_frame,
    closest_pre_contact_run,
    first_player_from_final_half,
    fit_path,
    measure_incoming_motion,
)

from annotator import point_winner
from annotator.calibration.fixtures import FIXTURES
from annotator.calibration.gt_scoring import canonical_tolerance, load_gt_tables
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
MIN_PATH_FRAMES = 5
MIN_TOTAL_MOVEMENT_BH = 0.25
MAX_LARGEST_STEP_RATIO = 4.0
PRIMARY_MIN_NET_CLOSURE_BH = 0.25
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
    row: dict[str, object] = {
        "fixture": data.fixture.name,
        "video_id": data.fixture.video_id,
        "fps": data.fixture.fps,
        "set_id": rally.set_id,
        "rally": rally.rally,
        "boundary": boundary.value,
        "span_id": span_id,
        "gt_server": _half_text(truth["gt_server"]),
        "gt_receiver": _half_text(truth["gt_receiver"]),
        "gt_first_frame": truth["gt_first_frame"],
        "gt_second_frame": truth["gt_second_frame"],
        "baseline_server": None,
        "baseline_correct": False,
        "frozen_server_failure": False,
        "accepted_contact_count": 0,
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
    }
    for variant in PATH_VARIANTS:
        row.update(
            {
                f"{variant}_path_start": None,
                f"{variant}_path_end": None,
                f"{variant}_path_frames": 0,
                f"{variant}_frames_to_contact": None,
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
    if run.end - run.start < 2:
        return None

    motion = measure_incoming_motion(
        data.sticky.distances_per_slot[run_slice, slot],
        data.track[run_slice, :2],
        data.sticky.bbox_height[run_slice, slot],
        data.fixture.resolution,
    )
    fit = fit_path(data.track[run_slice, :2])
    row[f"{variant}_start_distance_bh"] = motion.start_distance_bh
    row[f"{variant}_end_distance_bh"] = motion.end_distance_bh
    row[f"{variant}_net_closure_bh"] = motion.net_closure_bh
    row[f"{variant}_movements_towards_player"] = motion.closing_fraction
    row[f"{variant}_total_movement_bh"] = motion.total_movement_bh
    row[f"{variant}_largest_step_ratio"] = motion.largest_step_ratio
    row[f"{variant}_linear_rmse"] = fit.linear_rmse
    row[f"{variant}_quadratic_rmse"] = fit.quadratic_rmse
    row[f"{variant}_quadratic_improvement"] = fit.quadratic_improvement
    path_available = motion.n_frames >= MIN_PATH_FRAMES and run.frames_to_contact <= maximum_frames_to_contact
    path_quality_pass = (
        path_available
        and motion.total_movement_bh >= MIN_TOTAL_MOVEMENT_BH
        and motion.largest_step_ratio <= MAX_LARGEST_STEP_RATIO
    )
    row[f"{variant}_path_available"] = path_available
    row[f"{variant}_path_quality_pass"] = path_quality_pass
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
) -> None:
    """Add direct contact attribution and pre-contact motion for one covered rally."""
    gt_server = normalise_half(row["gt_server"])
    baseline_server = normalise_half(data.annotations["fitted_first_all"][span_id])
    row["baseline_server"] = _half_text(baseline_server)
    row["baseline_correct"] = baseline_server == gt_server
    row["frozen_server_failure"] = baseline_server != gt_server

    accepted = sorted(data.accepted_by_span.get(span_id, []))
    raw_contacts = data.raw_contacts_by_span.get(span_id, [])
    row["accepted_contact_count"] = len(accepted)
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
    row["anchor_gt_match"] = classify_anchor_frame(
        anchor,
        rally.stroke_frames,
        canonical_tolerance(data.fixture.fps),
    )
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
        )
        if evidence is not None:
            for name, values in evidence.items():
                case_paths[key][f"{variant}_{name}"] = values
    primary_start = row[f"{PRIMARY_PATH_VARIANT}_path_start"]
    row["path_reaches_scene_start"] = primary_start is not None and int(primary_start) == segment_start


def build_feature_rows() -> tuple[pd.DataFrame, dict[tuple[int, str, int], dict[str, np.ndarray]]]:
    """Build one result row per ShuttleSet rally."""
    shared_gt_tables = load_gt_tables()
    rows: list[dict[str, object]] = []
    case_paths: dict[tuple[int, str, int], dict[str, np.ndarray]] = {}
    for fixture in FIXTURES:
        print(f"{fixture.name}: loading frozen data and rebuilding player geometry")
        data = load_video_data(fixture, shared_gt_tables)
        for rally, (boundary, span_id) in zip(data.gt_rallies, data.boundaries, strict=True):
            row = _empty_row(data, rally, boundary, span_id)
            if boundary is RallyBoundary.COVERED and span_id is not None:
                _populate_covered_row(row, data, rally, span_id, case_paths)
            rows.append(row)
        print(f"{fixture.name}: measured {len(data.gt_rallies)} ShuttleSet rallies")
        del data
        gc.collect()

    features = pd.DataFrame(rows)
    if len(features) != 292 or features[["video_id", "set_id", "rally"]].duplicated().any():
        raise ValueError("feature table must contain 292 unique ShuttleSet rallies")
    return features, case_paths


def build_threshold_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Measure first-return precision and recall at every plain percentage threshold."""
    labelled = features[features["anchor_gt_match"].isin(["contact_1", "contact_2"])]
    if labelled.empty:
        raise ValueError("no anchors have clear contact-1 or contact-2 truth")
    truth = labelled["anchor_gt_match"].eq("contact_2").to_numpy()
    rows: list[dict[str, object]] = []
    for variant in PATH_VARIANTS:
        quality = labelled[f"{variant}_path_quality_pass"].astype(bool).to_numpy()
        closure = labelled[f"{variant}_net_closure_bh"].to_numpy(dtype=float)
        closing_fraction = labelled[f"{variant}_movements_towards_player"].to_numpy(dtype=float)
        for minimum_closure in CLOSURE_COMPARISONS_BH:
            for percentage in INCOMING_PERCENTAGES:
                predicted = (
                    quality
                    & (closure >= minimum_closure)
                    & (closing_fraction >= percentage / 100.0)
                )
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
                f1 = (
                    0.0
                    if precision + recall == 0
                    else 2.0 * precision * recall / (precision + recall)
                )
                rows.append(
                    {
                        "path_definition": variant,
                        "minimum_net_closure_bh": minimum_closure,
                        "minimum_movements_towards_player_percent": int(percentage),
                        "clear_contact_1": int(np.count_nonzero(~truth)),
                        "clear_contact_2": int(np.count_nonzero(truth)),
                        "paths_passing_quality_checks": int(np.count_nonzero(quality)),
                        "tp": true_positive,
                        "fp": false_positive,
                        "fn": false_negative,
                        "tn": true_negative,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                )
    return pd.DataFrame(rows)


def choose_threshold(thresholds: pd.DataFrame) -> pd.Series:
    """Choose highest first-return F1, then precision, then the stricter percentage."""
    primary = thresholds[
        (thresholds["path_definition"] == PRIMARY_PATH_VARIANT)
        & np.isclose(thresholds["minimum_net_closure_bh"], PRIMARY_MIN_NET_CLOSURE_BH)
    ]
    if primary.empty:
        raise ValueError("primary first-return threshold curve is empty")
    ordered = primary.sort_values(
        ["f1", "precision", "minimum_movements_towards_player_percent"],
        ascending=[False, False, False],
    )
    return ordered.iloc[0]


def _return_prediction(row: pd.Series, variant: str, closure_bh: float, percentage: int) -> bool:
    """Apply the declared incoming-motion rule to one rally row."""
    return bool(
        row[f"{variant}_path_quality_pass"]
        and float(row[f"{variant}_net_closure_bh"]) >= closure_bh
        and float(row[f"{variant}_movements_towards_player"]) >= percentage / 100.0
    )


def _parse_guesses(value: object) -> list[point_winner.Half | None]:
    """Parse the compact direct-contact sequence stored in the feature table."""
    if not isinstance(value, str) or not value:
        return []
    guesses: list[point_winner.Half | None] = []
    for item in value.split("|"):
        guesses.append(None if item == "Unknown" else normalise_half(item))
    return guesses


def apply_selected_rule(features: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    """Apply the incoming decision and both one-shot prepend experiments."""
    variant = str(selected["path_definition"])
    minimum_closure = float(selected["minimum_net_closure_bh"])
    percentage = int(selected["minimum_movements_towards_player_percent"])
    results = features.copy()
    return_detected: list[bool] = []
    forced_servers: list[str | None] = []
    evidence_servers: list[str | None] = []
    parity_servers: list[str | None] = []
    labelled_servers: list[str | None] = []
    labelled_final_changed: list[bool] = []

    for _, row in results.iterrows():
        anchor_player = normalise_half(row["anchor_player"])
        path_measured = bool(row[f"{variant}_path_quality_pass"])
        detected = anchor_player is not None and _return_prediction(
            row,
            variant,
            minimum_closure,
            percentage,
        )
        return_detected.append(detected)

        inferred_server = other_half(anchor_player) if detected and anchor_player is not None else anchor_player
        forced_servers.append(_half_text(inferred_server))
        evidence_servers.append(_half_text(inferred_server) if path_measured else None)

        guesses = _parse_guesses(row["direct_contact_guesses"])
        natural_final = point_winner.fit_alternation(guesses)
        if not detected or anchor_player is None:
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

    results["incoming_motion_found"] = return_detected
    results["forced_anchor_server"] = forced_servers
    results["evidence_only_server"] = evidence_servers
    results["missing_contact_refit_server"] = parity_servers
    results["inferred_player_refit_server"] = labelled_servers
    results["inferred_player_vote_changed_final_fit"] = labelled_final_changed
    for column in (
        "forced_anchor_server",
        "evidence_only_server",
        "missing_contact_refit_server",
        "inferred_player_refit_server",
    ):
        results[f"{column}_correct"] = results[column] == results["gt_server"]
    return results


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
        "released alternating fit": classification_metrics(frame, "baseline_server"),
        "first accepted contact player": classification_metrics(frame, "forced_anchor_server"),
        "motion evidence only": classification_metrics(frame, "evidence_only_server"),
        "add one missing contact": classification_metrics(frame, "missing_contact_refit_server"),
        "add one contact by inferred player": classification_metrics(frame, "inferred_player_refit_server"),
    }


def build_metrics(results: pd.DataFrame, selected: pd.Series) -> dict[str, object]:
    """Collect direct answers, denominators and server scores."""
    covered = results[results["boundary"] == RallyBoundary.COVERED.value]
    failures = covered[covered["frozen_server_failure"].astype(bool)]
    clear = results[results["anchor_gt_match"].isin(["contact_1", "contact_2"])]
    clear_returns = clear[clear["anchor_gt_match"] == "contact_2"]
    triggers = covered[covered["incoming_motion_found"].astype(bool)]
    triggered_returns = clear_returns[clear_returns["incoming_motion_found"].astype(bool)]
    no_quality_path = covered[~covered[f"{PRIMARY_PATH_VARIANT}_path_quality_pass"].astype(bool)]
    earlier_raw = covered[covered["earlier_raw_candidates"] > 0]
    return {
        "question": (
            "Do we see the shuttle travelling towards the player at the first accepted contact, "
            "and does adding one earlier shot by the other player make the server attribution correct?"
        ),
        "selected_rule": {
            "maximum_lookback_base30_frames": LOOKBACK_BASE30_FRAMES,
            "minimum_path_frames": MIN_PATH_FRAMES,
            "maximum_frames_to_contact_base30": MAX_FRAMES_TO_CONTACT_BASE30,
            "minimum_total_movement_bh": MIN_TOTAL_MOVEMENT_BH,
            "maximum_largest_step_ratio": MAX_LARGEST_STEP_RATIO,
            "minimum_net_closure_bh": float(selected["minimum_net_closure_bh"]),
            "minimum_movements_towards_player_percent": int(
                selected["minimum_movements_towards_player_percent"]
            ),
            "selection": (
                "highest first-return F1; ties use higher precision, then a higher incoming-movement percentage"
            ),
            "in_sample_eda": True,
        },
        "first_return_test": {
            "clear_contact_1": int((clear["anchor_gt_match"] == "contact_1").sum()),
            "clear_contact_2": len(clear_returns),
            "paths_passing_quality_checks": int(
                clear[f"{PRIMARY_PATH_VARIANT}_path_quality_pass"].astype(bool).sum()
            ),
            "tp": int(selected["tp"]),
            "fp": int(selected["fp"]),
            "fn": int(selected["fn"]),
            "tn": int(selected["tn"]),
            "precision": float(selected["precision"]),
            "recall": float(selected["recall"]),
            "f1": float(selected["f1"]),
        },
        "counts": {
            "all_gt_rallies": len(results),
            "covered_rallies": len(covered),
            "frozen_server_failures": len(failures),
            "anchors_with_player": int(covered["anchor_player"].notna().sum()),
            "no_quality_path": len(no_quality_path),
            "incoming_motion_found": len(triggers),
            "clear_returns_found": len(triggered_returns),
            "earlier_raw_candidate_rallies": len(earlier_raw),
            "triggers_with_earlier_raw_candidate": int((triggers["earlier_raw_candidates"] > 0).sum()),
            "paths_reaching_court_scene_start": int(covered["path_reaches_scene_start"].astype(bool).sum()),
            "anchor_equal_distance_ties": int(covered["anchor_equal_distance_tie"].astype(bool).sum()),
        },
        "what_happens_when_motion_is_found": {
            "n": len(triggers),
            "released_correct": int(triggers["baseline_correct"].astype(bool).sum()),
            "first_contact_answer_correct": int(triggers["forced_anchor_server_correct"].astype(bool).sum()),
            "missing_contact_refit_correct": int(
                triggers["missing_contact_refit_server_correct"].astype(bool).sum()
            ),
            "inferred_player_refit_correct": int(
                triggers["inferred_player_refit_server_correct"].astype(bool).sum()
            ),
            "extra_vote_changed_final_fit": int(
                triggers["inferred_player_vote_changed_final_fit"].astype(bool).sum()
            ),
        },
        "scores": {
            "all_292": _score_methods(results),
            "covered": _score_methods(covered),
            "frozen_failures": _score_methods(failures),
            "clear_first_or_second_contact": _score_methods(clear),
        },
        "anchor_groups": {
            str(group): int(count)
            for group, count in results["anchor_gt_match"].value_counts(dropna=False).items()
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
        xlabel="Minimum consecutive shuttle movements towards the contact player",
        ylabel="Percentage",
        title=(
            "Does incoming motion identify a first return?\n"
            f"Path must finish at least {PRIMARY_MIN_NET_CLOSURE_BH:g} body heights closer"
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
        label="Displayed threshold",
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
    )
    axes[1].axhline(PRIMARY_MIN_NET_CLOSURE_BH, color="#333333", linestyle=":")
    axes[1].set(
        xlabel="Movements towards the contact player",
        ylabel="How much closer the path finished (body heights)",
        title="Direction and total closing distance",
    )
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.savefig(PLOT_DIR / "incoming_motion_measurements.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_server_accuracy(metrics: dict[str, object]) -> None:
    """Plot correct server counts for the main methods without hiding denominators."""
    scores = metrics["scores"]
    methods = (
        "released alternating fit",
        "first accepted contact player",
        "add one missing contact",
        "add one contact by inferred player",
    )
    short_labels = (
        "Released fit",
        "First contact player",
        "Add missing contact",
        "Add inferred player",
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
        "| Method | Correct | Known answers | Accuracy | Macro-F1 |",
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
    report = f"""# Incoming shuttle before the first accepted contact

## Answer

This experiment asks: did the shuttle travel towards the player at the first accepted contact, and if so, did adding one earlier shot by the other player make the server attribution correct?

There were {first_return['clear_contact_2']} clear cases where the first accepted contact matched ShuttleSet's second contact, meaning it was the first return. The incoming-motion rule found {first_return['tp']} of them and incorrectly called {first_return['fp']} known first contacts returns. This is {first_return['precision']:.1%} precision, {first_return['recall']:.1%} recall and {first_return['f1']:.3f} F1. The raw counts matter because the known-return group is small.

Across the {triggered['n']} covered rallies where incoming motion was found, the released alternating fit had the right server in {triggered['released_correct']}/{triggered['n']}. Adding one missing contact, without naming its player, was right in {triggered['missing_contact_refit_correct']}/{triggered['n']}. Adding the missing contact and assigning it to the other player was right in {triggered['inferred_player_refit_correct']}/{triggered['n']}.

The extra player label changed the fitted final player in {triggered['extra_vote_changed_final_fit']}/{triggered['n']} triggered rallies. Usually the change comes from correcting the contact count, not from the added player vote overpowering the later contacts.

## What was measured

The anchor is the earliest accepted geometry/impulse contact in each predicted rally. Its player comes directly from `attribute_half` at that frame. The old fitted server label is never used to select the player or measure the incoming path.

The script searches at most {rule['maximum_lookback_base30_frames']} base-30 frames before the anchor. It uses the closest continuous path in the same court scene. A path needs at least {rule['minimum_path_frames']} frames, at least {rule['minimum_total_movement_bh']} body heights of total movement, and no one-frame jump more than {rule['maximum_largest_step_ratio']} times its typical movement. The displayed rule also requires the path to finish {rule['minimum_net_closure_bh']} body heights closer and at least {rule['minimum_movements_towards_player_percent']}% of its movements to reduce distance to the contact player.

The displayed setting was chosen by first-return F1 on these same three videos. It is exploratory, not a held-out estimate.

No usable path was available in {counts['no_quality_path']}/{counts['covered_rallies']} covered rallies. The tables therefore show both a forced answer, which calls the anchor player the server when no return is found, and an evidence-only answer, which abstains when the path is unavailable.

## Server attribution

### All {counts['all_gt_rallies']} ShuttleSet rallies

{_score_table(scores['all_292'])}

### {counts['covered_rallies']} rallies covered by one predicted span

{_score_table(scores['covered'])}

### {counts['frozen_server_failures']} rallies where the released fit was wrong or missing

{_score_table(scores['frozen_failures'])}

## Checks and useful subsets

- Clear anchor truth: {first_return['clear_contact_1']} first contacts and {first_return['clear_contact_2']} second contacts.
- Clear rows with a path passing the fixed quality checks: {first_return['paths_passing_quality_checks']}.
- Covered rallies with an earlier rejected raw impulse: {counts['earlier_raw_candidate_rallies']}.
- Incoming-motion triggers with an earlier rejected raw impulse: {counts['triggers_with_earlier_raw_candidate']}.
- Paths that begin exactly when the court scene begins: {counts['paths_reaching_court_scene_start']}.
- Exact equal-distance anchor ties that would favour Top: {counts['anchor_equal_distance_ties']}.
- Case plots written: {case_plot_count}; all clear false positives plus a small sample of true positives and misses.

## Plots

- `outputs/plots/first_return_threshold.png`: the plain precision, recall and F1 curve.
- `outputs/plots/incoming_motion_measurements.png`: percentage of movements towards the contact player and net closing distance.
- `outputs/plots/server_accuracy.png`: correct server counts before and after adding a shot.
- `outputs/plots/cases/`: at most twelve labelled shuttle paths.

## Limits

Only {first_return['clear_contact_2']} clear first-return anchors are available, so the threshold can move with a few rallies. A recurrence-clean TrackNet point is not guaranteed to be real. The producer-original comparison remains in the compressed tables. The experiment infers the player and order of a missing shot; it does not recover the serve frame or prove that the serve itself was visible.
"""
    (RUN_DIR / "report.md").write_text(report, encoding="utf-8")


def write_json_gz(path: Path, payload: dict[str, object]) -> None:
    """Write compressed JSON using the repository's required format."""
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> None:
    """Run the two experiments and write their checked source tables and plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    features, case_paths = build_feature_rows()
    thresholds = build_threshold_rows(features)
    selected = choose_threshold(thresholds)
    results = apply_selected_rule(features, selected)
    metrics = build_metrics(results, selected)
    plot_threshold_curve(thresholds, selected)
    plot_motion_measurements(results, selected)
    plot_server_accuracy(metrics)
    case_plot_count = plot_selected_cases(results, case_paths)
    metrics["counts"]["case_plots"] = case_plot_count
    results.to_csv(OUTPUT_DIR / "rallies.csv.gz", index=False, compression="gzip")
    thresholds.to_csv(OUTPUT_DIR / "thresholds.csv.gz", index=False, compression="gzip")
    write_json_gz(OUTPUT_DIR / "metrics.json.gz", metrics)
    write_report(metrics, case_plot_count)
    print(RUN_DIR / "report.md")


if __name__ == "__main__":
    main()
