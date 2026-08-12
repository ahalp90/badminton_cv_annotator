"""Committed ShuttleSet ground-truth scoring for the annotator chain."""
from __future__ import annotations

import argparse
import csv
import logging
import math
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from annotator.fps_constants import ScalingKind
from annotator.calibration.fixtures import (
    FIXTURES, REPO_ROOT, SHARED_FILES, Fixture, fixtures_root, verify_file,
    verify_run_video_fixture,
)
from annotator.run_video import AnnotatorResult, run_video
from annotator.inpaint_guard import code_counts, grade_track
from annotator import point_winner
from annotator.point_winner import Half, OTHER_HALF, SHIPPED_LANDING_FILTER_OPTIONS, Verdict
from annotator.replay_mask import _read_homography_rows
from annotator.calibration.scoring import (
    CANONICAL_CONTACT_TOLERANCE_BASE30,
    RallyBoundary,
    classify_all,
    greedy_match,
    load_gt_rallies,
    safe_f1,
    score_boundaries,
    score_contacts,
)
from shared.court import load_all_court_info

log = logging.getLogger(__name__)


# Re-pinned after the measured 2026-07-28 smoothing and re-entry default change
# (eee3e29). The tracked pre/post record is in
# docs/archive/completed_general_refactors/annotator_cleanup/w2_9_delta.diff.
# Floors read from here.
REFERENCE_SCORES = {
    'sset_01': {
        'covered_fraction': 0.9734513274336283,
        'covered': 110,
        'n_gt_rallies': 113,
        'split': 2,
        'missed': 1,
        'merged_spans': 4,
        'spurious_spans': 5,
        'start_offset_mean': -88.57272727272728,
        'start_offset_median': -28.0,
        'start_offset_n': 110,
        'end_offset_mean': 105.02727272727273,
        'end_offset_median': 56.0,
        'end_offset_n': 110,
        'ball_round_absdiff_mean': 3.890909090909091,
        'ball_round_absdiff_median': 2.0,
        'ball_round_absdiff_n': 110,
        'timing_abserr_mean': 2.1149012567324954,
        'timing_abserr_median': 2.0,
        'timing_abserr_n': 1114,
        'ball_round_primary': 0.18584070796460178,
        'ball_round_primary_correct': 21,
        'ball_round_primary_total': 113,
        'ball_round_covered': 0.19090909090909092,
        'ball_round_covered_correct': 21,
        'ball_round_covered_total': 110,
        'timing_primary_recall': 0.6788543570993297,
        'timing_primary_matched': 1114,
        'timing_primary_total': 1641,
        'timing_covered_recall': 0.6919254658385093,
        'timing_covered_matched': 1114,
        'timing_covered_total': 1610,
        'player_primary': 0.5486725663716814,
        'player_primary_correct': 62,
        'player_primary_total': 113,
        'player_covered': 0.5636363636363636,
        'player_covered_correct': 62,
        'player_covered_total': 110,
        'server_primary': 0.504424778761062,
        'server_primary_correct': 57,
        'server_primary_total': 113,
        'server_covered': 0.5181818181818182,
        'server_covered_correct': 57,
        'server_covered_total': 110,
        'hit_height_primary': 0.30956733698964045,
        'hit_height_primary_correct': 508,
        'hit_height_primary_total': 1641,
        'hit_height_covered': 0.315527950310559,
        'hit_height_covered_correct': 508,
        'hit_height_covered_total': 1610,
        'landing_primary': 0.3008849557522124,
        'landing_primary_correct': 34,
        'landing_primary_total': 113,
        'landing_covered': 0.3090909090909091,
        'landing_covered_correct': 34,
        'landing_covered_total': 110,
        'getpoint_primary': 0.41964285714285715,
        'getpoint_primary_correct': 47,
        'getpoint_primary_total': 112,
        'getpoint_covered': 0.43119266055045874,
        'getpoint_covered_correct': 47,
        'getpoint_covered_total': 109,
        'contact_f1': 0.6572271386430679,
        'contact_precision': 0.6369353916523728,
        'contact_recall': 0.6788543570993297,
        'contact_matches': 1114,
        'contact_filtered_total': 1749,
        'contact_gt_total': 1641,
        'n_raw_contacts': 3096,
        'n_filtered_contacts': 1749,
        'hit_height_failures': 0,
    },
    'sset_15': {
        'covered_fraction': 0.8076923076923077,
        'covered': 84,
        'n_gt_rallies': 104,
        'split': 4,
        'missed': 16,
        'merged_spans': 0,
        'spurious_spans': 54,
        'start_offset_mean': -50.61904761904762,
        'start_offset_median': -32.0,
        'start_offset_n': 84,
        'end_offset_mean': 91.4047619047619,
        'end_offset_median': 62.0,
        'end_offset_n': 84,
        'ball_round_absdiff_mean': 2.3095238095238093,
        'ball_round_absdiff_median': 2.0,
        'ball_round_absdiff_n': 84,
        'timing_abserr_mean': 1.425894378194208,
        'timing_abserr_median': 1.0,
        'timing_abserr_n': 587,
        'ball_round_primary': 0.09615384615384616,
        'ball_round_primary_correct': 10,
        'ball_round_primary_total': 104,
        'ball_round_covered': 0.11904761904761904,
        'ball_round_covered_correct': 10,
        'ball_round_covered_total': 84,
        'timing_primary_recall': 0.712378640776699,
        'timing_primary_matched': 587,
        'timing_primary_total': 824,
        'timing_covered_recall': 0.7496807151979565,
        'timing_covered_matched': 587,
        'timing_covered_total': 783,
        'player_primary': 0.375,
        'player_primary_correct': 39,
        'player_primary_total': 104,
        'player_covered': 0.4642857142857143,
        'player_covered_correct': 39,
        'player_covered_total': 84,
        'server_primary': 0.41346153846153844,
        'server_primary_correct': 43,
        'server_primary_total': 104,
        'server_covered': 0.5119047619047619,
        'server_covered_correct': 43,
        'server_covered_total': 84,
        'hit_height_primary': 0.33495145631067963,
        'hit_height_primary_correct': 276,
        'hit_height_primary_total': 824,
        'hit_height_covered': 0.3524904214559387,
        'hit_height_covered_correct': 276,
        'hit_height_covered_total': 783,
        'landing_primary': 0.25,
        'landing_primary_correct': 25,
        'landing_primary_total': 100,
        'landing_covered': 0.30120481927710846,
        'landing_covered_correct': 25,
        'landing_covered_total': 83,
        'getpoint_primary': 0.49411764705882355,
        'getpoint_primary_correct': 42,
        'getpoint_primary_total': 85,
        'getpoint_covered': 0.525,
        'getpoint_covered_correct': 42,
        'getpoint_covered_total': 80,
        'contact_f1': 0.5707340787554691,
        'contact_precision': 0.47607461476074614,
        'contact_recall': 0.712378640776699,
        'contact_matches': 587,
        'contact_filtered_total': 1233,
        'contact_gt_total': 824,
        'n_raw_contacts': 2210,
        'n_filtered_contacts': 1233,
        'hit_height_failures': 0,
    },
    'sset_21': {
        'covered_fraction': 0.72,
        'covered': 54,
        'n_gt_rallies': 75,
        'split': 20,
        'missed': 1,
        'merged_spans': 2,
        'spurious_spans': 26,
        'start_offset_mean': -148.0,
        'start_offset_median': -97.5,
        'start_offset_n': 54,
        'end_offset_mean': 127.64814814814815,
        'end_offset_median': 90.5,
        'end_offset_n': 54,
        'ball_round_absdiff_mean': 5.351851851851852,
        'ball_round_absdiff_median': 3.5,
        'ball_round_absdiff_n': 54,
        'timing_abserr_mean': 2.1050119331742243,
        'timing_abserr_median': 2.0,
        'timing_abserr_n': 419,
        'ball_round_primary': 0.04,
        'ball_round_primary_correct': 3,
        'ball_round_primary_total': 75,
        'ball_round_covered': 0.05555555555555555,
        'ball_round_covered_correct': 3,
        'ball_round_covered_total': 54,
        'timing_primary_recall': 0.6319758672699849,
        'timing_primary_matched': 419,
        'timing_primary_total': 663,
        'timing_covered_recall': 0.8073217726396917,
        'timing_covered_matched': 419,
        'timing_covered_total': 519,
        'player_primary': 0.37333333333333335,
        'player_primary_correct': 28,
        'player_primary_total': 75,
        'player_covered': 0.5185185185185185,
        'player_covered_correct': 28,
        'player_covered_total': 54,
        'server_primary': 0.4,
        'server_primary_correct': 30,
        'server_primary_total': 75,
        'server_covered': 0.5555555555555556,
        'server_covered_correct': 30,
        'server_covered_total': 54,
        'hit_height_primary': 0.2854984894259819,
        'hit_height_primary_correct': 189,
        'hit_height_primary_total': 662,
        'hit_height_covered': 0.36486486486486486,
        'hit_height_covered_correct': 189,
        'hit_height_covered_total': 518,
        'landing_primary': 0.24324324324324326,
        'landing_primary_correct': 18,
        'landing_primary_total': 74,
        'landing_covered': 0.33962264150943394,
        'landing_covered_correct': 18,
        'landing_covered_total': 53,
        'getpoint_primary': 0.33783783783783783,
        'getpoint_primary_correct': 25,
        'getpoint_primary_total': 74,
        'getpoint_covered': 0.4716981132075472,
        'getpoint_covered_correct': 25,
        'getpoint_covered_total': 53,
        'contact_f1': 0.4780376497432972,
        'contact_precision': 0.38440366972477064,
        'contact_recall': 0.6319758672699849,
        'contact_matches': 419,
        'contact_filtered_total': 1090,
        'contact_gt_total': 663,
        'n_raw_contacts': 2246,
        'n_filtered_contacts': 1090,
        'hit_height_failures': 0,
    },
}

DISPLAY_METRICS = (
    "covered_fraction", "covered", "n_gt_rallies", "split", "missed", "merged_spans",
    "spurious_spans", "start_offset_mean", "start_offset_median", "start_offset_n",
    "end_offset_mean", "end_offset_median", "end_offset_n", "ball_round_absdiff_mean",
    "ball_round_absdiff_median", "ball_round_absdiff_n", "timing_abserr_mean",
    "timing_abserr_median", "timing_abserr_n", "ball_round_primary", "ball_round_primary_correct",
    "ball_round_primary_total", "ball_round_covered", "ball_round_covered_correct",
    "ball_round_covered_total", "timing_primary_recall", "timing_primary_matched",
    "timing_primary_total", "timing_covered_recall", "timing_covered_matched",
    "timing_covered_total", "player_primary", "player_primary_correct", "player_primary_total",
    "player_covered", "player_covered_correct", "player_covered_total", "server_primary",
    "server_primary_correct", "server_primary_total", "server_covered", "server_covered_correct",
    "server_covered_total", "hit_height_primary", "hit_height_primary_correct",
    "hit_height_primary_total", "hit_height_covered", "hit_height_covered_correct",
    "hit_height_covered_total", "landing_primary", "landing_primary_correct",
    "landing_primary_total", "landing_covered", "landing_covered_correct", "landing_covered_total",
    "getpoint_primary", "getpoint_primary_correct", "getpoint_primary_total", "getpoint_covered",
    "getpoint_covered_correct", "getpoint_covered_total", "contact_f1", "contact_precision",
    "contact_recall", "contact_matches", "contact_filtered_total", "contact_gt_total",
    "n_raw_contacts", "n_filtered_contacts", "hit_height_failures",
)


class GtWinner(NamedTuple):
    getpoint_ab: str | None
    winner_half: str | None
    win_reason: str | None
    lose_reason: str | None
    landing_px: tuple[float, float] | None
    ab_to_half: dict[str, str]
    ab_half_flipped: bool


class Reconciliation(NamedTuple):
    winners: list[GtWinner]
    n_exact: int
    n_dedup: int
    n_mismatch: int
    n_ab_flip_rallies: int
    n_ab_flip_boundaries: int
    offset: int


class RallyRow(NamedTuple):
    """One ground-truth rally's boundary mapping and column-level scores.

    Identity and boundary fields run from ``gt_index`` through ``mapped_span``.
    Ball-round and timing fields compare stroke counts and matched frame error.
    Player, server, hit-height, getpoint, and landing fields then hold each
    column's eligible ground truth, prediction, and correctness values.
    """

    gt_index: int
    set_id: str
    rally: int
    n_gt_strokes: int
    classification: str
    mapped_span: int | None
    ball_round_gt: int
    ball_round_pred: int | None
    ball_round_correct: bool
    timing_matched_n: int
    timing_mean_abs_err: float | None
    player_gt: str
    player_pred: str | None
    player_correct: bool
    server_gt: str
    server_pred: str | None
    server_correct: bool
    hit_height_eligible_n: int
    hit_height_correct_n: int
    getpoint_eligible: bool
    getpoint_gt: str | None
    getpoint_pred: str | None
    getpoint_correct: bool | None
    landing_eligible: bool
    landing_gt: str | None
    landing_pred: str | None
    landing_correct: bool | None


class ColumnAgg(NamedTuple):
    """Correct and eligible counts for primary and covered-only views.

    Primary totals include every eligible ground-truth item. Secondary totals
    include only items whose rally has a covered boundary mapping.
    """

    primary_correct: int
    primary_total: int
    secondary_correct: int
    secondary_total: int


class VideoScoring(NamedTuple):
    """All per-video calibration rows, aggregates, counts, and diagnostics.

    ``rows`` and ``boundary_metrics`` retain rally-level and boundary results.
    ``ColumnAgg`` fields and timing pairs hold primary and covered-only counts.
    Contact totals support precision and recall, while the remaining lists and
    mappings retain hit-height failures, absolute errors, and verdict details.
    """

    name: str
    rows: list[RallyRow]
    boundary_metrics: dict
    ball_round: ColumnAgg
    timing_primary: tuple[int, int]
    timing_covered: tuple[int, int]
    player: ColumnAgg
    server: ColumnAgg
    hit_height: ColumnAgg
    landing: ColumnAgg
    getpoint: ColumnAgg
    contact_matches: int
    contact_filtered_total: int
    contact_gt_total: int
    n_raw_contacts: int
    n_filtered_contacts: int
    hit_height_failures: list[tuple[int, int, int, str]]
    ball_round_diffs: list[int]
    timing_errs: list[int]
    geometric_verdict_rows: dict[int, object]


class RunVideoInputs(NamedTuple):
    """Fixture-invariant arguments for one ``run_video`` invocation.

    Calibration callers overlay only the candidate configuration, serve request,
    or an explicitly supplied replacement dead mask.
    """

    positional: tuple[object, object, object, object, object]
    keyword: dict[str, object]
    master: pd.DataFrame
    courts: dict


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def canonical_tolerance(fps: float) -> int:
    return int(ScalingKind.FRAME_COUNT.scale(CANONICAL_CONTACT_TOLERANCE_BASE30, fps))


def _norm_half(side: str) -> str:
    return "Top" if side == "Top" else "Bot"


def load_fixture_arrays(fixture: Fixture) -> tuple[np.ndarray, ...]:
    verify_run_video_fixture(fixture)
    root = fixtures_root()
    return (
        np.load(root / fixture.track_path), np.load(root / fixture.pose_path("bboxes")),
        np.load(root / fixture.pose_path("scores")), np.load(root / fixture.pose_path("kps")),
        np.load(root / fixture.pose_path("ndet")), np.load(root / fixture.mask_path),
    )


def load_gt_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    for pin in SHARED_FILES:
        verify_file(pin)
    master = pd.read_csv(REPO_ROOT / "training/data/shuttleset/annotations/shots_master.csv")
    homography = REPO_ROOT / "training/data/shuttleset/annotations/set/homography.csv"
    homo_df = pd.read_csv(homography).set_index("id")
    court_info = load_all_court_info(homography)
    resolution = pd.read_csv(REPO_ROOT / "training/data/shuttleset/annotations/my_raw_video_resolution.csv").set_index("id")
    return master, homo_df, court_info, resolution


def build_run_video_inputs(fixture: Fixture) -> RunVideoInputs:
    """Load one fixture and assemble its invariant ``run_video`` inputs."""
    arrays = load_fixture_arrays(fixture)
    master, homo, courts, resolution = load_gt_tables()
    gate_courts = {str(video_id): info for video_id, info in courts.items()}
    gate_resolution = resolution.copy()
    gate_resolution.index = gate_resolution.index.astype(str)
    track, bboxes, scores, kps, ndet, committed_mask = arrays
    inpaint_codes, guard_info = grade_track(track)
    counts = code_counts(inpaint_codes)
    log.info(
        "inpaint guard video=%s length=%d threshold=%s margin=%s presence=%s counts=%s",
        fixture.video_id,
        len(track),
        guard_info["threshold"],
        guard_info["margin"],
        guard_info["presence_validation"],
        counts,
    )
    root = fixtures_root()
    court_present = np.load(root / fixture.court_present_path)
    if court_present.shape != (len(track),):
        raise ValueError(
            f"{fixture.name}: court_present shape {court_present.shape} != ({len(track)},)"
        )
    if court_present.dtype != np.bool_:
        raise ValueError(f"{fixture.name}: court_present dtype {court_present.dtype} is not bool")
    homography_rows = _read_homography_rows(
        root / fixture.scene_rows_path, str(fixture.video_id)
    )
    if not homography_rows:
        raise ValueError(f"{fixture.name}: homography_rows are missing or empty")
    keyword: dict[str, object] = {
        "fps": fixture.fps,
        "landing_options": SHIPPED_LANDING_FILTER_OPTIONS,
        "net_band": fixture.net_band,
        "resolution": fixture.resolution,
        "video_id": fixture.video_id,
        "court_info": courts[fixture.video_id],
        "homo_df": homo,
        "gate_court_info": gate_courts,
        "gate_resolution_table": gate_resolution,
        "raw_exclusion_mask": committed_mask,
        "court_present": court_present,
        "homography_rows": homography_rows,
    }
    keyword["inpaint_codes"] = inpaint_codes
    return RunVideoInputs((track, bboxes, scores, kps, ndet), keyword, master, courts)


def load_set_tables(fixture: Fixture, gt_rallies: list) -> dict[str, pd.DataFrame]:
    directory = REPO_ROOT / fixture.gt_set_dir
    tables = {path.stem: pd.read_csv(path) for path in sorted(directory.glob("*.csv"))}
    expected = {rally.set_id for rally in gt_rallies}
    if set(tables) != expected:
        raise ValueError(f"{fixture.name}: set CSVs {sorted(tables)} != GT set ids {sorted(expected)}")
    return tables


def reconcile_sets(fixture: Fixture, master: pd.DataFrame, gt_rallies: list,
                   set_dfs: dict[str, pd.DataFrame]) -> Reconciliation:
    per_vid = master[master["vid"] == fixture.video_id]
    sm_side = {int(frame): _norm_half(str(side)) for frame, side in zip(per_vid["frame_num"], per_vid["player_side"])}
    players = {
        set_id: {int(frame): list(group["player"].astype(str))
                 for frame, group in sdf.groupby("frame_num")}
        for set_id, sdf in set_dfs.items()
    }
    winners: list[GtWinner] = []
    exact = dedup = mismatch = flipped_n = 0
    offsets: set[int] = set()
    for rally in gt_rallies:
        group = set_dfs[rally.set_id]
        group = group[group["rally"] == rally.rally].sort_values("frame_num")
        if group.empty:
            raise ValueError(f"{fixture.name} {rally.set_id}/r{rally.rally}: no CSV rally")
        frames = [int(frame) for frame in group["frame_num"]]
        unique = sorted(set(frames))
        gt_frames = list(rally.stroke_frames)
        if frames == gt_frames:
            exact += 1
        elif unique == gt_frames:
            dedup += 1
        else:
            mismatch += 1
        offsets.add(unique[-1] - gt_frames[-1])
        last = group.iloc[-1]
        getpoint = str(last["getpoint_player"]) if pd.notna(last["getpoint_player"]) else None
        landing = (float(last["landing_x"]), float(last["landing_y"])) if pd.notna(last["landing_x"]) and pd.notna(last["landing_y"]) else None
        ab_to_half: dict[str, str] = {}
        flipped = False
        for frame in gt_frames:
            at_frame = players[rally.set_id].get(frame, [])
            if len(at_frame) != 1 or frame not in sm_side:
                continue
            ab, half = at_frame[0], sm_side[frame]
            if ab in ab_to_half and ab_to_half[ab] != half:
                flipped = True
            else:
                ab_to_half[ab] = half
        flipped_n += int(flipped)
        winners.append(GtWinner(getpoint, ab_to_half.get(getpoint),
            str(last["win_reason"]) if pd.notna(last["win_reason"]) else None,
            str(last["lose_reason"]) if pd.notna(last["lose_reason"]) else None, landing, ab_to_half, flipped))
    if len(offsets) != 1:
        raise ValueError(f"{fixture.name}: last-stroke offsets are {sorted(offsets)}")
    prior = None
    boundary_flips = 0
    for winner in winners:
        current = winner.ab_to_half.get("A")
        if prior is not None and current is not None and current != prior:
            boundary_flips += 1
        if current is not None:
            prior = current
    return Reconciliation(winners, exact, dedup, mismatch, flipped_n, boundary_flips, offsets.pop())


def _hit_height_gt_map(set_dfs: dict[str, pd.DataFrame]) -> dict[tuple[str, int], list[float]]:
    result = {}
    for set_id, sdf in set_dfs.items():
        for rally, group in sdf.groupby("rally"):
            ordered = group.sort_values("frame_num").drop_duplicates("frame_num", keep="first")
            result[(set_id, int(rally))] = [float(value) for value in ordered["hit_height"]]
    return result


def score_video(fixture: Fixture, result: AnnotatorResult, master: pd.DataFrame, court_info: dict,
                tolerance: int) -> VideoScoring:
    gt = load_gt_rallies(master, fixture.video_id)
    sets = load_set_tables(fixture, gt)
    recon = reconcile_sets(fixture, master, gt, sets)
    classifications = classify_all(result.spans, gt)
    boundaries = score_boundaries(result.spans, gt)
    contact_gate = score_contacts(result.spans, result.filtered_contacts, gt, tolerances=(tolerance,))["count_gate"]["covered"]
    per_vid = master[master["vid"] == fixture.video_id]
    frame_side = {int(frame): str(side) for frame, side in zip(per_vid["frame_num"], per_vid["player_side"])}
    heights = _hit_height_gt_map(sets)
    rows: list[RallyRow] = []
    br = [0, 0, 0, 0]
    timing = [0, 0, 0, 0]
    player = [0, 0, 0, 0]
    server = [0, 0, 0, 0]
    height = [0, 0, 0, 0]
    landing = [0, 0, 0, 0]
    winner = [0, 0, 0, 0]
    ball_round_diffs: list[int] = []
    timing_errs: list[int] = []
    contact_matches = 0
    for index, (rally, (category, span)) in enumerate(zip(gt, classifications)):
        covered = category is RallyBoundary.COVERED
        candidates = result.filtered_by_rally.get(span, []) if covered else []
        pred_count = len(candidates) if covered else None
        br_ok = covered and pred_count == rally.n_strokes
        br[0] += int(br_ok)
        br[1] += 1
        if covered:
            br[2] += int(br_ok)
            br[3] += 1
            ball_round_diffs.append(abs(pred_count - rally.n_strokes))
        matches = greedy_match(rally.stroke_frames, candidates, tolerance) if covered else []
        contact_matches += len(matches)
        timing_errs.extend(abs(rally.stroke_frames[g] - candidates[c]) for g, c in matches)
        timing[0] += len(matches)
        timing[1] += rally.n_strokes
        if covered:
            timing[2] += len(matches)
            timing[3] += rally.n_strokes
        gt_heights = heights.get((rally.set_id, rally.rally), [])
        hh_eligible = hh_correct = 0
        for stroke_index in range(rally.n_strokes):
            gt_height = gt_heights[stroke_index] if stroke_index < len(gt_heights) else math.nan
            if not math.isnan(gt_height):
                height[1] += 1
                hh_eligible += 1
                if covered:
                    height[3] += 1
        for gt_index, candidate_index in matches:
            gt_height = gt_heights[gt_index] if gt_index < len(gt_heights) else math.nan
            detected = result.hit_height_by_frame.get(candidates[candidate_index])
            if not math.isnan(gt_height) and detected is not None and int(gt_height) == detected:
                height[0] += 1
                height[2] += 1
                hh_correct += 1
        gt_player = Half.TOP if frame_side[rally.stroke_frames[-1]] == "Top" else Half.BOT
        pred_player = result.striker_halves[span] if covered else None
        p_ok = covered and pred_player == gt_player
        for values, ok in ((player, p_ok),):
            values[0] += int(ok)
            values[1] += 1
            if covered:
                values[2] += int(ok)
                values[3] += 1
        gt_server = Half.TOP if frame_side[rally.stroke_frames[0]] == "Top" else Half.BOT
        pred_server = result.fitted_first_all[span] if covered else None
        s_ok = covered and pred_server == gt_server
        server[0] += int(s_ok)
        server[1] += 1
        if covered:
            server[2] += int(s_ok)
            server[3] += 1
        gt_winner = recon.winners[index].winner_half
        pred_winner = None
        if covered and span in result.verdict_rows:
            verdict = result.verdict_rows[span]
            if verdict.verdict == Verdict.WON:
                pred_winner = verdict.striker_half
            elif verdict.verdict == Verdict.LOST:
                pred_winner = OTHER_HALF[verdict.striker_half]
        gp_ok = None
        if gt_winner is not None:
            gp_ok = covered and pred_winner is not None and pred_winner == gt_winner
            winner[0] += int(gp_ok)
            winner[1] += 1
            if covered:
                winner[2] += int(gp_ok)
                winner[3] += 1
        gt_landing = recon.winners[index].landing_px
        landing_half = pred_landing = None
        ld_ok = None
        if gt_landing is not None:
            projected = point_winner.project_pixels_to_court(np.array([[gt_landing[0]], [gt_landing[1]]]), point_winner.HOMOGRAPHY_RESOLUTION, court_info[fixture.video_id])
            landing_half = Half.TOP if float(projected[1, 0]) < point_winner.NET_COURT_Y else Half.BOT
            detected_landing = result.landings.get(span) if covered else None
            pred_landing = detected_landing.half if detected_landing is not None else None
            ld_ok = covered and pred_landing == landing_half
            landing[0] += int(ld_ok)
            landing[1] += 1
            if covered:
                landing[2] += int(ld_ok)
                landing[3] += 1
        mean_err = float(np.mean([abs(rally.stroke_frames[g] - candidates[c]) for g, c in matches])) if matches else None
        rows.append(RallyRow(index, rally.set_id, rally.rally, rally.n_strokes, category.value, span,
            rally.n_strokes, pred_count, br_ok, len(matches), mean_err, gt_player.value,
            pred_player.value if pred_player else None, p_ok, gt_server.value, pred_server.value if pred_server else None,
            s_ok, hh_eligible, hh_correct, gt_winner is not None, gt_winner,
            pred_winner.value if pred_winner else None, gp_ok, gt_landing is not None,
            landing_half.value if landing_half else None, pred_landing.value if pred_landing else None, ld_ok))
    if (contact_gate["pass"], contact_gate["total"]) != (br[2], br[3]):
        raise ValueError(f"{fixture.name}: ball_round count-gate mismatch vs score_contacts")
    return VideoScoring(fixture.name, rows, boundaries, ColumnAgg(*br),
        (timing[0], timing[1]), (timing[2], timing[3]),
        ColumnAgg(*player), ColumnAgg(*server), ColumnAgg(*height), ColumnAgg(*landing), ColumnAgg(*winner),
        contact_matches, len(result.filtered_contacts), sum(r.n_strokes for r in gt), len(result.contacts),
        len(result.filtered_contacts), result.hit_height_failures, ball_round_diffs, timing_errs,
        result.geometric_verdict_rows)


def flatten_metrics(scoring: VideoScoring) -> dict[str, int | float | None]:
    bm = scoring.boundary_metrics
    values: dict[str, int | float | None] = {key: bm[key] for key in ("covered_fraction", "covered", "n_gt_rallies", "split", "missed", "merged_spans", "spurious_spans")}
    start_alignment = bm.get("start_alignment") or {}
    end_alignment = bm.get("end_alignment") or {}
    values.update(
        start_offset_mean=start_alignment.get("mean"), start_offset_median=start_alignment.get("median"),
        start_offset_n=start_alignment.get("n"), end_offset_mean=end_alignment.get("mean"),
        end_offset_median=end_alignment.get("median"), end_offset_n=end_alignment.get("n"),
        ball_round_absdiff_mean=float(np.mean(scoring.ball_round_diffs)) if scoring.ball_round_diffs else None,
        ball_round_absdiff_median=float(np.median(scoring.ball_round_diffs)) if scoring.ball_round_diffs else None,
        ball_round_absdiff_n=len(scoring.ball_round_diffs) if scoring.ball_round_diffs else None,
        timing_abserr_mean=float(np.mean(scoring.timing_errs)) if scoring.timing_errs else None,
        timing_abserr_median=float(np.median(scoring.timing_errs)) if scoring.timing_errs else None,
        timing_abserr_n=len(scoring.timing_errs) if scoring.timing_errs else None,
    )
    for name in ("ball_round", "player", "server", "hit_height", "landing", "getpoint"):
        agg = getattr(scoring, name)
        values.update({f"{name}_primary": _ratio(agg.primary_correct, agg.primary_total), f"{name}_primary_correct": agg.primary_correct, f"{name}_primary_total": agg.primary_total, f"{name}_covered": _ratio(agg.secondary_correct, agg.secondary_total), f"{name}_covered_correct": agg.secondary_correct, f"{name}_covered_total": agg.secondary_total})
    for name, pair in (("timing_primary", scoring.timing_primary), ("timing_covered", scoring.timing_covered)):
        values.update({f"{name}_recall": _ratio(*pair), f"{name}_matched": pair[0], f"{name}_total": pair[1]})
    precision = _ratio(scoring.contact_matches, scoring.contact_filtered_total)
    recall = _ratio(scoring.contact_matches, scoring.contact_gt_total)
    f1 = None if precision is None or recall is None else safe_f1(precision, recall)
    values.update(contact_f1=f1, contact_precision=precision, contact_recall=recall, contact_matches=scoring.contact_matches, contact_filtered_total=scoring.contact_filtered_total, contact_gt_total=scoring.contact_gt_total, n_raw_contacts=scoring.n_raw_contacts, n_filtered_contacts=scoring.n_filtered_contacts, hit_height_failures=len(scoring.hit_height_failures))
    return values


def render_table(scores: dict[str, dict[str, int | float | None]], references=REFERENCE_SCORES) -> str:
    lines = ["fixture metric reference current"]
    for fixture, current in scores.items():
        reference = references.get(fixture, {}) if references else {}
        for metric in DISPLAY_METRICS:
            lines.append(f"{fixture} {metric} {reference.get(metric)!r} {current[metric]!r}")
    return "\n".join(lines)


def run_fixture(
    fixture: Fixture, diagnostics_dir: Path | None = None, *, no_replay_mask: bool = False,
) -> VideoScoring:
    log.info('config: no_replay_mask=%s', no_replay_mask)
    inputs = build_run_video_inputs(fixture)
    rejection_rows: list[dict[str, object]] = []
    keyword = dict(inputs.keyword)
    if no_replay_mask:
        track = inputs.positional[0]
        if not isinstance(track, np.ndarray):
            raise TypeError('fixture track must be a numpy array')
        keyword['raw_exclusion_mask'] = np.zeros(len(track), dtype=bool)
    keyword["rejection_diagnostics"] = rejection_rows
    result = run_video(*inputs.positional, **keyword)
    if diagnostics_dir is not None:
        write_guard_diagnostics(fixture, inputs, rejection_rows, diagnostics_dir)
    return score_video(fixture, result, inputs.master, inputs.courts, canonical_tolerance(fixture.fps))


def write_rallies_csv(rows: Iterable[RallyRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RallyRow._fields)
        writer.writerows(rows)


def write_geometric_verdicts_csv(rows: Iterable[object], path: Path) -> None:
    """Write the additive per-rally geometric winner diagnostics."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "rally_id", "geometric_verdict", "geometric_winner", "agreement",
            "window_closed_by_mask",
        ))
        for row in rows:
            writer.writerow((
                row.rally_id,
                row.geometric_verdict.value if row.geometric_verdict is not None else "",
                row.geometric_winner.value if row.geometric_winner is not None else "",
                row.agreement if row.agreement is not None else "",
                row.window_closed_by_mask,
            ))


def write_guard_diagnostics(
    fixture: Fixture, inputs: RunVideoInputs, rows: Iterable[dict[str, object]], output_dir: Path,
) -> None:
    """Write unpinned inpaint grades and event-rejection rows for one video."""
    video_dir = output_dir / fixture.name
    video_dir.mkdir(parents=True, exist_ok=True)
    codes = inputs.keyword["inpaint_codes"]
    if not isinstance(codes, np.ndarray):
        raise ValueError("inpaint_codes are missing from fixture inputs")
    np.save(video_dir / "codes.npy", codes, allow_pickle=False)
    with (video_dir / "rejections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("rule", "rally_id", "start_frame", "end_frame", "trigger_frame", "trigger_code"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _literal(scores: dict[str, dict[str, int | float | None]]) -> str:
    for fixture, values in scores.items():
        for metric in ("covered_fraction", "contact_f1"):
            value = values[metric]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{fixture} {metric} is not captureable: {value!r}")
    lines = ["REFERENCE_SCORES = {"]
    for fixture, values in scores.items():
        lines.append(f"    {fixture!r}: {{")
        lines.extend(f"        {metric!r}: {values[metric]!r}," for metric in DISPLAY_METRICS)
        lines.append("    },")
    return "\n".join([*lines, "}"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-replay-mask", action="store_true")
    args = parser.parse_args()
    if not args.capture:
        parser.error("--capture is required")
    out = args.out.resolve() if args.out else None
    if out is not None and (out == REPO_ROOT or REPO_ROOT in out.parents):
        parser.error("--out must be outside the repo")
    scorings = {
        fixture.name: run_fixture(fixture, out, no_replay_mask=args.no_replay_mask)
        for fixture in FIXTURES
    }
    scores = {name: flatten_metrics(scoring) for name, scoring in scorings.items()}
    print(render_table(scores, None))
    print(_literal(scores))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        for fixture in FIXTURES:
            write_rallies_csv(scorings[fixture.name].rows, out / f"{fixture.name}.csv")
            write_geometric_verdicts_csv(
                scorings[fixture.name].geometric_verdict_rows.values(),
                out / f"{fixture.name}_geometric_verdicts.csv",
            )


if __name__ == "__main__":
    main()
