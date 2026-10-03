"""Run court_detector.run_video with a paired trial of a score-first court choice.

Each scene is searched and scored once, exactly as in production. Two arms then
choose from that one scoring record:

- baseline: the checkout's own choice and final refit, unchanged;
- trial: the highest combined score among camera-plausible candidates wins.
  Player support orders candidates only when their combined scores are exactly
  equal. The final refit keeps its validity and camera checks and drops the
  player check.

Each arm's courts pool across scenes in its own VideoPool, so both arms use the
same sharing code. Candidate generation is untouched: a scene that stops at the
person count, or is too short for the feet window, has no court in either arm.

Outputs, under --output-dir:
  videos/<id>.json.gz         baseline arm, the production result
  trial_videos/<id>.json.gz   trial arm, same schema
  choices/<id>.jsonl.gz       one line per scene that reached the choice: both
                              picks and every camera-plausible candidate

The trial lives in this file alone. It replaces two names in the imported
detector modules at run time, so the checkout stays at its commit and the
baseline arm runs production code.

Run from the checkout with PYTHONPATH=.:src and run_video's --manifest options.
Use the recorded trial commit when reproducing the historical methods.
"""

from __future__ import annotations

# run_video sets the numerical-library thread limits before NumPy loads, so it comes first.
from court_detector import run_video  # isort: skip

import gzip
import json
import logging
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO

import numpy as np

from court_detector import detect, net_choice, stripe_refit
from court_detector.detect import (
    NET_OVERRUN_WORKING_PX,
    NET_WEIGHT,
    CourtResult,
    Laps,
    LiveModules,
    SceneCourts,
    Switches,
)
from court_detector.view_pool import CourtMode, VideoPool

logger = logging.getLogger("score_first_trial")

PRODUCTION_CHOOSE_COURT = detect.choose_court
PRODUCTION_DETECT_VIDEO = run_video.detect_video

TRIAL_VIDEOS_DIR = "trial_videos"
CHOICES_DIR = "choices"
# Share of feet samples with a player on the court that earns the middle support tier.
MIN_ONE_PLAYER_FRACTION = 0.5
# What identifies a scene and its timings; the trial row takes these from the baseline row.
SHARED_ROW_KEYS = ("view_id", "start_frame", "end_frame", "frame_index", "stage_seconds", "seconds")


@dataclass
class TrialArm:
    """The trial's courts for the video being detected."""

    pool: VideoPool
    choices: TextIO  # the open choices/<id>.jsonl.gz
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)  # by view_id; only scenes that reached the choice
    changed_picks: int = 0
    failures: int = 0


def support_tier(candidate: dict[str, Any]) -> int:
    """A candidate's measured player support; lower is stronger.

    0 passes the production player rule: a player on the court in every feet sample
    and one in each half in at least half of them. 1 has a player on the court in at
    least MIN_ONE_PLAYER_FRACTION of the samples. 2 is every other candidate.
    The fractions are the ones scoring already measured (court_checks.gate_evidence).
    """
    if candidate["historical"]["historical_fullcourt"]:
        return 0
    one_player, _both_halves = candidate["gates"]["player_fractions"]
    return 1 if one_player >= MIN_ONE_PLAYER_FRACTION else 2


def break_exact_ties(top_key: str | None, scored: list[dict[str, Any]], tiers: dict[str, int]) -> tuple[str | None, int]:
    """Among the candidates whose combined score exactly equals the top one, keep the strongest support.

    :param top_key: net_choice.choose's pick with people optional, or None without candidates.
    :param scored: That call's rows, in the C ranking's order.
    :param tiers: support_tier of each row's candidate, by origin_key.
    :return: The chosen origin_key and how many rows share the top score.
    """
    if top_key is None:
        return None, 0
    top_score = next(row["combined_score"] for row in scored if row["origin_key"] == top_key)
    tied = [row for row in scored if row["combined_score"] == top_score]
    # min keeps the first of equal tiers, so the C ranking's order settles the rest, as in production.
    chosen = min(tied, key=lambda row: tiers[row["origin_key"]])
    return chosen["origin_key"], len(tied)


def candidate_table(rows: list[dict[str, Any]], scored: list[dict[str, Any]],
                    candidates: dict[str, dict[str, Any]]) -> dict[str, list]:
    """Every camera-plausible candidate in the C ranking's order, enough to replay a choice rule offline.

    :return: Parallel lists, one entry per candidate. Corners are native px, to 0.01 px.
    """
    table: dict[str, list] = {name: [] for name in (
        "origin_key", "source", "paint_score", "geometry_score", "net_reward", "combined_score",
        "one_player", "both_halves", "corners_native_px")}
    for row, score in zip(rows, scored, strict=True):
        candidate = candidates[row["origin_key"]]
        one_player, both_halves = candidate["gates"]["player_fractions"]
        table["origin_key"].append(row["origin_key"])
        table["source"].append(row["source"])
        table["paint_score"].append(row["paint_score"])
        table["geometry_score"].append(row["geometry_score"])
        table["net_reward"].append(score["net_reward"])
        table["combined_score"].append(score["combined_score"])
        table["one_player"].append(one_player)
        table["both_halves"].append(both_halves)
        table["corners_native_px"].append(np.round(candidate["corners_px"], 2).tolist())
    return table


def refit_player_fractions(refit: dict[str, Any] | None) -> list[float] | None:
    """The refitted court's (one player, both halves) fractions, when the refit could be measured."""
    if refit is None or refit["corrected"]["measurement"] is None:
        return None
    return refit["corrected"]["measurement"]["gates"]["player_fractions"]


def trial_choice(view_id: str, record: dict, context: Any, native_frame: np.ndarray, line_maps: np.ndarray,
                 live: LiveModules, switches: Switches, baseline_key: str | None,
                 baseline_refit: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The trial's pick from the scoring record, and its stripe refit.

    :param baseline_key: The production pick, whose refit is reused when the trial picks the same court.
    :return: The trial's scene row, and its part of the choice record with the candidate table.
    """
    candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
    rows = net_choice.net_rows(record, context, require_people=False)
    top_key, scored = net_choice.choose(rows, NET_WEIGHT, NET_OVERRUN_WORKING_PX, switches.geometry_weight,
                                        require_people=False)
    tiers = {row["origin_key"]: support_tier(candidates[row["origin_key"]]) for row in rows}
    chosen, top_score_ties = break_exact_ties(top_key, scored, tiers)

    row: dict[str, Any] = {"view_id": view_id, "status": "no_court", "corners_native_px": None, "chosen_key": chosen,
                           "no_court_reason": None, "reused_from": None, "composition": None}
    paint_score, refit = None, None
    if chosen is None:
        row["no_court_reason"] = "no_gated_court"
    else:
        if chosen == baseline_key:
            refit = baseline_refit
        else:
            refit = stripe_refit.refit_chosen(record, chosen, context, native_frame, live.verifier, live.runtime,
                                              line_maps, replay_check=switches.self_checks)
        corrected = refit["corrected"]
        if not corrected["valid"]:
            row["no_court_reason"] = corrected["validity_reason"]
        elif not corrected["measurement"]["historical"]["historical_camera"]:
            row["no_court_reason"] = "refit_camera_implausible"
        else:
            # Production also requires historical_fullcourt here; the trial leaves that check out.
            row.update(status="court", corners_native_px=corrected["corners_native_px"])
            paint_score = corrected["measurement"]["paint_score"]
    choice = {"chosen_key": chosen, "no_court_reason": row["no_court_reason"],
              "corners_native_px": row["corners_native_px"], "paint_score": paint_score,
              "refit_player_fractions": refit_player_fractions(refit), "top_score_ties": top_score_ties,
              "tie_break_changed_pick": chosen != top_key,
              "candidates": candidate_table(rows, scored, candidates)}
    return row, choice


def add_trial_scene(arm: TrialArm, baseline: CourtResult, record: dict, context: Any, native_frame: np.ndarray,
                    line_maps: np.ndarray, live: LiveModules, switches: Switches, artefacts: dict[str, Any]) -> None:
    """Choose the trial's court for one scene, save both picks, and place the court in the trial's pool.

    :param baseline: The production result for this scene.
    :param artefacts: The production choice's artefacts, for its refit.
    """
    started = perf_counter()
    baseline_refit = artefacts.get("stripe_refit")
    row, choice = trial_choice(baseline.view_id, record, context, native_frame, line_maps, live, switches,
                               baseline.chosen_key, baseline_refit)
    baseline_corners = None if baseline.corners_native_px is None else baseline.corners_native_px.tolist()
    line = {"view_id": baseline.view_id,
            "baseline": {"chosen_key": baseline.chosen_key, "no_court_reason": baseline.no_court_reason,
                         "corners_native_px": baseline_corners, "paint_score": baseline.paint_score,
                         "refit_player_fractions": refit_player_fractions(baseline_refit)},
            "trial": choice, "trial_seconds": perf_counter() - started}
    arm.choices.write(json.dumps(line, allow_nan=False, separators=(",", ":")) + "\n")
    arm.changed_picks += row["chosen_key"] != baseline.chosen_key
    arm.rows[baseline.view_id] = row
    if row["status"] == "court":
        corners = np.asarray(row["corners_native_px"])
        # In video order, as production's court joins its own pool after detect() returns.
        arm.pool.add(row, SceneCourts(context, native_frame, corners, corners))


def paired_choose_court(arm: TrialArm, view_id: str, record: dict, context: Any, native_frame: np.ndarray,
                        line_maps: np.ndarray, live: LiveModules, switches: Switches, laps: Laps,
                        artefacts: dict[str, Any]) -> CourtResult:
    """The production choice, returned unchanged, after the trial has chosen from the same scoring record."""
    baseline = PRODUCTION_CHOOSE_COURT(view_id, record, context, native_frame, line_maps, live, switches, laps,
                                       artefacts)
    try:
        add_trial_scene(arm, baseline, record, context, native_frame, line_maps, live, switches, artefacts)
    except Exception as error:
        # A paired run of several hours keeps its baseline arm when the trial's side fails on
        # one scene. The trial's side runs in this process and leaves the shared workers alone.
        logger.exception("%s: trial arm failed for this scene; the baseline court is unaffected", view_id)
        arm.failures += 1
        arm.rows[view_id] = {"view_id": view_id, "status": "detection_failed", "corners_native_px": None,
                             "chosen_key": None, "no_court_reason": None, "reused_from": None, "composition": None,
                             "error": repr(error)}
    return baseline


def save_trial_arm(arm: TrialArm, baseline: dict[str, Any], output_dir: Path) -> None:
    """Pool the trial's courts and write its result in the baseline result's schema."""
    if arm.failures:
        raise RuntimeError(f"{arm.failures} trial scenes failed; this video cannot be compared")
    video_id = baseline["video_id"]
    started = perf_counter()
    logger.info("%s: pooling the trial arm's courts", video_id)
    trial_groups = arm.pool.apply()
    failed_groups = [group for group in trial_groups if group.get("error")]
    if failed_groups:
        raise RuntimeError(f"{video_id}: {len(failed_groups)} trial groups failed")
    trial_scenes = []
    for baseline_row in baseline["scenes"]:
        trial_row = arm.rows.get(baseline_row["view_id"])
        if trial_row is None:
            # The scene stopped before the choice, so it has the same outcome in both arms.
            trial_scenes.append(baseline_row)
            continue
        trial_scenes.append({**{key: baseline_row[key] for key in SHARED_ROW_KEYS}, **trial_row})
    run_video.write_json(output_dir / TRIAL_VIDEOS_DIR / f"{video_id}.json.gz",
                         {**baseline, "scenes": trial_scenes, "view_groups": trial_groups,
                          "selection_rule": "score_then_player_tier_on_exact_ties",
                          "total_seconds": baseline["total_seconds"] + perf_counter() - started})
    logger.info("%s: trial arm chose a different court in %d of %d scenes that reached the choice; "
                "%d trial failures", video_id, arm.changed_picks, len(arm.rows), arm.failures)


def paired_detect_video(video: Path, tools: run_video.CourtTools, *, output_dir: Path,
                        **options: Any) -> dict[str, Any]:
    """Production detect_video for the baseline arm; the trial arm pools separately and is saved beside it.

    :param output_dir: run_video's --output-dir.
    :param options: detect_video's keyword arguments.
    :return: The baseline arm's result, as production returns it.
    """
    switches = tools.detector.switches
    court_mode, video_id = options["court_mode"], options["video_id"]
    # artefacts_dir would search scenes past the person count, which changes candidate generation.
    if court_mode != CourtMode.FAST_ROBUST or not switches.require_people or switches.artefacts_dir is not None:
        raise ValueError("the trial pairs fast-robust choices with people required and no artefacts directory")
    for directory in (TRIAL_VIDEOS_DIR, CHOICES_DIR):
        (output_dir / directory).mkdir(exist_ok=True)
    with gzip.open(output_dir / CHOICES_DIR / f"{video_id}.jsonl.gz", "wt") as choices:
        arm = TrialArm(VideoPool(tools.detector.live, switches, court_mode), choices)
        # score_and_choose looks choose_court up in detect's namespace at each call.
        detect.choose_court = partial(paired_choose_court, arm)
        try:
            baseline = PRODUCTION_DETECT_VIDEO(video, tools, **options)
        finally:
            detect.choose_court = PRODUCTION_CHOOSE_COURT
    save_trial_arm(arm, baseline, output_dir)
    return baseline


def main() -> int:
    output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    # run_batch looks detect_video up in run_video's namespace at each call.
    run_video.detect_video = partial(paired_detect_video, output_dir=output_dir)
    return run_video.main()


if __name__ == "__main__":
    raise SystemExit(main())
