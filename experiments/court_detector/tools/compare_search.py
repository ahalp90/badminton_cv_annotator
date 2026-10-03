"""Compare the search without player rejection with the court-sharing patch and score-first selection.

Reads the paired trial outputs retrieved from both hosts, checks that every run
finished and that the arms describe the same scenes, and writes compressed tables
to evidence/search/search_results/ by default.

Errors use the supplied default-camera court annotation at 1280x720. They are
counted only on the fixed main-view population: the scenes in the court-sharing-patched
run's view group with the most labelled rally frames. Other views have no court
annotation, so those scenes are described by detections and by how far a court's
corners move between arms. A corner move is a change, not an error.

The statistical comparison treats the video as the unit. Each of the three methods
gets one value per video, every video counts equally, and the paired differences
are taken within each video. One matrix of whole-video resamples serves every
method, contrast and job-time ratio. Its intervals describe reweighting these eight
purpose-selected videos, not performance on other videos or venues.

Run from the repo root:

    PYTHONPATH=.:src ~/.venvs/badminton-cicd/bin/python \
      experiments/court_detector/tools/compare_search.py
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from court_detector.view_pool import GROUP_SCENE_KEY, POOLED_KEY
from scripts import evaluate_courts_fast_robust as evaluator
from scripts import summarise_court_rally_views as rally_view_summary
from shared.court import HOMOGRAPHY_RESOLUTION

EVIDENCE_ROOT = Path(__file__).resolve().parents[1] / "evidence"
SEARCH_RESULTS = EVIDENCE_ROOT / "search" / "search_results"
EVALUATION_ROOT = EVIDENCE_ROOT / "inputs"
REVIEW_SAMPLES = EVIDENCE_ROOT / "search" / "selection_results" / "review_samples.csv.gz"

HOST_VIDEOS = {
    "carmack": ("sset_11", "sset_21", "sset_30", "sset_36"),
    "bourbaki": ("ss22_27", "ss22_43", "ss22_44", "ss22_51"),
}
STAGE_COMMITS = {
    "stage_a": "928ed398d04ff3f7819321891dc612846f3e3a08",
    "stage_b": "cd776fb24f24b770d577918c51afc6788f09e8d0",
}
ORIGINAL = "original"
COURT_SHARING_PATCHED = "court_sharing_patched"
SCORE_FIRST = "score_first"
SEARCH_SCORE_FIRST = "search_score_first"
# Changed search with the original final player veto. Auxiliary: not a control for anything.
SEARCH_PLAYER_VETO = "search_player_veto"
# arm -> (stage, output directory, that arm's key in the stage's choice records)
TRIAL_ARMS = {
    COURT_SHARING_PATCHED: ("stage_a", "videos", "baseline"),
    SCORE_FIRST: ("stage_a", "trial_videos", "trial"),
    SEARCH_SCORE_FIRST: ("stage_b", "trial_videos", "trial"),
    SEARCH_PLAYER_VETO: ("stage_b", "videos", "baseline"),
}
ARMS = (ORIGINAL, *TRIAL_ARMS)
# The three methods compared statistically, and their paired contrasts as (before, after).
METHODS = (COURT_SHARING_PATCHED, SCORE_FIRST, SEARCH_SCORE_FIRST)
METHOD_PAIRS = (
    (COURT_SHARING_PATCHED, SCORE_FIRST),
    (COURT_SHARING_PATCHED, SEARCH_SCORE_FIRST),
    (SCORE_FIRST, SEARCH_SCORE_FIRST),
)
# (before, after) arms whose scenes are compared one to one.
ARM_PAIRS = (*METHOD_PAIRS, (COURT_SHARING_PATCHED, SEARCH_PLAYER_VETO))
# The selection comparison's population, kept so the two comparisons describe the same scenes.
FIXED_MAIN_SCENES = 726
FIXED_MAIN_RALLY_SCENES = 610
# Worst-corner tolerances at 720p. The same two bin how far a court moves between arms.
TOLERANCES_PX = (10, 20)
SMALL_PX, LARGE_PX = TOLERANCES_PX
# The evaluator's replicate count with a new seed. Every method, contrast and job-time
# ratio indexes the same draws.
BOOTSTRAP_SEED = 20261003


class MetricSpec(NamedTuple):
    unit: str
    count: str | None  # the pooled numerator column; None where a pooled total means nothing
    denominator: str | None
    interval: bool  # False keeps a metric descriptive: no bootstrap interval or paired contrast


# Per-video metric -> how it is summarised. Percentages run 0-100, and a missing
# court counts against them.
METRICS = {
    "representative_error_mean_px": MetricSpec("px", None, None, True),
    "representative_iou": MetricSpec("IoU", None, None, False),
    f"fixed_main_final_within_{SMALL_PX}px_pct": MetricSpec(
        "%", f"fixed_main_final_within_{SMALL_PX}px", "fixed_main_scenes", True),
    f"fixed_main_before_sharing_within_{LARGE_PX}px_pct": MetricSpec(
        "%", f"fixed_main_before_sharing_within_{LARGE_PX}px", "fixed_main_scenes", True),
    "rally_time_coverage_pct": MetricSpec("%", "detected_rally_frames", "rally_frames", True),
    "rally_view_court_pct": MetricSpec("%", "rally_view_courts", "rallies", True),
    f"rally_view_within_{SMALL_PX}px_pct": MetricSpec("%", f"rally_view_within_{SMALL_PX}px", "rallies", True),
}
DIFFERENCE_UNITS = {"px": "px", "%": "percentage points"}
STAGE_JOBS = ("stage_a_jobs", "stage_b_jobs")
LABEL_ROOTS = {"ShuttleSet": Path("data/shuttleset/set"), "ShuttleSet22": Path("data/shuttleset22/set")}
SCENE_KEY = ["start_frame", "end_frame", "frame_index"]
COURT_SOURCES = {POOLED_KEY: "pooled", GROUP_SCENE_KEY: "group_scene"}


@dataclass
class VideoComparison:
    paired: pd.DataFrame  # one row per scene, every arm side by side
    video_rows: list[dict[str, object]]  # one per arm
    group_rows: list[dict[str, object]]  # one per trial arm and view group
    audit: dict[str, object]
    outputs: dict[str, dict]  # the raw run result of each arm
    rally_views: pd.DataFrame  # one row per method and labelled rally


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_stage(stage_dir: Path, stage: str, video_ids: tuple[str, ...]) -> tuple[list[dict], dict[str, float]]:
    """Check one host's stage finished at the expected commit with every video complete.

    :return: The stage's cohort entries, and each video's paired job seconds.
    """
    exit_path = stage_dir / "stage.exit"
    require(exit_path.exists(), f"{exit_path} is missing: the stage is unfinished or the transfer is partial")
    require(exit_path.read_text().strip() == "0", f"{stage_dir}: stage exit is not 0")
    commit = evaluator.read_json_gz(stage_dir / "run_config.json.gz")["commit"]
    require(commit == STAGE_COMMITS[stage], f"{stage_dir}: ran commit {commit}")
    cohort = evaluator.read_json_gz(stage_dir / "cohort.json.gz")
    require(sorted(entry["id"] for entry in cohort) == sorted(video_ids), f"{stage_dir}: unexpected cohort")
    job_seconds = {}
    for video_id in video_ids:
        attempts = sorted((stage_dir / "attempts" / video_id).iterdir())
        require(len(attempts) == 1, f"{stage_dir}: {video_id} has {len(attempts)} attempts; timings assume one")
        require((attempts[0] / "exit_code").read_text().strip() == "0", f"{attempts[0]}: exit code is not 0")
        summary = evaluator.read_json_gz(attempts[0] / "results" / "summary.json.gz")
        require(summary["finished"] and summary["videos"][0]["status"] == "complete", f"{attempts[0]}: incomplete")
        job_seconds[video_id] = summary["videos"][0]["seconds"]
    return cohort, job_seconds


def read_choices(path: Path) -> dict[str, dict]:
    """:return: One choice record per scene that reached the choice, by view id."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {record["view_id"]: record for record in map(json.loads, handle)}


def check_own_fits(output: dict, choices: dict[str, dict], choice_arm: str, label: str) -> None:
    """Check each scene's saved own court against the choice record that produced it."""
    view_ids = {scene["view_id"] for scene in output["scenes"]}
    require(set(choices) <= view_ids, f"{label}: choice records for unknown scenes")
    for scene in output["scenes"]:
        require(not scene.get("error") and scene["status"] != "detection_failed", f"{label}: {scene['view_id']} failed")
        require("error" not in (scene.get("view_pool") or {}), f"{label}: {scene['view_id']} failed to join a group")
        # Sharing moves a scene's own court to scene_corners_native_px when it replaces it.
        saved_own = scene.get("scene_corners_native_px", scene["corners_native_px"])
        choice = choices.get(scene["view_id"])
        chosen_own = None if choice is None else choice[choice_arm]["corners_native_px"]
        require(saved_own == chosen_own, f"{label}: {scene['view_id']} own court differs from its choice record")
    for group in output["view_groups"]:
        require(not group.get("error"), f"{label}: group {group['reference_view_id']} failed")


def corner_shift_px(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """How far a scene's court moves between two arms: the largest corner distance at 720p.

    Corner order may start at a different corner in each arm, so the closest of the
    four cyclic rotations is kept, as evaluator.score_courts does against the annotation.

    :param first: (scene, corner, xy) at 720p; NaN rows where a scene has no court.
    :param second: Same shape.
    :return: One per scene; NaN unless both arms have a court.
    """
    rotated = [np.roll(second, -rotation, axis=1) for rotation in range(4)]
    distances = np.stack([np.linalg.norm(first - courts, axis=2) for courts in rotated])  # (rotation, scene, corner)
    closest_rotation = distances.mean(axis=2).argmin(axis=0)  # one per scene
    return distances[closest_rotation, np.arange(len(first))].max(axis=1)


def arm_tables(entry: dict, output: dict, official: np.ndarray,
               rallies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One arm's scored scenes and rallies from the evaluator, with each scene's place in the arm's view groups."""
    scenes, scored_rallies = evaluator.analyse_video(entry, output, official, rallies)
    scene_index_of = dict(zip(scenes["view_id"], scenes["scene_index"]))
    scenes["group_reference_scene"] = -1
    scenes["group_size"] = 0
    scenes["group_chosen_scene"] = -1
    for group in output["view_groups"]:
        members = [scene_index_of[view_id] for view_id in group["member_view_ids"]]
        scenes.loc[members, "group_reference_scene"] = scene_index_of[group["reference_view_id"]]
        scenes.loc[members, "group_size"] = len(members)
        if group["chosen_view_id"] is not None:
            scenes.loc[members, "group_chosen_scene"] = scene_index_of[group["chosen_view_id"]]
    chosen_keys = pd.Series([scene.get("chosen_key") for scene in output["scenes"]])
    scenes["court_source"] = chosen_keys.map(COURT_SOURCES).fillna("own").where(scenes["court_detected"])
    return scenes, scored_rallies


def group_rows(video_id: str, arm: str, output: dict, scenes: pd.DataFrame, fixed_main: np.ndarray,
               court_sharing_patched_detected: np.ndarray) -> list[dict[str, object]]:
    """One row per view group, counting what a scene outside rallies or a newly admitted court supplies.

    A group's reference is its first member; every later member is aligned to it. Marking
    donors supply the samples for the pooled court. The chosen scene's court replaces the
    court of each member that accepts it.

    :param scenes: The arm's scene table from arm_tables.
    :param fixed_main: One per scene; True inside the fixed main-view population.
    :param court_sharing_patched_detected: One per scene; True where the court-sharing-patched run has a court.
    """
    scene_index_of = dict(zip(scenes["view_id"], scenes["scene_index"]))
    rally_frames = scenes["rally_frames"].to_numpy()
    replaced = scenes["court_source"].isin(COURT_SOURCES.values()).to_numpy()
    rows = []
    for group in output["view_groups"]:
        members = np.array([scene_index_of[view_id] for view_id in group["member_view_ids"]])
        reference = scene_index_of[group["reference_view_id"]]
        # A marking no member could measure has no donor.
        marking_donors = {scene_index_of[marking["donor_view_id"]] for marking in group.get("markings", [])
                          if marking["donor_view_id"] is not None}
        replaced_members = members[replaced[members]]
        row: dict[str, object] = {
            "video_id": video_id, "arm": arm, "reference_scene": reference, "members": len(members),
            "members_in_rallies": int((rally_frames[members] > 0).sum()),
            "member_rally_frames": int(rally_frames[members].sum()),
            "members_in_fixed_main": int(fixed_main[members].sum()),
            "members_new_since_court_sharing_patched": int((~court_sharing_patched_detected[members]).sum()),
            "is_arm_main_group": bool(scenes.loc[reference, "is_dominant_group"]),
            "reason": group["reason"], "chosen_court": group["chosen_court"],
            "reference_rally_frames": int(rally_frames[reference]),
            "reference_new_since_court_sharing_patched": bool(~court_sharing_patched_detected[reference]),
            "marking_donors": len(marking_donors),
            "marking_donors_outside_rallies": sum(rally_frames[donor] == 0 for donor in marking_donors),
            "marking_donors_new_since_court_sharing_patched": sum(
                not court_sharing_patched_detected[donor] for donor in marking_donors),
            "replaced_members": len(replaced_members),
            "replaced_members_in_rallies": int((rally_frames[replaced_members] > 0).sum()),
            "replaced_rally_frames": int(rally_frames[replaced_members].sum()),
        }
        if group["chosen_view_id"] is not None:
            chosen = scene_index_of[group["chosen_view_id"]]
            row |= {"chosen_scene": chosen, "chosen_rally_frames": int(rally_frames[chosen]),
                    "chosen_new_since_court_sharing_patched": bool(~court_sharing_patched_detected[chosen])}
        rows.append(row)
    return rows


def paired_scene_table(tables: dict[str, pd.DataFrame], outputs: dict[str, dict]) -> pd.DataFrame:
    """Join the arms' scenes one to one. Their scene partitions are already checked equal."""
    court_sharing_patched = tables[COURT_SHARING_PATCHED]
    paired = court_sharing_patched[["video_id", "scene_index", "view_id", *SCENE_KEY, "rally_frames",
                          "frame_index_in_rally"]].copy()
    paired["fixed_main"] = court_sharing_patched["is_dominant_group"]
    scale = np.array(HOMOGRAPHY_RESOLUTION) / np.array(outputs[COURT_SHARING_PATCHED]["native_size"])
    final_courts, own_courts = {}, {}
    for arm, scenes in tables.items():
        columns = {
            "court_detected": "court_detected", "no_court_reason": "no_court_reason", "court_source": "court_source",
            "error_mean_px": "error_mean_px", "error_max_px": "error_max_px",
            "before_pool_error_mean_px": "own_error_mean_px", "before_pool_error_max_px": "own_error_max_px",
            "is_dominant_group": "main_group", "group_reference_scene": "group_reference_scene",
            "group_size": "group_size", "group_chosen_scene": "group_chosen_scene",
        }
        for source, name in columns.items():
            paired[f"{arm}_{name}"] = scenes[source]
        final_courts[arm] = scenes[evaluator.CORNER_COLUMNS].to_numpy().reshape(-1, 4, 2) * scale
        own_courts[arm] = final_courts[arm].copy()
        for index, scene in enumerate(outputs[arm]["scenes"]):
            if scene.get("scene_corners_native_px") is not None:
                own_courts[arm][index] = np.array(scene["scene_corners_native_px"]) * scale
    for before, after in ARM_PAIRS:
        paired[f"{after}_shift_from_{before}_px"] = corner_shift_px(final_courts[before], final_courts[after])
        paired[f"{after}_own_shift_from_{before}_px"] = corner_shift_px(own_courts[before], own_courts[after])
    return paired


def population_masks(paired: pd.DataFrame) -> dict[str, pd.Series]:
    """Four populations that partition the scenes: fixed main view or not, rally overlap or not."""
    main, in_rally = paired["fixed_main"], paired["rally_frames"] > 0
    return {"main_rally": main & in_rally, "main_outside_rallies": main & ~in_rally,
            "other_rally": ~main & in_rally, "other_outside_rallies": ~main & ~in_rally}


def main_view_transitions(paired: pd.DataFrame) -> list[dict[str, object]]:
    """Repairs and damage against the annotation, for the fixed main-view scenes only.

    Each scene needs every corner within the tolerance. Own courts are the scene's
    individual fit before sharing; final courts are what the run saved after sharing.
    """
    rows = []
    for population, mask in population_masks(paired).items():
        if not population.startswith("main"):
            continue
        scenes = paired[mask]
        for before, after in ARM_PAIRS:
            for fit, infix in (("own", "own_"), ("final", "")):
                before_error = scenes[f"{before}_{infix}error_max_px"]
                after_error = scenes[f"{after}_{infix}error_max_px"]
                for tolerance in TOLERANCES_PX:
                    before_within, after_within = before_error <= tolerance, after_error <= tolerance
                    rows.append({
                        "video_id": paired["video_id"].iloc[0], "population": population,
                        "before": before, "after": after, "fit": fit, "tolerance_px": tolerance,
                        "scenes": len(scenes), "before_within": int(before_within.sum()),
                        "after_within": int(after_within.sum()),
                        "repaired": int(((before_error > tolerance) & after_within).sum()),
                        "damaged": int((before_within & (after_error > tolerance)).sum()),
                        "lost_within": int((before_within & after_error.isna()).sum()),
                        "new_within": int((before_error.isna() & after_within).sum()),
                        "new_outside": int((before_error.isna() & (after_error > tolerance)).sum()),
                    })
    return rows


def detection_changes(paired: pd.DataFrame) -> list[dict[str, object]]:
    """Courts gained, lost and moved between arms, for every scene. No annotation is used."""
    rows = []
    for population, mask in population_masks(paired).items():
        scenes = paired[mask]
        for before, after in ARM_PAIRS:
            had, has = scenes[f"{before}_court_detected"], scenes[f"{after}_court_detected"]
            shift = scenes[f"{after}_shift_from_{before}_px"]
            own_shift = scenes[f"{after}_own_shift_from_{before}_px"]
            small, large = TOLERANCES_PX
            rows.append({
                "video_id": paired["video_id"].iloc[0], "population": population, "before": before, "after": after,
                "scenes": len(scenes), "rally_frames": int(scenes["rally_frames"].sum()),
                "before_detected": int(had.sum()), "after_detected": int(has.sum()),
                "new_courts": int((~had & has).sum()), "lost_courts": int((had & ~has).sum()),
                "new_court_rally_frames": int(scenes.loc[~had & has, "rally_frames"].sum()),
                "lost_court_rally_frames": int(scenes.loc[had & ~has, "rally_frames"].sum()),
                "kept_courts": int((had & has).sum()),
                "kept_identical": int((shift == 0).sum()),
                f"kept_moved_up_to_{small}px": int(((shift > 0) & (shift <= small)).sum()),
                f"kept_moved_{small}_to_{large}px": int(((shift > small) & (shift <= large)).sum()),
                f"kept_moved_over_{large}px": int((shift > large).sum()),
                f"kept_moved_over_{large}px_rally_frames": int(scenes.loc[shift > large, "rally_frames"].sum()),
                f"own_moved_over_{large}px": int((own_shift > large).sum()),
            })
    return rows


def court_paths(paired: pd.DataFrame) -> pd.DataFrame:
    """Count scenes by which of the three main arms hold a court, within each population.

    It shows what the search arm does with the courts score-first selection added or
    dropped, and how many courts only the search arm has.
    """
    paths = paired[["video_id", "rally_frames"]].copy()
    paths["population"] = ""
    for population, mask in population_masks(paired).items():
        paths.loc[mask, "population"] = population
    for arm in METHODS:
        paths[f"{arm}_court"] = paired[f"{arm}_court_detected"]
    same_px = TOLERANCES_PX[0]
    for before in (COURT_SHARING_PATCHED, SCORE_FIRST):
        shift = paired[f"{SEARCH_SCORE_FIRST}_shift_from_{before}_px"]
        paths[f"search_identical_to_{before}"] = shift == 0
        paths[f"search_within_{same_px}px_of_{before}"] = shift <= same_px
    paths["scenes"] = 1
    keys = ["video_id", "population", *(f"{arm}_court" for arm in METHODS)]
    return paths.groupby(keys, as_index=False).sum()


def reviewed_scene_rows(samples: pd.DataFrame, paired: pd.DataFrame, outputs: dict[str, dict[str, dict]]) -> list[dict]:
    """What each arm holds for the scenes whose images the user reviewed.

    The user judged every outlined court in that sample wrong and every abstention
    reasonable. ``search_outcome`` says only whether the search arm keeps, moves or drops
    the reviewed court. A moved court is unjudged; an unmoved one keeps its judgement.

    :param outputs: outputs[video_id][arm], the raw run results.
    """
    rows = []
    reviewed = samples.merge(paired, left_on=["id", *SCENE_KEY], right_on=["video_id", *SCENE_KEY],
                             how="left", validate="one_to_one", suffixes=("_sample", ""))
    require(bool(reviewed["video_id"].notna().all()), "a reviewed scene has no matching frame interval")
    for _, scene in reviewed.iterrows():
        shown_arm = scene["overlay_from"] if isinstance(scene["overlay_from"], str) else None
        after_detected = bool(scene[f"{SEARCH_SCORE_FIRST}_court_detected"])
        row = {"number": scene["number"], "video_id": scene["video_id"], "scene_index": scene["scene_index"],
               "category": scene["category"], "shown_arm": shown_arm, "frame_index": scene["frame_index"],
               "rally_frames": scene["rally_frames"], "fixed_main": scene["fixed_main"]}
        for arm in TRIAL_ARMS:
            saved = outputs[scene["video_id"]][arm]["scenes"][scene["scene_index"]]
            row |= {f"{arm}_court": scene[f"{arm}_court_detected"],
                    f"{arm}_no_court_reason": saved.get("no_court_reason"),
                    f"{arm}_court_source": scene[f"{arm}_court_source"],
                    f"{arm}_group_size": scene[f"{arm}_group_size"],
                    f"{arm}_corners_native_px": json.dumps(saved["corners_native_px"])}
        shift = np.nan
        if shown_arm is None:
            outcome = "new_court_where_abstention_was_reasonable" if after_detected else "still_no_court"
        elif not after_detected:
            outcome = "reviewed_wrong_court_absent"
        else:
            shift = scene[f"{SEARCH_SCORE_FIRST}_shift_from_{shown_arm}_px"]
            outcome = "same_wrong_court" if shift == 0 else "different_court_unjudged"
        row |= {"search_shift_from_shown_px": shift, "search_outcome": outcome}
        rows.append(row)
    return rows


def compare_video(entry: dict, stage_dirs: dict[str, Path], evaluation_root: Path) -> VideoComparison:
    """Load one video's arms, check them against each other, and build its tables."""
    video_id, dataset = entry["id"], entry["dataset"]
    labels = LABEL_ROOTS[dataset]
    source_id = int(entry["source_id"])
    homography = evaluator.read_label_table(labels, "homography").set_index("id").loc[source_id]
    official, _ = evaluator.read_official_corners(homography, dataset)
    match_dir = labels / evaluator.read_label_table(labels, "match").set_index("id").loc[source_id, "video"]

    outputs = {ORIGINAL: evaluator.read_json_gz(evaluation_root / "videos" / f"{video_id}.json.gz")}
    audit: dict[str, object] = {"video_id": video_id}
    for arm, (stage, directory, choice_arm) in TRIAL_ARMS.items():
        outputs[arm] = evaluator.read_json_gz(stage_dirs[stage] / directory / f"{video_id}.json.gz")
        choices = read_choices(stage_dirs[stage] / "choices" / f"{video_id}.jsonl.gz")
        check_own_fits(outputs[arm], choices, choice_arm, f"{video_id} {arm}")
        audit[f"{stage}_choice_records"] = len(choices)
        # The score-first choice and its refit; the rest of a paired job is shared by both of its arms.
        audit[f"{stage}_score_first_choice_seconds"] = sum(choice["trial_seconds"] for choice in choices.values())
    court_sharing_patched = outputs[COURT_SHARING_PATCHED]
    partition = [[scene[key] for key in ("view_id", *SCENE_KEY)] for scene in court_sharing_patched["scenes"]]
    for arm, output in outputs.items():
        arm_partition = [[scene[key] for key in ("view_id", *SCENE_KEY)] for scene in output["scenes"]]
        require(arm_partition == partition, f"{video_id} {arm}: scene partition differs from the court-sharing patch")
        require(output["native_size"] == court_sharing_patched["native_size"], f"{video_id} {arm}: frame size differs")

    rallies, label_counts = evaluator.read_rallies(match_dir, court_sharing_patched["frame_count"])
    tables, video_rows, rally_views = {}, [], []
    for arm, output in outputs.items():
        tables[arm], scored_rallies = arm_tables(entry, output, official, rallies)
        video_rows.append(evaluator.video_row(entry, output, tables[arm], scored_rallies)
                          | {"arm": arm, "total_seconds": output["total_seconds"], **label_counts})
        if arm in METHODS:
            # Each rally's court from the view group with the most rally time, as the
            # original-corpus rally-view summary chooses it.
            views = rally_view_summary.rally_views(tables[arm], scored_rallies)
            views.insert(0, "method", arm)
            rally_views.append(views)
    paired = paired_scene_table(tables, outputs)
    fixed_main = paired["fixed_main"].to_numpy()
    court_sharing_patched_detected = paired[f"{COURT_SHARING_PATCHED}_court_detected"].to_numpy()
    groups = []
    for arm in TRIAL_ARMS:
        groups += group_rows(video_id, arm, outputs[arm], tables[arm], fixed_main, court_sharing_patched_detected)
        in_arm_main = paired[f"{arm}_main_group"].to_numpy()
        added = ~fixed_main & in_arm_main
        audit |= {f"{arm}_groups": len(outputs[arm]["view_groups"]),
                  f"{arm}_fixed_main_outside_main_group": int((fixed_main & ~in_arm_main).sum()),
                  f"{arm}_main_group_added_scenes": int(added.sum()),
                  f"{arm}_main_group_added_rally_frames": int(paired.loc[added, "rally_frames"].sum())}
    audit |= {"scenes": len(paired), "fixed_main_scenes": int(fixed_main.sum()),
              "fixed_main_rally_scenes": int((fixed_main & (paired["rally_frames"] > 0)).sum())}
    return VideoComparison(paired, video_rows, groups, audit, outputs, pd.concat(rally_views, ignore_index=True))


def video_metrics(videos: pd.DataFrame, paired: pd.DataFrame, rally_views: pd.DataFrame,
                  video_order: list[str]) -> pd.DataFrame:
    """One row per method and video: the values the summary and paired effects average.

    Each method's rows follow video_order, which the bootstrap draws index. Counts keep
    their denominators so pooled totals can sit beside the per-video percentages.
    """
    fixed_main = paired[paired["fixed_main"]]
    rows = []
    for method in METHODS:
        method_videos = videos[videos["arm"] == method].set_index("video_id")
        for video_id in video_order:
            video = method_videos.loc[video_id]
            main_scenes = fixed_main[fixed_main["video_id"] == video_id]
            rallies = rally_views[(rally_views["method"] == method) & (rally_views["video_id"] == video_id)]
            rows.append({
                "method": method, "video_id": video_id, "dataset": video["dataset"],
                "representative_error_mean_px": video["representative_error_mean_px"],
                "representative_iou": video["representative_iou"],
                "fixed_main_scenes": len(main_scenes),
                f"fixed_main_final_within_{SMALL_PX}px": int((main_scenes[f"{method}_error_max_px"] <= SMALL_PX).sum()),
                f"fixed_main_before_sharing_within_{LARGE_PX}px":
                    int((main_scenes[f"{method}_own_error_max_px"] <= LARGE_PX).sum()),
                "rally_frames": video["rally_frames"], "detected_rally_frames": video["detected_rally_frames"],
                "rallies": len(rallies), "rally_view_courts": int(rallies["court_detected"].sum()),
                f"rally_view_within_{SMALL_PX}px": int((rallies["error_mean_px"] <= SMALL_PX).sum()),
            })
    metrics = pd.DataFrame(rows)
    for metric, spec in METRICS.items():
        if spec.count is not None:
            metrics[metric] = 100 * metrics[spec.count] / metrics[spec.denominator]
    # A missing value would make bootstrap_mean silently average fewer videos.
    require(not metrics[list(METRICS)].isna().any(axis=None), "a method lacks a value for some video")
    return metrics


def method_summary(metrics: pd.DataFrame, draws: np.ndarray) -> pd.DataFrame:
    """Each method's per-video metrics, with every video weighted equally.

    Only the all-video rows get bootstrap intervals. Each dataset has four selected
    videos, so its rows stay descriptive. Pooled counts weight videos by their scenes,
    rallies or rally frames instead.

    :param metrics: From video_metrics, so each method's rows follow the order the draws index.
    :param draws: (replicate, video) positions into the fixed video order.
    """
    rows = []
    scopes = {"all": metrics, **dict(tuple(metrics.groupby("dataset")))}
    for scope, scope_metrics in scopes.items():
        for method in METHODS:
            method_metrics = scope_metrics[scope_metrics["method"] == method]
            for metric, spec in METRICS.items():
                values = method_metrics[metric]
                row = {"scope": scope, "method": method, "metric": metric, "unit": spec.unit,
                       "videos": len(values), "mean": values.mean()}
                if scope == "all" and spec.interval:
                    row |= evaluator.bootstrap_mean(values, draws)
                row |= evaluator.tail_spread(values, 10) | evaluator.tail_spread(values, 90)
                if spec.count is not None:
                    count, denominator = method_metrics[spec.count].sum(), method_metrics[spec.denominator].sum()
                    row |= {"pooled_count": int(count), "pooled_denominator": int(denominator),
                            "pooled_of": spec.denominator, "pooled_pct": 100 * count / denominator}
                rows.append(row)
    # Int64 keeps whole counts whole where descriptive-only rows leave them missing.
    return pd.DataFrame(rows).astype({"pooled_count": "Int64", "pooled_denominator": "Int64"})


def contrast_row(before: str, after: str, metric: str, unit: str, before_values: pd.Series, after_values: pd.Series,
                 draws: np.ndarray) -> dict[str, object]:
    """The mean within-video difference, its interval from the shared draws, and its spread over videos.

    Higher, lower and equal compare exact values, so a 0.0001 px change counts as higher.
    """
    difference = after_values - before_values
    interval = evaluator.bootstrap_mean(difference, draws)
    return {"before": before, "after": after, "metric": metric, "statistic": "mean_paired_difference", "unit": unit,
            "videos": len(difference), "estimate": interval["mean"], "ci95_low": interval["ci95_low"],
            "ci95_high": interval["ci95_high"], "video_min": difference.min(), "video_max": difference.max(),
            "videos_higher": int((difference > 0).sum()), "videos_lower": int((difference < 0).sum()),
            "videos_equal": int((difference == 0).sum())}


def paired_effects(metrics: pd.DataFrame, audit: pd.DataFrame, draws: np.ndarray) -> pd.DataFrame:
    """Paired method differences, then stage B job time against stage A on the same videos.

    A higher percentage is not automatically better: abstaining can be the right result.
    Each stage job ran two arms and shared search work between them, so the job times
    compare the two stages, not standalone methods.

    :param metrics: From video_metrics.
    :param audit: One row per video in the fixed video order, with both stages' job seconds.
    :param draws: (replicate, video) positions into the fixed video order.
    """
    rows = []
    for before, after in METHOD_PAIRS:
        for metric, spec in METRICS.items():
            if not spec.interval:
                continue
            by_video = metrics.pivot(index="video_id", columns="method", values=metric).loc[audit.index]
            row = contrast_row(before, after, metric, DIFFERENCE_UNITS[spec.unit], by_video[before], by_video[after],
                               draws)
            if spec.count is not None:
                counts = metrics.groupby("method")[[spec.count, spec.denominator]].sum()
                row |= {"pooled_before": counts.loc[before, spec.count], "pooled_after": counts.loc[after, spec.count],
                        "pooled_denominator": counts.loc[before, spec.denominator], "pooled_of": spec.denominator}
            rows.append(row)

    before_minutes, after_minutes = audit["stage_a_job_seconds"] / 60, audit["stage_b_job_seconds"] / 60
    summed = {"pooled_before": before_minutes.sum(), "pooled_after": after_minutes.sum(),
              "pooled_of": "summed job minutes"}
    rows.append(contrast_row(*STAGE_JOBS, "job_minutes", "minutes", before_minutes, after_minutes, draws) | summed)
    ratio = after_minutes / before_minutes
    # A percentile interval survives exp(), so the log-ratio interval converts directly.
    log_interval = evaluator.bootstrap_mean(np.log(ratio), draws)
    rows.append({"before": STAGE_JOBS[0], "after": STAGE_JOBS[1], "metric": "job_time_ratio",
                 "statistic": "geometric_mean_of_video_ratios", "unit": "ratio", "videos": len(ratio),
                 "estimate": np.exp(log_interval["mean"]), "ci95_low": np.exp(log_interval["ci95_low"]),
                 "ci95_high": np.exp(log_interval["ci95_high"]), "video_min": ratio.min(), "video_max": ratio.max()})
    rows.append({"before": STAGE_JOBS[0], "after": STAGE_JOBS[1], "metric": "job_time_ratio",
                 "statistic": "summed_after_over_summed_before", "unit": "ratio", "videos": len(ratio),
                 "estimate": after_minutes.sum() / before_minutes.sum()} | summed)
    return pd.DataFrame(rows).astype({"videos_higher": "Int64", "videos_lower": "Int64", "videos_equal": "Int64"})


def print_statistics(metrics: pd.DataFrame, summary: pd.DataFrame, effects: pd.DataFrame) -> None:
    print("\nPER-VIDEO METRICS (equal-weight inputs to the summary and paired effects)")
    print(metrics[["method", "video_id", *METRICS]].round(4).to_string(index=False))
    print("\nMETHOD SUMMARY (video is the unit; intervals reweight these eight videos only)")
    print(summary.round(4).to_string(index=False))
    print("\nPAIRED EFFECTS (after minus before within each video; same draws for every row)")
    print(effects.round(4).to_string(index=False))


def print_summary(videos: pd.DataFrame, transitions: pd.DataFrame, changes: pd.DataFrame, paths: pd.DataFrame,
                  groups: pd.DataFrame, audit: pd.DataFrame, reviewed: pd.DataFrame) -> None:
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
    by_arm = videos.pivot(index="video_id", columns="arm")
    arms = list(ARMS)
    print("REPRESENTATIVE COURT, mean corner error px at 720p")
    print(by_arm["representative_error_mean_px"][arms].round(3).to_string())
    print("\nRALLY-TIME COVERAGE, fraction of labelled rally frames in a scene with any court")
    print(by_arm["time_coverage"][arms].round(5).to_string())
    print((by_arm["time_coverage"][arms].mean() * 100).round(3).rename("mean per video, %").to_string())
    print("\nDETECTED SCENES")
    print(by_arm["detected_scenes"][arms].sum().to_string())
    print("\nFIXED MAIN VIEW, worst corner against the annotation (sum over videos)")
    print(transitions.drop(columns="video_id")
          .groupby(["before", "after", "population", "fit", "tolerance_px"], sort=False).sum().to_string())
    print("\nALL SCENES, courts gained, lost and moved (sum over videos; no annotation used)")
    print(changes.drop(columns="video_id").groupby(["before", "after", "population"], sort=False).sum().T.to_string())
    print("\nCOURT PATHS, scenes by which arms hold a court (sum over videos)")
    path_keys = ["population", *(f"{arm}_court" for arm in METHODS)]
    print(paths.drop(columns="video_id").groupby(path_keys).sum().to_string())

    sharing = groups[groups["chosen_court"].notna()]
    outside = sharing["chosen_rally_frames"] == 0
    new = sharing["chosen_new_since_court_sharing_patched"].eq(True)  # missing for a pooled court
    # A new first member becomes the reference that the group's existing courts are aligned to.
    has_existing_court = groups["members"] > groups["members_new_since_court_sharing_patched"]
    new_reference = groups["reference_new_since_court_sharing_patched"] & has_existing_court
    print("\nVIEW GROUPS")
    print(pd.DataFrame({
        "groups": groups.groupby("arm").size(),
        "groups_of_one": groups[groups["members"] == 1].groupby("arm").size(),
        "groups_sharing_a_court": sharing.groupby("arm").size(),
        "pooled_court_chosen": sharing[sharing["chosen_court"] == "pooled"].groupby("arm").size(),
        "chosen_scene_outside_rallies": sharing[outside].groupby("arm").size(),
        "rally_frames_replaced_by_those": sharing[outside].groupby("arm")["replaced_rally_frames"].sum(),
        "chosen_scene_new_since_court_sharing_patched": sharing[new].groupby("arm").size(),
        "rally_frames_replaced_by_new": sharing[new].groupby("arm")["replaced_rally_frames"].sum(),
        "new_reference_for_existing_courts": groups[new_reference].groupby("arm").size(),
    }).reindex(list(TRIAL_ARMS)).fillna(0).astype(int).T.to_string())
    print("\nGROUPS SHARING A COURT")
    print(sharing[["video_id", "arm", "is_arm_main_group", "reference_scene", "members", "members_in_fixed_main",
                   "members_new_since_court_sharing_patched", "member_rally_frames", "chosen_court", "chosen_scene",
                   "chosen_rally_frames", "chosen_new_since_court_sharing_patched", "replaced_members",
                   "replaced_rally_frames"]].sort_values(["video_id", "reference_scene", "arm"]).to_string(index=False))
    print("\nAUDIT AND PAIRED JOB TIME")
    print(audit.T.to_string())
    print("\nREVIEWED SCENES")
    print(reviewed[["number", "video_id", "scene_index", "category", "shown_arm", "rally_frames",
                    f"{COURT_SHARING_PATCHED}_court", f"{SCORE_FIRST}_court", f"{SEARCH_SCORE_FIRST}_court",
                    f"{SEARCH_SCORE_FIRST}_court_source", "search_shift_from_shown_px", "search_outcome"]]
          .to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION_ROOT,
                        help="holds videos/ (original extraction) and player_tiebreak_results/<host>/stage_{a,b}/")
    parser.add_argument("--review-samples", type=Path, default=REVIEW_SAMPLES)
    parser.add_argument("--output", type=Path, default=SEARCH_RESULTS)
    parser.add_argument("--hosts", nargs="+", choices=list(HOST_VIDEOS), default=list(HOST_VIDEOS),
                        help="a subset skips the fixed-population check and is only for a partial dry run")
    args = parser.parse_args()

    results: list[VideoComparison] = []
    for host in args.hosts:
        host_root = args.evaluation_root / "player_tiebreak_results" / host
        stage_dirs = {stage: host_root / stage for stage in STAGE_COMMITS}
        cohorts, job_seconds = {}, {}
        for stage, stage_dir in stage_dirs.items():
            cohorts[stage], job_seconds[stage] = check_stage(stage_dir, stage, HOST_VIDEOS[host])
        require(cohorts["stage_a"] == cohorts["stage_b"], f"{host}: the two stages ran different cohorts")
        for entry in cohorts["stage_a"]:
            result = compare_video(entry, stage_dirs, args.evaluation_root)
            before, after = job_seconds["stage_a"][entry["id"]], job_seconds["stage_b"][entry["id"]]
            result.audit |= {"host": host, "stage_a_job_seconds": before, "stage_b_job_seconds": after,
                             "job_seconds_ratio": after / before}
            results.append(result)

    paired = pd.concat([result.paired for result in results], ignore_index=True)
    audit = pd.DataFrame([result.audit for result in results]).set_index("video_id")
    if set(args.hosts) == set(HOST_VIDEOS):
        population = (int(audit["fixed_main_scenes"].sum()), int(audit["fixed_main_rally_scenes"].sum()))
        require(population == (FIXED_MAIN_SCENES, FIXED_MAIN_RALLY_SCENES), f"fixed main population is {population}")
    videos = pd.DataFrame([row for result in results for row in result.video_rows])
    groups = pd.DataFrame([row for result in results for row in result.group_rows])
    transitions = pd.DataFrame([row for result in results for row in main_view_transitions(result.paired)])
    changes = pd.DataFrame([row for result in results for row in detection_changes(result.paired)])
    paths = court_paths(paired)
    samples = pd.read_csv(args.review_samples)
    samples = samples[samples["id"].isin(paired["video_id"])]
    outputs = {str(result.audit["video_id"]): result.outputs for result in results}
    reviewed = pd.DataFrame(reviewed_scene_rows(samples, paired, outputs))

    # One fixed video order and one draw matrix serve every method and contrast.
    video_order = [video_id for host in args.hosts for video_id in HOST_VIDEOS[host]]
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(video_order), size=(evaluator.BOOTSTRAP_REPLICATES, len(video_order)))
    rally_views = pd.concat([result.rally_views for result in results], ignore_index=True)
    metrics = video_metrics(videos, paired, rally_views, video_order)
    summary = method_summary(metrics, draws)
    effects = paired_effects(metrics, audit.loc[video_order], draws)

    args.output.mkdir(parents=True, exist_ok=True)
    tables = {"paired_scenes": paired, "per_video": videos, "main_view_transitions": transitions,
              "detection_changes": changes, "court_paths": paths, "view_groups": groups,
              "audit": audit.reset_index(), "reviewed_scenes": reviewed, "rally_views": rally_views,
              "video_metrics": metrics, "method_summary": summary, "paired_effects": effects}
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv.gz", index=False)
    print_summary(videos, transitions, changes, paths, groups, audit, reviewed)
    print_statistics(metrics, summary, effects)


if __name__ == "__main__":
    main()
