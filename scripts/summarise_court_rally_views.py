"""Summarise courts from the camera group occupying the most time in each rally.

Consumes the existing fast-robust evaluation tables. Camera groups come from the
detector, so homography agreement remains a proxy for main-view accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import evaluate_courts_fast_robust as evaluator


def rally_views(scenes: pd.DataFrame, rallies: pd.DataFrame) -> pd.DataFrame:
    """Choose groups by duration, including missing courts in the group selection."""
    rows = []
    for video_id, video_rallies in rallies.groupby("video_id", sort=False):
        video_scenes = scenes[scenes["video_id"] == video_id].reset_index(drop=True)
        starts = video_scenes["start_frame"].to_numpy()
        ends = video_scenes["end_frame"].to_numpy()
        for rally in video_rallies.to_dict("records"):
            overlap = np.maximum(0, np.minimum(ends, rally["end_frame"]) - np.maximum(starts, rally["start_frame"]))
            members = video_scenes.loc[overlap > 0].copy()
            members["rally_frames"] = overlap[overlap > 0]
            group_frames = members.groupby("view_group", sort=False)["rally_frames"].sum()
            # sort=False keeps the earliest scene's group when durations tie.
            largest_group = group_frames.idxmax()
            group = members[members["view_group"] == largest_group].copy().reset_index(drop=True)
            group["frame_index_in_rally"] = group["frame_index"].between(
                rally["start_frame"], rally["end_frame"] - 1
            )
            corners = group[evaluator.CORNER_COLUMNS].to_numpy().reshape(-1, 4, 2)
            representative = evaluator.choose_representative(group, corners)
            row = {key: rally[key] for key in ["video_id", "dataset", "set", "rally", "start_frame", "end_frame"]}
            row.update(
                rally_frames=rally["end_frame"] - rally["start_frame"],
                largest_group=int(largest_group),
                largest_group_frames=int(group["rally_frames"].sum()),
                largest_group_detected_frames=int(group.loc[group["court_detected"], "rally_frames"].sum()),
                group_count=len(group_frames),
                largest_group_is_video_main=bool(group["is_dominant_group"].any()),
                court_detected=representative is not None,
                longest_scene_index=rally["scene_index"],
                longest_scene_error_mean_px=rally["error_mean_px"],
            )
            for column in ["scene_index", "view_id", "frame_index", "error_mean_px", "error_max_px", "iou"]:
                row[column] = group.loc[representative, column] if representative is not None else None
            rows.append(row)
    result = pd.DataFrame(rows)
    result["largest_group_share"] = result["largest_group_frames"] / result["rally_frames"]
    return result


def scene_populations(scenes: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    """Measure prediction presence inside and outside the video's main view group.

    A scene can contribute time to both a rally and the surrounding broadcast.
    Alternate-view and non-rally rows deliberately have no accuracy fields.
    """
    rows = []
    for video in videos.to_dict("records"):
        group = scenes[scenes["video_id"] == video["video_id"]]
        rally_frames = group["rally_frames"].to_numpy()
        main = group["is_dominant_group"].to_numpy()
        populations = {
            "main_group_rally": np.where(main, rally_frames, 0),
            "other_group_rally": np.where(~main, rally_frames, 0),
            "outside_labelled_rallies": (group["end_frame"] - group["start_frame"]).to_numpy() - rally_frames,
        }
        for population, frames in populations.items():
            detected = group["court_detected"].to_numpy()
            total = int(frames.sum())
            predicted = int(frames[detected].sum())
            rows.append({
                "video_id": video["video_id"], "dataset": video["dataset"], "population": population,
                "scenes_with_time": int((frames > 0).sum()),
                "frames": total, "detected_frames": predicted,
                "seconds": total / video["fps"], "detected_seconds": predicted / video["fps"],
                "detection_coverage": predicted / total if total else np.nan,
            })
    return pd.DataFrame(rows)


def summarise(rallies: pd.DataFrame, videos: pd.DataFrame, populations: pd.DataFrame) -> dict:
    """Keep video-level resampling separate from pooled descriptive rally counts."""
    result = {}
    rng = np.random.default_rng(evaluator.BOOTSTRAP_SEED)
    subsets = {"combined": videos, **dict(tuple(videos.groupby("dataset")))}
    for name, subset in subsets.items():
        selected = rallies[rallies["video_id"].isin(subset["video_id"])]
        per_video = selected.groupby("video_id").agg(
            detected=("court_detected", "mean"),
            within10=("error_mean_px", lambda values: (values <= 10).mean()),
        )
        draws = rng.integers(0, len(subset), size=(evaluator.BOOTSTRAP_REPLICATES, len(subset)))
        counts = {
            "videos": len(subset), "rallies": len(selected),
            "detected_rallies": int(selected["court_detected"].sum()),
            "largest_group_is_video_main": int(selected["largest_group_is_video_main"].sum()),
            "largest_group_over_half": int((selected["largest_group_share"] > 0.5).sum()),
            "multiple_groups": int((selected["group_count"] > 1).sum()),
            "median_largest_group_share": float(selected["largest_group_share"].median()),
            "selected_scene_differs_from_longest_scene": int(
                selected["scene_index"].notna().mul(selected["scene_index"] != selected["longest_scene_index"]).sum()
            ),
            "within": evaluator.within_tolerances(selected["error_mean_px"]),
            "per_video_agreement_within10": evaluator.bootstrap_mean(per_video["within10"], draws),
            "per_video_detection": evaluator.bootstrap_mean(per_video["detected"], draws),
        }
        representative_good = (subset["representative_error_mean_px"] <= 10).astype(float)
        counts["representative_within10"] = evaluator.bootstrap_mean(representative_good, draws)
        counts["populations"] = {}
        for population, rows in populations[populations["video_id"].isin(subset["video_id"])].groupby("population"):
            counts["populations"][population] = {
                "frames": int(rows["frames"].sum()), "detected_frames": int(rows["detected_frames"].sum()),
                "hours": float(rows["seconds"].sum() / 3600),
                "detected_hours": float(rows["detected_seconds"].sum() / 3600),
                "pooled_time_coverage": float(rows["detected_seconds"].sum() / rows["seconds"].sum()),
                "mean_per_video_coverage": float(rows["detection_coverage"].mean()),
            }
        result[name] = counts
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    videos = pd.read_csv(args.input / "per_video.csv.gz")
    scenes = pd.read_csv(args.input / "per_scene.csv.gz")
    rallies = pd.read_csv(args.input / "per_rally.csv.gz")
    sources = set(zip(videos["dataset"], videos["source_id"].astype(str)))
    if sources != evaluator.released_sources() or len(videos) != len(sources):
        raise ValueError("input videos differ from the eligible release corpus")
    per_rally = rally_views(scenes, rallies)
    populations = scene_populations(scenes, videos)
    args.output.mkdir(parents=True, exist_ok=True)
    per_rally.to_csv(args.output / "per_rally_view.csv.gz", index=False)
    populations.to_csv(args.output / "view_populations.csv.gz", index=False)
    evaluator.write_json_gz(args.output / "summary.json.gz", summarise(per_rally, videos, populations))
    print(f"Summarised {len(videos)} videos and {len(per_rally)} rallies")


if __name__ == "__main__":
    main()
