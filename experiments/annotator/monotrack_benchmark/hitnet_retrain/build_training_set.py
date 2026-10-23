"""Build both HitNet training inputs from saved MonoTrack courts and RTMPose detections."""
from __future__ import annotations

import argparse
import gzip
import json
import tomllib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ("kept", "zeroed")
SIDE_CLASSES = {"Bot": 1, "Top": 2}


def load_videos(spans_path: Path, split_path: Path) -> list[dict[str, Any]]:
    """Check the frozen input spans against the committed development split."""
    with gzip.open(spans_path, 'rt') as stream:
        videos = json.load(stream)["videos"]
    with gzip.open(split_path, 'rt') as stream:
        split = json.load(stream)
    expected = {row["fixture"]: row for row in split["videos"]}
    if len(videos) != len(expected) or {video["item_id"] for video in videos} != set(expected):
        raise ValueError("Span videos differ from the committed development split")
    for video in videos:
        entry = expected[video["item_id"]]
        if video["role"] != entry["role"] or video["fps"] != entry["fps"]:
            raise ValueError(f"{video['item_id']}: split role or frame rate changed")
        if video["resolution"] != [entry["width"], entry["height"]]:
            raise ValueError(f"{video['item_id']}: image resolution changed")
        previous_end = 0
        for span in video["spans"]:
            start, end = span["start_frame"], span["end_frame"]
            frames = span["label_frames"]
            if not previous_end <= start < end <= video["frame_count"] or end - start < 12:
                raise ValueError(f"{video['item_id']}: overlapping or invalid rally bounds")
            if not frames or any(not start <= frame < end for frame in frames):
                raise ValueError(f"{video['item_id']}: contacts outside rally bounds")
            if len(frames) != len(span["label_sides"]) or any(side not in SIDE_CLASSES for side in span["label_sides"]):
                raise ValueError(f"{video['item_id']}: missing or unknown ground-truth side")
            previous_end = end
    return videos


def build_video(task: tuple[dict[str, Any], Path, Path]) -> dict[str, Any]:
    from inputs import (
        Court,
        feature_rows,
        fill_players,
        read_array,
        read_json,
        select_players,
        shuttle_variants,
    )

    video, stages, courts = task
    fixture = video["item_id"]
    pose_dir = stages / "pose" / fixture
    kps = read_array(pose_dir / "pose_kps.npy.xz")
    scores = read_array(pose_dir / "pose_kp_scores.npy.xz")
    ndet = read_array(pose_dir / "pose_ndet.npy.xz")
    frames = video["frame_count"]
    if kps.shape[0] != frames or kps.shape[2:] != (17, 2) or scores.shape != kps.shape[:-1]:
        raise ValueError(f"{fixture}: pose arrays do not match the video timeline")
    if ndet.shape != (frames,) or np.any(ndet < 0) or np.any(ndet > kps.shape[1]):
        raise ValueError(f"{fixture}: invalid saved detection counts")
    shuttle_dir = stages / "shuttle" / fixture
    sidecars = list(shuttle_dir.glob("*_stride8_inpaint_mask.json.gz"))
    if len(sidecars) != 1:
        raise ValueError(f"{fixture}: expected one inpaint sidecar, found {len(sidecars)}")
    track = read_array(shuttle_dir / "shuttle_track.npy.xz")
    if len(track) != frames:
        raise ValueError(f"{fixture}: shuttle timeline differs from pose timeline")
    shuttles = shuttle_variants(track, read_json(sidecars[0]), video["resolution"])
    output: dict[str, Any] = {"rallies": [], "labels": [], "kept": [], "zeroed": []}
    for span in video["spans"]:
        start, end = span["start_frame"], span["end_frame"]
        record = {"fixture": fixture, "role": video["role"], "fps": video["fps"],
                  "span_id": span["span_id"], "rally_id": span["rally_id"],
                  "start_frame": start, "end_frame": end, "contacts": len(span["label_frames"])}
        result = read_json(courts / f"{fixture}_{span['span_id']:04d}.json.gz")
        if result["status"] == "failed":
            output["rallies"].append({**record, "excluded": "no_court"})
            continue
        if result["status"] != "completed":
            raise ValueError(f"{fixture}: court result is unfinished")
        # Stored corners are raw detector [0,3,2,1]. Both selection and notebook
        # training use raw [1,2,0,3]; do not apply inference's extra permutation.
        court = Court(np.asarray(result["corners_native_px"])[[3, 2, 0, 1]])
        players = select_players(kps[start:end], scores[start:end], ndet[start:end], court)
        record["missing_player_frames"] = np.isnan(players).all(axis=(2, 3)).sum(axis=0).tolist()
        singleton_coordinates = (~np.isnan(players)).sum(axis=0) == 1
        if singleton_coordinates.any():
            # Released slinear filling crashes with only one observation.
            # Exclude these training rallies; benchmark inference still runs them.
            output["rallies"].append({**record, "excluded": "pose_interpolation"})
            continue
        players = fill_players(players)
        labels = np.zeros(end - start, dtype=np.int8)
        for frame, side in zip(span["label_frames"], span["label_sides"], strict=True):
            labels[frame - start] = SIDE_CLASSES[side]
        output["labels"].append(labels)
        for arm in ARMS:
            rows = feature_rows(shuttles[arm][start:end], players, court)
            if not np.isfinite(rows).all():
                raise ValueError(f"{fixture}/{span['span_id']}: non-finite scaled features")
            output[arm].append(rows)
        output["rallies"].append({**record, "frames": end - start})
    return output


def build(config: dict[str, Any], output: Path, workers: int) -> dict[str, Any]:
    from inputs import write_array, write_json

    paths = config["inputs"]
    videos = load_videos(Path(paths["spans"]), HERE / "shuttleset_development_split.json.gz")
    output.mkdir(parents=True, exist_ok=False)
    tasks = [(video, Path(paths["stages"]), Path(paths["courts"])) for video in videos]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(build_video, tasks))
    rallies = [rally for result in results for rally in result["rallies"]]
    offset = 0
    counts: Counter[tuple[str, str]] = Counter()
    contacts: Counter[tuple[str, str]] = Counter()
    for rally in rallies:
        status = rally.get("excluded", "usable")
        counts[rally["role"], status] += 1
        contacts[rally["role"], status] += rally["contacts"]
        if status == "usable":
            rally["offset"] = offset
            offset += rally["frames"]
    for arm in ARMS:
        directory = output / arm
        directory.mkdir()
        rows = np.concatenate([rows for result in results for rows in result[arm]])
        write_array(directory / "frames.npy.xz", rows)
    labels = np.concatenate([labels for result in results for labels in result["labels"]])
    write_array(output / "labels.npy.xz", labels)
    write_json(output / "rallies.json.gz", rallies)
    summary: dict[str, Any] = {"videos": len(videos), "rallies": len(rallies), "feature_frames": offset,
                               "player_selection": "monotrack", "shuttle_interpolation": False, "by_split": {}}
    for role in ("train", "validation"):
        summary["by_split"][role] = {
            "usable_rallies": counts[role, "usable"], "no_court_rallies": counts[role, "no_court"],
            "usable_contacts": contacts[role, "usable"], "no_court_contacts": contacts[role, "no_court"],
            "pose_interpolation_rallies": counts[role, "pose_interpolation"],
            "pose_interpolation_contacts": contacts[role, "pose_interpolation"],
        }
    write_json(output / "summary.json.gz", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    build(config, args.output, args.workers)


if __name__ == "__main__":
    main()
