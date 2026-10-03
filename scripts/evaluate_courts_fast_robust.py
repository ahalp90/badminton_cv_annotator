"""Compare fast-robust court detections with the official ShuttleSet court corners.

``analyse`` reads the detector's whole-video outputs and the official ShuttleSet and
ShuttleSet22 labels. It writes per-scene, per-rally and per-video tables, a summary
and a list of frames to draw. ``render`` draws each requested court on its native
frame.

The official corners describe one static playing view per match, at 1280x720. The
labelled contact frames say when play is on screen, so every comparison is weighted
by, or restricted to, rally frames. A rally runs from its first to its last labelled
contact. The labels do not say when the shuttle lands, so no tail is added.

Run from the repo root with ``PYTHONPATH=src``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from court_detector.paint_geometry import CENTRE_SEGMENTS_M, STRIPE_WIDTH_M
from dataset_builder.source_annotations import logical_set_id
from shared.court import HOMOGRAPHY_RESOLUTION
from shared.court_model import CORNER_COURT_M

REPO_ROOT = Path(__file__).resolve().parents[1]
SHUTTLESET_MANIFEST = REPO_ROOT / "configs" / "dataset_builder" / "shuttleset_sources_v1.toml"
SHUTTLESET22_MANIFEST = REPO_ROOT / "configs" / "shuttleset22" / "sources.toml"

# Detector corners and the reordered official corners both run TL, TR, BR, BL.
CORNER_NAMES = ("tl", "tr", "br", "bl")
OFFICIAL_CORNER_COLUMNS = ("upleft", "upright", "downright", "downleft")
CORNER_COLUMNS = [f"{name}_{axis}_native" for name in CORNER_NAMES for axis in "xy"]
SCENE_COLUMNS = ["view_id", "start_frame", "end_frame", "frame_index", "status", "no_court_reason"]
SCORE_COLUMNS = ["error_mean_px", "error_max_px", "iou"]
# Scene fields copied into the per-video row under a ``representative_`` prefix.
REPRESENTATIVE_COLUMNS = ["scene_index", "view_id", "frame_index", "frame_index_in_rally", *SCORE_COLUMNS]

# Declared before looking at results. They describe the error spread; nothing here
# shows that a court within a tolerance is good enough for any later use.
TOLERANCES_PX = (5, 10, 20)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261002
HARD_SAMPLES_PER_DATASET = 4

DASH_PX = 7
GAP_PX = 5
# Vertex pairs of a stripe outline. Both long edges run from the same end of the
# stripe, so their dashes sit opposite each other; the last two are the end caps.
OUTLINE_EDGES = ((0, 1), (3, 2), (0, 3), (1, 2))
RED_BGR = (0, 0, 255)
FRAME_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_json_gz(path: Path) -> object:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_gz(path: Path, payload: object) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, allow_nan=False)


def read_label_table(root: Path, name: str) -> pd.DataFrame:
    """Read one plain or gzip-compressed official label table."""
    paths = sorted(Path(root).glob(f"{name}.csv*"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one {name}.csv or {name}.csv.gz under {root}, found {paths}")
    return pd.read_csv(paths[0])


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def released_sources() -> set[tuple[str, str]]:
    """:return: (dataset, source id) pairs the two release manifests keep in their default corpora."""
    with SHUTTLESET_MANIFEST.open("rb") as handle:
        shuttleset = tomllib.load(handle)["videos"]
    with SHUTTLESET22_MANIFEST.open("rb") as handle:
        shuttleset22 = tomllib.load(handle)["videos"]
    kept = {("ShuttleSet", video["source_id"]) for video in shuttleset if video["eligible"]}
    for video in shuttleset22:
        # Overlaps are scored under their ShuttleSet id; unresolved sources have no video.
        if video["source_kind"] == "download" and "excluded_reason" not in video:
            kept.add(("ShuttleSet22", str(video["id"])))
    return kept


def read_official_corners(row: pd.Series, dataset: str) -> tuple[np.ndarray, dict[str, object]]:
    """Invert the supplied homography at its dataset's doubles-court template.

    The CSV's explicit corners sometimes disagree slightly with its homography.
    Keep that disagreement as a measurement of reference uncertainty.
    """
    corners = np.array([[row[f"{name}_x"], row[f"{name}_y"]] for name in OFFICIAL_CORNER_COLUMNS], dtype=float)
    top_left, top_right, bottom_right, bottom_left = corners
    if not (top_left[0] < top_right[0] and max(top_left[1], top_right[1]) < min(bottom_left[1], bottom_right[1])):
        raise ValueError(f"official corners for match {row.name} are not a top pair above a bottom pair")
    bottom_pair_swapped = bool(bottom_left[0] > bottom_right[0])
    if bottom_pair_swapped:
        corners = corners[[0, 1, 3, 2]]
    left, right = (25.0, 325.0) if dataset == "ShuttleSet" and int(row.name) <= 20 else (27.4, 327.6)
    template = np.array([[left, 150, 1], [right, 150, 1], [right, 810, 1], [left, 810, 1]])
    projected = template @ np.linalg.inv(np.array(json.loads(row["homography_matrix"]))).T
    reference = projected[:, :2] / projected[:, 2, None]
    gap = np.linalg.norm(reference - corners, axis=1)
    return reference, {
        "official_bottom_pair_swapped": bottom_pair_swapped,
        "homography_corner_gap_mean_px": float(gap.mean()),
        "homography_corner_gap_max_px": float(gap.max()),
    }


def read_rallies(set_dir: Path, frame_count: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Turn a match's contact rows into one frame interval per (set, rally).

    :return: Rallies with ``start_frame`` and exclusive ``end_frame``, and row counts.
    """
    paths = sorted(set_dir.glob("set*.csv*"))
    if not paths:
        raise FileNotFoundError(f"no set*.csv or set*.csv.gz under {set_dir}")
    tables = [
        pd.read_csv(path, usecols=["rally", "ball_round", "frame_num"]).assign(
            set=logical_set_id(path)
        )
        for path in paths
    ]
    contacts = pd.concat(tables, ignore_index=True)
    frame = pd.to_numeric(contacts["frame_num"], errors="coerce")
    finite = np.isfinite(frame)
    outside_video = finite & ((frame < 0) | (frame >= frame_count))
    if outside_video.any() or contacts["rally"].isna().any():
        raise ValueError(f"{set_dir}: contact frames outside [0, {frame_count}) or rows without a rally number")

    counts = {
        "label_rows": len(contacts),
        "nonfinite_frame_rows": int((~finite).sum()),
        "excluded_incomplete_rallies": 0,
        "excluded_non_monotonic_rallies": 0,
    }
    rows = []
    for (set_id, rally_id), group in contacts.assign(frame=frame).groupby(["set", "rally"]):
        if not np.isfinite(group["frame"]).all():
            counts["excluded_incomplete_rallies"] += 1
            continue
        ordered = group.sort_values(["ball_round", "frame"], kind="stable")["frame"].to_numpy()
        # Match source_annotations._usable_rallies; a flaw flag alone does not
        # invalidate a court comparison when its contact timing is usable.
        if (np.diff(ordered) <= 0).any():
            counts["excluded_non_monotonic_rallies"] += 1
            continue
        rows.append({"set": set_id, "rally": rally_id, "start_frame": int(ordered[0]),
                     "end_frame": int(ordered[-1]) + 1, "contacts": len(ordered)})
    return pd.DataFrame(rows), counts


def score_courts(predicted: np.ndarray, official: np.ndarray) -> pd.DataFrame:
    """Corner distances and polygon overlap for each predicted court, at 1280x720.

    The detector's corner order may start at a different corner from the official
    one, so all four cyclic rotations are tried and the closest kept. Mirrored
    orders are not tried: a mirrored court is a different answer.

    :param predicted: (scene, corner, xy); NaN rows where a scene has no court.
    :param official: (corner, xy).
    """
    rotated = [np.roll(predicted, -rotation, axis=1) for rotation in range(4)]
    distances = np.stack([np.linalg.norm(courts - official, axis=2) for courts in rotated])  # (rotation, scene, corner)
    rotation = distances.mean(axis=2).argmin(axis=0)  # one per scene
    closest = distances[rotation, np.arange(len(predicted))]  # (scene, corner)

    official_f32 = official.astype(np.float32)
    iou = np.full(len(predicted), np.nan)
    for scene_index in np.flatnonzero(np.isfinite(predicted[:, 0, 0])):
        court = predicted[scene_index].astype(np.float32)
        if not cv2.isContourConvex(court):
            raise ValueError(f"scene {scene_index}: predicted court is not convex")
        intersection, _ = cv2.intersectConvexConvex(court, official_f32)
        iou[scene_index] = intersection / (cv2.contourArea(court) + cv2.contourArea(official_f32) - intersection)
    errors = {"error_mean_px": closest.mean(axis=1), "error_max_px": closest.max(axis=1)}
    return pd.DataFrame(errors | {"corner_rotation": rotation, "iou": iou})


def choose_representative(scenes: pd.DataFrame, predicted: np.ndarray) -> int | None:
    """Pick one detected scene court to stand for the video's playing view.

    The view group whose detected scenes share the most rally frames is chosen
    first. Within it, the representative is the weighted medoid: the member court
    with the smallest rally-frame-weighted mean corner distance to all member
    courts. Members whose sampled frame lies in a rally are preferred when any
    exist. The official corners play no part, and the result is always a court the
    detector output for a real scene.

    :param predicted: (scene, corner, xy) in detector corner order.
    :return: The scene index, or None when no detected scene overlaps a rally.
    """
    detected = scenes[scenes["court_detected"]]
    group_rally_frames = detected.groupby("view_group")["rally_frames"].sum()
    if group_rally_frames.empty or group_rally_frames.max() == 0:
        return None
    members = detected[detected["view_group"] == group_rally_frames.idxmax()]
    courts = predicted[members.index]  # (member, corner, xy)
    weights = members["rally_frames"].to_numpy()

    # pairwise[candidate, member]: mean corner distance between two member courts
    pairwise = np.linalg.norm(courts[:, None] - courts[None, :], axis=3).mean(axis=2)
    cost = (pairwise * weights).sum(axis=1) / weights.sum()
    sampled_in_rally = members["frame_index_in_rally"].to_numpy()
    if sampled_in_rally.any():
        cost[~sampled_in_rally] = np.inf
    return int(members.index[cost.argmin()])


def analyse_video(
    entry: dict, output: dict, official: np.ndarray, rallies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """:return: One video's per-scene and per-rally tables, scored against its official corners."""
    frame_count = output["frame_count"]
    scenes = pd.DataFrame(output["scenes"])
    final_corners = scenes["corners_native_px"]
    scenes = scenes.reindex(columns=SCENE_COLUMNS)
    starts, ends = scenes["start_frame"].to_numpy(), scenes["end_frame"].to_numpy()
    spans_whole_video = starts[0] == 0 and ends[-1] == frame_count and (starts[1:] == ends[:-1]).all()
    if not spans_whole_video:
        raise ValueError(f"{entry['id']}: scenes do not partition [0, {frame_count})")

    detected = final_corners.notna().to_numpy()
    corners = np.full((len(scenes), 4, 2), np.nan)  # (scene, corner, xy) in native px
    corners[detected] = np.array(final_corners[detected].tolist()).reshape(-1, 4, 2)
    scale = np.array(HOMOGRAPHY_RESOLUTION) / np.array(output["native_size"])
    predicted = corners * scale
    original_corners = corners.copy()
    for index, scene in enumerate(output["scenes"]):
        # A scene that took a shared court keeps its own beside it. A courtless receiver's own is None.
        if "scene_corners_native_px" in scene:
            own = scene["scene_corners_native_px"]
            original_corners[index] = np.nan if own is None else own

    in_rally = np.zeros(frame_count, dtype=bool)  # one per video frame; overlapping rallies count once
    for start, end in zip(rallies["start_frame"], rallies["end_frame"]):
        in_rally[start:end] = True
    rally_frames_before = np.concatenate([[0], np.cumsum(in_rally)])  # (frame_count + 1,)

    # Scenes outside every detector view group stand as groups of their own. Outputs
    # from before courtless receivers have no receiver_view_ids.
    group_of_view = {
        view_id: group_index
        for group_index, group in enumerate(output["view_groups"])
        for view_id in [*group["member_view_ids"], *group.get("receiver_view_ids", [])]
    }
    view_group = scenes["view_id"].map(group_of_view)
    ungrouped = view_group.isna()
    view_group[ungrouped] = len(output["view_groups"]) + np.arange(ungrouped.sum())

    scenes.insert(0, "scene_index", scenes.index)
    scenes["court_detected"] = detected
    scenes["view_group"] = view_group.astype(int)
    scenes["rally_frames"] = rally_frames_before[ends] - rally_frames_before[starts]
    scenes["frame_index_in_rally"] = in_rally[scenes["frame_index"].to_numpy()]
    scenes = pd.concat([scenes, score_courts(predicted, official)], axis=1)
    before_pool = score_courts(original_corners * scale, official).add_prefix("before_pool_")
    scenes = pd.concat([scenes, before_pool], axis=1)
    scenes[CORNER_COLUMNS] = corners.reshape(len(scenes), 8)

    rally_start, rally_end = rallies["start_frame"].to_numpy(), rallies["end_frame"].to_numpy()
    # shared[rally, scene]: frames a rally and a scene have in common
    shared = np.clip(np.minimum(rally_end[:, None], ends) - np.maximum(rally_start[:, None], starts), 0, None)
    # One scene per rally, the first of any tied scenes. A rally counts as missed when
    # this scene has no court, even if a shorter scene in the rally has one.
    majority = shared.argmax(axis=1)
    majority_scenes = scenes.loc[majority].reset_index(drop=True)
    rallies = rallies.assign(shared_frames=shared[np.arange(len(rallies)), majority])
    majority_columns = ["scene_index", "view_id", "court_detected", *SCORE_COLUMNS]
    rallies = pd.concat([rallies, majority_scenes[majority_columns]], axis=1)

    scenes["majority_rallies"] = np.bincount(majority, minlength=len(scenes))
    scenes["is_representative"] = scenes.index == choose_representative(scenes, predicted)
    selected = scenes.loc[scenes["is_representative"], "view_group"]
    scenes["is_dominant_group"] = scenes["view_group"].isin(selected)
    for table in (scenes, rallies):
        table.insert(0, "dataset", entry["dataset"])
        table.insert(0, "video_id", entry["id"])
    return scenes, rallies


def video_row(entry: dict, output: dict, scenes: pd.DataFrame, rallies: pd.DataFrame) -> dict[str, object]:
    """Roll one video's scene and rally tables up into its per-video row."""
    detected = scenes[scenes["court_detected"]]
    rally_frames = int(scenes["rally_frames"].sum())
    row: dict[str, object] = {
        "video_id": entry["id"],
        "dataset": entry["dataset"],
        "source_id": entry["source_id"],
        "source_video": entry["video"],
        "fps": output["fps"],
        "frame_count": output["frame_count"],
        "native_width": output["native_size"][0],
        "native_height": output["native_size"][1],
        "scenes": len(scenes),
        "detected_scenes": len(detected),
        "rallies": len(rallies),
        "rally_frames": rally_frames,
        "detected_rally_frames": int(detected["rally_frames"].sum()),
        "time_coverage": detected["rally_frames"].sum() / rally_frames,
        "majority_detected_rallies": int(rallies["court_detected"].sum()),
        "majority_coverage": rallies["court_detected"].mean(),
        "majority_error_mean_px_median": rallies["error_mean_px"].median(),
    }
    representative = scenes[scenes["is_representative"]]
    if len(representative) == 1:
        scene = representative.iloc[0]
        group = detected[detected["view_group"] == scene["view_group"]]
        row |= {f"representative_{name}": scene[name] for name in REPRESENTATIVE_COLUMNS}
        row["representative_group_scenes"] = len(group)
        row["representative_group_rally_frames"] = int(group["rally_frames"].sum())
    return row


def tail_spread(values: pd.Series, tail_percent: int) -> dict[str, float | None]:
    """:return: The median and one tail percentile over videos, ignoring missing values."""
    return {
        "median": finite_or_none(values.median()),
        f"p{tail_percent}": finite_or_none(values.quantile(tail_percent / 100)),
    }


def within_tolerances(errors: pd.Series) -> dict[str, dict[str, float]]:
    """:return: Count and share of all rows at or under each tolerance; a missing court is not under."""
    shares = {}
    for tolerance in TOLERANCES_PX:
        count = int((errors <= tolerance).sum())
        shares[f"le_{tolerance}px"] = {"count": count, "of": len(errors), "fraction": count / len(errors)}
    return shares


def bootstrap_mean(values: pd.Series, draws: np.ndarray) -> dict[str, float | None]:
    """Mean over videos with a percentile 95% interval from resampling whole videos.

    :param draws: (replicate, video) row positions drawn with replacement.
    """
    replicate_means = np.nanmean(values.to_numpy(dtype=float)[draws], axis=1)
    low, high = np.nanpercentile(replicate_means, [2.5, 97.5])
    return {"mean": finite_or_none(values.mean()), "ci95_low": finite_or_none(low), "ci95_high": finite_or_none(high)}


def summarise(videos: pd.DataFrame, rallies: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
    """Describe one group of videos. The video is the unit: each video counts once."""
    error = videos["representative_error_mean_px"]
    # Scenes and rallies from one video share a camera and a court, so only whole
    # videos are resampled.
    draws = rng.integers(0, len(videos), size=(BOOTSTRAP_REPLICATES, len(videos)))
    return {
        "videos": len(videos),
        "videos_with_representative": int(error.notna().sum()),
        "representative_frame_in_rally": int(videos["representative_frame_index_in_rally"].eq(True).sum()),
        "representative_error_mean_px": tail_spread(error, 90) | bootstrap_mean(error, draws),
        "representative_error_max_px": tail_spread(videos["representative_error_max_px"], 90),
        "representative_iou": tail_spread(videos["representative_iou"], 10),
        "representative_within": within_tolerances(error),
        "time_coverage": tail_spread(videos["time_coverage"], 10) | bootstrap_mean(videos["time_coverage"], draws),
        "rally_frames": int(videos["rally_frames"].sum()),
        "detected_rally_frames": int(videos["detected_rally_frames"].sum()),
        "rallies": len(rallies),
        "majority_detected_rallies": int(rallies["court_detected"].sum()),
        "majority_coverage_per_video": tail_spread(videos["majority_coverage"], 10),
        "majority_error_mean_px_per_video_median": tail_spread(videos["majority_error_mean_px_median"], 90),
        "majority_rallies_within": within_tolerances(rallies["error_mean_px"]),
    }


def render_request(scene: pd.Series, video: pd.Series, reason: str, output: str, caption: str) -> dict[str, object]:
    """Describe one frame to draw: which frame, the court's native corners, and why."""
    corners = scene[CORNER_COLUMNS].to_numpy(dtype=float).reshape(4, 2)
    return {
        "id": scene["video_id"],
        "dataset": scene["dataset"],
        "source_video": video["source_video"],
        "frame_index": int(scene["frame_index"]),
        "native_size": [int(video["native_width"]), int(video["native_height"])],
        "corners_native_px": corners.tolist(),
        "reason": reason,
        "metrics": {name: float(scene[name]) for name in SCORE_COLUMNS}
        | {"frame_index_in_rally": bool(scene["frame_index_in_rally"])}
        | {"majority_rallies": int(scene["majority_rallies"])},
        "output": output,
        "caption": caption,
    }


def build_render_requests(scenes: pd.DataFrame, videos: pd.DataFrame) -> list[dict[str, object]]:
    """List one representative frame per video, then the hard samples for each dataset."""
    videos = videos.set_index("video_id")
    requests = []
    for _, scene in scenes[scenes["is_representative"]].iterrows():
        caption = (
            f"{scene['video_id']} frame {scene['frame_index']}: detected court chosen to stand for the playing view. "
            f"Mean corner distance to the official corners is {scene['error_mean_px']:.1f} px at 1280x720."
        )
        output = f"courts/{scene['video_id']}.png"
        requests.append(render_request(scene, videos.loc[scene["video_id"]], "representative", output, caption))

    # Hard samples: scenes that carry most of at least one rally and whose sampled
    # frame is inside a rally. The worst per video is kept so the examples come from
    # different videos.
    eligible = scenes[scenes["court_detected"] & (scenes["majority_rallies"] > 0) & scenes["frame_index_in_rally"]]
    worst_per_video = eligible.loc[eligible.groupby("video_id")["error_mean_px"].idxmax()]
    for _, dataset_scenes in worst_per_video.groupby("dataset"):
        ranked = dataset_scenes.nlargest(HARD_SAMPLES_PER_DATASET, "error_mean_px")
        for _, scene in ranked.iterrows():
            caption = (
                f"{scene['video_id']} frame {scene['frame_index']}: disagreement example, not a proven detector error. "
                f"Labelled rallies that fall mostly in this scene: {scene['majority_rallies']}. Its court is "
                f"{scene['error_mean_px']:.1f} px from the official corners on average (largest "
                f"{scene['error_max_px']:.1f} px, overlap {scene['iou']:.2f}) at 1280x720. "
                f"The official corners are one static set per match."
            )
            output = f"hard_samples/{scene['video_id']}_frame_{scene['frame_index']}.png"
            requests.append(render_request(scene, videos.loc[scene["video_id"]], "hard_sample", output, caption))
    return requests


def analyse(input_root: Path, dataset_roots: dict[str, Path], output_dir: Path) -> None:
    """Score every cohort video and write the tables, summary and render requests."""
    cohort = read_json_gz(input_root / "cohort.json.gz")
    cohort_sources = {(entry["dataset"], entry["source_id"]) for entry in cohort}
    if cohort_sources != released_sources() or len(cohort) != len(cohort_sources):
        raise ValueError(f"cohort differs from the release manifests: {sorted(cohort_sources ^ released_sources())}")
    matches = {name: read_label_table(root, "match").set_index("id") for name, root in dataset_roots.items()}
    homographies = {
        name: read_label_table(root, "homography").set_index("id")
        for name, root in dataset_roots.items()
    }

    scene_tables, rally_tables, video_rows = [], [], []
    for entry in cohort:
        output = read_json_gz(input_root / "videos" / f"{entry['id']}.json.gz")
        match_id = int(entry["source_id"])
        official, reference_notes = read_official_corners(homographies[entry["dataset"]].loc[match_id], entry["dataset"])
        set_dir = dataset_roots[entry["dataset"]] / matches[entry["dataset"]].loc[match_id, "video"]
        rallies, label_counts = read_rallies(set_dir, output["frame_count"])
        scenes, rallies = analyse_video(entry, output, official, rallies)
        scene_tables.append(scenes)
        rally_tables.append(rallies)
        official_names = [f"official_{name}_{axis}" for name in CORNER_NAMES for axis in "xy"]
        official_columns = dict(zip(official_names, official.ravel().tolist()))
        video_rows.append(
            video_row(entry, output, scenes, rallies)
            | label_counts
            | reference_notes
            | official_columns
        )

    per_scene = pd.concat(scene_tables, ignore_index=True)
    per_rally = pd.concat(rally_tables, ignore_index=True)
    per_video = pd.DataFrame(video_rows)
    requests = build_render_requests(per_scene, per_video)

    no_representative = per_video["representative_error_mean_px"].isna()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {
        "schema": "court-evaluation/1",
        "pixel_space": f"{HOMOGRAPHY_RESOLUTION[0]}x{HOMOGRAPHY_RESOLUTION[1]} official annotation pixels",
        "tolerances_px": list(TOLERANCES_PX),
        "tolerance_note": "Declared in advance to describe the error spread; not validated acceptance bounds.",
        "bootstrap": {"unit": "video", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "cohort_matches_release_manifests": True,
        "label_rows": int(per_video["label_rows"].sum()),
        "nonfinite_frame_rows": int(per_video["nonfinite_frame_rows"].sum()),
        "excluded_incomplete_rallies": int(per_video["excluded_incomplete_rallies"].sum()),
        "excluded_non_monotonic_rallies": int(per_video["excluded_non_monotonic_rallies"].sum()),
        "official_bottom_pair_swapped": per_video.loc[per_video["official_bottom_pair_swapped"], "video_id"].tolist(),
        "videos_without_representative": per_video.loc[no_representative, "video_id"].tolist(),
        "render_requests": pd.Series([request["reason"] for request in requests]).value_counts().to_dict(),
        "overall": summarise(per_video, per_rally, rng),
        "by_dataset": {},
    }
    for dataset, dataset_videos in per_video.groupby("dataset"):
        summary["by_dataset"][dataset] = summarise(dataset_videos, per_rally[per_rally["dataset"] == dataset], rng)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_scene.to_csv(output_dir / "per_scene.csv.gz", index=False)
    per_rally.to_csv(output_dir / "per_rally.csv.gz", index=False)
    per_video.to_csv(output_dir / "per_video.csv.gz", index=False)
    captions = pd.json_normalize(requests).drop(columns=["corners_native_px", "source_video", "native_size"])
    captions.to_csv(output_dir / "render_captions.csv.gz", index=False)
    write_json_gz(output_dir / "summary.json.gz", summary)
    write_json_gz(output_dir / "render_requests.json.gz", requests)
    pooling_diagnostic(per_scene).to_csv(output_dir / "pooling_diagnostic.csv.gz", index=False)
    print(json.dumps(summary["overall"], indent=1))


def pooling_diagnostic(scenes: pd.DataFrame) -> pd.DataFrame:
    """Compare retained scene fits with final courts in each dominant rally group.

    This uses already saved fits. It does not rerun the detector or change the
    primary evaluation. The 20 px threshold applies to every corner here.
    """
    eligible = scenes[scenes["is_dominant_group"] & scenes["court_detected"] & (scenes["rally_frames"] > 0)]
    rows = []
    for video_id, group in eligible.groupby("video_id"):
        before = group["before_pool_error_max_px"] <= 20
        after = group["error_max_px"] <= 20
        rows.append({
            "video_id": video_id,
            "detected_rally_scenes_in_dominant_group": len(group),
            "all_corners_within20_before_pool": int(before.sum()),
            "all_corners_within20_final": int(after.sum()),
            "lost_after_pool": int((before & ~after).sum()),
            "gained_after_pool": int((~before & after).sum()),
        })
    return pd.DataFrame(rows)


def stripe_outlines_m() -> np.ndarray:
    """:return: (stripe, vertex, xy) court-metre rectangles covering each painted stripe."""
    starts, ends = CENTRE_SEGMENTS_M[:, 0], CENTRE_SEGMENTS_M[:, 1]  # (stripe, xy) centre-line ends
    along = (ends - starts) / np.linalg.norm(ends - starts, axis=1, keepdims=True)
    half_width = np.stack([-along[:, 1], along[:, 0]], axis=1) * (STRIPE_WIDTH_M / 2)
    # The centre segments already run to the outer edge of the paint they join, so
    # the end caps sit at the segment ends.
    return np.stack([starts + half_width, ends + half_width, ends - half_width, starts - half_width], axis=1)


def clip_to_frame(start: np.ndarray, end: np.ndarray, width: int, height: int) -> tuple[float, float] | None:
    """:return: The (enter, exit) fractions of start->end that lie on the frame, or None when none does."""
    enter, leave = 0.0, 1.0
    for axis, last_pixel in ((0, width - 1), (1, height - 1)):
        step = end[axis] - start[axis]
        if step == 0:
            if not 0 <= start[axis] <= last_pixel:
                return None
            continue
        crossings = sorted(((0 - start[axis]) / step, (last_pixel - start[axis]) / step))
        enter, leave = max(enter, crossings[0]), min(leave, crossings[1])
    return (enter, leave) if enter < leave else None


def draw_dashed_edge(image: np.ndarray, start: np.ndarray, end: np.ndarray) -> None:
    """Draw the on-frame part of one outline edge as 1 px red dashes.

    Dashes are counted from ``start`` wherever it lies, but only those inside the
    clipped range are visited. An edge running far off the frame costs no more than
    its visible part.
    """
    visible = clip_to_frame(start, end, image.shape[1], image.shape[0])
    if visible is None:
        return
    length = float(np.linalg.norm(end - start))
    direction = (end - start) / length
    visible_from, visible_to = visible[0] * length, visible[1] * length  # px along the edge
    period = DASH_PX + GAP_PX
    for dash_index in range(int(visible_from // period), int(visible_to // period) + 1):
        dash_from = max(dash_index * period, visible_from)
        dash_to = min(dash_index * period + DASH_PX - 1, visible_to)
        if dash_from <= dash_to:
            first = np.rint(start + direction * dash_from).astype(int)
            last = np.rint(start + direction * dash_to).astype(int)
            cv2.line(image, tuple(first.tolist()), tuple(last.tolist()), RED_BGR, 1, cv2.LINE_8)


def draw_court(image: np.ndarray, corners_native: np.ndarray) -> None:
    """Outline every painted stripe of the court whose outer corners are given."""
    metres_to_native = cv2.getPerspectiveTransform(CORNER_COURT_M, corners_native.astype(np.float32))
    outlines_m = stripe_outlines_m()
    homogeneous = np.concatenate([outlines_m, np.ones((*outlines_m.shape[:2], 1))], axis=2) @ metres_to_native.T
    depth = homogeneous[..., 2]  # (stripe, vertex)
    if not (np.isfinite(homogeneous).all() and (depth > 0).all()):
        raise ValueError("a stripe vertex projects at or behind the camera horizon")
    outlines_px = homogeneous[..., :2] / depth[..., None]
    for outline in outlines_px:
        for start_vertex, end_vertex in OUTLINE_EDGES:
            draw_dashed_edge(image, outline[start_vertex], outline[end_vertex])


def render(requests_path: Path, frames_dir: Path, output_dir: Path) -> int:
    """Draw each request on its native frame, named by the request's output stem.

    :return: How many requests could not be drawn.
    """
    failures = 0
    for request in read_json_gz(requests_path):
        stem = Path(request["output"]).stem
        frame_paths = [path for path in frames_dir.glob(f"{stem}.*") if path.suffix.lower() in FRAME_SUFFIXES]
        try:
            if len(frame_paths) != 1:
                raise ValueError(f"expected one frame named {stem}, found {len(frame_paths)}")
            image = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot read {frame_paths[0]}")
            if list(image.shape[1::-1]) != request["native_size"]:
                raise ValueError(f"frame is {image.shape[1]}x{image.shape[0]}, expected {request['native_size']}")
            if request["corners_native_px"] is not None:
                draw_court(image, np.array(request["corners_native_px"], dtype=float))
        except ValueError as error:
            # One bad frame should not stop the other drawings.
            failures += 1
            print(f"FAILED {request['output']}: {error}")
            continue
        destination = output_dir / request["output"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"cannot write {destination}")
    return failures


def fetch_frames(requests_path: Path, output_dir: Path) -> None:
    """Decode the exact requested frames on the host that holds the source videos."""
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = read_json_gz(requests_path)
    for index, request in enumerate(requests):
        capture = cv2.VideoCapture(request["source_video"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, request["frame_index"])
        ok, image = capture.read()
        next_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        capture.release()
        if not ok or next_frame != request["frame_index"] + 1:
            raise RuntimeError(f"{request['id']}: could not decode frame {request['frame_index']}; next={next_frame}")
        destination = output_dir / (Path(request["output"]).stem + ".png")
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"cannot write {destination}")
        print(f"{index + 1}/{len(requests)} {destination.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    analyse_parser = commands.add_parser("analyse", help="score detector outputs against the official labels")
    analyse_parser.add_argument("--input-root", type=Path, required=True, help="holds cohort.json.gz and videos/")
    analyse_parser.add_argument("--shuttleset-root", type=Path, required=True, help="ShuttleSet set/ directory")
    analyse_parser.add_argument("--shuttleset22-root", type=Path, required=True, help="ShuttleSet22 set/ directory")
    analyse_parser.add_argument("--output", type=Path, required=True)
    render_parser = commands.add_parser("render", help="draw requested courts on native frames")
    render_parser.add_argument("--requests", type=Path, required=True, help="render_requests.json.gz")
    render_parser.add_argument("--frames", type=Path, required=True, help="native frames named <output stem>.jpg/png")
    render_parser.add_argument("--output", type=Path, required=True)
    fetch_parser = commands.add_parser("fetch-frames", help="decode requested frames on the video host")
    fetch_parser.add_argument("--requests", type=Path, required=True)
    fetch_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "analyse":
        dataset_roots = {"ShuttleSet": args.shuttleset_root, "ShuttleSet22": args.shuttleset22_root}
        analyse(args.input_root, dataset_roots, args.output)
    elif args.command == "fetch-frames":
        fetch_frames(args.requests, args.output)
    else:
        failures = render(args.requests, args.frames, args.output)
        if failures:
            raise SystemExit(f"{failures} requests failed")


if __name__ == "__main__":
    main()
