"""Recompute the detector-audit findings from the saved run.

Run from the repository root with ``PYTHONPATH=src``: the net-band check calls the annotator's
own court functions. It reads the Git-ignored ``raw/run/`` pull as well as ``results/``.

Four checks:
- court rejections: which detector stages ran in each rejected scene, and whether the scene's
  3-second foot window overlaps a labelled rally;
- oversized courts: vote-failed scenes whose court runs far outside the frame or is much larger
  than the video's accepted courts;
- the false final hit: its score, its place in the chosen sequence, the detected landing and
  the point winner;
- side errors: wrong sides grouped by where a rally's misses fall, and the net bands of
  videos 17 and 42.
"""

from __future__ import annotations

import gzip
import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from summarise_court_change import labelled_court_states
from trace_label_rows import official_rows

from annotator.courts.evidence import (
    _as_ref_corners,
    build_net_band,
    detected_court_info,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "raw/run"
FRAME_SIZE = (1920.0, 1080.0)  # every ShuttleSet22 test video
WINDOW_HALF_FRAMES = 45  # 31 samples at 10 fps span three seconds at 30 fps
EARLY_EXIT_STAGES = {"feet", "context"}
OFF_FRAME_SHARE = 0.1
AREA_RATIO_LIMIT = 2.0
OTHER_HALF = {"Top": "Bot", "Bot": "Top"}


def test_rows(path: str) -> pd.DataFrame:
    table = pd.read_csv(ROOT / path, dtype={"video": str})
    return table[(table.split == "test") & (table.tolerance_base30 == 10)]


def read_json(path: Path) -> dict:
    with gzip.open(path, "rt") as source:
        return json.load(source)


@cache
def release_scenes(video: str) -> list[dict]:
    return read_json(RUN / f"courts/videos/ss22_{int(video):02d}.json.gz")["scenes"]


@cache
def run_scenes(video: str) -> list[dict]:
    return read_json(RUN / f"output/shared/court/test/{video}/court_evidence.json.gz")["scene_records"]


@cache
def annotator_result(video: str) -> dict:
    return read_json(RUN / f"output/base/eval/test/{video}/annotator_result.json.gz")["result"]


@cache
def stream(video: str) -> dict:
    return read_json(RUN / f"output/base/eval/test/{video}/stream.json.gz")


def scene_kind(scene: dict) -> str:
    stages = set(scene.get("stage_seconds") or {})
    if scene["status"] == "no_court" and stages and stages <= EARLY_EXIT_STAGES:
        return "stopped before search"
    return scene.get("no_court_reason") or scene["status"]


def window_in_rally(video: str, scene: dict, rallies: pd.DataFrame) -> bool:
    start, end = scene["frame_index"] - WINDOW_HALF_FRAMES, scene["frame_index"] + WINDOW_HALF_FRAMES
    video_rallies = rallies[rallies.video == video]
    return bool(((video_rallies.first_frame <= end) & (video_rallies.last_frame >= start)).any())


def court_rejections(labels: pd.DataFrame, rallies: pd.DataFrame) -> None:
    missed = labels[~labels.matched & (labels.court_rejection != "Accepted")]
    missed = missed.assign(kind=[scene_kind(release_scenes(row.video)[row.scene_index]) for row in missed.itertuples()],
                           scene=missed.video + ":" + missed.scene_index.astype(str))
    print("Rejected-scene misses by detector outcome:")
    print(missed.groupby("kind").agg(misses=("source_frame", "size"), scenes=("scene", "nunique")).to_string())
    stopped = missed[missed.kind == "stopped before search"].drop_duplicates(["video", "scene_index"])
    accepted = labels[labels.court_rejection == "Accepted"].drop_duplicates(["video", "scene_index"])
    for name, scenes in (("stopped before search", stopped), ("accepted with labelled hits", accepted)):
        records = [release_scenes(row.video)[row.scene_index] for row in scenes.itertuples()]
        overlaps = [window_in_rally(row.video, record, rallies) for row, record in zip(scenes.itertuples(), records,
                                                                                       strict=True)]
        lengths = [record["end_frame"] - record["start_frame"] for record in records]
        print(f"{name}: {len(records)} scenes, median {np.median(lengths):.0f} frames, "
              f"foot window overlaps a labelled rally in {100 * np.mean(overlaps):.1f}%")
    newly = labelled_court_states()
    newly = newly[newly.court_present_old & ~newly.court_present_new].assign(video=lambda table: table.video.astype(str))
    newly = newly.merge(labels[["video", "source_frame", "scene_index"]], on=["video", "source_frame"])
    kinds = pd.Series([scene_kind(release_scenes(row.video)[row.scene_index]) for row in newly.itertuples()])
    print(f"Newly rejected hits: {len(newly)}; stopped before search: {int((kinds == 'stopped before search').sum())}")


def polygon_area(corners: np.ndarray) -> float:
    x, y = corners.T
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def court_shape(video: str, scene_index: int) -> dict:
    scenes = run_scenes(video)
    accepted_areas = [polygon_area(np.asarray(scene["corners_native_px"], float))
                      for scene in scenes if scene.get("scene_valid")]
    corners = np.asarray(scenes[scene_index]["corners_native_px"], float)
    width, height = FRAME_SIZE
    x, y = corners.T
    off_frame = bool(((x < -OFF_FRAME_SHARE * width) | (x > (1 + OFF_FRAME_SHARE) * width)
                      | (y < -OFF_FRAME_SHARE * height) | (y > (1 + OFF_FRAME_SHARE) * height)).any())
    area_ratio = polygon_area(corners) / float(np.median(accepted_areas))
    return {"off_frame": off_frame, "area_ratio": area_ratio, "vote": scenes[scene_index]["exactly_two_fraction"]}


def oversized_courts(labels: pd.DataFrame, sample: pd.DataFrame) -> None:
    missed = labels[~labels.matched & (labels.court_rejection == "Court found; player check failed")]
    scenes = missed.groupby(["video", "scene_index"]).size().rename("misses").reset_index()
    shapes = pd.DataFrame([court_shape(row.video, row.scene_index) for row in scenes.itertuples()])
    scenes = pd.concat([scenes, shapes], axis=1)
    scenes["oversized"] = scenes.off_frame | (scenes.area_ratio > AREA_RATIO_LIMIT)
    print("Vote-failed misses by court shape:")
    print(scenes.groupby("oversized").agg(misses=("misses", "sum"), scenes=("misses", "size"),
                                          median_vote=("vote", "median")).to_string())
    print(f"Misses in vote-failed scenes at 40% or above: {int(scenes[scenes.vote >= 0.4].misses.sum())}")
    checked = sample[sample.court_rejection == "Court found; player check failed"]
    for row in checked.itertuples():
        shape = court_shape(row.video, int(row.scene_index))
        print(f"Footage-checked vote failure, video {row.video}: off frame {shape['off_frame']}, "
              f"area {shape['area_ratio']:.2f}x, vote {shape['vote']:.3f}")


def official_winner(final_row: pd.Series) -> str | None:
    if pd.isna(final_row.official_side) or pd.isna(final_row.getpoint_player):
        return None
    if final_row.getpoint_player == final_row.player:
        return final_row.official_side
    return OTHER_HALF[final_row.official_side]


def winner_agreement(groups: pd.DataFrame) -> pd.Series:
    agree = []
    for video, group in groups.groupby("video"):
        final_rows = official_rows(video).sort_values(["rally_id", "ball_round"]).groupby("rally_id").tail(1)
        final_rows = final_rows.set_index("rally_id")
        next_servers = annotator_result(video)["next_servers"]
        for row in group.itertuples():
            winner = official_winner(final_rows.loc[row.rally_id])
            if winner is not None:
                agree.append({"group": row.group, "agree": next_servers[row.span_id] == winner})
    return pd.DataFrame(agree).groupby("group").agree.agg(["size", "mean"])


def false_final_hit(predictions: pd.DataFrame, proposals: pd.DataFrame) -> None:
    tail = pd.read_csv(ROOT / "results/tail_rallies.csv.gz", dtype={"video": str})
    tail = tail[tail.sole_error].merge(predictions[["video", "source_frame", "probability", "span_id"]],
                                       left_on=["video", "query_frame"], right_on=["video", "source_frame"])
    tail = tail.assign(span_id=tail.span_id.astype(int))
    print(f"False final hits: {len(tail)}; scored 0.9 or above: {int((tail.probability >= 0.9).sum())}; "
          f"median score {tail.probability.median():.2f}")
    last_event, landing_after, landings, striker = 0, 0, 0, 0
    for row in tail.itertuples():
        sequence = next(item for item in stream(row.video)["sequences"] if item["span_id"] == row.span_id)
        frames = [event[0] for event in sequence["events"]]
        last_event += frames[-1] == row.query_frame
        result = annotator_result(row.video)
        landing = result["landings"].get(str(row.span_id))
        if landing:
            landings += 1
            landing_after += landing["frame"] > row.query_frame
        striker += result["striker_halves"][row.span_id] == row.predicted_side
    print(f"Last event of its sequence: {last_event}; landing found: {landings}, after the false hit: "
          f"{landing_after}; striker is the false hit's side: {striker}")
    correct = proposals[proposals.outcome == "correct"].dropna(subset=["rally_id"])
    groups = pd.concat([tail[["video", "rally_id", "span_id"]].assign(group="false final hit"),
                        correct[["video", "rally_id", "span_id"]].astype({"span_id": int})
                        .assign(group="fully correct")])
    print("Predicted point winner agrees with the official rows:")
    print(winner_agreement(groups.drop_duplicates(["video", "rally_id"])).round(3).to_string())


def miss_pattern(rally: pd.DataFrame, extras: float) -> str:
    missed = ~rally.matched.to_numpy()
    if len(missed) > 2 and missed[1:-1].any():
        return "interior miss"
    if missed[0] or missed[-1]:
        return "first or last hit missed"
    if extras > 0:
        return "extra hit only"
    return "clean"


def step_shape(rally: pd.DataFrame) -> str:
    wrong = rally.wrong.to_numpy()
    gap = int(np.argmax(~rally.matched.to_numpy()))
    before, after = wrong[:gap], wrong[gap + 1:]
    if not wrong.any():
        return "no wrong side"
    if (before.all() and not after.any()) or (after.all() and not before.any()):
        return "every hit on one side of the gap"
    return "other"


def net_band(video: str, scene_index: int) -> tuple[float, float]:
    corners = np.asarray(run_scenes(video)[scene_index]["corners_native_px"], float)
    return build_net_band(detected_court_info(_as_ref_corners(corners, FRAME_SIZE)), FRAME_SIZE)


def side_errors(labels: pd.DataFrame, proposals: pd.DataFrame) -> None:
    extras = proposals.dropna(subset=["rally_id"]).groupby(["video", "rally_id"]).extra.sum()
    labels = labels.sort_values(["video", "rally_id", "label_index"])
    labels = labels.assign(wrong=labels.matched & labels.target_side.notna() & ~labels.player_correct)
    rows = []
    for (video, rally_id), rally in labels.groupby(["video", "rally_id"]):
        rally_extras = extras.get((video, rally_id), np.nan)
        rows.append({"video": video, "pattern": miss_pattern(rally, rally_extras), "matched": int(rally.matched.sum()),
                     "wrong": int(rally.wrong.sum()),
                     "one_miss": int((~rally.matched).sum()) == 1 and not rally_extras > 0,
                     "shape": step_shape(rally) if (~rally.matched).any() else None})
    rallies = pd.DataFrame(rows)
    table = rallies.groupby("pattern").agg(rallies=("wrong", "size"), matched=("matched", "sum"), wrong=("wrong", "sum"))
    table["wrong_share"] = (table.wrong / table.matched).round(3)
    print("Wrong sides by where the rally's misses fall:")
    print(table.to_string())
    print("Rallies with one miss and no extra:", rallies[rallies.one_miss]["shape"].value_counts().to_dict())
    clean = rallies[rallies.pattern == "clean"]
    for video in ("17", "42"):
        in_video = clean[clean.video == video]
        print(f"Video {video}: clean-rally wrong sides {in_video.wrong.sum()} of {in_video.matched.sum()}")
    wrong = labels[labels.wrong]
    print("Video 42 wrong sides, labelled to predicted:",
          wrong[wrong.video == "42"].groupby(["target_side", wrong.predicted_side.fillna("none")]).size().to_dict())
    for video in ("17", "42"):
        scenes = run_scenes(video)
        accepted = [scene for scene in scenes if scene.get("scene_valid")]
        representative = max(accepted, key=lambda scene: scene["end_frame"] - scene["start_frame"])
        hit_scenes = labels[(labels.video == video) & (labels.court_rejection == "Accepted")].scene_index
        busiest = int(hit_scenes[hit_scenes != representative["scene_index"]].value_counts().index[0])
        print(f"Video {video}: video-wide band from scene {representative['scene_index']} "
              f"({representative['end_frame'] - representative['start_frame']} frames, vote "
              f"{representative['exactly_two_fraction']:.3f}) = {net_band(video, representative['scene_index'])}; "
              f"busiest other scene {busiest} band = {net_band(video, busiest)}")


def run() -> None:
    contexts = pd.read_csv(ROOT / "results/contexts.csv.gz", dtype={"video": str})
    labels = test_rows("results/contacts.csv.gz").merge(contexts, on=["video", "source_frame"], validate="many_to_one")
    rallies = test_rows("results/rallies.csv.gz")
    proposals = test_rows("results/proposals.csv.gz")
    predictions = test_rows("results/predictions.csv.gz")
    sample = pd.read_csv(ROOT / "results/miss_sample.csv.gz", dtype={"video": str})
    court_rejections(labels, rallies)
    oversized_courts(labels, sample)
    false_final_hit(predictions, proposals)
    side_errors(labels, proposals)


if __name__ == "__main__":
    run()
