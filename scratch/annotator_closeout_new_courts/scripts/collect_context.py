"""Join each labelled ShuttleSet22 contact to the court and player state at its frame.

Court state comes from the run's converted court evidence. Player picks come
from the rerun in ``recompute_player_state.py``, whose streams matched the
saved ones exactly.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COURT = ROOT / "raw/run/output/shared/court/test"
PLAYERS = ROOT / "raw/player_state"
NEARBY_FRAMES_BASE30 = 10
INPUT_STATES = ("Court rejected", "Court accepted; a player pick missing", "Court accepted; both players picked")


def scene_table(video: str) -> pd.DataFrame:
    with gzip.open(COURT / video / "court_evidence.json.gz", "rt") as source:
        records = json.load(source)["scene_records"]
    scenes = pd.DataFrame([{
        "video": int(video), "scene_index": record["scene_index"], "start": record["start_frame"],
        "end": record["end_frame"], "status": record["status"], "no_court_reason": record["no_court_reason"],
        "has_court": record["corners_native_px"] is not None, "reused_from": record["reused_from"],
        "exactly_two_fraction": record["exactly_two_fraction"], "scene_valid": record["scene_valid"],
    } for record in records])
    assert (scenes.start.to_numpy()[1:] == scenes.end.to_numpy()[:-1]).all(), video
    return scenes


def video_context(video: str, frames: np.ndarray, fps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenes = scene_table(video)
    state = np.load(PLAYERS / f"{video}.npz")
    court_present = state["court_present"]
    expanded = np.zeros(len(court_present), dtype=bool)
    for scene in scenes.itertuples(index=False):
        expanded[scene.start:scene.end] = scene.scene_valid
    assert np.array_equal(expanded, court_present), video
    assert np.array_equal(state["tracker_covered"], court_present), video

    scene_position = np.searchsorted(scenes.start.to_numpy(), frames, side="right") - 1
    at_frame = scenes.iloc[scene_position].reset_index(drop=True)
    far_picked = state["picks"][frames, 0] >= 0
    near_picked = state["picks"][frames, 1] >= 0
    accepted = court_present[frames]
    both_picked = far_picked & near_picked

    # Feature rows exist only where contact search ran; a nearby row means a scored candidate was possible.
    feature_frames = np.sort(state["frame"])
    window = round(NEARBY_FRAMES_BASE30 * fps / 30)
    lower = np.searchsorted(feature_frames, frames - window, side="left")
    upper = np.searchsorted(feature_frames, frames + window, side="right")
    context = pd.DataFrame({
        "video": int(video), "source_frame": frames, "scene_index": at_frame.scene_index,
        "scene_status": at_frame.status, "no_court_reason": at_frame.no_court_reason,
        "scene_has_court": at_frame.has_court, "scene_reused_court": at_frame.reused_from.notna(),
        "exactly_two_fraction": at_frame.exactly_two_fraction, "court_present": accepted,
        "far_picked": far_picked, "near_picked": near_picked, "nearby_feature_rows": upper - lower,
        "input_state": np.select([~accepted, ~both_picked], INPUT_STATES[:2], default=INPUT_STATES[2]),
        "court_rejection": np.select([accepted, ~at_frame.has_court.to_numpy()],
                                     ["Accepted", "No court found"], default="Court found; player check failed"),
    })
    return context, scenes


def run() -> None:
    contacts = pd.read_csv(ROOT / "results/contacts.csv.gz", dtype={"video": str})
    labelled = contacts[(contacts.split == "test") & (contacts.tolerance_base30 == 10)]
    labelled = labelled.assign(video=labelled.video.astype(int))
    contexts, scenes = [], []
    for video, group in labelled.groupby("video"):
        frames = np.unique(group.source_frame.to_numpy())
        context, scene_rows = video_context(str(video), frames, float(group.fps.iloc[0]))
        contexts.append(context)
        scenes.append(scene_rows)
    context = pd.concat(contexts, ignore_index=True)
    context.to_csv(ROOT / "results/contexts.csv.gz", index=False)
    pd.concat(scenes, ignore_index=True).to_csv(ROOT / "results/scenes.csv.gz", index=False)
    joined = labelled.merge(context, on=["video", "source_frame"], validate="many_to_one")
    assert len(joined) == len(labelled) == 37184
    print(joined.groupby(["input_state", "matched"]).size().unstack().to_string())
    print(joined.groupby(["court_rejection", "matched"]).size().unstack().to_string())


if __name__ == "__main__":
    run()
