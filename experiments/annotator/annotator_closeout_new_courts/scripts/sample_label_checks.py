"""Pick the footage cases that test whether bulk failures are label errors.

Three groups:
- tail: failed rallies whose labelled hits all match on the right side, where the first unmatched
  prediction after the last label follows it by 10–60 frames (30 fps clock);
- side: rallies where at least 80% of matched hits are assigned the other player, outside the
  time-shifted video 12 rallies;
- drift: one hit from each time-shifted video 12 rally not already in the miss sample.

Each case names one query frame. Judges see stills around it without knowing whether it came
from a label or a prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261004
TAIL_CASES = 16
TAIL_GAP_FRAMES = (10, 60)
SIDE_SHARE = 0.8
SIDE_MIN_MATCHED = 4
DRIFT_RALLIES = ("set1:3", "set1:8")


def test_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[(table.split == "test") & (table.tolerance_base30 == 10)]


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contacts = pd.read_csv(ROOT / "results/contacts.csv.gz", dtype={"video": str})
    predictions = pd.read_csv(ROOT / "results/predictions.csv.gz", dtype={"video": str})
    rallies = pd.read_csv(ROOT / "results/rallies.csv.gz", dtype={"video": str})
    return test_rows(contacts), test_rows(predictions), test_rows(rallies)


def tail_pool(contacts: pd.DataFrame, predictions: pd.DataFrame, rallies: pd.DataFrame) -> pd.DataFrame:
    all_right = contacts.groupby(["video", "rally_id"]).player_correct.all().rename("all_right")
    candidates = rallies.join(all_right, on=["video", "rally_id"])
    candidates = candidates[~candidates.fully_correct & candidates.all_right]
    unmatched = predictions[~predictions.matched]
    rows = []
    for rally in candidates.itertuples(index=False):
        after = unmatched[(unmatched.video == rally.video) & (unmatched.source_frame > rally.last_frame)]
        if after.empty:
            continue
        first_after = after.loc[after.source_frame.idxmin()]
        gap = (first_after.source_frame - rally.last_frame) * 30 / rally.fps
        if TAIL_GAP_FRAMES[0] <= gap <= TAIL_GAP_FRAMES[1]:
            rows.append({"video": rally.video, "fps": rally.fps, "rally_id": rally.rally_id,
                         "query_frame": int(first_after.source_frame), "last_label_frame": int(rally.last_frame),
                         "label_side": None, "predicted_side": first_after.predicted_side})
    return pd.DataFrame(rows)


def tail_cases(contacts: pd.DataFrame, predictions: pd.DataFrame, rallies: pd.DataFrame) -> pd.DataFrame:
    pool = tail_pool(contacts, predictions, rallies)
    print("tail pool:", len(pool), "rallies")
    sample = pool.sample(n=TAIL_CASES, random_state=SEED).assign(group="tail")
    last_sides = contacts.sort_values("source_frame").groupby(["video", "rally_id"]).target_side.last()
    sample["label_side"] = [last_sides[(row.video, row.rally_id)] for row in sample.itertuples()]
    return sample


def side_cases(contacts: pd.DataFrame) -> pd.DataFrame:
    matched = contacts[contacts.matched & contacts.predicted_side.notna()]
    wrong = matched.predicted_side != matched.target_side
    share = wrong.groupby([matched.video, matched.rally_id]).agg(["mean", "size"])
    swapped = share[(share["mean"] >= SIDE_SHARE) & (share["size"] >= SIDE_MIN_MATCHED)].index
    swapped = [key for key in swapped if key[0] != "12"]
    generator = np.random.default_rng(SEED)
    rows = []
    for video, rally_id in swapped:
        rally = matched[(matched.video == video) & (matched.rally_id == rally_id)
                        & (matched.predicted_side != matched.target_side)]
        chosen = rally.iloc[generator.integers(len(rally))]
        rows.append({"video": video, "fps": chosen.fps, "rally_id": rally_id, "query_frame": int(chosen.source_frame),
                     "last_label_frame": None, "label_side": chosen.target_side,
                     "predicted_side": chosen.predicted_side, "group": "side"})
    return pd.DataFrame(rows)


def drift_cases(contacts: pd.DataFrame) -> pd.DataFrame:
    generator = np.random.default_rng(SEED)
    rows = []
    for rally_id in DRIFT_RALLIES:
        missed = contacts[(contacts.video == "12") & (contacts.rally_id == rally_id) & ~contacts.matched]
        chosen = missed.iloc[generator.integers(len(missed))]
        rows.append({"video": "12", "fps": chosen.fps, "rally_id": rally_id, "query_frame": int(chosen.source_frame),
                     "last_label_frame": None, "label_side": chosen.target_side, "predicted_side": None,
                     "group": "drift"})
    return pd.DataFrame(rows)


def run() -> None:
    contacts, predictions, rallies = load()
    cases = pd.concat([tail_cases(contacts, predictions, rallies), side_cases(contacts), drift_cases(contacts)],
                      ignore_index=True)
    shuffled = cases.sample(frac=1, random_state=SEED).reset_index(drop=True)
    shuffled.insert(0, "case_id", [f"L{index + 1:02d}" for index in range(len(shuffled))])
    shuffled.to_csv(ROOT / "results/label_check_cases.csv.gz", index=False)
    print(shuffled.groupby("group").size().to_string())
    print(shuffled.to_string(index=False))


if __name__ == "__main__":
    run()
