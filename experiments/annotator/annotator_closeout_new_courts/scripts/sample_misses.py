"""Draw missed ShuttleSet22 contacts uniformly for direct footage checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261004
SAMPLE_SIZE = 24


def run() -> None:
    contacts = pd.read_csv(ROOT / "results/contacts.csv.gz", dtype={"video": str})
    labelled = contacts[(contacts.split == "test") & (contacts.tolerance_base30 == 10)]
    labelled = labelled.assign(video=labelled.video.astype(int))
    context = pd.read_csv(ROOT / "results/contexts.csv.gz")
    missed = labelled[~labelled.matched].merge(context, on=["video", "source_frame"], validate="many_to_one")
    assert len(missed) == 2984
    sample = missed.sample(SAMPLE_SIZE, random_state=SEED).sort_values(["video", "source_frame"])
    sample.insert(0, "sample_id", [f"M{number:02d}" for number in range(1, SAMPLE_SIZE + 1)])
    columns = ["sample_id", "video", "fps", "rally_id", "label_index", "labelled_contacts", "position",
               "source_frame", "target_side", "court_present", "court_rejection", "scene_index",
               "scene_status", "no_court_reason", "far_picked", "near_picked"]
    sample[columns].to_csv(ROOT / "results/miss_sample.csv.gz", index=False)
    print(sample[columns].to_string(index=False))


if __name__ == "__main__":
    run()
