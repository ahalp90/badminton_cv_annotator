"""Split rally gains and losses by whether the new courts changed each rally's scene decision.

A rally counts as court-rejected under a run when any of its labelled contacts
falls in a frame that run's court gate rejected. Old-court states come from the
earlier evaluation's saved contexts; the fresh old-court refit used the same
court inputs as that historical run.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[2]
EARLIER = REPOSITORY / "scratch/annotator_wrapup_evaluation"
GROUP_ORDER = ["Accepted under both courts", "Rescued by the new courts", "Rejected under both courts",
               "Rejected only under the new courts"]


def labelled_court_states() -> pd.DataFrame:
    contacts = pd.read_csv(ROOT / "results/contacts.csv.gz", dtype={"video": str})
    contacts = contacts[(contacts.split == "test") & (contacts.tolerance_base30 == 10)]
    contacts = contacts.assign(video=contacts.video.astype(int))
    new = pd.read_csv(ROOT / "results/contexts.csv.gz")[["video", "source_frame", "court_present", "court_rejection"]]
    old = pd.read_csv(EARLIER / "results/contexts.csv.gz", usecols=["fixture", "source_frame", "court_present"])
    old = old.rename(columns={"fixture": "video"}).drop_duplicates()
    historical = pd.read_csv(EARLIER / "results/contacts.csv.gz")
    historical = historical[(historical.population == "retained") & (historical.tolerance_base30 == 10)]
    historical = historical.rename(columns={"fixture": "video", "matched": "historical_matched"})
    joined = (contacts.merge(new, on=["video", "source_frame"], validate="many_to_one")
              .merge(old, on=["video", "source_frame"], validate="many_to_one", suffixes=("_new", "_old"))
              .merge(historical[["video", "rally_id", "label_index", "historical_matched"]],
                     on=["video", "rally_id", "label_index"], validate="one_to_one"))
    assert len(joined) == len(contacts)
    return joined


def label_transitions(joined: pd.DataFrame) -> list[dict]:
    rows = []
    for (old_present, new_present), group in joined.groupby(["court_present_old", "court_present_new"]):
        rows.append({"old_courts": "accepted" if old_present else "rejected",
                     "new_courts": "accepted" if new_present else "rejected",
                     "labels": len(group), "videos": group.video.nunique(),
                     "historical_matched": int(group.historical_matched.sum()), "new_matched": int(group.matched.sum()),
                     "new_rejection": {reason: int(count) for reason, count in group.court_rejection.value_counts().items()},
                     "largest_videos": {int(video): int(count) for video, count in group.video.value_counts().head(5).items()}})
    return rows


def rally_court_groups(joined: pd.DataFrame) -> pd.DataFrame:
    rallies = joined.groupby(["video", "rally_id"]).agg(old_rejected=("court_present_old", lambda present: not present.all()),
                                                       new_rejected=("court_present_new", lambda present: not present.all()))
    rallies["group"] = np.select(
        [~rallies.old_rejected & ~rallies.new_rejected, rallies.old_rejected & ~rallies.new_rejected,
         rallies.old_rejected & rallies.new_rejected],
        GROUP_ORDER[:3], default=GROUP_ORDER[3])
    return rallies


def changes_against_fresh_refit() -> pd.DataFrame:
    path = REPOSITORY / "experiments/annotator/good_court_refit/evidence/saved-stream-analysis.json.gz"
    with gzip.open(path, "rt") as source:
        per_video = json.load(source)["old_to_new"]["fresh_old_courts"]["per_video"]
    rows = [(int(video), rally_id, change) for video, record in per_video.items()
            for change in ("gained", "lost") for rally_id in record[change]]
    return pd.DataFrame(rows, columns=["video", "rally_id", "change"])


def changes_against_historical() -> pd.DataFrame:
    columns = ["video", "rally_id", "fully_correct"]
    old = pd.read_csv(EARLIER / "results/rallies.csv.gz")
    old = old[(old.population == "retained") & (old.tolerance_base30 == 10) & (old.fixture != 15)].rename(columns={"fixture": "video"})
    new = pd.read_csv(ROOT / "results/rallies.csv.gz", dtype={"video": str})
    new = new[(new.split == "test") & (new.tolerance_base30 == 10)]
    new = new.assign(video=new.video.astype(int))
    paired = new[columns].merge(old[columns], on=["video", "rally_id"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(paired) == 3327
    change = np.select([paired.fully_correct_new & ~paired.fully_correct_old,
                        ~paired.fully_correct_new & paired.fully_correct_old], ["gained", "lost"], default="")
    return paired.assign(change=change)[change != ""][["video", "rally_id", "change"]]


def tabulate(rallies: pd.DataFrame, changes: pd.DataFrame) -> list[dict]:
    joined = changes.merge(rallies.reset_index(), on=["video", "rally_id"], validate="one_to_one")
    assert len(joined) == len(changes)
    rows = []
    for group in GROUP_ORDER:
        in_group = joined[joined.group == group]
        gained, lost = int(sum(in_group.change == "gained")), int(sum(in_group.change == "lost"))
        rows.append({"group": group, "rallies": int(sum(rallies.group == group)), "gained": gained, "lost": lost,
                     "net": gained - lost})
    return rows


def run() -> None:
    joined = labelled_court_states()
    rallies = rally_court_groups(joined)
    summary = {"labels": label_transitions(joined),
               "fresh_old_courts": tabulate(rallies, changes_against_fresh_refit()),
               "historical": tabulate(rallies, changes_against_historical())}
    with gzip.open(ROOT / "results/court_change_groups.json.gz", "wt") as target:
        json.dump(summary, target, indent=1)
    for name, rows in summary.items():
        print(name)
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run()
