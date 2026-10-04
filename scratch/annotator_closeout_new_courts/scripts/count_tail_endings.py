"""Count the rallies that fail on an extra hit after the last label, and how the official rows say they ended.

The pool is the footage-check pool from ``sample_label_checks.py``. A pool rally fails on that extra hit
alone when one proposal span covers only that rally and holds exactly one extra prediction. Spans that
cover two or more rallies carry no rally ID in ``proposals.csv.gz``, so their extras cannot be counted
per rally.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sample_label_checks import load, tail_pool
from trace_label_rows import ENDING_NAMES, official_rows

ROOT = Path(__file__).resolve().parents[1]
ENDING_ORDER = ["shot into net", "shot out", "opponent's winner landed", "misjudged landing", "shot short of net",
                "fault", "not recorded"]


def official_endings(videos: list[str]) -> pd.DataFrame:
    """One row per official rally: the ending recorded on its final row."""
    endings = []
    for video in videos:
        rows = official_rows(video).sort_values(["rally_id", "ball_round"])
        final_rows = rows.groupby("rally_id").tail(1)
        endings.append(pd.DataFrame({"video": video, "rally_id": final_rows.rally_id,
                                     "ending": final_rows.lose_reason.map(ENDING_NAMES).fillna("not recorded")}))
    return pd.concat(endings, ignore_index=True)


def tail_rallies() -> tuple[pd.DataFrame, pd.DataFrame]:
    contacts, predictions, rallies = load()
    proposals = pd.read_csv(ROOT / "results/proposals.csv.gz", dtype={"video": str})
    proposals = proposals[(proposals.split == "test") & (proposals.tolerance_base30 == 10)]
    own_spans = proposals.dropna(subset=["rally_id"]).groupby(["video", "rally_id"]).agg(
        own_spans=("span_id", "size"), extras=("extra", "sum"))
    last_sides = contacts.sort_values("source_frame").groupby(["video", "rally_id"]).target_side.last()
    pool = tail_pool(contacts, predictions, rallies).join(own_spans, on=["video", "rally_id"])
    pool["last_label_side"] = [last_sides[(row.video, row.rally_id)] for row in pool.itertuples()]
    pool["sole_error"] = (pool.own_spans == 1) & (pool.extras == 1)
    endings = official_endings(sorted(rallies.video.unique()))
    tested = rallies[["video", "rally_id"]].merge(endings, on=["video", "rally_id"], how="left", validate="one_to_one")
    pool = pool.merge(endings, on=["video", "rally_id"], how="left", validate="one_to_one")
    return pool.drop(columns="label_side"), tested


def ending_table(pool: pd.DataFrame, tested: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame({"pool": pool.ending.value_counts(), "sole_error": pool[pool.sole_error].ending.value_counts(),
                          "all_rallies": tested.ending.value_counts()}).reindex(ENDING_ORDER).fillna(0).astype(int)
    table.loc["total"] = table.sum()
    table["sole_error_share"] = (100 * table.sole_error / table.all_rallies).round(1)
    return table


def run() -> None:
    pool, tested = tail_rallies()
    pool.to_csv(ROOT / "results/tail_rallies.csv.gz", index=False)
    other_player = pool.predicted_side != pool.last_label_side
    print("pool:", len(pool), "| sole error:", int(pool.sole_error.sum()),
          "| two or more extras in own span:", int((pool.extras > 1).sum()),
          "| in a span covering several rallies:", int(pool.own_spans.isna().sum()),
          "| extra by the other player:", int(other_player.sum()))
    print(ending_table(pool, tested).to_string())


if __name__ == "__main__":
    run()
