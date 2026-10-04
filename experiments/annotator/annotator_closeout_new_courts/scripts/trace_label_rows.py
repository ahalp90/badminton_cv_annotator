"""Trace the cleaned test labels and the footage-check cases back to the official ShuttleSet22 rows.

The cleaned labels copy ``frame_num`` unchanged and take the side from the annotated feet:
the hitter is ``Top`` when their annotated y is above the opponent's. The official coordinates
are image pixels on a 1280×720 grid; the source videos are 1920×1080.

This script checks that cleaning kept every usable official row unchanged, then writes each
footage case with the official rows that decide it: the labelled hit for timing and side
cases, and the rally's final row for the extra-final-hit cases.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[2]
OFFICIAL = REPOSITORY / "data/shuttleset22/set"
CLEANED = ROOT / "raw/run/output/shared/labels/test"
ENDING_NAMES = {
    "出界": "shot out", "掛網": "shot into net", "未過網": "shot short of net", "對手落地致勝": "opponent's winner landed",
    "落點判斷失誤": "misjudged landing", "犯規": "fault",
}
ROW_COLUMNS = ["rally", "ball_round", "frame_num", "player", "type", "roundscore_A", "roundscore_B", "lose_reason",
               "getpoint_player", "flaw", "player_location_y", "opponent_location_y"]


def official_rows(video: str) -> pd.DataFrame:
    matches = pd.read_csv(OFFICIAL / "match.csv.gz").set_index("id")
    folder = OFFICIAL / matches.loc[int(video), "video"]
    tables = [pd.read_csv(path).assign(set_number=int(path.name[3])) for path in sorted(folder.glob("set*.csv.gz"))]
    rows = pd.concat(tables, ignore_index=True)
    rows["rally_id"] = "set" + rows.set_number.astype(str) + ":" + rows.rally.astype(str)
    both_feet_marked = rows.player_location_y.notna() & rows.opponent_location_y.notna()
    hitter_higher = rows.player_location_y < rows.opponent_location_y
    rows["official_side"] = hitter_higher.map({True: "Top", False: "Bot"}).where(both_feet_marked)
    return rows


def check_cleaning(videos: list[str]) -> pd.DataFrame:
    """Every cleaned label must be an official row with the same frame and side."""
    rows = []
    for video in videos:
        cleaned = pd.read_csv(CLEANED / f"{video}.csv")
        official = official_rows(video)
        kept_rallies = official[official.rally_id.isin(cleaned.rally_id)]
        joined = cleaned.merge(kept_rallies, left_on=["rally_id", "frame"], right_on=["rally_id", "frame_num"],
                               how="left", validate="one_to_one")
        rows.append({"video": video, "cleaned": len(cleaned), "unmatched": int(joined.frame_num.isna().sum()),
                     "side_differs": int((joined.side.fillna("none") != joined.official_side.fillna("none")).sum()),
                     "official_rows_dropped_in_kept_rallies": len(kept_rallies) - len(cleaned),
                     "official_rallies": official.rally_id.nunique(), "kept_rallies": cleaned.rally_id.nunique()})
    return pd.DataFrame(rows)


def deciding_row(case: pd.Series, official: pd.DataFrame) -> pd.Series:
    rally = official[official.rally_id == case.rally_id].sort_values("ball_round")
    if case.group == "tail":
        return rally.iloc[-1]
    return rally[rally.frame_num == case.query_frame].iloc[0]


def trace_cases() -> pd.DataFrame:
    cases = pd.read_csv(ROOT / "results/label_check_cases.csv.gz", dtype={"video": str})
    sample = pd.read_csv(ROOT / "results/miss_sample.csv.gz", dtype={"video": str})
    sample = sample.rename(columns={"sample_id": "case_id", "source_frame": "query_frame", "target_side": "label_side"})
    sample["group"] = "miss sample"
    cases = pd.concat([cases, sample[["case_id", "video", "rally_id", "query_frame", "label_side", "group"]]],
                      ignore_index=True)
    rows = []
    for case in cases.itertuples(index=False):
        case = pd.Series(case._asdict())
        row = deciding_row(case, official_rows(case.video))
        traced = {name: row[name] for name in ROW_COLUMNS} | {"official_side": row.official_side}
        traced["ending"] = ENDING_NAMES.get(row.lose_reason) if isinstance(row.lose_reason, str) else None
        rows.append(case.to_dict() | traced)
    return pd.DataFrame(rows)


def run() -> None:
    videos = sorted(path.stem for path in CLEANED.glob("*.csv"))
    cleaning = check_cleaning(videos)
    print(cleaning[["cleaned", "unmatched", "side_differs", "official_rows_dropped_in_kept_rallies", "official_rallies",
                    "kept_rallies"]].sum().to_string())
    traced = trace_cases()
    traced.to_csv(ROOT / "results/label_trace.csv.gz", index=False)
    print(traced.drop(columns=["player_location_y", "opponent_location_y"]).to_string(index=False))


if __name__ == "__main__":
    run()
