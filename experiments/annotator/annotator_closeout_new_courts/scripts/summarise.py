"""Collect every count the closeout report quotes from the row-level tables.

Main population: 46 ShuttleSet22 videos, cleaned labels. ``without_53`` repeats
the counts without video 53 as a sensitivity check only.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.training.sequences import match_contacts
from dataset_builder.vision import load_json_gz

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[2]
STREAMS = ROOT / "raw/run/output/base/eval/test"
ERROR_NAMES = {"missing": "Missed contacts", "extra": "Extra contacts",
               "wrong_player": "Wrong player", "boundary_error": "Clip cuts off rally"}
RALLY_LENGTH_BINS = ((1, 5, "1-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 10**6, "more than 20"))


def read(name: str) -> pd.DataFrame:
    table = pd.read_csv(ROOT / f"results/{name}.csv.gz", dtype={"video": str})
    return table


def test_rows(table: pd.DataFrame, tolerance: int = 10) -> pd.DataFrame:
    selected = table[(table.split == "test") & (table.tolerance_base30 == tolerance)]
    return selected.assign(video=selected.video.astype(int))


def headline(contacts: pd.DataFrame, rallies: pd.DataFrame, proposals: pd.DataFrame,
             predictions: pd.DataFrame) -> dict[str, float]:
    serves = contacts[contacts.position == "serve"]
    selected = proposals[proposals.selected]
    correct = int(sum(selected.outcome == "correct"))
    judged = int(sum(selected.outcome != "unknown"))
    precision = int(contacts.matched.sum()) / len(predictions)
    recall = int(contacts.matched.sum()) / len(contacts)
    player_precision = int(contacts.player_correct.sum()) / len(predictions)
    player_recall = int(contacts.player_correct.sum()) / len(contacts)
    return {
        "videos": int(contacts.video.nunique()), "labelled_rallies": len(rallies), "labelled_contacts": len(contacts),
        "exact_sequence": int(rallies.exact_sequence.sum()), "fully_correct": int(rallies.fully_correct.sum()),
        "timing_matched": int(contacts.matched.sum()), "timing_player": int(contacts.player_correct.sum()),
        "serve_timing": int(serves.matched.sum()), "serve_timing_player": int(serves.player_correct.sum()),
        "contained": int(rallies.contained.sum()), "predictions": len(predictions),
        "contact_precision": precision, "contact_recall": recall,
        "contact_f1": 2 * precision * recall / (precision + recall),
        "player_f1": 2 * player_precision * player_recall / (player_precision + player_recall),
        "proposals": len(proposals), "selected": len(selected), "selected_correct": correct,
        "selected_wrong": int(sum(selected.outcome == "wrong")),
        "selected_unknown": int(sum(selected.outcome == "unknown")),
        "selected_precision": correct / judged, "selected_recall": correct / len(rallies),
    }


def coverage(rallies: pd.DataFrame, proposals: pd.DataFrame) -> dict[str, object]:
    state = np.select([rallies.fully_correct, rallies.contained, rallies.overlapping_proposals > 0],
                      ["Fully correct clip", "Whole rally fits in a clip; errors remain",
                       "Only partial rally coverage"], default="No clip reaches a labelled contact")
    one_rally = proposals[proposals.overlapping_rallies == 1]
    contained_single = one_rally[~one_rally.boundary_error].groupby("video").rally_id.unique()
    single_count = sum(len(ids) for ids in contained_single)
    outside = proposals[~proposals.selected]
    return {
        "best_output": dict(Counter(state)),
        "contained_in_single_rally_clip": single_count,
        "proposals_no_rally": int(sum(proposals.overlapping_rallies == 0)),
        "proposals_one_rally": len(one_rally),
        "proposals_several_rallies": int(sum(proposals.overlapping_rallies > 1)),
        "timing_wholly_missed": int(sum(~rallies.timing_reached)),
        "outside_queue": dict(Counter(outside.outcome)),
    }


def selected_errors(proposals: pd.DataFrame, contacts: pd.DataFrame) -> dict[str, object]:
    wrong = proposals[proposals.selected & (proposals.outcome == "wrong")]
    combinations = Counter(
        "Several rallies in one clip" if row.overlapping_rallies > 1 else
        " + ".join(name for key, name in ERROR_NAMES.items() if getattr(row, key) > 0)
        for row in wrong.itertuples(index=False)
    )
    flags = {name: int(sum(wrong[key].fillna(0).astype(float) > 0)) for key, name in ERROR_NAMES.items()}
    # Event positions inside each wrong one-rally clip, matched within that clip only.
    extra_positions, missed_positions = Counter(), Counter()
    for (video, span_id), clip in wrong[wrong.overlapping_rallies == 1].groupby(["video", "span_id"]):
        stream = load_json_gz(STREAMS / str(video) / "stream.json.gz")
        sequence = next(row for row in stream["sequences"] if row["span_id"] == span_id)
        frames = [event[0] for event in sequence["events"]]
        truth = contacts[(contacts.video == video) & (contacts.rally_id == clip.rally_id.iloc[0])]
        truth = truth.sort_values("label_index")
        matches = match_contacts(truth.source_frame.tolist(), frames, round(10 * truth.fps.iloc[0] / 30))
        matched_events = {prediction for _label, prediction, _offset in matches}
        matched_labels = {label for label, _prediction, _offset in matches}
        first, last = truth.source_frame.min(), truth.source_frame.max()
        for index, frame in enumerate(frames):
            if index in matched_events:
                continue
            extra_positions["before first label" if frame < first else
                            "after last label" if frame > last else "between labels"] += 1
        for index, position in enumerate(truth.position):
            if index not in matched_labels:
                missed_positions[position] += 1
    return {"clips": len(wrong), "flags": flags, "combinations": combinations.most_common(),
            "player_only": combinations.get("Wrong player", 0),
            "extra_event_positions": dict(extra_positions), "missed_event_positions": dict(missed_positions)}


def input_states(contacts: pd.DataFrame, context: pd.DataFrame) -> dict[str, object]:
    joined = contacts.merge(context, on=["video", "source_frame"], validate="many_to_one")
    result = {}
    for name, group in (("all", joined), ("without_53", joined[joined.video != 53])):
        rates = group.groupby("input_state").matched.agg(labelled="size", matched="sum")
        reasons = group[~group.matched].groupby("court_rejection").size()
        accepted = group[group.court_present]
        position = accepted.groupby("position").matched.agg(labelled="size", matched="sum")
        result[name] = {
            "states": {state: {"labelled": int(row.labelled), "matched": int(row.matched)}
                       for state, row in rates.iterrows()},
            "missed_by_court_reason": {reason: int(count) for reason, count in reasons.items()},
            "accepted_positions": {kind: {"labelled": int(row.labelled), "missed": int(row.labelled - row.matched)}
                                   for kind, row in position.iterrows()},
            "missed_without_nearby_feature_row": int(sum(~group.matched & ~group.court_present
                                                         & (group.nearby_feature_rows == 0))),
        }
    return result


def timing(contacts: pd.DataFrame) -> dict[str, float]:
    offsets = contacts[contacts.matched].offset_base30
    return {"matched": len(offsets), "exact": int(sum(offsets == 0)), "within_2": int(sum(offsets.abs() <= 2)),
            "within_5": int(sum(offsets.abs() <= 5)), "median": float(offsets.median()),
            "mean": float(offsets.mean())}


def players(contacts: pd.DataFrame) -> dict[str, object]:
    matched = contacts[contacts.matched]
    known = matched[matched.target_side.notna()]
    confusion = known.groupby(["target_side", known.predicted_side.fillna("None")]).size()
    return {"known_side_matches": len(known), "correct": int(known.player_correct.sum()),
            "confusion": {f"{target}->{predicted}": int(count) for (target, predicted), count in confusion.items()}}


def per_video(contacts: pd.DataFrame, rallies: pd.DataFrame, proposals: pd.DataFrame,
              context: pd.DataFrame) -> pd.DataFrame:
    joined = contacts.merge(context, on=["video", "source_frame"], validate="many_to_one")
    table = rallies.groupby("video").agg(labelled_rallies=("fully_correct", "size"),
                                         correct_rallies=("fully_correct", "sum"))
    table = table.join(contacts.groupby("video").agg(labelled_contacts=("matched", "size"),
                                                     matched_contacts=("matched", "sum"),
                                                     player_correct=("player_correct", "sum")))
    missed = joined[~joined.matched]
    table["missed_court_rejected"] = missed[~missed.court_present].groupby("video").size()
    table["labels_court_rejected"] = joined[~joined.court_present].groupby("video").size()
    table["predictions"] = read("predictions").pipe(test_rows).groupby("video").size()
    chosen = proposals[proposals.selected]
    for outcome in ("correct", "wrong", "unknown"):
        table[f"selected_{outcome}"] = chosen[chosen.outcome == outcome].groupby("video").size()
    table = table.fillna(0).astype(int)
    table["rally_percent"] = 100 * table.correct_rallies / table.labelled_rallies
    table["contact_percent"] = 100 * table.matched_contacts / table.labelled_contacts
    return table


def rally_lengths(rallies: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for low, high, label in RALLY_LENGTH_BINS:
        group = rallies[(rallies.labelled_contacts >= low) & (rallies.labelled_contacts <= high)]
        rows.append({"length": label, "rallies": len(group), "fully_correct": int(group.fully_correct.sum())})
    return rows


def development(contacts: pd.DataFrame, rallies: pd.DataFrame) -> dict[str, object]:
    result = {}
    for split in ("validation", "development_heldout"):
        for tolerance in (10, 5):
            own_contacts = contacts[(contacts.split == split) & (contacts.tolerance_base30 == tolerance)]
            own_rallies = rallies[(rallies.split == split) & (rallies.tolerance_base30 == tolerance)]
            serves = own_contacts[own_contacts.position == "serve"]
            result[f"{split}_{tolerance}"] = {
                "videos": int(own_contacts.video.nunique()), "rallies": len(own_rallies),
                "contacts": len(own_contacts), "fully_correct": int(own_rallies.fully_correct.sum()),
                "exact_sequence": int(own_rallies.exact_sequence.sum()),
                "contained": int(own_rallies.contained.sum()),
                "timing": int(own_contacts.matched.sum()), "timing_player": int(own_contacts.player_correct.sum()),
                "serve_timing": int(serves.matched.sum()), "serve_timing_player": int(serves.player_correct.sum()),
            }
    return result


def court_comparison() -> dict[str, object]:
    analysis = load_json_gz(REPOSITORY / "experiments/annotator/good_court_refit/evidence/saved-stream-analysis.json.gz")
    result = {}
    for name, paired in analysis["old_to_new"].items():
        rows = {video: {"old": row["old_correct"], "new": row["new_correct"],
                        "gained": len(row["gained"]), "lost": len(row["lost"])}
                for video, row in paired["per_video"].items()}
        result[name] = {"gained": paired["gained"], "lost": paired["lost"],
                        "old": sum(row["old"] for row in rows.values()),
                        "new": sum(row["new"] for row in rows.values()), "per_video": rows}
    return result


def run() -> None:
    contacts, rallies = read("contacts"), read("rallies")
    proposals, predictions = read("proposals"), read("predictions")
    context = pd.read_csv(ROOT / "results/contexts.csv.gz")
    main = {name: test_rows(table) for name, table in
            (("contacts", contacts), ("rallies", rallies), ("proposals", proposals), ("predictions", predictions))}
    narrow = {name: test_rows(table, 5) for name, table in
              (("contacts", contacts), ("rallies", rallies), ("proposals", proposals), ("predictions", predictions))}
    without_53 = {name: table[table.video != 53] for name, table in main.items()}
    summary = {
        "headline": {"all_10": headline(**main), "without_53_10": headline(**without_53), "all_5": headline(**narrow)},
        "coverage": {"all": coverage(main["rallies"], main["proposals"]),
                     "without_53": coverage(without_53["rallies"], without_53["proposals"])},
        "selected_errors": selected_errors(main["proposals"], main["contacts"]),
        "input_states": input_states(main["contacts"], context),
        "timing": timing(main["contacts"]),
        "players": players(main["contacts"]),
        "rally_lengths": rally_lengths(main["rallies"]),
        "development": development(contacts, rallies),
        "court_comparison": court_comparison(),
    }
    videos = per_video(main["contacts"], main["rallies"], main["proposals"], context)
    videos.to_csv(ROOT / "results/per_video.csv.gz")
    summary["per_video"] = {
        "median_rally_percent": float(videos.rally_percent.median()),
        "median_contact_percent": float(videos.contact_percent.median()),
        "best": videos.sort_values("rally_percent", ascending=False).head(4).reset_index().to_dict("records"),
        "worst": videos.sort_values("rally_percent").head(8).reset_index().to_dict("records"),
    }
    with gzip.open(ROOT / "results/summary.json.gz", "wt") as destination:
        json.dump(summary, destination, indent=1, default=int)
    shown = {key: value for key, value in summary.items() if key not in ("per_video", "court_comparison")}
    print(json.dumps(shown, default=int))


if __name__ == "__main__":
    run()
