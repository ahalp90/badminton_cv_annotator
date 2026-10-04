"""Draw the closeout figures from the saved tables, in the earlier report's styles."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#999999"
DARK_GREY = "#777777"
PURPLE = "#7651A8"
LIGHT_ORANGE = "#E69F00"
SUBTITLE = "46 ShuttleSet22 videos · new-court model · cleaned labels · ±10 frames at 30 fps"
POPULATIONS = ["All 46 videos", "Without video 53\nsensitivity"]
ERROR_NAMES = {"missing": "Missed contacts", "extra": "Extra contacts",
               "wrong_player": "Wrong player", "boundary_error": "Clip cuts off rally"}
STATES = ("No court found", "Court found; player check failed",
          "Court accepted; a player pick missing", "Court accepted; both players picked")
STATE_COLOURS = (ORANGE, LIGHT_ORANGE, PURPLE, BLUE)
# The two earlier problem videos plus the lowest new one, labelled clear of the cluster.
LABEL_OFFSETS = {53: (-150, -3), 17: (-150, -12), 42: (-60, -70)}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def save(figure: plt.Figure, name: str) -> None:
    figure.savefig(FIGURES / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def test_rows(name: str, tolerance: int | None = 10) -> pd.DataFrame:
    table = pd.read_csv(ROOT / f"results/{name}.csv.gz", dtype={"video": str})
    table = table[table.split == "test"]
    if tolerance is not None:
        table = table[table.tolerance_base30 == tolerance]
    return table.assign(video=table.video.astype(int))


def with_state(contacts: pd.DataFrame) -> pd.DataFrame:
    context = pd.read_csv(ROOT / "results/contexts.csv.gz")
    joined = contacts.merge(context, on=["video", "source_frame"], validate="many_to_one")
    rejected_state = joined.court_rejection.where(~joined.court_present, joined.input_state)
    return joined.assign(state=rejected_state)


def plot_video_variation(per_video: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.scatter(per_video.contact_percent, per_video.rally_percent, color=BLUE, alpha=0.75, s=50)
    for video, offset in LABEL_OFFSETS.items():
        row = per_video.loc[video]
        axis.annotate(f"Video {video}\n{row.correct_rallies:.0f}/{row.labelled_rallies:.0f} rallies",
                      (row.contact_percent, row.rally_percent), xytext=offset, textcoords="offset points",
                      fontsize=11, arrowprops={"arrowstyle": "-", "color": GREY})
    axis.set(xlim=(0, 100), ylim=(-4, 100), xlabel="Labelled contacts with a timing match (%)",
             ylabel="Labelled rallies with a fully correct clip (%)")
    axis.set_title("Contacts are now high everywhere; whole rallies still vary widely", loc="left",
                   weight="bold", pad=38)
    figure.text(0.125, 0.91, "One point per video · " + SUBTITLE, fontsize=10)
    axis.grid(alpha=0.15)
    save(figure, "video_variation")


def plot_video_pages(contacts: pd.DataFrame, per_video: pd.DataFrame) -> None:
    order = per_video.sort_values(["rally_percent", "correct_rallies"], ascending=False).index.tolist()
    pages = (order[:23], order[23:])
    for page, videos in enumerate(pages, start=1):
        figure, axis = plt.subplots(figsize=(10, 9.2))
        positions = np.arange(len(videos))[::-1]
        for position, video in zip(positions, videos, strict=True):
            group = contacts[contacts.video == video]
            counts = [int(group.player_correct.sum()), int(sum(group.matched & ~group.player_correct)),
                      int(sum(~group.matched))]
            shares = 100 * np.array(counts) / len(group)
            left = 0.0
            for share, count, colour in zip(shares, counts, (BLUE, ORANGE, DARK_GREY), strict=True):
                axis.barh(position, share, left=left, color=colour, height=0.75)
                if share > 8:
                    axis.text(left + share / 2, position, f"{count:,}", ha="center", va="center",
                              color="white", fontsize=9)
                left += share
            row = per_video.loc[video]
            axis.text(101.5, position, f"{row.correct_rallies:.0f}/{row.labelled_rallies:.0f} rallies "
                      f"({row.rally_percent:.1f}%)", va="center", fontsize=10)
        axis.set_yticks(positions, [f"Video {video}" for video in videos])
        axis.set(xlim=(0, 100), xlabel="Share of labelled contacts (%) · each bar sums to 100%", ylabel="Video ID")
        axis.grid(axis="x", alpha=0.15)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for colour in (BLUE, ORANGE, DARK_GREY)]
        axis.legend(handles, ["Matched + correct player", "Matched, player not confirmed", "Missed labelled contact"],
                    ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False)
        axis.set_title(f"Per-video contact outcomes, new-court model ({page}/2)", loc="left", weight="bold", pad=52)
        figure.text(0.125, 0.915, SUBTITLE + " · sorted by fully correct rally rate\n"
                    "Matched + correct player means timing and attribution are both correct.", fontsize=10)
        save(figure, f"video_outcome_breakdown_{page}")


def plot_rally_coverage(rallies: pd.DataFrame) -> None:
    order = ["Fully correct clip", "Whole rally fits in a clip; errors remain", "Only partial rally coverage",
             "No clip reaches a labelled contact"]
    state = np.select([rallies.fully_correct, rallies.contained, rallies.overlapping_proposals > 0], order[:3],
                      default=order[3])
    counts = pd.Series(Counter(state)).reindex(order)
    figure, axis = plt.subplots(figsize=(10, 4.6))
    bars = axis.barh(range(4), counts, color=[BLUE, PURPLE, ORANGE, DARK_GREY])
    axis.invert_yaxis()
    axis.set(yticks=range(4), yticklabels=order, xlabel=f"Number of cleaned labelled rallies ({len(rallies):,} total)",
             ylabel="Best available clip coverage")
    axis.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=5, fontsize=11)
    axis.set_xlim(0, counts.max() * 1.14)
    axis.set_title("Some rallies still lack a complete clip before contact details can be judged", loc="left",
                   weight="bold", pad=40)
    figure.text(0.125, 0.92, SUBTITLE + " · before selection\n"
                "Each labelled rally appears once. These groups describe output; they are not independent causes.",
                fontsize=10)
    save(figure, "rally_coverage")


def plot_selection(proposals: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.4))
    left = np.zeros(2)
    for outcome, colour in (("correct", BLUE), ("wrong", ORANGE), ("unknown", GREY)):
        counts = np.array([sum((proposals.selected == selected) & (proposals.outcome == outcome))
                           for selected in (True, False)])
        axis.barh([1, 0], counts, left=left, color=colour, label=outcome.capitalize())
        for position, count, start in zip([1, 0], counts, left, strict=True):
            if count > 150:
                axis.text(start + count / 2, position, f"{count:,}", ha="center", va="center", color="white")
        left += counts
    kept = proposals[proposals.selected]
    outcomes = Counter(kept.outcome)
    rejected = int(sum(~proposals.selected))
    axis.set_yticks([1, 0], [f"Selected\n{len(kept):,} clips", f"Not selected\n{rejected:,} clips"])
    axis.set_xlabel("Number of proposed rally clips")
    axis.set_ylabel("Historical review cutoff")
    left_behind = int(sum(~proposals.selected & (proposals.outcome == "correct")))
    axis.set_title(f"Selection keeps {outcomes['correct']:,} correct clips and leaves {left_behind:,} behind",
                   loc="left", weight="bold", pad=42)
    figure.text(0.125, 0.91, SUBTITLE + " · cutoff 0.757 carried over unchanged", fontsize=10)
    axis.text(len(kept) + 20, 1, f"{outcomes['correct']} correct · {outcomes['wrong']} wrong · "
              f"{outcomes['unknown']} unknown", va="center", fontsize=11)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    axis.set_xlim(0, left.max() * 1.03)
    save(figure, "selection")


def plot_selected_errors(proposals: pd.DataFrame) -> None:
    wrong = proposals[proposals.selected & (proposals.outcome == "wrong")]
    counts = Counter(" + ".join(name for key, name in ERROR_NAMES.items() if getattr(row, key) > 0)
                     for row in wrong.itertuples(index=False))
    ordered = counts.most_common()
    labels = [label.replace(" + ", "\n+ ") for label, _ in ordered]
    values = [value for _, value in ordered]
    figure, axis = plt.subplots(figsize=(11.5, 7.5))
    positions = np.arange(len(values))
    axis.barh(positions, values, color=BLUE)
    axis.set_yticks(positions, labels, fontsize=10)
    axis.invert_yaxis()
    for position, value in zip(positions, values, strict=True):
        axis.text(value + 0.5, position, str(value), va="center")
    axis.set_xlim(0, max(values) + 5)
    axis.set_xlabel(f"Number of known wrong selected clips ({len(wrong)} in total)")
    axis.set_ylabel("Errors occurring together in one clip")
    axis.set_title("Selected clips usually fail because contacts are extra or missing", loc="left",
                   weight="bold", pad=62)
    figure.text(0.125, 0.915, SUBTITLE + "\nEach clip appears once. "
                f"Player-only errors: {counts.get('Wrong player', 0)}.", fontsize=11)
    save(figure, "selected_errors")


def plot_timing(contacts: pd.DataFrame) -> None:
    matched = contacts[contacts.matched]
    counts = matched.groupby(matched.offset_base30.round().astype(int)).size().reindex(range(-10, 11), fill_value=0)
    figure, axis = plt.subplots(figsize=(10, 4.7))
    axis.bar(counts.index, counts.values, width=0.82, color=BLUE)
    axis.set(xlabel="Prediction frame minus label frame (30 fps; negative means early)",
             ylabel="Number of matched labelled contacts", xticks=range(-10, 11, 2))
    axis.set_title("Most timing matches are close to the labelled frame", loc="left", weight="bold", pad=42)
    figure.text(0.125, 0.92, SUBTITLE + f"\nOnly the {len(matched):,} timing matches appear; "
                "missing contacts are outside this plot.", fontsize=10)
    save(figure, "timing_offsets")


def plot_contact_position(contacts_by_tolerance: dict[int, pd.DataFrame]) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.3))
    positions = np.arange(3)
    kinds = ["serve", "middle", "last"]
    for shift, tolerance, colour in ((-0.2, 10, BLUE), (0.2, 5, ORANGE)):
        data = contacts_by_tolerance[tolerance]
        groups = data.groupby("position").matched.agg(["size", "sum"]).loc[kinds]
        rates = 100 * (groups["size"] - groups["sum"]) / groups["size"]
        axis.bar(positions + shift, rates, width=0.36, color=colour, label=f"±{tolerance} frames")
        for position, rate in zip(positions + shift, rates, strict=True):
            axis.text(position, rate + 0.7, f"{rate:.1f}%", ha="center", fontsize=11)
    sizes = contacts_by_tolerance[10].groupby("position").size().loc[kinds]
    axis.set_xticks(positions, [f"{label}\n{count:,} labelled contacts" for label, count in
                                zip(["Serve", "Middle contact", "Last contact"], sizes, strict=True)])
    axis.set_ylim(0, 36)
    axis.set_xlabel("Position within the labelled rally")
    axis.set_ylabel("Labelled contacts without a timing match (%)")
    axis.set_title("Serves and final contacts are still missed more often", loc="left", weight="bold", pad=43)
    figure.text(0.125, 0.91, SUBTITLE.replace(" · ±10 frames at 30 fps", "") +
                "\nOne-to-one matching across each full video. Frame clock: 30 fps.", fontsize=11)
    axis.legend(frameon=False)
    single = int(sum(contacts_by_tolerance[10].labelled_contacts == 1))
    figure.text(0.125, -0.07, f"The {single} one-contact rallies count as serves only. "
                "A timing match may still name the wrong player.", fontsize=10)
    save(figure, "contact_position")


def plot_input_states(joined: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.9))
    positions = [1, 0]
    left = np.zeros(2)
    totals = np.array([sum(~joined.matched), sum(joined.matched)])
    for state, colour in zip(STATES, STATE_COLOURS, strict=True):
        counts = np.array([sum((joined.state == state) & (joined.matched == matched)) for matched in (False, True)])
        width = 100 * counts / totals
        axis.barh(positions, width, left=left, height=0.55, color=colour, label=state)
        for position, start, percent, count in zip(positions, left, width, counts, strict=True):
            if percent > 8:
                axis.text(start + percent / 2, position, f"{percent:.1f}%\n{count:,} contacts",
                          color="white", ha="center", va="center", fontsize=12)
        left += width
    axis.set_yticks(positions, [f"Missed\n{totals[0]:,} contacts", f"Matched\n{totals[1]:,} contacts"])
    axis.set(xlim=(0, 100), xlabel="Share of labelled contacts in each timing outcome (%)",
             ylabel="Final timing result")
    axis.set_title("Over half of the remaining misses still fall in court-rejected scenes", loc="left",
                   weight="bold", pad=48)
    figure.text(0.125, 0.92, SUBTITLE + "\nInputs measured at each labelled frame. Orange shades: the court "
                "stage rejected the scene.", fontsize=10)
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.27), frameon=False, fontsize=10, ncol=2)
    save(figure, "upstream_context")


def plot_misses_by_state(joined: pd.DataFrame) -> None:
    missed = joined[~joined.matched]
    values = [int(sum(missed.state == state)) for state in STATES]
    total = sum(values)
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    labels = ["No court\nfound", "Court found;\nplayer check failed", "Court accepted;\nplayer missing",
              "Court accepted;\nboth players available"]
    bars = axis.bar(labels, values, color=STATE_COLOURS)
    axis.set_title(f"Where the {total:,} missed contacts occur\n46 videos, cleaned labels, ±10 frames")
    axis.set_ylabel("Missed labelled contacts")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 25,
                  f"{value:,}\n({100 * value / total:.1f}%)", ha="center", va="bottom", fontsize=10)
    axis.set_ylim(0, max(values) * 1.2)
    save(figure, "misses_by_input_state")


def population_values(summary: dict, keys: tuple[str, ...]) -> list[list[float]]:
    rows = [summary["headline"]["all_10"], summary["headline"]["without_53_10"]]
    return [[row[key] for row in rows] for key in keys]


def grouped_bars(summary: dict, keys: tuple[str, str], denominator: str, labels: tuple[str, str],
                 title: str, ylabel: str, ylim: int, name: str) -> None:
    first, second = population_values(summary, keys)
    totals = population_values(summary, (denominator,))[0]
    positions = np.arange(len(POPULATIONS))
    figure, axis = plt.subplots(figsize=(8, 5.6))
    for offset, values, label in ((-0.18, first, labels[0]), (0.18, second, labels[1])):
        shares = 100 * np.array(values) / np.array(totals)
        bars = axis.bar(positions + offset, shares, 0.36, label=label)
        for bar, share in zip(bars, shares, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, share + 1, f"{share:.1f}%", ha="center", fontsize=9)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions, POPULATIONS)
    axis.set_ylim(0, ylim)
    axis.legend(loc="lower right" if ylim == 100 else "upper right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save(figure, name)


def plot_review_queue(summary: dict) -> None:
    correct, wrong, unknown = (np.array(values) for values in
                               population_values(summary, ("selected_correct", "selected_wrong", "selected_unknown")))
    positions = np.arange(len(POPULATIONS))
    figure, axis = plt.subplots(figsize=(8, 5.8))
    axis.bar(positions, correct, label="Known correct")
    axis.bar(positions, wrong, bottom=correct, label="Known wrong")
    axis.bar(positions, unknown, bottom=correct + wrong, label="Labels cannot judge")
    axis.set_title("Clips kept by the historical review cutoff\nNew-court model; video 53 removed for sensitivity")
    axis.set_ylabel("Selected clips")
    axis.set_xticks(positions, POPULATIONS)
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    for index in positions:
        axis.text(index, correct[index] + wrong[index] + unknown[index] + 8,
                  f"{correct[index] + wrong[index] + unknown[index]} total", ha="center", fontsize=9)
        axis.text(index, correct[index] / 2, f"{correct[index]} correct", ha="center", va="center", fontsize=9)
        axis.text(index, correct[index] + wrong[index] / 2, f"{wrong[index]} wrong", ha="center", va="center",
                  fontsize=9)
    figure.tight_layout()
    save(figure, "review_queue")


def plot_court_comparison(summary: dict) -> None:
    comparison = summary["court_comparison"]
    labels = ["Historical model\nold courts", "Fresh refit\nold courts", "Fresh refit\nnew courts"]
    values = [comparison["historical"]["old"], comparison["fresh_old_courts"]["old"],
              comparison["fresh_old_courts"]["new"]]
    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1, 1.6]})
    positions = np.arange(len(values))
    # Scale bar, not a confidence band: one pair of seeds gives one observed difference.
    seed_difference = 33
    bar_bottom = min(values)
    bar_position = 3.25
    left.plot([bar_position, bar_position], [bar_bottom, bar_bottom + seed_difference], color=ORANGE, linewidth=3)
    for end in (bar_bottom, bar_bottom + seed_difference):
        left.plot([bar_position - 0.08, bar_position + 0.08], [end, end], color=ORANGE, linewidth=2)
    left.text(bar_position, bar_bottom - 5, "Two seeds of one\nrefit differed by\n33 rallies", ha="center", va="top",
              fontsize=10, color=ORANGE)
    left.scatter(positions, values, color=[GREY, GREY, BLUE], s=140, zorder=3)
    for position, value in zip(positions, values, strict=True):
        left.text(position + 0.12, value, f"{value:,} ({100 * value / 3327:.1f}%)", va="center", fontsize=11)
    left.set_xticks(positions, labels)
    left.set(xlim=(-0.4, 3.75), ylim=(1680, 1800), ylabel="Fully correct rallies (of 3,327)")
    left.grid(axis="y", alpha=0.2)
    left.set_title("The three totals differ by less than two seeds did", loc="left", fontsize=14)
    per_video = pd.DataFrame(comparison["fresh_old_courts"]["per_video"]).T
    per_video.index = per_video.index.astype(int)
    per_video["net"] = per_video.new - per_video.old
    per_video = per_video.sort_values("net")
    positions = np.arange(len(per_video))
    right.bar(positions, per_video.gained, color=BLUE, label="Rallies gained")
    right.bar(positions, -per_video.lost, color=ORANGE, label="Rallies lost")
    right.scatter(positions, per_video.net, color="black", s=14, zorder=3, label="Net change")
    right.axhline(0, color="black", linewidth=0.8)
    right.set_xticks(positions, per_video.index, fontsize=8, rotation=90)
    right.set(xlabel="Video ID, sorted by net change", ylabel="Complete rallies vs fresh old-court refit")
    right.set_title(f"{comparison['fresh_old_courts']['gained']} gained and "
                    f"{comparison['fresh_old_courts']['lost']} lost, spread across videos", loc="left", fontsize=14)
    right.legend(frameon=False, fontsize=10, loc="upper left")
    figure.suptitle("Better courts move many individual rallies but barely change the total", x=0.06, ha="left",
                    fontsize=17, weight="bold")
    figure.text(0.06, 0.885, "46 ShuttleSet22 videos · cleaned labels · ±10 frames at 30 fps · paired against the "
                "same labels", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    save(figure, "court_comparison")


def plot_court_change_groups() -> None:
    with gzip.open(ROOT / "results/court_change_groups.json.gz", "rt") as source:
        groups = json.load(source)
    labels_by_change = {(row["old_courts"], row["new_courts"]): row for row in groups["labels"]}
    rows = groups["fresh_old_courts"]
    label_keys = {"Accepted under both courts": ("accepted", "accepted"),
                  "Rescued by the new courts": ("rejected", "accepted"),
                  "Rejected under both courts": ("rejected", "rejected"),
                  "Rejected only under the new courts": ("accepted", "rejected")}
    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True, gridspec_kw={"width_ratios": [1, 1.1]})
    positions = np.arange(len(rows))[::-1]
    for position, row in zip(positions, rows, strict=True):
        change = labels_by_change[label_keys[row["group"]]]
        before, after = change["historical_matched"], change["new_matched"]
        left.barh(position + 0.18, 100 * before / change["labels"], 0.34, color=GREY)
        left.barh(position - 0.18, 100 * after / change["labels"], 0.34, color=BLUE)
        left.text(102, position, f"{before:,} → {after:,}\nof {change['labels']:,}", va="center", fontsize=11)
        right.barh(position, row["gained"], 0.6, color=BLUE)
        right.barh(position, -row["lost"], 0.6, color=ORANGE)
        right.text(max(row["gained"], 0) + 4, position, f"net {row['net']:+d} of {row['rallies']:,} rallies",
                   va="center", fontsize=11)
    left.set_yticks(positions, [row["group"].replace(" under", "\nunder").replace(" by", "\nby") for row in rows])
    left.set_xlim(0, 100)
    left.set_xlabel("Labelled contacts with a timing match (%)")
    left.set_title("Contacts: historical model (grey) → new courts (blue)", loc="left", fontsize=13)
    right.axvline(0, color="black", linewidth=0.8)
    right.set_xlim(-160, 290)
    right.set_xlabel("Complete rallies gained (blue) and lost (orange) vs fresh old-court refit")
    right.set_title("Complete rallies vs fresh old-court refit", loc="left", fontsize=13)
    figure.suptitle("The court change cuts both ways; accepted scenes barely move", x=0.06, ha="left", fontsize=17,
                    weight="bold")
    figure.text(0.06, 0.885, SUBTITLE + " · a rally's group depends on whether any of its labels fell in a rejected "
                "scene", fontsize=10)
    figure.subplots_adjust(top=0.78, left=0.12, right=0.97, wspace=0.42)
    save(figure, "court_change_groups")


def plot_original_comparison(summary: dict) -> None:
    development = summary["development"]
    test = summary["headline"]["all_10"]
    groups = [
        ("Original ShuttleSet\nvalidation (8 videos)", development["validation_10"], "rallies", "contacts"),
        ("Original ShuttleSet\nheld-out groups (32 videos)", development["development_heldout_10"], "rallies",
         "contacts"),
        ("ShuttleSet22\n(46 videos)", {"fully_correct": test["fully_correct"], "timing_player": test["timing_player"],
                                      "serve_timing_player": test["serve_timing_player"],
                                      "rallies": test["labelled_rallies"], "contacts": test["labelled_contacts"]},
         "rallies", "contacts"),
    ]
    measures = (("Fully correct rally", "fully_correct", "rallies"),
                ("Contact timing + correct player", "timing_player", "contacts"),
                ("Serve timing + correct player", "serve_timing_player", "rallies"))
    figure, axis = plt.subplots(figsize=(13, 6.5))
    positions = np.arange(len(measures))
    width = 0.26
    for offset, (label, values, _rallies, _contacts), colour in zip((-width, 0, width), groups,
                                                                   (PURPLE, GREY, BLUE), strict=True):
        shares = [100 * values[key] / values[denominator] for _name, key, denominator in measures]
        bars = axis.bar(positions + offset, shares, width, color=colour, label=label.replace("\n", " "))
        for bar, share in zip(bars, shares, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, share + 1, f"{share:.1f}%", ha="center", fontsize=10)
    axis.set_xticks(positions, [name for name, _key, _denominator in measures])
    axis.set(ylim=(0, 105), ylabel="Correct / labelled (%)")
    axis.legend(frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, -0.08))
    axis.set_title("Original ShuttleSet serves and whole rallies remain weaker than ShuttleSet22", loc="left",
                   weight="bold", pad=40)
    figure.text(0.125, 0.91, "New-court model · ±10 frames at 30 fps · each original video scored by a contact "
                "model that did not train on it", fontsize=11)
    save(figure, "original_comparison")


def run() -> None:
    FIGURES.mkdir(exist_ok=True)
    with gzip.open(ROOT / "results/summary.json.gz", "rt") as source:
        summary = json.load(source)
    contacts = test_rows("contacts")
    rallies, proposals = test_rows("rallies"), test_rows("proposals")
    per_video = pd.read_csv(ROOT / "results/per_video.csv.gz", index_col="video")
    plot_video_variation(per_video)
    plot_video_pages(contacts, per_video)
    plot_rally_coverage(rallies)
    plot_selection(proposals)
    plot_selected_errors(proposals)
    plot_timing(contacts)
    plot_contact_position({10: contacts, 5: test_rows("contacts", 5)})
    joined = with_state(contacts)
    plot_input_states(joined)
    plot_misses_by_state(joined)
    grouped_bars(summary, ("exact_sequence", "fully_correct"), "labelled_rallies",
                 ("Exact contact sequence", "Exact sequence + correct players"),
                 "Whole-rally correctness\nNew-court model; video 53 removed for sensitivity",
                 "Share of labelled rallies (%)", 75, "rally_correctness")
    grouped_bars(summary, ("timing_matched", "timing_player"), "labelled_contacts",
                 ("Timing match", "Timing + correct player"),
                 "Labelled contacts recovered\nNew-court model; video 53 removed for sensitivity",
                 "Recall of labelled contacts (%)", 100, "contact_correctness")
    plot_review_queue(summary)
    plot_court_comparison(summary)
    plot_court_change_groups()
    plot_original_comparison(summary)


if __name__ == "__main__":
    run()
