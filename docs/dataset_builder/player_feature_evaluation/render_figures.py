"""Render the six report figures from saved player summaries and fitted results.

The bundled tables supply player summaries, groups and previously fitted results.
No model is fitted, and only the six PNGs in figures/ are written.
"""
from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
INK, MUTED, GRID = "#213047", "#586577", "#E7EBF0"
COLORS = ("#276FBF", "#C68B17", "#B75583")
MARKERS = ("o", "^", "D")
METHODS = ("rally_weighted_binomial", "equal_match_fractional_logit", "spearman_player_permutation")


@dataclass(frozen=True)
class View:
    kind: str
    title: str
    fields: tuple[str, str]
    source_groups: tuple[str, str, str]
    display_groups: tuple[str, str, str]
    stem: str
    model_stem: str
    summary_note: str
    definition: str


VIEWS = (
    View(
        "rhythm", "Rally rhythm and rally win rate", ("contacts", "interval_seconds"),
        ("Shorter, faster rallies", "Shorter, slower rallies", "Longer rallies"),
        ("Shorter, faster rallies", "Shorter, slower rallies", "Longer rallies"),
        "12-player-rally-rhythm-results", "15-player-rally-rhythm-model-checks",
        "Player values: geometric means of per-match medians.",
        "The horizontal values are each player's geometric means of match medians, with equal match weights. "
        "Contacts per rally and time between contacts describe exchanges involving both opponents.",
    ),
    View(
        "movement", "Recovery movement and rally win rate", ("centre_distance", "excess_path"),
        ("Closer to centre, less extra travel", "Middle movement profile", "Farther from centre, more extra travel"),
        ("Closer to centre, less extra travel", "Intermediate centre distance and extra travel",
         "Farther from centre, more extra travel"),
        "13-player-recovery-movement-results", "16-player-recovery-movement-model-checks",
        "Distances use normalised court coordinates; definitions are in the caption.",
        "Centre distance measures the non-striking player's distance from the centre of their current court half "
        "around opponent contacts. Extra travel is tracked path length minus the straight-line distance between "
        "positions at successive contacts. Both use unitless normalised court coordinates. Each player's values "
        "are geometric means of match medians, with equal match weights. Tracking and court calibration were "
        "not independently checked for this analysis; lower distances are not established as better play.",
    ),
    View(
        "shots", "Shot choices and rally win rate", ("Net play", "Clears / lifts"),
        ("More net play", "More clears and lifts", "Mixed net and clear profile"),
        ("Higher net-shot share", "Higher clear/lift share", "Intermediate net-shot and clear/lift shares"),
        "14-player-shot-choices-results", "17-player-shot-choices-model-checks",
        "Player shot shares: averages of per-match shares, with equal match weights.",
        "The horizontal values are arithmetic means of each player's match shares of mapped, labelled non-serve "
        "contacts, with equal match weights. Grouping uses all six shot shares; the panels show net shots and "
        "clears/lifts. The labels describe proportions, rather than total shot counts. Group outcomes overlap, "
        "and the plots do not establish that changing shot choices causes better results.",
    ),
)

AXIS_LABELS = {
    "contacts": "Typical contacts per rally",
    "interval_seconds": "Typical time between contacts (s)",
    "centre_distance": "Centre distance around opponent contacts\n(normalised coordinates)",
    "excess_path": "Extra travel between contacts\n(normalised coordinates)",
    "Net play": "Net shots\n(% of non-serve contacts)",
    "Clears / lifts": "Clears and lifts\n(% of non-serve contacts)",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as file:
        return list(csv.DictReader(file))


def draw(view: View, players: list[dict[str, str]], linear: list[dict[str, str]],
         models: list[dict[str, str]], curves: list[dict[str, str]], follow_up: bool) -> None:
    """Draw saved points, line endpoints or logistic curve coordinates.

    :param follow_up: Select the logistic/Spearman figure instead of the linear fit.
    """
    fig = plt.figure(figsize=(8.8, 8.0 if follow_up else 6.6), facecolor="white")
    fig.text(.09, .94, view.title, fontsize=18, weight="bold", color=INK)
    fig.text(.09, .891, "43 players · 86 selected elite matches · one point per player", fontsize=12, color=MUTED)
    if follow_up:
        line_handles = [Line2D([], [], color=INK, linestyle="--", linewidth=2),
                        Line2D([], [], color=MUTED, linestyle="-", linewidth=2)]
        fig.legend(line_handles, ["Logistic fit: rally weights", "Logistic fit: equal player-match weights"],
                   loc="center", bbox_to_anchor=(.53, .845), ncol=2, frameon=False, fontsize=11)
    else:
        fig.text(.09, .844, "Behavioural groups were formed without using rally win rates.",
                 fontsize=11.5, color=MUTED)
    rates = [float(player["win_rate"]) * 100 for player in players]
    endpoints = {rates.index(min(rates)), rates.index(max(rates))}
    for panel, (field, left) in enumerate(zip(view.fields, (.105, .575))):
        ax = fig.add_axes([left, .54 if follow_up else .40, .39, .24 if follow_up else .38])
        multiplier = 100 if view.kind == "shots" else 1
        values = [float(player[field]) * multiplier for player in players]
        span = max(max(values) - min(values), .01)
        ax.set_xlim(min(values) - .16 * span, max(values) + .16 * span)
        ax.set_ylim(25, 65)
        ax.set_yticks(range(25, 66, 10))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if view.kind == "shots":
            ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.grid(color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=12, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color("#CFD6DF")
        ax.set_xlabel(AXIS_LABELS[field], fontsize=12, color=INK, labelpad=8)
        if panel == 0:
            ax.set_ylabel("Recorded rally win rate", fontsize=12.5, color=INK, labelpad=8)
        if follow_up:
            for method, color, style in zip(METHODS[:2], (INK, MUTED), ("--", "-")):
                saved = [row for row in curves if row["feature"] == field and row["method"] == method]
                ax.plot([float(row["x_display_value"]) for row in saved],
                        [float(row["fitted_win_percent"]) for row in saved],
                        color=color, linestyle=style, linewidth=2, zorder=2)
        else:
            saved = next(row for row in linear if row["x_field"] == field)
            ax.plot([float(saved["x_min"]), float(saved["x_max"])],
                    [float(saved["fitted_win_percent_at_x_min"]), float(saved["fitted_win_percent_at_x_max"])],
                    color=INK, linestyle="--", linewidth=1.8, zorder=2)
            label = f"R² = {float(saved['r_squared']):.3f} · p = {float(saved['two_sided_p_unadjusted']):.3f}"
            ax.text(.98, 1.02, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=11.5, color=INK)
        for index, (player, value, rate) in enumerate(zip(players, values, rates)):
            cluster = view.source_groups.index(player[f"{view.kind}_group"])
            ax.scatter(value, rate, s=48, marker=MARKERS[cluster], color=COLORS[cluster],
                              edgecolor=INK, linewidth=.45, zorder=3)
            if index in endpoints:
                count = int(player["outcome_matches"])
                coverage = f"{count} {'match' if count == 1 else 'matches'}"
                offset = 14 if rate > 50 else -16
                ax.annotate(f"{player['player_name']}\n{coverage}", (value, rate), xytext=(0, offset),
                            textcoords="offset points", ha="center", va="center", fontsize=10.5, color=INK,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1, "alpha": .93},
                            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": .5})
        if follow_up:
            estimates = {row["method"]: row for row in models if row["feature"] == field}
            table_rows = []
            for name, method in zip(("Rally\nweighted", "Equal\nmatch", "Spearman"), METHODS):
                result = estimates[method]
                table_rows.append([name, f"{float(result['p_unadjusted']):.3f}",
                                   f"{float(result['p_holm_six_features']):.3f}"])
            table_ax = fig.add_axes([left, .265, .39, .165])
            table_ax.axis("off")
            table = table_ax.table(cellText=table_rows,
                                   colLabels=["Method", "Unadjusted\np", "Holm\nadjusted p"],
                                   colWidths=[.30, .34, .36], bbox=[0, 0, 1, 1], cellLoc="center", colLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            for (row, _), cell in table.get_celld().items():
                cell.PAD = .025
                cell.set_edgecolor("white")
                cell.set_facecolor("#F0F3F7" if row == 0 else "white")
                cell.set_text_props(color=INK, weight="bold" if row == 0 else "normal")
            rho = float(estimates[METHODS[2]]["spearman_rho"])
            fig.text(left, .245, f"Spearman rₛ = {rho:.2f}", fontsize=11.5, color=INK)
    legend_start = .209 if follow_up else .232
    for cluster, group in enumerate(view.display_groups):
        count = sum(player[f"{view.kind}_group"] == view.source_groups[cluster] for player in players)
        handle = Line2D([], [], color=COLORS[cluster], marker=MARKERS[cluster], linestyle="none", markersize=6)
        fig.legend([handle], [f"{group} ({count} players)"], loc="center left",
                   bbox_to_anchor=(.085, legend_start - cluster * .035), frameon=False, fontsize=11.5,
                   borderpad=0, handletextpad=.6)
    notes = (
        ("Holm adjustment covers six features per method; all adjusted p > 0.05.",
         "Exploratory associations; prediction was not evaluated on separate matches.")
        if follow_up else
        (view.summary_note,
         "Dashed: linear fit with equal player weights. p: unadjusted slope test.",
         "Exploratory associations; prediction was not evaluated on separate matches.")
    )
    positions = (.079, .044) if follow_up else (.097, .060, .023)
    for y, note in zip(positions, notes):
        fig.text(.09, y, note, fontsize=11, color=MUTED)
    stem = view.model_stem if follow_up else view.stem
    fig.savefig(FIGURES / f"{stem}.png", dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "path.simplify": False})
    FIGURES.mkdir(exist_ok=True)
    players = read_rows(DATA / "player-lookup.csv.gz")
    linear = read_rows(DATA / "linear-fit-results.csv.gz")
    models = read_rows(DATA / "model-results.csv.gz")
    curves = read_rows(DATA / "logistic-curve-values.csv.gz")
    for view in VIEWS:
        draw(view, players, linear, models, curves, False)
        draw(view, players, linear, models, curves, True)
    print("Rendered the six report PNGs from the bundled tables; no models refitted.")


if __name__ == "__main__":
    main()
