"""Draw court-comparison and shuttle-guard figures from published report counts."""

from pathlib import Path

import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parents[1] / "figures"
BLUE = "#4477AA"
SAND = "#DDCC77"
GREY = "#999999"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

LABELLED_CONTACTS = 37_184
OLD_MATCHES = 33_529
NEW_MATCHES = 34_200
RALLY_CHANGE = [
    ("Accepted only by new detector", 47),
    ("Rejected only by new detector", -33),
    ("Accepted by both", 17),
    ("Rejected by both", -21),
]
GUARD_GROUPS = [
    ("False final contact", 232, 110),
    ("Real contact just before it\n(same rallies)", 232, 2),
    ("Final contact in a\nfully correct rally", 1_721, 12),
    ("Any matched contact", 33_781, 785),
]


def save(fig: plt.Figure, name: str, note: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.text(0.04, 0.03, note, fontsize=11, va="bottom", color="#454545")
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


def plot_court_input_contact_recall() -> None:
    missed = [LABELLED_CONTACTS - OLD_MATCHES, LABELLED_CONTACTS - NEW_MATCHES]
    reduction = missed[0] - missed[1]
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.25, top=0.82)
    fig.suptitle("New court inputs reduce missed contacts by 18.4%", x=0.04,
                 ha="left", fontsize=16, weight="bold")
    fig.text(0.04, 0.895, f"{reduction:,} fewer misses among {LABELLED_CONTACTS:,} labelled contacts")
    bars = ax.bar(["Old-court refit", "New-court annotator"], missed,
                  color=[GREY, BLUE], width=0.72)
    ax.bar_label(bars, labels=[f"{value:,}" for value in missed], padding=7, fontsize=13)
    ax.set_ylim(0, 4_300)
    ax.set_ylabel("Labelled contacts missed")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.15)
    save(fig, "court_input_contact_recall",
         "46 videos · ±10 frames at 30 fps · same fitting code and scikit-learn 1.9.1\n"
         "Contact precision: 85.04% with old court inputs; 84.98% with new court inputs.")


def plot_rally_change_decomposition() -> None:
    labels, values = zip(*RALLY_CHANGE, strict=True)
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    fig.subplots_adjust(left=0.30, right=0.97, bottom=0.25, top=0.82)
    fig.suptitle("Court changes produce gains and losses in complete rallies", x=0.04,
                 ha="left", fontsize=15, weight="bold")
    fig.text(0.04, 0.895, "223 rallies gained − 213 lost = 10 more fully correct rallies")
    bars = ax.barh(labels, values, color=[BLUE, SAND, GREY, GREY], height=0.70)
    ax.invert_yaxis()
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlim(-48, 62)
    ax.set_xlabel("Net change in fully correct rallies")
    for bar, value in zip(bars, values, strict=True):
        ax.text(value + (2 if value > 0 else -2), bar.get_y() + bar.get_height() / 2,
                f"{value:+d}", va="center", ha="left" if value > 0 else "right", weight="bold")
    save(fig, "rally_change_decomposition",
         "New-court annotator compared with the old-court refit · 3,327 labelled rallies\n"
         "Court groups follow acceptance at the rally's labelled contacts.")


def plot_final_hit_guard() -> None:
    labels = [row[0] for row in GUARD_GROUPS]
    rates = [100 * flagged / total for _, total, flagged in GUARD_GROUPS]
    fig, ax = plt.subplots(figsize=(11.5, 5.1))
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.25, top=0.82)
    fig.suptitle("The guard flags many false final contacts and few real ones", x=0.04,
                 ha="left", fontsize=15, weight="bold")
    fig.text(0.04, 0.895, "A flag marks an unreliable position in the tracked shuttle path")
    bars = ax.barh(labels, rates, color=[SAND, BLUE, BLUE, GREY], height=0.68)
    ax.invert_yaxis()
    ax.set_xlim(0, 65)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel("Contacts flagged on their own frame (%)")
    for bar, rate, (_, total, flagged) in zip(bars, rates, GUARD_GROUPS, strict=True):
        ax.text(rate + 1, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%  ({flagged:,} / {total:,})", va="center", fontsize=11)
    save(fig, "final_hit_guard",
         "Video 22 is excluded because its guard stream was unavailable.\n"
         "These test-output comparisons motivate a rule that still needs development-set validation.")


def plot_court_change_groups() -> None:
    # Contact diagnostics use the historical output; rally changes use the refit.
    groups = [
        ("Accepted by\nboth detectors", 33_982, 32_778, 32_707, 2_677, 154, 137),
        ("Accepted only by\nthe new detector", 1_367, 8, 1_278, 114, 48, 1),
        ("Rejected by\nboth detectors", 1_247, 232, 215, 443, 21, 42),
        ("Rejected only by\nthe new detector", 588, 533, 0, 93, 0, 33),
    ]
    fig, (contacts, rallies) = plt.subplots(
        1, 2, figsize=(14.5, 6.0), sharey=True,
        gridspec_kw={"width_ratios": [1, 1.05]},
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.28, top=0.79, wspace=0.46)
    fig.suptitle("Contact recovery and rally changes by court decision", x=0.06,
                 ha="left", fontsize=17, weight="bold")
    fig.text(0.06, 0.9, "46 ShuttleSet22 videos · 3,327 labelled rallies · 37,184 labelled contacts · ±10 frames at 30 fps",
             fontsize=12)
    positions = list(range(len(groups)))[::-1]
    for position, (_, labels, old, new, total_rallies, gained, lost) in zip(positions, groups, strict=True):
        contacts.barh(position + 0.17, 100 * old / labels, 0.3, color=GREY,
                      label="Historical model" if position == positions[0] else None)
        contacts.barh(position - 0.17, 100 * new / labels, 0.3, color=BLUE,
                      label="New-court annotator" if position == positions[0] else None)
        contacts.text(102, position, f"{old:,} → {new:,}\nof {labels:,}",
                      va="center", fontsize=11)
        rallies.barh(position, gained, 0.6, color=BLUE,
                     label="Rallies gained" if position == positions[0] else None)
        rallies.barh(position, -lost, 0.6, color=SAND,
                     label="Rallies lost" if position == positions[0] else None)
        if gained:
            rallies.text(gained + 3, position, f"+{gained}", va="center", fontsize=11)
        if lost:
            rallies.text(-lost - 3, position, f"−{lost}", va="center", ha="right", fontsize=11)
        rallies.text(215, position, f"Net {gained - lost:+d}\n{total_rallies:,} rallies",
                     va="center", fontsize=11)
    contacts.set_yticks(positions, [row[0] for row in groups])
    contacts.set(xlim=(0, 100), ylim=(-0.55, 3.55),
                 xlabel="Labelled contacts recovered (%)")
    contacts.set_title("Contact baseline: historical model", loc="left", fontsize=13, pad=12)
    rallies.set(xlim=(-170, 305), xticks=[-150, -100, -50, 0, 50, 100, 150],
                xlabel="Change in fully correct rallies")
    rallies.set_title("Rally baseline: old-court refit", loc="left", fontsize=13, pad=12)
    rallies.axvline(0, color="black", linewidth=0.8)
    rallies.spines["left"].set_visible(False)
    rallies.tick_params(axis="y", left=False)
    for axis in (contacts, rallies):
        axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=11)
    save(fig, "court_change_groups",
         "Contact groups use court status at each labelled frame. A rally counts as rejected if any of its labels falls in a rejected scene.\n"
         "Across all rally groups: 223 gained and 213 lost, giving 10 additional fully correct rallies.")


def main() -> None:
    plot_court_input_contact_recall()
    plot_rally_change_decomposition()
    plot_final_hit_guard()
    plot_court_change_groups()


if __name__ == "__main__":
    main()
