"""Render the stage-8 rally-segmentation sweep "menu" CSVs as trade-off charts.

Three .png files come out of a run:
  1. boundary_tradeoff.png   -- coverage vs merges for every boundary config,
     with the coverage-vs-merges frontier and the three candidate crowns marked.
  2. boundary_start_earliness.png -- how early each config opens its spans
     (start_alignment_median), against coverage, with the crowns marked.
  3. contact_tradeoff.png    -- contact-time precision vs recall for every
     contact config, the Pareto frontier joined and shipped row flagged.

Charts 1 and 2 gain a second side-by-side panel (unmasked | masked) when a
--masked-dir is supplied, so the replay-mask variant reads against the stock one.

Palette follows ~/Documents/protan_colour_scheme.md (mild protanopia, no red or
red/green encodings). On the white figure background the doc calls for the darker
accent variants: navy #1e40af for the frontier line, orange #e88806 / pink-dark
#be185d / green #1a8c3c for the three crowns, light-mode lavender #7c3aed for the
    contact shipped row, and grey #888a85 for the de-emphasised config cloud. Crowns are told
apart by marker SHAPE plus a text label, so hue never carries the distinction alone.

Basic and re-runnable: point --unmasked-dir / --masked-dir at any later sweep output.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless: we only ever savefig, never show a window.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_UNMASKED = Path(
    "/home/ariel/Documents/COSC594/badminton_cv_annotator/"
    "local_scratch/autograder_architecture/pilot_results/sweep_widened"
)
DEFAULT_OUT = Path(
    "/home/ariel/Documents/COSC594/badminton_cv_annotator/"
    "local_scratch/autograder_architecture/pilot_results/plots"
)

# Protan-safe accents on a white background (see module docstring for provenance).
GREY_CLOUD = "#888a85"  # de-emphasised scatter of every config
NAVY = "#1e40af"  # frontier line (structural cool, holds on white)
LAVENDER = "#7c3aed"  # shipped contact row (light-mode lavender)
SHIPPED_LABEL = "shipped_defaults"

# Each candidate boundary crown gets its own SHAPE so it reads without relying on hue.
# (colour, matplotlib marker, plain-language label).
CROWN_STYLES = {
    "as_built": ("#e88806", "o", "as-built (max coverage)"),
    "merge_penalised": ("#be185d", "s", "merge-penalised (fewest glued)"),
    "start_alignment_penalised": ("#1a8c3c", "D", "start-aligned (opens latest)"),
}

FRONTIER_PREFIX = "frontier_cov"  # crown_key form for the coverage-vs-merges frontier rows


def load_boundary(sweep_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a boundary sweep directory's two CSVs.

    :param sweep_dir: directory holding boundary_sweep.csv and boundary_crowns.csv
    :return: (all-configs frame, crowns frame). The crowns frame keeps its
             crown_key column so callers can split named crowns from frontier rows.
    """
    sweep = pd.read_csv(sweep_dir / "boundary_sweep.csv")
    crowns = pd.read_csv(sweep_dir / "boundary_crowns.csv")
    return sweep, crowns


def derive_n_rallies(sweep: pd.DataFrame, fallback: int) -> int:
    """Recover the ground-truth rally count from covered / covered_fraction.

    covered_fraction = covered / n_gt, so n_gt = covered / covered_fraction for any
    config that covered at least one rally. The most common estimate wins, since
    rounding noise can nudge a stray row by one.

    :param sweep: all-configs frame
    :param fallback: value to use if no row has a positive covered_fraction
    :return: ground-truth rally count
    """
    scored = sweep[(sweep["covered"] > 0) & (sweep["covered_fraction"] > 0)]
    if scored.empty:
        return fallback
    estimates = (scored["covered"] / scored["covered_fraction"]).round()
    return int(estimates.mode().iloc[0])


def split_crowns(crowns: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Split the crowns frame into the frontier rows and the named candidate crowns.

    :param crowns: frame from boundary_crowns.csv
    :return: (frontier frame sorted by covered, {crown_key: row} for the named crowns)
    """
    is_frontier = crowns["crown_key"].str.startswith(FRONTIER_PREFIX)
    frontier = crowns[is_frontier].sort_values("covered")
    named = {}
    for crown_key in CROWN_STYLES:
        match = crowns[crowns["crown_key"] == crown_key]
        if not match.empty:
            named[crown_key] = match.iloc[0]
    return frontier, named


def draw_tradeoff_panel(ax: plt.Axes, sweep: pd.DataFrame, crowns: pd.DataFrame,
                        n_rallies: int, title: str) -> None:
    """Draw one coverage-vs-merges panel: config cloud, frontier, crowns.

    Many configs land on the same integer (merged_spans, covered) point, so the
    cloud is de-duplicated and each dot sized by how many configs share it.

    :param ax: axes to draw onto
    :param sweep: all-configs frame for this variant
    :param crowns: crowns frame for this variant
    :param n_rallies: ground-truth rally count (for the y-axis label)
    :param title: panel title
    """
    stacked = sweep.groupby(["merged_spans", "covered"]).size().reset_index(name="n_configs")
    # Area grows with sqrt(count) so a busy point reads bigger without swamping the panel.
    sizes = 8.0 + 6.0 * np.sqrt(stacked["n_configs"])
    ax.scatter(stacked["merged_spans"], stacked["covered"], s=sizes, color=GREY_CLOUD,
               alpha=0.35, edgecolors="none", zorder=1,
               label="configs (dot size = how many share the point)")

    frontier, named = split_crowns(crowns)
    ax.plot(frontier["merged_spans"], frontier["covered"], drawstyle="steps-post",
            color=NAVY, lw=1.8, alpha=0.9, zorder=3, label="fewest-merges frontier")

    for crown_key, (colour, marker, label) in CROWN_STYLES.items():
        row = named.get(crown_key)
        if row is None:
            continue
        ax.scatter(row["merged_spans"], row["covered"], s=170, color=colour, marker=marker,
                   edgecolors="black", linewidths=0.8, zorder=5, label=label)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Rallies glued together (merged spans)")
    ax.set_ylabel(f"Rallies covered (of {n_rallies})")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)


def plot_boundary_tradeoff(unmasked: tuple[pd.DataFrame, pd.DataFrame],
                           masked: tuple[pd.DataFrame, pd.DataFrame] | None,
                           n_rallies: int, out_path: Path) -> None:
    """Render boundary_tradeoff.png (one panel, or unmasked | masked side by side)."""
    sweep_u, crowns_u = unmasked
    if masked is None:
        fig, ax = plt.subplots(figsize=(9, 7))
        draw_tradeoff_panel(ax, sweep_u, crowns_u, n_rallies, "Unmasked broadcast footage")
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        draw_tradeoff_panel(axes[0], sweep_u, crowns_u, n_rallies, "Unmasked broadcast footage")
        draw_tradeoff_panel(axes[1], masked[0], masked[1], n_rallies, "Replay-masked")

    fig.suptitle("Rally boundary trade-off: coverage against merged spans",
                 fontsize=13, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def draw_earliness_panel(ax: plt.Axes, sweep: pd.DataFrame, crowns: pd.DataFrame,
                         n_rallies: int, fps: float, title: str) -> None:
    """Draw one start-earliness panel: how early spans open, against coverage.

    Configs that covered no rallies have no start_alignment_median (nothing to align
    against), so those rows are dropped here. A top secondary axis restates the frame
    offset in seconds via fps.

    :param ax: axes to draw onto
    :param sweep: all-configs frame for this variant
    :param crowns: crowns frame for this variant
    :param n_rallies: ground-truth rally count (for the y-axis label)
    :param fps: frames per second, for the seconds secondary axis
    :param title: panel title
    """
    scored = sweep.dropna(subset=["start_alignment_median"])
    ax.scatter(scored["start_alignment_median"], scored["covered"], s=18, color=GREY_CLOUD,
               alpha=0.30, edgecolors="none", zorder=1, label="configs")

    _, named = split_crowns(crowns)
    for crown_key, (colour, marker, label) in CROWN_STYLES.items():
        row = named.get(crown_key)
        if row is None or pd.isna(row["start_alignment_median"]):
            continue
        ax.scatter(row["start_alignment_median"], row["covered"], s=170, color=colour,
                   marker=marker, edgecolors="black", linewidths=0.8, zorder=5, label=label)

    ax.axvline(0.0, color="black", lw=0.8, alpha=0.4, zorder=2)  # 0 = span opens on time
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Median start offset (frames; negative = span opens early)")
    ax.set_ylabel(f"Rallies covered (of {n_rallies})")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    # Seconds mirror on top. frames / fps <-> seconds, so fps is the round-trip constant.
    secondary = ax.secondary_xaxis("top", functions=(lambda f: f / fps, lambda s: s * fps))
    secondary.set_xlabel(f"Median start offset (seconds at {fps:g} fps)")


def plot_boundary_earliness(unmasked: tuple[pd.DataFrame, pd.DataFrame],
                            masked: tuple[pd.DataFrame, pd.DataFrame] | None,
                            n_rallies: int, fps: float, out_path: Path) -> None:
    """Render boundary_start_earliness.png (one panel, or unmasked | masked)."""
    sweep_u, crowns_u = unmasked
    if masked is None:
        fig, ax = plt.subplots(figsize=(9, 7))
        draw_earliness_panel(ax, sweep_u, crowns_u, n_rallies, fps, "Unmasked broadcast footage")
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        draw_earliness_panel(axes[0], sweep_u, crowns_u, n_rallies, fps, "Unmasked broadcast footage")
        draw_earliness_panel(axes[1], masked[0], masked[1], n_rallies, fps, "Replay-masked")

    fig.suptitle("Rally boundary trade-off: how early spans open, against coverage",
                 fontsize=13, fontweight="bold")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_contact_tradeoff(sweep_dir: Path, out_path: Path) -> None:
    """Render contact_tradeoff.png: contact-time precision vs recall.

    Every config is a grey dot; the Pareto frontier is joined. The shipped row
    (smooth_window 3) is starred.
    """
    sweep = pd.read_csv(sweep_dir / "contact_sweep.csv")
    frontier = pd.read_csv(sweep_dir / "contact_frontier.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(sweep["precision_5"], sweep["recall_5"], s=28, color=GREY_CLOUD, alpha=0.45,
               edgecolors="none", zorder=1, label="contact configs")

    front = frontier.sort_values("precision_5")
    ax.plot(front["precision_5"], front["recall_5"], color=NAVY, lw=1.6, marker="o",
            markersize=6, zorder=3, label="Pareto frontier")
    shipped = sweep[sweep["label"] == SHIPPED_LABEL]
    if len(shipped) != 1:
        raise ValueError(
            f"expected one shipped contact row in contact_sweep.csv, found {len(shipped)}"
        )
    shipped_row = shipped.iloc[0]
    ax.scatter(shipped_row["precision_5"], shipped_row["recall_5"], s=260, color=LAVENDER,
               marker="*", edgecolors="black", linewidths=0.8, zorder=5,
               label="shipped contact row (smooth 3)")

    ax.set_title("Contact-time detection trade-off: precision vs recall", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("Contact-time precision (within +/-5 frames)")
    ax.set_ylabel("Contact-time recall (within +/-5 frames)")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unmasked-dir", type=Path, default=DEFAULT_UNMASKED,
                        help="Sweep output dir with boundary_/contact_ CSVs (default: sweep_widened)")
    parser.add_argument("--masked-dir", type=Path, default=None,
                        help="Optional replay-masked sweep dir; adds a second panel to the boundary charts")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                        help="Where the .png files are written (default: pilot_results/plots)")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Frames per second, used only to annotate the seconds axis (default: 25.0)")
    parser.add_argument("--n-rallies", type=int, default=113,
                        help="Ground-truth rally count fallback if it can't be derived (default: 113)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    unmasked = load_boundary(args.unmasked_dir)
    masked = load_boundary(args.masked_dir) if args.masked_dir is not None else None
    n_rallies = derive_n_rallies(unmasked[0], args.n_rallies)

    plot_boundary_tradeoff(unmasked, masked, n_rallies, args.out_dir / "boundary_tradeoff.png")
    plot_boundary_earliness(unmasked, masked, n_rallies, args.fps,
                            args.out_dir / "boundary_start_earliness.png")
    plot_contact_tradeoff(args.unmasked_dir, args.out_dir / "contact_tradeoff.png")


if __name__ == "__main__":
    main()
