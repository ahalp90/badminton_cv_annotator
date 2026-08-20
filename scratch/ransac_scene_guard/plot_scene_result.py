"""Plot the decision-relevant changes from the retained scene experiment."""

from __future__ import annotations

# Matplotlib reads MPLCONFIGDIR during import, so configure it first.
# ruff: noqa: E402

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(EXPERIMENT_ROOT / "local_runs/matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_INPUT = EXPERIMENT_ROOT / "results/scene_aware_ransac.json.gz"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/scene_aware_change.png"
POLICY_LABELS = {
    "raw_candidate": "All RANSAC",
    "recurrence_v4_clean": "Recurrence-clean",
    "recurrence_v4_clean_impulse_veto_radius3": "Clean + impulse veto",
}
METRICS = (
    ("selected_frames", "Selected frames", "#5a7a9a", ""),
    ("exact_contacts", "Exact contacts", "#9070a0", "//"),
    ("contacts_with_candidate", "Contacts within ±10", "#8a6a30", ".."),
)


def load_result(path: Path) -> dict[str, Any]:
    """Load the retained compressed JSON result."""
    with gzip.open(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported scene-result schema")
    return payload


def percentage_change(before: int, after: int) -> float:
    """Return signed percentage change from a non-zero baseline."""
    if before <= 0:
        raise ValueError(f"percentage baseline must be positive, got {before}")
    return 100.0 * (after - before) / before


def metric_value(metrics: dict[str, Any], field: str) -> int:
    """Read a top-level metric or the base-30 ±10 contact count."""
    if field == "contacts_with_candidate":
        return int(metrics["contact_tolerances"]["base30_10"][field])
    return int(metrics[field])


def plot_changes(result: dict[str, Any], output: Path) -> None:
    """Render one compact comparison chart."""
    policies = result["totals"]["policies"]
    policy_names = list(POLICY_LABELS)
    positions = np.arange(len(policy_names), dtype=np.float64)
    width = 0.23

    figure, axis = plt.subplots(figsize=(8.4, 4.8), facecolor="white")
    axis.set_facecolor("white")
    for metric_index, (field, label, colour, hatch) in enumerate(METRICS):
        changes: list[float] = []
        for policy in policy_names:
            baseline = metric_value(policies[policy]["baseline"], field)
            scene_aware = metric_value(policies[policy]["scene_aware"], field)
            changes.append(percentage_change(baseline, scene_aware))
        offset = (metric_index - 1) * width
        bars = axis.bar(
            positions + offset,
            changes,
            width,
            label=label,
            color=colour,
            edgecolor="#1a1a1a",
            linewidth=0.6,
            hatch=hatch,
        )
        axis.bar_label(bars, fmt="%+.1f%%", padding=3, fontsize=8, color="#1a1a1a")

    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.set_ylim(-3.4, 1.5)
    axis.set_xticks(positions, [POLICY_LABELS[name] for name in policy_names])
    axis.set_ylabel("Change from the same baseline")
    figure.suptitle(
        "Scene boundaries barely change contact conflicts",
        x=0.1,
        y=0.98,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.1,
        0.9,
        "All stress-set coverage is unchanged. The impulse-veto arm selects 2.8% fewer frames "
        "while exact contact conflicts rise 1.3%.",
        fontsize=9,
        color="#333333",
        ha="left",
    )
    axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0.0, -0.27))
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.83))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse retained input and plot paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Load the retained result and render its main comparison."""
    args = parse_args()
    plot_changes(load_result(args.input), args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
