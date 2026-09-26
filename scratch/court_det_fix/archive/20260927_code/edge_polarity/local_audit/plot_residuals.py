"""Plot saved SS03-34 profiles and account for saved diagnostic objectives."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from math import dist
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results.json.gz"
DIAGNOSTICS = HERE / "diagnostics.json.gz"
FRAGMENT_CHECK = HERE / "fragment111.json.gz"
CASE_ID = "shuttleset_03_scene_0034"
FRAGMENT_IDS = (111, 236, 88, 24, 209, 179)
POSITIONS = {0: "centre", 1: "negative edge", 2: "positive edge"}
COMPARISONS = ("preferred_x_only", "preferred_y_only", "preferred_xy_only", "saved_preferred")
USER_MOVEMENT_PX = (-3.0, -2.0)


def read_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def case_by_id(document: dict[str, Any], case_id: str) -> dict[str, Any]:
    return next(case for case in document["cases"] if case["case_id"] == case_id)


def make_figure(fragments: list[dict[str, Any]]) -> None:
    by_id = {fragment["raw_fragment_id"]: fragment for fragment in fragments}
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), sharex=True)
    for axis, fragment_id in zip(axes.flat, FRAGMENT_IDS, strict=True):
        fragment = by_id[fragment_id]
        profile = fragment["profile"]
        grey = profile["median_grey"]
        assert grey is not None, fragment_id
        negative, positive = profile["projected_edge_offsets_median_px"]
        width_min, width_max = profile["projected_width_range_px"]

        axis.plot(profile["offsets_px"], grey, color="#9467bd", linewidth=2, label="Saved median grey")
        axis.axvspan(negative, positive, color="#1f77b4", alpha=0.14, label="Projected paint stripe")
        axis.axvline(negative, color="#1f77b4", linestyle="--", linewidth=1.5,
                     label="Projected model edges")
        axis.axvline(positive, color="#1f77b4", linestyle="--", linewidth=1.5)
        axis.axvline(0, color="black", linestyle="--", linewidth=1.3,
                     label="Raw fragment coordinate (0)")
        axis.set_title(f"Fragment {fragment_id} · {fragment['marking_name']} · "
                       f"{POSITIONS[fragment['new_position']]}", fontsize=10.5)
        axis.text(0.02, 0.97, f"Projected width range: {width_min:.2f}–{width_max:.2f} px",
                  transform=axis.transAxes, va="top", fontsize=9,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82})
        axis.grid(alpha=0.2)
        axis.set_xlim(profile["offsets_px"][0], profile["offsets_px"][-1])
        grey_span = max(grey) - min(grey)
        axis.set_ylim(min(grey) - 0.08 * grey_span, max(grey) + 0.18 * grey_span)
        axis.set_xlabel("Signed offset along raw fragment normal (working px)")
        axis.set_ylabel("Saved median grey intensity")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.15, wspace=0.27, hspace=0.37)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=9)
    figure.text(0.5, 0.025, "SS03-34. Purple: image intensity evidence. Blue: projected model location; "
                "edges are medians along each fragment. Intensity changes alone do not identify physical boundaries.",
                ha="center", fontsize=9)
    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    figure.savefig(output / "residual_profiles.png", dpi=180)
    figure.savefig(output / "residual_profiles.svg")
    plt.close(figure)


def comparison_summary(fragments: list[dict[str, Any]], totals: dict[str, float]) -> dict[str, Any]:
    arms = ("polarity", *COMPARISONS)
    for arm in arms:
        fragment_sum = sum(fragment["objectives_px2"][arm] for fragment in fragments)
        assert abs(fragment_sum - totals[arm]) <= 1e-8, (arm, fragment_sum, totals[arm])

    comparisons = {}
    for arm in COMPARISONS:
        changes = []
        grouped = defaultdict(lambda: {"before_px2": 0.0, "after_px2": 0.0, "weight_fraction": 0.0})
        for fragment in fragments:
            before = fragment["objectives_px2"]["polarity"]
            after = fragment["objectives_px2"][arm]
            marking = fragment["marking_name"]
            changes.append({
                "fragment_id": fragment["raw_fragment_id"],
                "marking": marking,
                "position": POSITIONS[fragment["new_position"]],
                "weight_fraction": fragment["weight_fraction"],
                "before_px2": before,
                "after_px2": after,
                "delta_px2": after - before,
            })
            grouped[marking]["before_px2"] += before
            grouped[marking]["after_px2"] += after
            grouped[marking]["weight_fraction"] += fragment["weight_fraction"]
        for values in grouped.values():
            values["delta_px2"] = values["after_px2"] - values["before_px2"]
        comparisons[arm] = {
            "before_total_px2": totals["polarity"],
            "after_total_px2": totals[arm],
            "delta_total_px2": totals[arm] - totals["polarity"],
            "top_six_fragment_increases": sorted(changes, key=lambda row: row["delta_px2"], reverse=True)[:6],
            "by_marking": dict(sorted(grouped.items())),
        }
    return {"objective_totals_px2": totals, "comparisons": comparisons}


def diagnostic_cases(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for case in diagnostics["cases"]:
        polarity_corners = case["fits"]["polarity"]["corners_px"]
        arms = {}
        for name, fit in case["fits"].items():
            corners = fit["corners_px"]
            arms[name] = {
                "upper_left_movement_vs_polarity_working_px": [
                    corners[0][axis] - polarity_corners[0][axis] for axis in (0, 1)
                ],
                "max_euclidean_corner_movement_working_px": max(
                    dist(corner, reference) for corner, reference in zip(corners, polarity_corners, strict=True)
                ),
                "paint_score": fit["measurement"]["paint_score"],
            }
        cases.append({
            "case_id": case["case_id"],
            "baseline_reproduction_max_absolute_difference_working_px":
                case["polarity_baseline_max_difference_px"],
            "arms": arms,
        })
    return cases


def original_movement(saved: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    original = saved["fits"]["baseline"]["corners_px"][0]
    polarity = saved["fits"]["polarity"]["corners_px"][0]
    diagnostic_polarity = diagnostic["fits"]["polarity"]["corners_px"][0]
    assert max(abs(a - b) for a, b in zip(polarity, diagnostic_polarity, strict=True)) <= 1e-8
    movement = [after - before for before, after in zip(original, polarity, strict=True)]
    assert max(abs(a - b) for a, b in zip(movement, saved["movement_working_px"][0], strict=True)) <= 1e-8
    centre_to_edge = diagnostic["fits"]["strong_centres"]["corners_px"][0]
    total_movement = [after - before for before, after in zip(original, centre_to_edge, strict=True)]
    return {
        "original_upper_left_working_px": original,
        "polarity_upper_left_working_px": polarity,
        "polarity_movement_from_original_working_px": movement,
        "centre_to_edge_upper_left_working_px": centre_to_edge,
        "centre_to_edge_movement_from_original_working_px": total_movement,
        "user_diagnostic_movement_from_original_working_px": list(USER_MOVEMENT_PX),
        "remaining_movement_from_polarity_to_user_diagnostic_working_px": [
            wanted - actual for wanted, actual in zip(USER_MOVEMENT_PX, movement, strict=True)
        ],
        "remaining_movement_from_centre_to_edge_to_user_diagnostic_working_px": [
            wanted - actual for wanted, actual in zip(USER_MOVEMENT_PX, total_movement, strict=True)
        ],
    }


def main() -> None:
    diagnostics = read_json(DIAGNOSTICS)
    saved_results = read_json(RESULTS)
    fragment_check = read_json(FRAGMENT_CHECK)
    diagnostic = case_by_id(diagnostics, CASE_ID)
    saved = case_by_id(saved_results, CASE_ID)
    assert fragment_check["case_id"] == CASE_ID
    assert fragment_check["target_fragment_id"] == 111
    assert diagnostic["parent_key"] == saved["parent_key"] == fragment_check["parent_key"]
    report = diagnostic["residual_report"]
    fragments = report["fragments"]
    assert set(FRAGMENT_IDS) <= {fragment["raw_fragment_id"] for fragment in fragments}

    summary = {
        "schema": "saved-court-residual-accounting/1",
        "case_id": CASE_ID,
        "units": "working pixels; objective in squared working pixels",
        "source_note": "Postfit accounting of saved data; no references or new fits enter these values.",
        **comparison_summary(fragments, report["objectives_px2"]),
        "diagnostic_cases": diagnostic_cases(diagnostics),
        "scene34_original_movement": original_movement(saved, diagnostic),
    }
    payload = json.dumps(summary, indent=2, allow_nan=False).encode("utf-8")
    (HERE / "residual_summary.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    make_figure(fragments)


if __name__ == "__main__":
    main()
