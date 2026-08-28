"""Regenerate the PR 80 follow-up figures from retained compact summaries."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

EXPECTED_SCHEMAS = {
    "1_scene_comparison.json.gz": "vlm-followup-scene-comparison-v1",
    "2_final_model_gate.json.gz": "vlm-followup2-final-model-gate-v1",
    "3_precision_first_dataset.json.gz": "vlm-precision-first-score/0.1",
    "4_serve_reconstruction.json.gz": "vlm-followup4-serve-reconstruction-result-v1",
    "5_pr88_serve_lookback.json.gz": "vlm-followup5-pr88-reconciliation-result-v1",
    "6_rally_opening_context.json.gz": "vlm-followup-6-rally-opening-context/1.0",
}


def load_summary(results_dir: Path, filename: str) -> dict[str, Any]:
    path = results_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing retained summary: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    expected = EXPECTED_SCHEMAS[filename]
    actual = data.get("schema")
    if actual != expected:
        raise ValueError(f"{filename}: expected schema {expected!r}, found {actual!r}")
    return data


def percent(correct: int, total: int) -> float:
    if total <= 0:
        raise ValueError("Total must be positive")
    return 100.0 * correct / total


def label_bars(ax: Any, bars: Any, labels: list[str], *, horizontal: bool = False) -> None:
    for bar, label in zip(bars, labels, strict=True):
        if horizontal:
            ax.text(
                bar.get_width() + 1.0,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=9,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )


def save(fig: Any, output_dir: Path, filename: str, *, bottom: float = 0.16) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def scene_routing(data: dict[str, Any], output_dir: Path) -> None:
    groups = [
        ("Standard live kept", "standard_view_live", "kept"),
        ("Unusual live kept", "unusual_view_live", "kept"),
        ("Non-live sent for checking", "targets_containing_nonlive", "sent_to_further_check"),
        ("Pure replay sent for checking", "pure_replay", "sent_to_further_check"),
    ]
    qwen = data["material_results"]["qwen"]
    intern = data["material_results"]["intern"]

    q_values, i_values, q_labels, i_labels = [], [], [], []
    for _, key, count_key in groups:
        q_num, q_den = qwen[key][count_key], qwen[key]["total"]
        i_num, i_den = intern[key][count_key], intern[key]["total"]
        q_values.append(percent(q_num, q_den))
        i_values.append(percent(i_num, i_den))
        q_labels.append(f"{q_num}/{q_den}")
        i_labels.append(f"{i_num}/{i_den}")

    y = np.arange(len(groups))
    height = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    q_bars = ax.barh(y - height / 2, q_values, height, label="Qwen")
    i_bars = ax.barh(y + height / 2, i_values, height, label="Intern")
    ax.set_yticks(y, [item[0] for item in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("Targets routed correctly for the intended role (%)")
    ax.set_ylabel("Scene target group")
    ax.set_title("Short local scene clips: neither model was a safe final filter")
    ax.legend()
    label_bars(ax, q_bars, q_labels, horizontal=True)
    label_bars(ax, i_bars, i_labels, horizontal=True)
    fig.text(
        0.5,
        0.035,
        "Higher is better in every row. Main comparison: 347 material targets; pure-replay subset: 25.",
        ha="center",
        fontsize=9,
    )
    save(fig, output_dir, "scene_routing.png", bottom=0.13)


def clean_serve_gate(data: dict[str, Any], output_dir: Path) -> None:
    scores = data["scores"]
    categories = ["Server correct", "Visible contact within tolerance"]
    intern_values = [
        percent(scores["internvideo3"]["server_correct"], scores["internvideo3"]["server_total"]),
        percent(
            scores["internvideo3"]["visible_contact_within_project_tolerance"],
            scores["internvideo3"]["visible_contact_total"],
        ),
    ]
    qwen_values = [
        percent(scores["qwen3-vl"]["server_correct"], scores["qwen3-vl"]["server_total"]),
        percent(
            scores["qwen3-vl"]["visible_contact_within_project_tolerance"],
            scores["qwen3-vl"]["visible_contact_total"],
        ),
    ]
    intern_labels = ["23/32", "1/19"]
    qwen_labels = ["14/32", "1/19"]

    x = np.arange(len(categories))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    i_bars = ax.bar(x - width / 2, intern_values, width, label="Intern")
    q_bars = ax.bar(x + width / 2, qwen_values, width, label="Qwen")
    ax.set_xticks(x, categories)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Output field")
    ax.set_ylabel("Correct cases (%)")
    ax.set_title("Reviewed rally starts: Intern was better on server ID; timing failed")
    ax.legend()
    label_bars(ax, i_bars, intern_labels)
    label_bars(ax, q_bars, qwen_labels)
    fig.text(
        0.5,
        0.035,
        "Both models answered 'visible' in all 32 cases and claimed an exact frame in all 13 non-visible cases.",
        ha="center",
        fontsize=9,
    )
    save(fig, output_dir, "clean_serve_gate.png", bottom=0.13)


def precision_first(data: dict[str, Any], output_dir: Path) -> None:
    primary = data["aggregate_by_tolerance"]["5"]
    all_total = primary["retained_records"] + primary["rejected_records"]
    all_correct = primary["baseline_correct_records"]
    strict_scores = data["primary_rule_scores_by_video"]
    strict_total = sum(v["outcome-corroboration"]["retained_records"] for v in strict_scores.values())
    strict_correct = sum(
        v["outcome-corroboration"]["correct_complete_records"] for v in strict_scores.values()
    )

    totals = np.array([all_total, strict_total], dtype=float)
    correct = np.array([all_correct, strict_correct], dtype=float)
    wrong = totals - correct
    correct_pct = 100.0 * correct / totals
    wrong_pct = 100.0 * wrong / totals

    x = np.arange(2)
    labels = ["All current predictions\n(311)", "Strictest rule\n(7)"]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.bar(x, correct_pct, label="Complete record")
    ax.bar(x, wrong_pct, bottom=correct_pct, label="Wrong or incomplete")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Population being inspected")
    ax.set_ylabel("Share of records (%)")
    ax.set_title("Strict filtering could not find a trustworthy rally subset")
    ax.legend()
    ax.text(x[0], 3, "1 complete\n310 not complete", ha="center", va="bottom", fontsize=9)
    ax.text(x[1], 17, "1 complete\n6 wrong", ha="center", va="bottom", fontsize=9)
    fig.text(
        0.5,
        0.035,
        "No rule was error-free on the development fixtures, so none was tried on the set-aside fixture.",
        ha="center",
        fontsize=9,
    )
    save(fig, output_dir, "precision_first.png", bottom=0.13)


def serve_support(data: dict[str, Any], output_dir: Path) -> None:
    baseline = data["baseline_reference"]
    observations = data["arms"]["observations"]["result"]
    proposals = data["arms"]["observations_plus_proposals"]["result"]

    names = ["Plain prompt", "Automatic observations", "Observations\n+ current proposals"]
    server = [
        percent(baseline["server_correct"], 32),
        percent(observations["server"]["correct"], observations["server"]["total"]),
        percent(proposals["server"]["correct"], proposals["server"]["total"]),
    ]
    timing = [
        percent(baseline["contact_within_project_tolerance"], 19),
        percent(
            observations["contact_timing_visible_truth"]["within_project_tolerance"],
            observations["contact_timing_visible_truth"]["truth_cases"],
        ),
        percent(
            proposals["contact_timing_visible_truth"]["within_project_tolerance"],
            proposals["contact_timing_visible_truth"]["truth_cases"],
        ),
    ]
    server_labels = ["23/32", "23/32", "18/32"]
    timing_labels = ["1/19", "9/19", "2/19"]

    x = np.arange(len(names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 6))
    s_bars = ax.bar(x - width / 2, server, width, label="Server correct (n=32)")
    t_bars = ax.bar(x + width / 2, timing, width, label="Contact timing close enough (n=19 visible contacts)")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Prompt version")
    ax.set_ylabel("Correct cases (%)")
    ax.set_title("The apparent timing gain came from repeating the supplied candidate frame")
    ax.legend()
    label_bars(ax, s_bars, server_labels)
    label_bars(ax, t_bars, timing_labels)
    fig.text(
        0.5,
        0.035,
        "With automatic observations, 30/31 parsed replies repeated the candidate contact frame named in the prompt.",
        ha="center",
        fontsize=9,
    )
    save(fig, output_dir, "serve_support.png", bottom=0.13)


def pr88_development(data: dict[str, Any], output_dir: Path) -> None:
    baseline = data["pr88"]["baseline"]
    rule = data["pr88"]["preferred_rule"]
    categories = ["Server", "First visible stroke", "Both correct"]
    baseline_counts = [
        baseline["server_correct"],
        baseline["visible_start_correct"],
        baseline["joint_correct"],
    ]
    rule_counts = [
        rule["server_correct"],
        rule["visible_start_correct"],
        rule["joint_correct"],
    ]
    total = data["pr88"]["population"]["one_to_one_rallies"]
    baseline_values = [percent(v, total) for v in baseline_counts]
    rule_values = [percent(v, total) for v in rule_counts]

    x = np.arange(len(categories))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 6))
    b_bars = ax.bar(x - width / 2, baseline_values, width, label="PR 82")
    r_bars = ax.bar(x + width / 2, rule_values, width, label="PR 88")
    ax.set_xticks(x, categories)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Measure")
    ax.set_ylabel("Correct rallies (%)")
    ax.set_title("PR 88 improved the development result; unseen validation is still needed")
    ax.legend()
    label_bars(ax, b_bars, [f"{v}/{total}" for v in baseline_counts])
    label_bars(ax, r_bars, [f"{v}/{total}" for v in rule_counts])
    fig.text(
        0.5,
        0.035,
        "Development only. Server changes: 20 repairs, 13 damages; exact paired two-sided p = 0.296.",
        ha="center",
        fontsize=9,
    )
    save(fig, output_dir, "pr88_development.png", bottom=0.13)


def rally_opening(data: dict[str, Any], output_dir: Path) -> None:
    results = data["results"]
    arms = [
        ("Every second frame", results["clean_half_native"]),
        ("Every second frame\n+ timing hint", results["cued_half_native"]),
        ("Every frame\n+ timing hint", results["cued_native"]),
    ]
    values = [percent(v["correct"], v["total"]) for _, v in arms]
    labels = [f'{v["correct"]}/{v["total"]}' for _, v in arms]

    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    bars = ax.bar(x, values)
    ax.set_xticks(x, [name for name, _ in arms])
    ax.set_ylim(0, 100)
    ax.set_xlabel("22-second input version")
    ax.set_ylabel("Correct server answers (%)")
    ax.set_title("The clean 22-second input matched the earlier 8/12 total;\nthe timing hint and extra frames did not improve it")
    label_bars(ax, bars, labels)
    fig.text(
        0.5,
        0.035,
        "Several variables differed from the earlier short-clip setup, so this does not isolate clip duration as the cause.",
        ha="center",
        fontsize=9,
    )
    save(fig, output_dir, "rally_opening.png", bottom=0.13)


def auto_annotator_status(
    scene: dict[str, Any],
    gate: dict[str, Any],
    precision_data: dict[str, Any],
    support: dict[str, Any],
    pr88: dict[str, Any],
    output_dir: Path,
) -> None:
    primary = precision_data["aggregate_by_tolerance"]["5"]
    record_total = primary["retained_records"] + primary["rejected_records"]
    complete_records = primary["baseline_correct_records"]
    retained_records = primary["retained_records"]

    qwen_replay = scene["material_results"]["qwen"]["pure_replay"]
    intern_replay = scene["material_results"]["intern"]["pure_replay"]

    intern_gate = gate["scores"]["internvideo3"]
    qwen_gate = gate["scores"]["qwen3-vl"]
    false_frame_text = (
        f"Intern {intern_gate['false_exact_frames_on_nonvisible_cases']}/"
        f"{intern_gate['nonvisible_cases']}; Qwen "
        f"{qwen_gate['false_exact_frames_on_nonvisible_cases']}/"
        f"{qwen_gate['nonvisible_cases']}"
    )

    observation_rows = support["arms"]["observations"]["result"]["rows"]
    parsed_rows = [
        row for row in observation_rows if row.get("predicted_clip_contact_frame") is not None
    ]
    copied_points = sum(
        row["predicted_clip_contact_frame"] in {40, 80} for row in parsed_rows
    )
    parsed_answers = len(parsed_rows)

    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.axis("off")
    ax.set_title(
        "Branch status at close: useful diagnosis, but no trustworthy automatic annotator yet",
        fontsize=15,
        pad=20,
    )

    replay_summary = (
        f"Qwen accepted {qwen_replay['accepted_as_live']}/{qwen_replay['total']} pure replays; "
        f"Intern accepted {intern_replay['accepted_as_live']}/{intern_replay['total']}"
    )
    lines = [
        (
            "Completely correct predicted rally spans",
            f"{complete_records} of {record_total} at the main timing tolerance",
        ),
        (
            "Strict filtering on set-aside data",
            f"None qualified; {retained_records} of {record_total} retained",
        ),
        ("Replay filtering", replay_summary),
        ("When serve contact was not visible", f"Exact frames claimed on non-visible cases: {false_frame_text}"),
        (
            "Apparent timing improvement",
            f"{copied_points}/{parsed_answers} parsed answers repeated the candidate frame named in the prompt",
        ),
        ("Historical branch-close next test", "PR 88 unchanged on unseen rallies"),
    ]

    y = 0.83
    for heading, value in lines:
        ax.text(0.04, y, heading, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.text(0.43, y, value, transform=ax.transAxes, fontsize=11, va="top", wrap=True)
        y -= 0.115

    ax.text(
        0.04,
        0.08,
        "Bottom line",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.18,
        0.08,
        "Cross-broadcast generalisation remained unproven; the scene benchmark exposed convention sensitivity, "
        "and the branch did not produce a trustworthy standalone automatic annotator.",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        wrap=True,
    )
    fig.text(
        0.5,
        0.02,
        "Source: the six retained compact follow-up summaries on branch vlm-pr80-followups.",
        ha="center",
        fontsize=9,
    )
    save(fig, output_dir, "auto_annotator_status.png", bottom=0.08)


def parse_args() -> argparse.Namespace:
    evaluation_root = Path(__file__).resolve().parents[1]
    default_results = evaluation_root / "followups" / "evidence"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results,
        help="Directory containing the six compact .json.gz summaries",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=evaluation_root / "figures",
        help="Directory to receive PNG figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {
        filename: load_summary(args.results_dir, filename) for filename in EXPECTED_SCHEMAS
    }

    scene_routing(summaries["1_scene_comparison.json.gz"], args.output_dir)
    clean_serve_gate(summaries["2_final_model_gate.json.gz"], args.output_dir)
    precision_first(summaries["3_precision_first_dataset.json.gz"], args.output_dir)
    serve_support(summaries["4_serve_reconstruction.json.gz"], args.output_dir)
    pr88_development(summaries["5_pr88_serve_lookback.json.gz"], args.output_dir)
    rally_opening(summaries["6_rally_opening_context.json.gz"], args.output_dir)
    auto_annotator_status(
        summaries["1_scene_comparison.json.gz"],
        summaries["2_final_model_gate.json.gz"],
        summaries["3_precision_first_dataset.json.gz"],
        summaries["4_serve_reconstruction.json.gz"],
        summaries["5_pr88_serve_lookback.json.gz"],
        args.output_dir,
    )


if __name__ == "__main__":
    main()
