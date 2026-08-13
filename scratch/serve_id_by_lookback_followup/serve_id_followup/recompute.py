"""Build the checked development results from the bundled frozen evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .rules import (
    Decision,
    other_side,
    paired_outcome,
    preferred_decision,
    rank1_sensitivity_decision,
    temporal_slot_is_correct,
)

KEY = ("fixture", "video_id", "set_id", "rally")
SOURCE_FILES = (
    "strict_outgoing_search_results.csv.gz",
    "strict_outgoing_search_summary.json.gz",
    "relaxed_contact_evidence.csv.gz",
    "relaxed_search_results.csv.gz",
    "relaxed_trajectory_summary.json.gz",
    "high_shot_correction_results.csv.gz",
    "high_shot_correction_summary.json.gz",
    "serve_setup_sensitivity_summary.json.gz",
)


def row_key(row: Mapping[str, str]) -> tuple[str, str, str, str]:
    return tuple(row[name] for name in KEY)  # type: ignore[return-value]


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gzip_json(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def write_gzip_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fields = list(rows[0])
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = text.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw_handle, mtime=0
    ) as zipped:
        zipped.write(raw)


def write_gzip_json(path: Path, value: object) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw_handle, mtime=0
    ) as zipped:
        zipped.write(raw)


def bool_text(value: bool) -> str:
    return "True" if value else "False"


def text_bool(value: str) -> bool:
    return value == "True"


def unique_index(
    rows: Iterable[dict[str, str]], name: str
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    index: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = row_key(row)
        if key in index:
            raise ValueError(f"duplicate {name} row for {key}")
        index[key] = row
    return index


def score_decision(
    search: Mapping[str, str],
    baseline: Mapping[str, str],
    decision: Decision,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    # The prediction is already frozen. Ground truth is read only below.
    gt_server = baseline["gt_server"]
    server_correct = decision.predicted_server == gt_server
    pr82_server_correct = baseline["baseline_server"] == gt_server
    temporal_correct = temporal_slot_is_correct(
        decision.temporal_claim, decision.temporal_gt_label_at_10
    )
    row: dict[str, object] = {
        **{name: search[name] for name in KEY},
        "prediction_branch": decision.branch,
        "predicted_server": decision.predicted_server,
        "claimed_frame": decision.claimed_frame,
        "temporal_claim": decision.temporal_claim,
        "temporal_gt_label_at_10": decision.temporal_gt_label_at_10,
        "pr82_server": baseline["baseline_server"],
        "gt_server": gt_server,
        "server_correct": bool_text(server_correct),
        "temporal_slot_correct": bool_text(temporal_correct),
        "joint_temporal_and_server_correct": bool_text(
            server_correct and temporal_correct
        ),
        "pr82_server_correct": bool_text(pr82_server_correct),
        "changed_vs_pr82": bool_text(
            decision.predicted_server != baseline["baseline_server"]
        ),
        "paired_outcome": paired_outcome(server_correct, pr82_server_correct),
    }
    if extra:
        row.update(extra)
    return row


def build_preferred_rows(
    searches: Sequence[dict[str, str]],
    baseline_by_key: Mapping[tuple[str, str, str, str], dict[str, str]],
) -> list[dict[str, object]]:
    return [
        score_decision(
            search,
            baseline_by_key[row_key(search)],
            preferred_decision(search, baseline_by_key[row_key(search)]),
        )
        for search in searches
    ]


def build_rank1_rows(
    searches: Sequence[dict[str, str]],
    baseline_by_key: Mapping[tuple[str, str, str, str], dict[str, str]],
    rank1_by_key: Mapping[tuple[str, str, str, str], dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for search in searches:
        key = row_key(search)
        baseline = baseline_by_key[key]
        rank1 = rank1_by_key[key]
        decision = rank1_sensitivity_decision(search, baseline, rank1)
        rows.append(
            score_decision(
                search,
                baseline,
                decision,
                {
                    "selected_outgoing_category": search["sequential_category"],
                    "rank1_player": rank1["player"],
                    "rank1_pre_verdict": rank1["pre_verdict"],
                    "rank1_pre_path_status": rank1["pre_path_status"],
                    "rank1_credible_outgoing": rank1["credible_outgoing"],
                },
            )
        )
    return rows


def exact_mcnemar_p(fixes: int, damages: int) -> float:
    changed = fixes + damages
    if changed == 0:
        return 1.0
    tail = min(fixes, damages)
    probability = 2.0 * sum(
        math.comb(changed, k) for k in range(tail + 1)
    ) / (2**changed)
    return min(1.0, probability)


def summarise_rule(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    paired = Counter(str(row["paired_outcome"]) for row in rows)
    branches: dict[str, dict[str, int]] = {}
    branch_names = sorted({str(row["prediction_branch"]) for row in rows})
    for branch in branch_names:
        group = [row for row in rows if row["prediction_branch"] == branch]
        branches[branch] = {
            "count": len(group),
            "server_correct": sum(
                text_bool(str(row["server_correct"])) for row in group
            ),
            "temporal_slot_correct": sum(
                text_bool(str(row["temporal_slot_correct"])) for row in group
            ),
            "joint_temporal_and_server_correct": sum(
                text_bool(str(row["joint_temporal_and_server_correct"]))
                for row in group
            ),
        }

    fixes = paired["fix"]
    damages = paired["damage"]
    return {
        "population": len(rows),
        "server_correct": sum(
            text_bool(str(row["server_correct"])) for row in rows
        ),
        "temporal_slot_correct": sum(
            text_bool(str(row["temporal_slot_correct"])) for row in rows
        ),
        "joint_temporal_and_server_correct": sum(
            text_bool(str(row["joint_temporal_and_server_correct"])) for row in rows
        ),
        "correct_visible_serve_frame_claims": sum(
            row["temporal_claim"] == "serve"
            and row["temporal_gt_label_at_10"] == "contact_1"
            for row in rows
        ),
        "changed_vs_pr82": sum(
            text_bool(str(row["changed_vs_pr82"])) for row in rows
        ),
        "fixes_vs_pr82": fixes,
        "damages_vs_pr82": damages,
        "paired_outcomes": dict(sorted(paired.items())),
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(fixes, damages),
        "branch_counts": branches,
    }


def eligible_min3(row: Mapping[str, str], prefix: str) -> bool:
    names = (
        f"{prefix}_n_frames",
        f"{prefix}_contact_gap",
        f"{prefix}_largest_step_ratio",
        f"{prefix}_fitted_decrease_bh",
    )
    if any(not row[name] for name in names):
        return False
    return (
        float(row[f"{prefix}_n_frames"]) >= 3
        and float(row[f"{prefix}_contact_gap"]) <= 2
        and float(row[f"{prefix}_largest_step_ratio"]) <= 8.0
    )


def minimum_path_3_score(
    contacts: Sequence[dict[str, str]],
    searches: Sequence[dict[str, str]],
    baseline_by_key: Mapping[tuple[str, str, str, str], dict[str, str]],
) -> tuple[int, dict[str, int]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for contact in contacts:
        groups[row_key(contact)].append(contact)
    for group in groups.values():
        group.sort(key=lambda row: int(row["accepted_rank"]))

    correct = 0
    categories: Counter[str] = Counter()
    for search in searches:
        key = row_key(search)
        direct_server: str | None = None
        category = "unresolved"
        for contact in groups[key]:
            if not eligible_min3(contact, "post"):
                continue
            if float(contact["post_fitted_decrease_bh"]) > -0.05:
                continue
            if eligible_min3(contact, "pre"):
                if float(contact["pre_fitted_decrease_bh"]) >= 0.05:
                    category = "first_visible_post_serve_contact"
                    direct_server = other_side(contact["player"])
                else:
                    category = "visible_serve"
                    direct_server = contact["player"]
            else:
                category = "pre_contact_unavailable"
            break

        categories[category] += 1
        baseline = baseline_by_key[key]
        final_server = direct_server or baseline["baseline_server"]
        correct += final_server == baseline["gt_server"]
    return correct, dict(sorted(categories.items()))


def direct_160_diagnostic(
    contacts: Sequence[dict[str, str]],
    searches: Sequence[dict[str, str]],
    baseline_by_key: Mapping[tuple[str, str, str, str], dict[str, str]],
) -> dict[str, object]:
    contact_by_rank = {
        (row_key(contact), contact["accepted_rank"]): contact for contact in contacts
    }
    source_counts: Counter[str] = Counter()
    source_correct: Counter[str] = Counter()

    for search in searches:
        predicted_server: str | None = None
        source: str | None = None
        category = search["sequential_category"]
        if category == "first_visible_post_serve_contact":
            predicted_server = other_side(search["sequential_selected_player"])
            source = "outgoing_selected"
        elif category == "visible_serve":
            predicted_server = search["sequential_selected_player"]
            source = "outgoing_selected"
        elif (
            category == "not_enough_shuttle_trajectory_to_tell"
            and search["sequential_selected_rank"] == "1"
        ):
            selected = contact_by_rank[(row_key(search), "1")]
            if selected["pre_path_status"] == "no_usable_run":
                predicted_server = search["sequential_selected_player"]
                source = "rank1_no_usable_run_default"

        if predicted_server is None or source is None:
            continue
        source_counts[source] += 1
        source_correct[source] += (
            predicted_server == baseline_by_key[row_key(search)]["gt_server"]
        )

    return {
        "coverage": sum(source_counts.values()),
        "correct_server_sides": sum(source_correct.values()),
        "sources": {
            name: {
                "count": source_counts[name],
                "correct": source_correct[name],
            }
            for name in sorted(source_counts)
        },
    }


def source_hashes(data_dir: Path) -> dict[str, dict[str, object]]:
    values: dict[str, dict[str, object]] = {}
    for name in SOURCE_FILES:
        raw = (data_dir / name).read_bytes()
        values[name] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return values


def build_outputs(data_dir: Path, output_dir: Path) -> dict[str, object]:
    missing = [name for name in SOURCE_FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing source files: {', '.join(missing)}")

    contacts = read_gzip_csv(data_dir / "relaxed_contact_evidence.csv.gz")
    searches = read_gzip_csv(data_dir / "relaxed_search_results.csv.gz")
    baselines = read_gzip_csv(data_dir / "high_shot_correction_results.csv.gz")
    baseline_by_key = unique_index(baselines, "baseline")
    search_by_key = unique_index(searches, "search")

    rank1_rows = [row for row in contacts if row["accepted_rank"] == "1"]
    rank1_by_key = unique_index(rank1_rows, "rank-1 contact")
    if len(searches) != 239 or len(baselines) != 239 or len(rank1_rows) != 239:
        raise ValueError("the frozen one-to-one population must contain 239 rows")
    if set(search_by_key) != set(baseline_by_key) or set(rank1_by_key) != set(
        baseline_by_key
    ):
        raise ValueError("source key sets do not match")

    searches = sorted(searches, key=row_key)
    preferred_rows = build_preferred_rows(searches, baseline_by_key)
    rank1_sensitivity_rows = build_rank1_rows(
        searches, baseline_by_key, rank1_by_key
    )

    strict = read_gzip_json(data_dir / "strict_outgoing_search_summary.json.gz")
    relaxed = read_gzip_json(data_dir / "relaxed_trajectory_summary.json.gz")
    additive = read_gzip_json(data_dir / "high_shot_correction_summary.json.gz")
    setup = read_gzip_json(data_dir / "serve_setup_sensitivity_summary.json.gz")

    min3_correct, min3_categories = minimum_path_3_score(
        contacts, searches, baseline_by_key
    )
    direct160 = direct_160_diagnostic(contacts, searches, baseline_by_key)
    contact_by_rank = {
        (row_key(contact), contact["accepted_rank"]): contact for contact in contacts
    }
    no_usable_run_rows = 0
    for search in searches:
        if search["sequential_category"] != "not_enough_shuttle_trajectory_to_tell":
            continue
        selected = contact_by_rank[
            (row_key(search), search["sequential_selected_rank"])
        ]
        no_usable_run_rows += selected["pre_path_status"] == "no_usable_run"

    baseline_server_correct = sum(
        baseline["baseline_server"] == baseline["gt_server"] for baseline in baselines
    )
    baseline_start_correct = sum(
        baseline["baseline_start_correct"] == "True" for baseline in baselines
    )
    baseline_joint = sum(
        baseline["baseline_start_correct"] == "True"
        and baseline["baseline_server"] == baseline["gt_server"]
        for baseline in baselines
    )

    metrics: dict[str, object] = {
        "schema": "serve_id_followup_development_metrics/2",
        "population": 239,
        "scope_note": (
            "All scored rules use the fixed 239 one-to-one rally population. "
            "Their branches do not read ground-truth labels. Rule composition was "
            "chosen after development-set review."
        ),
        "pr82_baseline_from_frozen_table": {
            "server_correct": baseline_server_correct,
            "visible_start_correct": baseline_start_correct,
            "joint_visible_start_and_server_correct": baseline_joint,
        },
        "original_followup": {
            "strict_outgoing_first": {
                "fixed_rule": strict["fixed_rule"],
                "categories": strict["opener_categories"],
                "tolerance_10": strict["tolerances"]["10"],
            },
            "relaxed_trajectory_evidence": {
                "fixed_rule": relaxed["fixed_rule"],
                "pre_verdicts_over_3200_contacts": relaxed["pre_verdicts"],
                "pre_path_statuses_over_3200_contacts": relaxed[
                    "pre_path_statuses"
                ],
                "sequential_categories": relaxed["sequential_categories"],
                "sequential_tolerance_10": relaxed["tolerances"]["10"][
                    "sequential"
                ],
                "incoming_predecessor_categories": relaxed[
                    "incoming_categories"
                ],
                "incoming_admissions": relaxed["incoming_admissions"],
                "incoming_tolerance_10": relaxed["tolerances"]["10"][
                    "incoming"
                ],
            },
            "narrow_high_shot_correction": {
                "headline": additive["headline"],
                "server_attribution": additive["server_attribution"],
                "high_shot_evidence_states": additive[
                    "high_shot_evidence_states"
                ],
                "rejected_integrated_candidates": additive[
                    "rejected_integrated_candidates"
                ],
                "rejected_relaxed_same_anchor_incoming": additive[
                    "rejected_relaxed_same_anchor_incoming"
                ],
            },
            "serve_setup_sensitivities": setup["variants"],
        },
        "preferred_server_rule": {
            "description": (
                "Use the selected contact when the shuttle path clearly shows whether "
                "the contact looks like a serve or a return. Otherwise keep the PR #82 answer."
            ),
            **summarise_rule(preferred_rows),
        },
        "rank1_fallback_sensitivity": {
            "description": (
                "Use the same classifiable branches, then infer from rank 1 instead "
                "of keeping the PR #82 fallback."
            ),
            **summarise_rule(rank1_sensitivity_rows),
        },
        "diagnostics": {
            "direct_160": direct160,
            "minimum_path_3_final_correct": min3_correct,
            "minimum_path_3_categories": min3_categories,
            "selected_rows_with_no_usable_pre_run": no_usable_run_rows,
            "selected_apparent_return_rows": sum(
                search["sequential_category"]
                == "first_visible_post_serve_contact"
                for search in searches
            ),
        },
        "source_files": source_hashes(data_dir),
    }

    write_gzip_csv(output_dir / "preferred_server_rule.csv.gz", preferred_rows)
    write_gzip_csv(
        output_dir / "rank1_fallback_sensitivity.csv.gz",
        rank1_sensitivity_rows,
    )
    write_gzip_json(output_dir / "development_metrics.json.gz", metrics)
    return metrics


def compare_outputs(expected: Path, actual: Path) -> list[str]:
    names = (
        "development_metrics.json.gz",
        "preferred_server_rule.csv.gz",
        "rank1_fallback_sensitivity.csv.gz",
    )
    return [name for name in names if (expected / name).read_bytes() != (actual / name).read_bytes()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute in a temporary directory and compare with committed results",
    )
    args = parser.parse_args()

    if args.check:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            metrics = build_outputs(args.data_dir, temporary)
            differences = compare_outputs(args.output_dir, temporary)
        if differences:
            raise SystemExit("result files differ: " + ", ".join(differences))
        print("Committed result files match a fresh recomputation.")
    else:
        metrics = build_outputs(args.data_dir, args.output_dir)

    preferred = metrics["preferred_server_rule"]
    sensitivity = metrics["rank1_fallback_sensitivity"]
    print(f"PR #82 server: {metrics['pr82_baseline_from_frozen_table']['server_correct']}/239")
    print(f"Preferred rule: {preferred['server_correct']}/239")
    print(f"Rank-1 sensitivity: {sensitivity['server_correct']}/239")


if __name__ == "__main__":
    main()
