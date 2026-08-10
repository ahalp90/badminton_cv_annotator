"""Independently recalculate the report from its compressed result tables."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RUN_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

PATH_DEFINITIONS = ("recurrence_clean", "producer_original")
NET_CLOSURES_BH = (0.0, 0.25, 0.5, 1.0)
MOVEMENT_PERCENTAGES = tuple(range(0, 101, 5))
PRIMARY_PATH_DEFINITION = "recurrence_clean"
PRIMARY_NET_CLOSURE_BH = 0.25

METHOD_COLUMNS = {
    "old alternating fit": "baseline_server",
    "use the anchor player": "assume_first_contact_is_serve",
    "use the other player when motion is incoming": "motion_rule_server",
    "motion answer only; abstain without a usable path": "evidence_only_server",
    "prepend a contact with unknown player": "missing_contact_refit_server",
    "prepend a contact by the other player": "inferred_player_refit_server",
}


def _load_metrics() -> dict[str, object]:
    """Load the compressed metrics object."""
    with gzip.open(OUTPUT_DIR / "metrics.json.gz", "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("metrics.json.gz must contain one object")
    return payload


def _assert_equal(actual: object, expected: object, label: str) -> None:
    """Compare ordinary values while allowing harmless floating-point rounding."""
    if isinstance(actual, float) or isinstance(expected, float):
        if not np.isclose(float(actual), float(expected)):
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def _confusion(truth: np.ndarray, predicted: np.ndarray) -> dict[str, int | float]:
    """Calculate binary counts and scores without importing experiment code."""
    true_positive = int(np.count_nonzero(predicted & truth))
    false_positive = int(np.count_nonzero(predicted & ~truth))
    false_negative = int(np.count_nonzero(~predicted & truth))
    true_negative = int(np.count_nonzero(~predicted & ~truth))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0
    )
    recall = true_positive / (true_positive + false_negative)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _recalculate_thresholds(rallies: pd.DataFrame) -> pd.DataFrame:
    """Build every threshold row directly from the per-rally measurements."""
    clear = rallies[rallies["anchor_gt_match"].isin(["contact_1", "contact_2"])]
    truth = clear["anchor_gt_match"].eq("contact_2").to_numpy()
    rows: list[dict[str, object]] = []
    for path_definition in PATH_DEFINITIONS:
        quality = clear[f"{path_definition}_path_quality_pass"].astype(bool).to_numpy()
        closure = clear[f"{path_definition}_net_closure_bh"].to_numpy(dtype=float)
        movement_share = clear[
            f"{path_definition}_movements_towards_player"
        ].to_numpy(dtype=float)
        for minimum_closure in NET_CLOSURES_BH:
            for percentage in MOVEMENT_PERCENTAGES:
                predicted = (
                    quality
                    & (closure >= minimum_closure)
                    & (movement_share >= percentage / 100.0)
                )
                rows.append(
                    {
                        "path_definition": path_definition,
                        "minimum_net_closure_bh": minimum_closure,
                        "minimum_movements_towards_player_percent": percentage,
                        "clear_contact_1": int(np.count_nonzero(~truth)),
                        "clear_contact_2": int(np.count_nonzero(truth)),
                        "paths_passing_quality_checks": int(np.count_nonzero(quality)),
                        **_confusion(truth, predicted),
                    }
                )
    return pd.DataFrame(rows)


def _validate_thresholds(
    rallies: pd.DataFrame,
    saved: pd.DataFrame,
    metrics: dict[str, object],
) -> None:
    """Check every saved threshold and both selected settings."""
    recalculated = _recalculate_thresholds(rallies)
    key = [
        "path_definition",
        "minimum_net_closure_bh",
        "minimum_movements_towards_player_percent",
    ]
    saved = saved.sort_values(key).reset_index(drop=True)
    recalculated = recalculated.sort_values(key).reset_index(drop=True)
    if list(saved.columns) != list(recalculated.columns):
        raise AssertionError("threshold columns differ from the independent calculation")
    for column in saved.columns:
        if pd.api.types.is_numeric_dtype(saved[column]):
            if not np.allclose(saved[column], recalculated[column]):
                raise AssertionError(f"saved threshold values differ in {column}")
        elif not saved[column].equals(recalculated[column]):
            raise AssertionError(f"saved threshold values differ in {column}")

    for path_definition, metric_key in (
        (PRIMARY_PATH_DEFINITION, "first_return_test"),
        ("producer_original", "producer_original_comparison"),
    ):
        curve = recalculated[
            (recalculated["path_definition"] == path_definition)
            & np.isclose(recalculated["minimum_net_closure_bh"], PRIMARY_NET_CLOSURE_BH)
        ]
        selected = curve.sort_values(
            ["f1", "precision", "minimum_movements_towards_player_percent"],
            ascending=[False, False, False],
        ).iloc[0]
        reported = metrics[metric_key]
        if metric_key == "first_return_test":
            reported_percentage = metrics["selected_rule"][
                "minimum_movements_towards_player_percent"
            ]
        else:
            reported_percentage = reported["minimum_movements_towards_player_percent"]
        _assert_equal(
            int(selected["minimum_movements_towards_player_percent"]),
            reported_percentage,
            f"{path_definition} selected movement percentage",
        )
        for field in ("tp", "fp", "fn", "precision", "recall", "f1"):
            _assert_equal(selected[field], reported[field], f"{path_definition} {field}")


def _class_score(truth: np.ndarray, predictions: np.ndarray, label: str) -> float:
    """Return one class F1 with unknown predictions counted as misses."""
    true_positive = int(np.count_nonzero((predictions == label) & (truth == label)))
    false_positive = int(np.count_nonzero((predictions == label) & (truth != label)))
    false_negative = int(np.count_nonzero((predictions != label) & (truth == label)))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def _method_score(frame: pd.DataFrame, column: str) -> dict[str, int | float]:
    """Recalculate the fields printed in one server table row."""
    truth = frame["gt_server"].astype(str).to_numpy()
    predictions = frame[column].fillna("Unknown").astype(str).to_numpy()
    return {
        "n": len(frame),
        "known": int(np.count_nonzero(predictions != "Unknown")),
        "correct": int(np.count_nonzero(predictions == truth)),
        "accuracy": float(np.mean(predictions == truth)),
        "macro_f1": float(
            np.mean([_class_score(truth, predictions, label) for label in ("Top", "Bot")])
        ),
    }


def _validate_method_scores(rallies: pd.DataFrame, metrics: dict[str, object]) -> None:
    """Recalculate every displayed server table without shared scoring code."""
    covered = rallies[rallies["boundary"] == "covered"]
    groups = {
        "all_292": rallies,
        "covered": covered,
        "frozen_failures": covered[covered["frozen_server_failure"].astype(bool)],
    }
    for group_name, frame in groups.items():
        for method, column in METHOD_COLUMNS.items():
            recalculated = _method_score(frame, column)
            reported = metrics["scores"][group_name][method]
            for field, value in recalculated.items():
                _assert_equal(value, reported[field], f"{group_name} {method} {field}")


def _parse_report_table(report: str, heading: str) -> dict[str, dict[str, str]]:
    """Read one generated Markdown score table into named cells."""
    section = report.split(heading, maxsplit=1)
    if len(section) != 2:
        raise AssertionError(f"report heading missing: {heading}")
    lines = section[1].lstrip().splitlines()
    if not lines or not lines[0].startswith("| Method |"):
        raise AssertionError(f"score table missing after {heading}")
    table: dict[str, dict[str, str]] = {}
    for line in lines[2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        method, correct, known, accuracy, macro_f1 = cells
        table[method] = {
            "correct": correct,
            "known": known,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }
    return table


def _validate_report(metrics: dict[str, object]) -> None:
    """Check the headline and every displayed server table against metrics."""
    report = (RUN_DIR / "report.md").read_text(encoding="utf-8")
    first_return = metrics["first_return_test"]
    headline = (
        f"{first_return['tp']} calls were right, {first_return['fp']} were false calls, "
        f"and {first_return['fn']} returns were missed."
    )
    if headline not in report:
        raise AssertionError("report headline first-return counts do not match metrics")

    counts = metrics["counts"]
    headings = {
        "all_292": f"### All {counts['all_gt_rallies']} ShuttleSet rallies",
        "covered": f"### {counts['covered_rallies']} rallies covered by one predicted span",
        "frozen_failures": (
            f"### {counts['frozen_server_failures']} rallies where the released fit was wrong or missing"
        ),
    }
    for group_name, heading in headings.items():
        parsed = _parse_report_table(report, heading)
        for method, values in metrics["scores"][group_name].items():
            if method not in parsed:
                raise AssertionError(f"report table is missing {group_name} row {method}")
            expected = {
                "correct": f"{values['correct']}/{values['n']}",
                "known": f"{values['known']}/{values['n']}",
                "accuracy": f"{values['accuracy']:.1%}",
                "macro_f1": f"{values['macro_f1']:.3f}",
            }
            _assert_equal(parsed[method], expected, f"report {group_name} {method}")


def validate() -> None:
    """Validate result rows, threshold choices, report tables and plots."""
    rallies = pd.read_csv(OUTPUT_DIR / "rallies.csv.gz")
    thresholds = pd.read_csv(OUTPUT_DIR / "thresholds.csv.gz")
    metrics = _load_metrics()
    if len(rallies) != 292 or rallies[["video_id", "set_id", "rally"]].duplicated().any():
        raise AssertionError("rallies.csv.gz must contain 292 unique ShuttleSet rallies")
    _validate_thresholds(rallies, thresholds, metrics)
    _validate_method_scores(rallies, metrics)
    _validate_report(metrics)

    required_plots = (
        "first_return_threshold.png",
        "incoming_motion_measurements.png",
        "server_accuracy.png",
        "tracknet_source_comparison.png",
    )
    for name in required_plots:
        if not (PLOT_DIR / name).is_file():
            raise FileNotFoundError(PLOT_DIR / name)
    case_count = len(list((PLOT_DIR / "cases").glob("*.png")))
    _assert_equal(case_count, metrics["counts"]["case_plots"], "case plot count")
    print("validated all threshold rows, server tables, headline counts and plots")


if __name__ == "__main__":
    validate()
