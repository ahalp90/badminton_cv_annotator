"""Measure simple two-model keep rules on completed event attempts."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .score_trials import _load_attempts, _load_truth
from .trial_schema import TrialArm

BACKENDS = ("qwen3-vl", "internvideo3")


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.96
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return [centre - margin, centre + margin]


def decision_metrics(decisions: list[bool], truth: list[bool]) -> dict[str, Any]:
    """Score one automatic keep rule, including 95% Wilson intervals."""
    kept = sum(decisions)
    positives = sum(truth)
    true_kept = sum(keep and expected for keep, expected in zip(decisions, truth))
    false_kept = kept - true_kept
    return {
        "cases": len(decisions),
        "kept": kept,
        "truth_positive": positives,
        "true_kept": true_kept,
        "false_kept": false_kept,
        "precision": _ratio(true_kept, kept),
        "precision_95ci": _wilson_interval(true_kept, kept),
        "recall": _ratio(true_kept, positives),
        "recall_95ci": _wilson_interval(true_kept, positives),
    }


def analyse_event_consensus(truth_path: Path, attempts_root: Path) -> dict[str, Any]:
    truth_by_id = _load_truth(truth_path)
    event_truth = {
        case_id: row for case_id, row in truth_by_id.items() if row["kind"] == "event"
    }
    attempts = _load_attempts(attempts_root)
    rules: dict[str, Callable[[dict[str, str]], bool]] = {
        "qwen_yes": lambda answers: answers["qwen3-vl"] == "yes",
        "intern_yes": lambda answers: answers["internvideo3"] == "yes",
        "both_yes": lambda answers: all(answer == "yes" for answer in answers.values()),
        "either_yes": lambda answers: any(
            answer == "yes" for answer in answers.values()
        ),
    }
    results: list[dict[str, Any]] = []
    for arm in TrialArm:
        answers_by_case: dict[str, dict[str, str]] = {}
        for case_id in event_truth:
            answers: dict[str, str] = {}
            for backend in BACKENDS:
                attempt = attempts[(backend, case_id, arm.value)]
                parsed = attempt["parsed_response"]
                answers[backend] = (
                    "invalid" if parsed is None else parsed["contact_at_marker"]
                )
            answers_by_case[case_id] = answers

        for role in ("all", "later-stroke", "serve"):
            case_ids = [
                case_id
                for case_id, row in event_truth.items()
                if role == "all" or row["event_role"] == role
            ]
            for tolerance in (10, 15):
                truth = [
                    bool(event_truth[case_id][f"usable_at_{tolerance}"])
                    for case_id in case_ids
                ]
                for rule_name, rule in rules.items():
                    decisions = [rule(answers_by_case[case_id]) for case_id in case_ids]
                    results.append(
                        {
                            "arm": arm.value,
                            "role": role,
                            "tolerance": tolerance,
                            "rule": rule_name,
                            **decision_metrics(decisions, truth),
                        }
                    )
    return {
        "schema": "vlm-cleanup-consensus/0.1",
        "truth": str(truth_path),
        "attempts_root": str(attempts_root),
        "backends": list(BACKENDS),
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = analyse_event_consensus(args.truth, args.attempts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
