"""Score immutable VLM attempts against the separate cleanup truth sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .trial_schema import (
    ATTEMPT_SCHEMA,
    TrialArm,
    TrialKind,
    load_manifest,
    parse_reply,
)

TRUTH_SCHEMA = "vlm-cleanup-truth/0.1"
SCORE_SCHEMA = "vlm-cleanup-score/0.1"
DEFAULT_EXPECTED_BACKENDS = ("qwen3-vl", "internvideo3")

_EVENT_TRUTH_KEYS = {
    "case_id",
    "kind",
    "video_id",
    "stratum",
    "nearest_gt_frame",
    "distance_to_gt_base30",
    "usable_at_5",
    "usable_at_10",
    "usable_at_15",
    "expected_actor",
    "set_id",
    "rally",
    "ball_round",
    "event_role",
}
_BROADCAST_TRUTH_KEYS = {
    "case_id",
    "kind",
    "video_id",
    "valid_rally",
    "dominant_scene_truth",
    "scene_fractions",
    "boundary_class",
    "mapped_set_id",
    "mapped_rally",
}
_TRACK_TRUTH_KEYS = {
    "case_id",
    "kind",
    "video_id",
    "tracker_real",
    "truth_source",
    "sample_id",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_truth(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema", "cases"}:
        raise ValueError("truth sidecar must contain exactly schema and cases")
    if payload["schema"] != TRUTH_SCHEMA:
        raise ValueError(f"unsupported truth schema {payload['schema']!r}")
    if not isinstance(payload["cases"], list):
        raise TypeError("truth cases must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for row in payload["cases"]:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise TypeError("every truth case must be an object with a string case_id")
        try:
            kind = TrialKind(row.get("kind"))
        except ValueError as exc:
            raise ValueError(f"invalid truth kind for {row['case_id']!r}") from exc
        if kind is TrialKind.EVENT:
            expected_keys = _EVENT_TRUTH_KEYS
        elif kind is TrialKind.BROADCAST:
            expected_keys = _BROADCAST_TRUTH_KEYS
        else:
            expected_keys = _TRACK_TRUTH_KEYS
        if set(row) != expected_keys:
            raise ValueError(f"truth keys differ for {row['case_id']!r}")
        case_id = row["case_id"]
        if case_id in by_id:
            raise ValueError(f"duplicate truth case_id {case_id!r}")
        by_id[case_id] = row
    return by_id


def _load_attempts(root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    attempts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != ATTEMPT_SCHEMA:
            raise ValueError(f"{path}: unsupported attempt schema")
        expected_payload_keys = {
            "schema",
            "backend",
            "model",
            "case",
            "arm",
            "prompt",
            "prompt_sha256",
            "raw_response",
            "parsed_response",
            "parser_error",
            "generation_error",
            "elapsed_seconds",
            "sampling",
        }
        if set(payload) != expected_payload_keys:
            raise ValueError(f"{path}: attempt keys differ")
        case = payload.get("case")
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise TypeError(f"{path}: invalid attempt case")
        expected_case_keys = {
            "case_id",
            "kind",
            "video_id",
            "clip_path",
            "clip_sha256",
            "source_start_frame",
            "source_end_frame",
            "candidate_frame",
        }
        if set(case) != expected_case_keys:
            raise ValueError(f"{path}: attempt case keys differ")
        TrialKind(case["kind"])
        TrialArm(payload["arm"])
        if (
            not isinstance(payload["prompt"], str)
            or hashlib.sha256(payload["prompt"].encode("utf-8")).hexdigest()
            != payload["prompt_sha256"]
        ):
            raise ValueError(f"{path}: prompt hash does not reproduce")
        key = (str(payload["backend"]), case["case_id"], str(payload["arm"]))
        if key in attempts:
            raise ValueError(f"duplicate attempt for {key}")
        reparsed = None
        parser_error = None
        if payload["generation_error"] is None:
            try:
                reparsed = parse_reply(
                    TrialKind(case["kind"]), str(payload["raw_response"])
                )
            except (TypeError, ValueError) as exc:
                parser_error = str(exc)
        elif any(
            payload[name] is not None
            for name in ("raw_response", "parsed_response", "parser_error")
        ):
            raise ValueError(f"{path}: failed generation contains response data")
        if reparsed != payload.get("parsed_response") or parser_error != payload.get(
            "parser_error"
        ):
            raise ValueError(f"{path}: stored parser result does not reproduce")
        attempts[key] = payload
    if not attempts:
        raise ValueError(f"no attempt JSON files found under {root}")
    return attempts


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _decision_counts(
    decisions: list[str], truth: list[bool]
) -> dict[str, int | float | None]:
    counts = Counter(decisions)
    yes_true = sum(
        decision == "yes" and expected for decision, expected in zip(decisions, truth)
    )
    yes_false = sum(
        decision == "yes" and not expected
        for decision, expected in zip(decisions, truth)
    )
    no_true = sum(
        decision == "no" and expected for decision, expected in zip(decisions, truth)
    )
    no_false = sum(
        decision == "no" and not expected
        for decision, expected in zip(decisions, truth)
    )
    return {
        "yes": counts["yes"],
        "no": counts["no"],
        "unclear": counts["unclear"],
        "invalid": counts["invalid"],
        "yes_true": yes_true,
        "yes_false": yes_false,
        "no_true": no_true,
        "no_false": no_false,
        "yes_precision": _ratio(yes_true, yes_true + yes_false),
        "yes_recall": _ratio(yes_true, sum(truth)),
        "attempt_accuracy": _ratio(yes_true + no_false, len(decisions)),
        "resolved_accuracy": _ratio(
            yes_true + no_false, yes_true + yes_false + no_true + no_false
        ),
    }


def _score_event(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    decisions = [
        "invalid"
        if attempt["parsed_response"] is None
        else attempt["parsed_response"]["contact_at_marker"]
        for attempt, _ in rows
    ]
    result: dict[str, Any] = {}
    for tolerance in (5, 10, 15):
        truth = [bool(row[f"usable_at_{tolerance}"]) for _, row in rows]
        result[f"contact_at_{tolerance}"] = _decision_counts(decisions, truth)

    actor_total = actor_correct = 0
    nearby = Counter()
    evidence_kind = Counter()
    for attempt, truth_row in rows:
        parsed = attempt["parsed_response"]
        if parsed is None:
            continue
        nearby[parsed["nearby_unmarked_contact"]] += 1
        evidence_kind[parsed["evidence_kind"]] += 1
        expected_actor = truth_row["expected_actor"]
        if (
            parsed["contact_at_marker"] == "yes"
            and truth_row["usable_at_10"]
            and expected_actor is not None
        ):
            actor_total += 1
            actor_correct += parsed["actor"] == expected_actor
    result["actor_when_yes"] = {
        "correct": actor_correct,
        "total": actor_total,
        "accuracy": _ratio(actor_correct, actor_total),
    }
    result["nearby_unmarked_contact"] = dict(sorted(nearby.items()))
    result["evidence_kind"] = dict(sorted(evidence_kind.items()))
    for event_role in ("serve", "later-stroke"):
        role_rows = [pair for pair in rows if pair[1]["event_role"] == event_role]
        role_decisions = [
            "invalid"
            if attempt["parsed_response"] is None
            else attempt["parsed_response"]["contact_at_marker"]
            for attempt, _ in role_rows
        ]
        role_truth = [bool(truth_row["usable_at_10"]) for _, truth_row in role_rows]
        result[f"contact_at_10_{event_role}"] = _decision_counts(
            role_decisions, role_truth
        )
    return result


def _score_broadcast(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    decisions = [
        "invalid"
        if attempt["parsed_response"] is None
        else attempt["parsed_response"]["valid_rally_evidence"]
        for attempt, _ in rows
    ]
    truth = [bool(row["valid_rally"]) for _, row in rows]
    result = _decision_counts(decisions, truth)
    content_predictions = [
        "invalid"
        if attempt["parsed_response"] is None
        else attempt["parsed_response"]["broadcast_content"]
        for attempt, _ in rows
    ]
    expected_content = []
    for _, truth_row in rows:
        fractions = truth_row["scene_fractions"]
        live_fraction = fractions.get("live", 0.0) + fractions.get(
            "live-non-standard", 0.0
        )
        if live_fraction >= 0.8:
            expected_content.append("live-play")
        elif fractions.get("replay", 0.0) >= 0.6:
            expected_content.append("replay")
        elif fractions.get("cutaway", 0.0) >= 0.6:
            expected_content.append("cutaway")
        else:
            expected_content.append("mixed")
    content_resolved = [
        (predicted, expected)
        for predicted, expected in zip(content_predictions, expected_content)
        if predicted not in {"unclear", "invalid"}
    ]
    result["broadcast_content"] = {
        "predictions": dict(sorted(Counter(content_predictions).items())),
        "expected": dict(sorted(Counter(expected_content).items())),
        "resolved_correct": sum(
            predicted == expected for predicted, expected in content_resolved
        ),
        "resolved_total": len(content_resolved),
        "resolved_accuracy": _ratio(
            sum(predicted == expected for predicted, expected in content_resolved),
            len(content_resolved),
        ),
    }
    result["contains_camera_cut"] = {
        "scored": False,
        "reason": "the human scene timeline does not label cuts within one scene class",
        "predictions": dict(
            sorted(
                Counter(
                    "invalid"
                    if attempt["parsed_response"] is None
                    else attempt["parsed_response"]["contains_camera_cut"]
                    for attempt, _ in rows
                ).items()
            )
        ),
    }
    return result


def _score_track(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pooled_scored": False,
        "pooled_score_limit": (
            "Only the hallucination intervals have human tracker-validity truth. "
            "ShuttleSet-near contacts are orientation proxies, not labelled real tracks."
        ),
    }
    by_source: dict[str, Any] = {}
    for source in sorted({truth_row["truth_source"] for _attempt, truth_row in rows}):
        source_rows = [pair for pair in rows if pair[1]["truth_source"] == source]
        source_decisions = [
            "invalid"
            if attempt["parsed_response"] is None
            else attempt["parsed_response"]["tracker_follows_real_shuttle"]
            for attempt, _truth in source_rows
        ]
        source_truth = [
            bool(truth_row["tracker_real"]) for _attempt, truth_row in source_rows
        ]
        by_source[source] = _decision_counts(source_decisions, source_truth)
    result["by_truth_source"] = by_source
    result["tracked_object"] = dict(
        sorted(
            Counter(
                "invalid"
                if attempt["parsed_response"] is None
                else attempt["parsed_response"]["tracked_object"]
                for attempt, _truth in rows
            ).items()
        )
    )
    return result


def score_trials(
    manifest_path: Path,
    truth_path: Path,
    attempts_root: Path,
    *,
    expected_backends: Sequence[str] = DEFAULT_EXPECTED_BACKENDS,
    expected_arms: Sequence[TrialArm] = tuple(TrialArm),
    case_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Join truth only after inference and report every unresolved decision separately."""
    all_cases = load_manifest(manifest_path)
    cases = all_cases
    if case_ids is not None:
        cases = tuple(case for case in all_cases if case.case_id in case_ids)
        missing = case_ids - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"unknown requested case IDs: {sorted(missing)}")
    if not cases:
        raise ValueError("no score cases selected")
    case_by_id = {case.case_id: case for case in cases}
    all_truth = _load_truth(truth_path)
    if set(all_truth) != {case.case_id for case in all_cases}:
        raise ValueError("manifest and truth case IDs differ")
    truth = {case_id: all_truth[case_id] for case_id in case_by_id}
    attempts = _load_attempts(attempts_root)

    grouped: dict[
        tuple[str, str, TrialKind], list[tuple[dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    for (backend, case_id, arm), attempt in attempts.items():
        if case_id not in case_by_id:
            raise ValueError(f"attempt refers to unknown case {case_id!r}")
        case = case_by_id[case_id]
        attempt_case = attempt["case"]
        expected_identity = {
            "kind": case.kind.value,
            "video_id": case.video_id,
            "clip_path": str(case.clip_path),
            "source_start_frame": case.source_start_frame,
            "source_end_frame": case.source_end_frame,
            "candidate_frame": case.candidate_frame,
        }
        for name, expected in expected_identity.items():
            if attempt_case[name] != expected:
                raise ValueError(f"attempt {backend, case_id, arm} has wrong {name}")
        if attempt_case["clip_sha256"] != _sha256(case.clip_path):
            raise ValueError(f"attempt {backend, case_id, arm} has wrong clip hash")
        truth_row = truth[case_id]
        if (
            truth_row["kind"] != case.kind.value
            or truth_row["video_id"] != case.video_id
        ):
            raise ValueError(f"truth identity differs for {case_id!r}")
        grouped[(backend, arm, case.kind)].append((attempt, truth_row))

    results: list[dict[str, Any]] = []
    for (backend, arm, kind), rows in sorted(grouped.items()):
        if kind is TrialKind.EVENT:
            metrics = _score_event(rows)
        elif kind is TrialKind.BROADCAST:
            metrics = _score_broadcast(rows)
        else:
            metrics = _score_track(rows)
        results.append(
            {
                "backend": backend,
                "arm": arm,
                "kind": kind.value,
                "attempted_cases": len(rows),
                "parsed_cases": sum(
                    attempt["parsed_response"] is not None for attempt, _ in rows
                ),
                "metrics": metrics,
            }
        )

    expected_keys = {
        (backend, case.case_id, arm.value)
        for backend in expected_backends
        for case in cases
        for arm in expected_arms
    }
    observed_keys = set(attempts)
    missing_attempts = sorted(expected_keys - observed_keys)
    unexpected_attempts = sorted(observed_keys - expected_keys)
    parser_failures = sorted(
        key
        for key, payload in attempts.items()
        if payload["generation_error"] is None and payload["parsed_response"] is None
    )
    generation_failures = sorted(
        key
        for key, payload in attempts.items()
        if payload["generation_error"] is not None
    )
    inference_complete = not missing_attempts and not unexpected_attempts
    return {
        "schema": SCORE_SCHEMA,
        "manifest": str(manifest_path),
        "truth": str(truth_path),
        "attempts_root": str(attempts_root),
        "case_count": len(cases),
        "attempt_count": len(attempts),
        "expected_backends": list(expected_backends),
        "expected_arms": [arm.value for arm in expected_arms],
        "expected_attempt_count": len(expected_keys),
        "missing_attempts": missing_attempts,
        "unexpected_attempts": unexpected_attempts,
        "parser_failures": parser_failures,
        "generation_failures": generation_failures,
        "inference_complete": inference_complete,
        "parse_complete": not parser_failures,
        "complete": inference_complete
        and not parser_failures
        and not generation_failures,
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument(
        "--expected-backend",
        action="append",
        choices=DEFAULT_EXPECTED_BACKENDS,
        dest="expected_backends",
        help="backend required for completeness; repeat for several (default: both)",
    )
    parser.add_argument(
        "--expected-arm",
        action="append",
        choices=tuple(TrialArm),
        dest="expected_arms",
        help="prompt arm required for completeness; repeat for several (default: both)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = score_trials(
        args.manifest,
        args.truth,
        args.attempts,
        expected_backends=(
            DEFAULT_EXPECTED_BACKENDS
            if args.expected_backends is None
            else tuple(args.expected_backends)
        ),
        expected_arms=(
            tuple(TrialArm)
            if args.expected_arms is None
            else tuple(map(TrialArm, args.expected_arms))
        ),
        case_ids=None if args.case_ids is None else set(args.case_ids),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
