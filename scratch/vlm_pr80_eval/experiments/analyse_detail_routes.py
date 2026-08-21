"""Analyse conservative routes from completed paired detail score files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detail_schema import DetailArm, DetailContent, load_detail_manifest
from .score_detail_trials import DETAIL_SCORE_SCHEMA

ROUTE_ANALYSIS_SCHEMA = "vlm-multiscale-detail-route-analysis-v1"
ROUTINE_LIVE = "routine_live"
CLOSE_CHECK = "close_check"
TRACK_VISIBLE_FRACTION_THRESHOLD = 0.8
_ROUTES = frozenset((ROUTINE_LIVE, CLOSE_CHECK))
_PRIORS_TRUTH_KEYS = frozenset(
    {
        "truth",
        "ground_truth",
        "truth_route",
        "truth_content_frames",
        "target_frames",
        "predicted_route",
    }
)


@dataclass(frozen=True)
class DetailScoreCase:
    """Truth-side fields shared by all paired short-only score files."""

    case_id: str
    target_frames: int
    truth_route: str
    truth_content_frames: dict[str, int]
    predicted_route: str


@dataclass(frozen=True)
class BroadCaseFact:
    """The one optional broad fact used to combine with a detail prediction."""

    content: str | None


def _reject_priors_truth_keys(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _PRIORS_TRUTH_KEYS:
                raise ValueError(f"{location} contains forbidden truth key {key!r}")
            _reject_priors_truth_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_priors_truth_keys(child, f"{location}[{index}]")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_score_spec(raw: str) -> tuple[str, Path]:
    backend, separator, score_path = raw.partition("=")
    if not separator or not backend or not score_path:
        raise ValueError("--score values must use BACKEND=SCORE_JSON form")
    return backend, Path(score_path)


def _required_case_id(raw: Any, location: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{location} must be a non-empty string")
    return raw


def _required_positive_int(raw: Any, location: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return raw


def _frame_counts(raw: Any, location: str) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise TypeError(f"{location} must be an object")
    counts: dict[str, int] = {}
    for content, count in raw.items():
        if not isinstance(content, str) or not content:
            raise ValueError(f"{location} keys must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{location}.{content} must be a non-negative integer")
        counts[content] = count
    return counts


def _load_short_only_rows(path: Path, backend: str) -> dict[str, DetailScoreCase]:
    payload = _load_json(path)
    if payload.get("schema") != DETAIL_SCORE_SCHEMA:
        raise ValueError(
            f"{path}: expected detail score schema {DETAIL_SCORE_SCHEMA!r}, "
            f"got {payload.get('schema')!r}"
        )
    if payload.get("backend") != backend:
        raise ValueError(
            f"{path}: score backend {payload.get('backend')!r} does not match {backend!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{path}: detail score rows must be a list")

    short_rows: dict[str, DetailScoreCase] = {}
    for index, raw_row in enumerate(rows):
        location = f"{path}: rows[{index}]"
        if not isinstance(raw_row, dict):
            raise TypeError(f"{location} must be an object")
        if raw_row.get("arm") != DetailArm.SHORT_ONLY.value:
            continue
        case_id = _required_case_id(raw_row.get("case_id"), f"{location}.case_id")
        if case_id in short_rows:
            raise ValueError(f"{path}: duplicate short_only case ID {case_id!r}")
        target_frames = _required_positive_int(
            raw_row.get("target_frames"),
            f"{location}.target_frames",
        )
        truth_route = raw_row.get("truth_route")
        if truth_route not in _ROUTES:
            raise ValueError(f"{location}.truth_route has an unsupported value")
        predicted_route = raw_row.get("predicted_route")
        if predicted_route not in _ROUTES:
            raise ValueError(f"{location}.predicted_route has an unsupported value")
        truth_content_frames = _frame_counts(
            raw_row.get("truth_content_frames"),
            f"{location}.truth_content_frames",
        )
        if sum(truth_content_frames.values()) != target_frames:
            raise ValueError(
                f"{location}.truth_content_frames do not sum to target_frames"
            )
        short_rows[case_id] = DetailScoreCase(
            case_id=case_id,
            target_frames=target_frames,
            truth_route=truth_route,
            truth_content_frames=truth_content_frames,
            predicted_route=predicted_route,
        )

    if not short_rows:
        raise ValueError(f"{path}: score has no short_only rows")
    cases = payload.get("cases")
    if cases is not None:
        if not isinstance(cases, list):
            raise TypeError(f"{path}: detail score cases must be a list")
        nested_case_ids: list[str] = []
        for index, raw_case in enumerate(cases):
            if not isinstance(raw_case, dict):
                raise TypeError(f"{path}: cases[{index}] must be an object")
            nested_case_ids.append(
                _required_case_id(raw_case.get("case_id"), f"{path}: cases[{index}].case_id")
            )
        if len(set(nested_case_ids)) != len(nested_case_ids):
            raise ValueError(f"{path}: detail score cases contain duplicate IDs")
        if set(nested_case_ids) != set(short_rows):
            raise ValueError(f"{path}: nested and flat detail score case IDs differ")
    return short_rows


def _check_shared_truth(
    scores: Mapping[str, Mapping[str, DetailScoreCase]],
) -> tuple[list[str], dict[str, DetailScoreCase]]:
    if not scores:
        raise ValueError("provide at least one detail score")
    first_backend, first_rows = next(iter(scores.items()))
    case_ids = set(first_rows)
    reference = dict(first_rows)
    for backend, rows in scores.items():
        if set(rows) != case_ids:
            missing = sorted(case_ids - set(rows))
            extra = sorted(set(rows) - case_ids)
            raise ValueError(
                f"{backend}: short_only case IDs differ from {first_backend}: "
                f"missing={missing}, extra={extra}"
            )
        if backend == first_backend:
            continue
        for case_id in sorted(case_ids):
            observed = rows[case_id]
            expected = reference[case_id]
            if (
                observed.truth_route != expected.truth_route
                or observed.target_frames != expected.target_frames
                or observed.truth_content_frames != expected.truth_content_frames
            ):
                raise ValueError(
                    f"{backend}: truth-side fields differ for case {case_id!r}"
                )
    return sorted(case_ids), reference


def _load_broad_facts(
    path: Path,
    case_ids: Sequence[str],
) -> dict[str, BroadCaseFact]:
    manifest = load_detail_manifest(path, require_clips=False, verify_clip_hash=False)
    if manifest.arm is not DetailArm.BROAD_FACTS:
        raise ValueError(f"{path}: broad manifest arm must be broad_facts")
    facts_by_case: dict[str, BroadCaseFact] = {}
    for case in manifest.cases:
        if len(case.target_segment_ids) != 1:
            raise ValueError(
                f"{path}: {case.case_id} must inspect exactly one target segment"
            )
        if case.broad_facts is None:
            facts_by_case[case.case_id] = BroadCaseFact(None)
            continue
        if len(case.broad_facts) != 1:
            raise ValueError(
                f"{path}: {case.case_id} must contain exactly one broad fact or null"
            )
        fact = case.broad_facts[0]
        if fact.segment_id != case.target_segment_ids[0]:
            raise ValueError(
                f"{path}: {case.case_id} broad fact does not match its target segment"
            )
        facts_by_case[case.case_id] = BroadCaseFact(fact.content.value)

    expected_ids = set(case_ids)
    observed_ids = set(facts_by_case)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise ValueError(
            f"{path}: broad manifest case IDs differ from detail scores: "
            f"missing={missing}, extra={extra}"
        )
    return facts_by_case


def _load_track_visible_fractions(
    path: Path,
    case_ids: Sequence[str],
) -> dict[str, float]:
    """Load one truth-blind track visibility prior for each scored case."""
    manifest = load_detail_manifest(path, require_clips=False, verify_clip_hash=False)
    if manifest.arm is not DetailArm.DETERMINISTIC:
        raise ValueError(f"{path}: priors manifest arm must be deterministic")

    fractions_by_case: dict[str, float] = {}
    for case in manifest.cases:
        if case.case_id in fractions_by_case:
            raise ValueError(f"{path}: duplicate priors case ID {case.case_id!r}")
        deterministic_facts = case.deterministic_facts
        if not isinstance(deterministic_facts, dict):
            raise TypeError(f"{path}: {case.case_id} must contain deterministic facts")
        _reject_priors_truth_keys(
            deterministic_facts,
            f"{path}: {case.case_id}.deterministic_facts",
        )
        pipeline_priors = deterministic_facts.get("pipeline_priors")
        if not isinstance(pipeline_priors, dict):
            raise TypeError(f"{path}: {case.case_id} must contain one pipeline_priors record")
        raw_fraction = pipeline_priors.get("track_visible_fraction")
        if isinstance(raw_fraction, bool) or not isinstance(raw_fraction, (int, float)):
            raise TypeError(
                f"{path}: {case.case_id}.pipeline_priors.track_visible_fraction must be numeric"
            )
        fraction = float(raw_fraction)
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError(
                f"{path}: {case.case_id}.pipeline_priors.track_visible_fraction must be in [0, 1]"
            )
        fractions_by_case[case.case_id] = fraction

    expected_ids = set(case_ids)
    observed_ids = set(fractions_by_case)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise ValueError(
            f"{path}: priors manifest case IDs differ from detail scores: "
            f"missing={missing}, extra={extra}"
        )
    return fractions_by_case


def _route_metrics(
    routes: Mapping[str, str],
    references: Mapping[str, DetailScoreCase],
) -> dict[str, Any]:
    if set(routes) != set(references):
        raise ValueError("route case IDs do not match score case IDs")
    truth_close = [case_id for case_id, row in references.items() if row.truth_route == CLOSE_CHECK]
    truth_routine = [case_id for case_id, row in references.items() if row.truth_route == ROUTINE_LIVE]
    predicted_close = [case_id for case_id, route in routes.items() if route == CLOSE_CHECK]
    predicted_routine = [case_id for case_id, route in routes.items() if route == ROUTINE_LIVE]
    correct = sum(routes[case_id] == row.truth_route for case_id, row in references.items())
    close_hits = sum(routes[case_id] == CLOSE_CHECK for case_id in truth_close)
    routine_hits = sum(routes[case_id] == ROUTINE_LIVE for case_id in truth_routine)
    target_frames = sum(row.target_frames for row in references.values())
    kept_frames = sum(
        references[case_id].target_frames
        for case_id, route in routes.items()
        if route == ROUTINE_LIVE
    )
    return {
        "cases": len(references),
        "route_accuracy": correct / len(references),
        "unsafe_close_check_recall": close_hits / len(truth_close) if truth_close else None,
        "routine_live_recall": routine_hits / len(truth_routine) if truth_routine else None,
        "routine_live_precision": (
            sum(references[case_id].truth_route == ROUTINE_LIVE for case_id in predicted_routine)
            / len(predicted_routine)
            if predicted_routine
            else None
        ),
        "close_check_precision": (
            sum(references[case_id].truth_route == CLOSE_CHECK for case_id in predicted_close)
            / len(predicted_close)
            if predicted_close
            else None
        ),
        "target_frames": target_frames,
        "target_frames_kept": kept_frames,
        "target_frame_coverage_kept": kept_frames / target_frames,
    }


def _rule(
    name: str,
    routes: Mapping[str, str],
    references: Mapping[str, DetailScoreCase],
    *,
    parent: str | None,
    changed_case_ids: Sequence[str] | Mapping[str, Sequence[str]] = (),
) -> dict[str, Any]:
    if isinstance(changed_case_ids, Mapping):
        changed_value: Any = {
            key: list(value) for key, value in sorted(changed_case_ids.items())
        }
    else:
        changed_value = list(changed_case_ids)
    return {
        "name": name,
        "parent": parent,
        "routes": {case_id: routes[case_id] for case_id in sorted(routes)},
        "metrics": _route_metrics(routes, references),
        "changed_case_ids_vs_parent": changed_value,
    }


def analyse_detail_routes(
    score_paths: Mapping[str, Path],
    broad_manifest: Path | None = None,
    priors_manifest: Path | None = None,
) -> dict[str, Any]:
    """Build conservative routing rules from one or more detail score files.

    :param score_paths: Mapping from backend name to completed detail score JSON.
    :param broad_manifest: Optional broad-facts detail manifest.
    :param priors_manifest: Optional deterministic detail manifest with pipeline priors.
    :return: Versioned routing report with paired routes and metrics.
    """
    if not score_paths:
        raise ValueError("provide at least one detail score")
    loaded_scores = {
        backend: _load_short_only_rows(Path(path), backend)
        for backend, path in score_paths.items()
    }
    case_ids, references = _check_shared_truth(loaded_scores)

    short_rules: dict[str, dict[str, Any]] = {}
    short_routes: dict[str, dict[str, str]] = {}
    for backend, rows in loaded_scores.items():
        routes = {case_id: row.predicted_route for case_id, row in rows.items()}
        short_routes[backend] = routes
        short_rules[backend] = _rule(
            f"{backend}:short_only",
            routes,
            references,
            parent=None,
        )

    all_models_routes = {
        case_id: (
            ROUTINE_LIVE
            if all(short_routes[backend][case_id] == ROUTINE_LIVE for backend in short_routes)
            else CLOSE_CHECK
        )
        for case_id in case_ids
    }
    all_changed = {
        backend: sorted(
            case_id
            for case_id in case_ids
            if all_models_routes[case_id] != short_routes[backend][case_id]
        )
        for backend in sorted(short_routes)
    }
    all_models_rule = _rule(
        "all_models_live",
        all_models_routes,
        references,
        parent="short_only",
        changed_case_ids=all_changed,
    )

    broad_rules: dict[str, dict[str, Any]] | None = None
    broad_facts: dict[str, BroadCaseFact] | None = None
    if broad_manifest is not None:
        broad_facts = _load_broad_facts(Path(broad_manifest), case_ids)
        broad_rules = {}
        for backend in sorted(short_routes):
            routes = {
                case_id: (
                    ROUTINE_LIVE
                    if (
                        short_routes[backend][case_id] == ROUTINE_LIVE
                        and broad_facts[case_id].content == DetailContent.LIVE.value
                    )
                    else CLOSE_CHECK
                )
                for case_id in case_ids
            }
            changed = sorted(
                case_id
                for case_id in case_ids
                if routes[case_id] != short_routes[backend][case_id]
            )
            broad_rules[backend] = _rule(
                f"{backend}:short_only_plus_broad_content",
                routes,
                references,
                parent=f"{backend}:short_only",
                changed_case_ids=changed,
            )

    prior_rule: dict[str, Any] | None = None
    prior_intersection_rules: dict[str, dict[str, Any]] | None = None
    track_visible_fractions: dict[str, float] | None = None
    if priors_manifest is not None:
        track_visible_fractions = _load_track_visible_fractions(
            Path(priors_manifest), case_ids
        )
        prior_routes = {
            case_id: (
                ROUTINE_LIVE
                if track_visible_fractions[case_id] >= TRACK_VISIBLE_FRACTION_THRESHOLD
                else CLOSE_CHECK
            )
            for case_id in case_ids
        }
        prior_rule = _rule(
            "track_visible_fraction_at_0_8",
            prior_routes,
            references,
            parent=None,
        )
        prior_intersection_rules = {}
        for backend in sorted(short_routes):
            routes = {
                case_id: (
                    ROUTINE_LIVE
                    if (
                        short_routes[backend][case_id] == ROUTINE_LIVE
                        and prior_routes[case_id] == ROUTINE_LIVE
                    )
                    else CLOSE_CHECK
                )
                for case_id in case_ids
            }
            changed = sorted(
                case_id
                for case_id in case_ids
                if routes[case_id] != short_routes[backend][case_id]
            )
            prior_intersection_rules[backend] = _rule(
                f"{backend}:short_only_plus_track_visible_fraction",
                routes,
                references,
                parent=f"{backend}:short_only",
                changed_case_ids=changed,
            )

    changed_case_ids: dict[str, Any] = {
        "all_models_live_vs_short_only": all_changed,
    }
    if broad_rules is not None:
        changed_case_ids["short_only_plus_broad_content_vs_short_only"] = {
            backend: broad_rules[backend]["changed_case_ids_vs_parent"]
            for backend in sorted(broad_rules)
        }
    if prior_intersection_rules is not None:
        changed_case_ids["short_only_plus_track_visible_fraction_vs_short_only"] = {
            backend: prior_intersection_rules[backend]["changed_case_ids_vs_parent"]
            for backend in sorted(prior_intersection_rules)
        }

    return {
        "schema": ROUTE_ANALYSIS_SCHEMA,
        "case_ids": case_ids,
        "backends": sorted(loaded_scores),
        "scores": {
            backend: {
                "path": str(Path(path)),
                "sha256": _sha256(Path(path)),
            }
            for backend, path in sorted(score_paths.items())
        },
        "broad_manifest": (
            None
            if broad_manifest is None
            else {
                "path": str(Path(broad_manifest)),
                "sha256": _sha256(Path(broad_manifest)),
            }
        ),
        "priors_manifest": (
            None
            if priors_manifest is None
            else {
                "path": str(Path(priors_manifest)),
                "sha256": _sha256(Path(priors_manifest)),
            }
        ),
        "rules": {
            "short_only": short_rules,
            "all_models_live": all_models_rule,
            "short_only_plus_broad_content": broad_rules,
            "track_visible_fraction_at_0_8": prior_rule,
            "short_only_plus_track_visible_fraction": prior_intersection_rules,
        },
        "changed_case_ids": changed_case_ids,
    }


analyse_routes = analyse_detail_routes


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        metavar="BACKEND=SCORE_JSON",
        help="completed detail score; repeat once per backend",
    )
    parser.add_argument(
        "--broad-manifest",
        "--broad-facts-manifest",
        "--manifest",
        dest="broad_manifest",
        type=Path,
    )
    parser.add_argument(
        "--priors-manifest",
        type=Path,
        help="deterministic detail manifest containing pipeline priors",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    score_paths: dict[str, Path] = {}
    for raw_score in args.score:
        backend, path = _parse_score_spec(raw_score)
        if backend in score_paths:
            raise ValueError(f"duplicate backend {backend!r}")
        score_paths[backend] = path
    report = analyse_detail_routes(
        score_paths,
        args.broad_manifest,
        args.priors_manifest,
    )
    _write_json(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
