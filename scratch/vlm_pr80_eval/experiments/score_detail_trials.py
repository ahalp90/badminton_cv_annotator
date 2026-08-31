"""Score paired short-detail replies against the broad-pass truth sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from .backends import BACKEND_KEYS
from .detail_schema import (
    DetailArm,
    DetailCase,
    DetailContent,
    load_detail_arms,
    parse_detail_reply,
)
from .multiscale_prompts import (
    DetailPromptMode,
    build_detail_prompt,
    validate_detail_prompt_mode,
)
from .run_detail_trials import DETAIL_ATTEMPT_SCHEMA

DETAIL_SCORE_SCHEMA = "vlm-multiscale-detail-score-v1"
TRUTH_SCHEMA = "vlm-multiscale-truth-v1"
ARM_ORDER = (DetailArm.SHORT_ONLY, DetailArm.DETERMINISTIC, DetailArm.BROAD_FACTS)
MATERIAL_TARGET_BASE_30_FRAMES = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected an object in {path}")
    return payload


def _truth_content(raw: Any) -> DetailContent:
    if raw == "live-non-standard":
        return DetailContent.LIVE
    try:
        return DetailContent(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported scene truth {raw!r}") from exc


def _normalise_manifest_paths(
    manifest_paths: Mapping[DetailArm | str, Path],
) -> dict[DetailArm, Path]:
    normalised: dict[DetailArm, Path] = {}
    for raw_arm, path in manifest_paths.items():
        arm = raw_arm if isinstance(raw_arm, DetailArm) else DetailArm(raw_arm)
        if arm in normalised:
            raise ValueError(f"duplicate detail arm {arm.value}")
        normalised[arm] = Path(path)
    if set(normalised) != set(ARM_ORDER):
        raise ValueError("paths must contain short_only, deterministic, and broad_facts")
    return normalised


def _normalise_selected_arms(
    selected_arms: tuple[DetailArm, ...] | None,
) -> tuple[DetailArm, ...]:
    if selected_arms is None:
        return ARM_ORDER
    if not selected_arms:
        raise ValueError("selected_arms must not be empty")
    normalised: tuple[DetailArm, ...]
    try:
        normalised = tuple(
            arm if isinstance(arm, DetailArm) else DetailArm(arm)
            for arm in selected_arms
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown selected detail arm in {selected_arms!r}") from exc
    if len(set(normalised)) != len(normalised):
        raise ValueError("selected_arms must not contain duplicate detail arms")
    return normalised


def _truth_for_case(
    case: DetailCase,
    truth_by_pair: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates = [case.pair_id, case.context_case_id]
    if "--" in case.context_case_id:
        candidates.append(case.context_case_id.rsplit("--", 1)[0])
    for candidate in candidates:
        truth = truth_by_pair.get(candidate)
        if truth is not None:
            if str(truth["video_id"]) != case.video_id:
                raise ValueError(f"{case.case_id}: truth video differs from detail case")
            return truth
    raise ValueError(f"{case.case_id}: no context truth row matches its pair IDs")


def _intersect_truth(
    case: DetailCase,
    truth: dict[str, Any],
) -> list[tuple[int, int, DetailContent]]:
    intervals = truth.get("truth_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError(f"{case.case_id}: truth_intervals must be a non-empty list")
    intersections: list[tuple[int, int, DetailContent]] = []
    for index, interval in enumerate(intervals):
        if not isinstance(interval, dict):
            raise TypeError(f"{case.case_id}: truth interval {index} must be an object")
        try:
            start = int(interval["source_start_frame"])
            end = int(interval["source_end_frame"])
            content = _truth_content(interval["truth"])
        except KeyError as exc:
            raise ValueError(f"{case.case_id}: truth interval lacks {exc.args[0]}") from exc
        overlap_start = max(case.target_start_frame, start)
        overlap_end = min(case.target_end_frame, end)
        if overlap_start < overlap_end:
            intersections.append((overlap_start, overlap_end, content))
    intersections.sort(key=lambda interval: interval[0])
    previous_end = case.target_start_frame
    for start, end, _content in intersections:
        if start != previous_end:
            raise ValueError(f"{case.case_id}: truth does not cover its detail TARGET")
        previous_end = end
    if previous_end != case.target_end_frame:
        raise ValueError(f"{case.case_id}: truth does not cover its detail TARGET")
    return intersections


def _expected_case_payload(case: DetailCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "context_case_id": case.context_case_id,
        "video_id": case.video_id,
        "clip_path": str(case.clip_path),
        "clip_sha256": case.clip_sha256,
        "source_start_frame": case.source_start_frame,
        "source_end_frame": case.source_end_frame,
        "source_frames": list(case.source_frames),
        "target_start_frame": case.target_start_frame,
        "target_end_frame": case.target_end_frame,
        "boundary_frame": case.boundary_frame,
        "source_fps": case.source_fps,
        "sample_fps": case.sample_fps,
        "expected_frames": case.expected_frames,
        "width": case.width,
        "height": case.height,
        "target_segment_ids": list(case.target_segment_ids),
    }


def _load_attempt(
    attempts_root: Path,
    backend: str,
    arm: DetailArm,
    case: DetailCase,
    manifest_sha256: str,
    prompt_mode: DetailPromptMode,
) -> dict[str, Any]:
    path = attempts_root / backend / arm.value / f"{case.case_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing detail attempt: {path}")
    attempt = _load_json(path)
    expected_keys = {
        "schema",
        "backend",
        "model",
        "arm",
        "manifest_sha256",
        "case",
        "prompt",
        "prompt_sha256",
        "raw_response",
        "parsed_response",
        "parser_error",
        "generation_error",
        "elapsed_seconds",
        "sampling",
    }
    if set(attempt) != expected_keys:
        raise ValueError(f"{path}: attempt keys differ")
    if attempt["schema"] != DETAIL_ATTEMPT_SCHEMA:
        raise ValueError(f"{path}: unexpected detail attempt schema")
    if attempt["backend"] != backend or attempt["arm"] != arm.value:
        raise ValueError(f"{path}: backend or arm identity differs")
    if attempt["manifest_sha256"] != manifest_sha256:
        raise ValueError(f"{path}: manifest identity differs")
    if not isinstance(attempt["model"], dict):
        raise TypeError(f"{path}: model identity must be an object")
    expected_case = _expected_case_payload(case)
    observed_case = attempt["case"]
    if not isinstance(observed_case, dict) or set(observed_case) != set(expected_case):
        raise ValueError(f"{path}: case identity differs")
    # The absolute clip path can change when a run is copied from the GPU host.
    # The content hash below is the identity that must remain stable.
    if any(
        observed_case[key] != expected_case[key]
        for key in expected_case
        if key != "clip_path"
    ):
        raise ValueError(f"{path}: case identity differs")
    if _sha256(case.clip_path) != case.clip_sha256:
        raise ValueError(f"{path}: clip on disk does not match manifest")
    prompt = attempt["prompt"]
    if not isinstance(prompt, str):
        raise TypeError(f"{path}: prompt must be text")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != attempt["prompt_sha256"]:
        raise ValueError(f"{path}: prompt hash does not reproduce")
    expected_prompt = build_detail_prompt(case, arm, prompt_mode=prompt_mode)
    if prompt != expected_prompt:
        raise ValueError(f"{path}: prompt differs from the manifest facts")

    generation_error = attempt["generation_error"]
    parser_error = attempt["parser_error"]
    parsed_response = attempt["parsed_response"]
    if generation_error is not None:
        if any(
            attempt[field] is not None
                for field in ("raw_response", "parsed_response", "parser_error")
            ):
                raise ValueError(f"{path}: failed generation contains response data")
        attempt["_parser_recovered"] = False
        return attempt
    if not isinstance(attempt["raw_response"], str):
        raise TypeError(f"{path}: successful generation must contain raw text")
    reparsed = None
    reproduced_error = None
    try:
        reparsed = {"target_content": parse_detail_reply(attempt["raw_response"]).target_content}
        reparsed["target_content"] = str(reparsed["target_content"])
    except (TypeError, ValueError) as exc:
        reproduced_error = str(exc)
    parser_recovered = False
    if reparsed != parsed_response or reproduced_error != parser_error:
        if (
            parsed_response is None
            and parser_error is not None
            and reparsed is not None
            and reproduced_error is None
        ):
            attempt["parsed_response"] = reparsed
            attempt["parser_error"] = None
            parsed_response = reparsed
            parser_error = None
            parser_recovered = True
        else:
            raise ValueError(f"{path}: stored parser result does not reproduce")
    if (parsed_response is None) == (parser_error is None):
        raise ValueError(f"{path}: parsed response and parser error are inconsistent")
    attempt["_parser_recovered"] = parser_recovered
    return attempt


def _case_score(
    case: DetailCase,
    truth: dict[str, Any],
    attempt: dict[str, Any],
    arm: DetailArm,
) -> dict[str, Any]:
    intersections = _intersect_truth(case, truth)
    parsed = attempt["parsed_response"]
    predicted = None
    if parsed is not None:
        predicted = DetailContent(parsed["target_content"])
    predicted_live = predicted is DetailContent.LIVE
    exact_correct = 0
    binary_correct = 0
    truth_live_only = True
    truth_frame_counts: dict[str, int] = {}
    for start, end, truth_content in intersections:
        overlap = end - start
        truth_frame_counts[truth_content.value] = truth_frame_counts.get(truth_content.value, 0) + overlap
        if truth_content is not DetailContent.LIVE:
            truth_live_only = False
        if predicted is truth_content:
            exact_correct += overlap
        if (
            predicted is not None
            and predicted is not DetailContent.UNCLEAR
            and predicted_live == (truth_content is DetailContent.LIVE)
        ):
            binary_correct += overlap
    target_frames = case.target_end_frame - case.target_start_frame
    minimum_material_target_frames = math.ceil(
        case.source_fps * MATERIAL_TARGET_BASE_30_FRAMES / 30
    )
    predicted_route = "routine_live" if predicted_live else "close_check"
    truth_route = "routine_live" if truth_live_only else "close_check"
    return {
        "arm": arm.value,
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "context_case_id": case.context_case_id,
        "video_id": case.video_id,
        "target_start_frame": case.target_start_frame,
        "target_end_frame": case.target_end_frame,
        "target_frames": target_frames,
        "minimum_material_target_frames": minimum_material_target_frames,
        "meets_material_target": target_frames >= minimum_material_target_frames,
        "valid_reply": parsed is not None,
        "parser_recovered": bool(attempt.get("_parser_recovered", False)),
        "predicted_content": None if predicted is None else predicted.value,
        "truth_content_frames": truth_frame_counts,
        "exact_scene_correct_frames": exact_correct,
        "exact_scene_accuracy": exact_correct / target_frames,
        "binary_live_nonlive_correct_frames": binary_correct,
        "binary_live_nonlive_accuracy": binary_correct / target_frames,
        "binary_scene_correct_frames": binary_correct,
        "binary_scene_accuracy": binary_correct / target_frames,
        "predicted_route": predicted_route,
        "truth_route": truth_route,
        "route_correct": predicted_route == truth_route,
        "parser_error": attempt["parser_error"],
        "generation_error": attempt["generation_error"],
    }


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty detail arm")
    target_frames = sum(int(row["target_frames"]) for row in rows)
    truth_close = [row for row in rows if row["truth_route"] == "close_check"]
    truth_routine = [row for row in rows if row["truth_route"] == "routine_live"]
    predicted_close = [row for row in rows if row["predicted_route"] == "close_check"]
    predicted_routine = [row for row in rows if row["predicted_route"] == "routine_live"]
    close_correct = sum(row["predicted_route"] == "close_check" for row in truth_close)
    routine_correct = sum(row["predicted_route"] == "routine_live" for row in truth_routine)
    return {
        "cases": len(rows),
        "valid_replies": sum(bool(row["valid_reply"]) for row in rows),
        "invalid_replies": sum(not row["valid_reply"] for row in rows),
        "target_frames": target_frames,
        "mean_case_exact_scene_accuracy": mean(float(row["exact_scene_accuracy"]) for row in rows),
        "mean_case_binary_live_nonlive_accuracy": mean(
            float(row["binary_live_nonlive_accuracy"]) for row in rows
        ),
        "mean_case_binary_scene_accuracy": mean(float(row["binary_scene_accuracy"]) for row in rows),
        "target_frame_exact_scene_accuracy": sum(
            int(row["exact_scene_correct_frames"]) for row in rows
        )
        / target_frames,
        "target_frame_binary_live_nonlive_accuracy": sum(
            int(row["binary_live_nonlive_correct_frames"]) for row in rows
        )
        / target_frames,
        "target_frame_binary_scene_accuracy": sum(
            int(row["binary_scene_correct_frames"]) for row in rows
        )
        / target_frames,
        "route_accuracy": sum(bool(row["route_correct"]) for row in rows) / len(rows),
        "close_check_recall": close_correct / len(truth_close) if truth_close else None,
        "unsafe_close_check_recall": close_correct / len(truth_close) if truth_close else None,
        "routine_live_recall": routine_correct / len(truth_routine) if truth_routine else None,
        "routine_live_precision": (
            sum(row["truth_route"] == "routine_live" for row in predicted_routine)
            / len(predicted_routine)
            if predicted_routine
            else None
        ),
        "close_check_precision": (
            sum(row["truth_route"] == "close_check" for row in predicted_close) / len(predicted_close)
            if predicted_close
            else None
        ),
    }


def _aggregate_material_target(
    rows: Sequence[dict[str, Any]],
    excluded_cases: int,
) -> dict[str, Any]:
    """Aggregate material-length rows while retaining how many were excluded."""
    if rows:
        aggregate = _aggregate(rows)
    else:
        aggregate = {
            "cases": 0,
            "valid_replies": 0,
            "invalid_replies": 0,
            "target_frames": 0,
            "mean_case_exact_scene_accuracy": None,
            "mean_case_binary_live_nonlive_accuracy": None,
            "mean_case_binary_scene_accuracy": None,
            "target_frame_exact_scene_accuracy": None,
            "target_frame_binary_live_nonlive_accuracy": None,
            "target_frame_binary_scene_accuracy": None,
            "route_accuracy": None,
            "close_check_recall": None,
            "unsafe_close_check_recall": None,
            "routine_live_recall": None,
            "routine_live_precision": None,
            "close_check_precision": None,
        }
    aggregate["excluded_cases"] = excluded_cases
    return aggregate


def _comparison(
    aggregates: Mapping[DetailArm, dict[str, Any]],
    left: DetailArm,
    right: DetailArm,
) -> dict[str, float | None]:
    fields = (
        "mean_case_exact_scene_accuracy",
        "target_frame_exact_scene_accuracy",
        "mean_case_binary_live_nonlive_accuracy",
        "target_frame_binary_live_nonlive_accuracy",
        "close_check_recall",
        "routine_live_recall",
        "routine_live_precision",
    )
    result: dict[str, float | None] = {}
    for field in fields:
        left_value = aggregates[left][field]
        right_value = aggregates[right][field]
        result[f"{field}_delta"] = (
            None
            if left_value is None or right_value is None
            else float(left_value) - float(right_value)
        )
    return result


def score_detail_attempts(
    manifest_paths: Mapping[DetailArm | str, Path],
    truth_path: Path,
    attempts_root: Path,
    backend: str,
    *,
    selected_arms: tuple[DetailArm, ...] | None = None,
    prompt_mode: DetailPromptMode | str = DetailPromptMode.DEFAULT,
) -> dict[str, Any]:
    """Score selected arms, retaining invalid replies in every denominator."""
    arms = _normalise_selected_arms(selected_arms)
    prompt_mode = validate_detail_prompt_mode(prompt_mode, arms)
    normalised_paths = _normalise_manifest_paths(manifest_paths)
    manifests = load_detail_arms(
        normalised_paths,
        require_clips=True,
        verify_clip_hash=True,
    )
    truth_payload = _load_json(truth_path)
    if truth_payload.get("schema") != TRUTH_SCHEMA:
        raise ValueError(f"unexpected context truth schema {truth_payload.get('schema')!r}")
    truth_rows = truth_payload.get("cases")
    if not isinstance(truth_rows, list):
        raise TypeError("context truth cases must be a list")
    truth_by_pair: dict[str, dict[str, Any]] = {}
    for row in truth_rows:
        if not isinstance(row, dict) or not isinstance(row.get("pair_id"), str):
            raise TypeError("every context truth case must have a string pair_id")
        pair_id = row["pair_id"]
        if pair_id in truth_by_pair:
            raise ValueError(f"duplicate context truth pair_id {pair_id!r}")
        truth_by_pair[pair_id] = row

    manifest_hashes = {
        arm.value: _sha256(normalised_paths[arm])
        for arm in ARM_ORDER
    }
    cases_by_arm = {
        arm: {case.case_id: case for case in manifests[arm].cases}
        for arm in ARM_ORDER
    }
    baseline_cases = list(manifests[DetailArm.SHORT_ONLY].cases)
    rows_by_arm: dict[DetailArm, list[dict[str, Any]]] = {arm: [] for arm in arms}
    nested_cases: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for baseline_case in baseline_cases:
        truth = _truth_for_case(baseline_case, truth_by_pair)
        case_rows: dict[str, Any] = {
            "case_id": baseline_case.case_id,
            "pair_id": baseline_case.pair_id,
            "video_id": baseline_case.video_id,
            "by_arm": {},
        }
        for arm in arms:
            case = cases_by_arm[arm][baseline_case.case_id]
            attempt = _load_attempt(
                attempts_root,
                backend,
                arm,
                case,
                manifest_hashes[arm.value],
                prompt_mode,
            )
            row = _case_score(case, truth, attempt, arm)
            rows_by_arm[arm].append(row)
            flat_rows.append(row)
            case_rows["by_arm"][arm.value] = row
        nested_cases.append(case_rows)

    aggregates = {arm: _aggregate(rows_by_arm[arm]) for arm in arms}
    material_aggregates: dict[DetailArm, dict[str, Any]] = {}
    material_excluded_cases: dict[DetailArm, int] = {}
    for arm in arms:
        material_rows = [
            row for row in rows_by_arm[arm] if row["meets_material_target"]
        ]
        excluded_cases = len(rows_by_arm[arm]) - len(material_rows)
        material_excluded_cases[arm] = excluded_cases
        material_aggregates[arm] = _aggregate_material_target(
            material_rows,
            excluded_cases=excluded_cases,
        )
    comparison_specs = (
        ("broad_facts_vs_deterministic", DetailArm.BROAD_FACTS, DetailArm.DETERMINISTIC),
        ("broad_facts_vs_short_only", DetailArm.BROAD_FACTS, DetailArm.SHORT_ONLY),
        ("deterministic_vs_short_only", DetailArm.DETERMINISTIC, DetailArm.SHORT_ONLY),
    )
    selected_arm_set = set(arms)
    return {
        "schema": DETAIL_SCORE_SCHEMA,
        "backend": backend,
        "prompt_mode": prompt_mode.value,
        "manifest_sha256_by_arm": manifest_hashes,
        "truth_sha256": _sha256(truth_path),
        "by_arm": {arm.value: aggregates[arm] for arm in arms},
        "by_arm_material_target": {
            arm.value: material_aggregates[arm] for arm in arms
        },
        "material_target_excluded_cases_by_arm": {
            arm.value: material_excluded_cases[arm] for arm in arms
        },
        "comparison": {
            name: _comparison(aggregates, left, right)
            for name, left, right in comparison_specs
            if left in selected_arm_set and right in selected_arm_set
        },
        "cases": nested_cases,
        "rows": flat_rows,
    }


def _parse_arm_paths(raw_paths: Sequence[str]) -> dict[DetailArm, Path]:
    paths: dict[DetailArm, Path] = {}
    for raw in raw_paths:
        arm_name, separator, manifest = raw.partition("=")
        if not separator or not manifest:
            raise ValueError("--arm values must use ARM=MANIFEST form")
        try:
            arm = DetailArm(arm_name)
        except ValueError as exc:
            raise ValueError(f"unknown detail arm {arm_name!r}") from exc
        if arm in paths:
            raise ValueError(f"duplicate detail arm {arm.value}")
        paths[arm] = Path(manifest)
    if set(paths) != set(ARM_ORDER):
        raise ValueError("provide one manifest for short_only, deterministic, and broad_facts")
    return paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="ARM=MANIFEST")
    parser.add_argument(
        "--only-arm",
        action="append",
        dest="only_arms",
        choices=tuple(arm.value for arm in ARM_ORDER),
        help="score only this detail arm; repeat to select multiple arms (default: all arms)",
    )
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKEND_KEYS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=tuple(mode.value for mode in DetailPromptMode),
        default=DetailPromptMode.DEFAULT.value,
        help="detail prompt contract (conservative_replay_veto requires only short_only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    score = score_detail_attempts(
        _parse_arm_paths(args.arm),
        args.truth,
        args.attempts,
        args.backend,
        selected_arms=(
            None
            if args.only_arms is None
            else tuple(DetailArm(raw_arm) for raw_arm in args.only_arms)
        ),
        prompt_mode=args.prompt_mode,
    )
    _write_json(args.out, score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
