"""Run the three paired short-detail arms through one resident VLM backend."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from .detail_schema import (
    DetailArm,
    DetailCase,
    load_detail_arms,
    parse_detail_reply,
)
from .multiscale_prompts import (
    DetailPromptMode,
    build_detail_prompt,
    validate_detail_prompt_mode,
)

DETAIL_ATTEMPT_SCHEMA = "vlm-multiscale-detail-attempt-v1"
DEFAULT_MAX_NEW_TOKENS = 128
QWEN_MAX_MODEL_LEN = 16_384
ARM_ORDER = (DetailArm.SHORT_ONLY, DetailArm.DETERMINISTIC, DetailArm.BROAD_FACTS)


def _load_backend(
    backend_name: str,
    *,
    expected_input_frames: int,
    max_model_len: int | None,
) -> Any:
    from .backends import load_backend

    return load_backend(
        backend_name,
        expected_input_frames=expected_input_frames,
        max_model_len=max_model_len,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an attempt once so a resumed run cannot overwrite evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _case_payload(case: DetailCase) -> dict[str, Any]:
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


def _selected_cases(
    cases: Sequence[DetailCase],
    case_ids: set[str] | None,
    limit: int | None,
) -> list[DetailCase]:
    selected = list(cases)
    if case_ids is not None:
        selected = [case for case in selected if case.case_id in case_ids]
        missing = case_ids - {case.case_id for case in selected}
        if missing:
            raise ValueError(f"unknown requested case IDs: {sorted(missing)}")
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("no detail cases selected")
    return selected


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


def run_detail_trials(
    backend_name: str,
    manifest_paths: Mapping[DetailArm | str, Path],
    output_dir: Path,
    *,
    case_ids: set[str] | None = None,
    limit: int | None = None,
    selected_arms: tuple[DetailArm, ...] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    qwen_max_model_len: int = QWEN_MAX_MODEL_LEN,
    prompt_mode: DetailPromptMode | str = DetailPromptMode.DEFAULT,
) -> None:
    """Run selected arms while keeping one immutable attempt per case."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    arms = _normalise_selected_arms(selected_arms)
    prompt_mode = validate_detail_prompt_mode(prompt_mode, arms)
    normalised_paths = _normalise_manifest_paths(manifest_paths)
    manifests = load_detail_arms(
        normalised_paths,
        require_clips=True,
        verify_clip_hash=True,
    )
    baseline = manifests[DetailArm.SHORT_ONLY]
    cases = _selected_cases(baseline.cases, case_ids, limit)
    frame_counts = {manifest.expected_frames for manifest in manifests.values()}
    if len(frame_counts) != 1:
        raise ValueError(f"detail arms use different frame counts: {sorted(frame_counts)}")
    expected_frames = frame_counts.pop()
    max_model_len = qwen_max_model_len if backend_name == "qwen3-vl" else None
    backend = _load_backend(
        backend_name,
        expected_input_frames=expected_frames,
        max_model_len=max_model_len,
    )
    model_identity = asdict(backend.spec.identity(backend.backend_version))
    manifest_hashes = {
        arm: _sha256(normalised_paths[arm])
        for arm in ARM_ORDER
    }
    cases_by_arm = {
        arm: {case.case_id: case for case in manifests[arm].cases}
        for arm in ARM_ORDER
    }

    for arm in arms:
        for selected_case in cases:
            case = cases_by_arm[arm][selected_case.case_id]
            prompt = build_detail_prompt(case, arm, prompt_mode=prompt_mode)
            path = output_dir / backend_name / arm.value / f"{case.case_id}.json"
            base_payload = {
                "schema": DETAIL_ATTEMPT_SCHEMA,
                "backend": backend_name,
                "model": model_identity,
                "arm": arm.value,
                "manifest_sha256": manifest_hashes[arm],
                "case": _case_payload(case),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            base_sampling = {
                "expected_input_frames": expected_frames,
                "requested_fps": case.sample_fps,
                "requested_width": case.width,
                "requested_height": case.height,
                "max_new_tokens": max_new_tokens,
                "qwen_max_model_len": max_model_len,
            }
            started = perf_counter()
            try:
                evidence = backend.generate(
                    case.clip_path,
                    prompt,
                    requested_fps=case.sample_fps,
                    width=case.width,
                    height=case.height,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:
                payload = {
                    **base_payload,
                    "raw_response": None,
                    "parsed_response": None,
                    "parser_error": None,
                    "generation_error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": perf_counter() - started,
                    "sampling": {
                        **base_sampling,
                        "sampled_input_frames": None,
                        "width": None,
                        "height": None,
                        "visual_tokens": None,
                        "total_input_tokens": None,
                    },
                }
                _write_new_json(path, payload)
                print(path, flush=True)
                raise

            parsed_response = None
            parser_error = None
            try:
                parsed_response = asdict(parse_detail_reply(evidence.raw_response))
            except (TypeError, ValueError) as exc:
                parser_error = str(exc)
            payload = {
                **base_payload,
                "raw_response": evidence.raw_response,
                "parsed_response": parsed_response,
                "parser_error": parser_error,
                "generation_error": None,
                "elapsed_seconds": perf_counter() - started,
                "sampling": {
                    **base_sampling,
                    "sampled_input_frames": evidence.sampled_input_frames,
                    "width": evidence.width,
                    "height": evidence.height,
                    "visual_tokens": evidence.visual_tokens,
                    "total_input_tokens": evidence.total_input_tokens,
                },
            }
            _write_new_json(path, payload)
            print(path, flush=True)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qwen3-vl", "internvideo3"), required=True)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="ARM=MANIFEST",
        help="repeat once for short_only, deterministic, and broad_facts",
    )
    parser.add_argument(
        "--only-arm",
        action="append",
        dest="only_arms",
        choices=tuple(arm.value for arm in ARM_ORDER),
        help="run only this detail arm; repeat to select multiple arms (default: all arms)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--qwen-max-model-len", type=int, default=QWEN_MAX_MODEL_LEN)
    parser.add_argument(
        "--prompt-mode",
        choices=tuple(mode.value for mode in DetailPromptMode),
        default=DetailPromptMode.DEFAULT.value,
        help="detail prompt contract (conservative_replay_veto requires only short_only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    run_detail_trials(
        args.backend,
        _parse_arm_paths(args.arm),
        args.out,
        case_ids=None if args.case_ids is None else set(args.case_ids),
        limit=args.limit,
        selected_arms=(
            None
            if args.only_arms is None
            else tuple(DetailArm(raw_arm) for raw_arm in args.only_arms)
        ),
        max_new_tokens=args.max_new_tokens,
        qwen_max_model_len=args.qwen_max_model_len,
        prompt_mode=args.prompt_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
