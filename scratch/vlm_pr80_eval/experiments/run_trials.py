"""Run frozen VLM cleanup cases with one resident PR 80 backend."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from .prompts import build_prompt
from .trial_schema import (
    ATTEMPT_SCHEMA,
    EXPECTED_FRAMES,
    HEIGHT,
    WIDTH,
    TrialArm,
    load_manifest,
    parse_reply,
)

MAX_NEW_TOKENS = 384
QWEN_MAX_MODEL_LEN = 16_384


def _load_backend(
    backend_name: str, *, expected_input_frames: int, max_model_len: int | None
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run_trials(
    backend_name: str,
    manifest_path: Path,
    output_dir: Path,
    *,
    case_ids: set[str] | None = None,
    limit: int | None = None,
    arms: Sequence[TrialArm] = tuple(TrialArm),
) -> None:
    """Load one model once and retain an immutable record for every case and arm."""
    cases = list(load_manifest(manifest_path))
    if case_ids is not None:
        cases = [case for case in cases if case.case_id in case_ids]
        missing = case_ids - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"unknown requested case IDs: {sorted(missing)}")
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("no trial cases selected")

    max_model_len = QWEN_MAX_MODEL_LEN if backend_name == "qwen3-vl" else None
    backend = _load_backend(
        backend_name,
        expected_input_frames=EXPECTED_FRAMES,
        max_model_len=max_model_len,
    )
    model_identity = asdict(backend.spec.identity(backend.backend_version))

    if not arms:
        raise ValueError("at least one trial arm is required")
    if len(set(arms)) != len(arms):
        raise ValueError("trial arms must be unique")

    for case in cases:
        for arm in arms:
            prompt = build_prompt(case, arm)
            path = output_dir / backend_name / f"{case.case_id}--{arm.value}.json"
            base_payload = {
                "schema": ATTEMPT_SCHEMA,
                "backend": backend_name,
                "model": model_identity,
                "case": {
                    "case_id": case.case_id,
                    "kind": case.kind.value,
                    "video_id": case.video_id,
                    "clip_path": str(case.clip_path),
                    "clip_sha256": _sha256(case.clip_path),
                    "source_start_frame": case.source_start_frame,
                    "source_end_frame": case.source_end_frame,
                    "candidate_frame": case.candidate_frame,
                },
                "arm": arm.value,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            base_sampling = {
                "requested_fps": case.sample_fps,
                "max_new_tokens": MAX_NEW_TOKENS,
                "qwen_max_model_len": max_model_len,
            }
            started = perf_counter()
            try:
                evidence = backend.generate(
                    case.clip_path,
                    prompt,
                    requested_fps=case.sample_fps,
                    width=WIDTH,
                    height=HEIGHT,
                    max_new_tokens=MAX_NEW_TOKENS,
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
            elapsed = perf_counter() - started
            parsed = None
            parser_error = None
            try:
                parsed = parse_reply(case.kind, evidence.raw_response)
            except (TypeError, ValueError) as exc:
                parser_error = str(exc)
            payload = {
                **base_payload,
                "raw_response": evidence.raw_response,
                "parsed_response": parsed,
                "parser_error": parser_error,
                "generation_error": None,
                "elapsed_seconds": elapsed,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("internvideo3", "qwen3-vl"), required=True
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--arm",
        action="append",
        choices=tuple(TrialArm),
        dest="arms",
        help="prompt arm to run; repeat for several (default: both)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    run_trials(
        args.backend,
        args.manifest,
        args.out,
        case_ids=None if args.case_ids is None else set(args.case_ids),
        limit=args.limit,
        arms=tuple(TrialArm) if args.arms is None else tuple(map(TrialArm, args.arms)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
