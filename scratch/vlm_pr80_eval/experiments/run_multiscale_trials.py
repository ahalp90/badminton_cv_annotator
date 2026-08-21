"""Run cut-aware broad-context cases through one pinned VLM backend."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from .multiscale_prompts import build_broad_prompt
from .multiscale_schema import load_manifest, parse_broad_reply, target_route

ATTEMPT_SCHEMA = "vlm-multiscale-attempt-v1"
MAX_NEW_TOKENS = 1_024
QWEN_MAX_MODEL_LEN = 16_384


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run_multiscale_trials(
    backend_name: str,
    manifest_path: Path,
    output_dir: Path,
    *,
    case_ids: set[str] | None = None,
    limit: int | None = None,
    qwen_max_model_len: int = QWEN_MAX_MODEL_LEN,
) -> None:
    """Load one model once and retain one immutable attempt per broad case."""
    cases = list(load_manifest(manifest_path))
    if case_ids is not None:
        cases = [case for case in cases if case.case_id in case_ids]
        missing = case_ids - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"unknown requested case IDs: {sorted(missing)}")
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("no multiscale cases selected")
    frame_counts = {case.expected_frames for case in cases}
    if len(frame_counts) != 1:
        raise ValueError(f"selected cases use different frame counts: {sorted(frame_counts)}")
    expected_frames = frame_counts.pop()
    max_model_len = qwen_max_model_len if backend_name == "qwen3-vl" else None
    backend = _load_backend(
        backend_name,
        expected_input_frames=expected_frames,
        max_model_len=max_model_len,
    )
    model_identity = asdict(backend.spec.identity(backend.backend_version))
    manifest_sha256 = _sha256(manifest_path)

    for case in cases:
        prompt = build_broad_prompt(case)
        path = output_dir / backend_name / f"{case.case_id}.json"
        base_payload = {
            "schema": ATTEMPT_SCHEMA,
            "backend": backend_name,
            "model": model_identity,
            "manifest_sha256": manifest_sha256,
            "case": {
                "case_id": case.case_id,
                "pair_id": case.pair_id,
                "video_id": case.video_id,
                "context_seconds": case.context_seconds,
                "clip_path": str(case.clip_path),
                "clip_sha256": _sha256(case.clip_path),
                "source_start_frame": case.source_start_frame,
                "source_end_frame": case.source_end_frame,
                "target_start_frame": case.target_start_frame,
                "target_end_frame": case.target_end_frame,
                "target_segment_ids": case.target_segment_ids,
            },
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        base_sampling = {
            "expected_input_frames": expected_frames,
            "requested_fps": case.sample_fps,
            "requested_width": case.width,
            "requested_height": case.height,
            "max_new_tokens": MAX_NEW_TOKENS,
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
                max_new_tokens=MAX_NEW_TOKENS,
            )
        except Exception as exc:
            payload = {
                **base_payload,
                "raw_response": None,
                "parsed_response": None,
                "parser_error": None,
                "generation_error": f"{type(exc).__name__}: {exc}",
                "target_route": "close_check",
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
        parsed = None
        try:
            parsed = parse_broad_reply(case, evidence.raw_response)
            parsed_response = [asdict(reply) for reply in parsed]
        except (TypeError, ValueError) as exc:
            parser_error = str(exc)
        payload = {
            **base_payload,
            "raw_response": evidence.raw_response,
            "parsed_response": parsed_response,
            "parser_error": parser_error,
            "generation_error": None,
            "target_route": target_route(case, parsed).value,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qwen3-vl", "internvideo3"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--qwen-max-model-len", type=int, default=QWEN_MAX_MODEL_LEN)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_multiscale_trials(
        args.backend,
        args.manifest,
        args.out,
        case_ids=None if args.case_ids is None else set(args.case_ids),
        limit=args.limit,
        qwen_max_model_len=args.qwen_max_model_len,
    )


if __name__ == "__main__":
    main()
