"""Run one pinned Issue 38 VLM backend and retain reloadable evidence."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import socket

from .backends import GenerationEvidence, SceneBackend, backend_spec, load_backend
from .contracts import (
    BenchmarkRunRecord,
    RunOutcome,
    RuntimeTelemetry,
    SamplingObservation,
    SamplingRequest,
    write_run_record,
)
from .prepare import PreparedShardManifest, read_manifest, resolve_model_video
from .prompts import build_correction_prompt, build_scene_prompt
from .runtime import (
    GpuMemoryMonitor,
    is_strict_json_response,
    package_versions,
    parse_prediction_response,
    write_raw_response,
)


DEFAULT_MAX_NEW_TOKENS = 32_768


def _absolute_observation(
    evidence: GenerationEvidence,
    manifest: PreparedShardManifest,
) -> SamplingObservation:
    frame_map = manifest.sampled_source_frames
    absolute_frames: list[int] = []
    for local_frame in evidence.sampled_input_frames:
        if not 0 <= local_frame < len(frame_map):
            raise ValueError(
                f"backend sampled input frame {local_frame} outside [0, {len(frame_map)})"
            )
        absolute_frames.append(frame_map[local_frame])
    gaps = [right - left for left, right in zip(absolute_frames, absolute_frames[1:])]
    complete = evidence.sampled_input_frames == tuple(range(manifest.model_video.frame_count))
    uniform = len(gaps) < 2 or max(gaps) - min(gaps) <= 1
    return SamplingObservation(
        sampled_source_frames=tuple(absolute_frames),
        width=evidence.width,
        height=evidence.height,
        visual_tokens=evidence.visual_tokens,
        total_input_tokens=evidence.total_input_tokens,
        complete_source_coverage=complete,
        uniform_frame_grid=uniform,
    )


def _same_visual_input(left: SamplingObservation, right: SamplingObservation) -> bool:
    return (
        left.sampled_source_frames == right.sampled_source_frames
        and left.width == right.width
        and left.height == right.height
        and left.visual_tokens == right.visual_tokens
        and left.complete_source_coverage == right.complete_source_coverage
        and left.uniform_frame_grid == right.uniform_frame_grid
    )


def _raw_path(output_path: Path, attempt: int) -> Path:
    return output_path.with_name(f"{output_path.stem}.attempt-{attempt}.txt")


def _attempt(
    backend: SceneBackend,
    video_path: Path,
    prompt: str,
    request: SamplingRequest,
    max_new_tokens: int,
) -> GenerationEvidence:
    return backend.generate(
        video_path,
        prompt,
        requested_fps=request.fps,
        width=request.width,
        height=request.height,
        max_new_tokens=max_new_tokens,
    )


def run_benchmark(
    backend_name: str,
    manifest_path: Path,
    output_path: Path,
    *,
    run_id: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_model_len: int | None = None,
) -> BenchmarkRunRecord:
    """Run at most two generations and always retain a backend outcome record."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if max_model_len is not None and max_model_len <= max_new_tokens:
        raise ValueError("maximum model length must exceed maximum new tokens")
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    retained_paths = [output_path, _raw_path(output_path, 1), _raw_path(output_path, 2)]
    existing = [path for path in retained_paths if path.exists()]
    if existing:
        raise FileExistsError(f"benchmark output already exists: {existing[0]}")
    manifest = read_manifest(manifest_path)
    video_path = resolve_model_video(manifest_path, manifest)
    request = SamplingRequest(
        fps=manifest.model_video.fps,
        width=manifest.model_video.width,
        height=manifest.model_video.height,
    )
    initial_prompt = build_scene_prompt(
        manifest.shard,
        manifest.sampled_source_frames,
        manifest.cut_frames,
    )
    spec = backend_spec(backend_name)
    installed_versions = package_versions(spec.package_names)
    backend_version = dict(installed_versions).get(spec.backend_distribution, "not-installed")
    monitor = GpuMemoryMonitor()
    started = perf_counter()
    backend: SceneBackend | None = None
    observed: SamplingObservation | None = None
    first_observed: SamplingObservation | None = None
    segments = ()
    attempt_count = 0
    first_attempt_valid = False
    first_attempt_valid_prediction = False
    raw_digest: str | None = None
    raw_digests: list[str] = []
    failure_reason: str | None = None

    monitor.start()
    try:
        backend = load_backend(
            backend_name,
            expected_input_frames=manifest.model_video.frame_count,
            max_model_len=max_model_len,
        )
        backend_version = backend.backend_version
        evidence = _attempt(backend, video_path, initial_prompt, request, max_new_tokens)
        attempt_count = 1
        raw_digest = write_raw_response(_raw_path(output_path, attempt_count), evidence.raw_response)
        raw_digests.append(raw_digest)
        observed = _absolute_observation(evidence, manifest)
        first_observed = observed
        first_attempt_valid = is_strict_json_response(evidence.raw_response)
        try:
            segments = parse_prediction_response(
                evidence.raw_response,
                manifest.shard,
                manifest.sampled_source_frames,
            )
            first_attempt_valid_prediction = True
        except ValueError as first_error:
            correction = build_correction_prompt(
                initial_prompt,
                str(first_error),
            )
            corrected = _attempt(backend, video_path, correction, request, max_new_tokens)
            attempt_count = 2
            raw_digest = write_raw_response(
                _raw_path(output_path, attempt_count),
                corrected.raw_response,
            )
            raw_digests.append(raw_digest)
            observed = _absolute_observation(corrected, manifest)
            if not _same_visual_input(first_observed, observed):
                raise ValueError("correction retry consumed a different visual input")
            segments = parse_prediction_response(
                corrected.raw_response,
                manifest.shard,
                manifest.sampled_source_frames,
            )
    except Exception as error:
        failure_reason = f"{type(error).__name__}: {error}"
    finally:
        monitor.stop()

    if monitor.error is not None:
        monitor_failure = f"GPU monitoring failed: {monitor.error}"
        if failure_reason is None:
            failure_reason = monitor_failure
        else:
            failure_reason = f"{failure_reason}; {monitor_failure}"

    elapsed = perf_counter() - started
    installed_versions = package_versions(spec.package_names)
    runtime = RuntimeTelemetry(
        hostname=socket.gethostname(),
        device_name=monitor.device_name,
        peak_vram_mib=monitor.peak_used_memory_mib,
        elapsed_seconds=elapsed,
        cpu_offload=False if backend is None else backend.cpu_offload,
        cache_dtype=None if backend is None else backend.cache_dtype,
        package_versions=installed_versions,
    )
    base = BenchmarkRunRecord(
        run_id=run_id,
        outcome=RunOutcome.FAILED,
        model=spec.identity(backend_version),
        shard=manifest.shard,
        requested_sampling=request,
        observed_sampling=observed,
        runtime=runtime,
        attempt_count=attempt_count,
        first_attempt_valid_json=first_attempt_valid,
        first_attempt_valid_prediction=first_attempt_valid_prediction,
        raw_response_sha256=raw_digest,
        failure_reason=failure_reason or "backend did not produce a valid prediction partition",
        segments=(),
        attempt_response_sha256s=tuple(raw_digests),
    )
    if failure_reason is None:
        base = replace(
            base,
            outcome=RunOutcome.SUCCEEDED,
            failure_reason=None,
            segments=segments,
        )
    write_run_record(output_path, base)
    return base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("internvideo3", "qwen3-vl"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-model-len", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = run_benchmark(
            args.backend,
            args.manifest,
            args.out,
            run_id=args.run_id,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        build_parser().error(str(error))
    print(args.out)
    return 0 if record.outcome is RunOutcome.SUCCEEDED else 4


if __name__ == "__main__":
    raise SystemExit(main())
