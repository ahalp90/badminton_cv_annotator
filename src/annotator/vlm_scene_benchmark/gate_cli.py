"""Apply the Issue 38 deployment gate without reading human truth."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .backends import backend_spec
from .contracts import BenchmarkRunRecord, RunOutcome, read_run_record
from .runtime import is_strict_json_response, parse_prediction_response, sha256_file
from .scoring import deployment_failures


SUPPORTED_BACKENDS = ("internvideo3", "qwen3-vl")


@dataclass(frozen=True)
class RecordTarget:
    """Expected backend and retained run-record path."""

    backend: str
    path: Path


def parse_target(value: str) -> RecordTarget:
    """Parse one ``BACKEND=PATH`` command-line value."""
    backend, separator, raw_path = value.partition("=")
    if separator != "=" or backend not in SUPPORTED_BACKENDS or not raw_path:
        choices = ", ".join(SUPPORTED_BACKENDS)
        raise argparse.ArgumentTypeError(f"expected BACKEND=PATH with backend in: {choices}")
    return RecordTarget(backend=backend, path=Path(raw_path))


def raw_response_path(record_path: Path, record: BenchmarkRunRecord) -> Path | None:
    """Return the final retained response path, if generation was attempted."""
    if record.attempt_count < 1:
        return None
    return record_path.with_name(f"{record_path.stem}.attempt-{record.attempt_count}.txt")


def _response_failures(record_path: Path, record: BenchmarkRunRecord) -> list[str]:
    """Verify retained responses against their digest and normalized segments."""
    failures: list[str] = []
    final_path = raw_response_path(record_path, record)
    if final_path is None or record.raw_response_sha256 is None:
        return ["no retained model response"]
    attempt_digests = record.attempt_response_sha256s
    if attempt_digests is None:
        if not final_path.is_file():
            return [f"retained model response is missing: {final_path.name}"]
        if sha256_file(final_path) != record.raw_response_sha256:
            failures.append(f"retained model response digest differs: {final_path.name}")
        if record.attempt_count > 1:
            failures.append("legacy run record does not authenticate every model response")
    else:
        for attempt, expected_digest in enumerate(attempt_digests, start=1):
            attempt_path = record_path.with_name(f"{record_path.stem}.attempt-{attempt}.txt")
            if not attempt_path.is_file():
                failures.append(f"retained model response is missing: {attempt_path.name}")
            elif sha256_file(attempt_path) != expected_digest:
                failures.append(f"retained model response digest differs: {attempt_path.name}")

    first_path = record_path.with_name(f"{record_path.stem}.attempt-1.txt")
    if not first_path.is_file():
        failures.append(f"first model response is missing: {first_path.name}")
    else:
        first_response = first_path.read_text(encoding="utf-8")
        first_is_strict = is_strict_json_response(first_response)
        if first_is_strict != record.first_attempt_valid_json:
            failures.append(
                "first_attempt_valid_json differs from the retained first response"
            )
        observed = record.observed_sampling
        sampled_frames = None if observed is None else observed.sampled_source_frames
        try:
            parse_prediction_response(first_response, record.shard, sampled_frames)
        except ValueError:
            first_is_valid_prediction = False
        else:
            first_is_valid_prediction = True
        if first_is_valid_prediction != record.first_attempt_valid_prediction:
            failures.append(
                "first_attempt_valid_prediction differs from the retained first response"
            )

    if record.outcome is RunOutcome.SUCCEEDED:
        observed = record.observed_sampling
        sampled_frames = None if observed is None else observed.sampled_source_frames
        try:
            parsed = parse_prediction_response(
                final_path.read_text(encoding="utf-8"),
                record.shard,
                sampled_frames,
            )
        except (OSError, ValueError) as error:
            failures.append(f"retained model response cannot reconstruct predictions: {error}")
        else:
            if parsed != record.segments:
                failures.append("retained model response differs from recorded predictions")
    return failures


def gate_failures(target: RecordTarget) -> list[str]:
    """Return deployment, identity, and raw-evidence failures for one record."""
    record = read_run_record(target.path)
    spec = backend_spec(target.backend)
    failures = deployment_failures(record)
    if record.model.model_id != spec.model_id:
        failures.append(f"model ID {record.model.model_id!r} differs from {spec.model_id!r}")
    if record.model.model_revision != spec.model_revision:
        failures.append(
            f"model revision {record.model.model_revision!r} differs from {spec.model_revision!r}"
        )
    if record.model.backend != spec.backend_name:
        failures.append(f"backend {record.model.backend!r} differs from {spec.backend_name!r}")
    if record.model.backend_version != spec.expected_backend_version:
        failures.append(
            f"backend version {record.model.backend_version!r} differs from "
            f"{spec.expected_backend_version!r}"
        )
    if record.runtime.cache_dtype != spec.cache_dtype:
        failures.append(
            f"cache dtype {record.runtime.cache_dtype!r} differs from {spec.cache_dtype!r}"
        )

    installed_versions = dict(record.runtime.package_versions)
    for package_name in spec.package_names:
        package_version = installed_versions.get(package_name)
        if package_version is None:
            failures.append(f"runtime package is missing: {package_name}")
        elif package_version == "not-installed":
            failures.append(f"runtime package is not installed: {package_name}")
    runtime_backend_version = installed_versions.get(spec.backend_distribution)
    if (
        runtime_backend_version is not None
        and runtime_backend_version != "not-installed"
        and runtime_backend_version != spec.expected_backend_version
    ):
        failures.append(
            f"runtime {spec.backend_distribution} version {runtime_backend_version!r} differs from "
            f"{spec.expected_backend_version!r}"
        )

    failures.extend(_response_failures(target.path, record))
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=parse_target, metavar="BACKEND=PATH")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    any_failed = False
    for target in args.records:
        try:
            failures = gate_failures(target)
        except (OSError, TypeError, ValueError) as error:
            failures = [f"cannot reload record: {error}"]
        if failures:
            any_failed = True
            print(f"FAIL {target.backend} {target.path}")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print(f"PASS {target.backend} {target.path}")
    return 4 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
