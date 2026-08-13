"""One-command, fingerprinted coordinator for the issue 15 dataset builder."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
from typing import Protocol, TypeVar, cast

from dataset_builder.manifest import (
    build_stage_fingerprint,
    make_stage_record,
    record_stage,
    reuse_or_invalidate_stage,
    start_or_resume_run,
)
from dataset_builder.models import (
    InterpreterIdentity,
    RunManifest,
    SemanticValidation,
    StageFingerprint,
    StageOutcome,
)


PHASE_ORDER = (
    "search",
    "transcript",
    "triage",
    "selection",
    "download",
    "metadata",
    "commentary_cleaning",
    "tracknet_input",
    "shuttle",
    "pose",
    "court",
    "annotation",
    "commentary_pairing",
    "primitive_projection",
    "assembly",
    "report",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRAPER_WORKSPACE_NAME = "workspace"
_FINAL_PUBLICATIONS = {
    "assembly": ("rally_records.json.gz",),
    "report": ("dataset_builder_report.json.gz", "selected_videos.csv.gz"),
}

SemanticValidator = Callable[[Path], bool]
RuntimeFactory = Callable[["BuilderConfig", Path, str], "PipelineRuntime"]
ChoiceT = TypeVar("ChoiceT")


@dataclass(frozen=True)
class BuilderConfig:
    """Validated effective settings loaded from one TOML file."""

    source_dataset: str
    search_terms: dict[str, list[str]]
    search_count: int
    max_videos: int
    download_workers: int
    tracknet_python_environment: str
    pose_python_environment: str
    tracknet_dir: Path
    tracknet_model: Path
    inpaint_model: Path | None
    court_model: Path
    tracknet_workers: int
    tracknet_batch_size: int
    tracknet_stride: int
    tracknet_large_video: bool
    pose_device: str
    pose_n_max: int
    pose_shards: int
    court_device: str
    court_resize_mode: str
    commentary_enabled: bool
    commentary_api_key_environment: str


@dataclass(frozen=True)
class StageExecution:
    """Files and terminal outcome returned by one stage implementation."""

    outcome: StageOutcome
    outputs: Mapping[str, Path]
    counts: Mapping[str, int]
    reason: str | None = None
    semantic_validation: tuple[SemanticValidation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, StageOutcome):
            raise TypeError("stage execution outcome must be StageOutcome")
        if self.outcome is StageOutcome.PROCESSED and self.reason is not None:
            raise ValueError("processed stage execution cannot contain a reason")
        if self.outcome is not StageOutcome.PROCESSED and not self.reason:
            raise ValueError(f"stage execution {self.outcome.value!r} requires a reason")
        for name, value in self.counts.items():
            if not isinstance(name, str) or not name:
                raise ValueError("stage execution count names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"stage execution count {name!r} must be non-negative")


@dataclass(frozen=True)
class StagePlan:
    """Fingerprint inputs and callbacks for one global or per-video stage."""

    name: str
    contract_version: str
    dependencies: tuple[str, ...]
    command: tuple[str, ...]
    configuration: Mapping[str, object]
    interpreter: InterpreterIdentity
    model_weights: Mapping[str, Path]
    inputs: Mapping[str, Path]
    execute: Callable[[], StageExecution]
    restore: Callable[[], None]
    semantic_validators: Mapping[str, SemanticValidator]
    secret_values: tuple[str, ...] = ()
    failure_outcome: StageOutcome = StageOutcome.FAILED
    blocks_pipeline: bool = False
    on_failure: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.contract_version:
            raise ValueError("stage plan name and contract_version must be non-empty")
        if not self.command or any(not part for part in self.command):
            raise ValueError("stage plan command must contain non-empty arguments")
        if self.failure_outcome not in {StageOutcome.FAILED, StageOutcome.UNAVAILABLE}:
            raise ValueError("stage plan failure_outcome must be failed or unavailable")
        if any(not isinstance(value, str) or not value for value in self.secret_values):
            raise ValueError("stage plan secret_values must contain non-empty strings")


@dataclass(frozen=True)
class StageEvent:
    """One invocation or safe-reuse decision returned to the caller."""

    name: str
    outcome: StageOutcome
    reused: bool
    reason: str | None


@dataclass(frozen=True)
class DatasetBuilderRun:
    """Completed local coordinator result."""

    manifest: RunManifest
    events: tuple[StageEvent, ...]
    stopped_after: str | None = None
    terminal_error: str | None = None


class PipelineRuntime(Protocol):
    """Lazy real or fixture runtime consumed by the generic coordinator."""

    def preflight(self) -> None:
        """Validate required executables, models, and named environment variables."""

    def plans(self, phase: str, manifest: RunManifest) -> Sequence[StagePlan]:
        """Return source-ordered plans for one coordinator phase."""


def load_builder_config(path: Path, *, repo_root: Path = REPO_ROOT) -> BuilderConfig:
    """Read and strictly validate one dataset-builder TOML configuration."""
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    _exact_fields(
        payload,
        {"run", "search", "environment", "models", "vision", "commentary"},
        "dataset-builder configuration",
    )
    run = _section(payload, "run", {"source_dataset", "max_videos", "download_workers"})
    search = _section(payload, "search", {"result_count", "terms"})
    environment = _section(payload, "environment", {"tracknet_python", "pose_python"})
    models = _section(payload, "models", {"tracknet_dir", "tracknet", "inpaint", "court"})
    vision = _section(
        payload,
        "vision",
        {
            "tracknet_workers", "tracknet_batch_size", "tracknet_stride",
            "tracknet_large_video", "pose_device", "pose_n_max", "pose_shards", "court_device",
            "court_resize_mode",
        },
    )
    commentary = _section(payload, "commentary", {"enabled", "api_key_environment"})
    terms_payload = _object(search["terms"], "search.terms")
    if not terms_payload:
        raise ValueError("search.terms must contain at least one substream")
    search_terms = {
        _nonempty(name, "search substream"): _string_list(values, f"search.terms.{name}")
        for name, values in terms_payload.items()
    }
    inpaint_value = models["inpaint"]
    inpaint_model = None if inpaint_value in (None, "") else _repo_path(
        inpaint_value, "models.inpaint", repo_root,
    )
    return BuilderConfig(
        source_dataset=_nonempty(run["source_dataset"], "run.source_dataset"),
        search_terms=search_terms,
        search_count=_positive_integer(search["result_count"], "search.result_count"),
        max_videos=_positive_integer(run["max_videos"], "run.max_videos"),
        download_workers=_positive_integer(run["download_workers"], "run.download_workers"),
        tracknet_python_environment=_nonempty(
            environment["tracknet_python"], "environment.tracknet_python",
        ),
        pose_python_environment=_nonempty(
            environment["pose_python"], "environment.pose_python",
        ),
        tracknet_dir=_repo_path(models["tracknet_dir"], "models.tracknet_dir", repo_root),
        tracknet_model=_repo_path(models["tracknet"], "models.tracknet", repo_root),
        inpaint_model=inpaint_model,
        court_model=_repo_path(models["court"], "models.court", repo_root),
        tracknet_workers=_positive_integer(
            vision["tracknet_workers"], "vision.tracknet_workers",
        ),
        tracknet_batch_size=_positive_integer(
            vision["tracknet_batch_size"], "vision.tracknet_batch_size",
        ),
        tracknet_stride=_choice(vision["tracknet_stride"], {1, 8}, "vision.tracknet_stride"),
        tracknet_large_video=_boolean(
            vision["tracknet_large_video"], "vision.tracknet_large_video",
        ),
        pose_device=_choice(vision["pose_device"], {"cpu", "cuda"}, "vision.pose_device"),
        pose_n_max=_bounded_integer(vision["pose_n_max"], 1, 127, "vision.pose_n_max"),
        pose_shards=_positive_integer(vision["pose_shards"], "vision.pose_shards"),
        court_device=_choice(
            vision["court_device"], {"cpu", "cuda"}, "vision.court_device",
        ),
        court_resize_mode=_choice(
            vision["court_resize_mode"], {"pad", "squash"}, "vision.court_resize_mode",
        ),
        commentary_enabled=_boolean(commentary["enabled"], "commentary.enabled"),
        commentary_api_key_environment=_nonempty(
            commentary["api_key_environment"], "commentary.api_key_environment",
        ),
    )


def run_dataset_builder(
    config_path: Path,
    run_dir: Path,
    *,
    runtime_factory: RuntimeFactory | None = None,
    retry_unavailable: bool = False,
) -> DatasetBuilderRun:
    """Run or safely resume every configured dataset-builder phase."""
    config = load_builder_config(config_path)
    destination = Path(run_dir).resolve(strict=False)
    source_commit = _clean_source_commit(REPO_ROOT)
    workspace = destination / SCRAPER_WORKSPACE_NAME
    os.environ["BADMINTON_SCRAPE_DIR"] = os.fspath(workspace)
    factory = runtime_factory or _default_runtime_factory
    runtime = factory(config, destination, source_commit)
    runtime.preflight()
    manifest = start_or_resume_run(destination)
    events: list[StageEvent] = []
    stopped_after: str | None = None
    for phase in PHASE_ORDER:
        for plan in runtime.plans(phase, manifest):
            manifest, event = _run_stage_plan(
                destination,
                source_commit,
                manifest,
                plan,
                reuse_unavailable=not retry_unavailable,
            )
            events.append(event)
            if plan.blocks_pipeline and event.outcome is StageOutcome.FAILED:
                stopped_after = plan.name
                break
        if stopped_after is not None:
            break
    terminal_error = (
        None
        if stopped_after is not None
        else _empty_selected_run_reason(manifest)
    )
    return DatasetBuilderRun(manifest, tuple(events), stopped_after, terminal_error)


def _empty_selected_run_reason(manifest: RunManifest) -> str | None:
    stages = {stage.name: stage for stage in manifest.stages}
    selection = stages.get("selection")
    assembly = stages.get("assembly")
    if selection is None or assembly is None:
        return None
    selected = dict(selection.counts).get("selected")
    videos = dict(assembly.counts).get("videos")
    rallies = dict(assembly.counts).get("rallies")
    if selected is None or videos is None or rallies is None:
        return None
    if selected == 0:
        return "selection produced no videos"
    if videos == 0:
        return "every selected video was excluded before record assembly"
    if rallies == 0:
        return "selected videos produced no rally records"
    return None


def _run_stage_plan(
    run_dir: Path,
    source_commit: str,
    manifest: RunManifest,
    plan: StagePlan,
    *,
    reuse_unavailable: bool = False,
) -> tuple[RunManifest, StageEvent]:
    fingerprint = build_stage_fingerprint(
        source_commit=source_commit,
        contract_version=plan.contract_version,
        effective_configuration=plan.configuration,
        interpreter=plan.interpreter,
        model_weights=plan.model_weights,
        inputs=plan.inputs,
    )
    manifest, decision = reuse_or_invalidate_stage(
        run_dir,
        plan.name,
        fingerprint,
        semantic_validators=plan.semantic_validators,
        reuse_unavailable=reuse_unavailable,
    )
    _unpublish_invalidated_outputs(run_dir, decision.invalidated_stages)
    if decision.reusable:
        try:
            plan.restore()
        except Exception:
            manifest = _invalidate_unrestorable_stage(run_dir, plan.name, fingerprint)
        else:
            record = next(stage for stage in manifest.stages if stage.name == plan.name)
            return manifest, StageEvent(plan.name, record.outcome, True, decision.reason)

    started = time.monotonic()
    try:
        execution = plan.execute()
        validations = _validate_stage_outputs(run_dir, execution, plan.semantic_validators)
        execution = replace(
            execution,
            semantic_validation=(*execution.semantic_validation, *validations),
        )
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        if plan.on_failure is not None:
            plan.on_failure(reason)
        execution = StageExecution(plan.failure_outcome, {}, {}, reason)
    elapsed = time.monotonic() - started
    record = make_stage_record(
        name=plan.name,
        outcome=execution.outcome,
        fingerprint=fingerprint,
        run_dir=run_dir,
        command=plan.command,
        effective_configuration=plan.configuration,
        outputs=execution.outputs,
        dependencies=plan.dependencies,
        counts=execution.counts,
        elapsed_seconds=elapsed,
        semantic_validation=execution.semantic_validation,
        reason=execution.reason,
        secret_values=plan.secret_values,
    )
    manifest = record_stage(run_dir, record)
    return manifest, StageEvent(plan.name, execution.outcome, False, record.reason)


def _validate_stage_outputs(
    run_dir: Path,
    execution: StageExecution,
    validators: Mapping[str, SemanticValidator],
) -> tuple[SemanticValidation, ...]:
    validations: list[SemanticValidation] = []
    for name, validator in validators.items():
        try:
            passed = validator(run_dir)
        except Exception as error:
            raise ValueError(f"semantic validation {name!r} failed: {error}") from error
        if not passed:
            raise ValueError(f"semantic validation {name!r} failed")
        validations.append(SemanticValidation(name, True))
    return tuple(validations)


def _invalidate_unrestorable_stage(
    run_dir: Path,
    stage_name: str,
    fingerprint: StageFingerprint,
) -> RunManifest:
    replacement = "0" if fingerprint.digest[0] != "0" else "1"
    mismatched = replace(fingerprint, digest=f"{replacement}{fingerprint.digest[1:]}")
    manifest, decision = reuse_or_invalidate_stage(run_dir, stage_name, mismatched)
    if decision.reusable:
        raise AssertionError("mismatched fingerprint unexpectedly reused an unrestorable stage")
    _unpublish_invalidated_outputs(run_dir, decision.invalidated_stages)
    return manifest


def _unpublish_invalidated_outputs(
    run_dir: Path,
    invalidated_stages: Sequence[str],
) -> None:
    """Remove invalidated final artefacts before a later stage can fail."""
    filenames = {
        filename
        for stage_name in invalidated_stages
        for filename in _FINAL_PUBLICATIONS.get(stage_name, ())
    }
    for filename in sorted(filenames):
        path = Path(run_dir) / filename
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            raise ValueError(f"invalidated publication is not a file: {path}")


def _clean_source_commit(repo_root: Path) -> str:
    """Return HEAD only when tracked files exactly match that commit."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise RuntimeError(f"could not inspect tracked source state: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError("tracked files differ from HEAD; refusing to record a false source commit")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0 or not revision.stdout.strip():
        raise RuntimeError(f"could not resolve source commit: {revision.stderr.strip()}")
    return revision.stdout.strip()


def _default_runtime_factory(
    config: BuilderConfig,
    run_dir: Path,
    source_commit: str,
) -> PipelineRuntime:
    bst_x_root = REPO_ROOT / "src" / "bst_x"
    if os.fspath(bst_x_root) not in sys.path:
        sys.path.insert(0, os.fspath(bst_x_root))
    from dataset_builder._pipeline_runtime import DefaultPipelineRuntime

    return DefaultPipelineRuntime(config, run_dir, source_commit)


def _section(
    payload: Mapping[str, object],
    name: str,
    expected_fields: set[str],
) -> Mapping[str, object]:
    section = _object(payload.get(name), name)
    _exact_fields(section, expected_fields, name)
    return section


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a table with string keys")
    return value


def _exact_fields(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} fields differ: expected {sorted(expected)}, got {sorted(payload)}")


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    rows = [_nonempty(item, name) for item in value]
    if len(rows) != len(set(rows)):
        raise ValueError(f"{name} must not contain duplicates")
    return rows


def _positive_integer(value: object, name: str) -> int:
    number = _nonnegative_integer(value, name)
    if number == 0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _bounded_integer(value: object, minimum: int, maximum: int, name: str) -> int:
    number = _nonnegative_integer(value, name)
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return number


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _choice(value: object, choices: set[ChoiceT], name: str) -> ChoiceT:
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices, key=str)}")
    return cast(ChoiceT, value)


def _repo_path(value: object, name: str, repo_root: Path) -> Path:
    raw = Path(_nonempty(value, name))
    return raw if raw.is_absolute() else repo_root / raw


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run or resume one dataset build.")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument(
        "--retry-unavailable",
        action="store_true",
        help="Retry validated optional unavailable stages instead of reusing them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the one-command interface and return a process exit status."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command != "run":
        parser.error(f"unsupported command: {arguments.command}")
    try:
        result = run_dataset_builder(
            arguments.config,
            arguments.run_dir,
            retry_unavailable=arguments.retry_unavailable,
        )
    except Exception as error:
        print(f"dataset builder failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    if result.stopped_after is not None:
        print(f"dataset builder stopped after required stage {result.stopped_after}", file=sys.stderr)
        return 1
    if result.terminal_error is not None:
        print(f"dataset builder failed acceptance: {result.terminal_error}", file=sys.stderr)
        return 1
    print(f"dataset builder completed run {result.manifest.run_id}")
    return 0
