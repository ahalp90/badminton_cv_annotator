"""Resumable multi-fixture calibration sweep for Issue 96.

The legacy sweep remains the single-fixture compatibility path. This module
adds corpus aggregation and durable candidate attempts without changing the
production annotator or its selection rules.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from annotator.calibration import selection, sweep
from annotator.calibration.fixtures import FIXTURES, SHARED_FILES, Fixture
from annotator.calibration.gt_scoring import (
    RunVideoInputs,
    canonical_tolerance,
    flatten_metrics,
    load_gt_tables,
    score_video,
)
from annotator.calibration.scoring import load_gt_rallies, safe_f1
from annotator.run_video import run_video


RUN_SCHEMA = "annotator-corpus-sweep/1"
ATTEMPT_SCHEMA = "annotator-corpus-sweep-attempt/1"
MINIMUM_FIXTURE_COVERAGE = 0.6


@dataclass(frozen=True)
class CorpusFixtureInput:
    """One digest-validated fixture and its invariant production inputs."""

    fixture: Fixture
    inputs: RunVideoInputs
    guard_rejected_frames: int
    inpaint_filled_frames: int
    artifact_index: str
    artifact_index_sha256: str


@dataclass(frozen=True)
class CandidateOutcome:
    """One worker result before it is written to the attempt journal."""

    status: str
    elapsed_seconds: float
    aggregate: Mapping[str, Any] | None = None
    fixtures: Mapping[str, Mapping[str, Any]] | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None


CandidateRunner = Callable[[sweep.CandidateSpec, Sequence[CorpusFixtureInput]], CandidateOutcome]

_POOL_FIXTURES: tuple[CorpusFixtureInput, ...] = ()


def _pool_initialiser(fixtures: tuple[CorpusFixtureInput, ...]) -> None:
    global _POOL_FIXTURES
    _POOL_FIXTURES = fixtures


def _pool_candidate(spec: sweep.CandidateSpec) -> CandidateOutcome:
    if not _POOL_FIXTURES:
        raise RuntimeError("corpus process pool was not initialised")
    return run_production_candidate(spec, _POOL_FIXTURES)


def candidate_id(spec: sweep.CandidateSpec) -> str:
    """Return the stable identity of one fully routed configuration."""
    encoded = _canonical_json(sweep.serialise_spec(spec)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def search_space_document() -> dict[str, object]:
    """Return the exact ordered search space used by the corpus runner."""
    return {
        "boundary_keys": list(sweep.BOUNDARY_KEYS),
        "boundary_values": {
            key: list(sweep.BOUNDARY_VALUES[key]) for key in sweep.BOUNDARY_KEYS
        },
        "contact_keys": list(sweep.CONTACT_KEYS),
        "contact_values": {
            key: list(sweep.CONTACT_VALUES[key]) for key in sweep.CONTACT_KEYS
        },
        "boundary_candidates": len(sweep.build_boundary_grid()),
        "contact_candidates": len(
            sweep.build_contact_grid(
                {key: sweep.BOUNDARY_VALUES[key][0] for key in sweep.BOUNDARY_KEYS}
            )
        ),
        "serve_start": None,
    }


def build_run_document(
    *,
    source_revision: str,
    source_diff_sha256: str,
    fixtures: Sequence[CorpusFixtureInput],
) -> dict[str, object]:
    """Build the immutable run identity checked on every resume."""
    fixture_rows = []
    for item in fixtures:
        fixture_rows.append(
            {
                "name": item.fixture.name,
                "video_id": item.fixture.video_id,
                "fps": item.fixture.fps,
                "n_rallies": item.fixture.n_rallies,
                "n_strokes": item.fixture.n_strokes,
                "artifact_index": item.artifact_index,
                "artifact_index_sha256": item.artifact_index_sha256,
                "guard_rejected_frames": item.guard_rejected_frames,
                "inpaint_filled_frames": item.inpaint_filled_frames,
            }
        )
    return {
        "schema": RUN_SCHEMA,
        "source_revision": source_revision,
        "source_diff_sha256": source_diff_sha256,
        "fixtures": fixture_rows,
        "ground_truth_digests": {str(pin.path): pin.md5 for pin in SHARED_FILES},
        "search_space": search_space_document(),
        "minimum_fixture_coverage": MINIMUM_FIXTURE_COVERAGE,
        "baseline": sweep.serialise_spec(sweep.shipped_spec()),
    }


def load_pinned_replay_fixtures(
    *,
    config_path: Path,
    run_dir: Path,
    pinned_repo_dir: Path,
    source_root: Path,
    source_revision: str,
) -> tuple[CorpusFixtureInput, ...]:
    """Validate all replay indexes, then retain the three ground-truth videos.

    ``BADMINTON_SCRAPE_DIR`` must be exported before Python starts because the
    annotator path contract is bound during module import.
    """
    from annotator.config import BaseAnnotatorConfig
    from annotator.point_winner import SHIPPED_LANDING_FILTER_OPTIONS
    from dataset_builder._pipeline_runtime import DefaultPipelineRuntime
    from dataset_builder.artifact_index import artifact_index_path
    from dataset_builder.cli import SCRAPER_WORKSPACE_NAME, load_builder_config
    from dataset_builder.manifest import load_run_manifest
    from dataset_builder.shuttle_quality import summarize_shuttle_quality

    destination = Path(run_dir).resolve(strict=True)
    workspace = destination / SCRAPER_WORKSPACE_NAME
    bound_workspace = os.environ.get("BADMINTON_SCRAPE_DIR")
    if bound_workspace is None or Path(bound_workspace).resolve(strict=False) != workspace:
        raise RuntimeError(
            "BADMINTON_SCRAPE_DIR must name the Issue 103 workspace before Python starts"
        )
    pinned_repo = Path(pinned_repo_dir).resolve(strict=True)
    config = load_builder_config(config_path, repo_root=pinned_repo)
    if config.fixed_sources is None:
        raise ValueError("Issue 96 requires the fixed-source Issue 103 configuration")
    os.environ[config.fixed_sources.source_root_environment] = os.fspath(
        Path(source_root).resolve(strict=True)
    )
    runtime = DefaultPipelineRuntime(config, destination, source_revision)
    runtime.preflight_replay()
    manifest = load_run_manifest(destination)
    selected_ids = runtime.prepare_annotation_replay(manifest)
    if selected_ids != config.fixed_sources.video_ids:
        raise ValueError("restored replay video IDs differ from the fixed configuration")
    expected_fixture_ids = {fixture.name for fixture in FIXTURES}
    if not expected_fixture_ids.issubset(selected_ids):
        missing = sorted(expected_fixture_ids.difference(selected_ids))
        raise ValueError(f"Issue 103 replay inputs omit calibration fixtures: {missing}")

    master, _homo, _courts, _resolution = load_gt_tables()
    base = BaseAnnotatorConfig()
    loaded: list[CorpusFixtureInput] = []
    for fixture in FIXTURES:
        video_id = fixture.name
        metadata = runtime.state.metadata[video_id]
        if float(metadata.fps) != fixture.fps:
            raise ValueError(
                f"{video_id}: artifact FPS {float(metadata.fps)} differs from GT {fixture.fps}"
            )
        gt_rallies = load_gt_rallies(master, fixture.video_id)
        last_gt_frame = max(frame for rally in gt_rallies for frame in rally.stroke_frames)
        if last_gt_frame >= metadata.frame_count:
            raise ValueError(
                f"{video_id}: GT frame {last_gt_frame} exceeds artifact frame count {metadata.frame_count}"
            )
        shuttle = runtime.state.shuttles[video_id]
        pose = runtime.state.poses[video_id]
        court = runtime.state.courts[video_id]
        court_inputs = court.evidence.inputs
        if court_inputs is None:
            raise ValueError(f"{video_id}: pinned court evidence has no operational inputs")
        quality = summarize_shuttle_quality(
            shuttle.track,
            shuttle.inpaint_fill_mask,
            shuttle.guard_codes,
            base.rejected_grades,
        )
        cut_frames = [end for _start, end in court.raw_cuts[:-1]]
        keyword: dict[str, object] = {
            "fps": float(metadata.fps),
            "landing_options": SHIPPED_LANDING_FILTER_OPTIONS,
            "net_band": court_inputs.net_band,
            "resolution": court_inputs.resolution,
            "video_id": video_id,
            "court_info": court_inputs.court_info,
            "homo_df": None,
            "gate_court_info": court_inputs.gate_court_info,
            "gate_resolution_table": court_inputs.gate_resolution_table,
            "ref_err_px": 3.5,
            "raw_exclusion_mask": None,
            "court_present": court.evidence.court_present,
            "homography_rows": court_inputs.homography_rows,
            "cut_frames": cut_frames,
            "keep_vote": court.evidence.keep_vote,
            "inpaint_codes": shuttle.guard_codes,
            "court_invalid_is_excluded": True,
            "landing_error_band_m": court_inputs.landing_error_band_m,
        }
        inputs = RunVideoInputs(
            (shuttle.track, pose.bboxes, pose.scores, pose.kps, pose.ndet),
            keyword,
            master,
            {fixture.video_id: court_inputs.court_info},
        )
        index = artifact_index_path(destination, video_id)
        loaded.append(
            CorpusFixtureInput(
                fixture=fixture,
                inputs=inputs,
                guard_rejected_frames=quality.guard_rejected_frames,
                inpaint_filled_frames=quality.inpaint_filled_frames,
                artifact_index=os.fspath(index),
                artifact_index_sha256=_sha256_file(index),
            )
        )
    return tuple(loaded)


def initialise_run(out_dir: Path, document: Mapping[str, object]) -> None:
    """Create or validate the immutable run document without overwriting it."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run.json"
    expected = _canonical_json(document) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError("existing corpus sweep run.json differs from requested inputs")
        return
    _atomic_write_text(path, expected)


def run_production_candidate(
    spec: sweep.CandidateSpec,
    fixtures: Sequence[CorpusFixtureInput],
) -> CandidateOutcome:
    """Run one configuration on every fixture and aggregate integer counts."""
    started = time.monotonic()
    try:
        rows: dict[str, Mapping[str, Any]] = {}
        for item in fixtures:
            base, serve_start = sweep._base_and_serve(spec)
            keyword = dict(item.inputs.keyword)
            keyword.update(base=base, serve_start=serve_start)
            result = run_video(*item.inputs.positional, **keyword)
            row = sweep._row_for_result(
                item.fixture,
                spec,
                result,
                item.inputs.master,
            )
            row["_full_metrics"] = flatten_metrics(
                score_video(
                    item.fixture,
                    result,
                    item.inputs.master,
                    item.inputs.courts,
                    canonical_tolerance(item.fixture.fps),
                )
            )
            rows[item.fixture.name] = row
        aggregate = aggregate_fixture_rows(spec, fixtures, rows)
        return CandidateOutcome(
            status="succeeded",
            elapsed_seconds=time.monotonic() - started,
            aggregate=aggregate,
            fixtures={name: _serialisable_fixture_row(row) for name, row in rows.items()},
        )
    except Exception as error:  # noqa: BLE001 - failure evidence is part of the contract
        return CandidateOutcome(
            status="failed",
            elapsed_seconds=time.monotonic() - started,
            error_type=type(error).__name__,
            error_message=str(error),
            error_traceback=traceback.format_exc(),
        )


def aggregate_fixture_rows(
    spec: sweep.CandidateSpec,
    fixtures: Sequence[CorpusFixtureInput],
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Micro-average fixture rows using their integer scoring counts."""
    expected = {item.fixture.name for item in fixtures}
    if set(rows) != expected:
        raise ValueError("candidate fixture rows differ from the pinned fixture set")
    n_gt_rallies = sum(item.fixture.n_rallies for item in fixtures)
    covered = _sum(rows, "covered")
    clean_covered = _sum(rows, "clean_covered")
    spurious_spans = _sum(rows, "spurious_spans")
    strict_denominator = n_gt_rallies + clean_covered + spurious_spans
    offsets: list[float] = []
    for row in rows.values():
        offsets.extend(float(value) for value in row["_clean_offsets"].values())
    aggregate: dict[str, Any] = {
        "label": spec.label,
        "candidate_id": candidate_id(spec),
        "spec": sweep.serialise_spec(spec),
        "n_gt_rallies": n_gt_rallies,
        "n_spans": _sum(rows, "n_spans"),
        "covered": covered,
        "covered_fraction": covered / n_gt_rallies,
        "clean_covered": clean_covered,
        "split": _sum(rows, "split"),
        "missed": _sum(rows, "missed"),
        "merged_spans": _sum(rows, "merged_spans"),
        "spurious_spans": spurious_spans,
        "swallowed_rallies": _sum(rows, "swallowed_rallies"),
        "strict_f1": 0.0 if strict_denominator == 0 else 2 * clean_covered / strict_denominator,
        "strict_align_median": float(np.median(offsets)) if offsets else None,
        "strict_align_p90": float(np.percentile(offsets, 90)) if offsets else None,
        "changed_from_defaults": sweep._changed_from_defaults(spec),
        "guard_rejected_frames": sum(item.guard_rejected_frames for item in fixtures),
        "inpaint_filled_frames": sum(item.inpaint_filled_frames for item in fixtures),
    }
    fixture_strict_f1: dict[str, float] = {}
    fixture_coverage: dict[str, float] = {}
    for item in fixtures:
        row = rows[item.fixture.name]
        fixture_strict_f1[item.fixture.name] = _strict_f1(row, item.fixture.n_rallies)
        fixture_coverage[item.fixture.name] = float(row["covered_fraction"])
    aggregate["fixture_strict_f1"] = fixture_strict_f1
    aggregate["fixture_coverage"] = fixture_coverage
    fixture_boundary_counts: dict[str, dict[str, int]] = {}
    for item in fixtures:
        fixture_boundary_counts[item.fixture.name] = {
            "n_gt_rallies": item.fixture.n_rallies,
            "clean_covered": int(rows[item.fixture.name]["clean_covered"]),
            "spurious_spans": int(rows[item.fixture.name]["spurious_spans"]),
        }
    aggregate["fixture_boundary_counts"] = fixture_boundary_counts
    aggregate["minimum_fixture_strict_f1"] = min(fixture_strict_f1.values())
    aggregate["minimum_fixture_coverage"] = min(fixture_coverage.values())
    for band in sweep.CONTACT_TOLERANCES_BASE30:
        _add_contact_aggregate(aggregate, rows, band)
    if all("_full_metrics" in row for row in rows.values()):
        _add_full_metric_aggregates(aggregate, rows)
    return aggregate


def boundary_selection_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    """Return the deterministic corpus boundary ordering."""
    alignment = row["strict_align_median"]
    return (
        -float(row["strict_f1"]),
        -float(row["minimum_fixture_strict_f1"]),
        float("inf") if alignment is None else float(alignment),
        int(row["changed_from_defaults"]),
        str(row["candidate_id"]),
    )


def contact_selection_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    """Return the deterministic corpus raw-contact ordering."""
    return (
        -float(row["f1_raw_5"]),
        -float(row["minimum_fixture_f1_raw_5"]),
        int(row["changed_from_defaults"]),
        str(row["candidate_id"]),
    )


def select_boundary(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Select a complete grid row that clears every fixture coverage floor."""
    eligible = [
        row
        for row in rows
        if row["label"] == selection.GRID_LABEL
        and float(row["minimum_fixture_coverage"]) >= MINIMUM_FIXTURE_COVERAGE
    ]
    return None if not eligible else min(eligible, key=boundary_selection_key)


def select_contact(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Select a complete grid row with measurable contact metrics."""
    eligible = []
    for row in rows:
        if row["label"] != selection.GRID_LABEL:
            continue
        if row["f1_raw_5"] is None or row["minimum_fixture_f1_raw_5"] is None:
            continue
        eligible.append(row)
    return None if not eligible else min(eligible, key=contact_selection_key)


def decision_evidence(
    *,
    phase: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the fixed baseline, one-fixture, and downstream decision gates."""
    if phase not in {"boundary", "contact"}:
        raise ValueError(f"unsupported decision phase {phase!r}")
    metric = "strict_f1" if phase == "boundary" else "f1_raw_5"
    fixture_metric = "fixture_strict_f1" if phase == "boundary" else "fixture_f1_raw_5"
    baseline_fixture = baseline[fixture_metric]
    candidate_fixture = candidate[fixture_metric]
    deltas = {
        name: float(candidate_fixture[name]) - float(baseline_fixture[name])
        for name in baseline_fixture
    }
    leave_one_out: dict[str, float] = {}
    for excluded in baseline_fixture:
        baseline_score = _subset_objective(baseline, phase, excluded)
        candidate_score = _subset_objective(candidate, phase, excluded)
        leave_one_out[excluded] = candidate_score - baseline_score
    strongest_fixture = min(leave_one_out, key=leave_one_out.__getitem__)
    downstream = {}
    for name in ("player", "server", "hit_height", "landing", "getpoint"):
        key = f"{name}_primary_correct"
        baseline_correct = int(baseline[key])
        candidate_correct = int(candidate[key])
        downstream[name] = {
            "baseline_correct": baseline_correct,
            "candidate_correct": candidate_correct,
            "delta_correct": candidate_correct - baseline_correct,
            "passed": candidate_correct >= baseline_correct,
        }
    improves_overall = float(candidate[metric]) > float(baseline[metric])
    survives_strongest_removal = leave_one_out[strongest_fixture] > 0.0
    downstream_passed = all(item["passed"] for item in downstream.values())
    eligible = improves_overall and survives_strongest_removal and downstream_passed
    return {
        "schema": "annotator-corpus-sweep-decision/1",
        "phase": phase,
        "objective": metric,
        "baseline_candidate_id": baseline["candidate_id"],
        "provisional_candidate_id": candidate["candidate_id"],
        "baseline_objective": baseline[metric],
        "candidate_objective": candidate[metric],
        "objective_delta": float(candidate[metric]) - float(baseline[metric]),
        "fixture_objective_deltas": deltas,
        "strongest_fixture": strongest_fixture,
        "leave_one_out_objective_deltas": leave_one_out,
        "improves_overall": improves_overall,
        "survives_strongest_fixture_removal": survives_strongest_removal,
        "downstream_primary_correctness": downstream,
        "downstream_guardrails_passed": downstream_passed,
        "eligible_for_best_config": eligible,
        "recommended_spec": candidate["spec"] if eligible else baseline["spec"],
    }


def run_phase(
    *,
    phase: str,
    specs: Sequence[sweep.CandidateSpec],
    fixtures: Sequence[CorpusFixtureInput],
    out_dir: Path,
    workers: int,
    retry_failed: bool = False,
    candidate_runner: CandidateRunner = run_production_candidate,
) -> list[Mapping[str, Any]]:
    """Run or resume one phase and return every successful aggregate row."""
    if phase not in {"preflight", "boundary", "contact"}:
        raise ValueError(f"unsupported corpus sweep phase {phase!r}")
    if workers < 1:
        raise ValueError("workers must be at least one")
    pending: list[sweep.CandidateSpec] = []
    for spec in specs:
        status = latest_attempt(out_dir, phase, spec)
        if status is None or (retry_failed and status["status"] == "failed"):
            pending.append(spec)
    if workers > 1 and candidate_runner is run_production_candidate:
        _run_process_pool(phase, pending, tuple(fixtures), out_dir, workers)
    else:
        for spec in pending:
            outcome = candidate_runner(spec, fixtures)
            write_attempt(out_dir, phase, spec, outcome)
    attempts = [latest_attempt(out_dir, phase, spec) for spec in specs]
    failures = [attempt for attempt in attempts if attempt is None or attempt["status"] != "succeeded"]
    if failures:
        raise RuntimeError(f"{len(failures)} {phase} configurations lack successful evidence")
    return [attempt["aggregate"] for attempt in attempts if attempt is not None]


def run_calibration(
    *,
    phase: str,
    fixtures: Sequence[CorpusFixtureInput],
    out_dir: Path,
    workers: int,
    retry_failed: bool,
) -> dict[str, Mapping[str, Any]]:
    """Run the requested sweep phases and persist provisional selections."""
    selections: dict[str, Mapping[str, Any]] = {}
    boundary_winner: Mapping[str, Any] | None = None
    boundary_recommendation: object | None = None
    if phase in {"preflight", "boundary", "both"}:
        if phase == "preflight":
            grid = sweep.build_boundary_grid()
            indices = (0, len(grid) // 2, len(grid) - 1)
            specs = [sweep.shipped_spec(), *(grid[index] for index in indices)]
            rows = run_phase(
                phase="preflight",
                specs=specs,
                fixtures=fixtures,
                out_dir=out_dir,
                workers=workers,
                retry_failed=retry_failed,
            )
            summary = {
                "schema": "annotator-corpus-sweep-preflight/1",
                "candidate_count": len(rows),
                "candidate_ids": [row["candidate_id"] for row in rows],
            }
            _write_or_validate_json(out_dir / "preflight.json", summary)
            return selections
        boundary_rows = run_phase(
            phase="boundary",
            specs=[*sweep.build_boundary_grid(), sweep.shipped_spec()],
            fixtures=fixtures,
            out_dir=out_dir,
            workers=workers,
            retry_failed=retry_failed,
        )
        boundary_winner = select_boundary(boundary_rows)
        if boundary_winner is None:
            raise RuntimeError("no boundary candidate clears every fixture coverage floor")
        selections["boundary"] = boundary_winner
        _write_or_validate_json(out_dir / "boundary_selection.json", boundary_winner)
        baseline = _reference_row(boundary_rows)
        boundary_decision = decision_evidence(
            phase="boundary",
            baseline=baseline,
            candidate=boundary_winner,
        )
        _write_or_validate_json(out_dir / "boundary_decision.json", boundary_decision)
        boundary_recommendation = boundary_decision["recommended_spec"]
    if phase in {"contact", "both"}:
        if boundary_recommendation is None:
            boundary_decision = _load_json(out_dir / "boundary_decision.json")
            boundary_recommendation = boundary_decision["recommended_spec"]
        boundary_spec = _spec_from_payload(boundary_recommendation, selection.GRID_LABEL)
        contact_rows = run_phase(
            phase="contact",
            specs=[*sweep.build_contact_grid(boundary_spec.overrides_base30), sweep.shipped_spec()],
            fixtures=fixtures,
            out_dir=out_dir,
            workers=workers,
            retry_failed=retry_failed,
        )
        contact_winner = select_contact(contact_rows)
        if contact_winner is None:
            raise RuntimeError("no contact candidate has measurable metrics on every fixture")
        selections["contact"] = contact_winner
        _write_or_validate_json(out_dir / "contact_selection.json", contact_winner)
        baseline = _reference_row(contact_rows)
        contact_decision = decision_evidence(
            phase="contact",
            baseline=baseline,
            candidate=contact_winner,
        )
        _write_or_validate_json(out_dir / "contact_decision.json", contact_decision)
    return selections


def write_attempt(out_dir: Path, phase: str, spec: sweep.CandidateSpec, outcome: CandidateOutcome) -> Path:
    """Append one immutable atomic attempt file for a candidate."""
    directory = _candidate_directory(out_dir, phase, spec)
    directory.mkdir(parents=True, exist_ok=True)
    attempts = sorted(directory.glob("attempt-*.json.gz"))
    path = directory / f"attempt-{len(attempts) + 1:04d}.json.gz"
    if path.exists():
        raise FileExistsError(f"candidate attempt already exists: {path}")
    payload = {
        "schema": ATTEMPT_SCHEMA,
        "candidate_id": candidate_id(spec),
        "spec": sweep.serialise_spec(spec),
        "status": outcome.status,
        "elapsed_seconds": outcome.elapsed_seconds,
        "aggregate": outcome.aggregate,
        "fixtures": outcome.fixtures,
        "error": None
        if outcome.status == "succeeded"
        else {
            "type": outcome.error_type,
            "message": outcome.error_message,
            "traceback": outcome.error_traceback,
        },
    }
    _atomic_write_json_gz(path, payload)
    return path


def latest_attempt(out_dir: Path, phase: str, spec: sweep.CandidateSpec) -> dict[str, Any] | None:
    """Load the newest complete attempt and validate its candidate identity."""
    directory = _candidate_directory(out_dir, phase, spec)
    attempts = sorted(directory.glob("attempt-*.json.gz"))
    if not attempts:
        return None
    with gzip.open(attempts[-1], "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected_id = candidate_id(spec)
    if payload.get("schema") != ATTEMPT_SCHEMA or payload.get("candidate_id") != expected_id:
        raise ValueError(f"candidate attempt identity mismatch: {attempts[-1]}")
    if payload.get("spec") != sweep.serialise_spec(spec):
        raise ValueError(f"candidate attempt spec mismatch: {attempts[-1]}")
    if payload.get("status") not in {"succeeded", "failed"}:
        raise ValueError(f"candidate attempt status is invalid: {attempts[-1]}")
    return payload


def _run_process_pool(
    phase: str,
    specs: Sequence[sweep.CandidateSpec],
    fixtures: tuple[CorpusFixtureInput, ...],
    out_dir: Path,
    workers: int,
) -> None:
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_pool_initialiser,
        initargs=(fixtures,),
    ) as executor:
        future_specs = {executor.submit(_pool_candidate, spec): spec for spec in specs}
        for future in as_completed(future_specs):
            spec = future_specs[future]
            try:
                outcome = future.result()
            except Exception as error:  # noqa: BLE001 - retain broken-worker evidence
                outcome = CandidateOutcome(
                    status="failed",
                    elapsed_seconds=0.0,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    error_traceback=traceback.format_exc(),
                )
            write_attempt(out_dir, phase, spec, outcome)


def _add_contact_aggregate(
    aggregate: dict[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    band: int,
) -> None:
    matched = _sum(rows, f"_contact_matched_{band}")
    gt = _sum(rows, f"_contact_gt_{band}")
    candidates = _sum(rows, f"_contact_candidates_{band}")
    raw_matched = _sum(rows, f"_contact_raw_matched_{band}")
    raw_candidates = _sum(rows, f"_contact_raw_candidates_{band}")
    recall = _ratio(matched, gt)
    precision = _ratio(matched, candidates)
    precision_raw = _ratio(raw_matched, raw_candidates)
    aggregate.update(
        {
            f"contact_matched_{band}": matched,
            f"contact_gt_{band}": gt,
            f"contact_candidates_{band}": candidates,
            f"contact_raw_matched_{band}": raw_matched,
            f"contact_raw_candidates_{band}": raw_candidates,
            f"recall_{band}": recall,
            f"precision_{band}": precision,
            f"f1_{band}": None if recall is None or precision is None else safe_f1(precision, recall),
            f"precision_raw_{band}": precision_raw,
            f"f1_raw_{band}": (
                None if recall is None or precision_raw is None else safe_f1(precision_raw, recall)
            ),
        }
    )
    fixture_f1: dict[str, float | None] = {}
    for name, row in rows.items():
        fixture_f1[name] = selection.f1_raw_5(row) if band == 5 else _fixture_raw_f1(row, band)
    aggregate[f"fixture_f1_raw_{band}"] = fixture_f1
    fixture_contact_counts: dict[str, dict[str, int]] = {}
    for name, row in rows.items():
        fixture_contact_counts[name] = {
            "matched": int(row[f"_contact_matched_{band}"]),
            "gt": int(row[f"_contact_gt_{band}"]),
            "raw_matched": int(row[f"_contact_raw_matched_{band}"]),
            "raw_candidates": int(row[f"_contact_raw_candidates_{band}"]),
        }
    aggregate[f"fixture_contact_counts_{band}"] = fixture_contact_counts
    measured = [value for value in fixture_f1.values() if value is not None]
    aggregate[f"minimum_fixture_f1_raw_{band}"] = (
        min(measured) if len(measured) == len(fixture_f1) else None
    )


def _add_full_metric_aggregates(
    aggregate: dict[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
) -> None:
    metrics: dict[str, Mapping[str, Any]] = {}
    for name, row in rows.items():
        metrics[name] = row["_full_metrics"]
    aggregate["fixture_full_metrics"] = metrics
    for name in ("ball_round", "player", "server", "hit_height", "landing", "getpoint"):
        for view in ("primary", "covered"):
            correct_key = f"{name}_{view}_correct"
            total_key = f"{name}_{view}_total"
            correct = sum(int(values[correct_key]) for values in metrics.values())
            total = sum(int(values[total_key]) for values in metrics.values())
            aggregate[correct_key] = correct
            aggregate[total_key] = total
            aggregate[f"{name}_{view}"] = _ratio(correct, total)
    for view in ("timing_primary", "timing_covered"):
        matched_key = f"{view}_matched"
        total_key = f"{view}_total"
        matched = sum(int(values[matched_key]) for values in metrics.values())
        total = sum(int(values[total_key]) for values in metrics.values())
        aggregate[matched_key] = matched
        aggregate[total_key] = total
        aggregate[f"{view}_recall"] = _ratio(matched, total)
    contact_matches = sum(int(values["contact_matches"]) for values in metrics.values())
    filtered_total = sum(int(values["contact_filtered_total"]) for values in metrics.values())
    gt_total = sum(int(values["contact_gt_total"]) for values in metrics.values())
    contact_precision = _ratio(contact_matches, filtered_total)
    contact_recall = _ratio(contact_matches, gt_total)
    aggregate.update(
        {
            "contact_matches": contact_matches,
            "contact_filtered_total": filtered_total,
            "contact_gt_total": gt_total,
            "contact_precision": contact_precision,
            "contact_recall": contact_recall,
            "contact_f1": (
                None
                if contact_precision is None or contact_recall is None
                else safe_f1(contact_precision, contact_recall)
            ),
            "n_raw_contacts": sum(int(values["n_raw_contacts"]) for values in metrics.values()),
            "n_filtered_contacts": sum(
                int(values["n_filtered_contacts"]) for values in metrics.values()
            ),
            "hit_height_failures": sum(
                int(values["hit_height_failures"]) for values in metrics.values()
            ),
        }
    )


def _fixture_raw_f1(row: Mapping[str, Any], band: int) -> float | None:
    recall = row[f"recall_{band}"]
    precision = row[f"precision_raw_{band}"]
    if recall is None or precision is None:
        return None
    return safe_f1(float(precision), float(recall))


def _subset_objective(row: Mapping[str, Any], phase: str, excluded: str) -> float:
    if phase == "boundary":
        counts = row["fixture_boundary_counts"]
        n_gt = clean = spurious = 0
        for name, values in counts.items():
            if name == excluded:
                continue
            n_gt += int(values["n_gt_rallies"])
            clean += int(values["clean_covered"])
            spurious += int(values["spurious_spans"])
        denominator = n_gt + clean + spurious
        return 0.0 if denominator == 0 else 2 * clean / denominator
    counts = row["fixture_contact_counts_5"]
    matched = gt = raw_matched = raw_candidates = 0
    for name, values in counts.items():
        if name == excluded:
            continue
        matched += int(values["matched"])
        gt += int(values["gt"])
        raw_matched += int(values["raw_matched"])
        raw_candidates += int(values["raw_candidates"])
    recall = _ratio(matched, gt)
    precision = _ratio(raw_matched, raw_candidates)
    if recall is None or precision is None:
        return 0.0
    return safe_f1(precision, recall)


def _reference_row(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    references = [row for row in rows if row["label"] == sweep.LABEL_SHIPPED]
    if len(references) != 1:
        raise ValueError("corpus phase must contain exactly one shipped baseline")
    return references[0]


def _serialisable_fixture_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in row.items() if not key.startswith("_")}
    payload["contact_counts"] = {
        key.removeprefix("_contact_"): value
        for key, value in row.items()
        if key.startswith("_contact_")
    }
    payload["clean_start_offsets"] = list(row["_clean_offsets"].values())
    payload["split_log"] = row["_split_log"]
    if "_full_metrics" in row:
        payload["full_metrics"] = row["_full_metrics"]
    payload.pop("settings", None)
    return payload


def _candidate_directory(out_dir: Path, phase: str, spec: sweep.CandidateSpec) -> Path:
    return Path(out_dir) / "candidates" / phase / candidate_id(spec)


def _strict_f1(row: Mapping[str, Any], n_gt_rallies: int) -> float:
    true_positives = int(row["clean_covered"])
    denominator = n_gt_rallies + true_positives + int(row["spurious_spans"])
    return 0.0 if denominator == 0 else 2 * true_positives / denominator


def _sum(rows: Mapping[str, Mapping[str, Any]], key: str) -> int:
    return sum(int(row[key]) for row in rows.values())


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _spec_from_payload(payload: object, label: str) -> sweep.CandidateSpec:
    if not isinstance(payload, Mapping):
        raise ValueError("candidate spec payload must be an object")
    if set(payload) != {"overrides_base30", "strategies"}:
        raise ValueError("candidate spec payload fields differ")
    overrides = payload["overrides_base30"]
    strategies = payload["strategies"]
    if not isinstance(overrides, Mapping) or not isinstance(strategies, Mapping):
        raise ValueError("candidate spec mappings are malformed")
    spec = sweep.CandidateSpec(
        label,
        {str(key): float(value) for key, value in overrides.items()},
        {str(key): str(value) for key, value in strategies.items()},
    )
    sweep.serialise_spec(spec)
    return spec


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid corpus sweep JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"corpus sweep JSON must contain an object: {path}")
    return payload


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"existing selection differs from recomputed evidence: {path}")
        return
    _atomic_write_text(path, encoded)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json_gz(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with gzip.open(temporary, "xt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "boundary", "contact", "both"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pinned-repo-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-diff-sha256", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        fixtures = load_pinned_replay_fixtures(
            config_path=args.config,
            run_dir=args.run_dir,
            pinned_repo_dir=args.pinned_repo_dir,
            source_root=args.source_root,
            source_revision=args.source_revision,
        )
        document = build_run_document(
            source_revision=args.source_revision,
            source_diff_sha256=args.source_diff_sha256,
            fixtures=fixtures,
        )
        initialise_run(args.out_dir, document)
        run_calibration(
            phase=args.phase,
            fixtures=fixtures,
            out_dir=args.out_dir,
            workers=args.workers,
            retry_failed=args.retry_failed,
        )
    except Exception as error:  # noqa: BLE001 - command reports retained failure evidence
        print(f"corpus sweep failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"corpus sweep {args.phase} completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
