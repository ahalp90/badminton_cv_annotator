"""Single-fixture, base-30 calibration sweeps for the annotator pipeline.

The routing rule is deliberately centralised: numeric names in
``_OVERRIDABLE_BASE30_ROWS`` become ``BaseAnnotatorConfig.overrides_base30``;
the two segmentation fields are direct base fields; serve numerics belong to
``ServeStartConfig``.  Strategy names are enum names, never enum values.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from numbers import Real
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from annotator.calibration import selection
from annotator.calibration.fixtures import FIXTURES, SHARED_FILES, FilePin, Fixture, verify_file
from annotator.calibration.gt_scoring import RunVideoInputs, build_run_video_inputs
from annotator.calibration.scoring import (
    CONTACT_TOLERANCES_BASE30,
    RallyBoundary,
    classify_all,
    load_gt_rallies,
    merged_span_indices,
    score_boundaries,
    score_contacts,
)
from annotator.calibration.schemas import (
    CSV_COLUMNS_BY_FILENAME,
    WINNER_JSON_BOUNDARY_KEY,
    WINNER_JSON_META_KEY,
    WINNER_SPEC_OVERRIDES_KEY,
    WINNER_SPEC_STRATEGIES_KEY,
    winner_document,
    winner_spec,
)
from annotator.config import BaseAnnotatorConfig
from annotator.resolve import _OVERRIDABLE_BASE30_ROWS
from annotator.rally_segmentation import ServeStartClose, ServeStartMode
from annotator.run_video import run_video
from annotator.types import ScalingKind, ServeStartConfig, SpanOpen
from annotator.fps_constants import scale_for_fps

LABEL_SHIPPED = "shipped_defaults"
QUALITY_FLOOR = 0.6
WINNER_FILENAME = "config_winner.json"
WINNER_SCHEMA_VERSION = 1
# Persisted compatibility field from the retired direction-and-speed detector.
# Live contact detection does not read this value.
LEGACY_MIN_CONTACT_SPEED = 0.005
BOUNDARY_KEYS = ("rest_speed", "rest_window", "end_rest_frames", "start_speed", "start_min_frames")
CONTACT_KEYS = ("smooth_window", "impulse_floor_half_window_frames", "contact_dedup_radius_frames", "contact_impulse_multiple")
DIRECT_BASE_KEYS = frozenset({"gap_state_demotion_bound", "quiet_start_window"})
SERVE_NUMERIC_KEYS = frozenset({"threshold_bh", "stillness_threshold_bh", "serve_stillness_window_frames"})

BOUNDARY_VALUES = {
    "rest_speed": (1 / 600, 1 / 400, 1 / 240, 1 / 120, 1 / 60),
    "rest_window": (6.0, 8.4, 10.8, 18.0, 25.2),
    "end_rest_frames": (24.0, 36.0, 54.0, 72.0, 90.0, 108.0),
    "start_speed": (1 / 120, 1 / 80, 1 / 60, 1 / 40, 1 / 24),
    "start_min_frames": (1.2, 2.4, 3.6, 6.0),
}
CONTACT_VALUES = {
    "impulse_floor_half_window_frames": (8, 10, 12),
    "contact_impulse_multiple": (2, 3, 4, 6),
    "contact_dedup_radius_frames": (2, 3, 5),
    "smooth_window": (4, 5, 9),
}

_POOL_FIXTURE_INPUTS: RunVideoInputs | None = None


def _pool_initialiser(fixture_inputs: RunVideoInputs) -> None:
    global _POOL_FIXTURE_INPUTS
    _POOL_FIXTURE_INPUTS = fixture_inputs


def _pool_production_candidate(candidate_spec: CandidateSpec) -> dict[str, Any]:
    if _POOL_FIXTURE_INPUTS is None:
        raise RuntimeError("process pool fixture inputs were not initialised")
    return production_candidate_runner(
        fixture_inputs=_POOL_FIXTURE_INPUTS, candidate_spec=candidate_spec
    )


def _boundary_report_rows(rows: list[dict[str, Any]], n_rallies: int) -> list[dict[str, Any]]:
    """Return every pinned boundary report rule from grid rows only."""
    grid = selection.grid_rows(rows)
    live = min(
        selection.coverage_allowance_rows(rows, n_rallies),
        key=lambda row: selection.boundary_live_key_rally_id_f1(row, n_rallies),
    )
    rules = [
        ("rally_id_f1", live),
        (
            "fewest_merges",
            min(grid, key=selection.boundary_report_key_fewest_swallowed_rallies),
        ),
        ("coverage_first", min(grid, key=selection.boundary_report_key_coverage_first)),
        ("tightest_start", min(grid, key=selection.boundary_report_key_tightest_start)),
    ]
    for covered in sorted({row["covered"] for row in grid}, reverse=True):
        at_coverage = [row for row in grid if row["covered"] == covered]
        rules.append(
            (
                f"frontier_cov_{covered}",
                min(at_coverage, key=selection.boundary_report_key_fewest_swallowed_rallies),
            )
        )
    return [
        {
            "rule": rule,
            **row,
            "coverage_gap_from_best": max(item["covered"] for item in grid) - row["covered"],
            "needed_allowance": (max(item["covered"] for item in grid) - row["covered"]) / n_rallies,
        }
        for rule, row in rules
    ]


def _contact_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep non-dominated grid contact rows; duplicates intentionally remain."""
    candidates = [
        row for row in selection.grid_rows(rows)
        if row["recall_5"] is not None and row["precision_raw_5"] is not None
    ]
    frontier = [
        row for row in candidates
        if not any(
            other["recall_5"] >= row["recall_5"]
            and other["precision_raw_5"] >= row["precision_raw_5"]
            and (other["recall_5"] > row["recall_5"] or other["precision_raw_5"] > row["precision_raw_5"])
            for other in candidates
        )
    ]
    return sorted(frontier, key=lambda row: (-row["recall_5"], -row["precision_raw_5"], selection.standard_tail(row)))


def _alignment_rows(report_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build own and shared alignment summaries from runner-supplied identities."""
    own: list[dict[str, Any]] = []
    identity_sets = [set(row.get("_clean_offsets", {})) for row in report_rows]
    shared = set.intersection(*identity_sets) if identity_sets else set()
    shared_rows: list[dict[str, Any]] = []
    for row in report_rows:
        offsets = list(row.get("_clean_offsets", {}).values())
        shared_offsets = [row.get("_clean_offsets", {})[identity] for identity in shared]
        for target, values in ((own, offsets), (shared_rows, shared_offsets)):
            target.append({
                "rule": row["rule"], "n_rallies": len(values),
                "median_abs_start_offset": float(np.median(values)) if values else None,
                "p90_abs_start_offset": float(np.percentile(values, 90)) if values else None,
            })
    return own, shared_rows


def _run_candidates(
    specs: list[CandidateSpec], *, phase: str, fixture_inputs: RunVideoInputs,
    candidate_runner: Callable[..., dict[str, Any]], workers: int,
) -> list[dict[str, Any]]:
    """Run candidates while preserving grid order and retaining successful rows."""
    def one(index: int, spec: CandidateSpec) -> dict[str, Any] | None:
        try:
            return candidate_runner(fixture_inputs=fixture_inputs, candidate_spec=spec)
        except Exception as error:
            print(f"{phase} grid index {index} failed for {serialise_spec(spec)!r}: {error}", file=sys.stderr)
            return None

    # The injected test seam can be a closure.  Process workers are reserved for
    # the module-level production runner, which is safely picklable.
    if workers > 1 and candidate_runner is production_candidate_runner:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_pool_initialiser,
            initargs=(fixture_inputs,),
        ) as executor:
            futures = [
                executor.submit(_pool_production_candidate, spec)
                for spec in specs
            ]
            completed = []
            for index, (spec, future) in enumerate(zip(specs, futures)):
                try:
                    completed.append(future.result())
                except Exception as error:
                    print(f"{phase} grid index {index} failed for {serialise_spec(spec)!r}: {error}", file=sys.stderr)
                    completed.append(None)
    else:
        completed = [one(index, spec) for index, spec in enumerate(specs)]
    rows = [row for row in completed if row is not None]
    if not rows:
        raise RuntimeError(f"no {phase} grid row succeeded")
    return rows


@dataclass(frozen=True)
class CandidateSpec:
    """A serialisable candidate expressed exclusively in base-30 values."""

    label: str
    overrides_base30: Mapping[str, float]
    strategies: Mapping[str, str]


def build_boundary_grid() -> list[CandidateSpec]:
    """Build the pinned 3,000-cell boundary grid in product order."""
    return [CandidateSpec(selection.GRID_LABEL, dict(zip(BOUNDARY_KEYS, values)), {})
            for values in product(*(BOUNDARY_VALUES[key] for key in BOUNDARY_KEYS))]


def build_contact_grid(boundary: Mapping[str, float]) -> list[CandidateSpec]:
    """Build the pinned 108-cell contact grid with its boundary frozen."""
    return [CandidateSpec(selection.GRID_LABEL, {**boundary, **dict(zip(CONTACT_KEYS, values))}, {})
            for values in product(*(CONTACT_VALUES[key] for key in CONTACT_KEYS))]


def shipped_spec() -> CandidateSpec:
    """Return the unmodified configuration reference row."""
    return CandidateSpec(LABEL_SHIPPED, {}, {})


def _base_and_serve(spec: CandidateSpec) -> tuple[BaseAnnotatorConfig, ServeStartConfig | None]:
    """Apply the module routing rule shared by serialisation and execution."""
    overrides: dict[str, float] = {}
    direct: dict[str, float] = {}
    serve: dict[str, float] = {}
    for key, value in spec.overrides_base30.items():
        if key in _OVERRIDABLE_BASE30_ROWS:
            overrides[key] = value
        elif key in DIRECT_BASE_KEYS:
            direct[key] = value
        elif key in SERVE_NUMERIC_KEYS:
            serve[key] = value
        else:
            raise ValueError(f"cannot route numeric sweep key {key!r}")
    strategies: dict[str, str] = {}
    span_open: SpanOpen | None = None
    for key, name in spec.strategies.items():
        try:
            if key == "span_open":
                span_open = SpanOpen[name]
            elif key == "mode":
                ServeStartMode[name]
            elif key == "close":
                ServeStartClose[name]
            else:
                raise ValueError(f"cannot route strategy key {key!r}")
        except KeyError as error:
            raise ValueError(f"invalid strategy value for key {key!r}: {name!r}") from error
        if key in {"span_open", "mode", "close"}:
            strategies[key] = name
        else:
            raise ValueError(f"cannot route strategy key {key!r}")
    base_kwargs: dict[str, Any] = {"overrides_base30": overrides or None, **direct}
    if "span_open" in strategies:
        base_kwargs["span_open"] = span_open
    elif "quiet_start_window" in direct:
        base_kwargs["span_open"] = None
    base = BaseAnnotatorConfig(**base_kwargs)
    if not serve and "mode" not in spec.strategies and "close" not in spec.strategies:
        return base, None
    if "threshold_bh" not in serve or "mode" not in spec.strategies:
        raise ValueError("serve request requires threshold_bh and mode")
    close = ServeStartClose[spec.strategies["close"]] if "close" in spec.strategies else None
    return base, ServeStartConfig(serve["threshold_bh"], ServeStartMode[spec.strategies["mode"]], close, serve.get("stillness_threshold_bh"))


def serialise_spec(spec: CandidateSpec) -> dict[str, dict[str, object]]:
    """Serialise using the same closed routing validation as application."""
    _base_and_serve(spec)
    return winner_spec(spec.overrides_base30, spec.strategies)


def _settings(spec: CandidateSpec) -> tuple[tuple[int, object], ...]:
    keys = (*BOUNDARY_KEYS, *CONTACT_KEYS, "gap_state_demotion_bound", "quiet_start_window", "threshold_bh", "stillness_threshold_bh", "serve_stillness_window_frames", "span_open", "mode", "close")
    ranks = {"span_open": {None: 0, "REGION_START": 1, "BACK_FILL": 2}, "mode": {None: 0, "TRIM": 1, "REJECT": 2}, "close": {None: 0, "BURST": 1, "LAST_REST": 2}}
    values: list[tuple[int, object]] = []
    for key in keys:
        value = spec.strategies.get(key) if key in ranks else spec.overrides_base30.get(key)
        values.append((0, 0) if value is None else (1, ranks[key][value] if key in ranks else value))
    return tuple(values)


def _shipped_base30_values() -> dict[str, float]:
    constants = scale_for_fps(30.0)
    thresholds = BaseAnnotatorConfig().thresholds
    values: dict[str, float] = {}
    for key in _OVERRIDABLE_BASE30_ROWS:
        if hasattr(constants, key):
            values[key] = getattr(constants, key)
        elif hasattr(thresholds, key):
            values[key] = getattr(thresholds, key)
    return values


def _changed_from_defaults(spec: CandidateSpec) -> int:
    """Count applied axes that differ from the complete shipped configuration."""
    base, serve = _base_and_serve(spec)
    shipped = BaseAnnotatorConfig()
    shipped_base30 = _shipped_base30_values()
    changed = 0
    for key, value in spec.overrides_base30.items():
        if key in _OVERRIDABLE_BASE30_ROWS:
            if value != shipped_base30[key]:
                changed += 1
        elif key in DIRECT_BASE_KEYS:
            if value != getattr(shipped, key):
                changed += 1
        elif key in SERVE_NUMERIC_KEYS:
            changed += 1
        else:
            raise ValueError(f"cannot route numeric sweep key {key!r}")
    for key, value in spec.strategies.items():
        shipped_value = getattr(shipped, key, None)
        if key == "span_open":
            applied = base.span_open
            if applied != shipped_value:
                changed += 1
        else:
            # There is no shipped serve configuration. Any requested serve
            # axis is therefore a real configuration change.
            if key in {"mode", "close"}:
                changed += 1
            else:
                raise ValueError(f"cannot route strategy key {key!r}")
    if serve is not None:
        # Numeric serve axes were counted above. This assertion keeps the
        # complete application and counting paths aligned.
        assert any(key in SERVE_NUMERIC_KEYS for key in spec.overrides_base30)
    return changed


def _display_config(spec: CandidateSpec) -> dict[str, object]:
    shipped = _shipped_base30_values()
    return {
        key: spec.overrides_base30.get(key, shipped[key])
        for key in (*BOUNDARY_KEYS, *CONTACT_KEYS)
    }


def _row_for_result(fixture: Fixture, spec: CandidateSpec, result: Any, master: Any) -> dict[str, Any]:
    gt = load_gt_rallies(master, fixture.video_id)
    spans = result.spans
    boundary = score_boundaries(spans, gt)
    classifications = classify_all(spans, gt)
    merged = merged_span_indices(spans, gt)
    clean = [(index, rally, span) for index, (rally, (kind, span)) in enumerate(zip(gt, classifications)) if kind is RallyBoundary.COVERED and span not in merged]
    offsets = [abs(spans[span][0] - rally.extent[0]) for _index, rally, span in clean]
    contained = [sum(start <= first and last < end for first, last in (r.extent for r in gt)) for start, end in spans]
    tolerances = {
        band: ScalingKind.FRAME_COUNT.scale(band, fixture.fps)
        for band in CONTACT_TOLERANCES_BASE30
    }
    contacts = [(contact.rally_id, contact.contact_frame, contact.proximity_ok, contact.wrist_near) for contact in result.filtered_contacts]
    contact = score_contacts(spans, contacts, gt, tuple(tolerances.values()))
    row: dict[str, Any] = {
        "label": spec.label,
        **_display_config(spec),
        "min_contact_speed": LEGACY_MIN_CONTACT_SPEED,
        "n_spans": len(spans), **boundary, "clean_covered": len(clean), "swallowed_rallies": sum(max(0, count - 1) for count in contained), "max_rallies_in_one_span": max(contained, default=0),
        "strict_align_median": float(np.median(offsets)) if offsets else None, "strict_align_p90": float(np.percentile(offsets, 90)) if offsets else None,
        "total_candidates": len(result.filtered_contacts), "changed_from_defaults": _changed_from_defaults(spec), "settings": _settings(spec)}
    denominator = len(gt) + row["clean_covered"] + row["spurious_spans"]
    row["strict_f1"] = 0.0 if denominator == 0 else 2 * row["clean_covered"] / denominator
    start = boundary["start_alignment"]
    row["start_alignment_mean"] = None if start is None else start["mean"]
    row["start_alignment_median"] = None if start is None else start["median"]
    for band, frames in tolerances.items():
        metrics = contact["tolerances"][str(frames)]
        raw = contact["precision_raw"][str(frames)]
        row.update({f"tolerance_frames_{band}": frames, f"recall_{band}": metrics["recall"], f"precision_{band}": metrics["precision"], f"f1_{band}": metrics["f1"], f"precision_raw_{band}": raw["precision_raw"]})
        # Corpus sweeps need integer numerators and denominators. Aggregating
        # rounded per-fixture ratios would give videos equal weight regardless
        # of their number of ground-truth strokes.
        row[f"_contact_matched_{band}"] = metrics["matched"]
        row[f"_contact_gt_{band}"] = metrics["gt"]
        row[f"_contact_candidates_{band}"] = metrics["candidates"]
        row[f"_contact_raw_matched_{band}"] = raw["matched"]
        row[f"_contact_raw_candidates_{band}"] = raw["candidates"]
    row["count_gate_covered_fraction"] = contact["count_gate"]["covered"]["fraction"]
    row["count_gate_unmerged_fraction"] = contact["count_gate"]["unmerged"]["fraction"]
    row["f1_raw_5"] = selection.f1_raw_5(row)
    row["_clean_offsets"] = {
        (rally.set_id, rally.rally): abs(spans[span][0] - rally.extent[0])
        for _index, rally, span in clean
    }
    split_log: list[dict[str, Any]] = []
    for index, (rally, (kind, _mapped_span)) in enumerate(zip(gt, classifications)):
        pieces = [
            (span_index, spans[span_index])
            for span_index in sorted({
                span_index
                for stroke_frame in rally.stroke_frames
                for span_index in range(len(spans))
                if spans[span_index][0] <= stroke_frame < spans[span_index][1]
            })
        ]
        if kind is RallyBoundary.SPLIT:
            split_log.append({
                "video_id": fixture.video_id, "gt_rally_index": index,
                "gt_start": rally.extent[0], "gt_end": rally.extent[1],
                "piece_count": len(pieces),
                "piece_spans": ";".join(f"{start}-{end}" for _span, (start, end) in pieces),
            })
    row["_split_log"] = split_log
    return row


def production_candidate_runner(*, fixture_inputs: RunVideoInputs, candidate_spec: CandidateSpec) -> dict[str, Any]:
    """Run and score one candidate; resolution remains inside ``run_video``."""
    video_id = fixture_inputs.keyword["video_id"]
    fixture = next(
        fixture
        for fixture in FIXTURES
        if fixture.video_id == video_id or fixture.name == video_id
    )
    base, serve_start = _base_and_serve(candidate_spec)
    keyword = dict(fixture_inputs.keyword)
    keyword.update(base=base, serve_start=serve_start)
    result = run_video(*fixture_inputs.positional, **keyword)
    return _row_for_result(fixture, candidate_spec, result, fixture_inputs.master)


def _write_csv(path: Path, columns: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in writer.fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, Real) and not math.isfinite(value):
        raise ValueError("CSV value is not finite")
    return value.name if hasattr(value, "name") else value


def _empty_outputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, columns in CSV_COLUMNS_BY_FILENAME.items():
        _write_csv(out_dir / filename, columns, [])


def run_sweep(*, fixture: Fixture, out_dir: Path, phase: str = "both", boundary_spec: CandidateSpec | None = None,
              candidate_runner: Callable[..., dict[str, Any]] = production_candidate_runner,
              fixture_inputs: RunVideoInputs | None = None, workers: int = 1) -> int:
    """Execute the requested phases; the keyword runner is the fake-score seam."""
    if phase == "contact" and boundary_spec is None:
        raise ValueError("contact phase requires a boundary winner")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / WINNER_FILENAME).unlink(missing_ok=True)
    _empty_outputs(out_dir)
    inputs = build_run_video_inputs(fixture) if fixture_inputs is None else fixture_inputs
    boundary_rows: list[dict[str, Any]] = []
    winner: CandidateSpec | None = boundary_spec
    phases_run: list[str] = []
    if phase in {"boundary", "both"}:
        boundary_rows = _run_candidates(
            [*build_boundary_grid(), shipped_spec()], phase="boundary",
            fixture_inputs=inputs, candidate_runner=candidate_runner, workers=workers,
        )
        _write_csv(out_dir / "boundary_sweep.csv", CSV_COLUMNS_BY_FILENAME["boundary_sweep.csv"], boundary_rows)
        report_rows = _boundary_report_rows(boundary_rows, fixture.n_rallies)
        _write_csv(out_dir / "best_config_comparison.csv", CSV_COLUMNS_BY_FILENAME["best_config_comparison.csv"], report_rows)
        own_alignment, shared_alignment = _alignment_rows(report_rows)
        _write_csv(out_dir / "alignment_own_covered.csv", CSV_COLUMNS_BY_FILENAME["alignment_own_covered.csv"], own_alignment)
        _write_csv(out_dir / "alignment_shared.csv", CSV_COLUMNS_BY_FILENAME["alignment_shared.csv"], shared_alignment)
        live = report_rows[0]
        _write_csv(out_dir / "split_log.csv", CSV_COLUMNS_BY_FILENAME["split_log.csv"], live.get("_split_log", []))
        winner = CandidateSpec(live["label"], {key: live[key] for key in BOUNDARY_KEYS}, {})
        phases_run.append("boundary")
        # This deliberately uses min((-covered, tail)) rather than max: the
        # quality floor must inspect the greatest coverage, not its inverse.
        best_coverage = min(selection.grid_rows(boundary_rows), key=lambda row: (-row["covered"], selection.standard_tail(row)))
        if not selection.best_config_clears_quality_floor(best_coverage, QUALITY_FLOOR):
            _withheld("CALIBRATION VERDICT WITHHELD: best grid coverage is below the quality floor")
            return 0
    if phase in {"contact", "both"}:
        assert winner is not None
        contact_rows = _run_candidates(
            [*build_contact_grid(winner.overrides_base30), shipped_spec()], phase="contact",
            fixture_inputs=inputs, candidate_runner=candidate_runner, workers=workers,
        )
        _write_csv(out_dir / "contact_sweep.csv", CSV_COLUMNS_BY_FILENAME["contact_sweep.csv"], contact_rows)
        _write_csv(out_dir / "contact_frontier.csv", CSV_COLUMNS_BY_FILENAME["contact_frontier.csv"], _contact_frontier(contact_rows))
        contact_winner = selection.select_contact_live_winner(contact_rows)
        phases_run.append("contact")
        if contact_winner is None:
            _withheld("CALIBRATION VERDICT WITHHELD: no contact grid row has measurable logical-5 recall and raw precision")
            return 0
        contact_spec = CandidateSpec(contact_winner["label"], {key: contact_winner[key] for key in (*BOUNDARY_KEYS, *CONTACT_KEYS)}, {})
        stability_rows: list[dict[str, Any]] = []
        if phase == "both":
            assert boundary_rows
            report_by_rule = {row["rule"]: row for row in report_rows}
            auxiliary = {
                rule: CandidateSpec(
                    selection.GRID_LABEL,
                    {key: report_by_rule[rule][key] for key in BOUNDARY_KEYS},
                    {},
                )
                for rule in ("rally_id_f1", "fewest_merges", "coverage_first")
            }
            cached: dict[str, dict[str, Any] | None] = {
                json.dumps(serialise_spec(winner), sort_keys=True): contact_winner,
            }
            for rule, boundary_candidate in auxiliary.items():
                serialised = json.dumps(serialise_spec(boundary_candidate), sort_keys=True)
                if serialised not in cached:
                    rows = _run_candidates(
                        build_contact_grid(boundary_candidate.overrides_base30), phase="contact",
                        fixture_inputs=inputs, candidate_runner=candidate_runner, workers=workers,
                    )
                    cached[serialised] = selection.select_contact_live_winner(rows)
                auxiliary_winner = cached[serialised]
                stability_rows.append({
                    "rule": rule,
                    **({} if auxiliary_winner is None else auxiliary_winner),
                    "same_winner_as_live": auxiliary_winner is not None and serialise_spec(CandidateSpec(auxiliary_winner["label"], {key: auxiliary_winner[key] for key in (*BOUNDARY_KEYS, *CONTACT_KEYS)}, {})) == serialise_spec(contact_spec),
                })
        _write_csv(out_dir / "contact_stability.csv", CSV_COLUMNS_BY_FILENAME["contact_stability.csv"], stability_rows)
        document = winner_document(
            fixture.name,
            phases_run,
            boundary=serialise_spec(winner),
            contact=serialise_spec(contact_spec),
            schema_version=WINNER_SCHEMA_VERSION,
            tuning_video_ids=[fixture.video_id],
            input_digests=_input_digest_bundle(fixture),
        )
        (out_dir / WINNER_FILENAME).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    elif winner is not None:
        document = winner_document(
            fixture.name,
            phases_run,
            boundary=serialise_spec(winner),
            schema_version=WINNER_SCHEMA_VERSION,
            tuning_video_ids=[fixture.video_id],
            input_digests=_input_digest_bundle(fixture),
        )
        (out_dir / WINNER_FILENAME).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return 0


def _withheld(message: str) -> None:
    print(message)
    print(message, file=sys.stderr)


def _fixture(value: str) -> Fixture:
    matches = [fixture for fixture in FIXTURES if fixture.name == value]
    if len(matches) != 1:
        raise argparse.ArgumentTypeError(f"unknown fixture {value!r}")
    return matches[0]


def _input_pins(fixture: Fixture) -> tuple[FilePin, ...]:
    return (*fixture.files, *SHARED_FILES)


def _input_digest_bundle(fixture: Fixture) -> dict[str, str]:
    """Return the existing fixture-layer MD5 pins consumed by one sweep."""
    return {str(pin.path): pin.md5 for pin in _input_pins(fixture)}


def _validate_provenance(meta: dict[str, Any], fixture: Fixture) -> None:
    expected_keys = {"fixture", "phases_run", "verdict", "tolerances_base30", "schema_version", "tuning_video_ids", "input_digests"}
    if set(meta) != expected_keys:
        raise ValueError("config winner meta has unknown or missing keys")
    if meta["schema_version"] != WINNER_SCHEMA_VERSION:
        raise ValueError(f"unknown config winner schema_version {meta['schema_version']!r}")
    video_ids = meta["tuning_video_ids"]
    if not isinstance(video_ids, list) or not video_ids or any(isinstance(value, bool) or not isinstance(value, int) for value in video_ids):
        raise ValueError("config winner tuning_video_ids must be a non-empty list of integers")
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("config winner tuning_video_ids must not contain duplicates")
    if fixture.video_id not in video_ids:
        raise ValueError("config winner tuning_video_ids do not include the requested fixture video")
    digests = meta["input_digests"]
    if not isinstance(digests, dict) or not digests:
        raise ValueError("config winner input_digests must be a non-empty object")
    expected_digests = _input_digest_bundle(fixture)
    if digests != expected_digests:
        mismatched = sorted(set(digests) ^ set(expected_digests))
        mismatched.extend(key for key in set(digests) & set(expected_digests) if digests[key] != expected_digests[key])
        file_name = mismatched[0] if mismatched else "input digest bundle"
        raise ValueError(f"config winner input digest mismatch for {file_name}")
    for pin in _input_pins(fixture):
        verify_file(pin)


def load_boundary_winner(path: Path, fixture_name: str) -> CandidateSpec:
    """Load the one closed boundary spec accepted by contact-only sweeps."""
    try:
        def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            payload: dict[str, Any] = {}
            for key, value in pairs:
                if key in payload:
                    raise ValueError(f"duplicate winner-document key {key!r}")
                payload[key] = value
            return payload

        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid boundary winner document: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("invalid boundary winner document: offending key 'document' is not a dict")
    if WINNER_JSON_META_KEY not in document or not isinstance(document[WINNER_JSON_META_KEY], dict):
        raise ValueError("invalid boundary winner document: offending key 'meta' is not a dict")
    if WINNER_JSON_BOUNDARY_KEY not in document or not isinstance(document[WINNER_JSON_BOUNDARY_KEY], dict):
        raise ValueError("invalid boundary winner document: offending key 'boundary' is not a dict")
    meta = document[WINNER_JSON_META_KEY]
    phase = document[WINNER_JSON_BOUNDARY_KEY]
    if set(phase) != {WINNER_SPEC_OVERRIDES_KEY, WINNER_SPEC_STRATEGIES_KEY}:
        raise ValueError("boundary winner phase has unknown or missing spec keys")
    if not isinstance(phase.get(WINNER_SPEC_OVERRIDES_KEY), dict):
        raise ValueError("invalid boundary winner document: offending key 'overrides_base30' is not a dict")
    if not isinstance(phase.get(WINNER_SPEC_STRATEGIES_KEY), dict):
        raise ValueError("invalid boundary winner document: offending key 'strategies' is not a dict")
    overrides = phase[WINNER_SPEC_OVERRIDES_KEY]
    strategies = phase[WINNER_SPEC_STRATEGIES_KEY]
    if set(document) != {WINNER_JSON_META_KEY, WINNER_JSON_BOUNDARY_KEY}:
        raise ValueError("boundary winner document has unknown or missing phase keys")
    # A winner written since the provenance change carries three extra meta keys; accept and
    # validate them so a boundary-phase winner can still feed a contact-phase run.
    if "schema_version" in meta:
        _validate_provenance(meta, _fixture(fixture_name))
    elif set(meta) != {"fixture", "phases_run", "verdict", "tolerances_base30"}:
        raise ValueError("boundary winner meta has unknown or missing keys")
    if meta["fixture"] != fixture_name:
        raise ValueError("boundary winner fixture does not match --fixture")
    phases_run = meta["phases_run"]
    if (
        not isinstance(phases_run, list)
        or any(not isinstance(item, str) for item in phases_run)
        or "boundary" not in phases_run
        or len(set(phases_run)) != len(phases_run)
        or any(item not in {"boundary", "contact"} for item in phases_run)
    ):
        raise ValueError("boundary winner phases_run is invalid")
    if (
        meta["verdict"] != "issued"
        or meta["tolerances_base30"] != list(CONTACT_TOLERANCES_BASE30)
    ):
        raise ValueError("boundary winner meta does not describe an issued base-30 verdict")
    if set(overrides) != set(BOUNDARY_KEYS) or strategies != {}:
        raise ValueError("boundary winner must contain exactly the five boundary numeric keys")
    checked: dict[str, float] = {}
    for key, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid boundary winner numeric value for {key!r}")
        if value not in BOUNDARY_VALUES[key]:
            raise ValueError(f"boundary winner numeric value is outside the {key!r} grid domain")
        checked[key] = float(value)
    return CandidateSpec(selection.GRID_LABEL, checked, {})


def _replace_mask(inputs: RunVideoInputs, path: Path) -> RunVideoInputs:
    mask = np.load(path, allow_pickle=False)
    frame_count = len(inputs.positional[0])
    if mask.ndim != 1 or mask.dtype != np.bool_ or len(mask) != frame_count or mask.all():
        raise ValueError("--mask-npy must be a frame-aligned, non-all-True boolean vector")
    keyword = dict(inputs.keyword)
    keyword["raw_exclusion_mask"] = mask
    return inputs._replace(keyword=keyword)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=_fixture)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--phase", choices=("boundary", "contact", "both"), default="both")
    parser.add_argument("--boundary-winner-json", type=Path)
    parser.add_argument("--mask-npy", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    if args.boundary_winner_json is not None and args.phase != "contact":
        parser.error("--boundary-winner-json is accepted only with --phase contact")
    if args.phase == "contact" and args.boundary_winner_json is None:
        parser.error("contact phase requires --boundary-winner-json")
    try:
        boundary_spec = None
        if args.boundary_winner_json is not None:
            boundary_spec = load_boundary_winner(args.boundary_winner_json, args.fixture.name)
        inputs = build_run_video_inputs(args.fixture)
        if args.mask_npy is not None:
            inputs = _replace_mask(inputs, args.mask_npy)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    try:
        return run_sweep(fixture=args.fixture, out_dir=args.out_dir, phase=args.phase, boundary_spec=boundary_spec, fixture_inputs=inputs, workers=args.workers)
    except Exception as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
