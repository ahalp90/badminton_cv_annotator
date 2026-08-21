from __future__ import annotations

import gzip
import json

import pytest

from annotator.calibration import corpus_sweep, sweep
from annotator.calibration.fixtures import FIXTURES


def _inputs() -> tuple[corpus_sweep.CorpusFixtureInput, ...]:
    return tuple(
        corpus_sweep.CorpusFixtureInput(
            fixture=fixture,
            inputs=object(),  # type: ignore[arg-type]
            guard_rejected_frames=index + 1,
            inpaint_filled_frames=index + 10,
            artifact_index=f"/fixed/{fixture.name}.json.gz",
            artifact_index_sha256=f"sha-{fixture.name}",
        )
        for index, fixture in enumerate(FIXTURES)
    )


def _spec(value_index: int = 0) -> sweep.CandidateSpec:
    return sweep.CandidateSpec(
        "grid",
        {key: values[value_index] for key, values in sweep.BOUNDARY_VALUES.items()},
        {},
    )


def _fixture_row(spec: sweep.CandidateSpec, *, covered: int, matched: int) -> dict[str, object]:
    row: dict[str, object] = {
        "label": spec.label,
        "n_spans": covered + 2,
        "covered": covered,
        "covered_fraction": covered / 100,
        "clean_covered": covered - 1,
        "split": 1,
        "missed": 2,
        "merged_spans": 1,
        "spurious_spans": 3,
        "swallowed_rallies": 1,
        "strict_align_median": 2.0,
        "strict_align_p90": 4.0,
        "changed_from_defaults": 1,
        "settings": (),
        "_clean_offsets": {("set", covered): float(covered)},
        "_split_log": [],
        "_full_metrics": _full_metrics(matched),
    }
    for band in sweep.CONTACT_TOLERANCES_BASE30:
        row.update(
            {
                f"recall_{band}": matched / 20,
                f"precision_raw_{band}": matched / 10,
                f"_contact_matched_{band}": matched,
                f"_contact_gt_{band}": 20,
                f"_contact_candidates_{band}": 12,
                f"_contact_raw_matched_{band}": matched,
                f"_contact_raw_candidates_{band}": 10,
            }
        )
    return row


def _full_metrics(matched: int) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    for name in ("ball_round", "player", "server", "hit_height", "landing", "getpoint"):
        for view in ("primary", "covered"):
            metrics[f"{name}_{view}_correct"] = matched
            metrics[f"{name}_{view}_total"] = 20
    for view in ("timing_primary", "timing_covered"):
        metrics[f"{view}_matched"] = matched
        metrics[f"{view}_total"] = 20
    metrics.update(
        contact_matches=matched,
        contact_filtered_total=10,
        contact_gt_total=20,
        n_raw_contacts=30,
        n_filtered_contacts=10,
        hit_height_failures=0,
    )
    return metrics


def test_candidate_identity_and_run_document_are_deterministic(tmp_path) -> None:
    first = _spec()
    reordered = sweep.CandidateSpec(
        first.label,
        dict(reversed(list(first.overrides_base30.items()))),
        {},
    )
    assert corpus_sweep.candidate_id(first) == corpus_sweep.candidate_id(reordered)
    document = corpus_sweep.build_run_document(
        source_revision="abc",
        source_diff_sha256="def",
        fixtures=_inputs(),
    )
    corpus_sweep.initialise_run(tmp_path, document)
    corpus_sweep.initialise_run(tmp_path, document)
    changed = {**document, "source_revision": "different"}
    with pytest.raises(ValueError, match="run.json differs"):
        corpus_sweep.initialise_run(tmp_path, changed)


def test_aggregate_uses_integer_micro_counts_and_per_fixture_floors() -> None:
    spec = _spec()
    fixtures = _inputs()
    rows = {
        fixture.fixture.name: _fixture_row(spec, covered=70 + index, matched=5 + index)
        for index, fixture in enumerate(fixtures)
    }
    aggregate = corpus_sweep.aggregate_fixture_rows(spec, fixtures, rows)
    assert aggregate["covered"] == 213
    assert aggregate["guard_rejected_frames"] == 6
    assert aggregate["inpaint_filled_frames"] == 33
    assert aggregate["contact_matched_5"] == 18
    assert aggregate["contact_gt_5"] == 60
    assert aggregate["recall_5"] == pytest.approx(0.3)
    assert aggregate["precision_raw_5"] == pytest.approx(0.6)
    assert aggregate["f1_raw_5"] == pytest.approx(0.4)
    assert aggregate["minimum_fixture_coverage"] == pytest.approx(0.7)
    assert aggregate["landing_primary_correct"] == 18
    assert aggregate["landing_primary_total"] == 60
    assert aggregate["landing_primary"] == pytest.approx(0.3)
    assert aggregate["contact_f1"] == pytest.approx(0.4)


def test_selection_applies_fixture_floor_and_deterministic_objectives() -> None:
    eligible = {
        "label": "grid",
        "candidate_id": "b",
        "strict_f1": 0.8,
        "minimum_fixture_strict_f1": 0.6,
        "strict_align_median": 2.0,
        "changed_from_defaults": 2,
        "minimum_fixture_coverage": 0.7,
        "f1_raw_5": 0.7,
        "minimum_fixture_f1_raw_5": 0.5,
    }
    lower_floor = {**eligible, "candidate_id": "a", "strict_f1": 0.9, "minimum_fixture_coverage": 0.59}
    better_contact = {**eligible, "candidate_id": "c", "f1_raw_5": 0.8}
    assert corpus_sweep.select_boundary([lower_floor, eligible]) == eligible
    assert corpus_sweep.select_contact([eligible, better_contact]) == better_contact


def test_decision_requires_multi_fixture_gain_and_downstream_guardrails() -> None:
    fixtures = _inputs()
    baseline_spec = sweep.shipped_spec()
    candidate_spec = _spec()
    baseline_rows = {
        item.fixture.name: _fixture_row(baseline_spec, covered=70, matched=5)
        for item in fixtures
    }
    candidate_rows = {
        item.fixture.name: _fixture_row(candidate_spec, covered=72, matched=6)
        for item in fixtures
    }
    baseline = corpus_sweep.aggregate_fixture_rows(baseline_spec, fixtures, baseline_rows)
    candidate = corpus_sweep.aggregate_fixture_rows(candidate_spec, fixtures, candidate_rows)
    decision = corpus_sweep.decision_evidence(
        phase="contact",
        baseline=baseline,
        candidate=candidate,
    )
    assert decision["eligible_for_best_config"] is True
    assert decision["survives_strongest_fixture_removal"] is True

    regressed = dict(candidate)
    regressed["landing_primary_correct"] = baseline["landing_primary_correct"] - 1
    decision = corpus_sweep.decision_evidence(
        phase="contact",
        baseline=baseline,
        candidate=regressed,
    )
    assert decision["eligible_for_best_config"] is False
    assert decision["downstream_guardrails_passed"] is False


def test_decision_identifies_fixture_whose_removal_most_reduces_gain() -> None:
    fixtures = _inputs()
    baseline_spec = sweep.shipped_spec()
    candidate_spec = _spec()
    baseline_rows = {
        item.fixture.name: _fixture_row(baseline_spec, covered=70, matched=5)
        for item in fixtures
    }
    candidate_rows = {
        item.fixture.name: _fixture_row(
            candidate_spec,
            covered=72,
            matched={FIXTURES[0].name: 9, FIXTURES[1].name: 6, FIXTURES[2].name: 6}[
                item.fixture.name
            ],
        )
        for item in fixtures
    }
    baseline = corpus_sweep.aggregate_fixture_rows(baseline_spec, fixtures, baseline_rows)
    candidate = corpus_sweep.aggregate_fixture_rows(candidate_spec, fixtures, candidate_rows)

    decision = corpus_sweep.decision_evidence(
        phase="contact",
        baseline=baseline,
        candidate=candidate,
    )

    assert decision["strongest_fixture"] == FIXTURES[0].name
    assert decision["leave_one_out_objective_deltas"][FIXTURES[0].name] == min(
        decision["leave_one_out_objective_deltas"].values()
    )


def test_failed_attempt_is_preserved_and_explicitly_retried(tmp_path) -> None:
    specs = [_spec(0), _spec(1)]
    calls: list[str] = []

    def initial_runner(spec, fixtures):
        calls.append(corpus_sweep.candidate_id(spec))
        if spec is specs[1]:
            return corpus_sweep.CandidateOutcome(
                status="failed",
                elapsed_seconds=1.0,
                error_type="RuntimeError",
                error_message="boom",
                error_traceback="trace",
            )
        return corpus_sweep.CandidateOutcome(
            status="succeeded",
            elapsed_seconds=1.0,
            aggregate={"candidate_id": corpus_sweep.candidate_id(spec)},
            fixtures={},
        )

    with pytest.raises(RuntimeError, match="1 boundary configurations"):
        corpus_sweep.run_phase(
            phase="boundary",
            specs=specs,
            fixtures=_inputs(),
            out_dir=tmp_path,
            workers=1,
            candidate_runner=initial_runner,
        )
    assert len(calls) == 2

    retry_calls: list[str] = []

    def retry_runner(spec, fixtures):
        retry_calls.append(corpus_sweep.candidate_id(spec))
        return corpus_sweep.CandidateOutcome(
            status="succeeded",
            elapsed_seconds=2.0,
            aggregate={"candidate_id": corpus_sweep.candidate_id(spec)},
            fixtures={},
        )

    rows = corpus_sweep.run_phase(
        phase="boundary",
        specs=specs,
        fixtures=_inputs(),
        out_dir=tmp_path,
        workers=1,
        retry_failed=True,
        candidate_runner=retry_runner,
    )
    assert len(rows) == 2
    assert retry_calls == [corpus_sweep.candidate_id(specs[1])]
    directory = tmp_path / "candidates" / "boundary" / corpus_sweep.candidate_id(specs[1])
    attempts = sorted(directory.glob("attempt-*.json.gz"))
    assert len(attempts) == 2
    with gzip.open(attempts[0], "rt", encoding="utf-8") as handle:
        assert json.load(handle)["status"] == "failed"
    with gzip.open(attempts[1], "rt", encoding="utf-8") as handle:
        assert json.load(handle)["status"] == "succeeded"
