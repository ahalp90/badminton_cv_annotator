"""Focused pure-contract tests for the structured serve-prefix freeze."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import score_contact_serve_prefix as scorer


def _rows(
    frames: tuple[int, ...],
    scores: tuple[float, ...],
    *,
    intervals: tuple[int, ...] | None = None,
    decisions: tuple[int, ...] | None = None,
    fixture: str = "sset_01",
) -> np.ndarray:
    """Build a small score table with the production sidecar dtype."""
    if intervals is None:
        intervals = (0,) * len(frames)
    if decisions is None:
        decisions = (0,) * len(frames)
    rows = np.zeros(len(frames), dtype=scorer.tree_scorer.CANDIDATE_SCORE_DTYPE)
    rows["fixture"] = fixture.encode("ascii")
    rows["interval_id"] = intervals
    rows["frame"] = frames
    rows["timing_score"] = scores
    rows["threshold"] = 0.75
    rows["decision"] = decisions
    return rows


def _evidence(*spans: dict[str, object]) -> dict[str, object]:
    """Build the minimal evidence shape consumed by pure construction."""
    fixture_rows = [
        {"fixture": "sset_01", "spans": list(spans)},
        {"fixture": "sset_15", "spans": []},
        {"fixture": "sset_21", "spans": []},
    ]
    return {"fixtures": fixture_rows}


def _span(
    span_id: int,
    start: int,
    end: int,
    contacts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "span_id": span_id,
        "start_frame": start,
        "end_frame": end,
        "contacts": [] if contacts is None else contacts,
    }


def _contact(frame: int, *, filtered: bool = True) -> dict[str, object]:
    return {"contact_frame": frame, "filtered": filtered}


def _intervals() -> dict[str, tuple[tuple[int, int], ...]]:
    return {fixture: ((0, 100),) for fixture in scorer.tree_scorer.FIXTURE_SPECS}


def test_half_open_prefix_bounds_and_preceding_span_attachment_boundary() -> None:
    assert scorer.make_prefix_bounds("sset_01", 0, 20, 40, 20, 0, (0, 50), None) == (0, 20)
    assert scorer.make_prefix_bounds("sset_01", 1, 40, 60, 45, 0, (0, 50), 30) == (30, 45)
    # An end equal to the interval end is outside that half-open interval.
    assert scorer.make_prefix_bounds("sset_01", 1, 40, 60, 45, 0, (0, 50), 50) == (0, 45)
    with pytest.raises(ValueError, match="outside its search interval"):
        scorer.make_prefix_bounds("sset_01", 0, 0, 10, 50, 0, (0, 50), None)
    with pytest.raises(ValueError, match="lower bound exceeds"):
        scorer.make_prefix_bounds("sset_01", 1, 40, 60, 25, 0, (0, 50), 30)


def test_nplus_radius_scales_at_25_and_30_fps() -> None:
    assert scorer.scaled_nplus_radius(25.0) == 5
    assert scorer.scaled_nplus_radius(30.0) == 6


def test_threshold_free_peaks_use_nms_ties_and_earlier_frame() -> None:
    rows = _rows(
        (0, 5, 6, 11, 12, 20),
        (0.90, 0.80, 0.85, 0.80, 0.70, 0.60),
    )

    peaks = scorer.threshold_free_peaks(rows, "sset_01", 0, 0, 20, 5)

    assert peaks == ((0, 0.90), (6, 0.85), (12, 0.70))

    tied = _rows((0, 1, 7), (0.90, 0.90, 0.80))
    assert scorer.threshold_free_peaks(tied, "sset_01", 0, 0, 7, 5)[0] == (0, 0.90)


def test_prefix_has_at_most_five_candidates_and_merges_exact_sources() -> None:
    # Three distinct HGB peaks, a filtered heuristic row, and the anchor.
    rows = _rows(
        (0, 6, 12, 18, 24, 30),
        (0.90, 0.89, 0.88, 0.87, 0.86, 0.50),
        decisions=(0, 0, 0, 0, 0, 2),
    )
    evidence = _evidence(
        _span(0, 25, 31, [_contact(24)]),
    )

    frozen = scorer.construct_prefixes(rows, rows, evidence, _intervals())
    prefix = frozen.prefixes[0]

    assert len(prefix.candidates) == 5
    assert [candidate.frame for candidate in prefix.candidates] == [0, 6, 12, 24, 30]
    assert prefix.candidates[3].source_flags == (scorer.SOURCE_FILTERED_HEURISTIC,)
    assert prefix.candidates[4].source_flags == (scorer.SOURCE_ANCHOR,)
    assert frozen.candidate_source_count == 5
    assert frozen.candidate_count == 5
    assert frozen.exact_deduplication_count == 0

    # A heuristic candidate at an existing peak is one identity with both flags.
    merged_evidence = _evidence(_span(0, 25, 31, [_contact(12)]))
    merged_frozen = scorer.construct_prefixes(rows, rows, merged_evidence, _intervals())
    merged = merged_frozen.prefixes[0]
    merged_candidate = next(candidate for candidate in merged.candidates if candidate.frame == 12)
    assert merged_candidate.source_flags == (scorer.SOURCE_HGB_PEAK_3, scorer.SOURCE_FILTERED_HEURISTIC)
    assert merged_frozen.exact_deduplication_count == 1


def test_filtered_heuristic_requires_exact_fixture_interval_frame_join() -> None:
    rows = _rows((12,), (0.80,), intervals=(0,))
    evidence = _evidence(_span(0, 12, 13, [_contact(12)]))
    intervals = {
        "sset_01": ((0, 10), (10, 20)),
        "sset_15": ((0, 1),),
        "sset_21": ((0, 1),),
    }

    with pytest.raises(ValueError, match="no exact HGB row"):
        scorer.construct_prefixes(rows, rows, evidence, intervals)


def test_no_anchor_records_an_abstention_without_candidates() -> None:
    rows = _rows((10,), (0.80,), decisions=(0,))
    frozen = scorer.construct_prefixes(
        rows,
        rows,
        _evidence(_span(0, 10, 20)),
        _intervals(),
    )
    prefix = frozen.prefixes[0]

    assert prefix.anchor_frame is None
    assert prefix.candidates == ()
    assert prefix.abstention_reason == scorer.ABSTENTION_NO_ANCHOR
    assert prefix.fixed_action.action == "abstain"


def test_fixed_rule_requires_earlier_heuristic_farther_than_every_nplus_event() -> None:
    rows = _rows(
        (10, 30, 50),
        (0.20, 0.90, 0.80),
        decisions=(2, 2, 0),
    )
    prefix = scorer.PrefixRecord(
        fixture="sset_01",
        span_id=0,
        interval_id=0,
        lower_frame=0,
        upper_frame=30,
        anchor_frame=30,
        anchor_score=0.90,
        radius_frames=5,
        candidates=(
            scorer.PrefixCandidate(
                "sset_01", 0, 0, 10, 0.20, 1, (scorer.SOURCE_FILTERED_HEURISTIC,)
            ),
        ),
        fixed_action=scorer.PrefixAction("abstain", None, 30, "pending"),
    )
    assert scorer.choose_fixed_action(prefix, rows).action == "abstain"

    too_close = scorer.PrefixCandidate(
        "sset_01", 0, 0, 26, 0.20, 1, (scorer.SOURCE_FILTERED_HEURISTIC,)
    )
    close_prefix = scorer.PrefixRecord(
        "sset_01", 0, 0, 0, 30, 30, 0.90, 5, (too_close,), prefix.fixed_action
    )
    close_action = scorer.choose_fixed_action(close_prefix, rows)
    assert close_action.action == "abstain"
    assert close_action.reason == "heuristic_within_nplus_radius"


def test_freezing_twice_produces_identical_json_bytes() -> None:
    rows = _rows((0, 6, 12, 18, 24, 30), (0.90, 0.89, 0.88, 0.87, 0.86, 0.50), decisions=(0, 0, 0, 0, 0, 2))
    evidence = _evidence(_span(0, 25, 31, [_contact(24)]))

    frozen, first_bytes = scorer.freeze_construction_twice(rows, rows, evidence, _intervals())
    second_bytes = scorer.deterministic_json_bytes(scorer.construction_payload(frozen))

    assert first_bytes == second_bytes
    assert b'"labels_read": false' in first_bytes


def test_label_blind_loader_does_not_import_timing_or_side_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = argparse.Namespace(
        feature_manifest=Path("not-used-features.json"),
        candidate_manifest=Path("not-used-candidates.json"),
        tree_results=Path("not-used-tree.json.gz"),
        evidence_manifest=Path("not-used-evidence.json"),
    )
    features = argparse.Namespace(manifest={})
    candidates = argparse.Namespace(tree_result={}, rows=np.empty(0))
    evidence = object()
    monkeypatch.setattr(scorer.tree_scorer, "verify_freeze", lambda _path: features)
    monkeypatch.setattr(scorer.tree_scorer, "verify_candidate_scores", lambda *_args: candidates)
    monkeypatch.setattr(scorer.tree_scorer, "_result_variant", lambda _result: scorer.tree_scorer.CANDIDATE_VARIANT)
    monkeypatch.setattr(scorer.evidence_scorer, "verify_freeze", lambda _path: evidence)
    monkeypatch.setattr(scorer.decision_scorer, "_model_rows", lambda _features: np.empty(0))
    monkeypatch.setattr(scorer.decision_scorer, "replay_all_trials", lambda _rows, _model: {"N+": np.empty(0)})
    monkeypatch.setattr(scorer.tree_scorer, "_manifest_intervals", lambda _manifest, _field: {})

    def fail_if_loaded() -> None:
        raise AssertionError("labels must not load during label-blind input verification")

    monkeypatch.setattr(scorer.rally_scorer, "_load_timing_rallies", fail_if_loaded)
    monkeypatch.setattr(scorer.rally_scorer, "_load_side_labels", fail_if_loaded)

    verified = scorer.load_label_blind_inputs(arguments)

    assert verified.features is features
    assert verified.candidates is candidates
    assert verified.evidence is evidence


def _candidate(frame: int, score: float = 0.8) -> scorer.PrefixCandidate:
    return scorer.PrefixCandidate("sset_01", 0, 0, frame, score, 1, (scorer.SOURCE_HGB_PEAK_1,))


def _fixed_event(frame: int, score: float = 0.8) -> scorer.rally_scorer.FixedEvent:
    return scorer.rally_scorer.FixedEvent("sset_01", frame, score, "Top")


def test_insert_local_dedup_preserves_later_events_and_replaces_only_anchor() -> None:
    events = (_fixed_event(20), _fixed_event(40))
    attribution = {("sset_01", 10): "Top", ("sset_01", 16): "Bot"}

    inserted = scorer.apply_insert_with_local_dedup(events, _candidate(10), attribution, 20, 5)
    replaced = scorer.apply_insert_with_local_dedup(events, _candidate(16), attribution, 20, 5)

    assert [event.frame for event in inserted] == [10, 20, 40]
    assert [event.frame for event in replaced] == [16, 40]
    assert replaced[0].predicted_side == "Bot"


def test_prepend_changes_output_span_without_starting_another_lookback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(_span(0, 20, 50))
    events = {"sset_01": (_fixed_event(15), _fixed_event(30))}
    candidate = _candidate(10)
    prefix = scorer.PrefixRecord(
        "sset_01",
        0,
        0,
        0,
        30,
        30,
        0.9,
        5,
        (candidate,),
        scorer.PrefixAction("insert", candidate, 30, "test"),
    )

    def fail_if_prefix_search_restarts(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("serve prepend must not start another prefix search")

    monkeypatch.setattr(scorer, "build_prefix_record", fail_if_prefix_search_restarts)
    final_events = scorer.apply_actions_to_stream(
        events,
        (prefix,),
        {("sset_01", 10): "Top"},
    )
    attached = scorer.attach_actions_to_spans(
        evidence,
        final_events,
        (prefix,),
    )

    bounds = attached.bounds[0]
    assert (bounds.detected_span_start, bounds.detected_span_end) == (20, 50)
    assert bounds.serve_prepend_frame == 10
    assert (bounds.output_span_start, bounds.output_span_end) == (10, 50)
    assert bounds.output_start_source == scorer.OUTPUT_START_SERVE_PREPEND
    assert (attached.spans[0].start_frame, attached.spans[0].end_frame) == (10, 50)
    assert [event.frame for event in attached.spans[0].events] == [10, 15, 30]
    assert scorer.rally_scorer.unassigned_events(attached.spans, final_events) == ()


def test_timing_oracle_uses_scaled_tolerance_and_score_then_frame_ties() -> None:
    prefix = scorer.PrefixRecord(
        "sset_01",
        0,
        0,
        0,
        120,
        120,
        0.9,
        5,
        (
            _candidate(90, 0.7),
            _candidate(110, 0.8),
            _candidate(112, 0.8),
        ),
        scorer.PrefixAction("abstain", None, 120, "pending"),
    )
    rally = scorer.rally_scorer.RallyReference("sset_01", 0, "set:1", (100, 130))

    selected = scorer.choose_timing_oracle_action(prefix, (rally,), 10, span_start=20, span_end=150)
    assert selected.candidate is not None
    assert selected.candidate.frame == 110
    already_matched = scorer.choose_timing_oracle_action(
        prefix,
        (rally,),
        10,
        span_start=20,
        span_end=150,
        baseline_event_frames=(101, 130),
    )
    assert already_matched.candidate is None
    assert already_matched.reason == "serve_already_matched"

    earlier_tie = scorer.PrefixRecord(
        prefix.fixture,
        prefix.span_id,
        prefix.interval_id,
        prefix.lower_frame,
        prefix.upper_frame,
        prefix.anchor_frame,
        prefix.anchor_score,
        prefix.radius_frames,
        (_candidate(90, 0.8), _candidate(110, 0.8)),
        prefix.fixed_action,
    )
    assert scorer.choose_timing_oracle_action(
        earlier_tie, (rally,), 10, span_start=20, span_end=150
    ).candidate.frame == 90

    fps_25_tolerance = scorer.tree_scorer._scaled_frames(10, 25.0)
    assert fps_25_tolerance == 8
    fps_prefix = scorer.PrefixRecord(
        prefix.fixture,
        prefix.span_id,
        prefix.interval_id,
        prefix.lower_frame,
        prefix.upper_frame,
        prefix.anchor_frame,
        prefix.anchor_score,
        prefix.radius_frames,
        (_candidate(92, 0.8), _candidate(108, 0.8)),
        prefix.fixed_action,
    )
    assert scorer.choose_timing_oracle_action(
        fps_prefix, (rally,), fps_25_tolerance, span_start=20, span_end=150
    ).candidate is not None
    assert scorer.choose_timing_oracle_action(
        earlier_tie, (rally,), fps_25_tolerance - 1, span_start=20, span_end=150
    ).candidate is None


def test_event_comparison_reports_greedy_identity_gain_and_loss() -> None:
    ground_truth = scorer.tree_scorer.GroundTruth(
        frames={fixture: np.empty(0, dtype=np.int32) for fixture in scorer.tree_scorer.FIXTURE_SPECS},
        serves={fixture: set() for fixture in scorer.tree_scorer.FIXTURE_SPECS},
        rally_count=0,
    )
    ground_truth.frames["sset_01"] = np.asarray([10, 20], dtype=np.int32)
    ground_truth.serves["sset_01"] = {10}
    baseline = {fixture: np.empty(0, dtype=np.int32) for fixture in scorer.tree_scorer.FIXTURE_SPECS}
    alternative = {fixture: values.copy() for fixture, values in baseline.items()}
    baseline["sset_01"] = np.asarray([10], dtype=np.int32)
    alternative["sset_01"] = np.asarray([20], dtype=np.int32)

    report = scorer._event_comparison(ground_truth, baseline, alternative, 10)

    assert report["newly_matched_contact_identities"] == [("sset_01", 20)]
    assert report["lost_contact_identities"] == [("sset_01", 10)]
    assert report["newly_matched_serve_identities"] == []
    assert report["lost_serve_identities"] == [("sset_01", 10)]


def test_deterministic_gzip_output_uses_zero_mtime(tmp_path: Path) -> None:
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    payload = {"schema": scorer.RESULTS_SCHEMA, "values": [2, 1]}

    scorer.rally_scorer.write_results(first, payload)
    scorer.rally_scorer.write_results(second, payload)

    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as source:
        assert json.load(source) == payload
