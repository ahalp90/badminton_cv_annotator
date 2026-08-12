"""Focused tests for the recording-only serve-prepend measurement."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from annotator.broadcast_timeline_labels import (
    SceneTruth,
    VideoMetadata,
    make_interval,
    write_label_csv,
)
from annotator.calibration.fixtures import SSET_01
from annotator.calibration.scoring import GtRally


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/scraper_pipeline/serve_prepend_lookback/measure_serve_prepend_lookback.py"
)
SPEC = importlib.util.spec_from_file_location("measure_serve_prepend_lookback", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
measurement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measurement)


def _pose_arrays(n_frames: int) -> tuple[np.ndarray, ...]:
    track = np.column_stack((
        np.full(n_frames, 0.5),
        np.full(n_frames, 0.5),
        np.ones(n_frames),
    ))
    bboxes = np.full((n_frames, 3, 4), np.nan)
    scores = np.full((n_frames, 3), np.nan)
    kps = np.full((n_frames, 3, 17, 2), np.nan)
    ndet = np.ones(n_frames, dtype=int)
    bboxes[:, 0] = (25, 20, 75, 90)
    scores[:, 0] = 0.9
    kps[:, 0] = (50, 50)
    return track, bboxes, scores, kps, ndet


def test_release_profile_pins_all_three_static_stride8_inputs() -> None:
    profile = measurement.FIXTURE_PROFILES["une-189c5af-static-stride8"]

    assert profile.source_commit == measurement.PROFILE_SOURCE_COMMIT
    assert [fixture.name for fixture in profile.fixtures] == ["sset_01", "sset_15", "sset_21"]
    assert [fixture.digests.dead_mask for fixture in profile.fixtures] == [
        "70a2a4e9cbd7c6c02b497b468682c462",
        "281562f7933f1fd24301bdba48bb26b9",
        "4d2dfde901ccb5253a54542e60585d71",
    ]


def test_expand_truth_validates_and_expands_reviewed_partition(tmp_path: Path) -> None:
    metadata = VideoMetadata("sset_01", 25.0, 6)
    intervals = [
        make_interval(metadata, 0, 2, SceneTruth.CUTAWAY),
        make_interval(metadata, 2, 6, SceneTruth.LIVE_NON_STANDARD),
    ]
    path = tmp_path / "labels.csv.gz"
    write_label_csv(path, intervals, metadata)

    truth, interval_count = measurement.expand_truth(path, metadata)

    assert interval_count == 2
    assert truth.tolist() == [
        "cutaway", "cutaway", "live-non-standard", "live-non-standard",
        "live-non-standard", "live-non-standard",
    ]


def test_central_pose_band_uses_largest_valid_detection() -> None:
    track, bboxes, scores, kps, ndet = _pose_arrays(1)
    ndet[0] = 3
    bboxes[0] = [
        (0, 0, 34, 100),
        (30, 20, 70, 80),
        (40, 10, 60, 90),
    ]
    scores[0] = (0.8, 0.9, 0.95)
    kps[0] = (50, 50)

    middle_half = measurement.central_pose_evidence(
        track=track,
        bboxes=bboxes,
        scores=scores,
        kps=kps,
        ndet=ndet,
        frame=0,
        resolution=(100.0, 100.0),
        band_fraction=0.5,
    )
    middle_two_thirds = measurement.central_pose_evidence(
        track=track,
        bboxes=bboxes,
        scores=scores,
        kps=kps,
        ndet=ndet,
        frame=0,
        resolution=(100.0, 100.0),
        band_fraction=2 / 3,
    )

    assert middle_half.slot == 1
    assert middle_two_thirds.slot == 0
    assert middle_half.n_valid_detections == 3
    assert middle_half.n_central_detections == 2


def test_candidate_rule_records_truth_mask_and_pose_verdicts(monkeypatch) -> None:
    monkeypatch.setattr(
        measurement.candidate_measurement,
        "detect_contact_flags",
        lambda *_args, **_kwargs: [(7, 5.0), (8, 4.0), (16, 9.0)],
    )
    fixture = replace(SSET_01, resolution=(100.0, 100.0))
    arrays = _pose_arrays(30)
    truth = np.full(30, "live-non-standard", dtype="U18")
    truth[7] = "replay"
    inpaint_codes = np.zeros(30, dtype=np.uint8)
    court_present = np.zeros(30, dtype=bool)
    raw_mask = np.zeros(30, dtype=bool)
    definitive_mask = np.zeros(30, dtype=bool)
    raw_mask[7] = definitive_mask[7] = True
    rallies = [GtRally("set1", 1, (8, 20))]
    statuses = [{
        "rally_index": 0,
        "serve_frame": 8,
        "serve_matched": False,
        "later_strokes_matched": True,
        "target_miss": True,
    }]

    candidates, opportunities, summary = measurement.evaluate_serve_candidates(
        fixture=fixture,
        band="middle-half",
        band_fraction=0.5,
        arrays=arrays,
        truth=truth,
        inpaint_codes=inpaint_codes,
        court_present=court_present,
        raw_mask=raw_mask,
        definitive_mask=definitive_mask,
        spans=[(10, 25)],
        raw_by_span={},
        accepted_by_span={0: [20]},
        gt_rallies=rallies,
        statuses=statuses,
    )

    by_frame = {row["candidate_frame"]: row for row in candidates}
    assert by_frame[7]["evidence_pass"] is True
    assert by_frame[7]["anchor_suppression_pass"] is True
    assert by_frame[7]["policy_pass"] is False
    assert by_frame[7]["selected_evidence"] is True
    assert by_frame[7]["reject_reasons"] == "definitive-mask"
    assert by_frame[8]["manual_truth"] == "live-non-standard"
    assert by_frame[8]["selected_policy"] is True
    assert by_frame[16]["evidence_pass"] is True
    assert by_frame[16]["anchor_suppression_pass"] is False
    assert by_frame[16]["candidate_pass"] is False
    assert by_frame[16]["selected_evidence"] is False
    assert "contact-suppression-with-anchor" in by_frame[16]["reject_reasons"]
    assert opportunities[0]["selected_evidence_frame"] == 7
    assert opportunities[0]["selected_policy_frame"] == 8
    assert opportunities[0]["candidate_pass_count"] == 2
    assert summary["trigger_metrics"]["evidence_only"]["true_positives"] == 1
    assert summary["trigger_metrics"]["current_mask_policy"]["false_positives"] == 0
    assert summary["target_outcomes"]["current_mask_policy"]["selected_match"] == 1


def test_contact_injection_copies_accepted_contacts_and_keeps_spans(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_run_video(*_args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            spans=[(10, 25)],
            filtered_contacts=[
                SimpleNamespace(rally_id=0, contact_frame=8),
                SimpleNamespace(rally_id=0, contact_frame=20),
            ],
            filtered_by_rally={0: [8, 20]},
            n_strokes_list=[2],
            next_servers=["top"],
        )

    monkeypatch.setattr(measurement.candidate_measurement, "run_video", fake_run_video)
    natural = SimpleNamespace(
        spans=[(10, 25)],
        filtered_contacts=[SimpleNamespace(rally_id=0, contact_frame=20)],
        filtered_by_rally={0: [20]},
        n_strokes_list=[1],
        next_servers=["bottom"],
    )
    statuses = [{
        "rally_index": 0,
        "serve_frame": 8,
        "serve_matched": False,
        "later_strokes_matched": True,
        "target_miss": True,
    }]

    summary = measurement.run_contact_injection_counterfactual(
        inputs=SimpleNamespace(
            positional=(),
            keyword={"raw_exclusion_mask": np.ones(30, dtype=bool)},
        ),
        natural_result=natural,
        gt_rallies=[GtRally("set1", 1, (8, 20))],
        natural_statuses=statuses,
        candidate_rows=[{"selected_evidence": True, "span_id": 0, "candidate_frame": 8}],
        tolerance=4,
        selection_field="selected_evidence",
        exempt_selected_from_raw_mask=True,
    )

    assert seen["spans"] == [(10, 25)]
    assert seen["contacts"] == {0: [8, 20]}
    assert seen["raw_exclusion_mask"][8] == np.False_
    assert summary["exempted_raw_mask_frames"] == 1
    assert summary["target_serves_recovered"] == 1
    assert summary["all_unmatched_serves_recovered"] == 1
    assert summary["accepted_contact_delta"] == 1
    assert summary["span_delta"] == 0


def test_schema_csv_writer_is_deterministic_for_empty_rows(tmp_path: Path) -> None:
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"

    measurement._write_schema_csv_gz([], ("a", "b"), first)
    measurement._write_schema_csv_gz([], ("a", "b"), second)

    assert first.read_bytes() == second.read_bytes()
