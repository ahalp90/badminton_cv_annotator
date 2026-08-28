"""Focused tests for the 32-case rally-start VLM gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import rally_start_trials as trials


def _case(tmp_path: Path) -> trials.RallyStartCase:
    return trials.RallyStartCase(
        case_id="rally-start-sset_21-c00",
        video_id="sset_21",
        clip_path=tmp_path / "clip.mp4",
        source_start_frame=100,
        source_end_frame=220,
        sample_fps=30.0,
    )


def test_select_anchor_prefers_first_court_supported_cut() -> None:
    court_present = np.ones(300, dtype=np.bool_)

    selected = trials._select_anchor(
        [170, 220, 250],
        [[0, 140], [140, 200], [200, 300]],
        court_present,
        30.0,
    )

    assert selected == {
        "anchor_frame": 140,
        "selection_method": "court-supported-cut",
        "selected_cut_frame": 140,
        "contact_after_cut_frame": 170,
        "cut_to_contact_frames": 30,
        "court_confirm_fraction": 1.0,
    }


def test_select_anchor_falls_back_to_earliest_contact() -> None:
    court_present = np.zeros(300, dtype=np.bool_)

    selected = trials._select_anchor(
        [170, 220],
        [[0, 140], [140, 200], [200, 300]],
        court_present,
        30.0,
    )

    assert selected["selection_method"] == "earliest-accepted-contact"
    assert selected["anchor_frame"] == 170
    assert selected["selected_cut_frame"] is None


def test_nearby_contacts_preserve_recorded_anchor() -> None:
    result = {
        "spans": [[100, 200], [250, 300]],
        "filtered_by_rally": {"0": [120, 150, 180], "1": [260]},
    }

    guesses, rally_id = trials._nearby_current_contacts(result, 140, 30.0)

    assert guesses == [140, 150, 180]
    assert rally_id == 0


def test_nearby_contacts_keep_frozen_anchor_when_current_span_is_absent() -> None:
    result = {
        "spans": [[100, 130], [180, 220]],
        "filtered_by_rally": {"0": [120], "1": [190]},
    }

    guesses, rally_id = trials._nearby_current_contacts(result, 150, 30.0)

    assert guesses == [150]
    assert rally_id is None


def test_parse_response_requires_frame_only_for_visible_state() -> None:
    parsed = trials.parse_response(
        '{"server":"bottom","serve_state":"visible","contact_frame":51}',
    )
    assert parsed["contact_frame"] == 51

    with pytest.raises(ValueError, match="must be null"):
        trials.parse_response(
            '{"server":"bottom","serve_state":"off_frame","contact_frame":51}',
        )
    with pytest.raises(ValueError, match="requires contact_frame"):
        trials.parse_response(
            '{"server":"bottom","serve_state":"visible","contact_frame":null}',
        )


def test_scoring_normalises_only_leading_zero_contact_frame() -> None:
    raw_response = (
        '{"server":"top","serve_state":"visible","contact_frame":004}'
    )

    assert trials._normalise_leading_zero_contact_frame(raw_response) == (
        '{"server":"top","serve_state":"visible","contact_frame":4}'
    )
    assert trials._normalise_leading_zero_contact_frame(
        '{"server":"004","serve_state":"visible","contact_frame":4}'
    ) is None


def test_manifest_rejects_human_truth_keys(tmp_path: Path) -> None:
    manifest = {
        "schema": trials.MANIFEST_SCHEMA,
        "cases": [
            {
                "case_id": "case-1",
                "video_id": "sset_21",
                "clip_path": "clips/case-1.mp4",
                "source_start_frame": 100,
                "source_end_frame": 220,
                "sample_fps": 30.0,
                "expected_frames": 120,
                "width": 512,
                "height": 288,
                "serve_visibility": "visible",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden key 'serve_visibility'"):
        trials.load_manifest(path, require_clips=False)


def test_score_counts_timing_abstention_and_false_frame_claim(tmp_path: Path) -> None:
    visible_case = _case(tmp_path)
    omitted_case = trials.RallyStartCase(
        case_id="rally-start-sset_21-c01",
        video_id="sset_21",
        clip_path=tmp_path / "clip-2.mp4",
        source_start_frame=200,
        source_end_frame=320,
        sample_fps=30.0,
    )
    truth = {
        visible_case.case_id: {
            "expected_server": "bottom",
            "expected_serve_state": "visible",
            "visible_contact_frame": 150,
            "accepted_tolerance_frames": 5,
        },
        omitted_case.case_id: {
            "expected_server": "top",
            "expected_serve_state": "broadcast_omitted",
            "visible_contact_frame": None,
            "accepted_tolerance_frames": 5,
        },
    }
    attempts = {
        visible_case.case_id: {
            "parsed_response": {
                "server": "bottom",
                "serve_state": "visible",
                "contact_frame": 54,
            },
            "model": {"model_id": "test"},
            "prompt_sha256": "prompt",
            "sampling": {"sampled_input_frames": list(range(120))},
        },
        omitted_case.case_id: {
            "parsed_response": {
                "server": "unclear",
                "serve_state": "visible",
                "contact_frame": 50,
            },
            "model": {"model_id": "test"},
            "prompt_sha256": "prompt",
            "sampling": {"sampled_input_frames": list(range(120))},
        },
    }

    score = trials._score_backend((visible_case, omitted_case), truth, attempts)

    assert score["server"]["correct"] == 1
    assert score["serve_state"]["correct"] == 1
    assert score["contact_timing_visible_truth"]["within_project_tolerance"] == 1
    assert score["false_exact_frame_claims_on_nonvisible_truth"] == 1
    assert score["abstention"]["either_field_unclear"] == 1
    assert score["sampled_input_frame_counts"] == {"120": 2}
