from __future__ import annotations

import hashlib

import pytest

from scratch.vlm_pr80_eval.experiments.rally_opening_trials import (
    ATTEMPT_SCHEMA,
    _navigation_sentence,
    _paired_summary,
    _standard_window,
    _validate_attempt,
    parse_response,
)


def _case(**overrides: object) -> dict[str, object]:
    case: dict[str, object] = {
        "case_id": "case-1",
        "fps": 25.0,
        "total_video_frames": 10_000,
        "window_start_frame": 1_000,
        "window_end_frame_exclusive": 1_300,
        "qualifying_cut_frames": [1_125],
        "early_contact_frames": [1_130, 1_160, 1_200],
    }
    case.update(overrides)
    return case


def test_standard_window_contains_route_at_fixed_duration() -> None:
    start, end = _standard_window(_case())

    assert end - start == 550
    assert start <= 1_000
    assert end >= 1_300


def test_standard_window_shifts_inside_video_boundary() -> None:
    assert _standard_window(
        _case(window_start_frame=50, window_end_frame_exclusive=300)
    ) == (0, 550)


def test_standard_window_rejects_route_longer_than_trial_clip() -> None:
    with pytest.raises(ValueError, match="routed window exceeds"):
        _standard_window(_case(window_end_frame_exclusive=1_551))


def test_navigation_sentence_translates_frames_to_plain_times() -> None:
    sentence = _navigation_sentence(_case(), 900)

    assert "shot change at 9.0 seconds" in sentence
    assert "between 9.2 and 12.0 seconds" in sentence
    assert "do not identify the server" in sentence


def test_parse_response_accepts_exact_server_reply() -> None:
    assert parse_response('{"server":"bottom","evidence":"The near player served."}') == {
        "server": "bottom",
        "evidence": "The near player served.",
    }


@pytest.mark.parametrize(
    "reply",
    [
        '{"server":"bot","evidence":"Near player."}',
        '{"server":"top"}',
        '{"server":"top","evidence":""}',
    ],
)
def test_parse_response_rejects_unsupported_replies(reply: str) -> None:
    with pytest.raises(ValueError):
        parse_response(reply)


def test_validate_attempt_checks_identity_and_sampled_grid() -> None:
    case = {
        "trial_id": "case-1--cued_native",
        "case_id": "case-1",
        "video_id": "sset_21",
        "arm": "cued_native",
        "clip_sha256": "clip-hash",
        "prompt": "Prompt text",
        "expected_input_frames": 3,
        "sample_fps": 30.0,
    }
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "trial_id": case["trial_id"],
        "case_id": case["case_id"],
        "video_id": case["video_id"],
        "arm": case["arm"],
        "clip_sha256": case["clip_sha256"],
        "prompt": case["prompt"],
        "prompt_sha256": hashlib.sha256(b"Prompt text").hexdigest(),
        "generation_error": None,
        "sampling": {"sampled_input_frames": [0, 1, 2], "requested_fps": 30.0},
    }

    _validate_attempt(attempt, case)
    attempt["clip_sha256"] = "wrong"
    with pytest.raises(ValueError, match="clip_sha256 differs"):
        _validate_attempt(attempt, case)


def test_paired_summary_counts_changed_outcomes() -> None:
    rows = [
        {"case_id": "a", "arm": "left", "server_correct": True, "predicted_server": "top"},
        {"case_id": "b", "arm": "left", "server_correct": False, "predicted_server": "top"},
        {"case_id": "a", "arm": "right", "server_correct": False, "predicted_server": "bottom"},
        {"case_id": "b", "arm": "right", "server_correct": True, "predicted_server": "bottom"},
    ]

    assert _paired_summary(rows, "left", "right") == {
        "left_arm": "left",
        "right_arm": "right",
        "cases": 2,
        "both_correct": 0,
        "left_only_correct": 1,
        "right_only_correct": 1,
        "both_wrong": 0,
        "changed_predictions": 2,
    }
