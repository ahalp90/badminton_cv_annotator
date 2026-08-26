from pathlib import Path

from experiments.prompts import build_prompt
from experiments.trial_schema import TrialArm, TrialCase, TrialKind


def test_event_prior_prompt_explains_failed_wrist_evidence() -> None:
    case = TrialCase(
        case_id="event-serve",
        kind=TrialKind.EVENT,
        video_id="sset_01",
        clip_path=Path("unused.mp4"),
        source_start_frame=0,
        source_end_frame=50,
        candidate_frame=25,
        sample_fps=25.0,
        pipeline_priors={
            "court_present": True,
            "track_visible": True,
            "wrist_near": False,
            "proximity_ok": None,
            "suppressed": False,
            "raw_masked": False,
            "definitive_masked": False,
            "seconds_from_previous_raw_candidate": None,
            "seconds_to_next_raw_candidate": 0.12,
        },
    )

    prompt = build_prompt(case, TrialArm.PIPELINE_PRIORS)

    assert "inferred off-screen serve" in prompt
    assert "TOP as the far player and BOTTOM as the near player" in prompt
    assert "active rally play begins immediately after the cut" in prompt
    assert "pose wrist proximity supports this tracker point: no" in prompt
    assert "does not rule out a serve inferred across a broadcast cut" in prompt


def test_track_prompt_asks_about_tracker_identity_not_contact() -> None:
    case = TrialCase(
        case_id="track-test",
        kind=TrialKind.TRACK,
        video_id="sset_01",
        clip_path=Path("unused.mp4"),
        source_start_frame=0,
        source_end_frame=50,
        candidate_frame=25,
        sample_fps=25.0,
        pipeline_priors={"target_view": "full-frame"},
    )

    prompt = build_prompt(case, TrialArm.VIDEO_ONLY)

    assert "This is not a contact question" in prompt
    assert "repeats the short target interval slowly" in prompt
    assert "court text, a logo, a racket" in prompt


def test_track_prompt_explains_zoom_without_treating_it_as_evidence() -> None:
    case = TrialCase(
        case_id="track-zoom",
        kind=TrialKind.TRACK,
        video_id="sset_01",
        clip_path=Path("unused.mp4"),
        source_start_frame=0,
        source_end_frame=50,
        candidate_frame=25,
        sample_fps=25.0,
        pipeline_priors={"target_view": "tracker-centred-zoom"},
    )

    prompt = build_prompt(case, TrialArm.VIDEO_ONLY)

    assert "fixed enlarged view around the claimed track" in prompt
    assert "does not change the underlying frames" in prompt


def test_track_prompt_uses_clean_replay_to_counter_marker_anchoring() -> None:
    case = TrialCase(
        case_id="track-clean-zoom",
        kind=TrialKind.TRACK,
        video_id="sset_01",
        clip_path=Path("unused.mp4"),
        source_start_frame=0,
        source_end_frame=50,
        candidate_frame=25,
        sample_fps=25.0,
        pipeline_priors={"target_view": "clean-then-marked-zoom"},
    )

    prompt = build_prompt(case, TrialArm.VIDEO_ONLY)

    assert "first without the cyan marker, then with it" in prompt
    assert "Do not infer a shuttle from the marker" in prompt


def test_broadcast_prompt_explains_dense_target_sampling() -> None:
    case = TrialCase(
        case_id="broadcast-dense",
        kind=TrialKind.BROADCAST,
        video_id="sset_01",
        clip_path=Path("unused.mp4"),
        source_start_frame=0,
        source_end_frame=500,
        candidate_frame=None,
        sample_fps=2.5,
        pipeline_priors={"sampling_layout": "dense-four-second-target"},
    )

    prompt = build_prompt(case, TrialArm.VIDEO_ONLY)

    assert "30 consecutive gold-bordered frames" in prompt
    assert "four-second target" in prompt
    assert "sparsely show the surrounding broadcast order" in prompt
