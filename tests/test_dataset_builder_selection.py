"""Bounded visual selection and compressed decision-table contracts."""

from pathlib import Path

import pytest

from dataset_builder.selection import (
    COMMENTARY_FAILED,
    COMMENTARY_INELIGIBLE,
    COMMENTARY_NO_PAIR,
    COMMENTARY_NO_RETAINED_CHUNK,
    COMMENTARY_UNAVAILABLE_TRANSCRIPT,
    COMMENTARY_UNAVAILABLE_TRIAGE,
    load_selection,
    resolve_visual_selection,
    selected_video_ids,
    with_commentary_statuses,
    write_selection,
)


def _candidate(
    video_id: str,
    *,
    keep: str = "",
    substream: str = "match",
    doubles: bool = False,
    duration: bool = False,
    upload: bool = False,
) -> dict[str, object]:
    return {
        "video_id": video_id,
        "keep": keep,
        "substream": substream,
        "doubles_suspect": doubles,
        "duration_suspect": duration,
        "upload_date_suspect": upload,
    }


def test_selection_prefers_triage_then_transcript_fallback_and_applies_cap() -> None:
    candidates = [
        _candidate("fallback-no-transcript"),
        _candidate("triage-keep", keep="True"),
        _candidate("triage-reject", keep="False"),
        _candidate("fallback-with-transcript"),
        _candidate("suspect", doubles=True),
        _candidate("instructional", keep="True", substream="instructional"),
    ]

    decisions = resolve_visual_selection(
        candidates,
        max_videos=2,
        transcript_video_ids={
            "triage-keep", "triage-reject", "fallback-with-transcript",
            "suspect", "instructional",
        },
    )

    assert selected_video_ids(decisions) == ("triage-keep", "fallback-with-transcript")
    assert [decision.source_order for decision in decisions] == list(range(len(candidates)))
    by_id = {decision.video_id: decision for decision in decisions}
    assert by_id["triage-keep"].selection_source == "triage"
    assert by_id["fallback-with-transcript"].selection_source == "metadata_fallback"
    assert by_id["fallback-no-transcript"].selection_reason == "video_cap_reached"
    assert by_id["triage-reject"].selection_reason == "triage_rejected"
    assert by_id["triage-reject"].commentary_status == COMMENTARY_NO_RETAINED_CHUNK
    assert by_id["suspect"].selection_reason == "metadata_suspect"
    assert by_id["instructional"].commentary_status == COMMENTARY_INELIGIBLE


def test_fallback_never_selects_rejected_or_suspect_candidates() -> None:
    decisions = resolve_visual_selection(
        [
            _candidate("rejected", keep="False"),
            _candidate("doubles", doubles=True),
            _candidate("duration", duration=True),
            _candidate("upload", upload=True),
            _candidate("eligible"),
        ],
        max_videos=5,
    )

    assert selected_video_ids(decisions) == ("eligible",)
    assert decisions[0].commentary_status == COMMENTARY_UNAVAILABLE_TRANSCRIPT


def test_transcript_presence_is_only_a_fallback_ordering_preference() -> None:
    decisions = resolve_visual_selection(
        [_candidate("first"), _candidate("second"), _candidate("third")],
        max_videos=2,
        transcript_video_ids={"second"},
    )

    assert selected_video_ids(decisions) == ("first", "second")
    by_id = {decision.video_id: decision for decision in decisions}
    assert by_id["second"].commentary_status == COMMENTARY_UNAVAILABLE_TRIAGE
    assert by_id["first"].commentary_status == COMMENTARY_UNAVAILABLE_TRANSCRIPT
    assert by_id["third"].selection_reason == "video_cap_reached"


def test_zero_cap_records_every_candidate_without_selecting() -> None:
    decisions = resolve_visual_selection(
        [_candidate("kept", keep="True"), _candidate("fallback")],
        max_videos=0,
        transcript_video_ids={"kept"},
    )

    assert selected_video_ids(decisions) == ()
    assert all(decision.selection_reason == "video_cap_reached" for decision in decisions)


def test_selection_round_trip_is_deterministic_and_supports_late_statuses(
    tmp_path: Path,
) -> None:
    decisions = resolve_visual_selection(
        [_candidate("one", keep="True"), _candidate("two"), _candidate("three")],
        max_videos=3,
        transcript_video_ids={"one", "two"},
    )
    updated = with_commentary_statuses(
        decisions,
        {"one": COMMENTARY_NO_PAIR, "two": COMMENTARY_FAILED},
    )
    path = tmp_path / "selected_videos.csv.gz"

    write_selection(path, updated)
    first_bytes = path.read_bytes()
    write_selection(path, updated)

    assert path.read_bytes() == first_bytes
    assert load_selection(path) == updated


def test_selection_rejects_invalid_inputs_and_status_updates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate video_id"):
        resolve_visual_selection([_candidate("same"), _candidate("same")], max_videos=1)
    with pytest.raises(ValueError, match="non-negative integer"):
        resolve_visual_selection([], max_videos=-1)
    with pytest.raises(ValueError, match="not one string"):
        resolve_visual_selection(
            [_candidate("known")],
            max_videos=1,
            transcript_video_ids="known",
        )

    decisions = resolve_visual_selection([_candidate("known")], max_videos=1)
    with pytest.raises(ValueError, match="unknown videos"):
        with_commentary_statuses(decisions, {"other": COMMENTARY_NO_PAIR})
    with pytest.raises(ValueError, match="unsupported"):
        with_commentary_statuses(decisions, {"known": "pending"})
    with pytest.raises(ValueError, match="must end in .csv.gz"):
        write_selection(tmp_path / "selected.csv", decisions)
