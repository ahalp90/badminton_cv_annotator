"""Focused tests for the issue #104 commentary benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from annotator.calibration.commentary_benchmark import (
    _cleaning_aggregate,
    _leave_one_video_out,
    _pairing_aggregate,
    evaluate_video,
)
from annotator.calibration.commentary_benchmark_inputs import (
    EXPECTED_SHUTTLESET22_IDS,
    EXPECTED_SHUTTLESET_IDS,
    VideoInputs,
    _validate_repair_metadata,
    _validate_rallies,
    _validate_timed_rows,
    _validate_unique_chunk_starts,
)


def _chunk(
    chunk_id: str, start: float, end: float, text: str = "raw words"
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "start": start,
        "end": end,
        "text": text,
        "text_clean": "clean words",
        "alt_phrasings": ["one", "two", "three"],
        "bert_f1": 0.9,
        "clean_pass": True,
    }


def test_exact_populations_are_unique() -> None:
    assert len(EXPECTED_SHUTTLESET_IDS) == 40
    assert len(EXPECTED_SHUTTLESET22_IDS) == 47
    assert EXPECTED_SHUTTLESET_IDS.isdisjoint(EXPECTED_SHUTTLESET22_IDS)


def test_repaired_bundle_identity_is_explicit() -> None:
    _validate_repair_metadata(
        {
            "repair": {
                "source_manifest_sha256": (
                    "52a9933bcfd8d4d1cf7c032473181fa637bd296f073fbf09ff69ab0f5334342c"
                ),
                "identity_policy": "one chunk per video and coarse source start",
                "removed_overlap_row_count": 533,
                "affected_video_count": 86,
                "paid_request_count": 0,
            }
        },
        "fixture",
    )
    with pytest.raises(ValueError, match="repair identity differs"):
        _validate_repair_metadata(
            {
                "repair": {
                    "source_manifest_sha256": "wrong",
                    "identity_policy": "one chunk per video and coarse source start",
                    "removed_overlap_row_count": 533,
                    "affected_video_count": 86,
                    "paid_request_count": 0,
                }
            },
            "fixture",
        )


def test_direct_timeline_and_rally_validation_rejects_bad_associations() -> None:
    _validate_timed_rows(
        [
            {"start": 0.0, "end": 0.5, "text": "first"},
            {"start": 0.5, "end": 1.0, "text": "second"},
        ],
        1.0,
        "transcript",
    )
    _validate_rallies("video", [(0, 0, 10)], 10, "human")

    with pytest.raises(ValueError, match="not timestamp-ordered"):
        _validate_timed_rows(
            [
                {"start": 0.5, "end": 1.0, "text": "first"},
                {"start": 0.4, "end": 1.1, "text": "second"},
            ],
            2.0,
            "transcript",
        )
    with pytest.raises(ValueError, match="outside the local source"):
        _validate_rallies("video", [(0, 0, 11)], 10, "human")

    with pytest.raises(ValueError, match="duplicate start time"):
        _validate_unique_chunk_starts(
            [
                _chunk("video_c0", 0.0, 0.5, "same"),
                _chunk("video_c1", 0.0, 0.7, "wider variant"),
            ],
            "video cleaned chunks",
        )


def test_evaluate_video_reuses_post_rally_pairing_and_reports_five_second_sensitivity() -> (
    None
):
    inputs = VideoInputs(
        dataset="ShuttleSet",
        video_id="video",
        fps=10.0,
        frame_count=200,
        transcript_source="youtube_asr",
        transcript_segments=12,
        chunks=(
            _chunk("video_c0", 5.5, 5.8),
            _chunk("video_c1", 7.0, 7.2),
            _chunk("video_c2", 15.5, 15.8),
            _chunk("video_c3", 16.0, 16.2),
        ),
        rallies=((0, 0, 50), (1, 60, 100), (2, 110, 150)),
        human_rallies=((0, 0, 50), (1, 60, 100), (2, 110, 150)),
        replay_mask=None,
        annotation_population=None,
    )

    result, gaps, scores, word_deltas = evaluate_video(inputs)

    pairing = result["primary_pairing"]
    assert pairing["rallies"] == 3
    assert pairing["paired_rallies_8s"] == 3
    assert pairing["paired_rallies_5s"] == 2
    assert pairing["paired_chunks"] == 3
    assert pairing["unpaired_cleaned_chunks"] == 1
    assert pairing["in_rally_chunk_starts"] == 1
    assert pairing["paired_chunks_starting_in_another_rally"] == 0
    assert gaps == {
        "primary": [0.5, 5.5, 1.0],
        "human_contact": [0.5, 5.5, 1.0],
    }
    assert scores == [0.9] * 4
    assert word_deltas == [0] * 4


def test_five_second_sensitivity_runs_an_independent_greedy_join() -> None:
    inputs = VideoInputs(
        dataset="ShuttleSet22",
        video_id="video",
        fps=10.0,
        frame_count=100,
        transcript_source="youtube_asr",
        transcript_segments=1,
        chunks=(_chunk("video_c0", 6.0, 6.5),),
        rallies=((0, 0, 1), (1, 10, 50)),
        human_rallies=((0, 0, 1), (1, 10, 50)),
        replay_mask=None,
        annotation_population=None,
    )

    result, _gaps, _scores, _word_deltas = evaluate_video(inputs)

    pairing = result["primary_pairing"]
    assert pairing["paired_rallies_8s"] == 1
    assert pairing["paired_rallies_5s"] == 1


def test_replay_mask_holds_rally_out_and_leaves_chunk_for_later_rally() -> None:
    replay_mask = np.zeros(200, dtype=np.bool_)
    replay_mask[65:95] = True
    inputs = VideoInputs(
        dataset="ShuttleSet",
        video_id="video",
        fps=10.0,
        frame_count=200,
        transcript_source="whisper",
        transcript_segments=8,
        chunks=(
            _chunk("video_c0", 5.5, 5.8),
            _chunk("video_c1", 15.5, 15.8),
        ),
        rallies=((0, 0, 50), (1, 60, 100), (2, 110, 150)),
        human_rallies=((0, 0, 50), (1, 60, 100), (2, 110, 150)),
        replay_mask=replay_mask,
        annotation_population=None,
    )

    result, gaps, _scores, _word_deltas = evaluate_video(inputs)

    pairing = result["primary_pairing"]
    assert pairing["replay_masked_rallies"] == 1
    assert pairing["pairing_eligible_rallies"] == 2
    assert pairing["paired_rallies_8s"] == 2
    assert gaps["primary"] == [0.5, 0.5]


def test_triage_dropped_video_remains_an_explicit_zero_commentary_case() -> None:
    inputs = VideoInputs(
        dataset="ShuttleSet22",
        video_id="ss22_17",
        fps=30.0,
        frame_count=300,
        transcript_source="whisper",
        transcript_segments=4,
        chunks=(),
        rallies=((0, 0, 30),),
        human_rallies=((0, 0, 30),),
        replay_mask=None,
        annotation_population={"usable_rallies": 1},
    )

    result, gaps, scores, word_deltas = evaluate_video(inputs)

    assert result["primary_pairing"]["rallies"] == 1
    assert result["cleaned_chunks"] == 0
    assert result["primary_pairing"]["paired_rallies_8s"] == 0
    assert result["primary_pairing"]["unpaired_cleaned_chunks"] == 0
    assert result["bert_f1"]["count"] == 0
    assert gaps == {"primary": [], "human_contact": []}
    assert scores == []
    assert word_deltas == []


def test_evaluate_video_rejects_empty_cleaned_text() -> None:
    chunk = _chunk("video_c0", 1.1, 1.2)
    chunk["text_clean"] = " "
    inputs = VideoInputs(
        dataset="ShuttleSet22",
        video_id="video",
        fps=10.0,
        frame_count=100,
        transcript_source="youtube_asr",
        transcript_segments=1,
        chunks=(chunk,),
        rallies=((0, 0, 10),),
        human_rallies=((0, 0, 10),),
        replay_mask=None,
        annotation_population=None,
    )

    with pytest.raises(ValueError, match="empty cleaned text"):
        evaluate_video(inputs)


def test_aggregate_and_leave_one_video_out_use_eligible_rally_denominator() -> None:
    rows = [
        {
            "video_id": "a",
            "transcript_source": "youtube_asr",
            "transcript_segments": 1,
            "cleaned_chunks": 1,
            "raw_words": 2,
            "clean_words": 2,
            "primary_pairing": {
                "rallies": 10,
                "replay_masked_rallies": 2,
                "pairing_eligible_rallies": 8,
                "paired_rallies_8s": 4,
                "paired_rallies_5s": 3,
                "paired_chunks": 4,
                "unpaired_cleaned_chunks": 0,
                "in_rally_chunk_starts": 0,
                "paired_chunks_starting_in_another_rally": 0,
            },
        },
        {
            "video_id": "b",
            "transcript_source": "whisper",
            "transcript_segments": 1,
            "cleaned_chunks": 1,
            "raw_words": 2,
            "clean_words": 2,
            "primary_pairing": {
                "rallies": 4,
                "replay_masked_rallies": 0,
                "pairing_eligible_rallies": 4,
                "paired_rallies_8s": 1,
                "paired_rallies_5s": 1,
                "paired_chunks": 1,
                "unpaired_cleaned_chunks": 0,
                "in_rally_chunk_starts": 0,
                "paired_chunks_starting_in_another_rally": 0,
            },
        },
    ]

    cleaning = _cleaning_aggregate(rows, [0.9, 0.91])
    aggregate = _pairing_aggregate(rows, [1.0, 2.0], "primary_pairing")
    leave_one_out = _leave_one_video_out(rows, "primary_pairing")

    assert aggregate["pairing_eligible_rallies"] == 12
    assert aggregate["paired_rallies_8s"] == 5
    assert aggregate["paired_rally_rate_8s"] == 5 / 12
    assert cleaning["transcript_sources"] == {"youtube_asr": 1, "whisper": 1}
    assert leave_one_out == {
        "paired_rally_rate_8s_min": 0.25,
        "paired_rally_rate_8s_max": 0.5,
    }
