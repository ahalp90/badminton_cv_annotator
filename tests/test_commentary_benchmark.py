"""Focused tests for the issue #104 commentary benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from annotator.calibration.commentary_benchmark import (
    LAG_SWEEP_SECONDS,
    _cleaning_aggregate,
    _dataset_aggregate,
    _leave_one_video_out,
    _multi_rally_aggregate,
    _multi_rally_pairing_summary,
    _pairing_aggregate,
    evaluate_video,
)
from annotator.calibration.commentary_benchmark_inputs import (
    EXPECTED_SHUTTLESET22_IDS,
    EXPECTED_SHUTTLESET_IDS,
    RETIMED_RELATIVE_DIR,
    VideoInputs,
    _load_retimed_chunks_for_video,
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


def _retimed(
    chunk: dict[str, object],
    start: float,
    end: float,
    status: str = "aligned",
    match_ratio: float | None = 0.8,
) -> dict[str, object]:
    """Build the re-timed sidecar row for one cleaned chunk."""
    row = dict(chunk)
    row["coarse_start"] = float(chunk["start"])
    row["coarse_end"] = float(chunk["end"])
    row["start"] = start
    row["end"] = end
    row["align_status"] = status
    row["align_match_ratio"] = match_ratio
    row["align_shift_s"] = (
        start - float(chunk["start"]) if status == "aligned" else None
    )
    return row


def _write_retimed(root: Path, video_id: str, rows: list[dict[str, object]]) -> Path:
    """Write a re-timed sidecar under a fake commentary root and return its path."""
    directory = root / RETIMED_RELATIVE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{video_id}.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


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
        "aligned_primary": [],
        "aligned_human_contact": [],
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
    assert gaps == {
        "primary": [],
        "human_contact": [],
        "aligned_primary": [],
        "aligned_human_contact": [],
    }
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


def _inputs(
    chunks: tuple[dict[str, object], ...],
    rallies: tuple[tuple[int, int, int], ...],
    replay_mask: np.ndarray | None = None,
    retimed_chunks: tuple[dict[str, object], ...] | None = None,
    dataset: str = "ShuttleSet",
) -> VideoInputs:
    """A video at 10 fps with the same rallies in both rally views."""
    return VideoInputs(
        dataset=dataset,
        video_id="video",
        fps=10.0,
        frame_count=400,
        transcript_source="youtube_asr",
        transcript_segments=4,
        chunks=chunks,
        rallies=rallies,
        human_rallies=rallies,
        replay_mask=replay_mask,
        annotation_population=None,
        retimed_chunks=retimed_chunks,
        retimed_sha256=None if retimed_chunks is None else "sha",
    )


RALLIES = ((0, 0, 50), (1, 60, 100), (2, 200, 250))  # seconds: 0-5, 6-10, 20-25


def test_multi_rally_summary_ascribes_ambiguous_chunks_to_both_rallies() -> None:
    chunks = (
        _chunk("video_c0", 5.5, 5.8),  # 0.5 s after rally 0, before rally 1: one rally
        _chunk("video_c1", 7.0, 7.2),  # inside rally 1 and within 8 s of rally 0: both
        _chunk("video_c2", 26.0, 26.2),  # 1 s after rally 2, 16 s after rally 1: one rally
        _chunk("video_c3", 45.0, 45.2),  # 20 s after the last rally: nobody
    )
    summary = _multi_rally_pairing_summary(_inputs(chunks, RALLIES), RALLIES, 8.0)
    assert summary == {
        "rallies": 3,
        "replay_masked_rallies": 0,
        "pairing_eligible_rallies": 3,
        "paired_rallies": 3,
        "rallies_with_in_rally_chunk": 1,
        "in_rally_chunks": 1,
        "post_rally_chunks": 2,
        "single_rally_chunks": 2,
        "ambiguous_chunks": 1,
        "unassigned_chunks": 1,
        "masked_only_chunks": 0,
        "unpairable_masked_start_chunks": 0,
        "masked_rally_candidate_incidences": 0,
    }


def test_multi_rally_summary_window_decides_post_rally_reach() -> None:
    chunks = (_chunk("video_c0", 28.0, 28.2),)  # 3 s after rally 2
    inputs = _inputs(chunks, RALLIES)
    assert _multi_rally_pairing_summary(inputs, RALLIES, 8.0)["post_rally_chunks"] == 1
    narrow = _multi_rally_pairing_summary(inputs, RALLIES, 2.0)
    assert narrow["post_rally_chunks"] == 0
    assert narrow["unassigned_chunks"] == 1
    assert narrow["paired_rallies"] == 0


def test_multi_rally_summary_drops_masked_rallies_and_masked_chunk_starts() -> None:
    replay_mask = np.zeros(400, dtype=np.bool_)
    replay_mask[65:95] = True  # rally 1 is held out
    chunks = (
        _chunk("video_c0", 6.2, 6.4),  # inside masked rally 1, off the mask, 1.2 s after rally 0
        _chunk("video_c1", 7.0, 7.2),  # starts on a masked frame: unpairable
    )
    summary = _multi_rally_pairing_summary(_inputs(chunks, RALLIES, replay_mask), RALLIES, 8.0)
    assert summary["replay_masked_rallies"] == 1
    assert summary["pairing_eligible_rallies"] == 2
    assert summary["unpairable_masked_start_chunks"] == 1
    assert summary["masked_rally_candidate_incidences"] == 1
    assert summary["masked_only_chunks"] == 0
    assert summary["in_rally_chunks"] == 0
    assert summary["post_rally_chunks"] == 1
    assert summary["single_rally_chunks"] == 1
    assert summary["paired_rallies"] == 1


def test_evaluate_video_reports_aligned_views_only_with_retimed_chunks() -> None:
    chunks = (_chunk("video_c0", 50.0, 50.5), _chunk("video_c1", 60.0, 60.5))  # both far past every rally
    retimed = (
        _retimed(chunks[0], 5.5, 5.8),  # snapped back to just after rally 0
        _retimed(chunks[1], 60.0, 60.5, status="unmatched", match_ratio=None),
    )
    result, gaps, _scores, _deltas = evaluate_video(_inputs(chunks, RALLIES, retimed_chunks=retimed))
    assert result["primary_pairing"]["paired_rallies_8s"] == 0  # coarse times pair nothing
    assert result["aligned_primary_pairing"]["paired_rallies_8s"] == 1
    assert result["aligned_multi_human_contact_pairing"]["paired_rallies"] == 1
    assert list(result["aligned_lag_sweep_primary"]) == [f"{window:04.1f}" for window in LAG_SWEEP_SECONDS]
    assert result["retiming"]["status_counts"] == {"aligned": 1, "unmatched": 1, "collision": 0}
    assert result["retiming"]["abs_shift_seconds"]["max"] == pytest.approx(44.5)
    assert gaps["aligned_primary"] == [0.5]

    bare, bare_gaps, _scores, _deltas = evaluate_video(_inputs(chunks, RALLIES))
    assert bare["aligned_primary_pairing"] is None
    assert bare["aligned_lag_sweep_human_contact"] is None
    assert bare["retiming"] is None
    assert bare_gaps["aligned_primary"] == []


def test_retimed_sidecar_loader_ties_rows_to_the_cleaned_chunks(tmp_path: Path) -> None:
    chunks = [_chunk("video_c0", 30.0, 30.5), _chunk("video_c1", 40.0, 40.5)]
    assert _load_retimed_chunks_for_video(tmp_path, "video", chunks, 100.0) == (None, None)

    good = [_retimed(chunks[0], 5.5, 5.8), _retimed(chunks[1], 40.0, 40.5, status="unmatched", match_ratio=None)]
    path = _write_retimed(tmp_path, "video", good)
    rows, sha256 = _load_retimed_chunks_for_video(tmp_path, "video", chunks, 100.0)
    assert [row["chunk_id"] for row in rows] == ["video_c0", "video_c1"]
    assert len(sha256) == 64 and path.is_file()

    _write_retimed(tmp_path, "video", good[:1])
    with pytest.raises(ValueError, match="population"):
        _load_retimed_chunks_for_video(tmp_path, "video", chunks, 100.0)

    drifted = [dict(good[0], coarse_start=31.0), good[1]]
    _write_retimed(tmp_path, "video", drifted)
    with pytest.raises(ValueError, match="coarse times"):
        _load_retimed_chunks_for_video(tmp_path, "video", chunks, 100.0)

    moved = [good[0], dict(good[1], start=41.0)]
    _write_retimed(tmp_path, "video", moved)
    with pytest.raises(ValueError, match="unmatched"):
        _load_retimed_chunks_for_video(tmp_path, "video", chunks, 100.0)

    far = [dict(good[0], start=95.0, end=95.3, align_shift_s=65.0), good[1]]  # past coarse end + pad
    _write_retimed(tmp_path, "video", far)
    with pytest.raises(ValueError, match="search window"):
        _load_retimed_chunks_for_video(tmp_path, "video", chunks, 100.0)

    weak = [dict(good[0], align_match_ratio=0.2), good[1]]
    _write_retimed(tmp_path, "video", weak)
    with pytest.raises(ValueError, match="match floor"):
        _load_retimed_chunks_for_video(tmp_path, "video", chunks, 100.0)

    with pytest.raises(ValueError, match="no cleaned chunks"):
        _load_retimed_chunks_for_video(tmp_path, "video", [], 100.0)


def test_dataset_aggregate_ignores_the_chunkless_video_when_gating_aligned_results() -> None:
    chunks = (_chunk("video_c0", 50.0, 50.5),)
    aligned = _inputs(chunks, RALLIES, retimed_chunks=(_retimed(chunks[0], 5.5, 5.8),), dataset="ShuttleSet22")
    dropped = VideoInputs(
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
    per_video, gaps_by_video, scores_by_video = {}, {}, {}
    for inputs in (aligned, dropped):
        result, gaps, scores, _deltas = evaluate_video(inputs)
        per_video[inputs.video_id] = result
        gaps_by_video[inputs.video_id] = gaps
        scores_by_video[inputs.video_id] = scores

    aggregate = _dataset_aggregate("ShuttleSet22", per_video, gaps_by_video, scores_by_video)

    assert aggregate["retiming"] == {"videos": 1, "status_counts": {"aligned": 1, "unmatched": 0, "collision": 0}}
    assert aggregate["aligned_human_contact_pairing"]["paired_rallies_8s"] == 1
    assert aggregate["aligned_multi_human_contact_pairing"]["paired_rally_rate"] == pytest.approx(1 / 3)
    sweep = aggregate["aligned_lag_sweep_human_contact"]
    assert list(sweep) == sorted(sweep)  # zero-padded keys survive sorted JSON output
    assert "production_predicted_pairing" not in aggregate


def test_multi_rally_aggregate_rates_are_none_without_a_denominator() -> None:
    empty = _multi_rally_aggregate([])
    assert empty["paired_rally_rate"] is None
    assert empty["ambiguous_chunk_rate"] is None
    assert empty["paired_rallies"] == 0
