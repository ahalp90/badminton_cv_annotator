"""Benchmark prepared commentary against ShuttleSet rally populations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from annotator.calibration.commentary_benchmark_inputs import (
    COMMENTARY_CODE_COMMIT,
    COMMENTARY_INVENTORY_SHA256,
    COMMENTARY_MANIFEST_SHA256,
    COMMENTARY_MODEL,
    COMMENTARY_PROVIDER,
    COMMENTARY_REMOVED_OVERLAP_ROWS,
    COMMENTARY_SOURCE_MANIFEST_SHA256,
    COMMENTARY_STATUS_SHA256,
    SHUTTLESET_SHOTS_MASTER_SHA256,
    VideoInputs,
    _canonical_inventory,
    _issue103_inputs,
    _load_video_inputs,
    _manifest_index,
    _mapping,
    _sequence,
)
from annotator.calibration.shuttleset22_features import ANNOTATION_TREE_SHA256, _tree_digest
from annotator.fps_constants import scale_for_fps
from annotator.replay_mask import filter_short_exclusion_runs
from dataset_builder.vision import save_json_gz
from scraper.commentary_pairing import (
    PAIR_WINDOW_S,
    _believed_replay_in_rally_interior,
    pair_video,
)


RESULT_SCHEMA = "issue104-commentary-benchmark/3"
EVALUATOR_BASE_COMMIT = "002238dc62ac0390c2e2b4005780cf3d81420255"
FIVE_SECOND_WINDOW = 5.0
PAIRING_POLICY: dict[str, object] = {
    "pairing": "first unclaimed cleaned chunk starting strictly after the rally",
    "pair_window_seconds": PAIR_WINDOW_S,
    "sensitivity_window_seconds": FIVE_SECOND_WINDOW,
    "in_rally_commentary": (
        "not intentionally supported; measured separately because the post-rally join "
        "can claim a chunk after a preceding rally even when a later rally has begun"
    ),
    "replay": "duration-filtered issue #103 replay masks; no ShuttleSet22 replay masks",
    "timestamp_precision": "coarse transcript timestamps; fine WhisperX alignment not run",
    "semantic_outputs": "not implemented by the supported cleaning or pairing contracts",
    "rally_views": (
        "ShuttleSet production predictions are reported separately; the corpus comparison "
        "uses human contact intervals for both datasets"
    ),
}


def _distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("distribution contains non-finite values")
    return {
        "count": len(values),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def _masked_rally_ids(
    inputs: VideoInputs,
    rallies: Sequence[tuple[int, int, int]],
) -> set[int]:
    if inputs.replay_mask is None:
        return set()
    minimum_run = scale_for_fps(inputs.fps).replay_mask_min_frames
    filtered = filter_short_exclusion_runs(inputs.replay_mask, minimum_run)
    return {
        rally_id
        for rally_id, start_frame, end_frame in rallies
        if _believed_replay_in_rally_interior(
            filtered, start_frame, end_frame, minimum_run
        )
    }


def _in_rally_chunk_ids(
    inputs: VideoInputs,
    rallies: Sequence[tuple[int, int, int]],
) -> set[str]:
    spans = sorted((start / inputs.fps, end / inputs.fps) for _, start, end in rallies)
    result = set()
    for chunk in inputs.chunks:
        start = float(chunk["start"])
        if any(rally_start <= start < rally_end for rally_start, rally_end in spans):
            result.add(str(chunk["chunk_id"]))
    return result


def _pairing_summary(
    inputs: VideoInputs,
    rallies: Sequence[tuple[int, int, int]],
) -> tuple[dict[str, object], list[float]]:
    chunks = [dict(chunk) for chunk in inputs.chunks]
    rows = pair_video(
        inputs.video_id,
        list(rallies),
        chunks,
        inputs.replay_mask,
        inputs.fps,
        pair_window_s=PAIR_WINDOW_S,
    )
    five_second_rows = pair_video(
        inputs.video_id,
        list(rallies),
        chunks,
        inputs.replay_mask,
        inputs.fps,
        pair_window_s=FIVE_SECOND_WINDOW,
    )
    paired_rows = [row for row in rows if row["chunk_id"]]
    five_second_paired_rows = [row for row in five_second_rows if row["chunk_id"]]
    paired_ids = {str(row["chunk_id"]) for row in paired_rows}
    if len(paired_ids) != len(paired_rows):
        raise ValueError(f"{inputs.video_id} paired one chunk more than once")
    gaps = [
        float(row["commentary_start"]) - float(row["rally_end"]) / inputs.fps
        for row in paired_rows
    ]
    if any(gap <= 0 or gap > PAIR_WINDOW_S + 1e-9 for gap in gaps):
        raise ValueError(f"{inputs.video_id} has a pair outside the supported window")
    five_second_gaps = [
        float(row["commentary_start"]) - float(row["rally_end"]) / inputs.fps
        for row in five_second_paired_rows
    ]
    if any(gap <= 0 or gap > FIVE_SECOND_WINDOW + 1e-9 for gap in five_second_gaps):
        raise ValueError(f"{inputs.video_id} has a pair outside the sensitivity window")

    masked_ids = _masked_rally_ids(inputs, rallies)
    eligible_rallies = len(rallies) - len(masked_ids)
    in_rally_ids = _in_rally_chunk_ids(inputs, rallies)
    spans = {
        rally_id: (start_frame / inputs.fps, end_frame / inputs.fps)
        for rally_id, start_frame, end_frame in rallies
    }
    cross_rally_pairs = []
    for row in paired_rows:
        chunk_start = float(row["commentary_start"])
        paired_rally_id = int(row["rally_id"])
        for containing_rally_id, (start, end) in spans.items():
            if containing_rally_id != paired_rally_id and start <= chunk_start < end:
                cross_rally_pairs.append(
                    {
                        "chunk_id": str(row["chunk_id"]),
                        "chunk_start_seconds": chunk_start,
                        "paired_rally_id": paired_rally_id,
                        "paired_rally_end_seconds": float(row["rally_end"])
                        / inputs.fps,
                        "containing_rally_id": containing_rally_id,
                    }
                )

    return {
        "rallies": len(rallies),
        "replay_masked_rallies": len(masked_ids),
        "pairing_eligible_rallies": eligible_rallies,
        "paired_rallies_8s": len(paired_rows),
        "paired_rallies_5s": len(five_second_paired_rows),
        "paired_chunks": len(paired_ids),
        "unpaired_cleaned_chunks": len(inputs.chunks) - len(paired_ids),
        "in_rally_chunk_starts": len(in_rally_ids),
        "paired_chunks_starting_in_another_rally": len(
            {str(row["chunk_id"]) for row in cross_rally_pairs}
        ),
        "cross_rally_pairs": cross_rally_pairs,
        "post_rally_gap_seconds": _distribution(gaps),
    }, gaps


def evaluate_video(
    inputs: VideoInputs,
) -> tuple[dict[str, object], dict[str, list[float]], list[float], list[int]]:
    """Evaluate one video's supported pairing and cleaning coverage."""
    primary_pairing, primary_gaps = _pairing_summary(inputs, inputs.rallies)
    human_pairing, human_gaps = _pairing_summary(inputs, inputs.human_rallies)
    scores = [float(chunk["bert_f1"]) for chunk in inputs.chunks]
    raw_word_counts = [len(str(chunk["text"]).split()) for chunk in inputs.chunks]
    clean_word_counts = [
        len(str(chunk["text_clean"]).split()) for chunk in inputs.chunks
    ]
    semantic_fields = ("sentiment", "concept", "player", "player_link", "court_slot")
    semantic_counts = {
        field: sum(
            field in chunk and chunk[field] not in (None, "") for chunk in inputs.chunks
        )
        for field in semantic_fields
    }
    if any(not math.isfinite(score) or score < 0.8 for score in scores):
        raise ValueError(f"{inputs.video_id} has a failed cleaned chunk")
    if any(chunk["clean_pass"] is not True for chunk in inputs.chunks):
        raise ValueError(f"{inputs.video_id} has a non-passing cleaned chunk")
    if any(not str(chunk["text_clean"]).strip() for chunk in inputs.chunks):
        raise ValueError(f"{inputs.video_id} has empty cleaned text")
    if any(
        len(alternates) != 3
        or any(not str(alternate).strip() for alternate in alternates)
        for alternates in (
            _sequence(chunk["alt_phrasings"], f"{inputs.video_id} alternate phrasings")
            for chunk in inputs.chunks
        )
    ):
        raise ValueError(
            f"{inputs.video_id} has an invalid alternate-phrasing population"
        )

    result: dict[str, object] = {
        "dataset": inputs.dataset,
        "video_id": inputs.video_id,
        "fps": inputs.fps,
        "frame_count": inputs.frame_count,
        "transcript_source": inputs.transcript_source,
        "transcript_segments": inputs.transcript_segments,
        "cleaned_chunks": len(inputs.chunks),
        "raw_words": sum(raw_word_counts),
        "clean_words": sum(clean_word_counts),
        "bert_f1": _distribution(scores),
        "semantic_output_counts": semantic_counts,
        "primary_rally_source": (
            "production_predicted"
            if inputs.dataset == "ShuttleSet"
            else "human_contact"
        ),
        "primary_pairing": primary_pairing,
        "human_contact_pairing": human_pairing,
        "annotation_population": inputs.annotation_population,
    }
    return (
        result,
        {"primary": primary_gaps, "human_contact": human_gaps},
        scores,
        [clean - raw for clean, raw in zip(clean_word_counts, raw_word_counts)],
    )


def _cleaning_aggregate(
    rows: Sequence[Mapping[str, object]],
    scores: Sequence[float],
) -> dict[str, object]:
    sum_fields = (
        "transcript_segments",
        "cleaned_chunks",
        "raw_words",
        "clean_words",
    )
    return {
        "videos": len(rows),
        **{field: sum(int(row[field]) for row in rows) for field in sum_fields},
        "bert_f1": _distribution(scores),
        "transcript_sources": {
            source: sum(row["transcript_source"] == source for row in rows)
            for source in ("youtube_asr", "whisper")
        },
    }


def _pairing_aggregate(
    rows: Sequence[Mapping[str, object]],
    gaps: Sequence[float],
    field: str,
) -> dict[str, object]:
    sum_fields = (
        "rallies",
        "replay_masked_rallies",
        "pairing_eligible_rallies",
        "paired_rallies_8s",
        "paired_rallies_5s",
        "paired_chunks",
        "unpaired_cleaned_chunks",
        "in_rally_chunk_starts",
        "paired_chunks_starting_in_another_rally",
    )
    aggregate: dict[str, object] = {
        **{
            sum_field: sum(
                int(_mapping(row[field], f"{field} summary")[sum_field]) for row in rows
            )
            for sum_field in sum_fields
        },
        "post_rally_gap_seconds": _distribution(gaps),
    }
    eligible = int(aggregate["pairing_eligible_rallies"])
    aggregate["paired_rally_rate_8s"] = (
        None if eligible == 0 else int(aggregate["paired_rallies_8s"]) / eligible
    )
    aggregate["paired_rally_rate_5s"] = (
        None if eligible == 0 else int(aggregate["paired_rallies_5s"]) / eligible
    )
    return aggregate


def _leave_one_video_out(
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, float | None]:
    rates = []
    for excluded in rows:
        eligible = sum(
            int(_mapping(row[field], f"{field} summary")["pairing_eligible_rallies"])
            for row in rows
            if row["video_id"] != excluded["video_id"]
        )
        paired = sum(
            int(_mapping(row[field], f"{field} summary")["paired_rallies_8s"])
            for row in rows
            if row["video_id"] != excluded["video_id"]
        )
        if eligible:
            rates.append(paired / eligible)
    return {
        "paired_rally_rate_8s_min": None if not rates else min(rates),
        "paired_rally_rate_8s_max": None if not rates else max(rates),
    }


def _validate_corpus_inputs(
    commentary_root: Path,
    issue103_rally_records: Path,
    issue103_artifacts: Path,
    shuttleset22_root: Path,
) -> tuple[
    dict[str, Mapping[str, object]],
    list[Mapping[str, Any]],
    dict[str, list[tuple[int, int, int]]],
    dict[str, np.ndarray],
    dict[str, object],
    str,
]:
    """Validate the commentary, issue #103, and ShuttleSet22 input identities."""
    manifest_index = _manifest_index(commentary_root)
    inventory_records = _canonical_inventory(commentary_root)
    issue103_rallies, issue103_masks, issue103_provenance = _issue103_inputs(
        issue103_rally_records,
        issue103_artifacts,
    )
    annotation_digest = _tree_digest(shuttleset22_root / "annotations")
    if annotation_digest != ANNOTATION_TREE_SHA256:
        raise ValueError(
            "ShuttleSet22 annotation tree SHA-256 differs: "
            f"expected {ANNOTATION_TREE_SHA256}, found {annotation_digest}"
        )
    return (
        manifest_index,
        inventory_records,
        issue103_rallies,
        issue103_masks,
        issue103_provenance,
        annotation_digest,
    )


def _evaluate_videos(
    videos: Sequence[VideoInputs],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, list[float]]],
    dict[str, list[float]],
    list[int],
]:
    """Run evaluate_video across the corpus and collect its per-video outputs."""
    per_video: dict[str, dict[str, object]] = {}
    gaps_by_video: dict[str, dict[str, list[float]]] = {}
    scores_by_video: dict[str, list[float]] = {}
    word_deltas: list[int] = []
    for inputs in videos:
        result, gaps, scores, deltas = evaluate_video(inputs)
        per_video[inputs.video_id] = result
        gaps_by_video[inputs.video_id] = gaps
        scores_by_video[inputs.video_id] = scores
        word_deltas.extend(deltas)
    return per_video, gaps_by_video, scores_by_video, word_deltas


def _dataset_aggregate(
    dataset: str,
    per_video: Mapping[str, Mapping[str, object]],
    gaps_by_video: Mapping[str, Mapping[str, list[float]]],
    scores_by_video: Mapping[str, list[float]],
) -> dict[str, object]:
    """Aggregate one dataset's cleaning and pairing summaries across its videos."""
    rows = [row for row in per_video.values() if row["dataset"] == dataset]
    scores = [score for row in rows for score in scores_by_video[str(row["video_id"])]]
    primary_gaps = [
        gap for row in rows for gap in gaps_by_video[str(row["video_id"])]["primary"]
    ]
    human_gaps = [
        gap
        for row in rows
        for gap in gaps_by_video[str(row["video_id"])]["human_contact"]
    ]
    dataset_result: dict[str, object] = {
        "cleaning": _cleaning_aggregate(rows, scores),
        "human_contact_pairing": _pairing_aggregate(
            rows, human_gaps, "human_contact_pairing"
        ),
        "human_contact_leave_one_video_out": _leave_one_video_out(
            rows, "human_contact_pairing"
        ),
    }
    if dataset == "ShuttleSet":
        dataset_result["production_predicted_pairing"] = _pairing_aggregate(
            rows, primary_gaps, "primary_pairing"
        )
        dataset_result["production_predicted_leave_one_video_out"] = (
            _leave_one_video_out(rows, "primary_pairing")
        )
    return dataset_result


def _population_summary(videos: Sequence[VideoInputs]) -> dict[str, object]:
    """Summarize the canonical video population by dataset and cleaning status."""
    return {
        "canonical_videos": len(videos),
        "shuttleset_videos": sum(video.dataset == "ShuttleSet" for video in videos),
        "shuttleset22_videos": sum(video.dataset == "ShuttleSet22" for video in videos),
        "cleaned_videos": sum(bool(video.chunks) for video in videos),
        "triage_dropped_videos": [
            video.video_id for video in videos if not video.chunks
        ],
    }


def evaluate_corpus(
    commentary_root: Path,
    issue103_rally_records: Path,
    issue103_artifacts: Path,
    shuttleset_ground_truth_root: Path,
    shuttleset22_root: Path,
) -> dict[str, object]:
    """Validate all inputs and evaluate the complete commentary population."""
    (
        manifest_index,
        inventory_records,
        issue103_rallies,
        issue103_masks,
        issue103_provenance,
        annotation_digest,
    ) = _validate_corpus_inputs(
        commentary_root, issue103_rally_records, issue103_artifacts, shuttleset22_root
    )
    videos = _load_video_inputs(
        commentary_root,
        shuttleset_ground_truth_root,
        shuttleset22_root,
        inventory_records,
        manifest_index,
        issue103_rallies,
        issue103_masks,
    )
    per_video, gaps_by_video, scores_by_video, word_deltas = _evaluate_videos(videos)

    by_dataset = {
        dataset: _dataset_aggregate(dataset, per_video, gaps_by_video, scores_by_video)
        for dataset in ("ShuttleSet", "ShuttleSet22")
    }
    all_rows = list(per_video.values())
    all_human_gaps = [
        gap for gaps in gaps_by_video.values() for gap in gaps["human_contact"]
    ]
    all_scores = [score for scores in scores_by_video.values() for score in scores]
    semantic_totals = {
        field: sum(
            int(_mapping(row["semantic_output_counts"], "semantic counts")[field])
            for row in all_rows
        )
        for field in ("sentiment", "concept", "player", "player_link", "court_slot")
    }
    return {
        "schema": RESULT_SCHEMA,
        "provenance": {
            "evaluator_base_commit": EVALUATOR_BASE_COMMIT,
            "commentary_code_commit": COMMENTARY_CODE_COMMIT,
            "commentary_provider": COMMENTARY_PROVIDER,
            "commentary_model": COMMENTARY_MODEL,
            "commentary_manifest_sha256": COMMENTARY_MANIFEST_SHA256,
            "commentary_source_manifest_sha256": COMMENTARY_SOURCE_MANIFEST_SHA256,
            "commentary_removed_overlap_rows": COMMENTARY_REMOVED_OVERLAP_ROWS,
            "commentary_status_sha256": COMMENTARY_STATUS_SHA256,
            "commentary_inventory_sha256": COMMENTARY_INVENTORY_SHA256,
            "shuttleset_shots_master_sha256": SHUTTLESET_SHOTS_MASTER_SHA256,
            "shuttleset22_annotation_tree_sha256": annotation_digest,
            **issue103_provenance,
        },
        "policy": PAIRING_POLICY,
        "population": _population_summary(videos),
        "aggregate": {
            **_cleaning_aggregate(all_rows, all_scores),
            "word_count_delta": _distribution([float(value) for value in word_deltas]),
            "semantic_output_counts": semantic_totals,
            "human_contact_pairing": _pairing_aggregate(
                all_rows, all_human_gaps, "human_contact_pairing"
            ),
        },
        "by_dataset": by_dataset,
        "per_video": per_video,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commentary-root", type=Path, required=True)
    parser.add_argument("--issue103-rally-records", type=Path, required=True)
    parser.add_argument("--issue103-artifacts", type=Path, required=True)
    parser.add_argument("--shuttleset-ground-truth-root", type=Path, required=True)
    parser.add_argument("--shuttleset22-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = evaluate_corpus(
        args.commentary_root,
        args.issue103_rally_records,
        args.issue103_artifacts,
        args.shuttleset_ground_truth_root,
        args.shuttleset22_root,
    )
    save_json_gz(args.output, result)
    print(json.dumps(result["population"], sort_keys=True))
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
