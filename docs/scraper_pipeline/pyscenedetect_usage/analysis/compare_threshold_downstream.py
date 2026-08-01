"""Compare ContentDetector thresholds through the composition mask."""

from __future__ import annotations

import argparse
import csv
import gzip
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd
from scenedetect import open_video

from annotator.calibration.scoring import _scale_base30_frames, load_gt_rallies
from annotator.composition_mask import (
    CompositionSegment,
    build_composition_mask,
    detect_cuts,
)
from annotator.config import COMPOSITION_KEEP_VOTE
from annotator.fps_constants import probe_fps, scale_for_fps


VIDEO_IDS = (1, 15, 21)
THRESHOLDS = (22, 27)
HOMOGRAPHIES = {
    "static": "static_shuttleset_homography",
    "detected": "detected_ckn_opencv_consensus",
}
VOTE_RELATIVE_PATH = Path("tracknet-stride-8/keep_vote.npy")
STRIDE_1_VOTE_RELATIVE_PATH = Path("tracknet-stride-1/keep_vote.npy")


def parse_args() -> argparse.Namespace:
    """Parse paths while keeping the default invocation rooted at the repository."""
    default_repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root)
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="directory containing sset_XX_288p.mp4 files (default: local_scratch/.../videos_288p)",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="annotator run directory (default: experiments/annotator/runs/20260730-041328)",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="analysis directory containing baseline cuts and receiving outputs",
    )
    return parser.parse_args()


def load_keep_vote(path: Path) -> np.ndarray:
    """Load one one-dimensional boolean court-view vote array."""
    keep_vote = np.load(path)
    if keep_vote.ndim != 1 or keep_vote.dtype != np.bool_:
        raise ValueError(f"{path}: expected a one-dimensional bool array, got {keep_vote.shape} {keep_vote.dtype}")
    return keep_vote


def video_frame_count(video_path: Path) -> int:
    """Read the frame count from the video container through PySceneDetect."""
    video = open_video(str(video_path))
    return int(video.duration.frame_num)


def load_baseline(path: Path) -> tuple[list[int], int]:
    """Read a tracked cut list and its repeated source frame count."""
    with gzip.open(path, "rt", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"{path}: baseline cut list is empty")
    n_frames_values = {int(row["n_frames"]) for row in rows}
    if len(n_frames_values) != 1:
        raise ValueError(f"{path}: expected one n_frames value, got {sorted(n_frames_values)}")
    cut_frames = [int(row["cut_frame"]) for row in rows]
    return cut_frames, n_frames_values.pop()


def first_cut_mismatch(actual: list[int], expected: list[int]) -> tuple[int, int | None, int | None] | None:
    """Return the first differing position and values, or None for exact equality."""
    for index, (actual_frame, expected_frame) in enumerate(zip(actual, expected)):
        if actual_frame != expected_frame:
            return index, actual_frame, expected_frame
    if len(actual) != len(expected):
        index = min(len(actual), len(expected))
        actual_frame = actual[index] if index < len(actual) else None
        expected_frame = expected[index] if index < len(expected) else None
        return index, actual_frame, expected_frame
    return None


def segment_list_after_all_dead_error(
    cut_frames: np.ndarray, keep_vote: np.ndarray, n_frames: int
) -> list[CompositionSegment]:
    """Reconstruct segment spans for reporting after the builder's all-dead error."""
    boundaries = np.unique(np.concatenate([[0], cut_frames.astype(int), [n_frames]]))
    return [
        CompositionSegment(
            int(start),
            int(end),
            float(keep_vote[start:end].mean()),
            True,
        )
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def segment_for_frame(frame: int, segments: list[CompositionSegment]) -> CompositionSegment:
    """Find the half-open composition segment containing one frame."""
    for segment in segments:
        if segment.start <= frame < segment.end:
            return segment
    raise ValueError(f"frame {frame} is outside the composition segments")


def rally_row(rally, segments: list[CompositionSegment]) -> dict[str, object]:
    """Build the downstream dead-segment measurements for one ground-truth rally."""
    first_stroke, last_stroke = rally.extent
    first_segment = segment_for_frame(first_stroke, segments)
    contact_segments = [segment_for_frame(frame, segments) for frame in rally.stroke_frames]
    n_contacts_in_dead = sum(segment.is_dead for segment in contact_segments)
    dead_gaps_inside_extent = sum(
        segment.is_dead
        and first_stroke < segment.start
        and segment.end <= last_stroke
        for segment in segments
    )
    return {
        "set_id": rally.set_id,
        "rally": rally.rally,
        "first_stroke": first_stroke,
        "last_stroke": last_stroke,
        "live_segment_start": first_segment.start,
        "live_segment_end": first_segment.end,
        "first_contact_in_dead": first_segment.is_dead,
        "n_contacts_in_dead": n_contacts_in_dead,
        "live_start_gap": None if first_segment.is_dead else first_stroke - first_segment.start,
        "n_dead_gaps_inside_extent": dead_gaps_inside_extent,
    }


def percentile_or_none(values: list[int], percentile: int) -> float | None:
    """Return a percentile for live starts, with None when no live start exists."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def summary_row(
    video_id: int,
    threshold: int,
    homog: str,
    fps: float,
    cut_frames: np.ndarray,
    segments: list[CompositionSegment],
    rally_rows: list[dict[str, object]],
    composition_error: str | None,
) -> dict[str, object]:
    """Build one summary row using all rallies as the band-fraction denominator."""
    band_5_frames = _scale_base30_frames(5, fps)
    band_10_frames = _scale_base30_frames(10, fps)
    live_start_gaps = [
        int(row["live_start_gap"])
        for row in rally_rows
        if row["live_start_gap"] is not None
    ]
    n_rallies = len(rally_rows)
    dead_frames = sum(segment.end - segment.start for segment in segments if segment.is_dead)
    return {
        "video_id": video_id,
        "threshold": threshold,
        "homog": homog,
        "fps": fps,
        "n_rallies": n_rallies,
        "n_cuts": len(cut_frames),
        "n_segments": len(segments),
        "n_live_segments": sum(not segment.is_dead for segment in segments),
        "dead_frame_fraction": dead_frames / sum(segment.end - segment.start for segment in segments),
        "live_start_gap_n": len(live_start_gaps),
        "live_start_gap_median": percentile_or_none(live_start_gaps, 50),
        "live_start_gap_p75": percentile_or_none(live_start_gaps, 75),
        "live_start_gap_p90": percentile_or_none(live_start_gaps, 90),
        "live_start_gap_band_5_frames": band_5_frames,
        "live_start_gap_band_10_frames": band_10_frames,
        "live_start_gap_frac_within_5_base30": sum(
            gap is not None and gap <= band_5_frames for gap in (row["live_start_gap"] for row in rally_rows)
        ) / n_rallies,
        "live_start_gap_frac_within_10_base30": sum(
            gap is not None and gap <= band_10_frames for gap in (row["live_start_gap"] for row in rally_rows)
        ) / n_rallies,
        "n_rallies_first_contact_dead": sum(row["first_contact_in_dead"] for row in rally_rows),
        "n_rallies_any_contact_dead": sum(row["n_contacts_in_dead"] > 0 for row in rally_rows),
        "n_rallies_split_by_dead": sum(row["n_dead_gaps_inside_extent"] > 0 for row in rally_rows),
        "composition_error": composition_error,
    }


def timecode(frame: int, fps: float) -> str:
    """Format a frame timestamp as HH:MM:SS.mmm at the native frame rate."""
    total_milliseconds = int(round(frame * 1000 / fps))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def extra_cut_rows(video_id: int, fps: float, cuts_22: np.ndarray, cuts_27: np.ndarray) -> list[dict[str, object]]:
    """Return threshold-22 cuts without a threshold-27 neighbour within two frames."""
    rows = []
    cuts_27_list = [int(frame) for frame in cuts_27]
    for cut_frame in (int(frame) for frame in cuts_22):
        index = bisect_left(cuts_27_list, cut_frame)
        has_neighbour = any(
            abs(cuts_27_list[neighbour_index] - cut_frame) <= 2
            for neighbour_index in (index - 1, index)
            if 0 <= neighbour_index < len(cuts_27_list)
        )
        if not has_neighbour:
            rows.append({"video_id": video_id, "cut_frame": cut_frame, "timecode": timecode(cut_frame, fps)})
    return rows


def write_csv_gz(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write a small table as a new gzipped CSV file."""
    with gzip.open(path, "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    videos_dir = args.videos_dir or repo_root / "local_scratch/autograder_architecture/videos_288p"
    run_root = args.run_root or repo_root / "experiments/annotator/runs/20260730-041328"
    analysis_dir = args.analysis_dir or repo_root / "docs/scraper_pipeline/pyscenedetect_usage/analysis"
    data_dir = analysis_dir / "data"

    output_paths = [
        analysis_dir / "data/downstream_summary.csv.gz",
        analysis_dir / "data/extra_cuts_22.csv.gz",
        *(
            data_dir / f"downstream_{video_id}_{threshold}_{homog}.csv.gz"
            for video_id in VIDEO_IDS
            for threshold in THRESHOLDS
            for homog in HOMOGRAPHIES
        ),
    ]
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"refusing to overwrite existing output: {existing_outputs[0]}")

    shots_master_path = repo_root / "training/data/shuttleset/annotations/shots_master.csv"
    shots_master = pd.read_csv(shots_master_path)
    inputs: dict[int, dict[str, object]] = {}

    print("Preflight")
    for video_id in VIDEO_IDS:
        video_path = videos_dir / f"sset_{video_id:02d}_288p.mp4"
        fps = probe_fps(video_path)
        n_frames = video_frame_count(video_path)
        votes: dict[str, np.ndarray] = {}
        print(f"video {video_id}: fps={fps:g} n_frames={n_frames}")
        for homog, config_name in HOMOGRAPHIES.items():
            vote_path = run_root / config_name / f"sset_{video_id:02d}" / VOTE_RELATIVE_PATH
            keep_vote = load_keep_vote(vote_path)
            if len(keep_vote) != n_frames:
                raise ValueError(
                    f"video {video_id} {homog}: keep_vote length {len(keep_vote)} != video frame count {n_frames}"
                )
            votes[homog] = keep_vote
            print(f"  {homog}: keep_vote length={len(keep_vote)} true={int(keep_vote.sum())}")
            if video_id == 1:
                stride_1_path = run_root / config_name / "sset_01" / STRIDE_1_VOTE_RELATIVE_PATH
                stride_independent = stride_1_path.read_bytes() == vote_path.read_bytes()
                print(f"    stride-1_vs_stride-8_byte_equal={stride_independent}")

        baseline_path = analysis_dir / f"data/cuts_{video_id}_content27.csv.gz"
        baseline_cuts, baseline_n_frames = load_baseline(baseline_path)
        if baseline_n_frames != n_frames:
            raise ValueError(
                f"video {video_id}: baseline n_frames {baseline_n_frames} != video frame count {n_frames}"
            )
        inputs[video_id] = {
            "video_path": video_path,
            "fps": fps,
            "n_frames": n_frames,
            "votes": votes,
            "baseline_cuts": baseline_cuts,
        }

    print("Threshold-27 gate")
    cut_lists: dict[int, dict[int, np.ndarray]] = {22: {}, 27: {}}
    for video_id in VIDEO_IDS:
        video_input = inputs[video_id]
        min_scene_len = scale_for_fps(video_input["fps"]).composition_min_scene_len
        detected_cuts = detect_cuts(
            video_input["video_path"],
            video_input["n_frames"],
            27,
            min_scene_len,
        )
        actual = [int(frame) for frame in detected_cuts]
        mismatch = first_cut_mismatch(actual, video_input["baseline_cuts"])
        if mismatch is not None:
            index, actual_frame, expected_frame = mismatch
            raise RuntimeError(
                f"threshold-27 gate failed for video {video_id}: "
                f"actual_count={len(actual)} expected_count={len(video_input['baseline_cuts'])} "
                f"first_mismatch_index={index} actual={actual_frame} expected={expected_frame}"
            )
        cut_lists[27][video_id] = detected_cuts
        print(f"video {video_id}: PASS ({len(actual)} cuts)")

    print("Threshold-27 gate passed for all videos; detecting threshold 22")
    for video_id in VIDEO_IDS:
        video_input = inputs[video_id]
        min_scene_len = scale_for_fps(video_input["fps"]).composition_min_scene_len
        cut_lists[22][video_id] = detect_cuts(
            video_input["video_path"],
            video_input["n_frames"],
            22,
            min_scene_len,
        )
        print(f"video {video_id}: threshold-22 cuts={len(cut_lists[22][video_id])}")

    summary_rows: list[dict[str, object]] = []
    all_extra_cut_rows: list[dict[str, object]] = []
    rally_fieldnames = [
        "set_id",
        "rally",
        "first_stroke",
        "last_stroke",
        "live_segment_start",
        "live_segment_end",
        "first_contact_in_dead",
        "n_contacts_in_dead",
        "live_start_gap",
        "n_dead_gaps_inside_extent",
    ]
    summary_fieldnames = [
        "video_id",
        "threshold",
        "homog",
        "fps",
        "n_rallies",
        "n_cuts",
        "n_segments",
        "n_live_segments",
        "dead_frame_fraction",
        "live_start_gap_n",
        "live_start_gap_median",
        "live_start_gap_p75",
        "live_start_gap_p90",
        "live_start_gap_band_5_frames",
        "live_start_gap_band_10_frames",
        "live_start_gap_frac_within_5_base30",
        "live_start_gap_frac_within_10_base30",
        "n_rallies_first_contact_dead",
        "n_rallies_any_contact_dead",
        "n_rallies_split_by_dead",
        "composition_error",
    ]

    for video_id in VIDEO_IDS:
        video_input = inputs[video_id]
        gt_rallies = load_gt_rallies(shots_master, video_id)
        for threshold in THRESHOLDS:
            cut_frames = cut_lists[threshold][video_id]
            if threshold == 22:
                all_extra_cut_rows.extend(
                    extra_cut_rows(video_id, video_input["fps"], cut_frames, cut_lists[27][video_id])
                )
            for homog in HOMOGRAPHIES:
                keep_vote = video_input["votes"][homog]
                composition_error = None
                try:
                    _dead_mask, segments = build_composition_mask(
                        cut_frames,
                        keep_vote,
                        video_input["n_frames"],
                        COMPOSITION_KEEP_VOTE,
                    )
                except ValueError as error:
                    if "all dead" not in str(error):
                        raise
                    composition_error = str(error)
                    print(f"video {video_id} threshold {threshold} {homog}: {composition_error}")
                    segments = segment_list_after_all_dead_error(
                        cut_frames,
                        keep_vote,
                        video_input["n_frames"],
                    )

                rows = [rally_row(rally, segments) for rally in gt_rallies]
                output_path = data_dir / f"downstream_{video_id}_{threshold}_{homog}.csv.gz"
                write_csv_gz(output_path, rows, rally_fieldnames)
                summary_rows.append(
                    summary_row(
                        video_id,
                        threshold,
                        homog,
                        video_input["fps"],
                        cut_frames,
                        segments,
                        rows,
                        composition_error,
                    )
                )

    write_csv_gz(data_dir / "downstream_summary.csv.gz", summary_rows, summary_fieldnames)
    write_csv_gz(
        data_dir / "extra_cuts_22.csv.gz",
        all_extra_cut_rows,
        ["video_id", "cut_frame", "timecode"],
    )
    print(f"Wrote {len(summary_rows)} summary rows and {len(all_extra_cut_rows)} extra threshold-22 cuts")


if __name__ == "__main__":
    main()
