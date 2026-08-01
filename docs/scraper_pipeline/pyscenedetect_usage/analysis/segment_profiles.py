"""Build segment profiles and serve lead-in tables from baseline cut lists.

The --stats input is a PySceneDetect StatsManager CSV (per-frame content_val and
delta_* columns). The tracked ones were too big for the repo; regenerate with:

    from scenedetect import ContentDetector, SceneManager, StatsManager, open_video
    stats = StatsManager()
    manager = SceneManager(stats)
    manager.add_detector(ContentDetector(threshold=27, min_scene_len=15))
    manager.detect_scenes(open_video("<288p video>"))
    stats.save_to_csv("content_stats_<vid>.csv")
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.calibration.scoring import GtRally, load_gt_rallies


PROFILE_STATS = ("content_val", "delta_lum", "delta_edges")
def read_cuts(cuts_path: Path) -> tuple[int, list[int]]:
    """Read and validate the baseline cut starts."""
    cuts = pd.read_csv(cuts_path)
    if cuts.empty:
        raise ValueError(f"cut list is empty: {cuts_path}")
    frame_counts = cuts["n_frames"].dropna().unique()
    if len(frame_counts) != 1:
        raise ValueError(f"cut list has inconsistent n_frames values: {cuts_path}")

    n_frames = int(frame_counts[0])
    cut_frames = [int(frame) for frame in cuts["cut_frame"]]
    if cut_frames != sorted(set(cut_frames)):
        raise ValueError(f"cut frames are not strictly increasing: {cuts_path}")
    if not all(0 < frame < n_frames for frame in cut_frames):
        raise ValueError(f"cut frame is outside (0, n_frames): {cuts_path}")
    return n_frames, cut_frames


def read_stats(stats_path: Path, n_frames: int) -> pd.DataFrame:
    """Read per-frame stats and map one-based CSV frames to zero-based frames."""
    stats = pd.read_csv(stats_path)
    expected_columns = {
        "Frame Number",
        "Timecode",
        "content_val",
        "delta_edges",
        "delta_hue",
        "delta_lum",
        "delta_sat",
    }
    if set(stats.columns) != expected_columns:
        raise ValueError(f"unexpected stats columns in {stats_path}: {list(stats.columns)}")
    if len(stats) != n_frames - 1:
        raise ValueError(f"{stats_path}: {len(stats)} rows, expected {n_frames - 1}")

    frame_numbers = stats["Frame Number"].to_numpy(dtype=int)
    expected_frame_numbers = np.arange(2, n_frames + 1)
    if not np.array_equal(frame_numbers, expected_frame_numbers):
        raise ValueError(f"{stats_path}: expected Frame Number values 2..{n_frames}")

    stats = stats.assign(frame=frame_numbers - 1).set_index("frame")
    return stats


def build_segments(n_frames: int, cut_frames: list[int]) -> pd.DataFrame:
    """Build half-open timeline segments from cut frames used as segment starts."""
    starts = [0, *cut_frames]
    ends = [*cut_frames, n_frames]
    segments = pd.DataFrame(
        {
            "segment_index": np.arange(len(starts), dtype=int),
            "start": starts,
            "end": ends,
        }
    )
    segments["duration"] = segments["end"] - segments["start"]
    if int(segments.iloc[0]["start"]) != 0 or int(segments.iloc[-1]["end"]) != n_frames:
        raise AssertionError("segments do not cover the timeline endpoints")
    if not np.array_equal(segments["end"].to_numpy()[:-1], segments["start"].to_numpy()[1:]):
        raise AssertionError("segments have a gap or overlap")
    return segments


def segment_index_for_frame(starts: list[int], frame: int) -> int:
    """Return the unique half-open segment containing a zero-based frame."""
    index = bisect_right(starts, frame) - 1
    if index < 0 or index >= len(starts):
        raise ValueError(f"frame {frame} is outside the segment timeline")
    return index


def classify_segment(
    start: int,
    end: int,
    gt_rallies: list[GtRally],
) -> str:
    """Apply the requested segment category precedence."""
    contains_rally_start = any(start <= rally.extent[0] < end for rally in gt_rallies)
    if contains_rally_start:
        return "rally_start"

    inside_rallies = [
        rally for rally in gt_rallies if start > rally.extent[0] and end <= rally.extent[1]
    ]
    if inside_rallies:
        contains_stroke = any(
            start <= stroke < end
            for rally in inside_rallies
            for stroke in rally.stroke_frames
        )
        return "inside_with_strokes" if contains_stroke else "inside_no_strokes"

    overlaps_rally = any(start <= rally.extent[1] and end > rally.extent[0] for rally in gt_rallies)
    return "other" if overlaps_rally else "outside"


def add_segment_profiles(
    segments: pd.DataFrame,
    stats: pd.DataFrame,
    gt_rallies: list[GtRally],
) -> pd.DataFrame:
    """Add categories and mean/median stats to the segment table."""
    rows: list[dict[str, int | float | str | None]] = []
    for segment in segments.itertuples(index=False):
        start = int(segment.start)
        end = int(segment.end)
        segment_stats = stats.reindex(range(start, end))
        row: dict[str, int | float | str | None] = {
            "segment_index": int(segment.segment_index),
            "start": start,
            "end": end,
            "duration": int(segment.duration),
            "category": classify_segment(start, end, gt_rallies),
        }
        for stat_name in PROFILE_STATS:
            row[f"{stat_name}_mean"] = float(segment_stats[stat_name].mean())
            row[f"{stat_name}_median"] = float(segment_stats[stat_name].median())
        rows.append(row)
    return pd.DataFrame(rows)


def build_leadin_table(
    profiles: pd.DataFrame,
    cut_frames: list[int],
    gt_rallies: list[GtRally],
) -> pd.DataFrame:
    """Build one serve lead-in row for each ground-truth rally."""
    starts = [0, *cut_frames]
    rows: list[dict[str, int | float | str | None]] = []
    for rally in gt_rallies:
        first_stroke = rally.extent[0]
        segment_index = segment_index_for_frame(starts, first_stroke)
        segment = profiles.iloc[segment_index]
        if not int(segment["start"]) <= first_stroke < int(segment["end"]):
            raise AssertionError(f"rally {rally.set_id}/{rally.rally} has no containing segment")

        preceding_index = segment_index - 1
        if preceding_index < 0:
            preceding_duration = None
            preceding_category = None
            preceding_mean_content = None
        else:
            preceding = profiles.iloc[preceding_index]
            preceding_duration = int(preceding["duration"])
            preceding_category = str(preceding["category"])
            preceding_mean_content = float(preceding["content_val_mean"])

        rows.append(
            {
                "set_id": rally.set_id,
                "rally": rally.rally,
                "first_stroke": first_stroke,
                "segment_start": int(segment["start"]),
                "segment_end": int(segment["end"]),
                "start_gap": first_stroke - int(segment["start"]),
                "preceding_segment_duration": preceding_duration,
                "preceding_segment_category": preceding_category,
                "preceding_segment_mean_content_val": preceding_mean_content,
            }
        )
    return pd.DataFrame(rows)


def run(video_id: int, cuts_path: Path, stats_path: Path, out_dir: Path) -> None:
    """Generate the requested profile and lead-in CSVs for one video."""
    n_frames, cut_frames = read_cuts(cuts_path)
    stats = read_stats(stats_path, n_frames)
    shots_master_path = Path(__file__).resolve().parents[4] / "training/data/shuttleset/annotations/shots_master.csv"
    gt_rallies = load_gt_rallies(pd.read_csv(shots_master_path), video_id)
    if not all(0 <= stroke < n_frames for rally in gt_rallies for stroke in rally.stroke_frames):
        raise ValueError(f"GT stroke is outside the video timeline for video {video_id}")

    segments = build_segments(n_frames, cut_frames)
    profiles = add_segment_profiles(segments, stats, gt_rallies)
    # .gz suffix makes pandas write gzip-compressed CSV; read_csv above accepts .csv.gz inputs the same way
    profiles.to_csv(out_dir / f"segment_profiles_{video_id}.csv.gz", index=False)

    leadin = build_leadin_table(profiles, cut_frames, gt_rallies)
    leadin.to_csv(out_dir / f"serve_leadin_{video_id}.csv.gz", index=False)

    print(
        f"video={video_id} n_frames={n_frames} cuts={len(cut_frames)} "
        f"segments={len(profiles)} rallies={len(gt_rallies)} stats_rows={len(stats)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--cuts", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.video_id, args.cuts, args.stats, args.out_dir)


if __name__ == "__main__":
    main()
