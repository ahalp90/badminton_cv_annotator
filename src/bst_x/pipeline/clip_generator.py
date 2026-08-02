"""Automated clip generation from raw ShuttleSet match videos.

Replaces the 6 manual runs of gen_my_dataset.py with a single script that:
  1. Generates clips for all splits, both players, all stroke types
  2. Filters out individually removed shots (from flaw_shot_records.csv)
  3. Applies class merging per the active taxonomy with English folder names

Usage:
    python -m pipeline.clip_generator
"""
import argparse
import shutil

from moviepy import VideoFileClip
import pandas as pd
from pathlib import Path

from pipeline.config import (
    SET_INFO_DIR, RAW_VIDEO_DIR, CLIPS_OUTPUT_DIR,
    SPLITS,
    REMOVED_SHOTS, CLIP_WINDOW, PLAYERS,
    NOSIDE_FOLDERS,
)

from classifier_shared.dataset import compute_clip_bounds, compute_temporal_bounds
from classifier_shared.player_mapping import collect_shots
from classifier_shared.taxonomy import (
    STROKE_TYPES_19,
    STROKE_TYPES_19_ZH,
    Taxonomy,
    taxonomy_lookup,
)
from pipeline.video_metadata import find_video_files

# Default taxonomy when callers don't pass one. Matches the project's
# working baseline; override via the function arg for one-off runs.
_DEFAULT_TAXONOMY = taxonomy_lookup('une_v1_14')

# Single source for the runtime guard and the CLI --clip-window choices.
VALID_CLIP_WINDOWS = frozenset(
    {'middle_in_a_sec', 'between_2_hits', 'between_2_hits_with_max_limits'}
)


def _frame_to_time(frame_number: int, fps: float) -> str:
    """Convert a frame number to HH:MM:SS.ssssss time string for MoviePy.

    Adapted from ShuttleSet/utils.py.

    :param frame_number: Frame index in the video.
    :param fps: Video frames per second.
    :return: Time string formatted as HH:MM:SS.ssssss.
    """
    # Half-frame offset added before the split so it flows through the
    # hours/minutes/seconds carry (same total as adding it after, but the string
    # stays canonical near minute boundaries). Rounding to the printed 6 dp
    # happens before the split too: a float a hair under a minute boundary
    # would otherwise format as seconds=60.000000 instead of carrying.
    #
    # TODO(unverified): the same +0.5 offset in bric/preprocessing/slice_rallies.py
    # is off by one. Measured 2026-07-22 on ffmpeg 6.1.1: with raw `-ss`, seeking to
    # (n + 0.5)/fps selects source frame n+1, while n/fps selects n. The offset was
    # documented as a guard against landing on the PREVIOUS frame, so it overshoots
    # rather than protects. This path is not that path: the string below goes to
    # MoviePy's subclipped(), which fetches frames its own way, and nobody has
    # measured whether the same slip happens here. If it does, every generated
    # training clip starts one frame late.
    # To settle it: generate one clip, then compare its first frame byte-for-byte
    # against the source frame at start_f, decoded from zero with
    # `select='between(n,start_f,start_f)'`. Equal means this is fine.
    # Background: local_scratch/autograder_architecture/briefs/visual_verifier_build_brief.md
    total_seconds = round((frame_number + 0.5) / fps, 6)
    hours = int(total_seconds // 3600)
    minutes = int(total_seconds % 3600 // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:09.6f}"


def _write_clips_for_video(
    raw_video_dir: Path,
    out_folder: Path,
    video_id: int,
    shots_df: pd.DataFrame,
    clip_window: str,
    stroke_types: list[str],
    players: tuple[str, ...],
) -> int:
    """Write clip .mp4 files for one source video. Returns count of clips written.

    :param raw_video_dir: Directory containing raw match videos.
    :param out_folder: Split-level output directory (e.g. clips/train/).
    :param video_id: Numeric ID of the video (from match.csv).
    :param shots_df: DataFrame with columns: player, type, set, rally, ball_round,
        frame_num, start_f, end_f. Each row is one shot to clip.
    :param clip_window: Temporal clipping window name.
    :param stroke_types: List of English stroke type names (for folder creation).
    :param players: Tuple of player names ('Top', 'Bottom').
    :return: Number of clips written.
    """
    if clip_window not in VALID_CLIP_WINDOWS:
        raise ValueError(f"Unknown clip window: {clip_window!r}")

    for typ in stroke_types:
        if typ in NOSIDE_FOLDERS:
            (out_folder / typ).mkdir(parents=True, exist_ok=True)
        else:
            for player in players:
                (out_folder / f'{player}_{typ}').mkdir(parents=True, exist_ok=True)

    # New downloads use ``{id}.ext``; existing ShuttleSet files may retain
    # ``{id} {name}.ext``. The shared index rejects a directory containing both.
    video_path = find_video_files(raw_video_dir).get(video_id)
    if video_path is None:
        print(f"Warning: Raw video for ID {video_id} not found. Skipping.")
        return 0
    video = VideoFileClip(str(video_path))
    fps = video.fps
    clips_written = 0

    try:
        for _, row in shots_df.iterrows():
            typ = row['type']
            folder_name = typ if typ in NOSIDE_FOLDERS else f'{row["player"]}_{typ}'
            out_path = (out_folder
                        / folder_name
                        / f'{video_id}_{row["set"]}_{row["rally"]}_{int(row["ball_round"])}.mp4')
            if out_path.exists():
                continue

            start_f, end_f = compute_clip_bounds(row, clip_window, fps)

            clip = video.subclipped(
                _frame_to_time(start_f, fps),
                _frame_to_time(end_f, fps),
            )
            clip.write_videofile(str(out_path), logger=None)
            clips_written += 1
    finally:
        video.close()

    return clips_written


def _filter_removed_shots(
    shots_df: pd.DataFrame,
    vid: int,
    removed_shots: set[tuple[int, int, int, int]],
) -> pd.DataFrame:
    """Drop rows matching entries in removed_shots.

    Builds a ``(set, rally, ball_round)`` tuple per row and checks membership
    against the removals for this video. Numpy int tuples hash equal to plain
    int tuples, so no per-column casting is needed.

    :param shots_df: DataFrame with 'set', 'rally', 'ball_round' columns.
    :param vid: Video ID (first element of each removed_shots tuple).
    :param removed_shots: Set of (video_id, set, rally, ball_round) tuples.
    :return: Filtered DataFrame.
    """
    # Keep only the removals for this video
    to_remove = {(s, r, b) for v, s, r, b in removed_shots if v == vid}
    if not to_remove:
        return shots_df

    row_keys = pd.Series(
        list(zip(shots_df['set'], shots_df['rally'], shots_df['ball_round'])),
        index=shots_df.index,
    )
    return shots_df[~row_keys.isin(to_remove)]


def generate_all_clips(
    raw_video_dir: Path = RAW_VIDEO_DIR,
    set_info_dir: Path = SET_INFO_DIR,
    output_dir: Path = CLIPS_OUTPUT_DIR,
    clip_window: str = CLIP_WINDOW,
) -> None:
    """Generate labeled clip .mp4s for all splits, both players, all videos.

    :param raw_video_dir: Directory containing downloaded match videos.
    :param set_info_dir: Directory containing match.csv and per-match set CSVs.
    :param output_dir: Root output directory for clips (split subdirs created inside).
    :param clip_window: Temporal clipping window name.
    """
    # Load match metadata.
    # Each row is a pd.Series with fields: 'video' (str), 'downcourt' (bool).
    # The index (accessed via .name) is the integer video ID.
    match_df = pd.read_csv(set_info_dir / 'match.csv')[['id', 'video', 'downcourt']]
    match_df['downcourt'] = match_df['downcourt'].astype(bool)
    match_df = match_df.set_index('id')

    total_clips = 0
    for split_name, vid_ids in SPLITS.items():
        print(f'\n=== Split: {split_name} ({len(vid_ids)} videos) ===')
        out_folder = output_dir / split_name
        out_folder.mkdir(parents=True, exist_ok=True)

        for vid in vid_ids:
            if vid not in match_df.index:
                continue
            v_info = match_df.loc[vid]

            # Collect all shots (both players, English types)
            shots_df = collect_shots(set_info_dir, v_info, STROKE_TYPES_19_ZH)
            if shots_df.empty:
                continue

            # Filter out individually removed shots
            before = len(shots_df)
            shots_df = _filter_removed_shots(shots_df, vid, REMOVED_SHOTS)
            removed_count = before - len(shots_df)

            # Add temporal boundaries (start_f, end_f)
            folder_path = set_info_dir / v_info['video']
            shots_df = compute_temporal_bounds(folder_path, shots_df)

            # Write clips with English folder names
            n = _write_clips_for_video(
                raw_video_dir, out_folder,
                video_id=vid,
                shots_df=shots_df,
                clip_window=clip_window,
                stroke_types=STROKE_TYPES_19,
                players=PLAYERS,
            )
            total_clips += n
            status = f'video {vid:2d}: {n} new clips'
            if removed_count:
                status += f' ({removed_count} removed shots filtered)'
            print(f'  {status}')

    print(f'\nTotal new clips written: {total_clips}')


def apply_class_merge(
    output_dir: Path = CLIPS_OUTPUT_DIR,
    taxonomy: Taxonomy = _DEFAULT_TAXONOMY,
) -> None:
    """Merge rare subtype folders into their parent type folders.

    For example, Top_wrist_smash/*.mp4 -> Top_smash/*.mp4.
    Source folders are removed after merging.

    :param output_dir: Root clips directory containing split subdirs.
    :param taxonomy: Taxonomy whose merge_map defines which subtypes to merge.
    """
    if not taxonomy.merge_map:
        print('Taxonomy has no merge_map — nothing to merge.')
        return

    split_dirs = [d for d in sorted(output_dir.iterdir()) if d.is_dir()]
    move_ops = []
    for split_dir in split_dirs:
        for src_type, dst_type in taxonomy.merge_map.items():
            noside_src = src_type in NOSIDE_FOLDERS
            noside_dst = dst_type in NOSIDE_FOLDERS
            if noside_dst and not noside_src:
                raise NotImplementedError(
                    f'sided source {src_type!r} -> NOSIDE destination {dst_type!r}: '
                    f'clips live under Top_/Bottom_ and this branch would '
                    f'silently merge zero clips.'
                )
            if noside_src:
                # driven_flight -> drive: NOSIDE source folders carry no Top_/Bottom_,
                # so these clips merge into a flat drive/ beside the sided pair.
                # Harmless: downstream lookups are stem-keyed.
                src = split_dir / src_type
                dst = split_dir / dst_type
                if src.exists():
                    move_ops.append((src, dst))
            else:
                for player in PLAYERS:
                    src = split_dir / f'{player}_{src_type}'
                    dst = split_dir / f'{player}_{dst_type}'
                    if src.exists():
                        move_ops.append((src, dst))

    moved = 0
    for src, dst in move_ops:
        dst.mkdir(parents=True, exist_ok=True)
        for clip_file in src.iterdir():
            shutil.move(str(clip_file), str(dst / clip_file.name))
            moved += 1
        try:
            src.rmdir()  # may fail if stray non-clip files remain (e.g. .DS_Store)
        except OSError:
            pass

    print(f'Class merge complete: {moved} clips moved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate labeled ShuttleSet stroke clips from raw match videos.',
    )
    parser.add_argument('--clip-window', default=CLIP_WINDOW,
                        choices=sorted(VALID_CLIP_WINDOWS),
                        help='Temporal clipping window')
    parser.add_argument('--no-merge', action='store_true',
                        help='Skip class merging (keep all 19 types)')
    args = parser.parse_args()

    print('=== Generating clips ===')
    generate_all_clips(clip_window=args.clip_window)

    if not args.no_merge:
        print('\n=== Applying class merge ===')
        apply_class_merge()
