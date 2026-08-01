#!/usr/bin/env python3
"""Build a labelled thumbnail pack for baseline cuts inside GT rallies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd

from annotator.calibration.scoring import GtRally, load_gt_rallies
from annotator.composition_mask import CompositionSegment, build_composition_mask
from annotator.config import COMPOSITION_KEEP_VOTE


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[3]
DATA_DIR = ANALYSIS_DIR / 'data'
IMAGE_DIR = ANALYSIS_DIR / 'imgs'
INDEX_PATH = DATA_DIR / 'midcut_index.csv.gz'
SHOTS_MASTER_PATH = REPO_ROOT / 'training/data/shuttleset/annotations/shots_master.csv'

VIDEO_IDS = (1, 15, 21)
VIDEO_PATHS = {
    1: REPO_ROOT / 'local_scratch/autograder_architecture/videos_288p/sset_01_288p.mp4',
    21: REPO_ROOT / 'local_scratch/autograder_architecture/videos_288p/sset_21_288p.mp4',
}
CUT_PATHS = {
    video_id: DATA_DIR / f'cuts_{video_id}_content27.csv.gz'
    for video_id in VIDEO_IDS
}
PRIMARY_VOTE_ROOT = REPO_ROOT / 'experiments/annotator/runs/20260730-041328/detected_ckn_opencv_consensus'
SECONDARY_VOTE_ROOT = REPO_ROOT / 'experiments/annotator/runs/20260730-041328/static_shuttleset_homography'

PALETTE_SIZES = (64, 128, 256)
DEFAULT_PALETTE_SIZE = 128
TITLE_STRIP_HEIGHT = 24
TARGET_DISPLAYED_TEXT_PX = 16
FONT = cv2.FONT_HERSHEY_DUPLEX
PINK_BGR = (147, 20, 255)  # RGB #FF1493
BLACK_BGR = (0, 0, 0)
PINK_THICKNESS = 2
OUTLINE_THICKNESS = 4


@dataclass(frozen=True)
class Midcut:
    """One baseline cut strictly inside one ground-truth rally extent."""

    video_id: int
    rally: GtRally
    cut_frame: int
    before_primary: CompositionSegment
    after_primary: CompositionSegment
    before_secondary: CompositionSegment
    after_secondary: CompositionSegment


def load_cut_list(path: Path) -> tuple[np.ndarray, int]:
    """Load cut frames and the single declared frame count from a tracked CSV."""
    table = pd.read_csv(path)
    required_columns = {'cut_frame', 'n_frames'}
    if not required_columns.issubset(table.columns):
        raise ValueError(f'{path}: expected columns {sorted(required_columns)}')

    declared_counts = table['n_frames'].drop_duplicates().tolist()
    if len(declared_counts) != 1:
        raise ValueError(f'{path}: n_frames is not constant: {declared_counts}')
    n_frames = int(declared_counts[0])
    cut_frames = table['cut_frame'].to_numpy(dtype=int)
    if len(np.unique(cut_frames)) != len(cut_frames):
        raise ValueError(f'{path}: duplicate cut_frame values')
    if np.any(cut_frames < 0) or np.any(cut_frames >= n_frames):
        raise ValueError(f'{path}: cut frame outside [0, {n_frames})')
    return cut_frames, n_frames


def load_vote(path: Path) -> np.ndarray:
    """Load one one-dimensional boolean court-view vote array."""
    vote = np.load(path, allow_pickle=False)
    if vote.dtype != np.dtype(bool) or vote.ndim != 1:
        raise ValueError(f'{path}: expected a one-dimensional bool array, got {vote.dtype} {vote.shape}')
    return vote


def segment_at_cut(segments: list[CompositionSegment], cut_frame: int) -> tuple[CompositionSegment, CompositionSegment]:
    """Return the [start, cut) and [cut, end) segments around a cut."""
    before = [segment for segment in segments if segment.end == cut_frame]
    after = [segment for segment in segments if segment.start == cut_frame]
    if len(before) != 1 or len(after) != 1:
        raise ValueError(f'could not find unique neighbouring segments around cut {cut_frame}')
    return before[0], after[0]


def collect_midcuts() -> tuple[list[Midcut], dict[int, int]]:
    """Apply the frame-count gate, build both verdict sets, and find mid-rally cuts."""
    shots_master = pd.read_csv(SHOTS_MASTER_PATH)
    midcuts: list[Midcut] = []
    n_frames_by_video: dict[int, int] = {}

    for video_id in VIDEO_IDS:
        cut_frames, n_frames = load_cut_list(CUT_PATHS[video_id])
        n_frames_by_video[video_id] = n_frames
        primary_vote = load_vote(
            PRIMARY_VOTE_ROOT / f'sset_{video_id:02d}/tracknet-stride-8/keep_vote.npy'
        )
        secondary_vote = load_vote(
            SECONDARY_VOTE_ROOT / f'sset_{video_id:02d}/tracknet-stride-8/keep_vote.npy'
        )
        assert len(primary_vote) == n_frames, (
            f'video {video_id}: cut-list n_frames {n_frames} != primary keep_vote length {len(primary_vote)}'
        )
        assert len(secondary_vote) == n_frames, (
            f'video {video_id}: cut-list n_frames {n_frames} != secondary keep_vote length {len(secondary_vote)}'
        )

        _, primary_segments = build_composition_mask(
            cut_frames, primary_vote, n_frames, COMPOSITION_KEEP_VOTE
        )
        _, secondary_segments = build_composition_mask(
            cut_frames, secondary_vote, n_frames, COMPOSITION_KEEP_VOTE
        )
        rallies = load_gt_rallies(shots_master, video_id)
        video_midcuts: list[Midcut] = []

        for cut_frame in cut_frames:
            matching_rallies = [
                rally for rally in rallies
                if rally.extent[0] < int(cut_frame) < rally.extent[1]
            ]
            if len(matching_rallies) > 1:
                raise ValueError(f'cut {cut_frame} belongs to multiple rallies in video {video_id}')
            if not matching_rallies:
                continue

            rally = matching_rallies[0]
            before_primary, after_primary = segment_at_cut(primary_segments, int(cut_frame))
            before_secondary, after_secondary = segment_at_cut(secondary_segments, int(cut_frame))
            if (before_primary.start, before_primary.end) != (before_secondary.start, before_secondary.end):
                raise ValueError(f'primary and secondary before spans differ around cut {cut_frame}')
            if (after_primary.start, after_primary.end) != (after_secondary.start, after_secondary.end):
                raise ValueError(f'primary and secondary after spans differ around cut {cut_frame}')

            video_midcuts.append(Midcut(
                video_id=int(video_id),
                rally=rally,
                cut_frame=int(cut_frame),
                before_primary=before_primary,
                after_primary=after_primary,
                before_secondary=before_secondary,
                after_secondary=after_secondary,
            ))

        midcuts.extend(video_midcuts)
        print(f'video {video_id}: {len(video_midcuts)} mid-rally cuts')

    return midcuts, n_frames_by_video


def midpoint_frame(segment: CompositionSegment) -> int:
    """Return the lower midpoint of a non-empty half-open segment."""
    if segment.start >= segment.end:
        raise ValueError(f'empty segment [{segment.start}, {segment.end})')
    return (segment.start + segment.end - 1) // 2


def requested_frames(midcut: Midcut) -> tuple[int, int, int, int]:
    """Return the four frame numbers in top-left, top-right, bottom-left, bottom-right order."""
    return (
        midpoint_frame(midcut.before_primary),
        midcut.cut_frame - 1,
        midcut.cut_frame,
        midpoint_frame(midcut.after_primary),
    )


def read_requested_frames(video_path: Path, frame_numbers: set[int], expected_n_frames: int) -> dict[int, np.ndarray]:
    """Decode one video up to its highest requested frame and retain only requested frames."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'could not open {video_path}')
    reported_n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if reported_n_frames != expected_n_frames:
        capture.release()
        raise ValueError(f'{video_path}: video frame count {reported_n_frames} != vote length {expected_n_frames}')

    frames: dict[int, np.ndarray] = {}
    remaining = set(frame_numbers)
    frame_number = 0
    while remaining:
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f'{video_path}: failed while reading frame {frame_number}')
        if frame_number in remaining:
            frames[frame_number] = frame.copy()
            remaining.remove(frame_number)
        frame_number += 1
    capture.release()
    return frames


def font_scale_for_height(target_height: int) -> float:
    """Choose the first OpenCV font scale whose measured glyph height reaches target_height."""
    font_scale = 0.1
    while cv2.getTextSize('Ag', FONT, font_scale, PINK_THICKNESS)[0][1] < target_height:
        font_scale += 0.01
    return font_scale


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], font_scale: float) -> None:
    """Draw neon-pink text with a one-pixel black outline around the pink stroke."""
    cv2.putText(image, text, origin, FONT, font_scale, BLACK_BGR, OUTLINE_THICKNESS, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, font_scale, PINK_BGR, PINK_THICKNESS, cv2.LINE_AA)


def span_text(segment: CompositionSegment) -> str:
    return f'[{segment.start},{segment.end})'


def verdict_text(segment: CompositionSegment) -> str:
    return 'DEAD' if segment.is_dead else 'LIVE'


def make_sheet(midcut: Midcut, frames: dict[int, np.ndarray]) -> np.ndarray:
    """Make one native-resolution 2x2 sheet with the requested overlays."""
    frame_numbers = requested_frames(midcut)
    sample_frame = frames[frame_numbers[0]]
    tile_height, tile_width = sample_frame.shape[:2]
    sheet_height = tile_height * 2 + TITLE_STRIP_HEIGHT
    sheet_width = tile_width * 2
    target_glyph_height = round(TARGET_DISPLAYED_TEXT_PX * sheet_height / 1080)
    font_scale = font_scale_for_height(target_glyph_height)
    sheet = np.zeros((sheet_height, sheet_width, 3), dtype=np.uint8)

    title = (
        f'v{midcut.video_id} {midcut.rally.set_id}/r{midcut.rally.rally} '
        f'cut f{midcut.cut_frame} | before {span_text(midcut.before_primary)} '
        f'after {span_text(midcut.after_primary)}'
    )
    title_size, title_baseline = cv2.getTextSize(title, FONT, font_scale, PINK_THICKNESS)
    if title_size[0] + 8 > sheet_width:
        raise ValueError(f'title does not fit on {sheet_width}px sheet: {title}')
    title_y = (TITLE_STRIP_HEIGHT + title_size[1] - title_baseline) // 2
    draw_label(sheet, title, (4, title_y), font_scale)

    before_disagreement = midcut.before_primary.is_dead != midcut.before_secondary.is_dead
    after_disagreement = midcut.after_primary.is_dead != midcut.after_secondary.is_dead
    tile_specs = (
        ('BEFORE-MID', frame_numbers[0], midcut.before_primary, before_disagreement, 0, TITLE_STRIP_HEIGHT),
        ('BEFORE-LAST', frame_numbers[1], midcut.before_primary, before_disagreement, tile_width, TITLE_STRIP_HEIGHT),
        ('AFTER-FIRST', frame_numbers[2], midcut.after_primary, after_disagreement, 0, TITLE_STRIP_HEIGHT + tile_height),
        ('AFTER-MID', frame_numbers[3], midcut.after_primary, after_disagreement, tile_width, TITLE_STRIP_HEIGHT + tile_height),
    )
    for role, frame_number, primary_segment, disagreement, x, y in tile_specs:
        tile = frames[frame_number]
        if tile.shape[:2] != (tile_height, tile_width):
            raise ValueError(f'frame {frame_number} has inconsistent dimensions {tile.shape[:2]}')
        sheet[y:y + tile_height, x:x + tile_width] = tile
        label_x = x + 4
        line_height = cv2.getTextSize('Ag', FONT, font_scale, PINK_THICKNESS)[0][1]
        line_one_y = y + 4 + line_height
        line_two_y = line_one_y + line_height + 2
        draw_label(sheet, f'{role} f{frame_number}', (label_x, line_one_y), font_scale)
        marker = '*' if disagreement else ''
        draw_label(
            sheet,
            f'{verdict_text(primary_segment)} {primary_segment.keep_fraction:.2f}{marker}',
            (label_x, line_two_y),
            font_scale,
        )
    return sheet


def run_checked(command: list[str]) -> None:
    """Run a compression command and include captured output on failure."""
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        raise RuntimeError(f'command failed with exit {result.returncode}: {command}\n{output}')


def quantise_and_optimise(raw_sheet: Path, output_path: Path, palette_size: int) -> None:
    """Apply pngquant's requested lossy pass, then oxipng's lossless pass."""
    quantised_path = raw_sheet.with_name(f'{raw_sheet.stem}-quantised.png')
    run_checked([
        'pngquant',
        '--quality=40-60',
        '--force',
        '--output',
        str(quantised_path),
        str(palette_size),
        str(raw_sheet),
    ])
    os.replace(quantised_path, output_path)
    run_checked(['oxipng', '-o', 'max', '--strip', 'all', str(output_path)])


def clear_png_outputs() -> None:
    """Remove stale PNG deliverables and probe images in the requested output directory."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for path in IMAGE_DIR.iterdir():
        if path.is_file() and path.suffix.lower() == '.png':
            path.unlink()


def image_filename(midcut: Midcut) -> str:
    set_number = midcut.rally.set_id.removeprefix('set')
    return f'midcut_{midcut.video_id}_s{set_number}r{midcut.rally.rally}_f{midcut.cut_frame}.png'


def make_index_row(midcut: Midcut, filename: str) -> dict[str, object]:
    before_disagreement = midcut.before_primary.is_dead != midcut.before_secondary.is_dead
    after_disagreement = midcut.after_primary.is_dead != midcut.after_secondary.is_dead
    return {
        'image_path': str(Path('imgs') / filename),
        'video_id': midcut.video_id,
        'set_id': midcut.rally.set_id,
        'rally': midcut.rally.rally,
        'cut_frame': midcut.cut_frame,
        'before_segment_span': span_text(midcut.before_primary),
        'before_primary_keep_fraction': midcut.before_primary.keep_fraction,
        'before_primary_verdict': verdict_text(midcut.before_primary),
        'before_secondary_keep_fraction': midcut.before_secondary.keep_fraction,
        'before_secondary_verdict': verdict_text(midcut.before_secondary),
        'before_verdict_disagreement': before_disagreement,
        'after_segment_span': span_text(midcut.after_primary),
        'after_primary_keep_fraction': midcut.after_primary.keep_fraction,
        'after_primary_verdict': verdict_text(midcut.after_primary),
        'after_secondary_keep_fraction': midcut.after_secondary.keep_fraction,
        'after_secondary_verdict': verdict_text(midcut.after_secondary),
        'after_verdict_disagreement': after_disagreement,
        'rally_first_contact_frame': midcut.rally.extent[0],
        'rally_last_contact_frame': midcut.rally.extent[1],
    }


def prepare_frames(midcuts: list[Midcut], n_frames_by_video: dict[int, int]) -> dict[int, dict[int, np.ndarray]]:
    """Read only the four-frame requests needed for videos with mid-rally cuts."""
    requests_by_video: dict[int, set[int]] = {}
    for midcut in midcuts:
        requests_by_video.setdefault(midcut.video_id, set()).update(requested_frames(midcut))

    frames_by_video: dict[int, dict[int, np.ndarray]] = {}
    for video_id, frame_numbers in requests_by_video.items():
        frames_by_video[video_id] = read_requested_frames(
            VIDEO_PATHS[video_id], frame_numbers, n_frames_by_video[video_id]
        )
        print(f'video {video_id}: decoded {len(frame_numbers)} unique requested frames')
    return frames_by_video


def check_tools() -> None:
    for command in ('pngquant', 'oxipng'):
        if shutil.which(command) is None:
            raise RuntimeError(f'required compression tool is not on PATH: {command}')


def build_pack(palette_size: int, midcuts: list[Midcut], n_frames_by_video: dict[int, int]) -> None:
    """Write the compressed image pack and gzipped index."""
    frames_by_video = prepare_frames(midcuts, n_frames_by_video)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix='.midcut-', dir=ANALYSIS_DIR) as temporary_name:
        temporary_dir = Path(temporary_name)
        for midcut in midcuts:
            filename = image_filename(midcut)
            output_path = IMAGE_DIR / filename
            raw_path = temporary_dir / filename
            sheet = make_sheet(midcut, frames_by_video[midcut.video_id])
            if not cv2.imwrite(str(raw_path), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                raise RuntimeError(f'could not write temporary sheet {raw_path}')
            quantise_and_optimise(raw_path, output_path, palette_size)
            rows.append(make_index_row(midcut, filename))

    index = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    index.to_csv(INDEX_PATH, index=False, compression='gzip')

    image_paths = [path for path in IMAGE_DIR.iterdir() if path.is_file() and path.suffix.lower() == '.png']
    pack_bytes = sum(path.stat().st_size for path in image_paths)
    segment_disagreements = sum(
        int(row['before_verdict_disagreement']) + int(row['after_verdict_disagreement'])
        for row in rows
    )
    sheets_with_disagreement = sum(
        row['before_verdict_disagreement'] or row['after_verdict_disagreement']
        for row in rows
    )
    print(f'palette: {palette_size} colours; image count: {len(image_paths)}')
    print(f'pack size: {pack_bytes} bytes ({pack_bytes / (1024 * 1024):.2f} MiB)')
    print(
        f'primary/secondary verdict disagreements: {segment_disagreements} adjacent-segment comparisons '
        f'across {sheets_with_disagreement} sheets'
    )
    print(f'index: {INDEX_PATH}')

    dead_neighbours = [
        (midcut, side, segment)
        for midcut in midcuts
        for side, segment in (
            ('before', midcut.before_primary),
            ('after', midcut.after_primary),
        )
        if segment.is_dead
    ]
    if dead_neighbours:
        print('primary DEAD neighbouring segments:')
        for midcut, side, segment in dead_neighbours:
            print(
                f'  v{midcut.video_id} {midcut.rally.set_id}/r{midcut.rally.rally} '
                f'cut {midcut.cut_frame} {side} {span_text(segment)} keep={segment.keep_fraction:.2f}'
            )
    else:
        print('primary DEAD neighbouring segments: none')


def write_palette_probes(midcuts: list[Midcut], n_frames_by_video: dict[int, int]) -> None:
    """Write one compressed sample at each candidate palette for visual inspection."""
    if not midcuts:
        raise ValueError('no mid-rally cuts available for palette probes')
    frames_by_video = prepare_frames([midcuts[0]], n_frames_by_video)
    sample = midcuts[0]
    with tempfile.TemporaryDirectory(prefix='.midcut-probe-', dir=ANALYSIS_DIR) as temporary_name:
        raw_path = Path(temporary_name) / 'probe-raw.png'
        sheet = make_sheet(sample, frames_by_video[sample.video_id])
        if not cv2.imwrite(str(raw_path), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
            raise RuntimeError(f'could not write temporary probe {raw_path}')
        for palette_size in PALETTE_SIZES:
            output_path = IMAGE_DIR / f'.probe_palette{palette_size}.png'
            quantise_and_optimise(raw_path, output_path, palette_size)
            print(f'palette probe: {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--palette-size', type=int, choices=PALETTE_SIZES, default=DEFAULT_PALETTE_SIZE)
    parser.add_argument(
        '--probe-palettes', action='store_true',
        help='write one sample at 64, 128, and 256 colours for visual inspection, without building the pack',
    )
    args = parser.parse_args()

    check_tools()
    clear_png_outputs()
    midcuts, n_frames_by_video = collect_midcuts()

    if args.probe_palettes:
        write_palette_probes(midcuts, n_frames_by_video)
        return

    build_pack(args.palette_size, midcuts, n_frames_by_video)


if __name__ == '__main__':
    main()
