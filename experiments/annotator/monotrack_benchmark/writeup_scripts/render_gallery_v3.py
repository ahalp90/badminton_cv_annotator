"""Draw the v3 court galleries, with a plain legend on every sheet.

Line styles follow the earlier galleries: our court red with long dashes, MonoTrack's court
fluoro pink with short dashes, both 2 px; ShuttleSet22's reference corners white with fine
1 px dashes. Outlines are drawn after resizing so line widths stay fixed.

MonoTrack fits its court on each rally's middle frame. When that frame is a close-up, the
other two outlines (both for the main camera view) cannot match it either. Those rallies
are therefore shown as pairs: the frame MonoTrack fitted on, and a main-camera frame from
the same rally.

Usage: python render_gallery_v3.py --courts CSV --frames DIR --main-frames DIR --out DIR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.annotator.monotrack_benchmark.writeup_scripts.render_gallery import (
    BLACK,
    FLUORO_PINK,
    JPEG_QUALITY,
    RED,
    WHITE,
    OutlineStyle,
    corners_of,
    dashed_line,
    dashed_outline,
    tag,
)

HIGH_ERROR_PX = 15.0
ZOOM_VIDEO = 17
ZOOM_EXAMPLE_RALLY = 47
OURS = OutlineStyle('Our court (the tag on each frame says per rally or per scene)', RED, 2, (18, 8))
MONOTRACK = OutlineStyle("MonoTrack's court", FLUORO_PINK, 2, (7, 7))
REFERENCE = OutlineStyle('ShuttleSet22 reference corners (one set per video, for the main camera view)',
                         WHITE, 1, (4, 4))
# On-frame tags name each outline's source, as in the first report's galleries.
OUR_MODE_TAGS = {'ours_rally_corners': 'ours (per rally)', 'ours_discovery_corners': 'ours (per scene)'}
MONOTRACK_TAG = OutlineStyle('MonoTrack', FLUORO_PINK, 2, (7, 7))
REFERENCE_TAG = OutlineStyle('reference corners', WHITE, 1, (4, 4))
FONT = cv2.FONT_HERSHEY_SIMPLEX
BAND = (30, 30, 30)


def text_size(text: str, scale: float) -> tuple[int, int]:
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, max(1, round(scale * 2)))
    return width, height + baseline


def put(image: np.ndarray, text: str, x: int, y: int, scale: float, colour=WHITE) -> None:
    cv2.putText(image, text, (x, y), FONT, scale, colour, max(1, round(scale * 2)), cv2.LINE_AA)


def legend_band(width: int, title: str, notes: list[str], scale: float = 0.85) -> np.ndarray:
    """A header with the sheet title, one line sample per outline style and short notes."""
    line = round(46 * scale)
    sample = round(110 * scale)
    # Lay out the samples first, wrapping before any that would run off the edge,
    # so the band is tall enough for however many rows they need.
    placements = []
    x, row = 16, 0
    for style in (OURS, MONOTRACK, REFERENCE):
        item_width = sample + 12 + text_size(style.label, scale)[0]
        if x > 16 and x + item_width > width - 16:
            x, row = 16, row + 1
        placements.append((style, x, row))
        x += item_width + 48
    first_y = round(line * 2.1)
    notes_y = first_y + line * row
    band = np.full((notes_y + line * len(notes) + round(line * 0.6), width, 3), BAND, dtype=np.uint8)
    put(band, title, 16, line, scale * 1.25)
    for style, x, row in placements:
        y = first_y + line * row
        # Draw the sample twice as long as a dash cycle so the pattern is recognisable.
        dashed_line(band, np.array([x, y - line * 0.25]), np.array([x + sample, y - line * 0.25]), style)
        put(band, style.label, x + sample + 12, y, scale)
    for index, note in enumerate(notes):
        put(band, note, 16, notes_y + line * (index + 1), scale * 0.9, (220, 220, 220))
    return band


def draw_courts(frame: np.ndarray, row: pd.Series, scale: float, ours_column: str, tag_scale: float) -> np.ndarray:
    """Draw the outlines, tag each with its source, and flag either of our modes that gave no court."""
    image = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    monotrack = corners_of(row['monotrack_corners'])
    ours = corners_of(row[ours_column])
    reference = corners_of(row['official']) * scale
    if monotrack is not None:
        dashed_outline(image, monotrack * scale, MONOTRACK)
    if ours is not None:
        dashed_outline(image, ours * scale, OURS)
    dashed_outline(image, reference, REFERENCE)
    # Each tag starts at a different corner so coincident outlines keep readable labels.
    offset = 16 * tag_scale
    placed: list[tuple[int, int, int, int]] = []
    for column, label in OUR_MODE_TAGS.items():
        if corners_of(row[column]) is None:
            tag(image, np.zeros(2), OutlineStyle(f'{label} gave no court', RED, 2, (18, 8)), tag_scale, placed,
                bottom_right=True)
    tag(image, reference[3] + [offset, offset], REFERENCE_TAG, tag_scale, placed)
    if ours is not None:
        tag(image, ours[0] * scale + [-offset * 2, -offset * 4], OutlineStyle(OUR_MODE_TAGS[ours_column], RED, 2,
            (18, 8)), tag_scale, placed)
    if monotrack is not None:
        tag(image, monotrack[1] * scale + [-offset * 14, -offset * 4], MONOTRACK_TAG, tag_scale, placed)
    return image


def titled(image: np.ndarray, lines: list[str], scale: float) -> np.ndarray:
    """Stack caption lines above an image."""
    line = round(40 * scale)
    band = np.full((line * len(lines) + round(line * 0.4), image.shape[1], 3), BLACK, dtype=np.uint8)
    for index, text in enumerate(lines):
        put(band, text, 8, line * (index + 1) - round(8 * scale), scale)
    return np.vstack([band, image])


def error_text(value: float) -> str:
    return 'no court' if pd.isna(value) else f'{value:,.1f} px'


def grid(panels: list[np.ndarray], columns: int, gap: int = 12) -> np.ndarray:
    height = max(panel.shape[0] for panel in panels)
    width = panels[0].shape[1]
    rows = -(-len(panels) // columns)
    canvas = np.full((rows * (height + gap), columns * (width + gap) - gap, 3), BAND, dtype=np.uint8)
    for index, panel in enumerate(panels):
        top, left = (index // columns) * (height + gap), (index % columns) * (width + gap)
        canvas[top:top + panel.shape[0], left:left + width] = panel
    return canvas


def save(image: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def middle_frame(frames: Path, row: pd.Series) -> np.ndarray:
    path = frames / f'gallery_{row["video"]}_{row["rally"]:03d}_f{row["midpoint_frame"]}.jpg'
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def monotrack_failures(courts: pd.DataFrame, frames: Path, main_frames: Path, out: Path) -> None:
    rows = courts[(courts['monotrack_error_px'] > HIGH_ERROR_PX) & (courts['video'] != ZOOM_VIDEO)]
    scale, font, tag_scale = 0.5, 0.65, 0.55
    pairs = []
    for _, row in rows.sort_values(['video', 'rally']).iterrows():
        (main_path,) = main_frames.glob(f'main_{row["video"]}_{row["rally"]:03d}_f*.jpg')
        main_frame_number = int(main_path.stem.split('_f')[-1])
        left = titled(draw_courts(middle_frame(frames, row), row, scale, 'ours_discovery_corners', tag_scale),
                      [f'Video {row["video"]}, rally {row["rally"]}: middle frame {row["midpoint_frame"]:,}',
                       f'MonoTrack fitted its court here. Error {error_text(row["monotrack_error_px"])}.'], font)
        right = titled(draw_courts(cv2.imread(str(main_path)), row, scale, 'ours_discovery_corners', tag_scale),
                       [f'Same rally, main camera, frame {main_frame_number:,}',
                        (f'Our court (per scene): {error_text(row["ours_discovery_error_px"])}. '
                        'Per rally: no court.')], font)
        pairs.append(np.hstack([left, np.full((left.shape[0], 6, 3), BAND, dtype=np.uint8), right]))
    # One pair per row keeps each frame wide enough to read the dashed outlines.
    sheet = grid(pairs, 1, gap=24)
    header = legend_band(sheet.shape[1], f'The {len(pairs)} rallies where MonoTrack fitted its court to a close-up '
                         'or another camera', [
                             'Each pair: left, the rally\'s middle frame, where MonoTrack fits its court;',
                             'right, a main-camera frame from the same rally.',
                             'Our per-rally detector gave no court for these rallies, so red shows our per-scene court.',
                             'Error = mean corner distance from the reference corners.',
                             'MonoTrack is also over 15 px on 27 rallies in video 17, where the broadcast zoomed in',
                             'and the reference corners are out of date. Those are shown separately.',
                         ], scale=0.8)
    save(np.vstack([header, sheet]), out / 'fig7_court_gallery_monotrack_failures.jpg')


def discovery_failures(courts: pd.DataFrame, frames: Path, out: Path) -> None:
    only_ours = ((courts['ours_discovery_error_px'] > HIGH_ERROR_PX) & ~(courts['monotrack_error_px'] > HIGH_ERROR_PX)
                 & (courts['video'] != ZOOM_VIDEO))
    scale, font = 0.5, 0.6
    panels = []
    for _, row in courts[only_ours].sort_values(['video', 'rally']).iterrows():
        panels.append(titled(draw_courts(middle_frame(frames, row), row, scale, 'ours_discovery_corners', 0.55), [
            f'Video {row["video"]}, rally {row["rally"]}: middle frame {row["midpoint_frame"]:,}',
            (f'Our per-scene court {error_text(row["ours_discovery_error_px"])} | our per-rally court '
            f'{error_text(row["ours_rally_error_px"])} (not drawn) | MonoTrack {error_text(row["monotrack_error_px"])}'),
        ], font))
    sheet = grid(panels, 2)
    header = legend_band(sheet.shape[1], f'The {len(panels)} rallies where only our per-scene court is over 15 px off', [
        ('Red shows our per-scene court, used when the annotator finds rallies itself. Error = mean corner distance '
        'from the reference corners.'),
        ('Top row: the scene took a court from the wrong shot, mostly outside the frame. Bottom row: the right-hand '
        'sideline sits too far out.'),
    ], scale=0.7)
    save(np.vstack([header, sheet]), out / 'fig8_court_gallery_discovery_failures.jpg')


def zoom_example(courts: pd.DataFrame, frames: Path, out: Path) -> None:
    row = courts[(courts['video'] == ZOOM_VIDEO) & (courts['rally'] == ZOOM_EXAMPLE_RALLY)].iloc[0]
    panel = titled(draw_courts(middle_frame(frames, row), row, 0.5, 'ours_rally_corners', 0.55), [
        f'Video 17, rally 47: middle frame {row["midpoint_frame"]:,}',
        (f'Our per-rally court {error_text(row["ours_rally_error_px"])} | MonoTrack '
        f'{error_text(row["monotrack_error_px"])} from the reference corners'),
    ], 0.6)
    header = legend_band(panel.shape[1], 'Video 17 after the broadcast zoomed in (rallies 47-73)', [
        'Red and pink sit on the painted lines. The white reference was drawn for the wider view used earlier.',
    ], scale=0.6)
    save(np.vstack([header, panel]), out / 'fig9_court_gallery_video17_example.jpg')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--courts', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--main-frames', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    courts = pd.read_csv(args.courts)
    args.out.mkdir(parents=True, exist_ok=True)
    monotrack_failures(courts, args.frames, args.main_frames, args.out)
    discovery_failures(courts, args.frames, args.out)
    zoom_example(courts, args.frames, args.out)


if __name__ == '__main__':
    main()
