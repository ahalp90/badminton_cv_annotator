"""Draw the high-error court gallery for the MonoTrack report.

A rally enters the gallery when any court source is more than 15 px (mean corner error)
from ShuttleSet22's official corners. Each panel is the rally's midpoint frame with:

- our court: red, 2 px, long dashes
- MonoTrack's court: fluoro pink, 2 px, short dashes
- the official corners: white, 1 px, fine dashes

Red normally shows our per-rally court. Where that court is missing, or where only our
discovery-mode court is over 15 px, red shows the discovery-mode court instead. A text
tag on every outline names its source, so colour is never the only cue.

Usage: python render_gallery.py --frames DIR --courts CSV --out DIR
Writes one full-resolution JPEG per rally under ``--out/court_gallery/`` and two overview
sheets in ``--out``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HIGH_ERROR_PX = 15.0
CAMERA_ZOOM_VIDEO = 17
ZOOM_EXAMPLE_RALLY = 47
# BGR colours.
RED = (0, 0, 255)
FLUORO_PINK = (240, 16, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
JPEG_QUALITY = 92


@dataclass(frozen=True)
class OutlineStyle:
    label: str
    colour: tuple[int, int, int]
    thickness: int
    dash: tuple[int, int]


OURS_RALLY = OutlineStyle('ours (rally-bounded)', RED, 2, (18, 8))
OURS_DISCOVERY = OutlineStyle('ours (discovery mode)', RED, 2, (18, 8))
MONOTRACK = OutlineStyle('MonoTrack', FLUORO_PINK, 2, (7, 7))
OFFICIAL = OutlineStyle('official corners', WHITE, 1, (4, 4))


def corners_of(value: str) -> np.ndarray | None:
    return None if pd.isna(value) or value == '' else np.asarray(json.loads(value), dtype=float)


def dashed_line(image: np.ndarray, start: np.ndarray, end: np.ndarray, style: OutlineStyle) -> None:
    dash_px, gap_px = style.dash
    length = float(np.linalg.norm(end - start))
    direction = (end - start) / length
    position = 0.0
    while position < length:
        stop = min(position + dash_px, length)
        first = tuple(np.rint(start + direction * position).astype(int).tolist())
        second = tuple(np.rint(start + direction * stop).astype(int).tolist())
        # LINE_8 keeps 1 px lines crisp; anti-aliasing would blur the official outline.
        cv2.line(image, first, second, style.colour, style.thickness, cv2.LINE_8)
        position = stop + gap_px


def dashed_outline(image: np.ndarray, corners: np.ndarray, style: OutlineStyle) -> None:
    """Draw a closed outline; corners are image TL, TR, BR, BL."""
    for start, end in zip(corners, np.roll(corners, -1, axis=0), strict=True):
        dashed_line(image, start, end, style)


def overlaps(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = first
    other_left, other_top, other_right, other_bottom = second
    return left < other_right and other_left < right and top < other_bottom and other_top < bottom


def tag(image: np.ndarray, anchor: np.ndarray, style: OutlineStyle, font_scale: float,
        placed: list[tuple[int, int, int, int]], *, bottom_right: bool = False) -> None:
    """Label an outline next to one of its corners, inside the frame and clear of earlier tags."""
    text = style.label
    thickness = max(1, round(font_scale * 2))
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    swatch = height
    pad = max(3, round(font_scale * 6))
    box_width, box_height = swatch + 3 * pad + width, height + baseline + 2 * pad
    frame_height, frame_width = image.shape[:2]
    if bottom_right:
        x, y = frame_width - box_width - pad, frame_height - box_height - pad
    else:
        x = int(np.clip(anchor[0], 0, frame_width - box_width))
        y = int(np.clip(anchor[1], 0, frame_height - box_height))
    step = box_height + 2
    while any(overlaps((x, y, x + box_width, y + box_height), other) for other in placed):
        y = y + step if y + step <= frame_height - box_height else y - step
        step = abs(step)
    placed.append((x, y, x + box_width, y + box_height))
    cv2.rectangle(image, (x, y), (x + box_width, y + box_height), BLACK, -1)
    cv2.rectangle(image, (x + pad, y + pad), (x + pad + swatch, y + pad + swatch), style.colour, -1)
    cv2.putText(image, text, (x + 2 * pad + swatch, y + pad + height), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                WHITE, thickness, cv2.LINE_AA)


def legend_sample(image: np.ndarray, x: int, y: int, style: OutlineStyle, font_scale: float) -> int:
    """Draw a line sample and its label on the caption band; return the next free x."""
    sample_length = round(90 * font_scale / 0.7)
    dashed_line(image, np.array([x, y], float), np.array([x + sample_length, y], float), style)
    thickness = max(1, round(font_scale * 2))
    text_x = x + sample_length + round(10 * font_scale)
    cv2.putText(image, style.label, (text_x, y + round(8 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                WHITE, thickness, cv2.LINE_AA)
    (width, _), _ = cv2.getTextSize(style.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    return text_x + width + round(40 * font_scale)


def format_error(value: float) -> str:
    return 'no court' if pd.isna(value) else f'{value:.1f} px'


def our_court_for_panel(row: pd.Series) -> tuple[np.ndarray | None, OutlineStyle]:
    discovery_only = (row['ours_discovery_error_px'] > HIGH_ERROR_PX
                      and not row['ours_rally_error_px'] > HIGH_ERROR_PX)
    if discovery_only or pd.isna(row['ours_rally_error_px']):
        return corners_of(row['ours_discovery_corners']), OURS_DISCOVERY
    return corners_of(row['ours_rally_corners']), OURS_RALLY


def render_panel(frame: np.ndarray, row: pd.Series, scale: float, font_scale: float) -> np.ndarray:
    """Return the frame at ``scale`` with outlines, tags and a caption band beneath it.

    Outlines are drawn after resizing, so line widths stay 2 px and 1 px at every scale.
    """
    image = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else frame.copy()
    official = corners_of(row['official']) * scale
    monotrack = corners_of(row['monotrack_corners'])
    ours, ours_style = our_court_for_panel(row)
    if monotrack is not None:
        dashed_outline(image, monotrack * scale, MONOTRACK)
    if ours is not None:
        dashed_outline(image, ours * scale, ours_style)
    # The thin official outline goes on top so it stays visible where a fit coincides with it.
    dashed_outline(image, official, OFFICIAL)
    # Each tag starts at a different corner so coincident outlines keep readable labels.
    offset = 16 * font_scale
    placed: list[tuple[int, int, int, int]] = []
    for column, style in (('ours_rally_corners', OURS_RALLY), ('ours_discovery_corners', OURS_DISCOVERY)):
        if corners_of(row[column]) is None:
            missing = OutlineStyle(f'{style.label} gave no court', style.colour, style.thickness, style.dash)
            tag(image, np.zeros(2), missing, font_scale, placed, bottom_right=True)
    tag(image, official[3] + [offset, offset], OFFICIAL, font_scale, placed)
    if ours is not None:
        tag(image, ours[0] * scale + [-offset * 2, -offset * 4], ours_style, font_scale, placed)
    if monotrack is not None:
        tag(image, monotrack[1] * scale + [-offset * 14, -offset * 4], MONOTRACK, font_scale, placed)
    return add_caption(image, row, ours_style, font_scale)


def add_caption(image: np.ndarray, row: pd.Series, ours_style: OutlineStyle, font_scale: float) -> np.ndarray:
    line_height = round(34 * font_scale / 0.7)
    band = np.zeros((round(line_height * 3.4), image.shape[1], 3), dtype=np.uint8)
    thickness = max(1, round(font_scale * 2))
    asterisk = '*' if (row['video'], row['rally']) in {(49, 86), (54, 61)} else ''
    lines = [
        (f'Video {row["video"]}, rally {row["rally"]}{asterisk} ({row["rally_id"]}), '
         f'midpoint frame {row["midpoint_frame"]:,}'),
        (f'Mean corner error from official: ours per-rally {format_error(row["ours_rally_error_px"])} | '
         f'MonoTrack {format_error(row["monotrack_error_px"])} | '
         f'ours discovery {format_error(row["ours_discovery_error_px"])}'),
    ]
    for index, text in enumerate(lines):
        cv2.putText(band, text, (10, line_height * (index + 1) - round(8 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, WHITE, thickness, cv2.LINE_AA)
    y = round(line_height * 2.75)
    x = legend_sample(band, 10, y, ours_style, font_scale)
    x = legend_sample(band, x, y, MONOTRACK, font_scale)
    legend_sample(band, x, y, OFFICIAL, font_scale)
    return np.vstack([image, band])


def sheet(panels: list[np.ndarray], columns: int, title: str) -> np.ndarray:
    height, width = panels[0].shape[:2]
    rows = -(-len(panels) // columns)
    gap = 8
    canvas = np.full((rows * (height + gap) + 70, columns * (width + gap) - gap, 3), 30, dtype=np.uint8)
    cv2.putText(canvas, title, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2, cv2.LINE_AA)
    for index, panel in enumerate(panels):
        top = 70 + (index // columns) * (height + gap)
        left = (index % columns) * (width + gap)
        canvas[top:top + height, left:left + width] = panel
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--courts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    courts = pd.read_csv(args.courts)
    error_columns = ['ours_rally_error_px', 'monotrack_error_px', 'ours_discovery_error_px']
    gallery = courts[courts[error_columns].gt(HIGH_ERROR_PX).any(axis=1)].copy()
    gallery['zoom_video'] = gallery['video'] == CAMERA_ZOOM_VIDEO
    gallery = gallery.sort_values(['zoom_video', 'video', 'rally'])
    (args.out / 'court_gallery').mkdir(parents=True, exist_ok=True)
    detector_panels, zoom_panels, monotrack_panels, discovery_panels = [], [], [], []
    for _, row in gallery.iterrows():
        frame_path = args.frames / f'gallery_{row["video"]}_{row["rally"]:03d}_f{row["midpoint_frame"]}.jpg'
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(frame_path)
        full = render_panel(frame, row, 1.0, 0.9)
        name = f'video{row["video"]:02d}_rally{row["rally"]:03d}.jpg'
        cv2.imwrite(str(args.out / 'court_gallery' / name), full, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if row['zoom_video']:
            zoom_panels.append(render_panel(frame, row, 1 / 3, 0.42))
            if row['rally'] == ZOOM_EXAMPLE_RALLY:
                cv2.imwrite(str(args.out / 'fig7_court_gallery_video17_example.jpg'),
                            render_panel(frame, row, 0.5, 0.55), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            continue
        panel = render_panel(frame, row, 0.5, 0.55)
        detector_panels.append(panel)
        if row['monotrack_error_px'] > HIGH_ERROR_PX:
            monotrack_panels.append(panel)
        else:
            discovery_panels.append(panel)
    cv2.imwrite(str(args.out / 'fig5_court_gallery_detector_failures.jpg'),
                sheet(detector_panels, 2, 'Rallies over 15 px where a detector failed (12 rallies)'),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    cv2.imwrite(str(args.out / 'fig6_court_gallery_video17_zoom.jpg'),
                sheet(zoom_panels, 3, 'Video 17, rallies 47-73: camera zoomed in, official corners no longer fit (27 rallies)'),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    cv2.imwrite(str(args.out / 'fig5_court_gallery_monotrack_failures.jpg'),
                sheet(monotrack_panels, 2, f'MonoTrack court over 15 px ({len(monotrack_panels)} rallies)'),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    cv2.imwrite(str(args.out / 'fig6_court_gallery_discovery_failures.jpg'),
                sheet(discovery_panels, 2, f'Only our discovery-mode court over 15 px ({len(discovery_panels)} rallies)'),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    print(f'{len(gallery)} rallies: {len(detector_panels)} detector failures, {len(zoom_panels)} zoomed video 17')


if __name__ == '__main__':
    main()
