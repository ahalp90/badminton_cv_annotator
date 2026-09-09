"""Render saved stripe selections without fitting or ranking any new geometry."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detector import CORNER_COURT_M, SEGMENTS_M

SIZE = (1280, 720)
SELECTOR = 'complete_agreements_first'
FIT_COLOUR = '#cc00cc'
REFERENCE_COLOUR = '#0072b2'
FONT = ImageFont.truetype('DejaVuSans.ttf', 22)
SMALL_FONT = ImageFont.truetype('DejaVuSans.ttf', 17)


def read_packed(path: Path, member: str) -> dict:
    with ZipFile(path) as archive:
        return json.loads(gzip.decompress(archive.read(member)))


def court_lines(corners: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Project finite painted markings, then scale to the displayed frame."""
    transform = cv2.getPerspectiveTransform(CORNER_COURT_M, corners.astype(np.float32))
    points = cv2.perspectiveTransform(SEGMENTS_M.reshape(1, -1, 2), transform)
    return points.reshape(-1, 2, 2) * scale


def draw_lines(
    image: Image.Image, lines: np.ndarray, colour: str, dashed: bool, clip_rect: tuple[int, ...],
) -> None:
    draw = ImageDraw.Draw(image)
    for start, end in lines:
        # Clip before drawing dashes so far-off-screen corners cannot inflate work.
        visible, clipped_start, clipped_end = cv2.clipLine(
            clip_rect, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)),
        )
        if not visible:
            continue
        start, end = np.asarray(clipped_start), np.asarray(clipped_end)
        if dashed:
            vector = end - start
            length = np.linalg.norm(vector)
            if length == 0:
                continue
            for offset in np.arange(0, length, 12):
                first = start + vector * offset / length
                last = start + vector * min(offset + 7, length) / length
                draw.line([tuple(first), tuple(last)], fill=colour, width=1)
        else:
            draw.line([tuple(start), tuple(end)], fill=colour, width=2)


def svg_lines(lines: np.ndarray, layer: str, colour: str) -> str:
    dash = ' stroke-dasharray="7 5"' if layer == 'reference' else ''
    width = 1.3 if layer == 'reference' else 1.8
    parts = [f'<g class="{layer}" fill="none" stroke="{colour}" stroke-width="{width}"{dash}>']
    for start, end in lines:
        parts.append(f'<path d="M {start[0]:.5f} {start[1]:.5f} L {end[0]:.5f} {end[1]:.5f}"/>')
    return ''.join(parts) + '</g>'


def pick(record: dict, pool: str) -> dict | None:
    order = record['orders'][pool][SELECTOR]
    return next(entry for entry in record['entries'] if entry['id'] == order[0]) if order else None


def describe(entry: dict | None) -> str:
    if entry is None:
        return 'No eligible candidate — no fitted overlay'
    metrics = entry['metrics']
    error = metrics['corner_max_error_px']
    status = 'within 15 px' if error <= 15 else 'above 15 px'
    return f"Worst corner {error:.2f} px · landmark RMS {metrics['landmark_rms_px']:.2f} px · {status}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recorded', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True, help='Private gzip JSON case-to-source-image map')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    saved = read_packed(args.recorded / 'stripe_diagnostics.zip', 'refit_selection_v2.json.gz')
    packed = read_packed(args.recorded / 'marking_refit_replay.zip', 'marking_inputs.json.gz')
    images = json.loads(gzip.decompress(args.images.read_bytes()))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    sections, markdown, manifest = [], [], []
    counts = {'starts': 0, 'fixed_position': 0}
    for record in saved['records']:
        identifier = record['id']
        reference = packed['references'][identifier]
        reference_corners = np.asarray(reference['corners_px'])
        source = Path(images[identifier])
        with Image.open(source) as raw:
            native_size = (record['dimensions']['width'], record['dimensions']['height'])
            assert raw.size == native_size, (identifier, raw.size, native_size)
            display_scale = min(SIZE[0] / native_size[0], SIZE[1] / native_size[1])
            display_size = tuple(round(length * display_scale) for length in native_size)
            display_offset = (np.asarray(SIZE) - display_size) // 2
            frame = Image.new('RGB', SIZE, '#222222')
            resized = raw.convert('RGB').resize(display_size, Image.Resampling.LANCZOS)
            frame.paste(resized, tuple(display_offset))
        metric_scale = np.asarray(SIZE) / native_size
        reference_lines = court_lines(reference_corners, np.asarray(display_scale)) + display_offset
        frame.save(output / f'{identifier}__frame.jpg', quality=93)
        comparison = Image.new('RGB', (SIZE[0] * 2, SIZE[1] + 110), 'white')
        panels, records = [], {}
        for index, (pool, title) in enumerate((('starts', 'Before: starting pool'),
                                                ('fixed_position', 'Latest: starts + fixed-position fits'))):
            entry = pick(record, pool)
            lines = np.empty((0, 2, 2))
            if entry is not None:
                assert entry['eligible']
                corners = np.asarray(entry['corners_px'])
                error = float(np.linalg.norm((corners - reference_corners) * metric_scale, axis=1).max())
                assert abs(error - entry['metrics']['corner_max_error_px']) < 1e-8
                counts[pool] += error <= 15
                lines = court_lines(corners, np.asarray(display_scale)) + display_offset
            description = describe(entry)
            identity = entry['id'] if entry else 'Empty pool'
            fitted = frame.copy()
            clip_rect = (*map(int, display_offset), *display_size)
            draw_lines(fitted, reference_lines, REFERENCE_COLOUR, True, clip_rect)
            draw_lines(fitted, lines, FIT_COLOUR, False, clip_rect)
            comparison.paste(fitted, (index * SIZE[0], 110))
            draw = ImageDraw.Draw(comparison)
            left = index * SIZE[0] + 16
            draw.text((left, 10), title, fill='black', font=FONT)
            draw.text((left, 43), description, fill='black', font=SMALL_FONT)
            draw.text((left, 72), identity, fill='#444444', font=SMALL_FONT)
            clip_id = f'{identifier}_{pool}_clip'
            svg = (f'<svg viewBox="0 0 1280 720" role="img" aria-label="{html.escape(title)} court overlay">'
                   f'<defs><clipPath id="{clip_id}"><rect x="{display_offset[0]}" y="{display_offset[1]}" '
                   f'width="{display_size[0]}" height="{display_size[1]}"/></clipPath></defs>'
                   f'<image href="{identifier}__frame.jpg" width="1280" height="720"/>'
                   f'<g clip-path="url(#{clip_id})">'
                   + svg_lines(reference_lines, 'reference', REFERENCE_COLOUR)
                   + svg_lines(lines, 'fit', FIT_COLOUR) + '</g></svg>')
            panels.append(f'<div class="panel"><h3>{title}</h3><p>{description}<br><small>{identity}</small></p>{svg}</div>')
            records[pool] = None if entry is None else {
                'id': entry['id'], 'corners_px': entry['corners_px'], 'metrics': entry['metrics'],
            }
        comparison.save(output / f'{identifier}__comparison.jpg', quality=94)
        sections.append(f'<section id="{identifier}"><h2>{identifier}</h2><div class="pair">'
                        + ''.join(panels) + f'</div><p><a href="{identifier}__comparison.jpg">Full-size comparison image</a>'
                        f' · <a href="{identifier}__frame.jpg">Unmarked frame</a></p></section>')
        markdown.append(f'## {identifier}\n\n![Before and latest fitted courts]({identifier}__comparison.jpg)\n')
        manifest.append({'id': identifier, 'native_size': native_size, 'display_size': SIZE,
                         'source_image_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'display_scale': display_scale, 'display_offset_px': display_offset.tolist(),
                         'reference_corners_px': reference['corners_px'], 'picks': records})
    assert counts == {'starts': 9, 'fixed_position': 10}, counts
    intro = ('Before: 9/20; latest: 10/20. Both use the complete-junction-agreement ranking. '
             'Magenta solid lines: selected fit. Blue dashed lines: manual reference. '
             'Errors use 1280×720 pixels and include off-screen corners. Images preserve their aspect ratio and show the visible frame only. '
             'These are the same 20 development frames; no new fitting or ranking was performed. '
             'The empty pool has reference lines only. A /start winner means the original fit was retained.')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Latest fitted court overlays</title>
<style>body{font:16px system-ui;margin:24px;background:#fafafa;color:#222}header{max-width:1050px}
.controls{position:sticky;top:0;background:#fff;padding:12px;z-index:1;border-bottom:1px solid #ccc}
label{margin-right:24px}section{margin:36px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.panel{min-width:0}h3{margin-bottom:8px}svg{display:block;width:100%;background:#ddd}small{color:#555}
.hide-reference .reference,.hide-fit .fit{display:none}body.large .pair{grid-template-columns:1fr}
@media(max-width:900px){.pair{grid-template-columns:1fr}}</style>
<header><h1>Latest fitted court overlays</h1><p>INTRO</p><p>Use full-width panels or browser zoom for paint alignment.</p></header>
<div class="controls"><label><input type="checkbox" checked onchange="document.body.classList.toggle('hide-reference',!this.checked)"> Reference</label>
<label><input type="checkbox" checked onchange="document.body.classList.toggle('hide-fit',!this.checked)"> Selected fit</label>
<label><input type="checkbox" onchange="document.body.classList.toggle('large',this.checked)"> Full-width panels</label></div>
CONTENT</html>'''
    (output / 'index.html').write_text(document.replace('INTRO', html.escape(intro)).replace('CONTENT', ''.join(sections)))
    (output / 'README.md').write_text('# Latest fitted court overlays\n\n' + intro
                                    + '\n\n[Interactive gallery](index.html): toggle reference/fit lines and enlarge panels.\n\n'
                                    + '\n'.join(markdown))
    evidence = {'source': 'stripe_diagnostics.zip:refit_selection_v2.json.gz', 'selector': SELECTOR,
                'accurate_picks': counts, 'records': manifest}
    (output / 'manifest.json.gz').write_bytes(gzip.compress(json.dumps(evidence, allow_nan=False).encode(), mtime=0))
    print(f'Rendered {len(manifest)} frames; verified accurate selections {counts}')


if __name__ == '__main__':
    main()
