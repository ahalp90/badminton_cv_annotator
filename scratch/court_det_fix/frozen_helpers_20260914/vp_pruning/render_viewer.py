"""Reuse the existing court gallery for scored VP-pruned proposals."""

from __future__ import annotations

import argparse
import html
import importlib
import os
import re
import sys
from pathlib import Path

import numpy as np
from diagnose_targets import read

from experiments.annotator.independent_court.render_stripe_overlays import (
    court_lines,
    svg_lines,
)

ROOT = Path(__file__).resolve().parents[8]
CHECKS = ROOT / 'scratch/court_det_fix/worklog/checks/independent'
CASES = (
    'gxBQ_window_00_frame_0', 'gxBQ_window_00_frame_5',
    'am2_window_00_frame_150', 'am2_window_01_frame_28019', 'am3_window_00_frame_0',
    'shuttleset_03_scene_0017', 'shuttleset_03_scene_0019', 'shuttleset_03_scene_0016',
    'shuttleset_21_scene_0020',
)


def background(source: dict) -> Path:
    case_id = source['id']
    if case_id.startswith('gxBQ'):
        return CHECKS / 'player_guided/20260909/gx_extension/people' / source['image']
    if case_id.startswith('shuttleset'):
        return CHECKS / 'inputs/original' / source['image']
    video_id = case_id.split('_')[0]
    frame = int(case_id.rsplit('_', 1)[1])
    return CHECKS / f'inputs/amateur/{video_id}/frame_{frame:08d}.png'


def panel(title: str, description: str, corners: np.ndarray | None, source: dict, reference: dict, output: Path) -> str:
    """Draw direct SVG overlays over the original cached image."""
    dimensions = source['dimensions']
    scale = np.array([1280 / dimensions['width'], 720 / dimensions['height']])
    image = background(source)
    if not image.is_file():
        raise FileNotFoundError(image)
    href = html.escape(os.path.relpath(image, output.parent))
    manual = svg_lines(court_lines(np.asarray(reference['corners_px']), scale), 'reference', '#0072b2')
    fitted = '' if corners is None else svg_lines(court_lines(corners, scale), 'fit', '#cc00cc')
    return (
        f'<div class="panel"><h3>{html.escape(title)}</h3><p>{html.escape(description)}</p>'
        f'<svg viewBox="0 0 1280 720" role="img" aria-label="{html.escape(title)}">'
        f'<image href="{href}" width="1280" height="720"/>'
        f'<g class="evidence">{manual}{fitted}</g></svg></div>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    helper = CHECKS / 'player_guided/20260913/seed_selection'
    sys.path.insert(0, str(helper))
    viewer = importlib.import_module('render_visual_check')
    packs = [read(CHECKS / 'player_guided/20260909/gx_extension/inputs.json.gz'),
             read(CHECKS / 'player_guided/20260909/broadcast_extension/inputs.json.gz'),
             read(CHECKS / 'player_guided/20260908/marking_refit/marking_inputs.json.gz')]
    sources = {case['id']: case for pack in packs for case in pack['cases']}
    references = {case_id: reference for pack in packs for case_id, reference in pack['references'].items()}
    header = re.sub(
        r'<header>.*?</header>',
        '<header><h1>Courts from automatic vanishing-point pruning</h1>'
        '<p>Left: the top court retained by the unchanged detector, with its acceptance decision. '
        'Right: the closest generated court, chosen afterwards using the manual reference. '
        'The right panel diagnoses generation and may fail later gates.</p>'
        '<p>Does each new court follow the visible markings well enough? '
        'The recovered approved GX5 court keeps its earlier judgement. '
        'ShuttleSet panels use the existing median composite images.</p>'
        '<p>Candidate courts are magenta; optional manual references are blue. '
        'All errors are maximum corner errors at 1280×720. '
        'Production and acceptance gates are unchanged.</p></header>',
        viewer.document_header(), flags=re.DOTALL,
    ).replace('Frame-5 court usability check', 'VP-pruned court inspection')
    header = header.replace(
        '</header>',
        '<p>Inspect first: <a href="#gxBQ_window_00_frame_0">GX frame 0</a> · '
        '<a href="#shuttleset_03_scene_0016">ShuttleSet 03, scene 16</a> · '
        '<a href="#shuttleset_21_scene_0020">ShuttleSet 21, scene 20</a>.</p></header>',
    )
    sections = []
    for case_id in CASES:
        record = read(args.results / f'{case_id}.json.gz')
        source, reference = sources[case_id], references[case_id]
        final = record['final']
        decision = 'accepted' if final['accepted'] else f"not emitted ({final['reason']})"
        if final['candidates']:
            corners = np.asarray(final['candidates'][0]['corners_px'])
            description = f"{decision}; error {record['selected_max_corner_1280_px']:.2f} px."
        else:
            corners = None
            description = f'{decision}; no retained court.'
        left = panel('Top retained court', description, corners, source, reference, args.output)
        ranking = record['diagnostic_rankings']['geometry_before_players']['max_corner_native_px']
        if ranking:
            best = ranking[0]
            example = next(row for row in record['examples'] if row['generation_id'] == best['generation_id'])
            diagnostic = np.asarray(example['corners_px'])
            error = best['value'] * 1280 / source['dimensions']['width']
            description = f'Error {error:.2f} px. Reference-selected diagnostic; no new visual judgement yet.'
            if case_id == 'gxBQ_window_00_frame_5' and record['known_target']['geometry_valid']:
                same = np.allclose(diagnostic, record['known_target']['corners_px'], atol=1e-5, rtol=0)
                if same:
                    description = f'Error {error:.2f} px. Same geometry as the previously approved court.'
        else:
            diagnostic = None
            description = 'No geometry-valid generated court.'
        right = panel('Closest generated court', description, diagnostic, source, reference, args.output)
        if case_id.startswith('gxBQ'):
            title = f"GX — frame {case_id.rsplit('_', 1)[1]}"
        elif case_id.startswith('shuttleset'):
            _, video, _, scene = case_id.split('_')
            title = f'ShuttleSet {video} — scene {int(scene)}'
        else:
            video = case_id.split('_')[0][2:]
            title = f"Amateur-{video} — frame {case_id.rsplit('_', 1)[1]}"
        sections.append(
            f'<section id="{case_id}"><h2>{html.escape(title)}</h2>'
            f'<div class="pair">{left}{right}</div></section>'
        )
    args.output.write_text(header + ''.join(sections) + '</body></html>\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
