"""Reuse the court gallery for the given-direction spacing and paint comparison."""

from __future__ import annotations

import argparse
import html
import importlib
import re
import sys
from pathlib import Path

import numpy as np
from render_followup import case_title
from render_viewer import CHECKS, panel
from run_diagnosis import read


def describe(entry: dict) -> str:
    floor = 'passes' if entry['gates']['floor_score'] >= 0 else 'fails'
    return (f"Candidate {entry['candidate_id']}; reference corner error "
            f"{entry['reference_max_corner_1280_px']:.2f} px. "
            f"Paint-profile score {entry['profile']['score']:.3f}; {floor} the original floor gate.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--appearance', type=Path, required=True)
    parser.add_argument('--given', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(CHECKS / 'player_guided/20260913/seed_selection'))
    template = importlib.import_module('render_visual_check')
    packs = [read(CHECKS / 'player_guided/20260909/gx_extension/inputs.json.gz'),
             read(CHECKS / 'player_guided/20260909/broadcast_extension/inputs.json.gz'),
             read(CHECKS / 'player_guided/20260908/marking_refit/marking_inputs.json.gz')]
    sources = {source['id']: source for pack in packs for source in pack['cases']}
    references = {case_id: reference for pack in packs for case_id, reference in pack['references'].items()}
    records = {row['case_id']: row for row in read(args.appearance)['records']}
    heading = (
        '<header><h1>Court spacing and paint check — supplied directions</h1>'
        '<p><strong>This is a geometry diagnostic, not an automatic detector result.</strong> '
        'Court directions come from a previously approved fit or the manual reference. '
        'The new matcher then chooses line positions from image fragments; the paint check ranks the surviving courts. '
        'No new refitting is applied.</p>'
        '<p>Left: the court supplying those directions. Right: the new paint-ranked spacing match. '
        'Inspect the right-hand courts, especially their far horizontals and outer boundaries. '
        'Small corner errors and strong paint profiles do not establish a correct fit.</p>'
        '<p>Start with <a href="#am2_window_00_frame_150">Amateur-2 frame 150</a> · '
        '<a href="#am2_window_01_frame_28019">frame 28019</a> · '
        '<a href="#gxBQ_window_00_frame_0">GX0</a> · '
        '<a href="#shuttleset_03_scene_0017">ShuttleSet 03 scene 17</a>. '
        'GX0 moves farther from its reference under paint ranking. Scene17 remains uncertain. '
        'The three previously approved ShuttleSet views are included as controls.</p>'
        '<p>Expand “Effect of the paint check” to compare the line-score winner with the paint-ranked winner '
        'from exactly the same candidate pool. Magenta: court; optional blue: manual reference. '
        'Errors use a 1280×720 display. ShuttleSet backgrounds are cached median composites.</p></header>'
    )
    header = re.sub(r'<header>.*?</header>', heading, template.document_header(), flags=re.DOTALL)
    header = header.replace('Frame-5 court usability check', 'Given-direction spacing and paint check')
    header = header.replace('</style>', '.panel {display:flex;flex-direction:column} '
                            '.panel svg {margin-top:auto}</style>')
    ordered = ['am2_window_00_frame_150', 'am2_window_01_frame_28019', 'am3_window_00_frame_0',
               'gxBQ_window_00_frame_0', 'gxBQ_window_00_frame_5', 'shuttleset_03_scene_0017',
               'shuttleset_03_scene_0019', 'shuttleset_03_scene_0016', 'shuttleset_21_scene_0020']
    sections = []
    for case_id in ordered:
        record = records[case_id]
        given = read(args.given / f'{case_id}.json.gz')
        source, reference = sources[case_id], references[case_id]
        entries = {entry['candidate_id']: entry for entry in record['entries']}
        ridge, stripe = entries[record['ridge_winner_id']], entries[record['stripe_winner_id']]
        manual = given['given_direction_source'] == 'manual_reference_directions_only'
        control_title = 'Manual reference supplying directions' if manual else 'Previously approved court supplying directions'
        left = panel(control_title, 'Only its two directions constrain the new candidate search.',
                     np.asarray(given['control_corners_px']), source, reference, args.output)
        right = panel('New paint-ranked spacing match — given directions', describe(ridge),
                      np.asarray(ridge['corners_px']), source, reference, args.output)
        before = panel('Line-score winner from the same pool', describe(stripe),
                       np.asarray(stripe['corners_px']), source, reference, args.output)
        after = panel('Paint-ranked winner from the same pool', describe(ridge),
                      np.asarray(ridge['corners_px']), source, reference, args.output)
        details = f'<details><summary>Effect of the paint check</summary><div class="pair">{before}{after}</div></details>'
        sections.append(f'<section id="{case_id}"><h2>{html.escape(case_title(case_id))}</h2>'
                        f'<div class="pair">{left}{right}</div>{details}</section>')
    args.output.write_text(header + ''.join(sections) + '</body></html>\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
