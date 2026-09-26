"""Present automatic marking alternatives alongside the previously inspected courts."""

from __future__ import annotations

import argparse
import html
import importlib
import re
import sys
from pathlib import Path

import numpy as np
from run_diagnosis import read
from summarise import winners

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'vp_pruning'))
from render_viewer import CASES, CHECKS, panel


def case_title(case_id: str) -> str:
    if case_id.startswith('gxBQ'):
        return f"GX — frame {case_id.rsplit('_', 1)[1]}"
    if case_id.startswith('shuttleset'):
        _, video, _, scene = case_id.split('_')
        return f'ShuttleSet {video} — scene {int(scene)}'
    return f"Amateur-{case_id.split('_')[0][2:]} — frame {case_id.rsplit('_', 1)[1]}"


def description(entry: dict, fitted: bool = False) -> str:
    gates = entry['gates']
    floor = 'passes' if gates['floor_score'] >= 0 else 'fails'
    text = f"Error {entry['max_corner_1280_px']:.2f} px; {floor} the original floor gate."
    if fitted:
        text += ' Automatic shortlist and fixed-position fit; visual quality is unjudged.'
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(CHECKS / 'player_guided/20260913/seed_selection'))
    template = importlib.import_module('render_visual_check')
    packs = [read(CHECKS / 'player_guided/20260909/gx_extension/inputs.json.gz'),
             read(CHECKS / 'player_guided/20260909/broadcast_extension/inputs.json.gz'),
             read(CHECKS / 'player_guided/20260908/marking_refit/marking_inputs.json.gz')]
    sources = {source['id']: source for pack in packs for source in pack['cases']}
    references = {case_id: reference for pack in packs for case_id, reference in pack['references'].items()}
    header = re.sub(r'<header>.*?</header>',
                    '<header><h1>Marking-support follow-up</h1>'
                    '<p>Left: the previously inspected top retained court, or closest generated court '
                    'where none was retained. Right: an automatically chosen court from the new marking '
                    'shortlist, after fitting its fixed marking assignments.</p>'
                    '<p>Inspect first: <a href="#shuttleset_03_scene_0017">ShuttleSet 03 scene 17</a> · '
                    '<a href="#shuttleset_03_scene_0019">scene 19</a> · '
                    '<a href="#shuttleset_03_scene_0016">scene 16</a>. '
                    'Do the new far service and back lines follow the paint? '
                    'Do the sidelines and near lines remain usable?</p>'
                    '<p>Amateur ranking regressions are retained below as failure controls. '
                    'Reference corner error does not establish visual quality. '
                    'New selection uses image fragments and player/camera checks, without reference labels. '
                    'No new acceptance rule is applied. ShuttleSet images are cached median composites.</p>'
                    '<p>Magenta: generated court. Optional blue: manual reference. Errors use a 1280×720 display. '
                    'Expand the details beneath a pair to compare the new winning start with its own refit.</p></header>',
                    template.document_header(), flags=re.DOTALL).replace('Frame-5 court usability check',
                                                                          'Marking-support follow-up')
    header = header.replace('</style>', '.panel {display:flex;flex-direction:column} '
                            '.panel svg {margin-top:auto}</style>')
    sections = []
    ordered = [case_id for case_id in CASES if case_id.startswith('shuttleset')]
    ordered += [case_id for case_id in CASES if not case_id.startswith('shuttleset')]
    for case_id in ordered:
        source, reference = sources[case_id], references[case_id]
        previous = read(args.results.parent / 'results' / f'{case_id}.json.gz')
        old = next((entry for entry in previous['entries'] if entry['id'] == 'retained_0'), None)
        old_title = 'Previously inspected top retained court'
        if old is None:
            old = next(entry for entry in previous['entries'] if entry.get('inspected_closest'))
            old_title = 'Previously inspected closest court (reference-selected)'
        record = read(args.results / f'{case_id}.json.gz')
        selected = winners(record)
        fit, parent, start = selected['fit'], selected['fit_parent'], selected['start']
        left = panel(old_title, description(old), np.asarray(old['corners_px']), source, reference, args.output)
        right = panel('New automatic fixed-position refit', 'No eligible successful fit.' if fit is None else
                      description(fit, True), None if fit is None else np.asarray(fit['corners_px']),
                      source, reference, args.output)
        details = ''
        if fit is not None:
            before = panel('Start of the new winning refit', f"{parent['id']}. " + description(parent),
                           np.asarray(parent['corners_px']), source, reference, args.output)
            after = panel('Same start after fixed-assignment fitting', description(fit, True),
                          np.asarray(fit['corners_px']), source, reference, args.output)
            winner_note = 'No eligible automatic start.' if start is None else (
                f"The separately ranked best unfitted start is {start['id']} "
                f"(error {start['max_corner_1280_px']:.2f} px).")
            details = '<details><summary>New start versus its own refit</summary><p>' + html.escape(winner_note)
            details += f'</p><div class="pair">{before}{after}</div></details>'
        sections.append(f'<section id="{case_id}"><h2>{html.escape(case_title(case_id))}</h2>'
                        f'<div class="pair">{left}{right}</div>{details}</section>')
    args.output.write_text(header + ''.join(sections) + '</body></html>\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
