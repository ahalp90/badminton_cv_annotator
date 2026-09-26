"""Static gallery: B, M and R line and paint winners from the all-camera pools over the cached frames.

Reads the committed E4 diagnosis and E3 summaries plus the local input packs. Nothing is
recomputed; every error shown comes from the saved diagnosis. Layers toggle through CSS
``:has`` selectors, so the page works with JavaScript disabled.
"""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

import cv2
import numpy as np
from common import CASE_IDS, read

from experiments.annotator.independent_court.render_stripe_overlays import court_lines

REPO = Path(__file__).resolve().parents[3]
CHECKS = REPO / 'scratch/court_det_fix/evidence/independent_proposals/development'
PACKS = (
    CHECKS / 'player_guided/20260909/gx_extension/inputs.json.gz',
    CHECKS / 'player_guided/20260909/broadcast_extension/inputs.json.gz',
    CHECKS / 'player_guided/20260908/marking_refit/marking_inputs.json.gz',
)
BASELINE_ALL_CAMERA = CHECKS / 'player_guided/20260914/automatic_axes/collected/all_camera'
DISPLAY = np.array([1280., 720.])
ARM_COLOURS = {'B': '#d55e00', 'M': '#009e73', 'R': '#cc79a7'}
ARM_LABELS = {'B': 'B: baseline (foot anchor, greedy leader)', 'M': 'M: projected-midpoint anchor',
              'R': 'R: precision representative'}
CONTROL_COLOUR = '#0072b2'
REFERENCE_COLOUR = '#666666'
CORNER_MATCH_ATOL = 1e-6
JUDGEMENTS_PATH = REPO / ('experiments/annotator/independent_court/recorded/player_guided/projective_patterns/'
                          'automatic_axes_visual_judgements.md')
# The user's earlier rulings on the baseline all-camera winners, keyed by candidate ID so a
# ruling is shown only for the exact court it was made on. Phrases follow the judgement file.
PRIOR_JUDGEMENTS = {
    ('gxBQ_window_00_frame_0', 'line'): ('22:4579', 'not really usable; sheared'),
    ('gxBQ_window_00_frame_0', 'paint'): ('22:4588', 'worse than the line winner; same shear, wrong bottom-right corner'),
    ('gxBQ_window_00_frame_5', 'line'): ('181:29836', 'rejected; draws a court on the wall'),
    ('gxBQ_window_00_frame_5', 'paint'): ('10:1274', 'rejected; draws a court over the seated children'),
    ('am2_window_00_frame_150', 'line'): ('30:30', 'mistakes the blue mat boundary for the court'),
    ('am2_window_00_frame_150', 'paint'): ('30:33', 'described as perfect'),
    ('am2_window_01_frame_28019', 'line'): ('16:1800', 'extends to the mat borders and is sheared; not adequate'),
    ('am2_window_01_frame_28019', 'paint'): ('184:4123', 'rejected as an incoherent court near the net top'),
    ('am3_window_00_frame_0', 'line'): ('43:22603', 'essentially perfect'),
    ('am3_window_00_frame_0', 'paint'): ('43:22627', 'very usable'),
    ('shuttleset_03_scene_0017', 'line'): ('1:80', 'very usable with slight skew'),
    ('shuttleset_03_scene_0017', 'paint'): ('1:132', 'similar but worse; far baseline overshoots'),
    ('shuttleset_03_scene_0019', 'line'): ('1:60', 'usable'),
    ('shuttleset_03_scene_0019', 'paint'): ('165:6702', 'rejected as an unrelated, hallucinated court'),
    ('shuttleset_03_scene_0016', 'line'): ('1:60', 'very usable'),
    ('shuttleset_03_scene_0016', 'paint'): ('1:90', 'usable'),
    ('shuttleset_21_scene_0020', 'line'): ('0:2', 'very usable'),
    ('shuttleset_21_scene_0020', 'paint'): ('0:2', 'very usable'),
}
STYLE = """body{font:16px system-ui;margin:24px;background:#fafafa;color:#222}header{max-width:1050px}
.controls{position:sticky;top:0;background:#fff;padding:12px;z-index:1;border-bottom:1px solid #ccc}
label{margin-right:20px;white-space:nowrap}section{margin:36px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.panel{min-width:0}h3{margin-bottom:8px}svg{display:block;width:100%;background:#ddd}small{color:#555}
.swatch{display:inline-block;width:28px;height:0;border-top:4px solid;vertical-align:middle;margin-right:6px}
table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}th:first-child,td:first-child{text-align:left}
details{margin-top:12px}summary{cursor:pointer}
body:has(#image:not(:checked)) svg image{display:none}
body:has(#control:not(:checked)) .control{display:none}body:has(#reference:not(:checked)) .reference{display:none}
.court{display:none}
body:has(#armB:checked) .armB,body:has(#armM:checked) .armM,body:has(#armR:checked) .armR{display:inline}
@media(max-width:900px){.pair{grid-template-columns:1fr}}"""


def background(source: dict) -> Path:
    """Cached frame for the case, following the earlier gallery's layout."""
    case_id = source['id']
    if case_id.startswith('gxBQ'):
        return CHECKS / 'player_guided/20260909/gx_extension/people' / source['image']
    if case_id.startswith('shuttleset'):
        return CHECKS / 'inputs/original' / source['image']
    video_id = case_id.split('_')[0]
    frame = int(case_id.rsplit('_', 1)[1])
    return CHECKS / f'inputs/amateur/{video_id}/frame_{frame:08d}.png'


def case_title(case_id: str) -> str:
    if case_id.startswith('gxBQ'):
        return f"GX, frame {case_id.rsplit('_', 1)[1]}"
    if case_id.startswith('shuttleset'):
        _, video, _, scene = case_id.split('_')
        return f'ShuttleSet {video}, scene {int(scene)}'
    return f"Amateur-{case_id.split('_')[0][2:]}, frame {case_id.rsplit('_', 1)[1]}"


def court_layer(corners_native: np.ndarray, scale: np.ndarray, css_class: str, colour: str, dashed: bool) -> str:
    dash = ' stroke-dasharray="7 5"' if dashed else ''
    width = 1.3 if dashed else 1.8
    parts = [f'<g class="{css_class}" fill="none" stroke="{colour}" stroke-width="{width}"{dash}>']
    for start, end in court_lines(np.asarray(corners_native, dtype=float), scale):
        parts.append(f'<path d="M {start[0]:.3f} {start[1]:.3f} L {end[0]:.3f} {end[1]:.3f}"/>')
    return ''.join(parts) + '</g>'


def panel(title: str, description: str, layers: str, image_href: str) -> str:
    return (f'<div class="panel"><h3>{html.escape(title)}</h3><p>{description}</p>'
            f'<svg viewBox="0 0 1280 720" role="img" aria-label="{html.escape(title)}">'
            f'<image href="{html.escape(image_href)}" width="1280" height="720"/>{layers}</svg></div>')


def distinct_winners(records: list[dict], case_id: str) -> tuple[list[dict], list[str]]:
    """All-camera winners of B, M and R, merged when two labels share one court.

    Candidate IDs are local to each arm, so a shared court keeps one ID per label and is
    drawn in the colour of the first arm that chose it.
    """
    courts: list[dict] = []
    notes: list[str] = []
    for arm in ARM_COLOURS:
        record = next((entry for entry in records if entry['case_id'] == case_id and entry['arm'] == arm
                       and entry['stage'] == 'all_camera'), None)
        if record is None:
            notes.append(f'{arm}: no all-camera record in the diagnosis.')
            continue
        for ranking in ('line', 'paint'):
            winner = record['diagnosis']['winners'][ranking]
            if winner is None:
                notes.append(f'{arm} {ranking}: no winner; none camera eligible with an available paint profile.')
                continue
            corners = np.asarray(winner['corners_px'], dtype=float)
            label = f'{arm} {ranking}'
            for court in courts:
                if np.allclose(court['corners'], corners, atol=CORNER_MATCH_ATOL, rtol=0):
                    court['ids'][label] = winner['candidate_id']
                    court['arms'].add(arm)
                    break
            else:
                courts.append({'arm': arm, 'arms': {arm}, 'ids': {label: winner['candidate_id']}, 'corners': corners,
                               'winner': winner})
    return courts, notes


def court_classes(court: dict) -> str:
    return 'court ' + ' '.join(f'arm{arm}' for arm in ARM_COLOURS if arm in court['arms'])


def prior_judgement(case_id: str, court: dict) -> str:
    """The earlier ruling on a baseline winner, only when the candidate ID is the one that was judged."""
    rulings = []
    for label, candidate_id in court['ids'].items():
        arm, ranking = label.split()
        if arm != 'B':
            continue
        judged_id, phrase = PRIOR_JUDGEMENTS[(case_id, ranking)]
        if judged_id == candidate_id:
            # A ruling made on B's court applies to the same court when another arm also chose it.
            others = [other for other in ARM_COLOURS if other in court['arms'] and other != 'B']
            shared = f' (the same court as chosen by {" and ".join(others)})' if others else ''
            rulings.append(f'B {ranking} winner judged earlier: {phrase}{shared}')
        else:
            rulings.append(f'B {ranking} winner {candidate_id} is not the judged candidate {judged_id}')
    if not rulings:
        return 'No visual judgement in this run.'
    return html.escape('; '.join(rulings)) + '. No new visual judgement in this run.'


def winner_description(case_id: str, court: dict) -> str:
    winner = court['winner']
    labels = ', '.join(f'{label} (candidate {candidate_id})' for label, candidate_id in court['ids'].items())
    gates = winner['gates']
    floor = 'passes' if gates['floor_score'] is not None and gates['floor_score'] >= 0 else 'fails'
    return (f'<b>{html.escape(labels)}</b>. Control error {winner["control_working"]:.2f} working px; '
            f'manual-reference error {winner["reference_display"]:.2f} px at 1280×720. '
            f'Line score {winner["stripe_score"]:.3f}; paint score {winner["profile_score"]:.3f}; '
            f'camera error {gates["camera_error"]:.4f}; {floor} the original floor gate. '
            + prior_judgement(case_id, court))


def control_description(control: dict) -> str:
    status = 'visually approved' if control['visually_approved'] else 'manual reference; no visual approval'
    return (f'Control: <code>{html.escape(control["control_source"])}</code> ({status}), from '
            f'<code>{html.escape(control["record"])}</code>.')


def accounting_table(records: list[dict], case_id: str) -> str:
    columns = ('arm', 'stage', 'final', 'final_camera_eligible', 'final_floor_pass', 'line_winner_id', 'paint_winner_id',
               'line_control_working_px', 'paint_control_working_px', 'nearest_pre_global_px', 'nearest_final_px',
               'nearest_final_camera_eligible_px')
    rows = [entry['accounting'] for entry in records if entry['case_id'] == case_id and entry.get('accounting')]
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            cells.append(f'{value:.2f}' if isinstance(value, float) else html.escape(str(value)))
        body.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
    header = ''.join(f'<th>{html.escape(column)}</th>' for column in columns)
    return f'<table><tr>{header}</tr>{"".join(body)}</table>'


def potential_table(records: list[dict], case_id: str) -> str:
    entry = next((entry for entry in records if entry['case_id'] == case_id
                  and entry['stage'] == 'e3_direction_fit_potential'), None)
    if entry is None or entry['potential'] is None:
        return '<p>No E3 record.</p>'
    cells = []
    for name, best in entry['potential'].items():
        value = 'skipped' if best is None else f"{best['max_corner_working_px']:.3f} (pair {best['pair_id']})"
        cells.append(f'<tr><td>{html.escape(name)}</td><td>{html.escape(value)}</td></tr>')
    return ('<p>E3 best finite ordered-pair control fit per set, working px. These are least-squares fits to the '
            'control, not generated courts.</p><table><tr><th>set</th><th>best fit</th></tr>' + ''.join(cells) + '</table>')


def header(run: str, output: Path) -> str:
    legend = ''.join(f'<span class="swatch" style="border-color:{colour}"></span>{html.escape(ARM_LABELS[arm])}<br>'
                     for arm, colour in ARM_COLOURS.items())
    judgements_href = html.escape(os.path.relpath(JUDGEMENTS_PATH, output.parent))
    return (f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Direction agreement: B, M and R winners</title><style>{STYLE}</style>'
            f'<header><h1>Direction agreement: B, M and R winners</h1>'
            f'<p>Run <code>{html.escape(run)}</code>. Each case shows the line-score and paint-ranked winners of the '
            'baseline (B) and the two individual changes (M, R) from the all-camera pools, over the cached frame. '
            'A court chosen by more than one label is drawn once. The first panel overlays every distinct winner; the '
            'following panels show them one at a time.</p>'
            f'<p>{legend}<span class="swatch" style="border-color:{CONTROL_COLOUR};border-top-style:dashed"></span>'
            'Frozen control (dashed blue), labelled with its exact source and approval status<br>'
            f'<span class="swatch" style="border-color:{REFERENCE_COLOUR};border-top-style:dashed"></span>'
            'Manual reference (dashed grey), shown only where it differs from the control</p>'
            '<p>Errors are maximum corner distances allowing the 180-degree relabelling, from the saved diagnosis. '
            'No court here is emitted or accepted, and none received a visual judgement in this run. Baseline (B) '
            f'winners carry the user\'s earlier ruling from <a href="{judgements_href}">the automatic-axes '
            'judgements</a>, shown only when the candidate ID is the one judged; M and R winners are unjudged, except '
            'where an M or R winner is the same court as a judged B winner, which the panel says. A court chosen by '
            'several arms is drawn once, in the first arm\'s colour, and stays visible while any of its arms is '
            'checked.</p></header>'
            '<div class="controls"><label><input id="image" type="checkbox" checked> Image</label>'
            '<label><input id="control" type="checkbox" checked> Control</label>'
            '<label><input id="reference" type="checkbox"> Manual reference</label>'
            '<label><input id="armB" type="checkbox" checked> B</label>'
            '<label><input id="armM" type="checkbox" checked> M</label>'
            '<label><input id="armR" type="checkbox" checked> R</label></div>')


def validate_joins(run_dir: Path, case_id: str, courts: list[dict]) -> list[str]:
    """Each winner must be the record's own winner with the same corners, or the join is broken."""
    problems = []
    for court in courts:
        for label, candidate_id in court['ids'].items():
            arm, ranking = label.split()
            path = (BASELINE_ALL_CAMERA / f'{case_id}.json.gz' if arm == 'B'
                    else run_dir / 'e4' / arm / 'all_camera' / f'{case_id}.json.gz')
            if not path.exists():
                problems.append(f'{case_id} {label}: record {path} missing locally; join unchecked')
                continue
            record = read(path)
            winner_id = record[f'{ranking}_winner_id']
            if winner_id != candidate_id:
                problems.append(f'{case_id} {label}: diagnosis winner {candidate_id} but record winner {winner_id}')
                continue
            entry = next(entry for entry in record['entries'] if entry['candidate_id'] == winner_id)
            if not np.allclose(entry['corners_px'], court['corners'], atol=CORNER_MATCH_ATOL, rtol=0):
                problems.append(f'{case_id} {label}: corners differ between diagnosis and record')
    return problems


def render(run_dir: Path, output: Path) -> None:
    diagnosis = read(run_dir / 'e4' / 'diagnosis.json.gz')
    records = diagnosis['records']
    sources = {}
    references = {}
    for pack in PACKS:
        packed = read(pack)
        sources.update({source['id']: source for source in packed['cases']})
        references.update(packed['references'])
    sections = []
    problems = []
    for case_id in CASE_IDS:
        source, reference = sources[case_id], references[case_id]
        case_records = [entry for entry in records if entry['case_id'] == case_id]
        if not case_records:
            sections.append(f'<section id="{case_id}"><h2>{html.escape(case_title(case_id))}</h2>'
                            '<p>No diagnosis record for this case in the run.</p></section>')
            problems.append(f'{case_id}: absent from the diagnosis')
            continue
        control = case_records[0]['control']
        image = background(source)
        frame = cv2.imread(str(image))
        if frame is None:
            raise FileNotFoundError(image)
        if frame.shape[:2] != (source['dimensions']['height'], source['dimensions']['width']):
            problems.append(f'{case_id}: image {image} is {frame.shape[1]}x{frame.shape[0]}, source says {source["dimensions"]}')
        scale = DISPLAY / np.array([source['dimensions']['width'], source['dimensions']['height']])
        href = os.path.relpath(image, output.parent)
        courts, notes = distinct_winners(records, case_id)
        problems.extend(validate_joins(run_dir, case_id, courts))
        control_corners = np.asarray(control['corners_native_px'])
        reference_corners = np.asarray(reference['corners_px'], dtype=float)
        fixed_layers = court_layer(control_corners, scale, 'control', CONTROL_COLOUR, True)
        if not np.allclose(control_corners, reference_corners, atol=1e-3, rtol=0):
            fixed_layers += court_layer(reference_corners, scale, 'reference', REFERENCE_COLOUR, True)
        overlay = ''.join(court_layer(court['corners'], scale, court_classes(court), ARM_COLOURS[court['arm']], False)
                          for court in courts)
        panels = [panel('All distinct winners', 'Overlay of every distinct B, M and R winner in the arm colours; a court '
                        'chosen by several arms takes the first arm\'s colour. '
                        + ' '.join(html.escape(note) for note in notes), fixed_layers + overlay, href)]
        for court in courts:
            layer = court_layer(court['corners'], scale, court_classes(court), ARM_COLOURS[court['arm']], False)
            panels.append(panel(', '.join(court['ids']), winner_description(case_id, court), fixed_layers + layer, href))
        details = (f'<details><summary>Diagnostic accounting for {html.escape(case_title(case_id))}</summary>'
                   f'{accounting_table(records, case_id)}{potential_table(records, case_id)}</details>')
        sections.append(f'<section id="{case_id}"><h2>{html.escape(case_title(case_id))}</h2>'
                        f'<p>{control_description(control)}</p><div class="pair">{"".join(panels)}</div>{details}</section>')
    output.write_text(header(diagnosis['run'], output) + ''.join(sections) + '</body></html>\n', encoding='utf-8')
    print('wrote', output, 'sections', len(sections))
    for problem in problems:
        print('PROBLEM', problem)
    if problems:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'visual_check.html')
    args = parser.parse_args()
    render(args.run_dir, args.output)


if __name__ == '__main__':
    main()
