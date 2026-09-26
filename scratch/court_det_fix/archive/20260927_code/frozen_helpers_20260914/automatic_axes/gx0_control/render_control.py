"""Compare the approved GX0 court with one observed-direction matcher control."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from render_automatic_check import _load_template, _sources
from render_viewer import panel
from run_diagnosis import read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnosis = read(args.records / 'diagnosis.json.gz')
    control = read(args.records / 'control.json.gz')
    sources, references = _sources()
    case_id = control['case_id']
    source, reference = sources[case_id], references[case_id]
    approved = diagnosis['approved_entry']
    left = panel('Previously approved GX0 court',
                 'Supplied-direction line winner 89, which you judged effectively perfect.',
                 np.asarray(approved['corners_px']), source, reference, args.output)
    panels = []
    shown = set()
    for ranking in ['line', 'paint']:
        candidate_id = control[f'{ranking}_winner_id']
        if candidate_id in shown or candidate_id is None:
            continue
        shown.add(candidate_id)
        entry = next(entry for entry in control['entries'] if entry['candidate_id'] == candidate_id)
        both = control['line_winner_id'] == control['paint_winner_id']
        title = 'New observed-direction control' if both else f'New observed-direction control: {ranking} winner'
        description = (f"Candidate {candidate_id}. {'Both rankings choose this court. ' if both else ''}"
                       'Observed directions were selected using the approved court; this is a label-guided control.')
        right = panel(title, description, np.asarray(entry['corners_px']), source, reference, args.output)
        panels.append(f'<div class="pair">{left}{right}</div>')
    header = ('<header><h1>GX frame 0: direction-precision control</h1>'
              '<p>Does the new court remove the shear while retaining the approved boundary alignment?</p>'
              '<p>Left: the previously approved court. Right: an actual proposal from the unchanged matcher, '
              'using a more precise pair of image-derived directions. The approved court helped select '
              'those directions, so this does not establish automatic detection.</p>'
              '<p>Magenta is the fitted court; blue is the optional manual reference. '
              'No annotation or acceptance rule changed.</p></header>')
    template = re.sub(r'<header>.*?</header>', header, _load_template(), flags=re.DOTALL)
    template = template.replace('Frame-5 court usability check', 'GX0 direction-precision control')
    template = template.replace('</style>', '.panel {display:flex;flex-direction:column} '
                                '.panel svg {margin-top:auto}</style>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template + f'<section id="gx0_control">{"".join(panels)}</section></body></html>\n')
    print(args.output)


if __name__ == '__main__':
    main()
