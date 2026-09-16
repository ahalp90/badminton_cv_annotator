"""Summarise separate automatic and reference-selected marking experiments."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from run_diagnosis import read, write


def eligible(entry: dict) -> bool:
    gates = entry.get('gates', {})
    camera = gates.get('camera_error')
    one, two = gates.get('player_fractions', [0, 0])
    return bool(gates.get('geometry_valid') and one == 1 and two >= .5 and camera is not None and camera <= .1)


def winners(record: dict) -> dict:
    """Rank original starts and fixed-position fits separately, without reference metrics."""
    if any(entry['pool'] != 'automatic_continuous_shortlist' for entry in record['entries']):
        raise ValueError('Automatic ranking received a different candidate pool')
    starts = [entry for entry in record['entries'] if eligible(entry)]
    fitted = [(entry, entry['fits']['fixed_position']) for entry in record['entries']
              if entry['fits']['fixed_position']['successful'] and eligible(entry['fits']['fixed_position'])]
    start = max(starts, key=lambda entry: entry['stripe']['exclusive']['score'], default=None)
    best_fit = max(fitted, key=lambda pair: pair[1]['stripe_fixed']['exclusive']['score'], default=None)
    return {'start': start, 'fit_parent': None if best_fit is None else best_fit[0],
            'fit': None if best_fit is None else best_fit[1]}


def brief(entry: dict | None, parent: dict | None = None) -> dict | None:
    if entry is None:
        return None
    stripe = entry.get('stripe_fixed', entry.get('stripe'))
    return {'id': (parent or entry)['id'], 'max_corner_1280_px': entry['max_corner_1280_px'],
            'stripe_score': stripe['exclusive']['score'], 'coverage': stripe['exclusive_per_marking'],
            'gates': entry['gates'], 'corners_px': entry['corners_px']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = []
    total: Counter = Counter()
    for path in sorted(args.results.glob('*.json.gz')):
        if path.name.endswith('.shortlist.json.gz'):
            continue
        record = read(path)
        chosen = winners(record)
        statuses = Counter(fit['status'] for entry in record['entries'] for fit in entry['fits'].values())
        total.update(statuses)
        total['starts'] += len(record['entries'])
        result = {'case_id': record['case_id'], 'entries': len(record['entries']), 'statuses': dict(statuses),
                  'start': brief(chosen['start']), 'fit': brief(chosen['fit'], chosen['fit_parent']),
                  'shortlist': record.get('shortlist')}
        records.append(result)
        print(record['case_id'], 'start', None if result['start'] is None else
              (result['start']['id'], round(result['start']['max_corner_1280_px'], 2)),
              'fit', None if result['fit'] is None else
              (result['fit']['id'], round(result['fit']['max_corner_1280_px'], 2)),
              'coverage', None if result['fit'] is None else np.round(result['fit']['coverage'], 2), flush=True)
    write(args.output, {'records': records, 'totals': dict(total), 'reference_used_for_ranking': False})
    print(dict(total))


if __name__ == '__main__':
    main()
