"""Verify original global candidate IDs survive the saved-coordinate round trip."""

from pathlib import Path

import numpy as np
from run_automatic import select_pool
from run_diagnosis import read

from experiments.annotator.independent_court import detector


def main() -> None:
    sources = {}
    for path in ['gx_extension/inputs.json.gz', 'gx_seed_20260913/marking_inputs.json.gz',
                 'broadcast_extension/inputs.json.gz']:
        sources.update({source['id']: source for source in read(Path(path))['cases']})
    checked = 0
    for path in sorted(Path('automatic_axes_20260914/results').glob('*.json.gz')):
        result = read(path)
        source = sources[result['case_id']]
        scale = np.array([source['dimensions']['width'], source['dimensions']['height']]) / result['working_size']
        candidates, provenance = [], {}
        for pair in result['pairs']:
            for entry in pair.get('shortlist', []):
                candidate = detector.Candidate(np.asarray(entry['corners_px']) / scale,
                                               entry['shortlist_score'], (0., 0.), (0, 0))
                candidates.append(candidate)
                provenance[id(candidate)] = entry['candidate_id']
        selected = [provenance[id(candidate)] for candidate in select_pool(candidates)]
        assert selected == [entry['candidate_id'] for entry in result['entries']], result['case_id']
        checked += 1
        print(result['case_id'], 'exact unfiltered selection replay passed', flush=True)
    assert checked == 9, checked


if __name__ == '__main__':
    main()
