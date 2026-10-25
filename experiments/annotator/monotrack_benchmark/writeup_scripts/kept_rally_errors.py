"""Count the fixes that kept rallies need at validation-selected confidence cut-offs.

Reads the found rallies that ``metrics.py`` judged in discovery mode, keeps those at or above each
cut-off, and counts the missed, extra and wrong-player hits in every kept rally that is not fully
correct. Reads the exact operating points from confidence_cutoffs.json.gz.

Usage: python kept_rally_errors.py --data DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Bound inline: ruff's E402 allows sys.path calls above imports, but not assignments.
sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    read_json,
    write_json,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import SEEDS

ERROR_COLUMNS = ['missed', 'extra', 'wrong_player']


def fix_summary(found: pd.DataFrame, cutoff: float) -> dict:
    """Kept rallies that need a fix, and how many errors each holds. Counts are means over the seeds."""
    kept = found[found['confidence'] >= cutoff]
    needs_fix = kept[kept['outcome'] != 'fully_correct']
    # Rallies with the wrong start or end have no per-hit counts, so they never count as a single fix.
    errors = needs_fix[ERROR_COLUMNS].sum(axis=1, min_count=1)
    single = needs_fix[errors == 1]
    return {
        'kept_per_seed': len(kept) / len(SEEDS),
        'needs_fix_per_seed': len(needs_fix) / len(SEEDS),
        'single_error_share': len(single) / len(needs_fix) if len(needs_fix) else None,
        'two_or_fewer_errors_share': int((errors <= 2).sum()) / len(needs_fix) if len(needs_fix) else None,
        'single_error_type_shares': {
            column: float((single[column] == 1).mean()) if len(single) else None for column in ERROR_COLUMNS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='metrics.py output with found_rallies_*_seed*.csv.gz')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    found = pd.concat([pd.read_csv(args.data / f'found_rallies_annotator_discovery_seed{seed}.csv.gz').assign(seed=seed)
                       for seed in SEEDS])
    cutoffs = read_json(args.data / 'confidence_cutoffs.json.gz')['cutoffs']
    result = {}
    for name, cutoff in cutoffs.items():
        result[name] = None if cutoff is None else {'cutoff': cutoff, **fix_summary(found, cutoff)}
    write_json(args.out / 'kept_rally_errors.json.gz', result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
