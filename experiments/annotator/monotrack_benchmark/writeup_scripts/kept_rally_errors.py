"""Count the fixes that kept rallies need, at the two confidence cut-offs in the project README.

Reads the found rallies that ``metrics.py`` judged in discovery mode, keeps those at or above each
cut-off, and counts the missed, extra and wrong-player hits in every kept rally that is not fully
correct. The README's cut-offs were read off the test curve, so these numbers describe test
performance at those points, not a setting chosen on validation.

Usage: python kept_rally_errors.py --data DIR --out DIR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Bound inline: ruff's E402 allows sys.path calls above imports, but not assignments.
sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    write_json,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import SEEDS

README_CUTOFFS = (0.675, 0.5)
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
        'single_error_share': len(single) / len(needs_fix),
        'two_or_fewer_errors_share': int((errors <= 2).sum()) / len(needs_fix),
        'single_error_type_shares': {column: float((single[column] == 1).mean()) for column in ERROR_COLUMNS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='metrics.py output with found_rallies_*_seed*.csv.gz')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    found = pd.concat([pd.read_csv(args.data / f'found_rallies_annotator_discovery_seed{seed}.csv.gz').assign(seed=seed)
                       for seed in SEEDS])
    result = {f'{cutoff:g}': fix_summary(found, cutoff) for cutoff in README_CUTOFFS}
    write_json(args.out / 'kept_rally_errors.json.gz', result)
    for cutoff, summary in result.items():
        types = ', '.join(f'{column} {100 * share:.1f}%' for column, share in summary['single_error_type_shares'].items())
        print(f'{cutoff}: kept {summary["kept_per_seed"]:.0f}, need a fix {summary["needs_fix_per_seed"]:.0f}, '
              f'single error {100 * summary["single_error_share"]:.1f}% ({types}), '
              f'two or fewer {100 * summary["two_or_fewer_errors_share"]:.1f}%')


if __name__ == '__main__':
    main()
