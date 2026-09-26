"""Estimate the cascade's saving before and after the upright-camera filter, from each pair's court count.

Cost model, in units of fully scoring one court: a pair with n usable courts costs n today. With the
cascade it costs 0.43 n (the 16-sample cheap pass, measured in prefilter/summary.txt with the maps
rebuilt) plus K, once n exceeds K. On the pre-filter counts this model gives 50% at K = 2,048, against
the replay's measured 51%.

Court scoring's share of the run is scaled from the pre-filter figure of 30-33% of 6,434 s by the
drop in fully scored courts. That assumes its time follows its court count; nothing here is timed.

Usage: python cascade_cost.py > cascade_cost.txt  (from this folder)
"""
import glob
import json

import numpy as np

CHEAP_COST = 0.43
PRE_FILTER_RUN_S = 6434
PRE_FILTER_SCORING_SHARE = (0.30, 0.33)
POST_FILTER_RUN_S = 3703


def pair_counts(folder: str) -> np.ndarray:
    return np.array([pair['usable'] for path in sorted(glob.glob(f'{folder}/*.json'))
                     for pair in json.load(open(path))['pairs']], dtype=float)


counts = {folder: pair_counts(folder) for folder in ('unfiltered', 'filtered')}
for folder, n in counts.items():
    print(f'{folder}: {len(n)} pairs, {n.sum() / 1e6:.1f}M courts fully scored without the cascade')
    for k in (2048, 4096, 8192):
        cost = np.where(n > k, CHEAP_COST * n + k, n).sum() / n.sum()
        print(f'  K = {k}: cascade costs {cost:.0%} of court scoring; {np.mean(n > k):.0%} of pairs '
              f'exceed K and hold {n[n > k].sum() / n.sum():.0%} of courts')

n_after = counts['filtered']
court_ratio = n_after.sum() / counts['unfiltered'].sum()
cost_2048 = np.where(n_after > 2048, CHEAP_COST * n_after + 2048, n_after).sum() / n_after.sum()
for share in PRE_FILTER_SCORING_SHARE:
    scoring_s = share * PRE_FILTER_RUN_S * court_ratio
    saved_s = (1 - cost_2048) * scoring_s
    print(f'pre-filter scoring share {share:.0%}: with the filter, court scoring is about {scoring_s:.0f} s '
          f'({scoring_s / POST_FILTER_RUN_S:.0%} of {POST_FILTER_RUN_S} s); K = 2,048 saves about '
          f'{saved_s:.0f} s ({saved_s / POST_FILTER_RUN_S:.1%} of the run)')
