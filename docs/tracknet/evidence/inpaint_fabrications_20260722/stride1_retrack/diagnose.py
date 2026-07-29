"""Stress-test the '375 -> 0 zones' result before believing it.

Three ways the headline could be misleading:
  (a) weight-mode ensembling smears the loop so the exact-equality census goes
      blind, while the fills are still there
  (b) the flipped-contact check's periodic_at has no ptp>0 guard, so it fires on
      constant runs (including all-zero no-detection spans), not just fills
  (c) fills might survive but shorter than the census's 32-index floor
"""
import csv

import numpy as np

BASE = ('/home/ariel/Documents/COSC594/badminton_stroke_classification'
        '/local_scratch/autograder_architecture')
BASELINE = f'{BASE}/pilot_track_npy/1.npy'
WEIGHT = f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/pilot_weight.npy'
BISECT = f'{BASE}/inpaint_fabrications_investigation/c11_landing_bisect/bisect_per_rally.csv'

base = np.load(BASELINE)
wt = np.load(WEIGHT)


def zero_structure(track, label):
    """(0,0) frames are explicit no-detection; fills are non-zero inventions."""
    is_zero = (track[:, 0] == 0) & (track[:, 1] == 0)
    runs, start = [], None
    for i, z in enumerate(is_zero):
        if z and start is None:
            start = i
        elif not z and start is not None:
            runs.append(i - start)
            start = None
    if start is not None:
        runs.append(len(is_zero) - start)
    runs = np.array(runs) if runs else np.array([0])
    print(f'\n--- zero structure: {label} ---')
    print(f'frames exactly (0,0): {is_zero.sum()} ({is_zero.mean():.4f})')
    print(f'zero runs: {len(runs)}  longest {runs.max()}  '
          f'median {np.median(runs):.0f}  >=48-long {(runs >= 48).sum()}')
    return is_zero


bz = zero_structure(base, 'nonoverlap baseline')
wz = zero_structure(wt, 'weight mode')

# --- (c) does ANY varying 16-periodicity survive in weight mode? sweep the floor
print('\n--- relaxed census on weight mode (vary the run floor) ---')
for track, label in ((base, 'baseline'), (wt, 'weight')):
    y = track[:, 1]
    eq = y[:len(y) - 16] == y[16:]
    for floor in (32, 16, 8, 4):
        runs, start = [], None
        for i, m in enumerate(eq):
            if m and start is None:
                start = i
            elif not m and start is not None:
                if i - start >= floor:
                    runs.append((start, i))
                start = None
        osc = [(a, b) for a, b in runs if np.ptp(y[a:a + 16]) > 0]
        print(f'  {label:9s} floor>={floor:2d}: varying-periodic zones = {len(osc)}')

# --- (a) the InpaintNet prior fingerprint: do the baseline's 16 cycle y-values
#         still appear anywhere in the weight track?
ybase = base[:, 1]
eq = ybase[:len(ybase) - 16] == ybase[16:]
runs, start = [], None
for i, m in enumerate(eq):
    if m and start is None:
        start = i
    elif not m and start is not None:
        if i - start >= 32:
            runs.append((start, i))
        start = None
osc = [(a, b) for a, b in runs if np.ptp(ybase[a:a + 16]) > 0]
fingerprint = np.unique(ybase[osc[0][0]:osc[0][0] + 16])
print(f'\n--- InpaintNet prior fingerprint ({len(fingerprint)} distinct y values) ---')
print('values:', np.round(fingerprint, 4))
for track, label in ((base, 'baseline'), (wt, 'weight')):
    hit = np.isin(track[:, 1], fingerprint)
    print(f'  {label:9s}: frames whose y is one of the fingerprint values = '
          f'{hit.sum()} ({hit.mean():.4f})')

# --- (b) classify what periodic_at actually fires on, per flipped rally
def classify(track, label):
    y = track[:, 1]

    def span_at(f, span=32):
        seg = y[f:f + span + 16]
        if len(seg) != span + 16:
            return None
        if not np.array_equal(seg[:span], seg[16:span + 16]):
            return None
        window = seg[:16]
        if np.ptp(window) == 0:
            return 'constant-zero' if window[0] == 0 else 'constant-nonzero'
        return 'varying (true fill loop)'

    with open(BISECT) as fh:
        rows = list(csv.DictReader(fh))
    flipped = [r for r in rows if r['landing_pool'] and not r['landing_sticky']]
    kinds = {}
    for r in flipped:
        found = None
        for off in (-2, -1, 0, 1, 2):
            k = span_at(int(r['final_contact']) + off)
            if k:
                found = k
                break
        if found:
            kinds[found] = kinds.get(found, 0) + 1
    print(f'\n--- what the flipped-contact hits actually are: {label} ---')
    print(f'  flipped rallies: {len(flipped)}')
    for k, v in sorted(kinds.items()):
        print(f'  {k}: {v}')
    print(f'  no periodic span: {len(flipped) - sum(kinds.values())}')


classify(base, 'nonoverlap baseline')
classify(wt, 'weight mode')
