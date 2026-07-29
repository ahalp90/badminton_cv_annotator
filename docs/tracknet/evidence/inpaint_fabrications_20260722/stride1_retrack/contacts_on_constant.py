"""Direct test: does each flipped rally's final contact land on the constant?

The earlier flat-span test needed a 32-frame flat run around the contact, so it
under-counts contacts sitting near a gap edge. This checks the frame itself.
"""
import csv
from collections import Counter

import numpy as np

BASE = ('/home/ariel/Documents/COSC594/badminton_stroke_classification'
        '/local_scratch/autograder_architecture')
OLD = np.load(f'{BASE}/pilot_track_npy/1.npy')
NEW = np.load(f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/pilot_weight.npy')
CONST = (0.474609375, 0.2465277777777778)

with open(f'{BASE}/inpaint_fabrications_investigation/c11_landing_bisect/bisect_per_rally.csv') as fh:
    rows = list(csv.DictReader(fh))
flipped = [r for r in rows if r['landing_pool'] and not r['landing_sticky']]
print(f'flipped rallies: {len(flipped)}')


def classify(track, f, const):
    x, y = track[f, 0], track[f, 1]
    if x == 0 and y == 0:
        return 'blank'
    if (x, y) == const:
        return 'ON THE CONSTANT'
    return 'other'


# old-track fabrication test: is the frame inside a repeating loop zone?
y_old = OLD[:, 1]
eq = y_old[:len(y_old) - 16] == y_old[16:]
runs, start = [], None
for i, m in enumerate(eq):
    if m and start is None:
        start = i
    elif not m and start is not None:
        if i - start >= 32:
            runs.append((start, i))
        start = None
osc = [(a, b) for a, b in runs if np.ptp(y_old[a:a + 16]) > 0]
old_fab = np.zeros(len(OLD), bool)
for a, b in osc:
    old_fab[a:min(b + 16, len(OLD))] = True

new_kinds, old_kinds, both = Counter(), Counter(), 0
for r in flipped:
    f = int(r['final_contact'])
    k = classify(NEW, f, CONST)
    new_kinds[k] += 1
    old_kinds['in a fake loop' if old_fab[f] else 'not in a fake loop'] += 1
    if old_fab[f] and k == 'ON THE CONSTANT':
        both += 1

print('\n--- OLD track, at the exact final-contact frame ---')
for k, v in old_kinds.most_common():
    print(f'  {k}: {v}/38')
print('\n--- NEW track, at the exact final-contact frame ---')
for k, v in new_kinds.most_common():
    print(f'  {k}: {v}/38')
print(f'\nfabricated in BOTH tracks (old loop -> new constant): {both}/38')

# widen to +/-2 frames, matching the brief's tolerance
wide = sum(1 for r in flipped
           if any(classify(NEW, int(r['final_contact']) + o, CONST)
                  == 'ON THE CONSTANT' for o in (-2, -1, 0, 1, 2)))
print(f'on the constant within +/-2 frames: {wide}/38')

# ---------------------------------------------------------------- sanity
print('\n--- could the constant be a real resting shuttle? ---')
print(f'constant sits at y={CONST[1]:.4f} of frame height, '
      f'x={CONST[0]:.4f} of width')
print(f'that is row {CONST[1] * 288:.0f} of 288, i.e. the upper middle of the '
      f'picture: mid-air, not the floor')
n_const = ((NEW[:, 0] == CONST[0]) & (NEW[:, 1] == CONST[1])).sum()
print(f'it holds {n_const} frames = {n_const / len(NEW):.4f} of the video '
      f'({n_const / 25 / 60:.1f} minutes at 25 fps)')
print('a shuttle cannot hover at one mid-air pixel for that long, so the '
      'constant is invented, not rest')
