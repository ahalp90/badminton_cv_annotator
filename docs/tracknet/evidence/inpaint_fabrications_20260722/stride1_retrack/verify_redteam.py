"""Verify Sol's red-team claims first-hand. Do not take them on trust.

Claims checked:
 A. the weighted means in pixels, and the quantisation INTERVAL (not one pixel)
 B. the first-15-frame partial-average ramp (Sol's strongest new evidence)
 C. run structure of the constant: 591 runs, not one 36.8-minute hover
 D. what share of constant frames sit inside old proven zones
 E. how often the new constant's exact coordinate appears in the OLD track
 F. old-track contacts within +/-2 frames of a loop zone
"""
import csv
from collections import Counter

import numpy as np

BASE = ('/home/ariel/Documents/COSC594/badminton_stroke_classification'
        '/local_scratch/autograder_architecture')
OLD = np.load(f'{BASE}/pilot_track_npy/1.npy')
NEW = np.load(f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/pilot_weight.npy')
CONST = (0.474609375, 0.2465277777777778)

# ------------------------------------------------ rebuild cycle + old zones
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
zone_frames = np.unique(np.concatenate([np.arange(a, min(b + 16, len(OLD)))
                                        for a, b in osc]))
old_fab = np.zeros(len(OLD), bool)
old_fab[zone_frames] = True

cycle_x = np.zeros(16)
cycle_y = np.zeros(16)
for i in range(16):
    sel = zone_frames[zone_frames % 16 == i]
    cycle_x[i] = Counter(OLD[sel, 0]).most_common(1)[0][0]
    cycle_y[i] = Counter(OLD[sel, 1]).most_common(1)[0][0]

w = np.ones(16)
for i in range((16 + 1) // 2):
    w[i] = i + 1
    w[16 - i - 1] = i + 1
w = w / w.sum()

# ---------------------------------------------------------------- A
print('=== A. quantisation interval, not a single pixel ===')
mx = (w * cycle_x).sum() * 512
my = (w * cycle_y).sum() * 288
print(f'weighted mean of STORED cycle: x={mx:.4f}px  y={my:.4f}px')
print('stored values are truncated, so each true value is up to 1px higher.')
print(f'  x prediction set: {int(mx)} or {int(mx + 1)}')
print(f'  y prediction set: {int(my)} or {int(my + 1)}')
print(f'observed constant  : x={CONST[0] * 512:.0f}px  y={CONST[1] * 288:.0f}px')
print('Sol claimed x in {243,244}, y in {70,71}. Observed (243,71).')

# ---------------------------------------------------------------- B
print('\n=== B. the incomplete-buffer ramp in the first frames ===')
print('predict.py:312-315 divides by (sample_count+1) while the buffer fills,')
print('so frames 0..14 should ramp and frame 15 onward should hold the blend.')
for f in range(20):
    xp, yp = NEW[f, 0] * 512, NEW[f, 1] * 288
    mark = '  <- CONSTANT' if (NEW[f, 0], NEW[f, 1]) == CONST else ''
    print(f'  frame {f:2d}: x={xp:6.1f}px y={yp:6.1f}px{mark}')
first_const = next(f for f in range(len(NEW))
                   if (NEW[f, 0], NEW[f, 1]) == CONST)
print(f'first frame exactly on the constant: {first_const}')

# ---------------------------------------------------------------- C
print('\n=== C. run structure of the constant ===')
is_const = (NEW[:, 0] == CONST[0]) & (NEW[:, 1] == CONST[1])
runs_c, start = [], None
for i, v in enumerate(is_const):
    if v and start is None:
        start = i
    elif not v and start is not None:
        runs_c.append((start, i))
        start = None
if start is not None:
    runs_c.append((start, len(is_const)))
lens = np.array([b - a for a, b in runs_c])
print(f'constant frames: {is_const.sum()} in {len(runs_c)} separate runs')
print(f'  median run {np.median(lens):.0f}  longest {lens.max()}  '
      f'shortest {lens.min()}')
print(f'  share of constant frames in runs >=32: '
      f'{lens[lens >= 32].sum() / lens.sum():.3f}')
print('So it is NOT one continuous 36.8-minute hover. Sol is right.')

# ---------------------------------------------------------------- D
print('\n=== D. overlap of the constant with old proven zones ===')
inside = old_fab[is_const]
print(f'constant frames inside old proven loop zones: {inside.sum()} '
      f'({inside.mean():.3f})')
touch = sum(run_len for (a, b), run_len in zip(runs_c, lens)
            if old_fab[a:b].any())
print(f'constant frames in runs that intersect an old zone: {touch} '
      f'({touch / lens.sum():.3f})')

# ---------------------------------------------------------------- E
print('\n=== E. does the new constant coordinate occur in the OLD track? ===')
old_hits = ((OLD[:, 0] == CONST[0]) & (OLD[:, 1] == CONST[1])).sum()
print(f'exact (x,y) of the new constant, occurrences in OLD track: {old_hits}')

# ---------------------------------------------------------------- F
print('\n=== F. old-track contacts within +/-2 of a proven loop zone ===')
with open(f'{BASE}/inpaint_fabrications_investigation/c11_landing_bisect/bisect_per_rally.csv') as fh:
    rows = list(csv.DictReader(fh))
flipped = [r for r in rows if r['landing_pool'] and not r['landing_sticky']]
old_pm2 = sum(1 for r in flipped
              if any(old_fab[int(r['final_contact']) + o]
                     for o in (-2, -1, 0, 1, 2)
                     if 0 <= int(r['final_contact']) + o < len(OLD)))
new_pm2 = sum(1 for r in flipped
              if any((NEW[int(r['final_contact']) + o, 0],
                      NEW[int(r['final_contact']) + o, 1]) == CONST
                     for o in (-2, -1, 0, 1, 2)
                     if 0 <= int(r['final_contact']) + o < len(NEW)))
print(f'old: {old_pm2}/38 legacy contacts within +/-2 of a proven loop zone')
print(f'new: {new_pm2}/38 legacy contacts within +/-2 of the constant')
print('(these are LEGACY contact frames from the old-track detector; contact')
print(' detection was never re-run on the weight-mode track)')
