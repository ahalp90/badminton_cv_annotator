"""Test Fable's prediction: in weight mode the bobbing loop should collapse to
ONE CONSTANT position (the tent-weighted blend of the 16 cycle values), which is
ambiguous with a genuinely resting shuttle.

Source of the prediction: inpaint_fabrications_investigation/inpaint_fabrications_investigation.md, "Open questions",
weight-mode bullet. If it holds, the earlier "0/38 contacts on invented frames"
reading is wrong: the fabrications did not go away, they went flat.
"""
import csv
from collections import Counter

import numpy as np

BASE = ('/home/ariel/Documents/COSC594/badminton_stroke_classification'
        '/local_scratch/autograder_architecture')
OLD = np.load(f'{BASE}/pilot_track_npy/1.npy')
NEW = np.load(f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/pilot_weight.npy')
BISECT = f'{BASE}/inpaint_fabrications_investigation/c11_landing_bisect/bisect_per_rally.csv'

# ---------------------------------------------------------------- old zones
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
print(f'old fake zones: {len(osc)}')

zone_frames = np.unique(np.concatenate([np.arange(a, min(b + 16, len(OLD)))
                                        for a, b in osc]))
print(f'frames inside old fake zones: {len(zone_frames)} '
      f'({len(zone_frames) / len(OLD):.4f} of video)')

# ------------------------------------------- the cycle, indexed by frame % 16
# nonoverlap fill windows tile from frame 0, so within-window index == f % 16
cycle_x = np.zeros(16)
cycle_y = np.zeros(16)
for i in range(16):
    sel = zone_frames[zone_frames % 16 == i]
    cycle_x[i] = Counter(OLD[sel, 0]).most_common(1)[0][0]
    cycle_y[i] = Counter(OLD[sel, 1]).most_common(1)[0][0]
print('\ncycle x (by phase):', np.round(cycle_x, 4))
print('cycle y (by phase):', np.round(cycle_y, 4))

# ------------------------------------------------- predicted weight-mode blend
w = np.ones(16)
for i in range((16 + 1) // 2):
    w[i] = i + 1
    w[16 - i - 1] = i + 1
w = w / w.sum()
print('\ntent weights:', np.round(w * 72, 0).astype(int), '/ 72')

blend_x, blend_y = (w * cycle_x).sum(), (w * cycle_y).sum()
pred_x = int(blend_x * 512) / 512
pred_y = int(blend_y * 288) / 288
print(f'predicted blend      : x={blend_x:.5f}  y={blend_y:.5f}')
print(f'predicted saved value: x={pred_x:.5f} ({int(blend_x * 512)}px)  '
      f'y={pred_y:.5f} ({int(blend_y * 288)}px)')

# ----------------------------------------------- most common positions, actual
def top_positions(track, label, n=8):
    pairs = Counter(map(tuple, track[:, :2]))
    print(f'\n--- most common exact (x,y) in {label} ---')
    for (px, py), cnt in pairs.most_common(n):
        tag = ''
        if px == 0 and py == 0:
            tag = '  <- blank (no detection)'
        print(f'  x={px:.5f} y={py:.5f}  count={cnt:6d} '
              f'({cnt / len(track):.4f}){tag}')
    return pairs


old_top = top_positions(OLD, 'OLD track (nonoverlap)')
new_top = top_positions(NEW, 'NEW track (weight)')

# is the predicted constant actually there?
print('\n--- does the predicted blend appear in the NEW track? ---')
nonblank = {k: v for k, v in new_top.items() if k != (0.0, 0.0)}
top_nonblank, top_cnt = max(nonblank.items(), key=lambda kv: kv[1])
print(f'busiest non-blank NEW position: x={top_nonblank[0]:.5f} '
      f'y={top_nonblank[1]:.5f}  count={top_cnt} ({top_cnt / len(NEW):.4f})')
print(f'predicted                     : x={pred_x:.5f} y={pred_y:.5f}')
dx_px = abs(top_nonblank[0] - pred_x) * 512
dy_px = abs(top_nonblank[1] - pred_y) * 288
print(f'difference: {dx_px:.2f} px in x, {dy_px:.2f} px in y')

# for comparison, the busiest non-blank position in the OLD track
old_nonblank = {k: v for k, v in old_top.items() if k != (0.0, 0.0)}
o_pos, o_cnt = max(old_nonblank.items(), key=lambda kv: kv[1])
print(f'busiest non-blank OLD position: x={o_pos[0]:.5f} y={o_pos[1]:.5f} '
      f'count={o_cnt} ({o_cnt / len(OLD):.4f})')

# ------------------------------- what does the NEW track do at old-zone sites?
CONST = top_nonblank
new_at_zone = NEW[zone_frames]
is_blank = (new_at_zone[:, 0] == 0) & (new_at_zone[:, 1] == 0)
is_const = (new_at_zone[:, 0] == CONST[0]) & (new_at_zone[:, 1] == CONST[1])
print('\n--- NEW track values at frames that were fabricated in the OLD track ---')
print(f'  blank (0,0)              : {is_blank.sum():6d} ({is_blank.mean():.4f})')
print(f'  at the constant position : {is_const.sum():6d} ({is_const.mean():.4f})')
print(f'  something else           : {(~is_blank & ~is_const).sum():6d} '
      f'({(~is_blank & ~is_const).mean():.4f})')

# where does the constant sit OUTSIDE the old zones? (specificity check)
outside = np.setdiff1d(np.arange(len(NEW)), zone_frames)
oc = ((NEW[outside, 0] == CONST[0]) & (NEW[outside, 1] == CONST[1]))
print(f'  constant outside old zones: {oc.sum():6d} ({oc.mean():.4f} of '
      f'non-zone frames)')

# --------------------------------- desync from a straight-line interpolation
print('\n--- is the constant on a straight line between the real detections? ---')


def real_mask(track):
    blank = (track[:, 0] == 0) & (track[:, 1] == 0)
    const = (track[:, 0] == CONST[0]) & (track[:, 1] == CONST[1])
    return ~blank & ~const


rm = real_mask(NEW)
devs, consts_in_gap = [], 0
for a, b in osc:
    lo, hi = a, min(b + 16, len(NEW) - 1)
    pre = np.where(rm[:lo])[0]
    post = np.where(rm[hi:])[0]
    if len(pre) == 0 or len(post) == 0:
        continue
    f0, f1 = pre[-1], hi + post[0]
    if f1 <= f0:
        continue
    inside = np.arange(lo, hi)
    sel = inside[(NEW[inside, 0] == CONST[0]) & (NEW[inside, 1] == CONST[1])]
    if len(sel) == 0:
        continue
    consts_in_gap += 1
    t = (sel - f0) / (f1 - f0)
    lin_x = NEW[f0, 0] + t * (NEW[f1, 0] - NEW[f0, 0])
    lin_y = NEW[f0, 1] + t * (NEW[f1, 1] - NEW[f0, 1])
    d = np.hypot((NEW[sel, 0] - lin_x) * 512, (NEW[sel, 1] - lin_y) * 288)
    devs.append(np.median(d))

devs = np.array(devs)
print(f'old zones whose NEW frames hold the constant: {consts_in_gap}/{len(osc)}')
if len(devs):
    print(f'median distance from the straight line joining the bracketing real '
          f'detections: {np.median(devs):.1f} px')
    print(f'  quartiles: {np.percentile(devs, 25):.1f} / '
          f'{np.percentile(devs, 75):.1f} px   max {devs.max():.1f} px')
    print('  (a genuine bridge would sit near 0 px; a fixed model output '
          'ignores the endpoints)')

# ------------------------------- the 8 constant-nonzero flipped-rally contacts
print('\n--- the flipped-rally final contacts that sat on a constant span ---')
with open(BISECT) as fh:
    rows = list(csv.DictReader(fh))
flipped = [r for r in rows if r['landing_pool'] and not r['landing_sticky']]
y_new = NEW[:, 1]


def flat_span_at(f, span=32):
    seg = y_new[f:f + span + 16]
    if len(seg) != span + 16:
        return None
    if not np.array_equal(seg[:span], seg[16:span + 16]):
        return None
    return f


hits = []
for r in flipped:
    for off in (-2, -1, 0, 1, 2):
        f = flat_span_at(int(r['final_contact']) + off)
        if f is not None:
            hits.append(f)
            break
pos = Counter((NEW[f, 0], NEW[f, 1]) for f in hits)
print(f'flipped rallies with a flat span at the final contact: {len(hits)}/38')
for (px, py), cnt in pos.most_common():
    mark = '  <- THE CONSTANT' if (px, py) == CONST else (
        '  <- blank' if (px, py) == (0.0, 0.0) else '')
    print(f'  x={px:.5f} y={py:.5f}  {cnt} rallies{mark}')
