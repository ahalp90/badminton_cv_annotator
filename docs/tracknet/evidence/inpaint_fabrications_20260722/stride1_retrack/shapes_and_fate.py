"""Two questions about the flattened fabrication.

Q1 Is there periodicity or structure in the flattened locations? Is there ONE
   attractor or several, and do the runs share an entry/exit transient?
Q2 Where did the old fabricated frames go? Literally to (0,0), or smoothed into
   plausible values?
"""
from collections import Counter

import numpy as np

BASE = ('/home/ariel/Documents/COSC594/badminton_stroke_classification'
        '/local_scratch/autograder_architecture')
OLD = np.load(f'{BASE}/pilot_track_npy/1.npy')
NEW = np.load(f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/pilot_weight.npy')
CONST = (0.474609375, 0.2465277777777778)


def runs_of(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


# ------------------------------------------------------- old proven zones
y_old = OLD[:, 1]
eq = y_old[:len(y_old) - 16] == y_old[16:]
zr, start = [], None
for i, m in enumerate(eq):
    if m and start is None:
        start = i
    elif not m and start is not None:
        if i - start >= 32:
            zr.append((start, i))
        start = None
osc = [(a, b) for a, b in zr if np.ptp(y_old[a:a + 16]) > 0]
old_fab = np.zeros(len(OLD), bool)
for a, b in osc:
    old_fab[a:min(b + 16, len(OLD))] = True

# =====================================================================  Q1
print('=== Q1a. how many flat attractors are there? ===')
print('positions held in any run of >=16 frames, by total frames held:')
pos_run_frames = Counter()
seen = {}
for key in set(map(tuple, NEW[:, :2])):
    pass  # (full scan below is cheaper)

# find runs of constant value (any value), >=16 long
same = (NEW[1:, 0] == NEW[:-1, 0]) & (NEW[1:, 1] == NEW[:-1, 1])
flat_runs = runs_of(same)
for a, b in flat_runs:
    length = b - a + 1
    if length >= 16:
        pos_run_frames[(NEW[a, 0], NEW[a, 1])] += length
for (px, py), n in pos_run_frames.most_common(10):
    tag = ''
    if (px, py) == CONST:
        tag = '  <- THE CONSTANT'
    elif px == 0 and py == 0:
        tag = '  <- blank'
    print(f'  x={px * 512:6.1f}px y={py * 288:6.1f}px  frames={n:6d}{tag}')

print('\n=== Q1b. residual periodicity inside constant runs ===')
is_const = (NEW[:, 0] == CONST[0]) & (NEW[:, 1] == CONST[1])
cruns = [(a, b) for a, b in runs_of(is_const)]
print(f'constant runs: {len(cruns)}')
print('by construction a run holds ONE value, so there is no motion inside it.')
print('checking the frames BETWEEN/AROUND runs for a repeating transient.')

# entry transient: the k frames immediately before each long constant run
K = 6
entries, exits = [], []
for a, b in cruns:
    if b - a < 32:
        continue
    if a - K >= 0 and not is_const[a - K:a].any():
        entries.append(np.round(NEW[a - K:a, 1] * 288).astype(int))
    if b + K < len(NEW) and not is_const[b:b + K].any():
        exits.append(np.round(NEW[b:b + K, 1] * 288).astype(int))
entries = np.array(entries) if entries else np.zeros((0, K), int)
exits = np.array(exits) if exits else np.zeros((0, K), int)
print(f'long runs sampled: entries {len(entries)}, exits {len(exits)}')
if len(entries):
    uniq_e = Counter(map(tuple, entries))
    print(f'  distinct entry transients (y px, {K} frames): {len(uniq_e)}')
    for seq, n in uniq_e.most_common(3):
        print(f'    {list(seq)}  x{n}')
if len(exits):
    uniq_x = Counter(map(tuple, exits))
    print(f'  distinct exit transients (y px, {K} frames): {len(uniq_x)}')
    for seq, n in uniq_x.most_common(3):
        print(f'    {list(seq)}  x{n}')
print('many distinct transients means they are shaped by the surrounding real')
print('evidence, not a fixed pattern: no second periodic signature to exploit.')

print('\n=== Q1c. is the OLD 16-frame period detectable anywhere in NEW? ===')
y_new = NEW[:, 1]
for period in (8, 16, 32):
    eqp = y_new[:len(y_new) - period] == y_new[period:]
    nonflat = eqp & ~is_const[:len(eqp)]
    print(f'  period {period:2d}: matching non-constant comparison indices = '
          f'{nonflat.sum()} ({nonflat.mean():.4f})')

# =====================================================================  Q2
print('\n=== Q2. where did the old fabricated frames go? ===')
zf = np.where(old_fab)[0]
new_blank = (NEW[zf, 0] == 0) & (NEW[zf, 1] == 0)
new_const = (NEW[zf, 0] == CONST[0]) & (NEW[zf, 1] == CONST[1])
new_other = ~new_blank & ~new_const
print(f'old fabricated frames: {len(zf)}')
print(f'  -> literally (0,0) blank : {new_blank.sum():6d} '
      f'({new_blank.mean():.4f})')
print(f'  -> the constant          : {new_const.sum():6d} '
      f'({new_const.mean():.4f})')
print(f'  -> some other value      : {new_other.sum():6d} '
      f'({new_other.mean():.4f})')

print('\nare those "other" values smooth, or jumpy?')


def step_px(track, idx):
    idx = idx[(idx > 0) & (idx < len(track) - 1)]
    keep = []
    for f in idx:
        if (track[f - 1, 0] == 0 and track[f - 1, 1] == 0):
            continue
        if (track[f, 0] == 0 and track[f, 1] == 0):
            continue
        keep.append(f)
    keep = np.array(keep, int)
    if len(keep) == 0:
        return np.array([])
    dx = (track[keep, 0] - track[keep - 1, 0]) * 512
    dy = (track[keep, 1] - track[keep - 1, 1]) * 288
    return np.hypot(dx, dy)


other_idx = zf[new_other]
# reference: frames far from any old zone, not blank, not constant
far = np.ones(len(NEW), bool)
for a, b in osc:
    far[max(0, a - 200):min(len(NEW), b + 216)] = False
ref_idx = np.where(far & ~((NEW[:, 0] == 0) & (NEW[:, 1] == 0))
                   & ~is_const)[0]
s_other = step_px(NEW, other_idx)
s_ref = step_px(NEW, ref_idx)
for name, arr in (('old-zone "other" frames', s_other),
                  ('clearly-real frames', s_ref)):
    if len(arr):
        print(f'  {name}: n={len(arr):6d}  median step '
              f'{np.median(arr):6.2f}px  p90 {np.percentile(arr, 90):6.2f}px')
print('a smoothed bridge shows small steps; a jump shows large ones.')

print('\nand where do the blanked ones sit relative to the constant runs?')
blank_idx = zf[new_blank]
adj = sum(1 for f in blank_idx
          if (f > 0 and is_const[f - 1]) or (f + 1 < len(NEW) and is_const[f + 1]))
print(f'  of {len(blank_idx)} blanked old-fab frames, {adj} touch a constant run')
