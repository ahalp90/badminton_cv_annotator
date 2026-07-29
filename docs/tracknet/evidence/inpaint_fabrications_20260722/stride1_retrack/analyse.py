"""Convert the weight-mode CSV to the 1.npy convention, then run the brief's
census and flipped-contact check on both weight and the nonoverlap baseline.

The census and periodic_at helpers are the brief's scripts VERBATIM (including
the fact that census ignores a run still open at the array end). Keeping them
byte-faithful is what makes the 375-zone baseline comparable.
"""
import csv

import numpy as np

BASE = ('/home/ariel/Documents/COSC594/badminton_stroke_classification'
        '/local_scratch/autograder_architecture')
CSV_IN = f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/save_dir/pilot_288p_ball.csv'
NPY_OUT = f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/pilot_weight.npy'
BASELINE = f'{BASE}/pilot_track_npy/1.npy'
BISECT = f'{BASE}/inpaint_fabrications_investigation/c11_landing_bisect/bisect_per_rally.csv'

# --- convert: mirror of the block-1 recipe that reproduced 1.npy exactly ---
with open(CSV_IN) as fh:
    header = fh.readline().strip()
print('csv header:', header)
assert header == 'Frame,Visibility,X,Y', f'unexpected header: {header}'

data = np.genfromtxt(CSV_IN, delimiter=',', skip_header=1)  # Frame,Visibility,X,Y
assert data.shape[0] == 154393, f'frame count {data.shape[0]} != 154393'
arr = np.empty((data.shape[0], 3), dtype=float)
arr[:, 0] = data[:, 2] / 512.0   # X -> fraction of frame width
arr[:, 1] = data[:, 3] / 288.0   # Y -> fraction of frame height
arr[:, 2] = data[:, 1]           # Visibility
np.save(NPY_OUT, arr)
print(f'converted -> {NPY_OUT}  shape {arr.shape}  frames {data.shape[0]}')


def census(track, label):
    y = track[:, 1]
    eq = y[:len(y) - 16] == y[16:]
    runs, start = [], None
    for i, m in enumerate(eq):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= 32:
                runs.append((start, i))
            start = None
    osc = [(a, b) for a, b in runs if np.ptp(y[a:a + 16]) > 0]
    frames = sum(b - a for a, b in osc)
    phys = sum(b - a + 16 for a, b in osc)
    vis = (track[:, 2] == 1).mean()
    print(f'\n--- census: {label} ---')
    print('oscillating zones:', len(osc))
    print('comparison-index share:', frames / len(y))
    print('physical share:', phys / len(y))
    print('vis fraction:', vis)
    if osc:
        a0 = osc[0][0]
        cyc = np.sort(y[a0:a0 + 16])
        same = sum(1 for a, b in osc
                   if np.array_equal(np.sort(y[a:a + 16]), cyc))
        print('cycle range:', cyc[0], cyc[-1], '| zones sharing it:', same, '/', len(osc))
    return {'zones': len(osc), 'comp': frames / len(y),
            'phys': phys / len(y), 'vis': vis}


def flipped_check(track, label):
    y = track[:, 1]

    def periodic_at(f, span=32):
        seg = y[f:f + span + 16]
        return len(seg) == span + 16 and np.array_equal(
            seg[:span], seg[16:span + 16])

    with open(BISECT) as fh:
        rows = list(csv.DictReader(fh))
    flipped = [r for r in rows
               if r['landing_pool'] and not r['landing_sticky']]
    hits = sum(1 for r in flipped
               if any(periodic_at(int(r['final_contact']) + o)
                      for o in (-2, -1, 0, 1, 2)))
    print(f'\n--- flipped-contact check: {label} ---')
    print('flipped rallies still at a fill edge:', hits, '/', len(flipped))
    return {'hits': hits, 'n': len(flipped)}


base_track = np.load(BASELINE)
weight_track = np.load(NPY_OUT)

b = census(base_track, 'nonoverlap baseline (1.npy)')
w = census(weight_track, 'weight mode (new)')
fb = flipped_check(base_track, 'nonoverlap baseline')
fw = flipped_check(weight_track, 'weight mode')

print('\n=== COMPARISON (nonoverlap -> weight) ===')
print(f'oscillating zones     : {b["zones"]} -> {w["zones"]}')
print(f'comparison-index share: {b["comp"]:.4f} -> {w["comp"]:.4f}')
print(f'physical share        : {b["phys"]:.4f} -> {w["phys"]:.4f}')
print(f'visibility fraction   : {b["vis"]:.4f} -> {w["vis"]:.4f}')
print(f'flipped at fill edge  : {fb["hits"]}/{fb["n"]} -> {fw["hits"]}/{fw["n"]}')
