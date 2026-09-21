import gzip, json, sys
import numpy as np
from itertools import combinations
exec(open('/tmp/inspect_bank.py').read().split("est_path, bank_path")[0])
est_path, bank_path = sys.argv[1], sys.argv[2]
focus = [int(x) for x in sys.argv[3].split(',')]
saved = json.load(gzip.open(est_path)); est = saved['estimator']
bank = json.load(gzip.open(bank_path))
lines = np.asarray(est['direction_lines']); T = np.asarray(est['normalised_to_working']); nl = lines @ T
pairs = np.asarray(list(combinations(range(len(lines)), 2))).reshape(-1, 2)
raw = np.concatenate((np.cross(nl[pairs[:,0]], nl[pairs[:,1]]), np.column_stack((nl[:,1], -nl[:,0], np.zeros(len(lines))))))
cid = np.asarray(est['candidate_ids']); cands = raw[cid]/np.linalg.norm(raw[cid],axis=1)[:,None]
gens = [list(p) for p in pairs] + [[i] for i in range(len(lines))]
ctrl = [T @ (t/np.linalg.norm(t)) for t in np.asarray(bank['control_points_normalised'])]
for axis, c in enumerate(focus):
    w = ctrl[axis]; p = w/w[2]
    row = int(np.searchsorted(cid, c)); cw = T @ cands[row]
    m = angular_residuals(nl, cands[row][None])[0] <= 1.5
    ctrl_mask = angular_residuals(nl, (np.linalg.inv(T) @ w)[None])[0] <= 1.5
    perp = np.abs(lines @ p)
    loc = cw[:2]/cw[2]
    print(f'axis {axis} control VP {np.round(p[:2],1)} retained {c} gens {gens[c]} at {np.round(loc,1)} ({np.linalg.norm(loc-p[:2]):.1f}px from control)')
    print('   true members perp px:', {int(i): round(float(perp[i]),1) for i in np.flatnonzero(m & ctrl_mask)})
    print('   extra members perp px:', {int(i): round(float(perp[i]),1) for i in np.flatnonzero(m & ~ctrl_mask)})
    print('   missing control members perp px:', {int(i): round(float(perp[i]),1) for i in np.flatnonzero(~m & ctrl_mask)})
