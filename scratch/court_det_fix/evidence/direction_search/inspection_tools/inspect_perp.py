import gzip, json, sys
import numpy as np
from itertools import combinations
exec(open('/tmp/inspect_bank.py').read().split("est_path, bank_path")[0])
est_path, bank_path = sys.argv[1], sys.argv[2]
saved = json.load(gzip.open(est_path)); est = saved['estimator']
bank = json.load(gzip.open(bank_path))
lines = np.asarray(est['direction_lines']); T = np.asarray(est['normalised_to_working']); nl = lines @ T
W,H = saved['working_size']
print(saved['case_id'], 'normal norms of saved lines min/max', np.linalg.norm(lines[:,:2],axis=1).min(), np.linalg.norm(lines[:,:2],axis=1).max())
pairs = np.asarray(list(combinations(range(len(lines)), 2))).reshape(-1, 2)
raw = np.concatenate((np.cross(nl[pairs[:,0]], nl[pairs[:,1]]), np.column_stack((nl[:,1], -nl[:,0], np.zeros(len(lines))))))
cid = np.asarray(est['candidate_ids']); cands = raw[cid]/np.linalg.norm(raw[cid],axis=1)[:,None]
for axis, t in enumerate(np.asarray(bank['control_points_normalised'])):
    t = t/np.linalg.norm(t)
    w = T @ t
    print(f' axis {axis} control VP working', np.round(w[:2]/w[2],1) if abs(w[2])>1e-12 else 'infinite', 'homog', np.round(t,4))
    res = angular_residuals(nl, t[None])[0]
    sel = np.flatnonzero(res <= 15)
    # perpendicular distance in working px from control VP to each line (finite VP)
    if abs(w[2])>1e-9:
        p = w/w[2]
        perp = np.abs(lines @ p)
    else:
        perp = np.full(len(lines), np.nan)
    # algebraic residual used by SVD on normalised lines (unit vector t, lines normalised to unit normal in normalised coords)
    nln = nl/np.linalg.norm(nl[:,:2],axis=1)[:,None]
    alg = np.abs(nln @ t)
    print('   line  ang_deg  perp_px  algebraic')
    for i in sel[np.argsort(res[sel])]:
        print(f'   {i:4d} {res[i]:7.3f} {perp[i]:8.2f} {alg[i]:.5f}')
