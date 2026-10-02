import gzip, json, sys
import numpy as np
from itertools import combinations
exec(open('/tmp/inspect_bank.py').read().split("est_path, bank_path")[0])
est_path, bank_path = sys.argv[1], sys.argv[2]
ids = [int(x) for x in sys.argv[3:]]
saved = json.load(gzip.open(est_path)); est = saved['estimator']
bank = json.load(gzip.open(bank_path))
lines = np.asarray(est['direction_lines']); T = np.asarray(est['normalised_to_working']); nl = lines @ T
pairs = np.asarray(list(combinations(range(len(lines)), 2))).reshape(-1, 2)
raw = np.concatenate((np.cross(nl[pairs[:,0]], nl[pairs[:,1]]), np.column_stack((nl[:,1], -nl[:,0], np.zeros(len(lines))))))
cid = np.asarray(est['candidate_ids']); cands = raw[cid]/np.linalg.norm(raw[cid],axis=1)[:,None]
gens = [list(p) for p in pairs] + [[i] for i in range(len(lines))]
ctrl = [T @ (t/np.linalg.norm(t)) for t in np.asarray(bank['control_points_normalised'])]
for c in ids:
    row = int(np.searchsorted(cid, c)); w = T @ cands[row]
    g = gens[c]
    # crossing angle of generator lines (deg) in working px
    if len(g)==2:
        n0, n1 = lines[g[0],:2], lines[g[1],:2]
        cross = np.degrees(np.arcsin(min(1,abs(n0[0]*n1[1]-n0[1]*n1[0]))))
    else: cross=None
    loc = w[:2]/w[2] if abs(w[2])>1e-12 else None
    d = [None if loc is None else round(float(np.linalg.norm(loc - cw[:2]/cw[2])),2) for cw in ctrl]
    print(c, est['candidate_status'][row], 'gens', g, 'crossing_deg', None if cross is None else round(cross,2), 'loc', None if loc is None else np.round(loc,1), 'px to control VPs', d)
