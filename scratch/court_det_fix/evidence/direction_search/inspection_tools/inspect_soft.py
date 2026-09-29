import gzip, json, sys
import numpy as np
from itertools import combinations
exec(open('/tmp/inspect_bank.py').read().split("est_path, bank_path")[0])
est_path, bank_path = sys.argv[1], sys.argv[2]
saved = json.load(gzip.open(est_path)); est = saved['estimator']
bank = json.load(gzip.open(bank_path))
lines = np.asarray(est['direction_lines']); T = np.asarray(est['normalised_to_working']); nl = lines @ T
pairs = np.asarray(list(combinations(range(len(lines)), 2))).reshape(-1, 2)
raw = np.concatenate((np.cross(nl[pairs[:,0]], nl[pairs[:,1]]), np.column_stack((nl[:,1], -nl[:,0], np.zeros(len(lines))))))
cid = np.asarray(est['candidate_ids']); cands = raw[cid]/np.linalg.norm(raw[cid],axis=1)[:,None]
gens = [list(p) for p in pairs] + [[i] for i in range(len(lines))]
status = np.asarray(est['candidate_status']); counts=np.asarray(est['support_counts'])
retained = list(est['retained_candidate_ids'])
targets = np.asarray(bank['control_points_normalised'])
W,H = saved['working_size']
print('working size', W, H, 'diag', np.hypot(W,H))
for axis, t in enumerate(targets):
    t=t/np.linalg.norm(t)
    ctrl_res = angular_residuals(nl, t[None])[0]
    # euclidean location of control VP in normalised units
    print(f'axis {axis} control VP normalised radius', None if abs(t[2])<1e-9 else round(float(np.linalg.norm(t[:2]/t[2])),3))
    dists = np.degrees(np.arccos(np.clip(np.abs(cands@t),0,1)))
    near_ret = sorted([(float(dists[np.searchsorted(cid,r)]), r) for r in retained])[:2]
    ids = [r for _, r in near_ret] + [c['candidate_id'] for c in bank['axes'][axis]['candidates']]
    for c in ids:
        row = int(np.searchsorted(cid, c))
        ang = angular_residuals(nl, cands[row][None])[0]
        m = ang <= 1.5
        soft = np.maximum(0, 1-ang/1.5).sum()
        extra = np.flatnonzero(m & (ctrl_res>1.5)); missing = np.flatnonzero(~m & (ctrl_res<=1.5))
        print(f'  cand {c} [{status[row]}] gens {gens[c]} dist {dists[row]:.4f}deg count {m.sum()} (saved {counts[row]}) soft {soft:.3f} '
              f'rms_in_mask {np.sqrt(np.mean(ang[m]**2)):.3f} extra {extra.tolist()} ctrl_res_extra {np.round(ctrl_res[extra],2).tolist()} missing {missing.tolist()}')
    # radius of the neighbourhood: how many bank candidates are closer to control than the nearest retained
    print('   bank candidates closer than nearest retained:', int((dists < near_ret[0][0]).sum()),
          ' within 0.1deg:', int((dists<0.1).sum()), ' statuses within 0.1deg:', {s:int(((dists<0.1)&(status==s)).sum()) for s in np.unique(status)})
    # generator membership check: generators of nearest retained in control support?
    for d_, r in near_ret:
        print('   retained', r, 'generator lines control residual', [round(float(ctrl_res[g]),3) for g in gens[r]])
    b = bank['best_measured_fit']
