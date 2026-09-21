import gzip, json, sys
import numpy as np
from itertools import combinations
np.set_printoptions(linewidth=200, precision=4, suppress=True)

def angular_residuals(lines, points):
    normals = lines[:, :2]
    feet = -lines[:, 2, None] * normals / np.sum(normals**2, axis=1)[:, None]
    directions = np.stack((lines[:, 1], -lines[:, 0]), axis=1)
    rays = points[:, None, :2] - points[:, None, 2:] * feet[None]
    dot = np.abs(np.einsum("pli,li->pl", rays, directions))
    cross = np.abs(rays[..., 0] * directions[:, 1] - rays[..., 1] * directions[:, 0])
    angles = np.degrees(np.arctan2(cross, dot))
    return np.where(np.linalg.norm(rays, axis=2) > 1e-12, angles, 90.0)

est_path, bank_path = sys.argv[1], sys.argv[2]
saved = json.load(gzip.open(est_path))
est = saved['estimator']
bank = json.load(gzip.open(bank_path))
lines = np.asarray(est['direction_lines']); T = np.asarray(est['normalised_to_working'])
nl = lines @ T
pairs = np.asarray(list(combinations(range(len(lines)), 2))).reshape(-1, 2)
raw = np.concatenate((np.cross(nl[pairs[:,0]], nl[pairs[:,1]]), np.column_stack((nl[:,1], -nl[:,0], np.zeros(len(lines))))))
cid = np.asarray(est['candidate_ids'])
cands = raw[cid] / np.linalg.norm(raw[cid], axis=1)[:, None]
status = np.asarray(est['candidate_status']); counts = np.asarray(est['support_counts'])
gens = [list(p) for p in pairs] + [[i] for i in range(len(lines))]
retained = np.asarray(est['retained_candidate_ids'])
rows = np.searchsorted(cid, retained)
masks_saved = np.asarray(est['retained_support_masks'], dtype=bool)
print('case', saved['case_id'], 'lines', len(lines), 'cands', len(cands), 'status counts', {s:int((status==s).sum()) for s in np.unique(status)})
targets = np.asarray(bank['control_points_normalised'])
for axis, t in enumerate(targets):
    t = t/np.linalg.norm(t)
    dist_ret = np.degrees(np.arccos(np.clip(np.abs(cands[rows] @ t), 0, 1)))
    order = np.argsort(dist_ret)
    print(f'axis {axis}: retained pencils nearest control (pencil idx, cand id, count, deg, gens):')
    for k in order[:4]:
        print('   ', int(k), int(retained[k]), int(counts[rows[k]]), round(float(dist_ret[k]),4), gens[retained[k]])
    tmask = angular_residuals(nl, t[None])[0]
    member = tmask <= 1.5
    print('   control-VP support at 1.5deg:', int(member.sum()), 'lines', np.flatnonzero(member).tolist())
    print('   residuals of those lines (deg):', np.round(tmask[member], 3).tolist())
    for k in order[:3]:
        m = masks_saved[k]
        inter = int((m & member).sum()); uni = int((m | member).sum())
        print(f'   pencil {int(k)} mask {np.flatnonzero(m).tolist()} IoU with control-support {inter}/{uni}')
    # bank candidates nearest to control
    for c in bank['axes'][axis]['candidates'][:2]:
        r = int(np.searchsorted(cid, c['candidate_id']))
        cm = angular_residuals(nl, cands[r][None])[0] <= 1.5
        print('   bank cand', c['candidate_id'], c['candidate_status'], 'mask', np.flatnonzero(cm).tolist())
        # which retained pencil made it redundant: IoU>0.8
        ious = [(int(k), round(float((masks_saved[k]&cm).sum()/(masks_saved[k]|cm).sum()),3)) for k in range(len(rows))]
        print('     IoU with retained pencils >0.5:', [x for x in ious if x[1]>0.5])
