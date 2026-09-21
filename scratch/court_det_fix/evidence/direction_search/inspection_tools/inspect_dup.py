import gzip, json, sys
import numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
for path in sys.argv[1:]:
    s = json.load(gzip.open(path)); est = s['estimator']
    T = np.asarray(est['normalised_to_working'])
    pts = np.asarray(est['points_working']); nrm = (np.linalg.inv(T) @ pts.T).T
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    ang = np.degrees(np.arccos(np.clip(np.abs(nrm @ nrm.T), 0, 1)))
    masks = np.asarray(est['retained_support_masks'], dtype=bool)
    inter = masks[:, None] & masks[None]; uni = masks[:, None] | masks[None]
    iou = inter.sum(2) / np.maximum(uni.sum(2), 1)
    close = [(i, j, round(float(ang[i, j]), 3), round(float(iou[i, j]), 2)) for i in range(16) for j in range(i+1, 16) if ang[i, j] < 1.0]
    counts = masks.sum(1)
    radius = [None if abs(p[2]) < 1e-12 else round(float(np.hypot(*(p[:2]/p[2] - np.array(s['working_size'])/2)) / np.hypot(*s['working_size'])), 2) for p in pts]
    print(s['case_id'], 'counts', counts.tolist())
    print('   VP radius from centre (diagonals):', radius)
    print('   retained pairs < 1 deg apart (i, j, deg, mask IoU):', close)
    print('   timing', s.get('timing'))
