"""Read-only extension of follow-up 3; no fits or detector runs."""
from pathlib import Path
import gzip, json
import numpy as np

ROOT = Path(__file__).parent / 'task3_inputs/scratch/court_det_fix'
RUNDIR = ROOT / 'direction_agreement/runs/direction_agreement_20260915_144900'

def read(path):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)

rows = []
for path in sorted((RUNDIR/'e3').glob('*.json.gz')):
    rec = read(path)
    b, s = rec['sets']['B'], rec['sets']['B_svd']
    e = read(ROOT/'frozen_views/baseline_directions'/path.name)['estimator']
    a = read(RUNDIR/'e2'/path.name)['arms']['B']
    groups = s['groups']
    n = np.array([g['line_count'] for g in groups])
    residual = np.array([g['algebraic_rms'] for g in groups])
    sig = np.array([g['singular_values'] for g in groups])
    order = sorted(range(len(groups)),key=lambda i:(residual[i],-n[i],i))
    rank = {idx:r+1 for r,idx in enumerate(order)}
    best = b['fits']['best_finite']
    svd_best = s['fits']['best_finite']
    fixed_svd = next(f for f in s['fits']['records'] if f['pair_id']==best['pair_id'])
    # Recompute actual B residual under the same normalized, unit-line convention.
    lines = np.asarray(e['direction_lines']) @ np.asarray(e['normalised_to_working'])
    lines /= np.linalg.norm(lines[:,:2],axis=1)[:,None]
    points = np.asarray(a['points_normalised']); points /= np.linalg.norm(points,axis=1)[:,None]
    original_rms=np.array([np.sqrt(np.mean((lines[np.asarray(mask,dtype=bool)]@v)**2)) for mask,v in zip(a['support_masks'],points)])
    # Tests of mathematical statements under the existing fixed metric.
    np.testing.assert_allclose(residual, sig[:,-1]/np.sqrt(n), rtol=1e-10,atol=1e-12)
    assert np.all(residual <= original_rms+1e-12)
    saved_g=[]
    for i in range(len(n)):
        saved_g.append({'group':i,'support_count':int(n[i]),'rms_rank':rank[i],
           'svd_rms':float(residual[i]),'original_direction_rms':float(original_rms[i]),
           'singular_values':sig[i].tolist(),'normalized_gap':groups[i]['normalised_nullspace_gap'],
           'movement_deg':groups[i]['movement_deg']})
    rows.append({'case':rec['case_id'],'approved':rec['control']['visually_approved'],
       'B_best_px':best['max_corner_working_px'],'B_best_groups':best['groups'],
       'B_best_group_ranks':[rank[i] for i in best['groups']],
       'B_svd_best_px':svd_best['max_corner_working_px'],'B_svd_best_groups':svd_best['groups'],
       'B_svd_same_pair_px':fixed_svd.get('max_corner_working_px'),
       'svd_saved_ms':s['svd_elapsed_s']*1e3,
       'top4_support_counts':[int(n[i]) for i in order[:4]],'best_pair_support_counts':[int(n[i]) for i in best['groups']],
       'support_count_range':[int(n.min()),int(n.max())],
       'groups':saved_g})
output={'note':'Read-only re-analysis of uploaded cached records. B_svd errors are existing control-guided E3 fit potentials, not automatic detection. No fits rerun.', 'cases':rows}
with gzip.open(Path(__file__).parent/'cached_extension.json.gz','wt') as f: json.dump(output,f,allow_nan=False,indent=2)
for r in rows:
 print(r['case'], 'B',round(r['B_best_px'],4),'SVD best',round(r['B_svd_best_px'],4),'SVD same pair',round(r['B_svd_same_pair_px'],4),'best ranks',r['B_best_group_ranks'],'best supports',r['best_pair_support_counts'],'top4 supports',r['top4_support_counts'],'counts',r['support_count_range'],'ms',round(r['svd_saved_ms'],3))
print('SVD recorded ms min median max:',np.min([r['svd_saved_ms'] for r in rows]),np.median([r['svd_saved_ms'] for r in rows]),np.max([r['svd_saved_ms'] for r in rows]))
print('SVD best improved',sum(r['B_svd_best_px']<r['B_best_px'] for r in rows),'of',len(rows))
