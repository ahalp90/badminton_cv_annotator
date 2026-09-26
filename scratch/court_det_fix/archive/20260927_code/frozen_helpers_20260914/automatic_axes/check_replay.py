"""Compare a frozen given-direction result with its extracted-helper replay."""

import gzip
import json
import math
from pathlib import Path

name = 'gxBQ_window_00_frame_5.json.gz'
with gzip.open(Path('axis_matching_20260914/given_finite') / name, 'rt') as stream:
    old = json.load(stream)
with gzip.open(Path('automatic_axes_20260914/role_replay') / name, 'rt') as stream:
    new = json.load(stream)
old.pop('elapsed_s')
new.pop('elapsed_s')
assert len(old['entries']) == len(new['entries'])
differences = []
for old_entry, new_entry in zip(old['entries'], new['entries'], strict=True):
    old_score = old_entry.pop('shortlist_score')
    new_score = new_entry.pop('shortlist_score')
    # Replay permits tiny numerical drift only in finite-support scores. Geometry,
    # candidate order, downstream scores and every other saved field stay exact.
    assert math.isclose(old_score, new_score, rel_tol=0., abs_tol=1e-10)
    if old_score != new_score:
        differences.append(abs(old_score - new_score))
assert old == new
print('GX5 replay passed: all fields exact except finite-support scores;',
      len(differences), 'changed scores; maximum absolute difference', max(differences, default=0.))
