"""Compare two real SS03-19 pairs through saved and live generation, including scoring.

Run from the repository root with the project Python environment. This is a
bounded smoke check, not a full detector evaluation. Raw arm outputs are saved
before comparisons so any failure remains inspectable.
"""

import gzip
import json
import os
import sys
from pathlib import Path

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
repo = Path(__file__).resolve().parents[3]
root = repo / 'scratch/court_det_fix'
sys.path[:0] = [str(repo), str(repo / 'src'), str(root / 'wider_evaluation')]
import cv2
import run_cases

_, verifier, runtime = run_cases.load_runtime(root)
import automatic_generation
import run_automatic

sys.path.insert(0, str(root / 'svd_runtime'))
from run_benchmark import equal

output = root / 'svd_runtime/integration_smoke_raw'
output.mkdir(exist_ok=True)
def save(name: str, result: dict) -> None:
    (output / f'{name}.json.gz').write_bytes(gzip.compress(json.dumps(result, allow_nan=False).encode(), mtime=0))

cv2.setNumThreads(1)
case_id = 'shuttleset_03_scene_0019'
context = verifier.prepare_view(root, case_id)
saved = verifier.read_json_gz(root / 'frozen_views/baseline_directions' / f'{case_id}.json.gz')
original_frame_path = run_automatic.frame_path
old_pairs = run_automatic.permutations
new_pairs = automatic_generation.permutations
run_automatic.frame_path = lambda source, _: verifier.frame_path(root, source, context.provenance)
# The first two pairs retain original IDs 0 and 1, including the approved pair (0, 2).
run_automatic.permutations = lambda *args: iter([(0, 1), (0, 2)])
automatic_generation.permutations = lambda *args: iter([(0, 1), (0, 2)])
try:
    old = run_automatic.generate(context.source, saved, runtime['zone'], root)
    save('seed16', old)
    full = automatic_generation.generate(context.source, saved, runtime['zone'], root, run_automatic, 16)
    save('live16', full)
    pruned = automatic_generation.generate(context.source, saved, runtime['zone'], root, run_automatic, 12)
    save('live12', pruned)
finally:
    run_automatic.frame_path = original_frame_path
    run_automatic.permutations = old_pairs
    automatic_generation.permutations = new_pairs

def without_timing(value: object) -> object:
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items()
                if key not in ('elapsed_s', 'direction_screen')}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    if isinstance(value, tuple):
        return [without_timing(item) for item in value]
    return value

assert without_timing(old) == without_timing(full), '16-family generator differs from seed'
compared = []
for original_pair, pruned_pair in zip(full['pairs'], pruned['pairs'], strict=True):
    if pruned_pair['status'] != 'skipped_svd_mask':
        equal(without_timing(original_pair), without_timing(pruned_pair), f"shared_pair_{original_pair['pair_id']}")
        compared.append(pruned_pair['pair_id'])
assert 1 in compared
receipt = {'case_id': case_id, 'scope': 'First two original pairs with real seed matcher and full scoring',
           'full16_matches_seed_exactly_except_elapsed': True, 'shared_pair_ids_equal_at_1e_minus_8_absolute': compared,
           'full16_entries': len(full['entries']), 'svd12_entries': len(pruned['entries']),
           'full16_winners': [full['line_winner_id'], full['paint_winner_id']],
           'svd12_winners': [pruned['line_winner_id'], pruned['paint_winner_id']]}
path = root / 'svd_runtime/integration_smoke.json.gz'
path.write_bytes(gzip.compress(json.dumps(receipt, allow_nan=False).encode(), mtime=0))
print(json.dumps(receipt))
