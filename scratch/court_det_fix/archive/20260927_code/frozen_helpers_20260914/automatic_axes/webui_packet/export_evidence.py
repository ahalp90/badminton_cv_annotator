"""Export allowlisted saved evidence and literal source; perform no geometry work."""

import ast
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
LOCAL = ROOT / 'scratch/court_det_fix/evidence/independent_proposals/development/player_guided/20260914'
PUBLIC = ROOT / 'experiments/annotator/independent_court/recorded/player_guided/projective_patterns'
OUTPUT = PUBLIC / 'evaluation'


def read(path: Path) -> dict[str, Any]:
    return json.loads(gzip.decompress(path.read_bytes()))


def digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write(name: str, value: dict[str, Any]) -> None:
    raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    for forbidden in [b'/home/', b'/scratch/', b'local_scratch', b'carmack', b'password', b'ssh ']:
        assert forbidden not in raw.lower(), forbidden
    path = OUTPUT / name
    path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    assert read(path) == value
    print(name, len(raw), 'uncompressed bytes;', path.stat().st_size, 'compressed bytes')


case_ids = ['gxBQ_window_00_frame_0', 'gxBQ_window_00_frame_5', 'am2_window_01_frame_28019']
estimator_keys = [
    'raw_fragments', 'visible_fragments', 'merge_input_cap', 'merge_input_excluded',
    'merged_direction_count', 'direction_cap_excluded', 'direction_lines', 'pair_candidates',
    'infinity_candidates', 'degenerate_candidates', 'candidate_ids', 'support_counts',
    'candidate_status', 'retained_candidate_ids', 'retained_support_masks', 'points_working',
    'normalised_to_working',
]
cases = []
for case_id in case_ids:
    saved_path = LOCAL / 'vp_pruning/coverage/results' / f'{case_id}.json.gz'
    svd_path = LOCAL / 'automatic_axes/svd_fixed/collected/records' / f'{case_id}.json.gz'
    saved, svd = read(saved_path), read(svd_path)
    assert svd['source_record_md5']['estimator'] == digest(saved_path)
    assert svd['original_candidate_ids'] == saved['estimator']['retained_candidate_ids']
    assert svd['original_points_working'] == saved['estimator']['points_working']
    estimator = {key: saved['estimator'][key] for key in estimator_keys}
    cases.append({
        'case_id': case_id, 'working_size': saved['working_size'], 'settings': saved['settings'],
        'source_record_md5': {'coverage': digest(saved_path), 'svd': digest(svd_path)},
        'estimator': estimator, 'svd_diagnostic': svd,
    })
write('direction_records.json.gz', {
    'schema': 'court-direction-review/1', 'snapshot_basis': 'b90518c',
    'export_kind': 'literal_saved_fields_no_new_geometry',
    'actual_fragment_merge_membership_available': False,
    'compatible_raw_ids_included': False, 'new_experiment_results_available': False, 'cases': cases,
})

source_packs = [
    LOCAL.parent / '20260909/gx_extension/inputs.json.gz',
    LOCAL.parent / '20260909/broadcast_extension/inputs.json.gz',
    LOCAL.parent / '20260908/marking_refit/marking_inputs.json.gz',
]
sources = {}
for path in source_packs:
    sources.update({case['id']: case for case in read(path)['cases']})
entries = []
entry_keys = ['candidate_id', 'homography_working', 'corners_px', 'shortlist_score', 'stripe', 'gates', 'profile']
all_camera = LOCAL / 'automatic_axes/collected/all_camera'
for path in sorted(all_camera.glob('*.json.gz')):
    saved = read(path)
    case_id = saved['case_id']
    winners = [saved['line_winner_id'], saved['paint_winner_id']]
    by_id = {entry['candidate_id']: entry for entry in saved['entries']}
    selected = []
    for candidate_id in dict.fromkeys(winners):
        if candidate_id is not None:
            selected.append({key: by_id[candidate_id][key] for key in entry_keys})
    entries.append({
        'case_id': case_id, 'population': 'automatic_all_camera', 'working_size': saved['working_size'],
        'native_dimensions': sources[case_id]['dimensions'], 'source_record_md5': digest(path),
        'line_winner_id': winners[0], 'paint_winner_id': winners[1], 'entries': selected,
    })
assert len(entries) == 9
path = LOCAL / 'automatic_axes/gx0_control/collected/control.json.gz'
saved = read(path)
by_id = {entry['candidate_id']: entry for entry in saved['entries']}
selected = [{key: by_id[saved[ranking]][key] for key in entry_keys}
            for ranking in ['line_winner_id', 'paint_winner_id']]
entries.append({
    'case_id': saved['case_id'], 'population': 'gx0_label_guided_bank_control',
    'working_size': saved['working_size'], 'native_dimensions': sources[saved['case_id']]['dimensions'],
    'source_record_md5': digest(path), 'line_winner_id': saved['line_winner_id'],
    'paint_winner_id': saved['paint_winner_id'], 'entries': selected,
})
write('ranking_records.json.gz', {
    'schema': 'court-ranking-review/1', 'snapshot_basis': 'b90518c',
    'export_kind': 'literal_saved_winners_no_new_scoring',
    'selection_bias': 'winners_only_not_a_full_population_or_new_negative_dataset', 'populations': entries,
})

excerpt_specs = [
    (ROOT / 'src/courtkeynet/court_corners.py', [
        'COURT_WIDTH_M', 'COURT_LENGTH_M', 'CORNER_COURT_M', '_SINGLES_INSET', '_LONG_SERVICE_INSET',
        '_SHORT_SERVICE_OFFSET', '_NET_Y', '_CENTRE_X', '_FAR_SHORT_Y', '_NEAR_SHORT_Y', 'PAINTED_SEGMENTS_M',
    ]),
    (LOCAL / 'vp_pruning/vp_pruning.py', ['Settings', 'normalisation', 'angular_residuals', 'retain_pencils', 'estimate']),
    (LOCAL / 'automatic_axes/svd_fixed/run_svd_fixed.py', ['svd_direction', 'fit_groups', 'fit_pairs']),
    (LOCAL / 'automatic_axes/diagnose_directions.py', ['control_fit']),
    (LOCAL / 'axis_matching/projective_seed.py', ['corner_errors']),
    (LOCAL / 'axis_matching/inspect_appearance.py', ['ridge_mask', 'profiles']),
    (LOCAL / 'automatic_axes/run_automatic.py', ['winner_ids']),
    (ROOT / 'experiments/annotator/independent_court/assignment.py', ['MARKINGS', 'MARKING_INTERVALS']),
    (ROOT / 'experiments/annotator/independent_court/detector.py', [
        'SEGMENTS_M', 'RIDGE_SAMPLES', 'RIDGE_CENTRE_SHIFTS', 'RIDGE_SIDE_DISTANCE', 'RIDGE_MIN_CONTRAST',
        'RIDGE_MIN_FRACTION', '_filter_painted_stripes', '_visible_samples',
    ]),
]
parts = ['# Exact method excerpts\n\nRead the [evidence guide](README.md) first. These are literal source excerpts,\nnot a standalone runnable module. Module names and whole-file MD5s identify the\nsource snapshots. Imports and unrelated functions are omitted. `np` means NumPy,\n`cv2` means OpenCV; other names refer to the named project modules. No new result\nis produced by publishing these excerpts.\n']
provenance = []
for path, names in excerpt_specs:
    text = path.read_text()
    tree = ast.parse(text)
    nodes = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nodes[node.name] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            nodes[node.target.id] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    nodes[target.id] = node
    parts.append(f'\n## {path.name}\n\nWhole-source MD5: `{digest(path)}`.\n')
    for name in names:
        node = nodes[name]
        start = min([node.lineno] + [item.lineno for item in getattr(node, 'decorator_list', [])])
        excerpt = '\n'.join(text.splitlines()[start - 1:node.end_lineno])
        parts.append(f'\n### {name}\n\n```python\n{excerpt}\n```\n')
    provenance.append({'path': str(path.relative_to(ROOT)), 'md5': digest(path), 'symbols': names})
(OUTPUT / 'method_excerpts.md').write_text(''.join(parts))
(LOCAL / 'automatic_axes/webui_packet/source_manifest.json.gz').write_bytes(
    gzip.compress(json.dumps(provenance, indent=2).encode(), mtime=0))
print('Exact excerpts:', sum(len(names) for _, names in excerpt_specs), 'symbols')
