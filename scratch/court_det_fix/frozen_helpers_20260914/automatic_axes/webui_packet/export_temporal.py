"""Copy cached temporal evidence; do not fit, align, score or decode video."""
import gzip
import hashlib
import html
import json
from pathlib import Path

ROOT = Path.cwd()
LOCAL = ROOT / 'scratch/court_det_fix/worklog/checks/independent/player_guided'
PUBLIC = ROOT / 'experiments/annotator/independent_court/recorded/player_guided/projective_patterns'
OUTPUT = PUBLIC / 'evaluation'


def read(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


packs = [LOCAL / '20260909/gx_extension/inputs.json.gz', LOCAL / '20260908/marking_refit/marking_inputs.json.gz']
rows = []
for pack_path in packs:
    packed = read(pack_path)
    for source in packed['cases']:
        case_id = source['id']
        if not case_id.startswith(('gxBQ_', 'am2_', 'am3_')):
            continue
        video = case_id.split('_')[0]
        frame_index = int(case_id.rsplit('_', 1)[1])
        saved_path = LOCAL / '20260914/vp_pruning/coverage/results' / f'{case_id}.json.gz'
        saved = read(saved_path)
        estimator = saved['estimator']
        if video == 'gxBQ':
            provenance = source['provenance']
            fps = provenance['source_fps']
            raw_path = LOCAL / '20260909/gx_extension/people' / source['image']
            assert md5(raw_path) == provenance['line_image_file_md5']
            if frame_index in (0, 5):
                image = f'../images/gxBQ_window_00_frame_{frame_index:08d}.png'
                image_kind = 'native frame image'
            else:
                image = f'../../gx_trace_overlays/{case_id}.jpg'
                image_kind = 'historical annotated rendering; geometry is not a stability judgement'
        else:
            fps = None
            image = f'../../stripe_overlays/{case_id}__frame.jpg'
            image_kind = 'published frame rendering from the earlier stripe experiment'
        image_path = (OUTPUT / image).resolve()
        assert image_path.exists()
        candidates = []
        for index, candidate in enumerate(source.get('candidates', [])):
            candidates.append({'input_candidate_index': index, **{key: candidate[key] for key in [
                'corners_px', 'original_score', 'original_rank', 'source_variant', 'source_experiment',
            ]}})
        rows.append({
            'case_id': case_id, 'video_group': video, 'frame_index': frame_index,
            'source_fps': fps, 'nominal_seconds_from_frame_zero': frame_index / fps if fps else None,
            'time_note': 'frame_index/source_fps, not presentation timestamp' if fps else 'FPS unavailable in selected pack',
            'camera_stability_verified': False, 'native_dimensions': source['dimensions'],
            'working_size': saved['working_size'], 'segments_native_px': source['segments_px'],
            'direction_estimator': {key: estimator[key] for key in [
                'direction_lines', 'retained_candidate_ids', 'retained_support_masks',
                'points_working', 'normalised_to_working',
            ]},
            'estimator_settings': saved['settings'], 'image': image, 'image_kind': image_kind,
            'image_md5': md5(image_path), 'input_pack_md5': md5(pack_path), 'estimator_record_md5': md5(saved_path),
            'historical_candidates': candidates,
            'candidate_note': 'Input-pack candidates from an earlier experiment; empty does not mean no court exists',
            'saved_reference': packed['references'][case_id],
            'reference_note': 'Existing annotation only; not evidence of camera stability or automatic generation',
        })
assert len(rows) == 13
rows.sort(key=lambda row: (['gxBQ', 'am2', 'am3'].index(row['video_group']), row['frame_index']))
payload = {'schema': 'cached-temporal-review/1', 'basis_commit': '7299ff3',
           'new_geometry_or_stability_measurement': False, 'rows': rows}
raw = json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()
for forbidden in [b'/home/', b'/scratch/', b'local_scratch', b'carmack', b'password']:
    assert forbidden not in raw.lower(), forbidden
path = OUTPUT / 'temporal_records.json.gz'
path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
assert read(path) == payload
parts = ['<!doctype html><meta charset="utf-8"><title>Cached frames for temporal assessment</title>',
         '<style>body{font-family:system-ui;max-width:1100px;margin:2em auto;padding:0 1em}'
         'img{width:100%;height:auto}figure{margin:2em 0}figcaption{margin:.5em 0}</style>',
         '<h1>Cached frames for temporal assessment</h1><p>These sparse frames are grouped by video, '
         'not by verified stable camera view. Existing overlays are historical hypotheses. '
         'No alignment, stability test or temporal fitting has been performed for this packet. '
         'Use static scene structure to assess the images; do not use annotation agreement as proof.</p>',
         '<p><a href="temporal_assessment.md">Evidence and limits</a> · '
         '<a href="temporal_records.json.gz">Saved numerical evidence</a></p>']
for row in rows:
    stamp = row['nominal_seconds_from_frame_zero']
    timing = f'{stamp:.3f} nominal seconds' if stamp is not None else 'seconds unavailable; frame index preserved'
    caption = f"{row['case_id']} — {timing}. {row['image_kind']}."
    parts.append(f'<figure><figcaption>{html.escape(caption)}</figcaption>'
                 f'<img loading="lazy" src="{html.escape(row["image"])}" alt="{html.escape(caption)}"></figure>')
(OUTPUT / 'temporal_view.html').write_text('\n'.join(parts))
print('Exported', len(rows), 'cached cases;', path.stat().st_size, 'compressed bytes; no new image files')
