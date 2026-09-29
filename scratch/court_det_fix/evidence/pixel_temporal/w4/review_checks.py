#!/usr/bin/env python3
"""Repeat the W4 calculations from saved numbers.

By default, this reads the copied fields included in inputs/.
--l3-csv reads your original local score_matrix.csv instead.
No detector runs, new courts, image scoring or visual judgements are involved.
Only Python's standard library is needed.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

HERE = Path(__file__).resolve().parent
FRAMES = [0, 5, 689, 5111, 5766, 77876, 86088]
PREFIX = 'gxBQ_window_00_frame_'


def read_json(path: Path) -> Any:
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def corner_error(corners: list, reference: list) -> tuple[float, float, float]:
    if len(corners) != 4 or len(reference) != 4:
        raise ValueError('Four corners are required.')
    direct = max(math.dist(a, b) for a, b in zip(corners, reference))
    rotated_ref = reference[2:] + reference[:2]
    rotated = max(math.dist(a, b) for a, b in zip(corners, rotated_ref))
    return min(direct, rotated), direct, rotated


def c2_check(path: Path) -> dict:
    source = read_json(path)
    computed = []
    for row in source['gx_rows']:
        value, direct, rotated = corner_error(row['corners'], source['gx_control'])
        if abs(value - row['reported_error']) > 1e-10:
            raise ValueError(f'The corner calculation does not match the saved value: {row["name"]}')
        computed.append(dict(name=row['name'], candidate_id=row['candidate_id'],
                             recomputed_error=value, direct=direct, rotated=rotated,
                             reported_error=row['reported_error']))
    ordered = sorted(source['am3_rows'], key=lambda row: -row['score'])
    same_key = [row for row in ordered if row['assignment'] == [7,197,32,3,19]]
    return {
        'source_revision': source['revision'], 'source_path': source['source_path'],
        'method': 'Measure all four corner distances and keep the largest. Repeat with labels turned 180 degrees; use the smaller result.',
        'gx': computed,
        'proxy_delta_px': computed[2]['recomputed_error'] - computed[0]['recomputed_error'],
        'pre_global_delta_px': computed[3]['recomputed_error'] - computed[1]['recomputed_error'],
        'am3_score_order_in_supplied_witnesses': [row['id'] for row in ordered],
        'same_key_representative_in_supplied_witnesses': same_key[0]['id'],
        'representative_score_advantage': same_key[0]['score'] - same_key[1]['score'],
        'am3_error_recalculated': False,
        'boundaries': 'The GX errors were calculated again from the copied corners. This did not repeat the full search or check every rank. The Am3 comparison checks assignment IDs and scores, not the reported corner errors.'
    }


def load_csv_scores(path: Path) -> list[dict]:
    """Read scores, not reference errors. Check that all 30 courts can compete on all seven frames."""
    cells: dict[str, dict[int, tuple[float, float]]] = {}
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            frame = int(row['frame_index'])
            cid = row['court_id']
            if frame not in FRAMES or row['status'] != 'ok':
                raise ValueError('A frame is unexpected or a court cannot compete. The script has not skipped that row.')
            if row['geometry_valid'].lower() != 'true' or float(row['camera_error']) > .1:
                raise ValueError('The court checks differ from those in the reviewed test.')
            item = cells.setdefault(cid, {})
            if frame in item:
                raise ValueError(f'Duplicate matrix cell: {cid}, {frame}')
            item[frame] = (float(row['line_score']), float(row['paint_profile_score']))
    if len(cells) != 30 or any(set(cell) != set(FRAMES) for cell in cells.values()):
        raise ValueError('Expected all 30 courts on all seven frames, with none missing.')
    return [dict(court_id=cid, line=[cell[f][0] for f in FRAMES],
                 paint=[cell[f][1] for f in FRAMES], status=['ok']*7)
            for cid, cell in sorted(cells.items())]


def rank_sum(rows: list[dict]) -> dict:
    """Rank each score from best to worst, starting at 1. Ties use the full court ID as text."""
    rows = sorted(rows, key=lambda row: row['court_id'])
    ids = [row['court_id'] for row in rows]
    if len(ids) != 30 or len(set(ids)) != 30:
        raise ValueError('Expected the same 30 source IDs. This calculation does not merge similar court shapes.')
    for row in rows:
        if row['status'] != ['ok']*7:
            raise ValueError('Not all courts pass the saved checks on every frame.')
        for cue in ('line', 'paint'):
            if len(row[cue]) != 7 or not all(math.isfinite(x) for x in row[cue]):
                raise ValueError('Each court needs seven finite numbers for each score.')
    ranks = {cue: [[0]*7 for _ in rows] for cue in ('line','paint')}
    for cue in ranks:
        for t in range(7):
            order = sorted(range(30), key=lambda i: (-rows[i][cue][t], ids[i]))
            for position, i in enumerate(order, 1):
                ranks[cue][i][t] = position
    sums = [[a+b for a,b in zip(ranks['line'][i], ranks['paint'][i])] for i in range(30)]
    order = sorted(range(30), key=lambda i: (sum(sums[i]), ids[i]))
    independent = [min(range(30), key=lambda i: (-rows[i]['line'][t],ids[i])) for t in range(7)]
    median_winner = min(range(30), key=lambda i: (-median(rows[i]['line']),ids[i]))
    nondominated = [ids[i] for i in range(30) if not any(
        all(b>=a for a,b in zip(rows[i]['line'],rows[j]['line'])) and
        any(b>a for a,b in zip(rows[i]['line'],rows[j]['line']))
        for j in range(30) if j != i)]
    return {
        'protocol': 'Rule from scratch/court_det_fix/next_steps_20260916/L3_temporal/W3_PACKET.md at b36402f: rank line and paint scores separately on each frame, best first. Tied scores use the full court ID as text. Add the ranks, then average over seven frames. A final tie would also use the ID; there is no tie for first place here.',
        'frames': FRAMES, 'panel_origin_ids': ids, 'common_eligible_count': 30,
        'same_panel_used_for_all_selectors': True,
        'winner': ids[order[0]],
        'independent_line_winners': [ids[i] for i in independent],
        'shared_median_line_winner': ids[median_winner],
        'line_nondominated_ids': nondominated,
        'ranking': [dict(court_id=ids[i], total_rank_sum=sum(sums[i]),
                         mean_rank_sum=mean(sums[i]), line_ranks=ranks['line'][i],
                         paint_ranks=ranks['paint'][i], per_frame_rank_sum=sums[i],
                         line_vector=rows[i]['line'], paint_vector=rows[i]['paint']) for i in order]
    }


def csv_references(path: Path) -> dict[str, list[float]]:
    """Read the saved errors only after the chosen court has been written to disk."""
    values: dict[str, dict[int,float]] = {}
    with path.open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            text = row.get('reference_max_corner_working_px', '')
            if text:
                values.setdefault(row['court_id'], {})[int(row['frame_index'])] = float(text)
    return {cid:[value[f] for f in FRAMES] for cid,value in values.items() if set(value)==set(FRAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'results')
    parser.add_argument('--l3-csv', type=Path, help='Read your original score_matrix.csv. Results go to --output, not to the input file.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    c2 = c2_check(HERE/'inputs/c2_compact.json')
    write_json(args.output/'c2_check.json', c2)
    compact = read_json(HERE/'inputs/l3_scores_compact.json')
    rows = load_csv_scores(args.l3_csv) if args.l3_csv else compact['rows']
    if args.l3_csv:
        # Check that this is the same data. Matching files does not show that a court is correct.
        actual = sorted(rows,key=lambda row:row['court_id'])
        expected = sorted(compact['rows'],key=lambda row:row['court_id'])
        if actual != expected:
            raise ValueError('The original CSV differs from the included copy. Check the files and revision before using this result.')
    result = rank_sum(rows)
    result['input_kind'] = 'original local CSV' if args.l3_csv else 'copied fields from the original result files; not the original CSV bytes'
    result['source_revision'] = compact['revision']
    result['source_path'] = compact['source_path']
    # rank_sum cannot read the reference errors. Write its choice before loading those errors.
    write_json(args.output/'selection_lock.json', result)
    references = csv_references(args.l3_csv) if args.l3_csv else read_json(
        HERE/'inputs/l3_recorded_reference_errors.json')['errors']
    origin_winners = [PREFIX+'0::22:4579',PREFIX+'0::22:4588',PREFIX+'5::181:29836',PREFIX+'5::10:1274']
    targets = list(dict.fromkeys([result['winner'],result['shared_median_line_winner'],*origin_winners]))
    indexed = {row['court_id']:row for row in result['ranking']}
    result['post_lock_reference_comparison'] = [dict(
        court_id=cid, line_vector=indexed[cid]['line_vector'], paint_vector=indexed[cid]['paint_vector'],
        mean_rank_sum=indexed[cid]['mean_rank_sum'],
        recorded_error_vector=references.get(cid),
        recorded_error_median=median(references[cid]) if cid in references else None,
        fresh_visual_judgement='not measured') for cid in targets]
    result['reference_boundary'] = 'The saved annotation errors were read after choosing the court. They were not measured from corners again and do not use C2s approved court-89 reference. No new image judgement was made.'
    result['original_median_line_independent_error_vector'] = [references[cid][t] for t,cid in enumerate(result['independent_line_winners'])]
    result['original_median_line_independent_error_median'] = median(result['original_median_line_independent_error_vector'])
    write_json(args.output/'l3_rank_sum.json', result)
    with (args.output/'l3_ranks.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream); writer.writerow(['court_id','total_rank_sum','mean_rank_sum','line_ranks','paint_ranks'])
        for row in result['ranking']:
            writer.writerow([row['court_id'],row['total_rank_sum'],row['mean_rank_sum'],json.dumps(row['line_ranks']),json.dumps(row['paint_ranks'])])
    print('C2: distances calculated from the saved GX corners')
    for row in c2['gx']:
        print(f"  {row['name']}: {row['recomputed_error']:.6f} px")
    print(f"  Pair-143 change: {c2['proxy_delta_px']:+.6f} px")
    print(f"  Change across saved shortlists: {c2['pre_global_delta_px']:+.6f} px")
    print('  Higher-scoring Am3 duplicate:', c2['same_key_representative_in_supplied_witnesses'])
    print('L3: the same 30 courts, scored on all seven frames')
    print('  Winner using only the median line score:', result['shared_median_line_winner'])
    print('  Winner combining line and paint ranks:', result['winner'])
    print('  Average rank total:', result['ranking'][0]['mean_rank_sum'])
    print('  Median saved corner error:', result['post_lock_reference_comparison'][0]['recorded_error_median'])
    print('No images were viewed or scored by this script. A lower error is not a new visual approval.')
    print('Result files written to:', args.output.resolve())

if __name__ == '__main__':
    try:
        main()
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise SystemExit(f'W4 calculation stopped: {exc}') from exc
