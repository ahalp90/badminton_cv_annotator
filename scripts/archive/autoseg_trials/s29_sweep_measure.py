"""Findings-only S29 threshold sweep over the four S28 sticky cells."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 's29_sweep_outputs'
PIN_PATH = HERE / 's28_sticky_pin.py'
REFERENCE = Path('/home/ariel/Documents/COSC594/badminton_stroke_classification/local_scratch/autograder_architecture')
CELL_ORDER = ('m4/r9', 'm4/r7', 'm2/r9', 'm2/r7')
VIDEOS = ('pilot', 'vid15')
BANDS = (0.95, 0.925, 0.90, 0.85, 0.80)
Z = 1.959963984540054


def load_modules():
    spec = importlib.util.spec_from_file_location('s28_sticky_pin', PIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError('cannot load s28 sticky pin')
    pin = importlib.util.module_from_spec(spec)
    sys.modules['s28_sticky_pin'] = pin
    spec.loader.exec_module(pin)
    import m_miss_junk_census as census
    assert Path(pin.__file__).resolve() == PIN_PATH.resolve()
    assert Path(census.__file__).resolve() == HERE / 'm_miss_junk_census.py'
    source = Path(inspect.getsourcefile(census.span_junctions)).resolve()
    assert source == REFERENCE / 'j4_miss_kinematics.py', source
    print(f'import stanza: pin={pin.__file__} census={census.__file__}')
    print(f'span_junctions source: {source}')
    return pin, census


def dec_range(start: int, stop: int, scale: int, step: int) -> list[float]:
    return [i / scale for i in range(start, stop + 1, step)]


NONE = 'NONE'
BURST = [NONE, 0.0] + dec_range(5, 100, 1000, 5) + dec_range(12, 50, 100, 2) + dec_range(55, 300, 100, 5)
RUN = list(range(31)) + list(range(35, 151, 5))
PCT = [NONE, 0.0] + dec_range(25, 500, 10000, 25) + dec_range(6, 50, 100, 1)
PRODUCT_T = dec_range(10, 600, 20, 10) + dec_range(31, 120, 1, 3) + dec_range(130, 400, 1, 10)


def fmt(value):
    if value is None or value == NONE:
        return '' if value is None else NONE
    if isinstance(value, (float, np.floating)):
        return '' if not np.isfinite(value) else f'{float(value):.6f}'
    return str(value)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt(row.get(column)) for column in columns})


def wilson(success: int, total: int) -> tuple[float | None, float | None]:
    if not total:
        return None, None
    p = success / total
    d = 1 + Z * Z / total
    centre = (p + Z * Z / (2 * total)) / d
    half = Z * math.sqrt(p * (1 - p) / total + Z * Z / (4 * total * total)) / d
    return centre - half, centre + half


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort')
    result = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        result[order[index:end]] = (index + 1 + end) / 2
        index = end
    return result


def signal_for(chain, cfg, track, dead, pin, census):
    masked = pin.stage8.apply_replay_mask(track, dead)
    by_frame = {}
    for contact in chain.filtered_contacts:
        if contact.wrist_near is False or contact.suppressed is True:
            continue
        by_frame[contact.contact_frame] = contact.rally_id
    result = []
    span_junction_cache = {}
    for contact in chain.filtered_contacts:
        if contact.wrist_near is False or contact.suppressed is True:
            continue
        start, end = chain.spans[contact.rally_id]
        if contact.rally_id not in span_junction_cache:
            span_junction_cache[contact.rally_id] = census.span_junctions(masked, start, end)
        junctions = span_junction_cache[contact.rally_id]
        if junctions is None:
            raise AssertionError(f'{cfg.name} span {contact.rally_id} has a surviving candidate')
        _angle, speed_in, speed_out, _visible = junctions
        frame = contact.contact_frame
        local = frame - start - 1
        assert 0 <= local < len(speed_in)
        incoming, outgoing = float(speed_in[local]), float(speed_out[local])
        burst = outgoing / incoming if np.isfinite(incoming) and incoming != 0 and np.isfinite(outgoing) else float('nan')
        visible_run = 0
        next_frame = frame + 1
        while next_frame < end and not dead[next_frame] and track[next_frame, 2] == 1:
            visible_run += 1
            next_frame += 1
        result.append({'cell': '', 'video': cfg.name, 'rally_id': contact.rally_id, 'frame': frame,
                       'burst_ratio': burst, 'visible_run': visible_run, 'span_end_gap': end - frame,
                       'run_truncated': visible_run == end - frame - 1})
    assert len(result) == len(by_frame) and len({row['frame'] for row in result}) == len(result)
    return result


def match_count(census, gt, records, keep):
    candidates = [row['frame'] for row, yes in zip(records, keep) if yes]
    return len(census._global_matches(gt, candidates))


def ranked_pairs(gt, records):
    return sorted((abs(gt_frame - row['frame']), gt_index, candidate_index)
                  for gt_index, gt_frame in enumerate(gt)
                  for candidate_index, row in enumerate(records)
                  if abs(gt_frame - row['frame']) <= 10)


def fast_match_count(pairs, keep):
    claimed_gt = set()
    claimed_candidates = set()
    for _distance, gt_index, candidate_index in pairs:
        if keep[candidate_index] and gt_index not in claimed_gt and candidate_index not in claimed_candidates:
            claimed_gt.add(gt_index)
            claimed_candidates.add(candidate_index)
    return len(claimed_gt)


def key_order(row):
    def none(v): return -1 if v == NONE else v
    if row['shape'] == 'PROD':
        return (row['product_T'], row['product_cap'], none(row['guard_run_floor']), none(row['guard_burst_floor']))
    return (row['run_floor'], none(row['burst_floor']))


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    pin, census = load_modules()
    stage8 = pin.stage8
    assert stage8.CONTACT_IMPULSE_MULTIPLE == 4.0
    run_root = OUT / '_rallies'
    runs = {}
    m4 = {9: pin.run_radius(9, run_root), 7: pin.run_radius(7, run_root)}
    expected = {('m4/r9','pilot'):'ddc4f60b058f03e85c326dd5f460924d', ('m4/r9','vid15'):'e0fa89414b44e1b4453e3a6f00f80ac6',
                ('m4/r7','pilot'):'c259e147f36d5e848ccabc27ca41ba0b', ('m4/r7','vid15'):'c332cfb257285daf4346f59a314e7bfe'}
    for cell, radius in (('m4/r9',9),('m4/r7',7)):
        for video in VIDEOS:
            assert m4[radius][video][1] == expected[(cell,video)]
            runs[(cell,video)] = m4[radius][video][0]
    print('m4 md5 tripwire: PASS')
    try:
        stage8.CONTACT_IMPULSE_MULTIPLE = 2.0
        for cell, radius in (('m2/r9',9),('m2/r7',7)):
            for cfg in (pin.harness.retest.PILOT, pin.harness.retest.VID15):
                runs[(cell,cfg.name)], _ = pin.run_video(cfg, radius, run_root, score_output=False)
    finally:
        stage8.CONTACT_IMPULSE_MULTIPLE = 4.0
    assert stage8.CONTACT_IMPULSE_MULTIPLE == 4.0
    print('m2 patch/restore: PASS')

    master = pin.harness.pd.read_csv(pin.harness.retest.SHOTS_MASTER)
    cfgs = {'pilot': pin.harness.retest.PILOT, 'vid15': pin.harness.retest.VID15}
    arrays = {v: (np.load(cfgs[v].track_path), np.load(cfgs[v].mask_path)) for v in VIDEOS}
    sticky = {}
    with (HERE / 's28_sticky_measure_outputs/sticky_cells.csv').open() as handle:
        for row in csv.DictReader(handle): sticky[(row['cell'],row['video'])] = row
    census_rows = {}
    with (HERE / 's27_census_outputs/junk_census.csv').open() as handle:
        for row in csv.DictReader(handle):
            k=(row['video'],int(row['frame'])); assert k not in census_rows; census_rows[k]=row

    all_records = {}; all_grid=[]; baselines={}; gt_by_video={}; pair_cache={}
    for video in VIDEOS:
        gt_rallies = pin.harness.retest.load_gt_rallies(master, cfgs[video].vid)
        gt_by_video[video] = [f for rally in gt_rallies for f in rally.stroke_frames]
        assert len(gt_by_video[video]) == (1641 if video == 'pilot' else 824)
    for cell in CELL_ORDER:
        for video in VIDEOS:
            records = signal_for(runs[(cell,video)], cfgs[video], *arrays[video], pin, census)
            for row in records: row['cell'] = cell
            all_records[(cell,video)] = records
            frames={r['frame'] for r in records}; shared=frames & {k[1] for k in census_rows if k[0]==video}
            cell_only=frames-shared; census_only={k[1] for k in census_rows if k[0]==video}-frames
            print(f'{cell} {video} signal cross-check shared={len(shared)} cell_only={len(cell_only)} census_only={len(census_only)}')
            for frame in shared:
                a=next(r for r in records if r['frame']==frame); b=census_rows[(video,frame)]
                assert fmt(a['burst_ratio']) == b['burst_ratio_speed_out_in'] and str(a['visible_run']) == b['post_flag_visible_run_frames']
            gt=gt_by_video[video]; matches=census._global_matches(gt,[r['frame'] for r in records])
            pair_cache[(cell, video)] = ranked_pairs(gt, records)
            matched={i for i,_ in matches};
            for i,r in enumerate(records): r['matched_baseline']=i in {ci for _,ci in matches}
            baselines[(cell,video)] = (len(matches),len(gt),len(records))
            print(f'{cell} {video}: extreme_burst={sum(r["burst_ratio"]>100 for r in records)} nan_burst={sum(not np.isfinite(r["burst_ratio"]) for r in records)}')
            finite=[r for r in records if np.isfinite(r['burst_ratio'])]
            for r in records: r['pct_burst']=float('nan'); r['pct_run']=0.0
            if finite:
                br=ranks(np.array([r['burst_ratio'] for r in finite])) / len(records)
                rr=ranks(np.array([r['visible_run'] for r in records])) / len(records)
                for r,x in zip(finite,br): r['pct_burst']=x
                for r,x in zip(records,rr): r['pct_run']=x
            paired=[r for r in records if np.isfinite(r['burst_ratio'])]
            corr=np.nan if len(paired)<2 else np.corrcoef(ranks(np.array([r['burst_ratio'] for r in paired])), ranks(np.array([r['visible_run'] for r in paired])))[0,1]
            print(f'{cell} {video} spearman paired={len(paired)} value={corr}')
        assert sum(baselines[(cell,v)][2] for v in VIDEOS) == {'m4/r9':2872,'m4/r7':2960,'m2/r9':3912,'m2/r7':4245}[cell]

    for cell in CELL_ORDER:
        for video in VIDEOS:
            s=sticky[(cell,video)]; base=baselines[(cell,video)]
            assert (base[0],base[1],base[2]) == (int(s['matches']),int(s['gt']),int(s['candidates']))
            assert fmt(base[0]/base[1]) == s['recall'] and fmt(base[0]/base[2]) == s['precision']
    print('no-op corner and candidate-count assertions: PASS')

    print('FEASIBILITY BUDGET')
    for cell in CELL_ORDER:
        for video in VIDEOS:
            baseline_matches, gt_count, _candidate_count = baselines[(cell, video)]
            entries = []
            for band in BANDS:
                required = math.ceil(band * gt_count)
                allowed = baseline_matches - required
                entries.append(f'{band:g}:losses={allowed}:feasible={allowed >= 0}')
            print(f'  {cell} {video}: baseline_matches={baseline_matches} GT={gt_count} '
                  f'recall={baseline_matches / gt_count:.6f} ' + ' '.join(entries))
    print('feasibility budget gate: PASS')

    # Add percentile thresholds to records, then materialise the deterministic grid.
    shapes = ('AND','OR','PROD','PCT-AND','PCT-OR')
    grid_columns=['cell','shape','burst_floor','run_floor','product_cap','product_T','guard_burst_floor','guard_run_floor']
    for prefix in ('pilot','vid15','pooled'):
        grid_columns += [f'{prefix}_{x}' for x in ('kept','matches','recall','precision')]
    grid_columns += ['pilot_baseline_matched_kept','pilot_baseline_junk_kept','vid15_baseline_matched_kept','vid15_baseline_junk_kept']
    frontiers=[]; pareto=[]; lovo=[]; overlap=[]; search_counts={}
    for cell in CELL_ORDER:
      for shape in shapes:
        params=[]
        if shape in ('AND','OR','PCT-AND','PCT-OR'):
            bs = BURST if shape in ('AND','OR') else PCT
            run_values = RUN if shape in ('AND','OR') else [value for value in PCT if value != NONE]
            for b in bs:
              for r in run_values: params.append({'burst_floor':b,'run_floor':r})
        else:
            for cap in (60,120):
              for t in PRODUCT_T:
               for b in (NONE,0.01,0.05,0.1,0.2):
                for r in (0,5,10,20): params.append({'product_cap':cap,'product_T':t,'guard_burst_floor':b,'guard_run_floor':r})
        rows=[]; noops=[]
        for count,p in enumerate(params,1):
            masks={}
            for video in VIDEOS:
                rec=all_records[(cell,video)]; b=p.get('burst_floor'); rf=p.get('run_floor')
                pct='PCT' in shape; burst_floor=b
                bp=np.ones(len(rec),bool) if shape == 'PROD' or b == NONE else np.array([(x['pct_burst'] if pct else x['burst_ratio']) >= b for x in rec])
                rp=np.ones(len(rec),bool) if shape == 'PROD' else np.array([(x['pct_run'] if pct else x['visible_run']) >= rf for x in rec])
                if shape.endswith('AND') or shape=='AND': keep=bp & rp
                elif shape.endswith('OR') or shape=='OR': keep=bp | rp
                else:
                    cap=p['product_cap']; t=p['product_T']; gb=p['guard_burst_floor']; gr=p['guard_run_floor']
                    guardb=np.ones(len(rec),bool) if gb==NONE else np.array([x['burst_ratio']>=gb for x in rec])
                    keep=guardb & (np.array([x['visible_run']>=gr for x in rec])) & bp & rp & np.array([np.isfinite(x['burst_ratio']) and x['burst_ratio']*min(x['visible_run'],cap)>=t for x in rec])
                masks[video]=keep
            operational=bool(masks['pilot'].all() and masks['vid15'].all())
            if operational or count % 50 == 0:
                for video in VIDEOS:
                    direct=match_count(census,gt_by_video[video],all_records[(cell,video)],masks[video])
                    pairs=[(abs(g-r['frame']),i,j) for i,g in enumerate(gt_by_video[video]) for j,r in enumerate(all_records[(cell,video)]) if masks[video][j] and abs(g-r['frame'])<=10]
                    # The direct call is the equivalence oracle; cached pairs are used below.
                    ranked=sorted(pairs); claimed=set(); claimed_c=set()
                    for _d,gi,ci in ranked:
                        if gi not in claimed and ci not in claimed_c: claimed.add(gi); claimed_c.add(ci)
                    assert direct == len(claimed)
                noops.append(count) if operational else None
            row={**p,'cell':cell,'shape':shape}
            for video in VIDEOS:
                rec=all_records[(cell,video)]; keep=masks[video]; matches=fast_match_count(pair_cache[(cell, video)], keep); k=int(keep.sum()); gt=len(gt_by_video[video])
                row[f'{video}_kept']=k; row[f'{video}_matches']=matches; row[f'{video}_recall']=matches/gt; row[f'{video}_precision']=matches/k if k else 0.0
                row[f'{video}_baseline_matched_kept']=sum(keep[i] and x['matched_baseline'] for i,x in enumerate(rec)); row[f'{video}_baseline_junk_kept']=sum(keep[i] and not x['matched_baseline'] for i,x in enumerate(rec))
            row['pooled_kept']=row['pilot_kept']+row['vid15_kept']; row['pooled_matches']=row['pilot_matches']+row['vid15_matches']; row['pooled_recall']=row['pooled_matches']/2465; row['pooled_precision']=row['pooled_matches']/row['pooled_kept'] if row['pooled_kept'] else 0.0
            rows.append(row)
        search_counts[(cell,shape)]=len(rows); all_grid.extend(rows)
        print(f'fast-path equivalence sample {cell} {shape}: PASS ({len(noops)} no-op points)')
        def best(band, metric='pooled'):
            feasible=[x for x in rows if x[f'{metric}_recall'] >= band]
            if not feasible:return None
            return max(feasible,key=lambda x:(x[f'{metric}_precision'],x[f'{metric}_recall'],tuple(-999 if v==NONE else -v for v in key_order(x))))
        for band in BANDS:
            for selection,metric in (('pooled','pooled'),):
                pick=best(band,metric)
                frontiers.append(frontier_row(cell,shape,band,selection,pick,search_counts[(cell,shape)]))
            if band in (0.95,0.90,0.85,0.80):
                pick=best(band,'min_video') if False else None
                candidates=[x for x in rows if min(x['pilot_recall'],x['vid15_recall'])>=band]
                if candidates: pick=max(candidates,key=lambda x:(min(x['pilot_precision'],x['vid15_precision']),min(x['pilot_recall'],x['vid15_recall']),tuple(-999 if v==NONE else -v for v in key_order(x))))
                frontiers.append(frontier_row(cell,shape,band,'min_video',pick,search_counts[(cell,shape)]))
        # Pareto retains every row sharing a non-dominated metric pair.
        for x in rows:
            if not any((y['pooled_recall']>=x['pooled_recall'] and y['pooled_precision']>=x['pooled_precision'] and (y['pooled_recall']>x['pooled_recall'] or y['pooled_precision']>x['pooled_precision'])) for y in rows): pareto.append({**x,'multiplicity':len(rows)})
        for tuning in VIDEOS:
          other='vid15' if tuning=='pilot' else 'pilot'
          for band in (0.95,0.90,0.85,0.80):
            feasible=[x for x in rows if x[f'{tuning}_recall']>=band]
            pick=max(feasible,key=lambda x:(x[f'{tuning}_precision'],x[f'{tuning}_recall'],tuple(-999 if v==NONE else -v for v in key_order(x)))) if feasible else None
            if pick is None: lovo.append({'cell':cell,'shape':shape,'band':band,'tuning_video':tuning,'status':'skipped'})
            else: lovo.append({'cell':cell,'shape':shape,'band':band,'tuning_video':tuning,'status':'scored','tuned_recall':pick[f'{tuning}_recall'],'tuned_precision':pick[f'{tuning}_precision'],'transferred_recall':pick[f'{other}_recall'],'transferred_precision':pick[f'{other}_precision'],'recall_gap':pick[f'{other}_recall']-pick[f'{tuning}_recall'],'precision_gap':pick[f'{other}_precision']-pick[f'{tuning}_precision'],**pick})
      # overlap uses highest feasible pooled RAW AND frontier.
      raw=[x for x in frontiers if x['cell']==cell and x['shape']=='AND' and x['selection']=='pooled' and x['status']=='feasible']
      if raw:
        chosen=max(raw,key=lambda x:x['band']); b=chosen['burst_floor']; r=chosen['run_floor']
        for matched in (True,False):
          failb=failr=both=neither=trunc=andtrunc=0
          for v in VIDEOS:
            for x in all_records[(cell,v)]:
              if x['matched_baseline'] != matched: continue
              fb=(b!=NONE and not (x['burst_ratio']>=b)); fr=x['visible_run']<r
              if fb and fr: both+=1
              elif fb: failb+=1
              elif fr: failr+=1
              else: neither+=1
              if fr and x['run_truncated']: trunc+=1
              if fb and fr and x['run_truncated']: andtrunc+=1
          overlap.append({'cell':cell,'band':chosen['band'],'matched_baseline':matched,'fails_burst_only':failb,'fails_run_only':failr,'fails_both':both,'fails_neither':neither,'and_killed_matched_run_truncated':andtrunc if matched else 0,'run_test_failed_matched_run_truncated':trunc if matched else 0})

    candidate_rows=[r for cell in CELL_ORDER for video in VIDEOS for r in all_records[(cell,video)]]
    write_csv(OUT/'candidates.csv','cell video rally_id frame burst_ratio visible_run span_end_gap run_truncated matched_baseline pct_burst pct_run'.split(),candidate_rows)
    write_csv(OUT/'grid.csv',grid_columns,all_grid)
    front_cols=['cell','shape','selection','band','status','burst_floor','run_floor','product_cap','product_T','guard_burst_floor','guard_run_floor','multiplicity','pooled_recall','pooled_precision','pooled_recall_low','pooled_recall_high','pooled_precision_low','pooled_precision_high','pilot_recall','pilot_precision','vid15_recall','vid15_precision']
    write_csv(OUT/'frontier.csv',front_cols,frontiers)
    write_csv(OUT/'pareto.csv',grid_columns+['multiplicity'],pareto)
    write_csv(OUT/'lovo.csv',['cell','shape','band','tuning_video','status','burst_floor','run_floor','product_cap','product_T','guard_burst_floor','guard_run_floor','tuned_recall','tuned_precision','transferred_recall','transferred_precision','recall_gap','precision_gap'],lovo)
    write_csv(OUT/'overlap.csv','cell band matched_baseline fails_burst_only fails_run_only fails_both fails_neither and_killed_matched_run_truncated run_test_failed_matched_run_truncated'.split(),overlap)
    elapsed=time.perf_counter()-started; print(f'feasibility budget printed for bands {BANDS}; pooled GT=2465')
    print(f'wall time seconds={elapsed:.3f}')


def frontier_row(cell, shape, band, selection, pick, multiplicity):
    if pick is None:return {'cell':cell,'shape':shape,'selection':selection,'band':band,'status':'infeasible','multiplicity':multiplicity}
    row={k:pick.get(k) for k in ('burst_floor','run_floor','product_cap','product_T','guard_burst_floor','guard_run_floor')}
    row.update({'cell':cell,'shape':shape,'selection':selection,'band':band,'status':'feasible','multiplicity':multiplicity})
    for metric in ('pooled','pilot','vid15'):
        recall=pick[f'{metric}_recall']; precision=pick[f'{metric}_precision']; total=2465 if metric=='pooled' else (1641 if metric=='pilot' else 824); matches=pick[f'{metric}_matches'] if metric in ('pilot','vid15') else pick['pooled_matches']; kept=pick[f'{metric}_kept'] if metric in ('pilot','vid15') else pick['pooled_kept']
        lo,hi=wilson(matches,total); pl,ph=wilson(matches,kept)
        row[f'{metric}_recall']=recall;row[f'{metric}_precision']=precision;row[f'{metric}_recall_low']=lo;row[f'{metric}_recall_high']=hi;row[f'{metric}_precision_low']=pl;row[f'{metric}_precision_high']=ph
    return row


if __name__ == '__main__': main()
