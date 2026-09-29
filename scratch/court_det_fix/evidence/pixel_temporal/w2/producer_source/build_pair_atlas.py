#!/usr/bin/env python3
"""One bounded W2 follow-up: put the two Am2 marking pairs and their pixels together.

No fitting, extraction, scoring change, candidate search, network access, or repository
imports. Reads three hash-pinned inputs using git show (or an exactly matching local
file); writes outside the checkout. Requires the existing NumPy/OpenCV environment.
Physical ownership is deliberately left UNRESOLVED by the script.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import math
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from analyze_fresh import read_rows, CSV_SHA256
from w2_probe import interval_probe, EXPECTED_PANEL

REF = 'bb6787adeca9a19b5f1d3c3324d54cfcb2ea6e68'
SEED = 'scratch/court_det_fix/next_steps_20260916/webui_seed'
WITNESS_PATH = SEED + '/L2_scoring/witnesses.json'
WITNESS_BLOB = '739af3d1dacc7f3fedb551eab01a2b4bad94fd06'
TARGETS = [
    {'case_id':'am2_window_01_frame_28019','candidate_id':'184:4123',
     'intervals':[10,11], 'label':'False Am2-28019: near long service / near baseline',
     'frame':SEED+'/frames/amateur/am2/frame_00028019.png',
     'frame_blob':'a6c1ace2fad3a6f350b7b91fad991840955fd866'},
    {'case_id':'am2_window_00_frame_150','candidate_id':'30:33',
     'intervals':[6,7], 'label':'Approved Am2-150: far baseline / far long service',
     'frame':SEED+'/frames/amateur/am2/frame_00000150.png',
     'frame_blob':'08ab1d43d219320bc18d2bfb25fbde5f5f3cf456'},
]
WORKING_SIZE = (960,540)
COLOURS = [(0,200,255),(255,220,0)]  # Annotation keys only: A circles, B crosses (BGR).


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def git(repo: Path, *args: str) -> bytes:
    proc = subprocess.run(['git','-C',str(repo),*args],capture_output=True,check=False)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors='replace').strip())
    return proc.stdout


def pinned_read(repo: Path, path: str, expected: str, ledger: list[dict]) -> bytes:
    # The fallback accommodates a local packet without requiring any large output commit.
    origin = 'git show '+REF+':'+path
    try:
        data = git(repo,'show',REF+':'+path)
    except RuntimeError:
        data = (repo/path).read_bytes()
        origin = 'verified local file'
    actual = blob_sha(data)
    if actual != expected:
        raise ValueError(f'Immutable input mismatch: {path}: {actual} != {expected}')
    local = repo/path
    local_hash = sha256(local.read_bytes()) if local.is_file() else None
    ledger.append({'path':path,'origin':origin,'git_blob':actual,'sha256':sha256(data),
                   'bytes':len(data),'local_worktree_sha256':local_hash,
                   'local_worktree_matches_used_bytes':local_hash==sha256(data) if local_hash else None})
    return data


def image_write(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path),image):
        raise OSError('Could not write '+str(path))


def sample(gray: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    h,w=gray.shape
    values=cv2.remap(gray,xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    inside=(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)
    return values,inside


def verify_pixel_trace(gray: np.ndarray, interval: dict) -> dict:
    coords = [np.asarray(interval[k],dtype=float) for k in [
        'tested_sample_coordinates_working_px','minus_side_coordinates_working_px',
        'plus_side_coordinates_working_px']]
    measured=[sample(gray,xy) for xy in coords]
    centre,minus,plus=[v[0] for v in measured]
    available=np.logical_and.reduce([v[1] for v in measured])
    c1,c2=centre-minus,centre-plus
    saved1=np.asarray(interval['contrast_centre_minus_minus_side'],dtype=float)
    saved2=np.asarray(interval['contrast_centre_minus_plus_side'],dtype=float)
    residual=max(float(np.max(np.abs(c1-saved1))),float(np.max(np.abs(c2-saved2))))
    passing=available&(np.minimum(c1,c2)>=10)
    checks={'interval':interval['interval'],'max_abs_contrast_replay_difference':residual,
            'availability_equal':np.array_equal(available,np.asarray(interval['available'],dtype=bool)),
            'passing_offsets_equal':np.array_equal(passing,np.asarray(interval['pass_by_shift'],dtype=bool))}
    # 1e-4 is a numerical reproduction tolerance, NOT a court/pixel acceptance threshold.
    checks['passed']=bool(residual<=1e-4 and checks['availability_equal'] and checks['passing_offsets_equal'])
    if not checks['passed']:
        raise ValueError('Pixel replay mismatch; no physical conclusions: '+json.dumps(checks))
    return checks


def verify_uploaded_table(payload: dict, csv_path: Path) -> dict:
    records=read_rows(csv_path)
    index={(r['case_id'],r['candidate_id'],r['interval']):r for r in records}
    panel=payload.get('diagnostic_panel',[])
    if payload.get('schema')!='l2-generation-scoring-witnesses/1' or len(panel)!=4 or {
        (r['case_id'],r['candidate_id']) for r in panel} != EXPECTED_PANEL:
        raise ValueError('Unexpected witness schema or diagnostic panel')
    checked=0
    for candidate in panel:
        trace=candidate['trace']
        for flag in ['interval_pass_matches_detector','saved_profile_reproduced','existing_profile_reproduced']:
            if trace.get(flag) is not True:
                raise ValueError('Saved replay check absent/failed: '+flag)
        if trace['settings'] != {'samples_along':24,'centre_offsets_working_px':[-4.,-2.,0.,2.,4.],
                                  'side_distance_working_px':6.,'minimum_contrast':10.,'minimum_fraction':.4}:
            raise ValueError('Changed sampling settings')
        if len(trace['intervals'])!=12:
            raise ValueError('Expected twelve intervals')
        for interval in trace['intervals']:
            row=index[(candidate['case_id'],candidate['candidate_id'],interval['interval'])]
            calculated=interval_probe(interval)
            for k in ['status','legacy_interval_pass','observed_pass_stations','stations_with_any_measured_offset',
                      'unknown_offset_tests','run_lower_stations','run_upper_stations',
                      'legacy_pass_station_fraction','run_lower_fraction','run_upper_fraction',
                      'clipped_span_working_px','station_spacing_working_px','lower_path_endpoint_span_working_px']:
                a,b=calculated.get(k),row.get(k)
                equal=math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9) if isinstance(a,float) and b is not None else a==b
                if not equal:
                    raise ValueError(f'Uploaded CSV differs from pinned trace: {candidate["candidate_id"]} {interval["interval"]} {k}')
            checked+=1
    return {'uploaded_csv_sha256':CSV_SHA256,'recomputed_interval_rows':checked,'all_equal':True}


def overlay(image: np.ndarray, pair: list[dict], scale: np.ndarray, origin: np.ndarray) -> np.ndarray:
    canvas=image.copy()
    for ordinal,interval in enumerate(pair):
        points=np.asarray(interval['tested_sample_coordinates_working_px'],dtype=float)
        ends=np.rint(points[[0,-1],2]*scale-origin).astype(np.int32)
        cv2.line(canvas,tuple(ends[0]),tuple(ends[1]),COLOURS[ordinal],1,cv2.LINE_AA)
        for xy in points[np.asarray(interval['pass_by_shift'],dtype=bool)]:
            pixel=tuple(np.rint(xy*scale-origin).astype(int))
            if ordinal==0:
                cv2.circle(canvas,pixel,3,(0,0,0),2,cv2.LINE_AA)
                cv2.circle(canvas,pixel,2,COLOURS[ordinal],1,cv2.LINE_AA)
            else:
                cv2.drawMarker(canvas,pixel,(0,0,0),cv2.MARKER_CROSS,7,3,cv2.LINE_AA)
                cv2.drawMarker(canvas,pixel,COLOURS[ordinal],cv2.MARKER_CROSS,5,1,cv2.LINE_AA)
    return canvas


def joint_strip(gray: np.ndarray, pair: list[dict]) -> dict:
    """Shared IMAGE chart, not a homography/court fit and not a new paint test.

    Same physical chart for both intervals; never align their unequal spans by
    station index. Subpixel interpolation here is for display, not new resolution.
    """
    endpoints=[np.asarray(r['tested_sample_coordinates_working_px'],dtype=float)[[0,-1],2] for r in pair]
    a,b=endpoints
    length=float(np.linalg.norm(a[1]-a[0])); u=(a[1]-a[0])/length
    n=np.array([-u[1],u[0]])
    tb=(b-a[0])@u; zb=(b-a[0])@n
    if abs(tb[1]-tb[0])<1e-10:
        raise ValueError('Pair has no usable longitudinal overlap')
    order=np.argsort(tb); tb,zb=tb[order],zb[order]
    lo,hi=max(0.,tb[0]),min(length,tb[-1])
    if hi<=lo:
        raise ValueError('Pair has no finite overlapping span')
    t=np.linspace(lo,hi,int(math.ceil((hi-lo)/.5))+1)
    z_b=np.interp(t,tb,zb)
    zlo=math.floor(min(0.,float(z_b.min()))-12.)
    zhi=math.ceil(max(0.,float(z_b.max()))+12.)
    z=np.linspace(zlo,zhi,int(math.ceil((zhi-zlo)/.25))+1)
    xy=a[0]+t[None,:,None]*u+z[:,None,None]*n
    values,available=sample(gray,xy)
    image=np.rint(values).clip(0,255).astype(np.uint8)
    # Unknown positions get a checker for display; NPZ keeps an explicit mask and NaN.
    iy,ix=np.indices(image.shape)
    checker=96+24*((ix//4+iy//4)%2)
    image=np.where(available,image,checker).astype(np.uint8)
    annotated=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    def to_display(points):
        d=np.asarray(points)-a[0]
        return np.stack((((d@u)-t[0])/(t[-1]-t[0])*(len(t)-1),
                         ((d@n)-z[0])/(z[-1]-z[0])*(len(z)-1)),axis=-1)
    for ordinal,interval in enumerate(pair):
        points=np.asarray(interval['tested_sample_coordinates_working_px'],dtype=float)
        ends=np.rint(to_display(points[[0,-1],2])).astype(np.int32)
        cv2.line(annotated,tuple(ends[0]),tuple(ends[1]),COLOURS[ordinal],1,cv2.LINE_AA)
        for p in np.rint(to_display(points[np.asarray(interval['pass_by_shift'],dtype=bool)])).astype(np.int32):
            cv2.drawMarker(annotated,tuple(p),COLOURS[ordinal],cv2.MARKER_CROSS if ordinal else cv2.MARKER_SQUARE,4,1)
    return {'image':image,'overlay':annotated,'xy':xy,'intensity':np.where(available,values,np.nan),
            'available':available,'t':t,'z':z,'z_b':z_b,'origin':a[0],'tangent':u,'normal':n}


def render_pair(native: np.ndarray, candidate: dict, target: dict, folder: Path) -> dict:
    folder.mkdir()
    work=cv2.resize(native,WORKING_SIZE,interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY).astype(np.float32)
    by_id={r['interval']:r for r in candidate['trace']['intervals']}
    pair=[by_id[i] for i in target['intervals']]
    if not all(r['visible'] for r in pair):
        raise ValueError('Expected visible target intervals')
    checks=[verify_pixel_trace(gray,r) for r in pair]
    scale=np.array([native.shape[1],native.shape[0]])/np.array(WORKING_SIZE)
    cloud=np.concatenate([np.asarray(r[k]).reshape(-1,2) for r in pair for k in [
        'tested_sample_coordinates_working_px','minus_side_coordinates_working_px','plus_side_coordinates_working_px']])
    cloud=cloud[np.isfinite(cloud).all(axis=1)]
    low=np.floor((cloud.min(axis=0)-24)*scale).astype(int)
    high=np.ceil((cloud.max(axis=0)+24)*scale).astype(int)+1
    low=np.maximum(low,0); high=np.minimum(high,[native.shape[1],native.shape[0]])
    if (high<=low).any():
        raise ValueError('Invalid native crop')
    crop=native[low[1]:high[1],low[0]:high[0]]
    image_write(folder/'native_pair_raw.png',crop)
    image_write(folder/'native_pair_overlay.png',overlay(crop,pair,scale,low))
    image_write(folder/'context_raw.png',work)
    image_write(folder/'context_overlay.png',overlay(work,pair,np.ones(2),np.zeros(2)))
    strip=joint_strip(gray,pair)
    image_write(folder/'shared_strip_raw.png',strip['image'])
    image_write(folder/'shared_strip_overlay.png',strip['overlay'])
    np.savez_compressed(folder/'shared_strip.npz',
        coordinates_working_px=strip['xy'],gray_intensity=strip['intensity'],available=strip['available'],
        longitudinal_working_px=strip['t'],normal_working_px=strip['z'],
        second_nominal_line_normal_working_px=strip['z_b'])
    result={'case_id':target['case_id'],'candidate_id':target['candidate_id'],
            'population':'ORIGINAL automatic_all_camera','label':target['label'],
            'existing_ruling':candidate['existing_ruling'],'physical_ownership':'UNRESOLVED; not assigned by this script',
            'frame_git_blob':target['frame_blob'],'working_size':list(WORKING_SIZE),
            'native_size':[native.shape[1],native.shape[0]],'native_per_working_xy':scale.tolist(),
            'native_crop_origin_xy':low.tolist(),'native_crop_end_exclusive_xy':high.tolist(),
            'pixel_replay_checks':checks,'interval_metrics':[interval_probe(r) for r in pair],
            'original_intervals':pair,
            'shared_chart':{'origin_working_px':strip['origin'].tolist(),'tangent':strip['tangent'].tolist(),
                            'normal':strip['normal'].tolist(),'common_span_working_px':[float(strip['t'][0]),float(strip['t'][-1])],
                            'display_sampling_only':'<=0.5 px along, <=0.25 px normal; interpolation is NOT resolution or acceptance',
                            'unknowns':'explicit NPZ mask/NaN; checker in the display'},
            'annotation':'A/first interval: amber circles; B/second: cyan crosses. Nominal tracks are not observed ridges.'}
    write_json(folder/'pair_trace.json',result)
    return result


def make_html(output: Path, targets: list[dict]) -> None:
    body=['<!doctype html><meta charset="utf-8"><title>W2 paired pixel witness</title>',
          '<style>body{font:17px sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem}img{max-width:100%;height:auto}.row{display:grid;grid-template-columns:1fr 1fr;gap:1rem}figure{margin:0}section{margin-bottom:3rem}pre{white-space:pre-wrap}h2{margin-top:2rem}</style>',
          '<h1>Two fixed Am2 pairs</h1><p>Inspect raw pixels before overlays. The script has not labelled physical ownership or changed a score. A: first interval, amber circles. B: second interval, cyan crosses.</p>']
    for target in targets:
        name=target['candidate_id'].replace(':','_'); label=html.escape(target['label'])
        body.append(f'<section><h2>{label}</h2>')
        for a,b,caption in [('context_raw.png','context_overlay.png','Full working view: raw / saved passing tests'),
                            ('native_pair_raw.png','native_pair_overlay.png','Native paired crop: raw / saved passing tests'),
                            ('shared_strip_raw.png','shared_strip_overlay.png','Same image chart: raw grayscale / nominal tracks and saved passes')]:
            body.append(f'<h3>{caption}</h3><div class="row"><img src="{name}/{a}"><img src="{name}/{b}"></div>')
        body.append('<p>Shared strips are interpolated displays of the 960×540 working image, not additional resolution. Checker areas are unavailable. Use the native crop for context; neither colour denotes court identity.</p></section>')
    body.append('<h2>One readout</h2><p>For each pair: identify which visible structure supplies the passing tests; report whether the two named markings use distinct physical ridges, one ridge, or unresolved evidence. For the approved pair, explain any visible weak/crowded/occluded support without inventing a rejection. No threshold fitting or automatic acceptance claim.</p>')
    (output/'atlas.html').write_text('\n'.join(body),encoding='utf-8')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe-csv',type=Path,default=Path(__file__).parent/'inputs/real_interval_probe.csv')
    args=parser.parse_args(); repo=args.repo.resolve(); output=args.output.resolve()
    # The committed handover keeps generated evidence under a distinct child
    # directory. Rejecting only the checkout itself preserves that boundary.
    if output==repo:
        raise ValueError('Choose a distinct output directory')
    output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    manifest={'status':'started','utc':datetime.now(timezone.utc).isoformat(),'pinned_revision':REF,
              'used_inputs':[],'script_sha256':sha256(Path(__file__).read_bytes()),
              'helper_sha256':{name:sha256((Path(__file__).parent/name).read_bytes()) for name in ['analyze_fresh.py','w2_probe.py']},
              'environment':{'python':sys.version,'numpy':np.__version__,'opencv':cv2.__version__,'platform':platform.platform()},
              'scientific_change':'diagnostic export only: two paired pixel witnesses in one batch',
              'new_extractor_or_scores_or_candidates':False,'physical_ownership':'UNRESOLVED'}
    try:
        try:
            manifest['repository_head']=git(repo,'rev-parse','HEAD').decode().strip()
            manifest['tracked_worktree_status']=git(repo,'status','--porcelain=v1','--untracked-files=no').decode()
            manifest['tracked_dirty_patch_sha256']=sha256(git(repo,'diff','--binary','HEAD'))
        except RuntimeError as e:
            manifest['git_metadata_unavailable']=str(e)
        payload=json.loads(pinned_read(repo,WITNESS_PATH,WITNESS_BLOB,manifest['used_inputs']))
        manifest['csv_trace_reproduction']=verify_uploaded_table(payload,args.probe_csv)
        index={(r['case_id'],r['candidate_id']):r for r in payload['diagnostic_panel']}
        metrics=[]
        for target in TARGETS:
            data=pinned_read(repo,target['frame'],target['frame_blob'],manifest['used_inputs'])
            image=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Could not decode native frame')
            if not math.isclose(image.shape[1]/image.shape[0],960/540,rel_tol=0,abs_tol=1e-9):
                raise ValueError('Native aspect ratio differs from working convention')
            record=render_pair(image,index[(target['case_id'],target['candidate_id'])],target,
                               output/target['candidate_id'].replace(':','_'))
            metrics.append({k:record[k] for k in ['case_id','candidate_id','interval_metrics','pixel_replay_checks']})
        make_html(output,TARGETS)
        write_json(output/'pair_summary.json',metrics)
        manifest['status']='complete'
        manifest['note']='Exactly four targeted intervals pixel-replayed; all 48 uploaded table rows checked against pinned trace. No physical rulings.'
        write_json(output/'manifest.json',manifest)
        with zipfile.ZipFile(output/'return_pack.zip','w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(output.rglob('*')):
                if path.is_file() and path.name!='return_pack.zip':
                    z.write(path,path.relative_to(output))
        print('COMPLETE:',output/'return_pack.zip')
    except Exception as e:
        manifest['status']='BLOCKED';manifest['error']=f'{type(e).__name__}: {e}'
        write_json(output/'manifest.json',manifest)
        raise

if __name__=='__main__':
    main()
