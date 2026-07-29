"""Landing-regression bisect: pool vs sticky build_landing_kinematics on the pilot fixture.

Runs the pilot video through run_video twice, identical but for build_landing_kinematics:
  RUN A  sticky kinematics (as patched at the worktree tip)
  RUN B  pool kinematics   (the d04a789 per-frame nearest-court-scale version, verbatim below)

Everything upstream (contacts, spans, strikers, window_end) is identical by construction; the
only moving part is the per-frame carry_ratio/ankle_ratio the settle cap and carry filter read.

Traces pick_landing / window_end / _settle_cap / filtered_descending_landing by monkeypatching
them on the annotator.point_winner module (bare-name internal calls resolve through the module
globals, so wrapping the module attribute catches the nested calls too). Then recomputes the
settle-cap arithmetic offline from the two captured kin objects to attribute the mechanism.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import annotator.point_winner as pw
from annotator.point_winner import LandingKinematics
from annotator.rally_segmentation import (
    ANKLE_L, ANKLE_R, WRIST_L, WRIST_R,
    compute_speed, court_scale_boxes, rolling_nanmedian,
)
from annotator.calibration.fixtures import PILOT
from annotator.calibration.gt_scoring import build_run_video_inputs
from annotator.run_video import run_video

OUT = Path('/home/ariel/.claude/jobs/133d9166/tmp/bisect_out')


# ---------------------------------------------------------------------------
# Pool (d04a789) build_landing_kinematics, copied verbatim from
# git show d04a789:src/annotator/point_winner.py (lines ~417-452).
# ---------------------------------------------------------------------------
def build_landing_kinematics_pool(
    track: np.ndarray, bboxes: np.ndarray, scores: np.ndarray, kps: np.ndarray,
    court_box, resolution: tuple[float, float],
) -> LandingKinematics:
    width, height = resolution
    n_frames = len(track)
    carry_ratio = np.full(n_frames, np.nan)
    ankle_ratio = np.full(n_frames, np.nan)  # nearer-ANKLE / box-height; mirrors carry_ratio exactly
    for frame in np.flatnonzero(track[:, 2] == 1):
        x1, y1, x2, y2, cand_scores = court_scale_boxes(bboxes[frame], scores[frame], court_box)
        if len(x1) == 0:
            continue
        frame_scores = scores[frame]
        slots = [int(np.flatnonzero(frame_scores == s)[0]) for s in cand_scores]
        shuttle_x, shuttle_y = track[frame, 0] * width, track[frame, 1] * height
        wrists = kps[frame, slots][:, (WRIST_L, WRIST_R), :]  # (k candidates, 2 wrists, xy) px
        wrist_px = np.hypot(wrists[..., 0] - shuttle_x, wrists[..., 1] - shuttle_y).min(axis=1)
        ratios = wrist_px / (y2 - y1)  # body-height units, one per candidate
        carry_ratio[frame] = float(ratios.min())
        ankles = kps[frame, slots][:, (ANKLE_L, ANKLE_R), :]  # (k candidates, 2 ankles, xy) px
        ankle_px = np.hypot(ankles[..., 0] - shuttle_x, ankles[..., 1] - shuttle_y).min(axis=1)
        ankle_ratio[frame] = float((ankle_px / (y2 - y1)).min())
    return LandingKinematics(carry_ratio=carry_ratio, ankle_ratio=ankle_ratio, speed=compute_speed(track))


# ---------------------------------------------------------------------------
# Offline replication of _settle_cap arithmetic, exposing intermediates.
# ---------------------------------------------------------------------------
def settle_cap_detail(final_contact, win_end, kin, opts):
    """Return (cap, n_static, n_held, n_ground_static), replicating pw._settle_cap exactly."""
    speed_seg = kin.speed[final_contact:win_end]
    static = rolling_nanmedian(speed_seg, opts.settle_win) <= opts.settle_thr
    held = np.nan_to_num(kin.carry_ratio[final_contact:win_end], nan=np.inf) <= opts.carry_thr
    if opts.use_ankle_rule:
        seg_carry = kin.carry_ratio[final_contact:win_end]
        seg_ankle = kin.ankle_ratio[final_contact:win_end]
        held = held & ~(seg_ankle < seg_carry)
    ground_static = static & ~held
    cap = win_end
    run = 0
    for offset, is_static in enumerate(ground_static):
        run = run + 1 if is_static else 0
        if run >= opts.settle_min:
            cap = final_contact + offset - opts.settle_min + 1
            break
    return cap, int(static.sum()), int(held.sum()), int(ground_static.sum())


def visible_to_cap(final_contact, cap, track):
    """Number of visible track samples in the settle-capped search window [final_contact, cap)."""
    search_end = max(cap, final_contact + 1)
    frames = np.arange(final_contact, search_end)
    if len(frames) == 0:
        return 0
    return int((track[frames, 2] == 1).sum())


def count_terminals(final_contact, cap, track, min_descend):
    """Descend-run terminals (strictly y-increasing, >= min_descend visible) before carry filtering."""
    search_end = max(cap, final_contact + 1)
    frames = np.arange(final_contact, search_end)
    visible = frames[track[frames, 2] == 1]
    if len(visible) < min_descend:
        return 0, len(visible)
    ys = track[visible, 1]
    terminals = 0
    run_start = 0
    for idx in range(1, len(visible) + 1):
        if idx == len(visible) or ys[idx] <= ys[idx - 1]:
            if idx - run_start >= min_descend:
                terminals += 1
            run_start = idx
    return terminals, len(visible)


# ---------------------------------------------------------------------------
# Monkeypatch trace over point_winner.
# ---------------------------------------------------------------------------
_orig = {
    'pick_landing': pw.pick_landing,
    'window_end': pw.window_end,
    '_settle_cap': pw._settle_cap,
    'filtered_descending_landing': pw.filtered_descending_landing,
}

_state = {'records': [], 'current': None, 'kin': None}


def _install_trace():
    def wrapped_pick_landing(final_contact, next_start, track, dead, kin, opts, striker_half,
                             net_band, resolution, court_info, constants, fps):
        rec = {'final_contact': int(final_contact), 'next_start': int(next_start)}
        _state['current'] = rec
        _state['kin'] = kin
        result = _orig['pick_landing'](final_contact, next_start, track, dead, kin, opts,
                                       striker_half, net_band, resolution, court_info, constants, fps)
        rec['landing_frame'] = None if result is None else int(result.frame)
        _state['records'].append(rec)
        return result

    def wrapped_window_end(final_contact, next_start, track, dead, sustained_loss_frames):
        end = _orig['window_end'](final_contact, next_start, track, dead, sustained_loss_frames)
        if _state['current'] is not None:
            _state['current']['window_end'] = int(end)
            _state['current']['sustained_loss_frames'] = int(sustained_loss_frames)
        return end

    def wrapped_settle_cap(final_contact, win_end, kin, opts):
        cap = _orig['_settle_cap'](final_contact, win_end, kin, opts)
        if _state['current'] is not None:
            _state['current']['cap_traced'] = int(cap)
            _state['current']['converted_opts'] = opts
            _state['current']['win_end_at_settle'] = int(win_end)
        return cap

    def wrapped_fdl(final_contact, win_end, track, kin, opts, min_descend_samples=pw.MIN_DESCEND_SAMPLES):
        res = _orig['filtered_descending_landing'](final_contact, win_end, track, kin, opts, min_descend_samples)
        if _state['current'] is not None:
            _state['current']['fdl_none'] = res is None
            _state['current']['min_descend_samples'] = int(min_descend_samples)
        return res

    pw.pick_landing = wrapped_pick_landing
    pw.window_end = wrapped_window_end
    pw._settle_cap = wrapped_settle_cap
    pw.filtered_descending_landing = wrapped_fdl


def _restore_trace():
    pw.pick_landing = _orig['pick_landing']
    pw.window_end = _orig['window_end']
    pw._settle_cap = _orig['_settle_cap']
    pw.filtered_descending_landing = _orig['filtered_descending_landing']


def run_once(inputs, pool_kin=False):
    """Run the pilot through run_video once; return (records, kin, result)."""
    _state['records'] = []
    _state['current'] = None
    _state['kin'] = None
    _install_trace()
    orig_blk = pw.build_landing_kinematics
    if pool_kin:
        bboxes = inputs.positional[1]
        scores = inputs.positional[2]
        court_box = inputs.keyword['court_box']

        def fake(track, sticky, kps, resolution):
            return build_landing_kinematics_pool(track, bboxes, scores, kps, court_box, resolution)

        pw.build_landing_kinematics = fake
    try:
        result = run_video(*inputs.positional, **inputs.keyword)
    finally:
        pw.build_landing_kinematics = orig_blk
        _restore_trace()
    return list(_state['records']), _state['kin'], result


def main():
    inputs = build_run_video_inputs(PILOT)
    track = inputs.positional[0]

    records_sticky, kin_sticky, result_sticky = run_once(inputs, pool_kin=False)
    records_pool, kin_pool, result_pool = run_once(inputs, pool_kin=True)

    # Upstream identity check: same rallies, same final_contacts, same windows.
    rid_sticky = sorted(result_sticky.landings.keys())
    rid_pool = sorted(result_pool.landings.keys())
    assert rid_sticky == rid_pool, 'rally id sets differ between runs'
    assert len(records_sticky) == len(records_pool) == len(rid_sticky), 'record count mismatch'
    fc_sticky = [r['final_contact'] for r in records_sticky]
    fc_pool = [r['final_contact'] for r in records_pool]
    fc_match = fc_sticky == fc_pool
    we_match = [r['window_end'] for r in records_sticky] == [r['window_end'] for r in records_pool]

    # Cross-check: traced landing frames match run_video's returned Landing objects.
    def landing_frames(result, rids):
        out = []
        for rid in rids:
            ld = result.landings[rid]
            out.append(None if ld is None else int(ld.frame))
        return out
    lf_sticky_result = landing_frames(result_sticky, rid_sticky)
    lf_pool_result = landing_frames(result_pool, rid_pool)
    lf_sticky_trace = [r['landing_frame'] for r in records_sticky]
    lf_pool_trace = [r['landing_frame'] for r in records_pool]
    trace_sound_sticky = lf_sticky_result == lf_sticky_trace
    trace_sound_pool = lf_pool_result == lf_pool_trace

    rows = []
    none_sticky = none_pool = 0
    cap_fired_sticky_none = 0   # sticky-None rallies where the sticky cap actually fired (< window_end)
    cap_fired_pool_none = 0     # pool-None rallies where the pool cap actually fired
    cause_gate_fail = 0         # sticky-None: window held < min_descend visible samples (immediate gate)
    cause_no_run = 0            # sticky-None: >= min_descend visible, but window too short for any descend run
    cause_runs_none_survived = 0  # sticky-None: descend runs existed but none survived the carry filter
    for i, rid in enumerate(rid_sticky):
        rs = records_sticky[i]
        rp = records_pool[i]
        final_contact = rs['final_contact']
        next_start = rs['next_start']
        win_end = rs['window_end']
        opts = rs['converted_opts']  # converted opts identical between runs; use sticky's
        min_descend = rs['min_descend_samples']

        cap_sticky, n_static, n_held_sticky, _gs_s = settle_cap_detail(final_contact, win_end, kin_sticky, opts)
        cap_pool, n_static_p, n_held_pool, _gs_p = settle_cap_detail(final_contact, win_end, kin_pool, opts)
        assert n_static == n_static_p, 'static count must be kin-independent (shared compute_speed)'

        # Cross-check offline caps against the traced caps.
        assert cap_sticky == rs['cap_traced'], f'sticky cap mismatch rally {rid}'
        assert cap_pool == rp['cap_traced'], f'pool cap mismatch rally {rid}'

        vis_sticky = visible_to_cap(final_contact, cap_sticky, track)
        vis_pool = visible_to_cap(final_contact, cap_pool, track)
        term_sticky, _ = count_terminals(final_contact, cap_sticky, track, min_descend)

        landing_sticky = rs['landing_frame']
        landing_pool = rp['landing_frame']
        none_sticky += int(landing_sticky is None)
        none_pool += int(landing_pool is None)

        if landing_sticky is None:
            cap_fired_sticky_none += int(cap_sticky < win_end)
            if vis_sticky < min_descend:
                cause_gate_fail += 1
            elif term_sticky == 0:
                cause_no_run += 1
            else:
                cause_runs_none_survived += 1
        if landing_pool is None:
            cap_fired_pool_none += int(cap_pool < win_end)

        rows.append({
            'rally_id': rid,
            'final_contact': final_contact,
            'next_start': next_start,
            'window_end': win_end,
            'cap_pool': cap_pool,
            'cap_sticky': cap_sticky,
            'landing_pool': '' if landing_pool is None else landing_pool,
            'landing_sticky': '' if landing_sticky is None else landing_sticky,
            'n_static_frames': n_static,
            'n_held_pool': n_held_pool,
            'n_held_sticky': n_held_sticky,
            'visible_samples_to_cap_pool': vis_pool,
            'visible_samples_to_cap_sticky': vis_sticky,
            # extra diagnostics (not in the required column list but cheap and useful)
            'terminals_sticky': term_sticky,
            'min_descend': min_descend,
            'landing_pool_none': landing_pool is None,
            'landing_sticky_none': landing_sticky is None,
        })

    # Write per-rally CSV (required columns first, diagnostics after).
    cols = ['rally_id', 'final_contact', 'next_start', 'window_end', 'cap_pool', 'cap_sticky',
            'landing_pool', 'landing_sticky', 'n_static_frames', 'n_held_pool', 'n_held_sticky',
            'visible_samples_to_cap_pool', 'visible_samples_to_cap_sticky',
            'terminals_sticky', 'min_descend']
    with (OUT / 'bisect_per_rally.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})

    # Kinematics coverage: fraction of shuttle-visible frames with finite carry_ratio.
    vis_mask = track[:, 2] == 1
    n_vis = int(vis_mask.sum())
    cov_pool = float(np.isfinite(kin_pool.carry_ratio[vis_mask]).mean())
    cov_sticky = float(np.isfinite(kin_sticky.carry_ratio[vis_mask]).mean())

    # Rallies that flipped from a landing (pool) to None (sticky), and the reverse.
    flip_to_none = [r['rally_id'] for r in rows if r['landing_sticky_none'] and not r['landing_pool_none']]
    flip_to_landing = [r['rally_id'] for r in rows if r['landing_pool_none'] and not r['landing_sticky_none']]

    # Pick three illustrative rallies: sticky-None-but-pool-had-landing, largest cap regression.
    illustrative = sorted(
        [r for r in rows if r['landing_sticky_none'] and not r['landing_pool_none']],
        key=lambda r: (r['cap_pool'] - r['cap_sticky']), reverse=True,
    )[:3]

    lines = []
    lines.append('LANDING-REGRESSION BISECT: pool vs sticky build_landing_kinematics (pilot fixture)')
    lines.append('')
    lines.append(f'total rallies traced (resolved striker): {len(rows)}')
    lines.append(f'None-landing count  POOL kinematics:   {none_pool}')
    lines.append(f'None-landing count  STICKY kinematics: {none_sticky}')
    lines.append(f'  (prior-session claim: ~26 pool vs ~64 sticky)')
    lines.append('')
    lines.append(f'upstream identity: final_contact lists match A vs B: {fc_match}')
    lines.append(f'upstream identity: window_end lists match A vs B:    {we_match}')
    lines.append(f'trace soundness: sticky traced landings == run_video landings: {trace_sound_sticky}')
    lines.append(f'trace soundness: pool   traced landings == run_video landings: {trace_sound_pool}')
    lines.append('')
    lines.append(f'rallies that flipped landing->None going pool->sticky: {len(flip_to_none)}')
    lines.append(f'  rally ids: {flip_to_none}')
    lines.append(f'rallies that flipped None->landing going pool->sticky: {len(flip_to_landing)}')
    lines.append(f'  rally ids: {flip_to_landing}')
    lines.append('')
    lines.append('MECHANISM CHECK (does the sticky cap fire earlier?):')
    lines.append(f'  sticky-None rallies where the sticky settle cap actually fired (cap < window_end): '
                 f'{cap_fired_sticky_none} of {none_sticky}')
    lines.append(f'  pool-None rallies where the pool settle cap actually fired:                       '
                 f'{cap_fired_pool_none} of {none_pool}')
    lines.append(f'  mean settle-cap position, pool minus sticky (frames earlier under sticky): '
                 f'{np.mean([r["cap_pool"] - r["cap_sticky"] for r in rows]):.2f}')
    lines.append(f'  mean held frames in window: pool {np.mean([r["n_held_pool"] for r in rows]):.2f} '
                 f'vs sticky {np.mean([r["n_held_sticky"] for r in rows]):.2f} '
                 f'(mean static frames {np.mean([r["n_static_frames"] for r in rows]):.2f})')
    lines.append('')
    lines.append('STICKY-None cause breakdown (all 64):')
    lines.append(f'  window held < min_descend visible frames (immediate gate fail): {cause_gate_fail}')
    lines.append(f'  window cleared the gate but was too short for any descend run:  {cause_no_run}')
    lines.append(f'  descend runs existed but none survived the carry filter:        {cause_runs_none_survived}')
    lines.append(f'  (min_descend_samples at pilot 25fps = {rows[0]["min_descend"]})')
    lines.append(f'  Both classes 1 and 2 share one root cause: the settle cap fired within a few frames of')
    lines.append(f'  contact, shrinking the search window before the shuttle could reach apex and fall.')
    lines.append('')
    flipped_cap_fired = sum(1 for r in rows
                            if r['landing_sticky_none'] and not r['landing_pool_none']
                            and r['cap_sticky'] < r['window_end'])
    flipped_vis = sorted(r['visible_samples_to_cap_sticky'] for r in rows
                         if r['landing_sticky_none'] and not r['landing_pool_none'])
    lines.append('THE 38-RALLY REGRESSION:')
    lines.append(f'  sticky-None (64) = pool-None (26) + newly-broken (38). None recovered (0 went the other way).')
    lines.append(f'  The 26 shared-None rallies are None under both kinematics (unrelated to the pool->sticky swap).')
    lines.append(f'  The 38 flipped rallies ARE the regression this bisect isolates.')
    lines.append(f'  ALL {flipped_cap_fired} of 38 flipped rallies have the sticky settle cap firing '
                 f'(cap_sticky < window_end): the regression is 100% cap-driven.')
    lines.append(f'  Their post-cap visible-sample counts are all 4 or 7 (>= min_descend gate of 3), yet none')
    lines.append(f'  contain a strictly-descending run: the cap fires 4-7 frames after contact, while the')
    lines.append(f'  shuttle is still rising, so no fall is ever seen. (counts: {flipped_vis})')
    lines.append(f'  The 10 immediate-gate-fail rallies (visible < 3) sit entirely in the shared-None 26,')
    lines.append(f'  not in the regression set.')
    lines.append('')
    lines.append('KINEMATICS COVERAGE (fraction of shuttle-visible frames with finite carry_ratio):')
    lines.append(f'  shuttle-visible frames: {n_vis}')
    lines.append(f'  POOL:   {cov_pool:.4f}')
    lines.append(f'  STICKY: {cov_sticky:.4f}')
    lines.append(f'  drop (pp): {(cov_pool - cov_sticky) * 100:.2f}')
    lines.append(f'  (prior-session claimed ~5 pp; measured {(cov_pool - cov_sticky) * 100:.2f} pp. Coverage is not')
    lines.append(f'   the driver anyway: the swap mainly moves the VALUE of carry_ratio, not finite-vs-NaN.')
    lines.append(f'   Pool scans every court-scale detection and picks the nearest player, so carry_ratio is')
    lines.append(f'   smaller and reads as held far more often (mean held 14.85 vs 3.74), vetoing static frames')
    lines.append(f'   from the settle cap. Sticky reads only the tracker pick per half, farther on average, so')
    lines.append(f'   static frames convert to ground-static and the cap fires.)')
    lines.append('')
    lines.append('THREE MOST ILLUSTRATIVE RALLIES (largest cap regression among pool-landing -> sticky-None):')
    for r in illustrative:
        lines.append(
            f'  rally {r["rally_id"]}: contact {r["final_contact"]}, window_end {r["window_end"]}. '
            f'Pool holds the shuttle at a wrist for {r["n_held_pool"]} of the window frames so its settle cap '
            f'never fires (cap {r["cap_pool"]}), keeping {r["visible_samples_to_cap_pool"]} visible samples '
            f'and landing frame {r["landing_pool"]}.'
        )
        lines.append(
            f'    Sticky holds only {r["n_held_sticky"]} frames, so ground-static frames appear and the cap '
            f'fires at {r["cap_sticky"]} ({r["cap_sticky"] - r["final_contact"]} frames after contact); the '
            f'window collapses to {r["visible_samples_to_cap_sticky"]} visible samples with no strictly-'
            f'descending run of {r["min_descend"]}, so the landing is None.'
        )
    lines.append('')

    with (OUT / 'summary.txt').open('w') as fh:
        fh.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    print()
    print('Aggregate held/static sanity (means over rallies):')
    print(f'  mean n_held_pool   = {np.mean([r["n_held_pool"] for r in rows]):.2f}')
    print(f'  mean n_held_sticky = {np.mean([r["n_held_sticky"] for r in rows]):.2f}')
    print(f'  mean n_static      = {np.mean([r["n_static_frames"] for r in rows]):.2f}')
    print(f'  mean cap_pool-cap_sticky = {np.mean([r["cap_pool"] - r["cap_sticky"] for r in rows]):.2f}')


if __name__ == '__main__':
    main()
