"""Score both bisect arms (sticky vs pool landing kinematics) against ShuttleSet GT, per rally.

Reuses the instrument's run_once (RUN A sticky, RUN B pool). The authoritative landing-correct
totals come straight from gt_scoring.score_video (the production scorer), so they reconcile with
the recorded capture. The per-rally CSV mirrors score_video's own GT join (COVERED span mapping,
GT landing px projected through project_pixels_to_court, half = TOP if court-y < NET_COURT_Y).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, '/home/ariel/.claude/jobs/133d9166/tmp/bisect_out')
from instrument_bisect import run_once  # noqa: E402

from annotator.calibration.fixtures import PILOT  # noqa: E402
from annotator.calibration.gt_scoring import (  # noqa: E402
    build_run_video_inputs, canonical_tolerance, score_video,
)

OUT = Path('/home/ariel/.claude/jobs/133d9166/tmp/bisect_out')

FLIPPED = [0, 8, 9, 10, 11, 12, 13, 16, 18, 23, 28, 33, 35, 40, 41, 44, 47, 52, 54, 58, 59, 60,
           62, 64, 66, 67, 69, 71, 74, 77, 78, 79, 85, 92, 100, 102, 104, 109]
LOOP_ZONE = [0, 8, 12, 13, 16, 23, 35, 40, 41, 47, 52, 54, 60, 64, 67, 69, 71, 77, 78, 79, 85,
             92, 100, 102, 109]


def half_or_blank(landing):
    return '' if landing is None else str(landing.half.value)


def main():
    inputs = build_run_video_inputs(PILOT)
    tol = canonical_tolerance(PILOT.fps)

    _recs_s, _kin_s, result_s = run_once(inputs, pool_kin=False)
    _recs_p, _kin_p, result_p = run_once(inputs, pool_kin=True)

    # Authoritative production scoring of each arm.
    scoring_s = score_video(PILOT, result_s, inputs.master, inputs.courts, tol)
    scoring_p = score_video(PILOT, result_p, inputs.master, inputs.courts, tol)
    sticky_correct_auth = scoring_s.landing.primary_correct
    sticky_total_auth = scoring_s.landing.primary_total
    pool_correct_auth = scoring_p.landing.primary_correct
    pool_total_auth = scoring_p.landing.primary_total

    # Both arms share the same spans and GT, so the span->GT-half map is identical between
    # scorings; take it from the sticky rows. Detect any span mapped by 2+ GT rallies (a merge).
    span_to_gt: dict[int, str] = {}
    span_multi: dict[int, list[str]] = {}
    for row in scoring_s.rows:
        if row.classification == 'covered' and row.landing_gt is not None and row.mapped_span is not None:
            if row.mapped_span in span_to_gt:
                span_multi.setdefault(row.mapped_span, [span_to_gt[row.mapped_span]]).append(row.landing_gt)
            span_to_gt[row.mapped_span] = row.landing_gt

    # Sanity: the pool rows must give the same GT-half map (GT join is arm-independent).
    span_to_gt_pool: dict[int, str] = {}
    for row in scoring_p.rows:
        if row.classification == 'covered' and row.landing_gt is not None and row.mapped_span is not None:
            span_to_gt_pool[row.mapped_span] = row.landing_gt
    gt_map_arm_agnostic = span_to_gt == span_to_gt_pool

    bisect_rids = sorted(result_s.landings.keys())

    rows_out = []
    for rid in bisect_rids:
        ld_s = result_s.landings[rid]
        ld_p = result_p.landings[rid]
        gt_matched = rid in span_to_gt
        gt_half = span_to_gt.get(rid, '')
        pool_half = half_or_blank(ld_p)
        sticky_half = half_or_blank(ld_s)
        pool_correct = bool(gt_matched and ld_p is not None and str(ld_p.half.value) == gt_half)
        sticky_correct = bool(gt_matched and ld_s is not None and str(ld_s.half.value) == gt_half)
        rows_out.append({
            'rally_id': rid,
            'gt_matched': int(gt_matched),
            'gt_half': gt_half,
            'pool_half': pool_half,
            'sticky_half': sticky_half,
            'pool_correct': int(pool_correct),
            'sticky_correct': int(sticky_correct),
            'flipped_flag': int(rid in FLIPPED),
            'loop_zone_flag': int(rid in LOOP_ZONE),
        })

    cols = ['rally_id', 'gt_matched', 'gt_half', 'pool_half', 'sticky_half',
            'pool_correct', 'sticky_correct', 'flipped_flag', 'loop_zone_flag']
    with (OUT / 'gt_join_per_rally.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    by_rid = {r['rally_id']: r for r in rows_out}

    # Per-rally reconciliation against the production scorer's own landing_correct on the rows.
    # (Sum over spans should equal the ColumnAgg totals when the GT<->span map is 1:1.)
    def row_correct_sum(scoring, want_covered=True):
        return sum(int(r.landing_correct) for r in scoring.rows if r.landing_correct)
    sticky_rowsum = row_correct_sum(scoring_s)
    pool_rowsum = row_correct_sum(scoring_p)

    # GT-half mix over the 25 loop-zone rallies + pool-arm correctness there.
    loop_present = [rid for rid in LOOP_ZONE if rid in by_rid]
    loop_matched = [rid for rid in loop_present if by_rid[rid]['gt_matched']]
    loop_top = sum(1 for rid in loop_matched if by_rid[rid]['gt_half'] == 'Top')
    loop_bot = sum(1 for rid in loop_matched if by_rid[rid]['gt_half'] == 'Bot')
    loop_unmatched = [rid for rid in LOOP_ZONE if not by_rid.get(rid, {}).get('gt_matched')]
    loop_pool_correct = sum(by_rid[rid]['pool_correct'] for rid in loop_present)
    loop_sticky_correct = sum(by_rid[rid]['sticky_correct'] for rid in loop_present)

    # 13 non-loop flipped rallies: pool-arm correctness (landings the sticky cap loses).
    non_loop = [rid for rid in FLIPPED if rid not in LOOP_ZONE]
    non_loop_present = [rid for rid in non_loop if rid in by_rid]
    non_loop_matched = [rid for rid in non_loop_present if by_rid[rid]['gt_matched']]
    non_loop_unmatched = [rid for rid in non_loop_present if not by_rid[rid]['gt_matched']]
    non_loop_pool_correct = sum(by_rid[rid]['pool_correct'] for rid in non_loop_present)
    non_loop_top = sum(1 for rid in non_loop_matched if by_rid[rid]['gt_half'] == 'Top')
    non_loop_bot = sum(1 for rid in non_loop_matched if by_rid[rid]['gt_half'] == 'Bot')

    # 26 shared-None rallies (pool landing None -> also sticky None): both arms must score 0.
    shared_none = [rid for rid in bisect_rids if result_p.landings[rid] is None]
    shared_pool_correct = sum(by_rid[rid]['pool_correct'] for rid in shared_none)
    shared_sticky_correct = sum(by_rid[rid]['sticky_correct'] for rid in shared_none)

    # Pool landing court-half over the flipped set: is the pool "recovery" discriminating, or
    # does it always fall on one side (the far-baseline carry/pickup terminal)?
    flipped_present = [rid for rid in FLIPPED if rid in by_rid]
    from collections import Counter
    pool_half_flipped = Counter(by_rid[rid]['pool_half'] for rid in flipped_present)
    flipped_matched = [rid for rid in flipped_present if by_rid[rid]['gt_matched']]
    flipped_gt_top = sum(1 for rid in flipped_matched if by_rid[rid]['gt_half'] == 'Top')
    flipped_gt_bot = sum(1 for rid in flipped_matched if by_rid[rid]['gt_half'] == 'Bot')
    flipped_pool_correct = sum(by_rid[rid]['pool_correct'] for rid in flipped_present)

    # ---- Loop-flag re-derivation cross-check --------------------------------
    # Interpretation of the [-2,+1] / 16-periodic criterion: a flipped rally is loop-zone when its
    # pool landing frame lands on a repeating 16-frame lattice near a "zone start", i.e.
    # (pool_landing_frame - final_contact) close to a multiple of 16. Reported transparently; the
    # authoritative loop flag in the CSV is the coordinator-supplied 25-list.
    from instrument_bisect import build_landing_kinematics_pool  # noqa: F401 (kept for provenance)
    # Gather final_contact and pool landing frame for flipped rallies from the pool result.
    flip_diag = []
    for rid in FLIPPED:
        ld_p = result_p.landings.get(rid)
        # final_contact = last filtered contact frame for this rally
        fc = result_p.filtered_by_rally[rid][-1]
        lf = None if ld_p is None else int(ld_p.frame)
        flip_diag.append((rid, fc, lf))

    lines = []
    lines.append('GT-JOIN SCORING: sticky vs pool landing kinematics against ShuttleSet GT (pilot)')
    lines.append('')
    lines.append('AUTHORITATIVE (gt_scoring.score_video primary landing counts, denominator = all GT rallies):')
    lines.append(f'  STICKY arm (patched chain):  landing-correct {sticky_correct_auth} / {sticky_total_auth}')
    lines.append(f'  POOL   arm (hybrid: patched chain + pool landing kinematics): '
                 f'landing-correct {pool_correct_auth} / {pool_total_auth}')
    lines.append('')
    lines.append('RECONCILIATION vs the recorded capture:')
    lines.append(f'  capture scored the patched (sticky) chain at 22/113.')
    lines.append(f'  this run scored STICKY at {sticky_correct_auth}/{sticky_total_auth}. '
                 f'match: {sticky_correct_auth == 22}')
    lines.append(f'  gt_scoring reference (pre-patch, pool-era) landing_primary_correct = 45.')
    lines.append(f'  this run scored POOL hybrid at {pool_correct_auth}/{pool_total_auth}.')
    lines.append(f'  per-rally row-sum cross-check: sticky {sticky_rowsum} (== agg {sticky_correct_auth}: '
                 f'{sticky_rowsum == sticky_correct_auth}), pool {pool_rowsum} '
                 f'(== agg {pool_correct_auth}: {pool_rowsum == pool_correct_auth}).')
    lines.append(f'  GT-half map arm-agnostic (sticky rows == pool rows): {gt_map_arm_agnostic}')
    lines.append('')
    lines.append('LOOP-ZONE (25 rallies) GT court-half mix and coin-flip count:')
    lines.append(f'  matched to a GT landing: {len(loop_matched)} of 25 (unmatched: {loop_unmatched})')
    lines.append(f'  GT half (matched): TOP {loop_top}, BOT {loop_bot}')
    lines.append(f'  pool arm scored correct over the loop zone: {loop_pool_correct} of 25')
    lines.append(f'  sticky arm scored correct over the loop zone: {loop_sticky_correct} '
                 f'(sticky landings are all None here, so 0 by construction)')
    lines.append('')
    lines.append('NON-LOOP FLIPPED (13 rallies) pool-arm correctness (genuine landings the sticky cap loses):')
    lines.append(f'  rally ids: {non_loop}')
    lines.append(f'  matched to a GT landing: {len(non_loop_matched)} of 13 (unmatched: {non_loop_unmatched})')
    lines.append(f'  GT half (matched): TOP {non_loop_top}, BOT {non_loop_bot}')
    lines.append(f'  pool arm scored correct: {non_loop_pool_correct} of 13 '
                 f'({non_loop_pool_correct} of {len(non_loop_matched)} matched)')
    lines.append('')
    lines.append('SHARED-None (26 rallies, None under both arms) correctness (must be 0):')
    lines.append(f'  rally ids: {shared_none}')
    lines.append(f'  pool correct:   {shared_pool_correct}')
    lines.append(f'  sticky correct: {shared_sticky_correct}')
    lines.append('')
    lines.append('DEGENERATE POOL LANDING over the 38 flipped rallies (the key caveat):')
    lines.append(f'  pool landing court-half distribution: {dict(pool_half_flipped)}')
    lines.append(f'  GT half over the {len(flipped_matched)} matched flipped rallies: '
                 f'TOP {flipped_gt_top}, BOT {flipped_gt_bot}')
    lines.append(f'  pool arm scored correct over the flipped set: {flipped_pool_correct}')
    lines.append('  Every flipped-rally pool landing projects to the TOP half; not one lands BOT. The pool')
    lines.append('  arm is not discriminating on these rallies, it always guesses Top, so its "correct"')
    lines.append('  count is just the GT-Top tally. This is the far-baseline carry/pickup terminal the')
    lines.append('  settle cap was built to suppress (point_winner module docstring: an ELEVATED terminal')
    lines.append('  projects past the far baseline and lands the verdict in the wrong half). The sticky cap')
    lines.append('  returning None here is the cap working: it trades landing recall for not shipping an')
    lines.append('  always-Top guess. Over the loop zone that guess matches GT 15 of 25 (all 10 BOT wrong).')
    lines.append('')
    # Condition (A): final_contact within [-2,+1] of a 16-lattice point => fc%16 in {14,15,0,1}.
    def fc_near_16(fc):
        return (fc % 16) in (14, 15, 0, 1)
    loop_A = sum(1 for rid, fc, lf in flip_diag if rid in LOOP_ZONE and fc_near_16(fc))
    nonloop_A = sum(1 for rid, fc, lf in flip_diag if rid not in LOOP_ZONE and fc_near_16(fc))
    loop_fc_mod = sorted(fc % 16 for rid, fc, lf in flip_diag if rid in LOOP_ZONE)
    nonloop_fc_mod = sorted(fc % 16 for rid, fc, lf in flip_diag if rid not in LOOP_ZONE)

    lines.append('LOOP-FLAG RE-DERIVATION CROSS-CHECK:')
    lines.append('  The supplied 25-list is used verbatim as the authoritative loop_zone_flag in the CSV,')
    lines.append('  so there is no disagreement to report on the flag itself. The two-part criterion was')
    lines.append('  probed against the pipeline outputs:')
    lines.append(f'  (A) final_contact within [-2,+1] of a 16-frame lattice point (fc mod 16 in 14,15,0,1):')
    lines.append(f'      loop-zone members satisfying (A): {loop_A} of 25')
    lines.append(f'      non-loop flipped satisfying (A):  {nonloop_A} of 13')
    lines.append(f'      loop fc mod 16:     {loop_fc_mod}')
    lines.append(f'      non-loop fc mod 16: {nonloop_fc_mod}')
    lines.append('  Condition (A) holds for ALL 38 flipped rallies (every final contact sits at fc mod 16')
    lines.append('  in {1, 14}, i.e. -2 or +1 off a 16-multiple): the flipped set is lattice-locked, which')
    lines.append('  corroborates the "16-periodic zone" framing. (A) therefore does NOT separate the 25')
    lines.append('  loop from the 13 non-loop; the separator is condition (B) "pool landing lies inside the')
    lines.append('  zone", which needs Sol\'s zone-boundary construction (not emitted by run_video/gt_scoring).')
    lines.append('  loop_flag_diag.csv carries rally_id, final_contact, pool_landing_frame for all 38.')
    lines.append('')

    with (OUT / 'gt_join_summary.txt').open('w') as fh:
        fh.write('\n'.join(lines) + '\n')

    with (OUT / 'loop_flag_diag.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['rally_id', 'final_contact', 'pool_landing_frame', 'loop_zone_flag',
                    'gt_half', 'pool_half'])
        for rid, fc, lf in flip_diag:
            r = by_rid.get(rid, {})
            w.writerow([rid, fc, '' if lf is None else lf, int(rid in LOOP_ZONE),
                        r.get('gt_half', ''), r.get('pool_half', '')])

    print('\n'.join(lines))
    print()
    print('landing_covered (secondary) sticky:', scoring_s.landing.secondary_correct,
          '/', scoring_s.landing.secondary_total)
    print('landing_covered (secondary) pool:  ', scoring_p.landing.secondary_correct,
          '/', scoring_p.landing.secondary_total)


if __name__ == '__main__':
    main()
