"""Brief H: the honest end-to-end number.

Runs the whole GT-free detection chain (crowned rally segmentation + composition mask, wrist-
filtered contact detection, wrist-body-unit attribution, alternation-rhythm fit, next-server
verdict, kinematic landing filter, hit_height) on pilot (vid 1) and vid15 with NO ground truth
anywhere upstream of scoring, then scores every column current_dimension_extraction_infra.md
tracks against ShuttleSet's per-rally / per-stroke labels: rally boundaries, ball_round (stroke
count), hit-frame timing, player (striker), server, hit_height, landing side, getpoint_player
(winner). This is the number that decides whether hit detection needs new design work before
per-shot rows are viable (hit-list tier 2).

Two-tree setup, copied verbatim from d5_stage10_pin.py's `_bind_worktree_package` mechanism (see
that module's docstring for the full why): `scraper` and `scripts` are bound to the WORKTREE
before `d5_winner_retest` is imported, so every `scraper.*` / `scripts.*` import anywhere in the
process resolves to the worktree regardless of this repo checkout's current branch. Only the input
fixtures come from `d5_winner_retest` (VideoCfg PILOT/VID15, load_gt_rallies, reconcile,
load_all_court_info, SHOTS_MASTER, HOMOGRAPHY_CSV); its own GT-anchored per-rally detector
(`run_rally`/`run_video`) is never called — this script's detection chain is built fresh from the
worktree's promoted `scraper.point_winner` module, GT-free until the scoring section.

Deterministic: fixed rally/span/contact ordering throughout, floats formatted to 6dp, no
timestamps or randomness; the CSVs are byte-stable on re-run and their md5s print at the end.

Run with ~/.venvs/badminton-cicd/bin/python from anywhere:
    ~/.venvs/badminton-cicd/bin/python \
        local_scratch/autograder_architecture/h_end_to_end.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = Path(
    '/tmp/claude-1000/-home-ariel-Documents-COSC594-badminton-stroke-classification/'
    '3c0a259c-d52c-4627-be84-028e5b2aa0d6/scratchpad/wt_s24'
)

RESOLUTION = (1920.0, 1080.0)  # working-res the shuttle track / pose arrays share (both videos)
REF_ERR_PX = 3.5  # the reference corner-error band (refpx), the D5/G ruled value
TIMING_TOLERANCE = 30  # frames, greedy-match tolerance for the hit-frame timing column

# Crowned per-video masks (the re-crown picks per video, not the uniform config default). Also
# exactly what d5_winner_retest.PILOT/VID15.mask_path already point at; restated here per the
# brief so the choice is visible without cross-referencing that module.
CROWNED_MASK = {
    'pilot': 'comp_content27_v0p5.npy',
    'vid15': 'comp_content27_v0p7.npy',
}


# ---------------------------------------------------------------------------
# Two-tree import setup (verbatim mechanism from d5_stage10_pin.py)
# ---------------------------------------------------------------------------
def _bind_worktree_package(name: str, package_dir: Path) -> ModuleType:
    """Register `name` (e.g. 'scraper', 'scripts') as a regular package backed by `package_dir`.

    Writes straight into `sys.modules`, so any later `import name` or `from name.sub import ...`
    anywhere in the process (including inside d5_winner_retest.py, which we do not control)
    reuses this exact module object instead of re-resolving `name` off sys.path.
    """
    spec = importlib.util.spec_from_file_location(
        name, package_dir / '__init__.py', submodule_search_locations=[str(package_dir)])
    if spec is None or spec.loader is None:
        raise ImportError(f'could not build a spec for {package_dir}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_worktree_imports(worktree_root: Path) -> None:
    """Bind `scraper` and `scripts` to the worktree, then patch the one pre-rename compatibility
    shim d5_winner_retest.py's own import needs, before that import runs.
    """
    _bind_worktree_package('scraper', worktree_root / 'src' / 'scraper')
    _bind_worktree_package('scripts', worktree_root / 'scripts')
    stage8 = importlib.import_module('scraper.stage8_rally_segmentation')
    stage8._court_scale_boxes = stage8.court_scale_boxes  # noqa: SLF001 — runtime compat shim only


_prepare_worktree_imports(WORKTREE_ROOT)

sys.path.insert(0, str(HERE))
import d5_winner_retest as retest  # noqa: E402

point_winner = importlib.import_module('scraper.point_winner')
stage8_seg = importlib.import_module('scraper.stage8_rally_segmentation')
stage8_score = importlib.import_module('scripts.stage8_score')

Half = point_winner.Half
Verdict = point_winner.Verdict
OTHER_HALF = point_winner.OTHER_HALF
LANDING_OPTS = point_winner.LandingFilterOptions(
    settle_win=7, settle_thr=0.004, settle_min=5, carry_win=7, carry_thr=0.75)


# ---------------------------------------------------------------------------
# Stage 1: the GT-free detection chain
# ---------------------------------------------------------------------------
class DetectedChain(NamedTuple):
    """Everything the chain produces for one video, before any GT is read.

    :param spans: detected rally spans, `[(start, end), ...]`; rally_id is the list index.
    :param contacts: RAW contact candidates with separate `wrist_near` and `suppressed` verdicts.
    :param filtered_contacts: rows where `wrist_near is not False and suppressed is not True` —
        the set `scripts.stage8_score.score_contacts` scores the ball_round column against.
    :param filtered_by_rally: rally_id -> ascending contact frames from `filtered_contacts`.
    :param striker_halves: fitted final-contact half per rally_id (None: no contacts, or a tied
        fit); index-aligned to `spans`.
    :param n_strokes_list: `len(filtered_by_rally[rally_id])` per rally_id (0 when contact-less).
    :param next_servers: winner-serves-next half per rally_id (point_winner.next_server_half).
    :param fitted_first_all: each rally's OWN fitted first-stroke half (the server prediction),
        index-aligned to `spans`; None where `striker_halves[rally_id]` is None.
    :param verdict_rows: rally_id -> VerdictRow, only for rallies with a resolved striker.
    :param landings: rally_id -> Landing or None, same keys as `verdict_rows`.
    :param hit_height_by_frame: contact_frame -> ShuttleSet-coded hit_height (1/2), one entry per
        filtered contact that scored successfully.
    :param hit_height_failures: `(rally_id, stroke_idx, contact_frame, error)` for filtered
        contacts where hit_height raised (shuttle not visible at that exact frame).
    """

    spans: list[tuple[int, int]]
    contacts: list
    filtered_contacts: list
    filtered_by_rally: dict[int, list[int]]
    striker_halves: list
    n_strokes_list: list[int]
    next_servers: list
    fitted_first_all: list
    verdict_rows: dict[int, object]
    landings: dict[int, object | None]
    hit_height_by_frame: dict[int, int]
    hit_height_failures: list[tuple[int, int, int, str]]


def _first_stroke_half(final_half, n_strokes: int):
    """The rally's own fitted first-stroke half, from its fitted final-contact half.

    Same parity formula as point_winner's private `_phase_assignment` at index 0 (last =
    n_strokes - 1; step back from the last stroke, flipping each step): duplicated here as a
    one-line arithmetic fact rather than reaching into that module-private helper, since
    `next_server_half` only ever exposes rally n+1's fitted first stroke (as rally n's winner),
    never a rally's own — Brief H's "server" column needs the latter for every rally.
    """
    return final_half if (n_strokes - 1) % 2 == 0 else OTHER_HALF[final_half]


def build_chain(cfg, track, bboxes, scores, kps, ndet, dead, homo_df, court_info,
                gate_info, gate_res) -> DetectedChain:
    """The GT-free chain: segmentation -> wrist filter -> attribution -> verdict -> hit_height.

    No ground truth is read here. `homo_df`/`court_info` are camera calibration (the recorded
    homography), not stroke-level GT, and are needed structurally for the landing projection and
    the corner-error band, same as the chain steps in the brief list them.

    s28 sticky swap: the gate's candidate source is the sticky_anchor picker, so segment_video
    now also takes ndet and the string-keyed homography / resolution context (`gate_info`,
    `gate_res`) alongside the pose arrays and court box. `wrist_near` is the pure wrist-gate
    verdict; a separate `suppressed` bool records a radius-contest loss.
    """
    spans, contacts = stage8_seg.segment_video(
        track, positions=None, thresholds=None, span_open=stage8_seg.SpanOpen.BACK_FILL,
        replay_mask=dead, pose_bboxes=bboxes, pose_scores=scores, pose_kps=kps,
        pose_ndet=ndet, court_box=cfg.court_box, gate_video_id=str(cfg.vid),
        gate_court_info=gate_info, gate_resolution_table=gate_res, resolution=RESOLUTION)

    filtered_contacts = [
        contact for contact in contacts
        if contact.wrist_near is not False and contact.suppressed is not True
    ]
    filtered_by_rally: dict[int, list[int]] = {}
    for contact in filtered_contacts:
        filtered_by_rally.setdefault(contact.rally_id, []).append(contact.contact_frame)

    striker_halves = []
    for rally_id in range(len(spans)):
        frames = filtered_by_rally.get(rally_id, [])
        guesses = [point_winner.attribute_half(f, track, bboxes, scores, kps, cfg.court_box,
                                               cfg.net_band, RESOLUTION)
                  for f in frames]
        striker_halves.append(point_winner.fit_alternation(guesses))
    n_strokes_list = [len(filtered_by_rally.get(rally_id, [])) for rally_id in range(len(spans))]
    next_servers = point_winner.next_server_half(striker_halves, n_strokes_list)
    fitted_first_all = [
        _first_stroke_half(half, n) if half is not None else None
        for half, n in zip(striker_halves, n_strokes_list)
    ]

    kin = point_winner.build_landing_kinematics(track, bboxes, scores, kps, cfg.court_box, RESOLUTION)
    band_m = point_winner.corner_error_band_m(cfg.vid, homo_df, court_info, REF_ERR_PX)

    verdict_rows: dict[int, object] = {}
    landings: dict[int, object | None] = {}
    for rally_id in range(len(spans)):
        striker = striker_halves[rally_id]
        if striker is None:
            continue  # no contacts, or a tied fit: no verdict row (an automatic miss downstream)
        frames = filtered_by_rally[rally_id]  # non-empty: fit_alternation([]) always ties to None
        final_contact = frames[-1]
        next_start = spans[rally_id + 1][0] if rally_id + 1 < len(spans) else len(track)
        landing = point_winner.pick_landing(final_contact, next_start, track, dead, kin,
                                            LANDING_OPTS, striker, cfg.net_band, RESOLUTION,
                                            court_info)
        verdict_rows[rally_id] = point_winner.rally_verdict(
            rally_id, striker, next_servers[rally_id], landing, band_m)
        landings[rally_id] = landing

    hit_height_by_frame: dict[int, int] = {}
    hit_height_failures: list[tuple[int, int, int, str]] = []
    for rally_id in range(len(spans)):
        for stroke_idx, contact_frame in enumerate(filtered_by_rally.get(rally_id, [])):
            try:
                rows = point_winner.build_hit_height_rows(
                    [(rally_id, stroke_idx, contact_frame)], track, cfg.net_band, RESOLUTION)
            except ValueError as exc:
                hit_height_failures.append((rally_id, stroke_idx, contact_frame, str(exc)))
                continue
            hit_height_by_frame[contact_frame] = rows[0].hit_height

    return DetectedChain(
        spans=spans, contacts=contacts, filtered_contacts=filtered_contacts,
        filtered_by_rally=filtered_by_rally,
        striker_halves=striker_halves, n_strokes_list=n_strokes_list, next_servers=next_servers,
        fitted_first_all=fitted_first_all, verdict_rows=verdict_rows, landings=landings,
        hit_height_by_frame=hit_height_by_frame, hit_height_failures=hit_height_failures,
    )


# ---------------------------------------------------------------------------
# Stage 2: scoring (GT enters only here)
# ---------------------------------------------------------------------------
def _gt_half(player_side: str):
    return Half.TOP if player_side == 'Top' else Half.BOT


def _hit_height_gt_map(cfg) -> dict[tuple[str, int], list[float]]:
    """(set_id, rally) -> per-stroke GT hit_height, ascending frame_num order (may contain NaN).

    Position-based join with `GtRally.stroke_frames` (both ascending): frame_num-collision rows
    are deduped first (drop_duplicates keeps the first), mirroring reconcile()'s own
    `set_unique == sm_frames` alignment assumption, which holds for every rally but the pilot's
    three documented annotation artefacts (reconcile's n_dedup/n_mismatch counts).
    """
    result: dict[tuple[str, int], list[float]] = {}
    for n in (1, 2, 3):
        set_id = f'set{n}'
        sdf = pd.read_csv(cfg.set_dir / f'{set_id}.csv')
        for rally, group in sdf.groupby('rally'):
            ordered = group.sort_values('frame_num').drop_duplicates(subset='frame_num', keep='first')
            result[(set_id, int(rally))] = [float(v) for v in ordered['hit_height']]
    return result


class RallyRow(NamedTuple):
    """One CSV row: a GT rally, its boundary classification, and every scored column."""

    gt_index: int
    set_id: str
    rally: int
    n_gt_strokes: int
    classification: str
    mapped_span: int | None
    ball_round_gt: int
    ball_round_pred: int | None
    ball_round_correct: bool
    timing_matched_n: int
    timing_mean_abs_err: float | None
    player_gt: str
    player_pred: str | None
    player_correct: bool
    server_gt: str
    server_pred: str | None
    server_correct: bool
    hit_height_eligible_n: int
    hit_height_correct_n: int
    getpoint_eligible: bool
    getpoint_gt: str | None
    getpoint_pred: str | None
    getpoint_correct: bool | None
    landing_eligible: bool
    landing_gt: str | None
    landing_pred: str | None
    landing_correct: bool | None


class ColumnAgg(NamedTuple):
    """Primary (over ALL GT rallies/strokes) vs secondary (covered/matched-only) accuracy."""

    primary_correct: int
    primary_total: int
    secondary_correct: int
    secondary_total: int


class VideoScoring(NamedTuple):
    name: str
    rows: list[RallyRow]
    boundary_metrics: dict
    ball_round: ColumnAgg
    ball_round_abs_diffs: list[int]  # over covered rallies, diagnostic distribution
    timing_primary_recall: tuple[int, int]  # matched, total GT strokes (all rallies)
    timing_secondary_recall: tuple[int, int]  # matched, total GT strokes (covered rallies)
    timing_abs_errs: list[int]  # over matched pairs, pooled
    player: ColumnAgg
    server: ColumnAgg
    hit_height: ColumnAgg
    landing: ColumnAgg
    getpoint: ColumnAgg
    hit_height_failures: list[tuple[int, int, int, str]]
    n_raw_contacts: int
    n_filtered_contacts: int


def score_video(cfg, chain: DetectedChain, master: pd.DataFrame, homo_df: pd.DataFrame,
                all_court_info: dict) -> VideoScoring:
    gt_rallies = retest.load_gt_rallies(master, cfg.vid)
    recon = retest.reconcile(cfg, master, gt_rallies)
    court_info = all_court_info[cfg.vid]
    classifications = stage8_score.classify_all(chain.spans, gt_rallies)
    boundary_metrics = stage8_score.score_boundaries(chain.spans, gt_rallies)
    # ball_round's covered-only count-gate, sourced from score_contacts (not hand-rolled): a
    # covered rally passes when its mapped span's filtered-contact count equals its GT stroke
    # count. The per-rally loop below recomputes the same test (it needs the predicted count and
    # the per-rally correct flag for the CSV anyway); the two are cross-checked with an assert.
    contact_metrics = stage8_score.score_contacts(chain.spans, chain.filtered_contacts, gt_rallies)

    per_vid = master[master['vid'] == cfg.vid]
    frame_side = {int(f): str(s) for f, s in zip(per_vid['frame_num'], per_vid['player_side'])}
    hit_height_gt = _hit_height_gt_map(cfg)

    rows: list[RallyRow] = []
    ball_round_correct = ball_round_total = ball_round_covered_correct = ball_round_covered_total = 0
    ball_round_diffs: list[int] = []
    timing_matched_all = timing_gt_all = 0
    timing_matched_covered = timing_gt_covered = 0
    timing_errs: list[int] = []
    player_correct = player_total = player_cov_correct = player_cov_total = 0
    server_correct = server_total = server_cov_correct = server_cov_total = 0
    hh_correct = hh_total = hh_cov_correct = hh_cov_total = 0
    landing_correct = landing_total = landing_cov_correct = landing_cov_total = 0
    getpoint_correct = getpoint_total = getpoint_cov_correct = getpoint_cov_total = 0

    for gt_index, (rally, (category, span_idx)) in enumerate(zip(gt_rallies, classifications)):
        covered = category is stage8_score.RallyBoundary.COVERED
        candidates = chain.filtered_by_rally.get(span_idx, []) if covered else []

        # ball_round: predicted stroke count vs GT stroke count, over ALL / covered rallies.
        ball_round_pred = len(candidates) if covered else None
        br_correct = covered and ball_round_pred == rally.n_strokes
        ball_round_total += 1
        ball_round_correct += int(br_correct)
        if covered:
            ball_round_covered_total += 1
            ball_round_covered_correct += int(br_correct)
            ball_round_diffs.append(abs(ball_round_pred - rally.n_strokes))

        # hit-frame timing + hit_height ride the same greedy match within the covered rally.
        matched: list[tuple[int, int]] = []
        if covered:
            matched = stage8_score.greedy_match(rally.stroke_frames, candidates, TIMING_TOLERANCE)
        timing_gt_all += rally.n_strokes
        timing_matched_all += len(matched)
        if covered:
            timing_gt_covered += rally.n_strokes
            timing_matched_covered += len(matched)
        timing_errs.extend(abs(rally.stroke_frames[gi] - candidates[ci]) for gi, ci in matched)

        gt_heights = hit_height_gt.get((rally.set_id, rally.rally), [])
        rally_hh_eligible = rally_hh_correct = 0
        for stroke_idx in range(rally.n_strokes):
            gt_h = gt_heights[stroke_idx] if stroke_idx < len(gt_heights) else math.nan
            if math.isnan(gt_h):
                continue
            hh_total += 1
            rally_hh_eligible += 1
            if covered:
                hh_cov_total += 1
        for gi, ci in matched:
            gt_h = gt_heights[gi] if gi < len(gt_heights) else math.nan
            if math.isnan(gt_h):
                continue
            det_h = chain.hit_height_by_frame.get(candidates[ci])
            if det_h is None:
                continue
            is_correct = int(gt_h) == det_h
            hh_correct += int(is_correct)
            hh_cov_correct += int(is_correct)
            rally_hh_correct += int(is_correct)

        # player (striker), server: GT always defined; predicted only when covered.
        gt_player = _gt_half(frame_side[rally.stroke_frames[-1]])
        pred_player = chain.striker_halves[span_idx] if covered else None
        p_correct = covered and pred_player is not None and pred_player == gt_player
        player_total += 1
        player_correct += int(p_correct)
        if covered:
            player_cov_total += 1
            player_cov_correct += int(p_correct)

        gt_server = _gt_half(frame_side[rally.stroke_frames[0]])
        pred_server = chain.fitted_first_all[span_idx] if covered else None
        s_correct = covered and pred_server is not None and pred_server == gt_server
        server_total += 1
        server_correct += int(s_correct)
        if covered:
            server_cov_total += 1
            server_cov_correct += int(s_correct)

        # getpoint_player/winner: only eligible when GT names a winner half.
        gt_winner = recon.winners[gt_index].winner_half
        getpoint_eligible = gt_winner is not None
        pred_winner = None
        if covered and span_idx in chain.verdict_rows:
            vr = chain.verdict_rows[span_idx]
            if vr.verdict == Verdict.WON:
                pred_winner = vr.striker_half
            elif vr.verdict == Verdict.LOST:
                pred_winner = OTHER_HALF[vr.striker_half]
        gp_correct = None
        if getpoint_eligible:
            gp_correct = covered and pred_winner is not None and pred_winner == gt_winner
            getpoint_total += 1
            getpoint_correct += int(gp_correct)
            if covered:
                getpoint_cov_total += 1
                getpoint_cov_correct += int(gp_correct)

        # landing side: only eligible when GT records landing pixel coords.
        gt_landing_px = recon.winners[gt_index].landing_px
        landing_eligible = gt_landing_px is not None
        gt_landing_half = None
        pred_landing_half = None
        ld_correct = None
        if landing_eligible:
            proj = point_winner.project_pixels_to_court(
                np.array([[gt_landing_px[0]], [gt_landing_px[1]]]),
                point_winner.HOMOGRAPHY_RESOLUTION, court_info)
            gt_landing_half = Half.TOP if float(proj[1, 0]) < point_winner.NET_COURT_Y else Half.BOT
            landing = chain.landings.get(span_idx) if covered else None
            if landing is not None:
                pred_landing_half = landing.half
            ld_correct = covered and pred_landing_half is not None and pred_landing_half == gt_landing_half
            landing_total += 1
            landing_correct += int(ld_correct)
            if covered:
                landing_cov_total += 1
                landing_cov_correct += int(ld_correct)

        timing_mean_this = float(np.mean([abs(rally.stroke_frames[gi] - candidates[ci])
                                          for gi, ci in matched])) if matched else None
        rows.append(RallyRow(
            gt_index=gt_index, set_id=rally.set_id, rally=rally.rally, n_gt_strokes=rally.n_strokes,
            classification=category.value, mapped_span=span_idx,
            ball_round_gt=rally.n_strokes, ball_round_pred=ball_round_pred, ball_round_correct=br_correct,
            timing_matched_n=len(matched), timing_mean_abs_err=timing_mean_this,
            player_gt=gt_player.value, player_pred=(pred_player.value if pred_player else None),
            player_correct=p_correct,
            server_gt=gt_server.value, server_pred=(pred_server.value if pred_server else None),
            server_correct=s_correct,
            hit_height_eligible_n=rally_hh_eligible, hit_height_correct_n=rally_hh_correct,
            getpoint_eligible=getpoint_eligible,
            getpoint_gt=gt_winner,  # already a plain 'Top'/'Bot' str (retest.GtWinner.winner_half)
            getpoint_pred=(pred_winner.value if pred_winner else None), getpoint_correct=gp_correct,
            landing_eligible=landing_eligible,
            landing_gt=(gt_landing_half.value if gt_landing_half else None),
            landing_pred=(pred_landing_half.value if pred_landing_half else None),
            landing_correct=ld_correct,
        ))

    # Cross-check the hand-rolled covered-only count-gate against score_contacts's own (same
    # definition, computed independently over the pooled candidate/GT-stroke counts): a mismatch
    # would mean this loop's per-rally bookkeeping has drifted from the library function.
    gate = contact_metrics['count_gate']['covered']
    if (gate['pass'], gate['total']) != (ball_round_covered_correct, ball_round_covered_total):
        raise ValueError(
            f'{cfg.name}: ball_round count-gate mismatch vs score_contacts: '
            f'mine {ball_round_covered_correct}/{ball_round_covered_total}, '
            f'score_contacts {gate["pass"]}/{gate["total"]}')

    return VideoScoring(
        name=cfg.name, rows=rows, boundary_metrics=boundary_metrics,
        ball_round=ColumnAgg(ball_round_correct, ball_round_total,
                             ball_round_covered_correct, ball_round_covered_total),
        ball_round_abs_diffs=ball_round_diffs,
        timing_primary_recall=(timing_matched_all, timing_gt_all),
        timing_secondary_recall=(timing_matched_covered, timing_gt_covered),
        timing_abs_errs=timing_errs,
        player=ColumnAgg(player_correct, player_total, player_cov_correct, player_cov_total),
        server=ColumnAgg(server_correct, server_total, server_cov_correct, server_cov_total),
        hit_height=ColumnAgg(hh_correct, hh_total, hh_cov_correct, hh_cov_total),
        landing=ColumnAgg(landing_correct, landing_total, landing_cov_correct, landing_cov_total),
        getpoint=ColumnAgg(getpoint_correct, getpoint_total, getpoint_cov_correct, getpoint_cov_total),
        hit_height_failures=chain.hit_height_failures,
        n_raw_contacts=len(chain.contacts), n_filtered_contacts=len(chain.filtered_contacts),
    )


# ---------------------------------------------------------------------------
# Output: CSV + console summary
# ---------------------------------------------------------------------------
CSV_COLUMNS = list(RallyRow._fields)


def _fmt(value) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return '' if not np.isfinite(value) else f'{value:.6f}'
    return str(value)


def write_rallies_csv(rows: list[RallyRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([_fmt(v) for v in row])


def _acc(agg: ColumnAgg) -> tuple[float | None, float | None]:
    primary = agg.primary_correct / agg.primary_total if agg.primary_total else None
    secondary = agg.secondary_correct / agg.secondary_total if agg.secondary_total else None
    return primary, secondary


def _acc_str(agg: ColumnAgg) -> str:
    primary, secondary = _acc(agg)
    p_s = 'n/a' if primary is None else f'{primary:.4f}'
    s_s = 'n/a' if secondary is None else f'{secondary:.4f}'
    return (f'primary {p_s} ({agg.primary_correct}/{agg.primary_total})   '
           f'covered-only {s_s} ({agg.secondary_correct}/{agg.secondary_total})')


def render_summary(scoring: VideoScoring) -> list[str]:
    out = ['=' * 100, f'BRIEF H END-TO-END — {scoring.name}', '=' * 100]
    bm = scoring.boundary_metrics
    out.append(f'  rally (boundary):  covered {bm["covered"]}/{bm["n_gt_rallies"]} '
               f'({bm["covered_fraction"]:.4f})   split {bm["split"]}   missed {bm["missed"]}   '
               f'merged_spans {bm["merged_spans"]}   spurious_spans {bm["spurious_spans"]}')
    if bm['start_alignment']:
        sa = bm['start_alignment']
        out.append(f'    start offset (span-first_stroke): mean {sa["mean"]:.2f}  '
                   f'median {sa["median"]:.2f}  n={sa["n"]}')
    if bm['end_alignment']:
        ea = bm['end_alignment']
        out.append(f'    end offset (span-last_stroke):    mean {ea["mean"]:.2f}  '
                   f'median {ea["median"]:.2f}  n={ea["n"]}')

    out.append(f'  ball_round:  {_acc_str(scoring.ball_round)}')
    if scoring.ball_round_abs_diffs:
        diffs = np.array(scoring.ball_round_abs_diffs)
        out.append(f'    |detected-GT strokes| over covered rallies: mean {diffs.mean():.4f}  '
                   f'median {float(np.median(diffs)):.4f}  n={len(diffs)}')

    matched_all, gt_all = scoring.timing_primary_recall
    matched_cov, gt_cov = scoring.timing_secondary_recall
    recall_p = matched_all / gt_all if gt_all else None
    recall_s = matched_cov / gt_cov if gt_cov else None
    out.append(f'  time/frame_num (hit-frame timing, tolerance {TIMING_TOLERANCE}f):  '
               f'match recall primary {"n/a" if recall_p is None else f"{recall_p:.4f}"} '
               f'({matched_all}/{gt_all})   covered-only '
               f'{"n/a" if recall_s is None else f"{recall_s:.4f}"} ({matched_cov}/{gt_cov})')
    if scoring.timing_abs_errs:
        errs = np.array(scoring.timing_abs_errs)
        out.append(f'    |gt-detected| over matched pairs: mean {errs.mean():.4f}  '
                   f'median {float(np.median(errs)):.4f}  n={len(errs)}')

    out.append(f'  player (striker):  {_acc_str(scoring.player)}')
    out.append(f'  server:  {_acc_str(scoring.server)}')
    out.append(f'  hit_height:  {_acc_str(scoring.hit_height)}')
    out.append(f'  landing side:  {_acc_str(scoring.landing)}')
    out.append(f'  getpoint_player (winner):  {_acc_str(scoring.getpoint)}')
    out.append(f'  contacts: raw {scoring.n_raw_contacts}  filtered {scoring.n_filtered_contacts}')
    if scoring.hit_height_failures:
        out.append(f'  hit_height failures (caught, shuttle not visible at contact): '
                   f'{len(scoring.hit_height_failures)}')
    out.append('')
    return out


def render_pooled(scorings: list[VideoScoring]) -> list[str]:
    out = ['=' * 100, 'POOLED (pilot + vid15)', '=' * 100]

    def pool(attr: str) -> ColumnAgg:
        aggs = [getattr(s, attr) for s in scorings]
        return ColumnAgg(
            sum(a.primary_correct for a in aggs), sum(a.primary_total for a in aggs),
            sum(a.secondary_correct for a in aggs), sum(a.secondary_total for a in aggs))

    n_gt_total = sum(s.boundary_metrics['n_gt_rallies'] for s in scorings)
    n_covered_total = sum(s.boundary_metrics['covered'] for s in scorings)
    out.append(f'  rally (boundary): covered {n_covered_total}/{n_gt_total} '
               f'({n_covered_total / n_gt_total:.4f})')
    out.append(f'  ball_round:  {_acc_str(pool("ball_round"))}')

    matched_all = sum(s.timing_primary_recall[0] for s in scorings)
    gt_all = sum(s.timing_primary_recall[1] for s in scorings)
    matched_cov = sum(s.timing_secondary_recall[0] for s in scorings)
    gt_cov = sum(s.timing_secondary_recall[1] for s in scorings)
    recall_p = matched_all / gt_all if gt_all else None
    recall_s = matched_cov / gt_cov if gt_cov else None
    out.append(f'  time/frame_num:  match recall primary '
               f'{"n/a" if recall_p is None else f"{recall_p:.4f}"} ({matched_all}/{gt_all})   '
               f'covered-only {"n/a" if recall_s is None else f"{recall_s:.4f}"} '
               f'({matched_cov}/{gt_cov})')
    all_errs = [e for s in scorings for e in s.timing_abs_errs]
    if all_errs:
        errs = np.array(all_errs)
        out.append(f'    |gt-detected| over matched pairs: mean {errs.mean():.4f}  '
                   f'median {float(np.median(errs)):.4f}  n={len(errs)}')

    out.append(f'  player (striker):  {_acc_str(pool("player"))}')
    out.append(f'  server:  {_acc_str(pool("server"))}')
    out.append(f'  hit_height:  {_acc_str(pool("hit_height"))}')
    out.append(f'  landing side:  {_acc_str(pool("landing"))}')
    out.append(f'  getpoint_player (winner):  {_acc_str(pool("getpoint"))}')
    n_raw = sum(s.n_raw_contacts for s in scorings)
    n_filt = sum(s.n_filtered_contacts for s in scorings)
    n_hh_fail = sum(len(s.hit_height_failures) for s in scorings)
    out.append(f'  contacts: raw {n_raw}  filtered {n_filt}   hit_height failures {n_hh_fail}')
    out.append('')
    return out


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
def main() -> None:
    master = pd.read_csv(retest.SHOTS_MASTER)
    homo_df = pd.read_csv(retest.HOMOGRAPHY_CSV).set_index('id')
    all_court_info = retest.load_all_court_info(retest.HOMOGRAPHY_CSV)
    # segment_video's sticky gate joins by string video id (s28 convention).
    gate_info = {str(video_id): info for video_id, info in all_court_info.items()}
    gate_res = pd.read_csv(retest.RESOLUTION_CSV).set_index('id')
    gate_res.index = gate_res.index.astype(str)

    scorings: list[VideoScoring] = []
    md5s: dict[str, str] = {}
    for cfg in (retest.PILOT, retest.VID15):
        track = np.load(cfg.track_path)
        bboxes = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_bboxes.npy')
        scores = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_scores.npy')
        kps = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_kps.npy')
        ndet = np.load(cfg.pose_dir / f'{cfg.pose_prefix}_ndet.npy')
        dead = np.load(cfg.mask_path)  # crowned per-video mask; True = dead
        court_info = all_court_info[cfg.vid]

        chain = build_chain(cfg, track, bboxes, scores, kps, ndet, dead, homo_df, court_info,
                            gate_info, gate_res)
        scoring = score_video(cfg, chain, master, homo_df, all_court_info)
        scorings.append(scoring)

        out_path = HERE / f'{cfg.name}_results' / 'h_end_to_end' / 'rallies.csv'
        write_rallies_csv(scoring.rows, out_path)
        md5s[cfg.name] = _md5(out_path)

        print('\n'.join(render_summary(scoring)))

    print('\n'.join(render_pooled(scorings)))
    print('=== MD5SUMS ===')
    for name, digest in sorted(md5s.items()):
        print(f'  {digest}  {name}_results/h_end_to_end/rallies.csv')


if __name__ == '__main__':
    main()
