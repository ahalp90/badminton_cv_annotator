"""``sticky_anchor`` heuristic: per-slot EMA + fixed-prior tracking.

Picks one detection per slot (Top, Bottom) by proximity to a weighted
anchor. The anchor for slot ``s`` is:

    effective_anchor[s] = prior_weight * halfcourt_centre[s]
                        + (1 - prior_weight) * ema[s]

``halfcourt_centre`` is derived per-clip from the homography borders and
collapses to ``(0.5, 0.25)`` and ``(0.5, 0.75)`` for ShuttleSet's
canonical rectangle. The EMA starts at ``halfcourt_centre`` and updates
with the slot's picked court position each frame, gated on that pick
landing inside the court by ``update_gate_eps`` (pollution guard).

Per-frame flow:

A. Score-filter raw detections, project each bbox-bottom-centre to
   normalised court coords, drop NaN projections.
B. Compute effective anchors and the (candidate, slot) distance matrix.
C. For each slot in order (Bottom first, then Top):
   - Drop candidates past ``sanity_ceiling``.
   - Drop candidates closer to the OTHER slot's anchor (Voronoi partition
     against cross-half capture).
   - For Top, drop whichever candidate Bottom assigned.
   - ``argmin`` D on survivors. If any survivor is within
     ``tiebreaker_tol`` of the winner's D: drop sitting candidates
     (body-frame knee/torso projection), pick largest bbox area among
     what remains. Fall back to the original ``argmin`` if the sitting
     filter drops everyone.
D. Rally-presence check: if both slots picked but neither pick lands
   inside ``[-generous_margin, 1 + generous_margin]`` on both axes, zero
   both (catches cutaway / pure-bystander frames).
E. Write outputs. EMA resets to ``halfcourt_centre[s]`` on zero; updates
   by ``ema_alpha`` on a picked slot whose pick is within the
   ``update_gate_eps`` in-court test.

Operational reference: ``docs/architecture_notes/mmpose_heuristic/mmpose_heuristic.md``
(algorithm spec, hyperparameters, calibration, failsafe gate). Design log + rejected
variants: ``docs/architecture_notes/mmpose_heuristic/historical_mmpose_heuristic_investigation.md``,
"Sticky_anchor design, finalised (2026-04-22)" section.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from pipeline.config import COCO_N_JOINTS
from shared.court import normalize_position, to_court_coordinate

from .base import (
    DOUBLES_COUNT_MARGIN,
    SITTING_THRESHOLD,
    ClipContext,
    HeuristicOutput,
    RawClip,
    count_standing_in_court,
    is_sitting,
)


@dataclass(frozen=True)
class StickyAnchorParams:
    """Hyperparameters for the sticky_anchor heuristic.

    Single source of truth: ``apply_heuristic.py`` derives its argparse
    ``--<field>`` block from these fields, and ``apply`` constructs an
    instance from any keyword overrides at the registry boundary.
    """
    # Tested at 25/30 fps. No vid fps scaling. At 60fps anchor drifts faster, but also snaps back faster.
    prior_weight: float = 0.75
    ema_alpha: float = 0.1
    sanity_ceiling: float = 0.6
    generous_margin: float = 0.15
    score_filter: float = 0.2
    tiebreaker_tol: float = 0.05
    sitting_threshold: float = SITTING_THRESHOLD
    update_gate_eps: float = 0.01
    count_margin: float = DOUBLES_COUNT_MARGIN


SLOT_TOP = 0
SLOT_BOTTOM = 1
SLOT_ORDER = (SLOT_BOTTOM, SLOT_TOP)  # pick order: Bottom first, Top second
OTHER_SLOT = {SLOT_TOP: SLOT_BOTTOM, SLOT_BOTTOM: SLOT_TOP}


def compute_halfcourt_centres(court_info: dict) -> np.ndarray:
    """Halfcourt centres in normalised [0, 1] coords, returned as (2, 2).

    Row 0 = Top (y = 0.25 for ShuttleSet), row 1 = Bottom (y = 0.75).
    Computed from the homography borders rather than hardcoded so the
    code stays correct for any future homography that maps to a
    non-ShuttleSet canonical rectangle. For ShuttleSet the formula
    collapses to the constants above.
    """
    bL, bR = court_info["border_L"], court_info["border_R"]
    bU, bD = court_info["border_U"], court_info["border_D"]

    cx = (bL + bR) / 2
    y_top = bU + (bD - bU) / 4
    y_bot = bU + 3 * (bD - bU) / 4

    # normalize_position wants (axis, n); build it directly that way.
    xs = np.array([cx, cx], dtype=np.float64)
    ys = np.array([y_top, y_bot], dtype=np.float64)
    raw = np.stack([xs, ys], axis=0)  # (axis=2, slot=2)
    return normalize_position(raw, court_info).T  # (slot, axis)


def project_bbox_bottom_centre(
    bboxes: np.ndarray, ctx: ClipContext,
) -> np.ndarray:
    """Project (n, 4) pixel-space bboxes to (n, 2) normalised court coords.

    Uses bbox bottom-centre ``((x1+x2)/2, y2)`` as the foot proxy.
    """
    x1, _, x2, y2 = bboxes.T
    bottom_centres = np.stack([(x1 + x2) / 2, y2], axis=0)  # (2, n)
    court = to_court_coordinate(
        bottom_centres, ctx.vid, ctx.all_court_info, ctx.res_df,
    )  # (2, n)
    normalised = normalize_position(court, ctx.all_court_info[ctx.vid])  # (2, n)
    return normalised.T  # (n, 2)


def in_generous_court(pos: np.ndarray, margin: float) -> bool:
    return bool(
        -margin <= pos[0] <= 1 + margin and -margin <= pos[1] <= 1 + margin
    )


class FrameAnalysis(NamedTuple):
    """Observable sticky-anchor state and picks for one frame.

    ``picks`` uses filtered-candidate indices and is ``None`` only on a
    whole-frame picker failure. ``filtered_to_raw`` maps that index space back
    to raw pose slots. Candidate arrays and the mapping are retained whenever
    the score-filtered, NaN-filtered projection population is non-empty.
    """

    standing_in_court_count: int
    picks: list[int] | None
    court_base_pos: np.ndarray | None
    kps: np.ndarray | None
    bboxes: np.ndarray | None
    filtered_to_raw: np.ndarray | None


# Compatibility aliases for frozen callers; remove at Stage 7.
_compute_halfcourt_centres = compute_halfcourt_centres
_project_bbox_bottom_centre = project_bbox_bottom_centre
_in_generous_court = in_generous_court


def analyse_frame(
    raw: RawClip,
    f: int,
    ema: np.ndarray,
    halfcourt_centre: np.ndarray,
    ctx: ClipContext,
    params: StickyAnchorParams,
) -> FrameAnalysis:
    """Analyse and pick one frame without discarding observable evidence.

    The picker candidate pipeline is deliberately score filter followed by a
    NaN-only projection filter. In particular, infinite projections remain in
    the filtered candidate space exactly as they do for ``pick_one_frame``.
    """
    n = int(raw.ndet[f])
    if n == 0:
        return FrameAnalysis(0, None, None, None, None, None)

    # Step A: score filter on real detections.
    scores_f = raw.scores[f, :n]
    pass_score = scores_f > params.score_filter
    if not pass_score.any():
        return FrameAnalysis(0, None, None, None, None, None)

    keep_idx = np.nonzero(pass_score)[0]
    bboxes_f = raw.bboxes[f, keep_idx].astype(np.float64)  # (k, 4)
    kps_f = raw.kps[f, keep_idx].astype(np.float64)  # (k, J, 2)

    court_base_pos = project_bbox_bottom_centre(bboxes_f, ctx)  # (k, 2)
    valid = ~np.isnan(court_base_pos).any(axis=1)
    if not valid.any():
        return FrameAnalysis(0, None, None, None, None, None)

    filtered_to_raw = keep_idx[valid]
    bboxes_f = bboxes_f[valid]
    kps_f = kps_f[valid]
    court_base_pos = court_base_pos[valid]
    # Per-candidate invariant from here: bboxes_f, kps_f, court_base_pos,
    # sitting, bbox_areas all share the same [0, k) index space, and
    # the eligible/tied boolean masks below operate on it.

    # Step B: effective anchors + full distance matrix.
    effective_anchor = (
        params.prior_weight * halfcourt_centre + (1 - params.prior_weight) * ema
    )  # (2, 2); row = slot
    # Outer-difference via broadcasting: (k, 1, 2) against (1, 2, 2)
    # broadcasts to (k, 2, 2); L2 over the last axis collapses to (k, 2).
    distances = np.linalg.norm(
        court_base_pos[:, None, :] - effective_anchor[None, :, :],
        axis=-1,
    )  # (k, 2); distances[c, s]

    # Precompute per-candidate sitting + bbox area (tiebreaker only).
    # Eager rather than lazy: k is small (~2-6 after filtering) and the
    # vectorised per-candidate cost is trivial. Eager removes one more
    # place to get an off-by-one wrong when slicing into the tiebreaker.
    sitting = is_sitting(kps_f, params.sitting_threshold)  # (k,) bool
    x1, y1, x2, y2 = bboxes_f.T
    bbox_areas = (x2 - x1) * (y2 - y1)

    # Head-count for the doubles guard, not the pick: counted over the full
    # candidate set because doubles evidence is the crowd of in-court feet, not
    # who wins the two slots. Margin narrower than generous_margin and
    # sitting-exempt so persistent seated officials behind the lines never count
    # (D26).
    n_counted = count_standing_in_court(court_base_pos, sitting, params.count_margin)

    # Step C: process slots Bottom first, then Top.
    picks: list[int] = [-1, -1]
    for s in SLOT_ORDER:
        other = OTHER_SLOT[s]

        within_sanity = distances[:, s] <= params.sanity_ceiling
        # Closer-to-own-anchor rule (Voronoi partition): keep candidates
        # where the own slot's anchor is closer than/equidistant to the other slot's.
        # Equality passes both slots; Bottom-first order resolves any tie.
        in_own_voronoi_cell = distances[:, s] <= distances[:, other]

        # Bool mask over k candidate bboxes; True iff both filters pass.
        eligible = within_sanity & in_own_voronoi_cell
        # picks[other]: candidate bbox index.
        if picks[other] >= 0:
            eligible[picks[other]] = False

        if not eligible.any():
            continue

        eligible_idx = np.nonzero(eligible)[0]
        winner = int(eligible_idx[np.argmin(distances[eligible_idx, s])])
        winner_d = distances[winner, s]

        # Tiebreaker: any other eligible within tiebreaker_tol.
        tied = eligible & (
            np.abs(distances[:, s] - winner_d) < params.tiebreaker_tol
        )
        if tied.sum() > 1:
            standing_tied = tied & ~sitting
            if standing_tied.any():
                st_idx = np.nonzero(standing_tied)[0]
                winner = int(st_idx[np.argmax(bbox_areas[st_idx])])
            # Else: sitting dropped everyone; revert to original argmin.

        picks[s] = winner

    # Step D: rally-presence check.
    if picks[SLOT_TOP] >= 0 and picks[SLOT_BOTTOM] >= 0:
        top_p = court_base_pos[picks[SLOT_TOP]]
        bot_p = court_base_pos[picks[SLOT_BOTTOM]]
        if not in_generous_court(top_p, params.generous_margin) and not in_generous_court(
            bot_p, params.generous_margin
        ):
            return FrameAnalysis(
                n_counted, None, court_base_pos, kps_f, bboxes_f, filtered_to_raw,
            )

    if picks == [-1, -1]:
        return FrameAnalysis(
            n_counted, None, court_base_pos, kps_f, bboxes_f, filtered_to_raw,
        )

    return FrameAnalysis(
        n_counted, picks, court_base_pos, kps_f, bboxes_f, filtered_to_raw,
    )


def pick_one_frame(
    raw: RawClip,
    f: int,
    ema: np.ndarray,
    halfcourt_centre: np.ndarray,
    ctx: ClipContext,
    params: StickyAnchorParams,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Pick (Bottom, Top) detections for a single frame.

    Returns ``None`` for a full-frame failure: no detections, score
    filter empties, all projections NaN, rally-presence rejects both
    picks, or no slot ended up with a winner. The caller treats ``None``
    as ``failed[f] = True`` plus a full EMA reset.

    Otherwise returns ``(picks, court_base_pos, kps_f, bboxes_f, n_counted)``,
    where ``picks`` is a length-2 list (``-1`` in any unpicked slot),
    the three arrays are per-candidate after the score + NaN filters, and
    ``n_counted`` is the doubles-guard head count over the full candidate set:
    > 2 standing candidates within ``count_margin`` is doubles evidence, even
    though only two are ever picked. The caller writes outputs and updates the
    per-slot EMA based on the picks.
    """
    analysis = analyse_frame(raw, f, ema, halfcourt_centre, ctx, params)
    if analysis.picks is None:
        return None
    assert analysis.court_base_pos is not None
    assert analysis.kps is not None
    assert analysis.bboxes is not None
    return (
        analysis.picks,
        analysis.court_base_pos,
        analysis.kps,
        analysis.bboxes,
        analysis.standing_in_court_count,
    )


def _run_clip(
    raw: RawClip, ctx: ClipContext, normalize_joints, params: StickyAnchorParams,
) -> tuple[HeuristicOutput, np.ndarray]:
    """Drive the per-frame loop and return ``(output, ema_history)``.

    ``normalize_joints`` is injected so the caller controls how keypoints
    are normalised. ``ema_history`` has shape ``(F, 2, 2)`` and records the
    post-update EMA at the end of every frame; the public ``apply`` wrapper
    discards it, tests use it.
    """
    court_info = ctx.all_court_info[ctx.vid]
    halfcourt_centre = compute_halfcourt_centres(court_info)  # (2, 2)

    num_frames = raw.kps.shape[0]
    failed = np.zeros(num_frames, dtype=bool)
    pos = np.zeros((num_frames, 2, 2), dtype=np.float64)
    joints = np.zeros((num_frames, 2, COCO_N_JOINTS, 2), dtype=np.float64)
    ema_history = np.zeros((num_frames, 2, 2), dtype=np.float64)
    # (F,) doubles evidence: True where > 2 standing candidates projected within
    # count_margin on a success frame. Left False on failure frames (init default):
    # a fully-failed frame contributes no doubles evidence.
    overcount = np.zeros(num_frames, dtype=bool)

    # Per-slot EMA, initialised to halfcourt_centre.
    ema = halfcourt_centre.copy()

    for f in range(num_frames):
        result = pick_one_frame(raw, f, ema, halfcourt_centre, ctx, params)
        if not result:
            failed[f] = True
            ema[:] = halfcourt_centre
            ema_history[f] = ema
            continue

        picks, court_base_pos, kps_f, bboxes_f, n_counted = result
        overcount[f] = n_counted > 2

        # Step E: write outputs + update EMAs. Mixed result (one slot
        # picked, one not) still resets the unpicked slot's EMA below.
        frame_has_zero = False
        for s in (SLOT_TOP, SLOT_BOTTOM):
            if picks[s] < 0:
                frame_has_zero = True
                ema[s] = halfcourt_centre[s]
                continue
            cbp = court_base_pos[picks[s]]
            pos[f, s] = cbp
            joints[f, s] = normalize_joints(
                arr=kps_f[picks[s]][None, :, :],
                bbox=bboxes_f[picks[s]][None, :],
                v_height=None,
                center_align=True,
            )[0]
            if in_generous_court(cbp, params.update_gate_eps):
                ema[s] = params.ema_alpha * cbp + (1 - params.ema_alpha) * ema[s]

        failed[f] = frame_has_zero
        ema_history[f] = ema

    return (
        HeuristicOutput(pos=pos, joints=joints, failed=failed, overcount=overcount),
        ema_history,
    )


def apply(raw: RawClip, ctx: ClipContext, **hyperparams) -> HeuristicOutput:
    """Apply the sticky_anchor heuristic to a raw clip.

    Keeps the registry-contract ``apply(raw, ctx, **kw)`` signature; the
    ``StickyAnchorParams`` instance is constructed at this boundary.
    """
    # Lazy import: prepare_train_on_shuttleset pulls in torch/pandas at module
    # load (its rtmlib import is itself lazy).
    from preparing_data.prepare_train_on_shuttleset import (  # noqa: PLC0415
        normalize_joints,
    )

    params = StickyAnchorParams(**hyperparams)
    output, _ema_history = _run_clip(raw, ctx, normalize_joints, params)
    return output
