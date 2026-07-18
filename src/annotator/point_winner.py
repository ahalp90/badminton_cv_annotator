"""Stage 10: point-winner verdicts (D5 chain — attribution, alternation fit, landing, verdict).

Wrist-anchored striker attribution in body-height units, an alternation-rhythm fit for the
final-contact half, a kinematic landing filter (a settle cap plus a carry filter, both refined by
an ankle rule), and a next-server winner call with a landing-geometry best-guess fallback. Promoted
from the D5 point-winner detector proven out in
local_scratch/autograder_architecture/d5_winner_retest.py and d5_landing_arms.py (measured on the
ShuttleSet pilot and trial videos, GT-anchored against the per-set winner labels). Only the SHIPPED
chain lands here: the box-height attribution arm, the window fix (a lob that leaves the frame top
waits for re-entry), the combined landing filter with the ankle rule on, and the next-server
verdict. The three attribution ablation arms, the parameter sweeps, and the GT reconciliation that
measured all of this stay in the scratch harness — they answer "is this the right chain", which is
already settled; this module only carries the chain itself.

The chain assumes full-frame broadcast footage: the top-exit wait treats the frame's top edge as
sky (a lob leaving it will fall back into view), which a tight crop whose top edge cuts through
play would break. That assumption rides the measured configuration and is not a parameter.

Library-only: no argparse main. Every function here reads precomputed per-video arrays (a shuttle
track, court-scale pose boxes, a replay/dead mask, a homography) for one rally or one frame at a
time; there is no established pipeline path convention yet for wiring stage 8/9's outputs into a
point-winner CLI, so this stays a library the caller composes over a rally list, the way the
harness's own per-rally loop does. See
local_scratch/autograder_architecture/d5_stage10_pin.py for a runnable example that reproduces the
D5 retest's arm-2 verdict CSVs from this module.
"""
from __future__ import annotations

from enum import StrEnum
import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from shared.court import (
    HOMOGRAPHY_RESOLUTION,
    convert_homogeneous,
    get_corner_camera,
    normalize_position,
    project,
    scale_pos_by_resolution,
)

from .rally_segmentation import (
    ANKLE_L,
    ANKLE_R,
    WRIST_L,
    WRIST_R,
    CourtBox,
    compute_speed,
    court_scale_boxes,
    rolling_nanmedian,
    true_runs,
)
from .fps_constants import FpsConstants


class Half(StrEnum):
    """Which court half a striker, receiver, or landing sits in. Byte-identical to the harness's
    plain `'Top'`/`'Bot'` strings, so a CSV written from these serialises the same way."""

    TOP = 'Top'
    BOT = 'Bot'


OTHER_HALF = {Half.TOP: Half.BOT, Half.BOT: Half.TOP}


class Verdict(StrEnum):
    """A rally's outcome relative to the striker. Byte-identical to the harness's 'won'/'lost'."""

    WON = 'won'
    LOST = 'lost'


class VerdictSource(StrEnum):
    """Where a verdict row's winner call came from. Byte-identical to the harness's strings."""

    NEXT_SERVER = 'next_server'      # winner-serves-next: rally n+1's fitted first-stroke half
    LANDING_GEOMETRY = 'landing_geometry'  # in/out of the receiver's singles half
    NET_RULE = 'net_rule'            # flight never crossed the net and died in the net band


# Singles-court geometry in the normalised court space (doubles outline -> unit square, x = court
# WIDTH 6.10 m, y = court LENGTH 13.40 m). Singles sidelines sit inset (6.10-5.18)/2 = 0.46 m each
# side; baselines are the unit-square y edges; net at y=0.5.
COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
SINGLES_INSET_M = 0.46
SINGLES_X_LO = SINGLES_INSET_M / COURT_WIDTH_M          # ~0.07541
SINGLES_X_HI = 1.0 - SINGLES_INSET_M / COURT_WIDTH_M    # ~0.92459
NET_COURT_Y = 0.5

# Landing-window constants. A sustained track loss is a run of >= this many consecutive invisible
# frames (mirrors stage 8's BLIP_MAX_FRAMES, a gap longer than a blip); the descending run needs
# >= 3 visible samples.
SUSTAINED_LOSS_FRAMES = 10
MIN_DESCEND_SAMPLES = 3

# Image-y fraction that counts as the frame's TOP edge for the window fix (a lob that exits the
# top leaves its last visible sample this close to y=0). Also the terminal-at-border threshold
# (2% of any edge): matches the harness's single source of truth for "at the top edge".
TOP_EDGE_FRAC = 0.02

# +/- frames around a contact frame for the box-height attribution's body-scale denominator.
BODY_UNIT_HALF_WINDOW = 12


# ---------------------------------------------------------------------------
# Court projection
# ---------------------------------------------------------------------------
def project_pixels_to_court(
    px_xy: np.ndarray, resolution: tuple[float, float], court_info: dict,
) -> np.ndarray:
    """(2, N) pixels at `resolution` -> (2, N) normalised court coords (doubles outline = unit sq).

    One source of truth for the two projections the harness kept separate: homography-resolution
    pixels (pass `shared.court.HOMOGRAPHY_RESOLUTION`; the resolution scale is then an exact 1.0
    no-op) and working-resolution pixels (pass the pose/track resolution; scaled down to the
    homography's recorded resolution before the matrix multiply). `H` lives inside `court_info`
    (the `shared.court.get_court_info`/`load_all_court_info` shape).

    :param px_xy: (2, N) pixel coordinates at `resolution`.
    :param resolution: (width, height) `px_xy` is expressed in.
    :param court_info: dict carrying `'H'` plus the court boundary keys `normalize_position` reads.
    :return: (2, N) normalised court coordinates.
    """
    scaled = scale_pos_by_resolution(px_xy, width=resolution[0], height=resolution[1])
    court = project(court_info['H'], convert_homogeneous(scaled))
    return normalize_position(court, court_info)


# ---------------------------------------------------------------------------
# Striker attribution (the shipped wrist_boxh arm: nearer-wrist px / mean windowed box height)
# ---------------------------------------------------------------------------
def body_unit_gaps(
    frame: int, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
    cand_slots: list[int], bboxes: np.ndarray, scores: np.ndarray, kps: np.ndarray,
    court_box: CourtBox, track: np.ndarray, width: float, height: float, half_window: int,
) -> np.ndarray:
    """Body-unit (box-height) shuttle-to-candidate distance, the shipped attribution arm.

    Numerator: the contact-frame candidate's nearer-wrist distance to the shuttle in working-res
    pixels. Denominator: that candidate's mean box-pixel-height over a +/-12-frame window. Per
    window frame we re-detect the court-scale boxes and associate the one whose centre is nearest
    this candidate's contact-frame bbox centre, accepting only within one contact-frame box
    height. The contact frame always self-matches at distance 0, so every candidate's denominator
    has at least that sample.

    :param x1..y2: the contact-frame candidates' court-scale boxes (px), from court_scale_boxes.
    :param cand_slots: each contact-frame candidate's kps/scores slot id.
    :return: gaps (k,), one per contact-frame candidate; argmin picks the nearest as elsewhere.
    """
    shuttle_x_px = track[frame, 0] * width
    shuttle_y_px = track[frame, 1] * height
    wrists = kps[frame, cand_slots][:, (WRIST_L, WRIST_R), :]  # (k candidates, 2 wrists, xy) px
    wrist_px = np.hypot(wrists[..., 0] - shuttle_x_px, wrists[..., 1] - shuttle_y_px).min(axis=1)

    cand_cx = (x1 + x2) / 2.0  # contact-frame candidate bbox centres (px)
    cand_cy = (y1 + y2) / 2.0
    cand_h = y2 - y1           # contact-frame box heights (px); the association threshold

    n_cand = len(x1)
    n_frames = len(bboxes)
    lo = max(0, frame - half_window)
    hi = min(n_frames - 1, frame + half_window)
    denom_sum = np.zeros(n_cand)
    denom_count = np.zeros(n_cand, dtype=int)
    for g in range(lo, hi + 1):
        gx1, gy1, gx2, gy2, _ = court_scale_boxes(bboxes[g], scores[g], court_box)
        if len(gx1) == 0:
            continue
        g_cx = (gx1 + gx2) / 2.0
        g_cy = (gy1 + gy2) / 2.0
        for cand in range(n_cand):
            dists = np.hypot(g_cx - cand_cx[cand], g_cy - cand_cy[cand])
            nearest = int(np.argmin(dists))
            if dists[nearest] > cand_h[cand]:  # association beyond one contact-frame box height
                continue
            denom_sum[cand] += float(gy2[nearest] - gy1[nearest])
            denom_count[cand] += 1
    if (denom_count == 0).any():  # the contact frame should always self-match; empty => a bug
        raise ValueError(f'body-unit gap: a candidate had no accepted window frame at contact {frame}')
    denom_px = denom_sum / denom_count
    if not np.all(np.isfinite(denom_px)) or (denom_px <= 0.0).any():
        raise ValueError(f'body-unit gap: non-finite or non-positive body-scale denominator at contact {frame}')
    return wrist_px / denom_px


def attribute_half(
    frame: int, track: np.ndarray, bboxes: np.ndarray, scores: np.ndarray, kps: np.ndarray,
    court_box: CourtBox, net_band: tuple[float, float], resolution: tuple[float, float],
    body_unit_half_window: int,
) -> Half | None:
    """Court half of the nearest court-scale detection to the shuttle at `frame`, or None.

    None (ambiguous) when the shuttle is invisible, no court-scale detection is present, or the
    nearest detection's foot sits inside the net band. Shuttle-to-candidate distance is the
    shipped wrist_boxh arm: the nearer-wrist pixel gap divided by that candidate's mean windowed
    box height (body units), not image-fraction — see `_body_unit_gaps`. The foot point is that
    nearest candidate's bbox bottom-centre y (pixels) tested against the band.

    :param resolution: (width, height) the shuttle xy and the wrist pixels normalise by.
    """
    if track[frame, 2] != 1:
        return None
    x1, y1, x2, y2, cand_scores = court_scale_boxes(bboxes[frame], scores[frame], court_box)
    if len(x1) == 0:
        return None
    # court_scale_boxes drops padding and reorders to the court-scale subset without slot ids; its
    # 5th element is those detections' scores, an exact slice of scores[frame], so match them back
    # to recover each candidate's original slot (kps aligns with scores/bboxes by slot; finite
    # detection scores are unique within a frame).
    frame_scores = scores[frame]
    cand_slots = [int(np.flatnonzero(frame_scores == s)[0]) for s in cand_scores]
    width, height = resolution

    gaps = body_unit_gaps(frame, x1, y1, x2, y2, cand_slots, bboxes, scores, kps, court_box,
                           track, width, height, body_unit_half_window)

    foot_y = float(y2[int(np.argmin(gaps))])  # bbox bottom-centre y, pixels
    band_lo, band_hi = net_band
    if foot_y < band_lo:
        return Half.TOP
    if foot_y > band_hi:
        return Half.BOT
    return None  # inside the net band


# Compatibility alias for frozen callers; remove when the old sticky seam retires.
_body_unit_gaps = body_unit_gaps


# ---------------------------------------------------------------------------
# Alternation-rhythm fit
# ---------------------------------------------------------------------------
def _phase_assignment(final_half: Half, n_strokes: int) -> list[Half]:
    """Alternating half per stroke for a rally whose LAST stroke is `final_half`.

    Stroke i counts back from the last (index n-1 = final_half); each step back flips halves.
    The two possible phases (final_half Top vs Bot) are exact complements of each other.
    """
    last = n_strokes - 1
    return [final_half if (last - i) % 2 == 0 else OTHER_HALF[final_half]
            for i in range(n_strokes)]


def fit_alternation(guesses: list[Half | None]) -> Half | None:
    """Fitted final-stroke half from the two alternating phases, or None on a tie.

    Score each phase (final stroke Top, or final stroke Bot) by the count of non-None per-stroke
    guesses it matches; the higher-scoring phase names the final-contact striker. An equal score
    is a genuine tie (no phase resolved) -> None.
    """
    phase_score = {
        final_half: sum(1 for guess, assigned in zip(guesses, _phase_assignment(final_half, len(guesses)))
                        if guess is not None and guess == assigned)
        for final_half in (Half.TOP, Half.BOT)
    }
    if phase_score[Half.TOP] == phase_score[Half.BOT]:
        return None
    return Half.TOP if phase_score[Half.TOP] > phase_score[Half.BOT] else Half.BOT


def next_server_half(striker_halves: list[Half | None], n_strokes: list[int]) -> list[Half | None]:
    """Winner half per rally from winner-serves-next; None where no next serve is attributable.

    Winner(n) = the attributed half of rally n+1's FIRST stroke, read off the fit already resolved
    for rally n+1 (its final-stroke half back-propagated through the alternation to stroke 0). None
    for the last rally (no next serve) or where rally n+1's fit tied. Game/set boundaries need no
    special case: the game winner takes the last point AND serves first next game, so winner(n) ==
    server(n+1) across a boundary too.
    """
    fitted_first = [_phase_assignment(half, n)[0] if half is not None else None
                    for half, n in zip(striker_halves, n_strokes)]
    return [fitted_first[n + 1] if n + 1 < len(striker_halves) else None
            for n in range(len(striker_halves))]


# ---------------------------------------------------------------------------
# Landing search window
# ---------------------------------------------------------------------------
def _gap_after_top_exit(final_contact: int, run_start: int, track: np.ndarray) -> bool:
    """Did the last visible sample before an invisible run sit at the frame's TOP edge?

    The invisible run begins at absolute frame ``final_contact + 1 + run_start``, so the sample
    just before it is ``final_contact + run_start`` (the contact frame itself when run_start == 0).
    A visible sample there with image-y within TOP_EDGE_FRAC of 0 means the shuttle left the frame
    upward (a lob) and will fall back into view; a non-visible or non-top sample reads False.
    """
    last_vis = final_contact + run_start
    return bool(track[last_vis, 2] == 1 and track[last_vis, 1] < TOP_EDGE_FRAC)


def window_end(
    final_contact: int, next_start: int, track: np.ndarray, dead: np.ndarray,
    sustained_loss_frames: int,
) -> int:
    """Earliest of the next rally's GT start, a sustained track loss, or replay-mask onset.

    Half-open: the window is [final_contact, window_end). Scans forward from final_contact + 1.

    A sustained-loss gap does NOT close the window when the last visible sample before it sits at
    the frame top: the shuttle lobbed out of the top of the picture and will descend back into
    view, so the search waits for the re-entry, still bounded by the next serve and the replay
    mask (and by any LATER sustained-loss gap that did not follow a top exit). This is the shipped
    window-fix behaviour (the harness's ``--window-fix``); there is no toggle here, it always
    applies.
    """
    cap = min(next_start, len(track))
    end = cap
    seg_dead = dead[final_contact + 1:cap]  # first masked frame after contact
    if seg_dead.any():
        end = min(end, final_contact + 1 + int(np.argmax(seg_dead)))
    invisible = track[final_contact + 1:cap, 2] != 1  # first sustained-loss run start
    for run_start, run_end in true_runs(invisible):
        if run_end - run_start >= sustained_loss_frames:
            if _gap_after_top_exit(final_contact, run_start, track):
                continue  # lob left the frame top; wait for the shuttle to re-enter
            end = min(end, final_contact + 1 + run_start)
            break
    return max(end, final_contact + 1)


def _at_frame_border(xy: np.ndarray) -> bool:
    """Terminal sample within 2% of any image edge (normalised coords).

    True => the descending run ran off-frame: the shuttle left the picture mid-fall rather than
    being seen to come down, so there is no trustworthy landing.
    """
    x, y = float(xy[0]), float(xy[1])
    return x < TOP_EDGE_FRAC or x > 1.0 - TOP_EDGE_FRAC or y < TOP_EDGE_FRAC or y > 1.0 - TOP_EDGE_FRAC


# ---------------------------------------------------------------------------
# Kinematic landing filter (settle cap + carry filter, both ankle-rule refined)
# ---------------------------------------------------------------------------
# Purpose: the "last descending run" a naive search keeps is often the post-rally pickup / carry /
# toss-back, whose ELEVATED terminal (shuttle in a hand, mid-air) projects past the far baseline
# through the floor homography and lands the verdict in the wrong court half. This filter excludes
# fallen/carried shuttle spans from the descending-run search KINEMATICALLY, repurposing stage 8's
# serve gate's low-displacement-over-a-window machinery (its body-height form) into two signals:
#   - SETTLE: the shuttle goes static (self-speed rolling-median <= settle_thr) for >= settle_min
#     frames and is NOT held at a wrist (the "not a trajectory inversion around the nearest wrist"
#     carve-out, proxied kinematically by wrist-proximity in body-height units). A ground settle =
#     the shuttle has fallen and come to rest, so later motion is out-of-play: the search window is
#     CAPPED at the first settle onset.
#   - CARRY: a descending run whose terminal sat sustained-close to the nearest court-scale player's
#     wrist (median wrist/box-height distance over the trailing carry_win frames <= carry_thr) is a
#     lowering-by-hand, not a fall; it is dropped from the run set.
# The last SURVIVING run wins, exactly as a naive search keeps the last run. NO geometric clamp
# lives anywhere here: every decision reads shuttle speed and shuttle-to-wrist body-height distance
# only.
class LandingKinematics(NamedTuple):
    """Per-frame kinematic signals for the landing filter, built once per video.

    :param carry_ratio: (t,) shuttle-to-nearest-court-scale-player nearer-WRIST distance divided by
        that player's bbox height (body-height units); NaN where the shuttle is invisible or no
        court-scale detection is present. The body-height form of the serve gate's proximity signal,
        with the wrist numerator matching the shipped wrists-per-box-height attribution.
    :param ankle_ratio: (t,) the same signal built from the nearer ANKLE (COCO 15/16) instead of the
        wrist, in the same body-height units. A terminal (or a settle frame) nearer an ankle than a
        wrist is the shuttle grounded by the feet, not held.
    :param speed: (t,) shuttle self-speed (norm-units/frame, compute_speed); NaN on non-visible steps.
    """

    carry_ratio: np.ndarray
    ankle_ratio: np.ndarray
    speed: np.ndarray


class LandingFilterOptions(NamedTuple):
    """One landing-filter setting: the settle cap + carry filter kinematics.

    :param settle_win: rolling-median window (frames) for the shuttle self-speed static test.
    :param settle_thr: static speed threshold (norm-units/frame); at/below this reads as static.
    :param settle_min: consecutive ground-static frames that mark a settle onset (the window cap).
    :param carry_win: trailing window (frames) the carry proximity median runs over.
    :param carry_thr: carried when median trailing wrist/box-height distance <= this (body heights).
    :param use_settle: apply the settle window cap.
    :param use_carry: drop carried runs from the run set.
    :param null_if_all_carried: when every run is carried, null the landing (True) rather than fall
        back to the last run (False). False is the shipped default (keep-last-drop): nulling loses
        rallies the measurement showed were recoverable by keeping the last surviving run anyway.
    :param use_ankle_rule: a shuttle nearer a player's ANKLE than any player's wrist reads as
        grounded, not hand-held. Refines both the carry filter (keeps such a terminal as a landing)
        and the settle cap's held carve-out (does not veto such a frame). True is the shipped
        default: without it, a standing player over a genuinely fallen shuttle never lets the
        settle cap fire (measured on the pilot's rally 90).
    """

    settle_win: int
    settle_thr: float
    settle_min: int
    carry_win: int
    carry_thr: float
    use_settle: bool = True
    use_carry: bool = True
    null_if_all_carried: bool = False
    use_ankle_rule: bool = True


def convert_landing_options(opts: LandingFilterOptions, fps: float) -> LandingFilterOptions:
    """Convert base-30 landing options once; returned fields are final fps values."""
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError(f'fps must be positive and finite, got {fps!r}')
    time = lambda value: max(1, math.floor(value * fps / 30.0 + 0.5))
    return opts._replace(
        settle_win=time(opts.settle_win),
        settle_thr=opts.settle_thr * 30.0 / fps,
        settle_min=time(opts.settle_min),
        carry_win=time(opts.carry_win),
    )


def build_landing_kinematics(
    track: np.ndarray, bboxes: np.ndarray, scores: np.ndarray, kps: np.ndarray,
    court_box: CourtBox, resolution: tuple[float, float],
) -> LandingKinematics:
    """Per-frame carry proximity (wrist / box-height) and shuttle self-speed for the landing filter.

    Mirrors a single nearest-court-scale-player loop, but reads the nearer WRIST (COCO 9/10)
    rather than the bbox centre for the numerator, so the proximity signal is in the same
    body-height units the attribution arm uses. NaN where the shuttle is invisible or no
    court-scale detection is present.

    :param resolution: (width, height) the shuttle xy and the wrist/ankle pixels normalise by.
    """
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
        # Same machinery on the nearer ANKLE. The two minima are taken independently, so the
        # nearest ankle and nearest wrist may in principle come from different players; at a
        # terminal or a ground settle the shuttle is beside one player, so both minima are that
        # player's and the per-candidate box-height normaliser cancels in the ankle-vs-wrist test.
        ankles = kps[frame, slots][:, (ANKLE_L, ANKLE_R), :]  # (k candidates, 2 ankles, xy) px
        ankle_px = np.hypot(ankles[..., 0] - shuttle_x, ankles[..., 1] - shuttle_y).min(axis=1)
        ankle_ratio[frame] = float((ankle_px / (y2 - y1)).min())
    return LandingKinematics(carry_ratio=carry_ratio, ankle_ratio=ankle_ratio, speed=compute_speed(track))


def _settle_cap(final_contact: int, win_end: int, kin: LandingKinematics,
                opts: LandingFilterOptions) -> int:
    """First ground-settle onset in the window, or win_end when none.

    Ground-static = the shuttle self-speed rolling-median is <= settle_thr AND the shuttle is NOT
    held at a wrist (carry_ratio <= carry_thr, the carve-out that keeps a hand-held pause from
    reading as a floor rest). The cap is the onset frame of the first run of >= settle_min such
    frames: the moment the shuttle came to rest, after which all motion is out-of-play.
    """
    speed_seg = kin.speed[final_contact:win_end]
    # Unseen frames stay out of the median (nanmedian) but take their visible
    # neighbours' verdict; an all-unseen window reads not-static (NaN <= thr is False).
    static = rolling_nanmedian(speed_seg, opts.settle_win) <= opts.settle_thr
    held = np.nan_to_num(kin.carry_ratio[final_contact:win_end], nan=np.inf) <= opts.carry_thr
    if opts.use_ankle_rule:
        # Ankle refinement: a static frame whose shuttle is nearer an ankle than a wrist is resting
        # on the ground by the feet, not paused in a hand, so the held carve-out must not veto it. A
        # standing player sits ~0.5 box-heights from a grounded shuttle's nearest wrist (reads as
        # held) but nearer its ankle; without this the settle cap never fires on such a rest. NaN in
        # either ratio -> the comparison is False -> held unchanged.
        seg_carry = kin.carry_ratio[final_contact:win_end]
        seg_ankle = kin.ankle_ratio[final_contact:win_end]
        held = held & ~(seg_ankle < seg_carry)
    ground_static = static & ~held
    run = 0
    for offset, is_static in enumerate(ground_static):
        run = run + 1 if is_static else 0
        if run >= opts.settle_min:
            return final_contact + offset - opts.settle_min + 1  # run onset
    return win_end


def _carried_terminal(terminal: int, kin: LandingKinematics, opts: LandingFilterOptions) -> bool:
    """Did the shuttle sit sustained-close to a wrist over the carry_win frames ending at terminal?

    With use_ankle_rule set, a carried verdict is overturned when the terminal sample itself sits
    nearer a player's ANKLE than any player's wrist: that is the shuttle grounded by the feet, a
    landing rather than a lowering-by-hand. Association mirrors carry_ratio (nearest ankle vs nearest
    wrist over the court-scale players, body-height units); see build_landing_kinematics.
    """
    lo = max(0, terminal - opts.carry_win + 1)
    window = kin.carry_ratio[lo:terminal + 1]
    finite = window[np.isfinite(window)]
    carried = len(finite) > 0 and bool(np.median(finite) <= opts.carry_thr)
    if carried and opts.use_ankle_rule:
        ankle_r = kin.ankle_ratio[terminal]
        wrist_r = kin.carry_ratio[terminal]
        if np.isfinite(ankle_r) and np.isfinite(wrist_r) and ankle_r < wrist_r:
            return False  # nearer an ankle than a wrist -> grounded, keep the run as a landing
    return carried


def filtered_descending_landing(
    final_contact: int, win_end: int, track: np.ndarray,
    kin: LandingKinematics, opts: LandingFilterOptions, min_descend_samples: int = MIN_DESCEND_SAMPLES,
) -> tuple[int, np.ndarray] | None:
    """The landing: the last descending run surviving the settle cap and carry filter.

    Descending = the shuttle physically falling = image-y INCREASING. Runs are >= MIN_DESCEND_SAMPLES
    consecutive VISIBLE samples, strictly image-y-increasing. The search window is first capped at
    the settle onset, then carried runs are dropped. Returns (landing_frame, [x, y]) (frame,
    normalised xy) or None when nothing survives.
    """
    cap = _settle_cap(final_contact, win_end, kin, opts) if opts.use_settle else win_end
    search_end = max(cap, final_contact + 1)
    frames = np.arange(final_contact, search_end)
    visible = frames[track[frames, 2] == 1]
    if len(visible) < min_descend_samples:
        return None
    ys = track[visible, 1]  # normalised image-y, ascending frame order
    terminals: list[int] = []
    run_start = 0
    for idx in range(1, len(visible) + 1):
        # a run breaks when the next step is not strictly increasing (falling), or at the end
        if idx == len(visible) or ys[idx] <= ys[idx - 1]:
            if idx - run_start >= min_descend_samples:
                terminals.append(int(visible[idx - 1]))
            run_start = idx
    if not terminals:
        return None
    if opts.use_carry:
        survivors = [t for t in terminals if not _carried_terminal(t, kin, opts)]
        if not survivors:
            if opts.null_if_all_carried:
                return None
            survivors = terminals
        terminals = survivors
    landing_frame = terminals[-1]  # FINAL surviving run wins
    return landing_frame, track[landing_frame, :2].copy()


# ---------------------------------------------------------------------------
# Net rule, in/out verdict, margins
# ---------------------------------------------------------------------------
def is_net_ender(
    final_contact: int, win_end: int, track: np.ndarray,
    striker_half: Half, net_band: tuple[float, float], resolution: tuple[float, float],
) -> bool:
    """Flight never crosses the net line's image y AND dies (terminal sample) in the net band.

    Image-space, per the only concrete net-band numbers available (the homography net band).
    A Top striker (small image-y) must never send the shuttle past band_hi (to the receiver's
    side); a Bot striker never below band_lo. The terminal sample is the last visible sample in
    the window. True => 'died at the net' => striker lost.

    :param resolution: (width, height) the shuttle image-y (normalised) scales to pixels by.
    """
    frames = np.arange(final_contact, win_end)
    visible = frames[track[frames, 2] == 1]
    if len(visible) < 2:
        return False
    _, height = resolution
    ys = track[visible, 1] * height  # image-y pixels
    band_lo, band_hi = net_band
    terminal_y = float(ys[-1])
    if striker_half == Half.TOP:
        never_crossed = bool(np.all(ys <= band_hi))
    else:
        never_crossed = bool(np.all(ys >= band_lo))
    dies_at_net = band_lo <= terminal_y <= band_hi
    return never_crossed and dies_at_net


def inout_verdict(landing_norm: np.ndarray, receiver_half: Half, margin_m: float) -> Verdict | None:
    """WON / LOST / None (ambiguous margin) for a landing vs the receiver's singles half.

    Receiver singles half-court rectangle in normalised court coords: x in the singles inset,
    y in the receiver's half ([0.5, 1] when receiver is Bot, [0, 0.5] when Top). Clearances to
    the four boundary lines are converted to metres (x*6.10, y*13.40). Inside with every
    clearance > M => won; outside (point-to-rectangle distance) by > M => lost; else null.
    """
    x, y = float(landing_norm[0]), float(landing_norm[1])
    y_lo, y_hi = (NET_COURT_Y, 1.0) if receiver_half == Half.BOT else (0.0, NET_COURT_Y)

    clear_xlo = (x - SINGLES_X_LO) * COURT_WIDTH_M
    clear_xhi = (SINGLES_X_HI - x) * COURT_WIDTH_M
    clear_ylo = (y - y_lo) * COURT_LENGTH_M
    clear_yhi = (y_hi - y) * COURT_LENGTH_M
    if min(clear_xlo, clear_xhi, clear_ylo, clear_yhi) > margin_m:
        return Verdict.WON

    out_x = max(0.0, (SINGLES_X_LO - x), (x - SINGLES_X_HI)) * COURT_WIDTH_M
    out_y = max(0.0, (y_lo - y), (y - y_hi)) * COURT_LENGTH_M
    if float(np.hypot(out_x, out_y)) > margin_m:
        return Verdict.LOST
    return None  # within +/-M of a boundary line


class LandingMargins(NamedTuple):
    """Signed court-metre clearances for one landing vs the receiver's singles half.

    :param margin_m: signed metres to the NEAREST boundary line overall; + inside the receiver
        half, - outside (point-to-rectangle distance).
    :param net_clear_m: unsigned metres from the landing to the net (halfway) line.
    :param line_clear_m: unsigned metres to the nearest of the two sidelines and the receiver's
        baseline (the in/out lines, net excluded).
    """

    margin_m: float
    net_clear_m: float
    line_clear_m: float


def landing_margins(landing_norm: tuple[float, float], receiver_half: Half) -> LandingMargins:
    """Court-metre clearances for a landing, reusing inout_verdict's boundary geometry."""
    x, y = float(landing_norm[0]), float(landing_norm[1])
    y_lo, y_hi = (NET_COURT_Y, 1.0) if receiver_half == Half.BOT else (0.0, NET_COURT_Y)
    baseline_y = y_hi if receiver_half == Half.BOT else y_lo  # the non-net y edge = receiver baseline

    net_clear_m = abs(y - NET_COURT_Y) * COURT_LENGTH_M
    line_clear_m = min(abs(x - SINGLES_X_LO) * COURT_WIDTH_M,
                       abs(SINGLES_X_HI - x) * COURT_WIDTH_M,
                       abs(y - baseline_y) * COURT_LENGTH_M)

    clear_xlo = (x - SINGLES_X_LO) * COURT_WIDTH_M
    clear_xhi = (SINGLES_X_HI - x) * COURT_WIDTH_M
    clear_ylo = (y - y_lo) * COURT_LENGTH_M
    clear_yhi = (y_hi - y) * COURT_LENGTH_M
    inside = min(clear_xlo, clear_xhi, clear_ylo, clear_yhi)
    if inside > 0:
        margin_m = float(inside)
    else:
        out_x = max(0.0, SINGLES_X_LO - x, x - SINGLES_X_HI) * COURT_WIDTH_M
        out_y = max(0.0, y_lo - y, y - y_hi) * COURT_LENGTH_M
        margin_m = -float(np.hypot(out_x, out_y))
    return LandingMargins(margin_m=margin_m, net_clear_m=net_clear_m, line_clear_m=line_clear_m)


def corner_error_band_m(vid: int, homo_df: pd.DataFrame, court_info: dict, err_px: float) -> float:
    """Corner error (refpx) propagated to court metres at the recorded-corner (line) locations.

    Shifts each recorded corner by err_px along +/-x and +/-y, re-projects it through the SAME
    homography, and measures how far the projected point moves in court metres. The median over the
    four corners x four directions is the band: how many metres of landing uncertainty a corner
    error of err_px buys at a court line. A forward-projection proxy for the homography's own corner
    uncertainty; both scale as err_px x the local metres-per-refpx.

    :param vid: the ShuttleSet video id, to key `homo_df`.
    :param homo_df: the homography.csv frame, indexed by id.
    :param court_info: this video's court info dict (carries `'H'`).
    :param err_px: assumed corner-marking error, in the recorded homography's own pixel space.
    """
    corners = get_corner_camera(homo_df.loc[vid])  # (2, 4) refpx, already at HOMOGRAPHY_RESOLUTION
    base = project_pixels_to_court(corners, HOMOGRAPHY_RESOLUTION, court_info)
    displacements: list[float] = []
    for corner in range(4):
        for dx, dy in ((err_px, 0.0), (-err_px, 0.0), (0.0, err_px), (0.0, -err_px)):
            shifted = corners.copy()
            shifted[0, corner] += dx
            shifted[1, corner] += dy
            proj = project_pixels_to_court(shifted, HOMOGRAPHY_RESOLUTION, court_info)
            move = proj[:, corner] - base[:, corner]
            displacements.append(float(np.hypot(move[0] * COURT_WIDTH_M, move[1] * COURT_LENGTH_M)))
    return float(np.median(displacements))


# ---------------------------------------------------------------------------
# Landing pick + verdict assembly
# ---------------------------------------------------------------------------
class Landing(NamedTuple):
    """One rally's picked shuttle landing: frame, projected court position, and quality flags.

    :param frame: whole-video frame index of the landing.
    :param norm: normalised court xy (doubles outline = unit square).
    :param half: court half the landing's projected position falls in (court-space y against the
        net line at 0.5): a side call only, never an in/out call — the singles boundary plays no
        part in it.
    :param at_border: True when the picked terminal sat within 2% of any image edge (the shuttle
        left frame mid-fall, not a seen landing).
    :param masked: True when a replay-masked frame sits between contact and the landing.
    :param net_ender: True when the rally's flight never crossed the net and died in the net band.
    """

    frame: int
    norm: tuple[float, float]
    half: Half
    at_border: bool
    masked: bool
    net_ender: bool


def pick_landing(
    final_contact: int, next_start: int, track: np.ndarray, dead: np.ndarray,
    kin: LandingKinematics, opts: LandingFilterOptions, striker_half: Half,
    net_band: tuple[float, float], resolution: tuple[float, float], court_info: dict,
    constants: FpsConstants, fps: float,
) -> Landing | None:
    """The picked landing for one rally: the filtered terminal, projected to court space, with
    quality flags. None when nothing survives the landing filter within the window.

    ``fps`` must be the rate for which ``constants`` was resolved; it is used only to convert
    landing options once.
    """
    win_end = window_end(final_contact, next_start, track, dead, constants.sustained_loss_frames)
    landing = filtered_descending_landing(
        final_contact, win_end, track, kin, convert_landing_options(opts, fps),
        constants.min_descend_samples,
    )
    if landing is None:
        return None
    landing_frame, landing_xy = landing
    px = np.array([[landing_xy[0] * resolution[0]], [landing_xy[1] * resolution[1]]])
    proj = project_pixels_to_court(px, resolution, court_info)
    norm = (float(proj[0, 0]), float(proj[1, 0]))
    half = Half.TOP if norm[1] < NET_COURT_Y else Half.BOT
    return Landing(
        frame=landing_frame, norm=norm, half=half,
        at_border=_at_frame_border(landing_xy),
        masked=bool(dead[final_contact:landing_frame + 1].any()),
        net_ender=is_net_ender(final_contact, win_end, track, striker_half, net_band, resolution),
    )


def geometric_verdict(
    striker_half: Half, landing: Landing | None, best_guess: bool = False,
) -> tuple[Verdict | None, Half | None, VerdictSource]:
    """(verdict, winner_half, source) from the landing geometry at M=0: net rule, else in/out.

    best_guess=False (the confident path): None where no confident call is available (off-frame /
    masked / exactly on a line). best_guess=True (the shipped next-server fallback, for a rally
    with no attributable next serve): the raw landing's side membership always yields won/lost, so
    the only blank is a rally with no landing at all.
    """
    receiver = OTHER_HALF[striker_half]
    if landing is not None and landing.net_ender:
        return Verdict.LOST, receiver, VerdictSource.NET_RULE
    if landing is None:
        return None, None, VerdictSource.LANDING_GEOMETRY
    if best_guess:
        # Which side of the receiver singles half does the raw terminal fall on? Never None.
        x, y = landing.norm
        y_lo, y_hi = (NET_COURT_Y, 1.0) if receiver == Half.BOT else (0.0, NET_COURT_Y)
        inside = (SINGLES_X_LO <= x <= SINGLES_X_HI) and (y_lo <= y <= y_hi)
        winner = striker_half if inside else receiver
        return (Verdict.WON if inside else Verdict.LOST), winner, VerdictSource.LANDING_GEOMETRY
    # Confident path: off-frame / masked / on-line => no call.
    if landing.at_border or landing.masked:
        return None, None, VerdictSource.LANDING_GEOMETRY
    verdict = inout_verdict(np.array(landing.norm), receiver, 0.0)
    if verdict is None:
        return None, None, VerdictSource.LANDING_GEOMETRY
    winner = striker_half if verdict == Verdict.WON else receiver
    return verdict, winner, VerdictSource.LANDING_GEOMETRY


class VerdictRow(NamedTuple):
    """One rally's verdict, in the production schema. No GT columns: a caller scoring against
    ground truth (the pin driver, or a future eval script) joins those on separately.

    The predicted winner half is not stored directly: with only two halves, it is exactly
    `striker_half` when `verdict is Verdict.WON` and `OTHER_HALF[striker_half]` when it is
    `Verdict.LOST`. A `None` verdict has no winner to derive; check for it first.
    """

    rally_id: int
    striker_half: Half
    verdict: Verdict | None
    verdict_source: VerdictSource | None
    margin_m: float | None
    within_line_margin: bool
    within_net_margin: bool


def rally_verdict(
    rally_id: int, striker_half: Half, next_server: Half | None, landing: Landing | None,
    band_m: float,
) -> VerdictRow:
    """One rally's verdict row: the shipped next-server-first call, geometry as a fallback.

    Rally n's winner is rally n+1's fitted first-stroke half whenever one is attributable
    (winner-serves-next), sidestepping the landing estimate for the winner call entirely — a
    next-server row never checks `landing.net_ender`. Only when no next serve is attributable
    does the call fall back to the landing's best-guess court-half membership (the ported
    best_guess=True semantics: the raw terminal's side always yields a call, so verdict is blank
    only when there is no landing at all). Margins and band flags always read the landing
    geometry, diagnostic even on a next-server row.
    """
    if next_server is not None:
        winner = next_server
        verdict = Verdict.WON if winner == striker_half else Verdict.LOST
        source = VerdictSource.NEXT_SERVER
    else:
        verdict, _winner, source = geometric_verdict(striker_half, landing, best_guess=True)

    margin_m = None
    within_line = within_net = False
    if landing is not None:
        margins = landing_margins(landing.norm, OTHER_HALF[striker_half])
        margin_m = margins.margin_m
        within_line = margins.line_clear_m < band_m
        within_net = margins.net_clear_m < band_m

    return VerdictRow(
        rally_id=rally_id, striker_half=striker_half, verdict=verdict,
        verdict_source=source if verdict is not None else None,
        margin_m=margin_m, within_line_margin=within_line, within_net_margin=within_net,
    )


# ---------------------------------------------------------------------------
# Hit height (ShuttleSet coding; decoupled from the verdict path)
# ---------------------------------------------------------------------------
def hit_height(
    track: np.ndarray, contact_frame: int, net_band: tuple[float, float],
    resolution: tuple[float, float],
) -> int:
    """ShuttleSet hit_height coding for one contact: 1 above the net-band centre, 2 at/below it.

    Reads the shuttle's own image-y at the contact frame against the net band's CENTRE line
    (resolution-scaled pixels): smaller image-y (higher in frame) than the centre is 1; equal to
    or below the centre is 2 (the centre itself resolves to 2, mirroring is_net_ender's >=
    convention for "below"). Fails loud when the shuttle is not visible at the contact frame:
    hit_height needs an actual detected position, not a guess.

    :param net_band: image-y net band (pixels, `resolution`-scaled), the striker/receiver
        half-split band.
    :param resolution: (width, height) the shuttle xy normalises by.
    :return: 1 (above the net-band centre) or 2 (at or below it), the ShuttleSet hit_height coding.
    """
    if track[contact_frame, 2] != 1:
        raise ValueError(f'shuttle not visible at contact frame {contact_frame}: cannot read hit_height')
    _, height = resolution
    shuttle_y_px = track[contact_frame, 1] * height
    band_lo, band_hi = net_band
    centre = (band_lo + band_hi) / 2.0
    return 1 if shuttle_y_px < centre else 2


class HitHeightRow(NamedTuple):
    """One stroke's ShuttleSet-coded hit_height, keyed the same way the verdict rows are."""

    rally_id: int
    stroke_idx: int
    contact_frame: int
    hit_height: int


def build_hit_height_rows(
    contacts: list[tuple[int, int, int]], track: np.ndarray,
    net_band: tuple[float, float], resolution: tuple[float, float],
) -> list[HitHeightRow]:
    """Per-stroke hit_height rows for a flat list of (rally_id, stroke_idx, contact_frame).

    Decoupled from the verdict path: the contact frame is the only shuttle-track input this
    reads (no attribution, no landing filter), so a caller can build these independently of
    rally_verdict.
    """
    return [
        HitHeightRow(rally_id, stroke_idx, contact_frame,
                    hit_height(track, contact_frame, net_band, resolution))
        for rally_id, stroke_idx, contact_frame in contacts
    ]
