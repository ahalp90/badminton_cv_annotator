"""Point-winner verdicts (D5 chain — attribution, alternation fit, landing, verdict).

Wrist-anchored striker attribution in body-height units, an alternation-rhythm fit for the
final-contact half, a kinematic landing filter (a settle cap plus a carry filter, both refined by
an ankle rule), and a next-server winner call with a landing-geometry best-guess fallback. Promoted
from the D5 point-winner detector proven out in
local_scratch/autograder_architecture/d5_winner_retest.py and d5_landing_arms.py (measured on the
ShuttleSet sset_01 and trial videos, GT-anchored against the per-set winner labels). Only the SHIPPED
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
time; there is no established path convention yet for wiring rally-segmentation and replay-mask
outputs into a point-winner CLI, so this stays a library the caller composes over a rally list, the way the
harness's own per-rally loop does. See
the pinned D5 example under local_scratch/autograder_architecture for a runnable reproduction of
the D5 retest's arm-2 verdict CSVs from this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
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

from .types import (
    ANKLE_L,
    ANKLE_R,
    WRIST_L,
    WRIST_R,
    StickyResult,
    compute_speed,
    rolling_nanmedian,
    true_runs,
)
from .fps_constants import FpsConstants, ScalingKind


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
SHUTTLESET_TO_COURTKEYNET_CORNER_ORDER = (0, 1, 3, 2)

# Image-y fraction that counts as the frame's TOP edge for the window fix (a lob that exits the
# top leaves its last visible sample this close to y=0). Also the terminal-at-border threshold
# (2% of any edge): matches the harness's single source of truth for "at the top edge".
TOP_EDGE_FRAC = 0.02

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
def attribute_half(
    frame: int, track: np.ndarray, sticky: StickyResult, bboxes: np.ndarray, net_band: tuple[float, float],
) -> Half | None:
    """Court half of the sticky pick nearest to the shuttle at `frame`, or None.

    The pick and its cached nearest-wrist body-unit distance come from the sticky tracker. The
    wrist measurement point is unchanged. The picked bbox's bottom y (pixels) is tested against
    the net band.
    """
    if track[frame, 2] != 1:
        return None
    distances = sticky.distances_per_slot[frame]
    if not np.isfinite(distances).any():
        return None
    # A finite value beats +inf/NaN sentinels; an exact two-half tie resolves to Top
    # (nanargmin takes the first index).
    half = int(np.nanargmin(distances))
    slot = sticky.picks[frame, half]
    foot_y = float(bboxes[frame, slot][3])
    band_lo, band_hi = net_band
    if foot_y < band_lo:
        return Half.TOP
    if foot_y > band_hi:
        return Half.BOT
    return None  # inside the net band


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
@dataclass(frozen=True)
class LandingWindow:
    """The safe half-open landing endpoint and the guards that closed it."""

    end_frame: int
    closure_reasons: tuple[str, ...]


def _gap_after_top_exit(
    final_contact: int, run_start: int, track: np.ndarray,
    shuttle_hallucination_mask: np.ndarray | None = None,
) -> bool:
    """Did the last visible sample before an invisible run sit at the frame's TOP edge?

    The invisible run begins at absolute frame ``final_contact + 1 + run_start``, so the sample
    just before it is ``final_contact + run_start`` (the contact frame itself when run_start == 0).
    A visible sample there with image-y within TOP_EDGE_FRAC of 0 means the shuttle left the frame
    upward (a lob) and will fall back into view; a non-visible or non-top sample reads False.
    """
    last_vis = final_contact + run_start
    visible = track[last_vis, 2] == 1
    if shuttle_hallucination_mask is not None:
        visible = visible and not shuttle_hallucination_mask[last_vis]
    return bool(visible and track[last_vis, 1] < TOP_EDGE_FRAC)


def landing_window(
    final_contact: int, next_start: int, track: np.ndarray, dead: np.ndarray,
    sustained_loss_frames: int, shuttle_hallucination_mask: np.ndarray | None = None,
) -> LandingWindow:
    """Find the safe landing endpoint and all guards tied at that endpoint.

    Half-open: the window is [final_contact, window_end). Scans forward from final_contact + 1.

    A sustained-loss gap does NOT close the window when the last visible sample before it sits at
    the frame top: the shuttle lobbed out of the top of the picture and will descend back into
    view, so the search waits for the re-entry, still bounded by the next serve and the replay
    mask (and by any LATER sustained-loss gap that did not follow a top exit). This is the shipped
    window-fix behaviour (the harness's ``--window-fix``); there is no toggle here, it always
    applies.
    """
    if shuttle_hallucination_mask is not None and len(shuttle_hallucination_mask) != len(track):
        raise ValueError('shuttle_hallucination_mask length must match track length')
    video_end = len(track)
    boundaries: dict[str, int] = {'video_end': video_end}
    if next_start < video_end:
        boundaries['next_rally'] = next_start

    scan_start = final_contact + 1
    scan_end = min(next_start, video_end)
    seg_dead = dead[scan_start:scan_end]  # first masked frame after contact
    if seg_dead.any():
        boundaries['definitive_exclusion'] = scan_start + int(np.argmax(seg_dead))
    effective_visibility = track[:, 2] == 1
    if shuttle_hallucination_mask is not None:
        effective_visibility &= ~shuttle_hallucination_mask
    invisible = ~effective_visibility[scan_start:scan_end]  # first sustained-loss run start
    for run_start, run_end in true_runs(invisible):
        if run_end - run_start >= sustained_loss_frames:
            if _gap_after_top_exit(
                final_contact, run_start, track, shuttle_hallucination_mask,
            ):
                continue  # lob left the frame top; wait for the shuttle to re-enter
            boundaries['sustained_loss'] = scan_start + run_start
            break
    safe_end = max(final_contact + 1, min(boundaries.values()))
    reason_order = ('next_rally', 'definitive_exclusion', 'sustained_loss', 'video_end')
    reasons = tuple(
        reason for reason in reason_order
        if boundaries.get(reason) == safe_end
    )
    return LandingWindow(end_frame=safe_end, closure_reasons=reasons)


def window_end(
    final_contact: int, next_start: int, track: np.ndarray, dead: np.ndarray,
    sustained_loss_frames: int, shuttle_hallucination_mask: np.ndarray | None = None,
) -> int:
    """Return the existing safe landing endpoint."""
    return landing_window(
        final_contact, next_start, track, dead, sustained_loss_frames,
        shuttle_hallucination_mask,
    ).end_frame


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
# fallen/carried shuttle spans from the descending-run search KINEMATICALLY, repurposing rally segmentation's
# serve gate's low-displacement-over-a-window machinery (its body-height form) into two signals:
#   - SETTLE: the shuttle goes static (self-speed rolling-median <= settle_thr) for >= settle_min
#     frames and is NOT held at a wrist (the "not a trajectory inversion around the nearest wrist"
#     carve-out, proxied kinematically by wrist-proximity in body-height units). A ground settle =
#     the shuttle has fallen and come to rest, so later motion is out-of-play: the search window is
#     CAPPED at the first settle onset.
#   - CARRY: a descending run whose terminal sat sustained-close to the nearest sticky-selected
#     player's wrist (median wrist/box-height distance over the trailing carry_win frames <=
#     carry_thr) is a lowering-by-hand, not a fall; it is dropped from the run set.
# The last SURVIVING run wins, exactly as a naive search keeps the last run. NO geometric clamp
# lives anywhere here: every decision reads shuttle speed and shuttle-to-wrist body-height distance
# only.
class LandingKinematics(NamedTuple):
    """Per-frame kinematic signals for the landing filter, built once per video.

    :param carry_ratio: (t,) shuttle-to-nearest sticky-selected player's nearer-WRIST distance
        divided by that player's bbox height (body-height units); NaN where the shuttle is invisible
        or no sticky pick has a finite height. The body-height form of the serve gate's proximity
        signal, with the wrist numerator matching the shipped wrists-per-box-height attribution.
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
        settle cap fire (measured on sset_01's rally 90).
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


SHIPPED_LANDING_FILTER_OPTIONS = LandingFilterOptions(
    settle_win=7,
    settle_thr=0.004,
    settle_min=5,
    carry_win=7,
    carry_thr=0.75,
)


def convert_landing_options(opts: LandingFilterOptions, fps: float) -> LandingFilterOptions:
    """Convert base-30 landing options once; returned fields are final fps values."""
    def frame_count(value: int) -> int:
        return int(ScalingKind.FRAME_COUNT.scale(value, fps))

    return opts._replace(
        settle_win=frame_count(opts.settle_win),
        settle_thr=float(ScalingKind.PER_FRAME_SPEED.scale(opts.settle_thr, fps)),
        settle_min=frame_count(opts.settle_min),
        carry_win=frame_count(opts.carry_win),
    )


def build_landing_kinematics(
    track: np.ndarray, sticky: StickyResult, kps: np.ndarray, resolution: tuple[float, float],
) -> LandingKinematics:
    """Per-frame carry proximity (wrist / box-height) and shuttle self-speed for the landing filter.

    Reads the sticky pick for each court half, then uses its nearer WRIST (COCO 9/10) rather than
    the bbox centre for the numerator. NaN where the shuttle is invisible or no sticky pick has a
    finite current-frame height.

    :param resolution: (width, height) the shuttle xy and the wrist/ankle pixels normalise by.
    """
    width, height = resolution
    n_frames = len(track)
    carry_ratio = np.full(n_frames, np.nan)
    ankle_ratio = np.full(n_frames, np.nan)  # nearer-ANKLE / box-height; mirrors carry_ratio exactly
    for frame in np.flatnonzero(track[:, 2] == 1):
        shuttle_x, shuttle_y = track[frame, 0] * width, track[frame, 1] * height
        wrist_ratios = []
        ankle_ratios = []
        for half in range(2):
            slot = sticky.picks[frame, half]
            box_height = sticky.bbox_height[frame, half]
            if slot < 0 or not np.isfinite(box_height):
                continue
            wrists = kps[frame, slot, (WRIST_L, WRIST_R), :]
            wrist_px = np.hypot(wrists[:, 0] - shuttle_x, wrists[:, 1] - shuttle_y).min()
            # NaN unreachable in practice (pose model always emits 17 keypoints); the 0-substitute
            # keeps the min() reductions below deterministic if that ever changes.
            wrist_ratio = wrist_px / box_height
            wrist_ratios.append(0.0 if np.isnan(wrist_ratio) else wrist_ratio)
        # Same machinery on the nearer ANKLE. The two minima are taken independently, so the
        # nearest ankle and nearest wrist may in principle come from different players; at a
        # terminal or a ground settle the shuttle is beside one player, so both minima are that
        # player's and the per-candidate box-height normaliser cancels in the ankle-vs-wrist test.
            ankles = kps[frame, slot, (ANKLE_L, ANKLE_R), :]
            ankle_px = np.hypot(ankles[:, 0] - shuttle_x, ankles[:, 1] - shuttle_y).min()
            ankle_ratio_val = ankle_px / box_height
            ankle_ratios.append(0.0 if np.isnan(ankle_ratio_val) else ankle_ratio_val)
        if wrist_ratios:
            carry_ratio[frame] = float(min(wrist_ratios))
            ankle_ratio[frame] = float(min(ankle_ratios))
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


def _carried_terminal(
    final_contact: int, terminal: int, kin: LandingKinematics, opts: LandingFilterOptions,
) -> bool:
    """Did the shuttle sit sustained-close to a wrist over the carry_win frames ending at terminal?

    With use_ankle_rule set, a carried verdict is overturned when the terminal sample itself sits
    nearer a player's ANKLE than any player's wrist: that is the shuttle grounded by the feet, a
    landing rather than a lowering-by-hand. Association mirrors carry_ratio (nearest ankle vs nearest
    wrist over the sticky-selected players, body-height units); see build_landing_kinematics.
    """
    lo = max(final_contact, terminal - opts.carry_win + 1)
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
    kin: LandingKinematics, opts: LandingFilterOptions, min_descend_samples: int,
    shuttle_hallucination_mask: np.ndarray | None = None,
    rejected_intervals: list[tuple[int, int]] | None = None,
) -> tuple[int, np.ndarray] | None:
    """The landing: the last descending run surviving the settle cap and carry filter.

    Descending = the shuttle physically falling = image-y INCREASING. Runs are
    at least ``min_descend_samples`` consecutive VISIBLE samples, strictly
    image-y-increasing. The search window is first capped at the settle onset,
    then carried runs are dropped. Returns (landing_frame, [x, y]) (frame,
    normalised xy) or None when nothing survives.
    """
    cap = _settle_cap(final_contact, win_end, kin, opts) if opts.use_settle else win_end
    search_end = max(cap, final_contact + 1)
    frames = np.arange(final_contact, search_end)
    visible = frames[track[frames, 2] == 1]
    if len(visible) < min_descend_samples:
        return None
    ys = track[visible, 1]  # normalised image-y, ascending frame order
    candidates: list[tuple[int, int, int]] = []
    run_start = 0
    for idx in range(1, len(visible) + 1):
        # a run breaks when the next step is not strictly increasing (falling), or at the end
        if idx == len(visible) or ys[idx] <= ys[idx - 1]:
            if idx - run_start >= min_descend_samples:
                start_frame = int(visible[run_start])
                terminal_frame = int(visible[idx - 1])
                candidates.append((start_frame, terminal_frame, terminal_frame + 1))
            run_start = idx
    if not candidates:
        return None

    if shuttle_hallucination_mask is not None:
        if len(shuttle_hallucination_mask) != len(track):
            raise ValueError('shuttle_hallucination_mask length must match track length')
        surviving_candidates: list[tuple[int, int, int]] = []
        for candidate in candidates:
            start_frame, _terminal_frame, end_frame = candidate
            if shuttle_hallucination_mask[start_frame:end_frame].any():
                if rejected_intervals is not None:
                    rejected_intervals.append((start_frame, end_frame))
            else:
                surviving_candidates.append(candidate)
        candidates = surviving_candidates
    if not candidates:
        return None

    terminals = [terminal_frame for _start_frame, terminal_frame, _end_frame in candidates]
    if opts.use_carry:
        survivors = [
            t for t in terminals if not _carried_terminal(final_contact, t, kin, opts)
        ]
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
    signed_margin = landing_margins(
        (float(landing_norm[0]), float(landing_norm[1])), receiver_half,
    ).margin_m
    if signed_margin > margin_m:
        return Verdict.WON
    if signed_margin < -margin_m:
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
    """Court-metre clearances for a landing against the receiver's singles half."""
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


def corner_error_band_from_corners(
    corners_refpx: np.ndarray, court_info: dict, err_px: float,
) -> float:
    """Return the median metre displacement from sixteen corner perturbations.

    ``corners_refpx`` uses the CourtKeyNet TL, TR, BR, BL order at
    :data:`HOMOGRAPHY_RESOLUTION`. The supplied ``court_info`` owns the active
    parent homography and court boundaries, so this helper is independent of
    ShuttleSet's homography table.
    """
    corners_refpx = np.asarray(corners_refpx, dtype=float)
    base = project_pixels_to_court(corners_refpx.T, HOMOGRAPHY_RESOLUTION, court_info)
    displacements: list[float] = []
    for corner in range(4):
        for dx, dy in ((err_px, 0.0), (-err_px, 0.0), (0.0, err_px), (0.0, -err_px)):
            shifted = corners_refpx.copy()
            shifted[corner, 0] += dx
            shifted[corner, 1] += dy
            projected = project_pixels_to_court(shifted.T, HOMOGRAPHY_RESOLUTION, court_info)
            move = projected[:, corner] - base[:, corner]
            displacements.append(float(np.hypot(move[0] * COURT_WIDTH_M, move[1] * COURT_LENGTH_M)))
    return float(np.median(displacements))


def corner_error_band_m(
    vid: int | str,
    homo_df: pd.DataFrame,
    court_info: dict,
    err_px: float,
) -> float:
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
    source_order = get_corner_camera(homo_df.loc[vid]).T
    corners = source_order[list(SHUTTLESET_TO_COURTKEYNET_CORNER_ORDER)]
    return corner_error_band_from_corners(corners, court_info, err_px)


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
    :param net_ender: True when the rally's flight never crossed the net and died in the net band.
    """

    frame: int
    norm: tuple[float, float]
    half: Half
    at_border: bool
    net_ender: bool


def pick_landing_to_end(
    final_contact: int, end_frame: int, track: np.ndarray,
    kin: LandingKinematics, opts: LandingFilterOptions, striker_half: Half,
    net_band: tuple[float, float], resolution: tuple[float, float], court_info: dict,
    constants: FpsConstants, fps: float,
    shuttle_hallucination_mask: np.ndarray | None = None,
    rejected_intervals: list[tuple[int, int]] | None = None,
) -> Landing | None:
    """Pick a landing in an explicit half-open window endpoint."""
    landing = filtered_descending_landing(
        final_contact, end_frame, track, kin, convert_landing_options(opts, fps),
        constants.min_descend_samples, shuttle_hallucination_mask, rejected_intervals,
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
        net_ender=is_net_ender(final_contact, end_frame, track, striker_half, net_band, resolution),
    )


def geometric_verdict(
    striker_half: Half, landing: Landing | None, best_guess: bool = False,
) -> tuple[Verdict | None, Half | None, VerdictSource]:
    """(verdict, winner_half, source) from the landing geometry at M=0: net rule, else in/out.

    best_guess=False (the confident path): None where no confident call is available (off-frame /
    exactly on a line). best_guess=True (the shipped next-server fallback, for a rally
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
    # Confident path: off-frame or on-line => no call.
    if landing.at_border:
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


class GeometricVerdictRow(NamedTuple):
    """The geometric winner arm and its consistency check for one rally."""

    rally_id: int
    geometric_verdict: Verdict | None
    geometric_winner: Half | None
    agreement: bool | None
    window_closed_by_mask: bool


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
