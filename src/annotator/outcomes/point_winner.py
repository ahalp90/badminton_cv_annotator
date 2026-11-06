"""Estimate rally outcomes from contact sides, shuttle flight and court geometry.

The next server determines the winner when known. Otherwise, a filtered landing
provides a geometric estimate. Outcomes use the already-fitted player order
without changing contact sides.

Landing filters reject carried shuttle motion and stop at ground settling. They
assume full-frame broadcast footage: a shuttle leaving the image's top edge may
be a lob that returns to view. Each landing search stays within one camera view.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import numpy as np

from annotator.calc_helpers import compute_speed, rolling_nanmedian, true_runs
from annotator.config import Thresholds, frames_at_fps, speed_at_fps
from annotator.types import (
    ANKLE_L,
    ANKLE_R,
    OTHER_HALF,
    WRIST_L,
    WRIST_R,
    CourtHalf,
    StickyResult,
)
from shared.court import project_pixels_to_court
from shared.court_model import COURT_LENGTH_M, COURT_WIDTH_M, SINGLES_INSET_M


class WinnerSource(StrEnum):
    """The observation or rule used to determine the winner."""

    NEXT_SERVER = 'next_server'      # winner-serves-next: rally n+1's fitted first-stroke half
    LANDING_GEOMETRY = 'landing_geometry'  # in/out of the receiver's singles half
    NET_RULE = 'net_rule'            # flight never crossed the net and died in the net band


# Singles-court geometry in the normalised court space (doubles outline -> unit square, x = court
# WIDTH 6.10 m, y = court LENGTH 13.40 m). Singles sidelines sit inset (6.10-5.18)/2 = 0.46 m each
# side; baselines are the unit-square y edges; net at y=0.5.
SINGLES_X_LO = SINGLES_INSET_M / COURT_WIDTH_M          # ~0.07541
SINGLES_X_HI = 1.0 - SINGLES_INSET_M / COURT_WIDTH_M    # ~0.92459
NET_COURT_Y = 0.5
# Image-y fraction that counts as the frame's TOP edge for the window fix (a lob that exits the
# top leaves its last visible sample this close to y=0). Also the terminal-at-border threshold
# (2% of any edge).
TOP_EDGE_FRAC = 0.02

# ---------------------------------------------------------------------------
# Landing search window
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LandingWindow:
    """The half-open landing search endpoint and reasons to stop there."""

    end_frame: int
    search_stop_reasons: tuple[str, ...]
    search_shortened_by_exclusion: bool


def gap_after_top_exit(
    final_contact: int, run_start: int, track: np.ndarray,
    shuttle_hallucination_mask: np.ndarray,
) -> bool:
    """Did the last visible sample before an invisible run sit at the frame's TOP edge?

    The invisible run begins at absolute frame ``final_contact + 1 + run_start``, so the sample
    just before it is ``final_contact + run_start`` (the contact frame itself when run_start == 0).
    A visible sample there with image-y within TOP_EDGE_FRAC of 0 means the shuttle left the frame
    upward (a lob) and will fall back into view; a non-visible or non-top sample reads False.
    """
    last_vis = final_contact + run_start
    visible = track[last_vis, 2] == 1
    visible = visible and not shuttle_hallucination_mask[last_vis]
    return bool(visible and track[last_vis, 1] < TOP_EDGE_FRAC)


def landing_window(
    final_contact: int, next_start: int, track: np.ndarray, dead: np.ndarray,
    sustained_loss_frames: int, shuttle_hallucination_mask: np.ndarray,
) -> LandingWindow:
    """Find the landing search endpoint and all reasons to stop there.

    Half-open: the window is [final_contact, window_end). Scans forward from final_contact + 1.

    A sustained-loss gap does not stop the search when the last visible sample before it sits at
    the frame top: the shuttle lobbed out of the top of the picture and will descend back into
    view, so the search waits for the re-entry, still bounded by the next serve and the replay
    mask and by any later sustained-loss gap that did not follow a top exit.
    """
    video_end = len(track)
    boundaries: dict[str, int] = {'video_end': video_end}
    if next_start < video_end:
        boundaries['next_rally'] = next_start

    scan_start = final_contact + 1
    scan_end = min(next_start, video_end)
    seg_dead = dead[scan_start:scan_end]  # first masked frame after contact
    if seg_dead.any():
        boundaries['definitive_exclusion'] = scan_start + int(np.argmax(seg_dead))
    effective_visibility = (track[:, 2] == 1) & ~shuttle_hallucination_mask
    invisible = ~effective_visibility[scan_start:scan_end]  # first sustained-loss run start
    for run_start, run_end in true_runs(invisible):
        if run_end - run_start >= sustained_loss_frames:
            if gap_after_top_exit(
                final_contact, run_start, track, shuttle_hallucination_mask,
            ):
                continue  # lob left the frame top; wait for the shuttle to re-enter
            boundaries['sustained_loss'] = scan_start + run_start
            break
    search_end = max(final_contact + 1, min(boundaries.values()))
    reason_order = ('next_rally', 'definitive_exclusion', 'sustained_loss', 'video_end')
    reasons = tuple(
        reason for reason in reason_order
        if boundaries.get(reason) == search_end
    )
    end_without_exclusion = max(final_contact + 1, min(
        end for reason, end in boundaries.items() if reason != 'definitive_exclusion'
    ))
    return LandingWindow(
        end_frame=search_end, search_stop_reasons=reasons,
        search_shortened_by_exclusion=search_end < end_without_exclusion,
    )


def at_frame_border(xy: np.ndarray) -> bool:
    """Terminal sample within 2% of any image edge (normalised coords).

    True => the descending run ran off-frame: the shuttle left the picture mid-fall rather than
    being seen to come down, so there is no trustworthy landing.
    """
    x, y = float(xy[0]), float(xy[1])
    return x < TOP_EDGE_FRAC or x > 1.0 - TOP_EDGE_FRAC or y < TOP_EDGE_FRAC or y > 1.0 - TOP_EDGE_FRAC


# ---------------------------------------------------------------------------
# Kinematic landing filter (settle cap + carry filter, both ankle-rule refined)
# ---------------------------------------------------------------------------
# A pickup or toss-back after a rally can look like a later landing. Projecting
# that airborne shuttle onto the court floor can put it in the wrong court half.
# Limit the search when the shuttle settles on the ground, and filter descending
# runs that stay close to a player's wrist. A shuttle nearer an ankle than a wrist
# is treated as grounded. These checks use motion and player distance, not court bounds.
class LandingKinematics(NamedTuple):
    """Shuttle motion and distance from players, measured at each video frame.

    :param carry_ratio: (t,) minimum wrist-to-shuttle distance in player body
        heights. Divide each selected player's distance by their box height before
        taking the minimum. NaN when the shuttle is invisible or no selected
        player has a finite height.
    :param ankle_ratio: (t,) minimum ankle-to-shuttle distance, calculated in the
        same way. The nearest ankle and wrist can belong to different players.
    :param speed: (t,) shuttle speed in normalised image coordinates per frame;
        NaN when either endpoint of a step is invisible.
    """

    carry_ratio: np.ndarray
    ankle_ratio: np.ndarray
    speed: np.ndarray


class LandingFilterOptions(NamedTuple):
    """Settings for distinguishing a landing from a pause or movement by hand.

    :param settle_win: rolling-median window for shuttle speed, in frames.
    :param settle_thr: maximum speed considered still, in normalised image units
        per frame.
    :param settle_min: consecutive still, grounded frames needed to end the search.
    :param carry_win: trailing window for median wrist distance, in frames.
    :param carry_thr: maximum median wrist distance considered carried, in body heights.
        Also used per frame to distinguish a held shuttle from a ground settle.
    """

    settle_win: int
    settle_thr: float
    settle_min: int
    carry_win: int
    carry_thr: float


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
        return frames_at_fps(value, fps)

    return opts._replace(
        settle_win=frame_count(opts.settle_win),
        settle_thr=speed_at_fps(opts.settle_thr, fps),
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
            ankles = kps[frame, slot, (ANKLE_L, ANKLE_R), :]
            ankle_px = np.hypot(ankles[:, 0] - shuttle_x, ankles[:, 1] - shuttle_y).min()
            ankle_ratio_val = ankle_px / box_height
            ankle_ratios.append(0.0 if np.isnan(ankle_ratio_val) else ankle_ratio_val)
        if wrist_ratios:
            carry_ratio[frame] = float(min(wrist_ratios))
            ankle_ratio[frame] = float(min(ankle_ratios))
    return LandingKinematics(carry_ratio=carry_ratio, ankle_ratio=ankle_ratio, speed=compute_speed(track))


def landing_settle_cap(final_contact: int, win_end: int, kin: LandingKinematics,
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
    # A shuttle nearer an ankle than a wrist is grounded, even beside a player.
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


def carried_terminal(
    final_contact: int, terminal: int, kin: LandingKinematics, opts: LandingFilterOptions,
) -> bool:
    """Did the shuttle sit sustained-close to a wrist over the carry_win frames ending at terminal?

    A carried classification is overturned when the terminal sample itself sits
    nearer a player's ANKLE than any player's wrist: that is the shuttle grounded by the feet, a
    landing rather than a lowering-by-hand. Association mirrors carry_ratio (nearest ankle vs nearest
    wrist over the sticky-selected players, body-height units); see build_landing_kinematics.
    """
    lo = max(final_contact, terminal - opts.carry_win + 1)
    window = kin.carry_ratio[lo:terminal + 1]
    finite = window[np.isfinite(window)]
    carried = len(finite) > 0 and bool(np.median(finite) <= opts.carry_thr)
    if carried:
        ankle_r = kin.ankle_ratio[terminal]
        wrist_r = kin.carry_ratio[terminal]
        if np.isfinite(ankle_r) and np.isfinite(wrist_r) and ankle_r < wrist_r:
            return False  # nearer an ankle than a wrist -> grounded, keep the run as a landing
    return carried


def filtered_descending_landing(
    final_contact: int, win_end: int, track: np.ndarray,
    kin: LandingKinematics, opts: LandingFilterOptions, min_descend_samples: int,
    shuttle_hallucination_mask: np.ndarray,
    rejected_intervals: list[tuple[int, int]] | None = None,
) -> tuple[int, np.ndarray] | None:
    """The landing: the last descending run surviving the settle cap and carry filter.

    Descending = the shuttle physically falling = image-y INCREASING. Runs are
    at least ``min_descend_samples`` consecutive VISIBLE samples, strictly
    image-y-increasing. The search window is first capped at the settle onset,
    then carried runs are dropped. Returns (landing_frame, [x, y]) (frame,
    normalised xy) or None when nothing survives.
    """
    cap = landing_settle_cap(final_contact, win_end, kin, opts)
    search_end = max(cap, final_contact + 1)
    frames = np.arange(final_contact, search_end)
    visible = frames[track[frames, 2] == 1]
    if len(visible) < min_descend_samples:
        return None
    ys = track[visible, 1]
    run_start = 0
    last_terminal = None
    last_uncarried_terminal = None
    for index in range(1, len(visible) + 1):
        if index < len(visible) and ys[index] > ys[index - 1]:
            continue
        if index - run_start >= min_descend_samples:
            start_frame = int(visible[run_start])
            terminal = int(visible[index - 1])
            if shuttle_hallucination_mask[start_frame:terminal + 1].any():
                if rejected_intervals is not None:
                    rejected_intervals.append((start_frame, terminal + 1))
            else:
                last_terminal = terminal
                if not carried_terminal(final_contact, terminal, kin, opts):
                    last_uncarried_terminal = terminal
        run_start = index
    # Keep the last observed descent if every accepted run looks carried.
    landing_frame = last_uncarried_terminal if last_uncarried_terminal is not None else last_terminal
    if landing_frame is None:
        return None
    return landing_frame, track[landing_frame, :2].copy()


# ---------------------------------------------------------------------------
# Net rule, in/out verdict, margins
# ---------------------------------------------------------------------------
def is_net_ender(
    final_contact: int, win_end: int, track: np.ndarray,
    striker_half: CourtHalf, net_band: tuple[float, float], resolution: tuple[float, float],
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
    if striker_half == CourtHalf.TOP:
        never_crossed = bool(np.all(ys <= band_hi))
    else:
        never_crossed = bool(np.all(ys >= band_lo))
    dies_at_net = band_lo <= terminal_y <= band_hi
    return never_crossed and dies_at_net


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


def landing_margins(landing_norm: tuple[float, float], receiver_half: CourtHalf) -> LandingMargins:
    """Court-metre clearances for a landing against the receiver's singles half."""
    x, y = float(landing_norm[0]), float(landing_norm[1])
    y_lo, y_hi = (NET_COURT_Y, 1.0) if receiver_half == CourtHalf.BOT else (0.0, NET_COURT_Y)
    baseline_y = y_hi if receiver_half == CourtHalf.BOT else y_lo  # the non-net y edge = receiver baseline

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
    half: CourtHalf
    at_border: bool
    net_ender: bool


def pick_landing_to_end(
    final_contact: int, end_frame: int, track: np.ndarray,
    kin: LandingKinematics, opts: LandingFilterOptions, striker_half: CourtHalf,
    net_band: tuple[float, float], resolution: tuple[float, float], court_info: dict,
    constants: Thresholds,
    shuttle_hallucination_mask: np.ndarray,
    rejected_intervals: list[tuple[int, int]] | None = None,
    *, infer_net_fault: bool = True,
) -> Landing | None:
    """Pick a landing within one camera view.

    ``opts`` must already be scaled to the video frame rate. Disable net-fault
    inference when the start is a returning view rather than the contact: the
    unseen flight may already have crossed the net.
    """
    landing = filtered_descending_landing(
        final_contact, end_frame, track, kin, opts,
        constants.min_descend_samples, shuttle_hallucination_mask, rejected_intervals,
    )
    if landing is None:
        return None
    landing_frame, landing_xy = landing
    px = np.array([[landing_xy[0] * resolution[0]], [landing_xy[1] * resolution[1]]])
    proj = project_pixels_to_court(px, resolution, court_info)
    norm = (float(proj[0, 0]), float(proj[1, 0]))
    half = CourtHalf.TOP if norm[1] < NET_COURT_Y else CourtHalf.BOT
    return Landing(
        frame=landing_frame, norm=norm, half=half,
        at_border=at_frame_border(landing_xy),
        net_ender=infer_net_fault and is_net_ender(final_contact, end_frame, track, striker_half, net_band, resolution),
    )


class RallyOutcome(NamedTuple):
    """Absolute point winner, landing evidence and a conservative landing-only estimate."""

    winner: CourtHalf | None
    winner_source: WinnerSource | None
    landing: Landing | None
    landing_margin_m: float | None
    within_line_margin: bool
    within_net_margin: bool
    conservative_landing_winner: CourtHalf | None
    search_shortened_by_exclusion: bool


def resolve_rally_outcome(
    striker_half: CourtHalf, next_server: CourtHalf | None, landing: Landing | None,
    band_m: float | None, *, search_shortened_by_exclusion: bool = False,
) -> RallyOutcome:
    """Use the next server when known, otherwise the best landing estimate.

    The conservative landing-only estimate withholds image-border and exact-line
    calls. Both landing estimates withhold an out landing on the striker's own
    half. A net fault takes priority over geometry; a next server takes priority
    over either landing estimate. Landing ambiguity flags are independent of winner.
    """
    receiver = OTHER_HALF[striker_half]
    landing_winner = conservative_landing_winner = None
    margin_m = None
    within_line = within_net = False
    if landing is not None:
        assert band_m is not None
        margins = landing_margins(landing.norm, receiver)
        margin_m = margins.margin_m
        within_line = margins.line_clear_m < band_m
        within_net = margins.net_clear_m < band_m
        out_on_own_half = (
            landing.half == striker_half and landing_margins(landing.norm, striker_half).margin_m < 0
        )
        if landing.net_ender:
            landing_winner = conservative_landing_winner = receiver
        elif not out_on_own_half:
            landing_winner = striker_half if margin_m >= 0 else receiver
            if not landing.at_border and margin_m != 0:
                conservative_landing_winner = landing_winner
    winner = next_server if next_server is not None else landing_winner
    source = None
    if next_server is not None:
        source = WinnerSource.NEXT_SERVER
    elif landing is not None and winner is not None:
        source = WinnerSource.NET_RULE if landing.net_ender else WinnerSource.LANDING_GEOMETRY
    return RallyOutcome(
        winner, source, landing, margin_m, within_line, within_net,
        conservative_landing_winner, search_shortened_by_exclusion,
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
