"""Estimate the point winner from the next server or the shuttle landing.

Use the next server when known. Otherwise, estimate a landing from the last
flight and compare it with the receiver's court. Contact sides have already
been assigned.

Landing search stops at ground settling. Prefer descents away from players'
wrists, falling back to the last accepted descent if all look carried.
Each search stays within one camera view. A shuttle leaving the image top
may be a high shot that returns to view.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import numpy as np

from annotator.array_calc_helpers import compute_speed, rolling_nanmedian, true_runs
from annotator.config import Thresholds
from annotator.types import (
    ANKLE_L,
    ANKLE_R,
    OTHER_HALF,
    WRIST_L,
    WRIST_R,
    CourtHalf,
    TrackedPlayers,
)
from shared.court import project_pixels_to_court
from shared.court_model import COURT_LENGTH_M, COURT_WIDTH_M, SINGLES_INSET_M
from shared.frame_timing import frames_at_fps, speed_at_fps


class WinnerSource(StrEnum):
    """The observation or rule used to determine the winner."""

    NEXT_SERVER = 'next_server'  # next rally's fitted first hitter
    LANDING_GEOMETRY = 'landing_geometry'  # in/out of the receiver's singles half
    NET_RULE = 'net_rule'  # flight stayed on the hitter's side and ended at the net


# Court xy uses the doubles outline as a unit square: x across, y along the
# court. Singles sidelines are inset 0.46 m; baselines at y=0 and y=1, net at 0.5.
SINGLES_X_LO = SINGLES_INSET_M / COURT_WIDTH_M  # ~0.07541
SINGLES_X_HI = 1.0 - SINGLES_INSET_M / COURT_WIDTH_M  # ~0.92459
NET_COURT_Y = 0.5
# Within 2% of the top edge: allow a high shot to return after disappearing.
# The same distance from any edge flags a possible exit before landing.
TOP_EDGE_FRAC = 0.02


# Landing result and point winner


class Landing(NamedTuple):
    """Selected landing, projected court position and uncertainty flags.

    :param frame: Original video frame index.
    :param court_xy: Court xy; doubles outline mapped to the unit square.
    :param half: Top/Bot from court y versus 0.5. This does not establish in/out.
    :param at_border: Position within 2% of an image edge; it may be an exit
        from view before the shuttle reaches the floor.
    :param net_ender: Observed flight stayed on the hitter's side and ended
        in the net band.
    """

    frame: int
    court_xy: tuple[float, float]
    half: CourtHalf
    at_border: bool
    net_ender: bool


class RallyOutcome(NamedTuple):
    """Estimated winner, landing and reasons for uncertainty.

    ``winner`` uses the next server when known, otherwise landing/net evidence.
    ``winner_source`` identifies that choice. ``conservative_landing_winner``
    ignores the next server and withholds image-border and exact-line calls
    unless a net fault determines the result.

    ``landing_margin_m`` is signed distance to the receiver's singles bounds.
    ``within_line_margin`` and ``within_net_margin`` mean the landing is close
    enough to a line or the net to fall within the court-position error estimate.
    These flags apply even when the winner is known.
    ``search_shortened_by_exclusion`` means excluded footage cut the search short.
    """

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
    """Choose the next server as winner when known; otherwise use landing evidence.

    A net fault awards the point to the receiver before any in/out check.
    For other landings, withhold an out landing on the hitter's own half.
    The conservative landing estimate also withholds image-border and
    exact-line calls. Keep ambiguity flags even when the next server is known.
    """
    receiver = OTHER_HALF[striker_half]
    landing_winner = conservative_landing_winner = None
    margin_m = None
    within_line = within_net = False
    if landing is not None:
        assert band_m is not None
        margins = landing_margins(landing.court_xy, receiver)
        margin_m = margins.margin_m
        within_line = margins.line_clear_m < band_m
        within_net = margins.net_clear_m < band_m
        out_on_own_half = (
            landing.half == striker_half and landing_margins(landing.court_xy, striker_half).margin_m < 0
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


# Landing search settings and measurements


class LandingFilterOptions(NamedTuple):
    """Settings for distinguishing a landing from a pause or movement by hand.

    Pass base-30-fps values to ``annotate_video``. ``convert_landing_options``
    scales them once; lower-level landing functions take the converted values.

    :param settle_win: Rolling-median speed window, in frames.
    :param settle_thr: Maximum still speed, in normalised image distance/frame.
    :param settle_min: Consecutive still, grounded frames needed to stop searching.
    :param carry_win: Trailing window for median wrist distance, in frames.
    :param carry_thr: Maximum wrist distance considered carried, in body heights.
        Also distinguishes a held pause from a ground settle.
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
    """Scale base-30-fps landing settings to the video frame rate; call once."""
    def frame_count(value: int) -> int:
        return frames_at_fps(value, fps)

    return opts._replace(
        settle_win=frame_count(opts.settle_win),
        settle_thr=speed_at_fps(opts.settle_thr, fps),
        settle_min=frame_count(opts.settle_min),
        carry_win=frame_count(opts.carry_win),
    )


@dataclass(frozen=True)
class LandingWindow:
    """Where to stop looking for a landing, excluding ``end_frame``.

    ``search_stop_reasons`` lists any limits reached at that frame.
    ``search_shortened_by_exclusion`` is True only when excluded footage made
    the search shorter than it would otherwise have been.
    """

    end_frame: int
    search_stop_reasons: tuple[str, ...]
    search_shortened_by_exclusion: bool


class LandingKinematics(NamedTuple):
    """Shuttle speed and nearest wrist/ankle distances, for landing filters.

    Divide each selected player's distance by their current box height, then
    take the minimum. NaN for an invisible shuttle or no usable player. The
    nearest wrist and ankle may belong to different players.

    Speed is normalised image distance per frame; NaN if either end of the
    step is invisible.
    """

    wrist_distance_body_heights: np.ndarray
    ankle_distance_body_heights: np.ndarray
    speed: np.ndarray


# Find and check a landing


def pick_landing_to_end(
    search_start: int, end_frame: int, track: np.ndarray,
    kin: LandingKinematics, opts: LandingFilterOptions, striker_half: CourtHalf,
    net_band: tuple[float, float], resolution: tuple[float, float], court_info: dict,
    constants: Thresholds,
    rejected_shuttle_frames: np.ndarray,
    *, infer_net_fault: bool = True,
) -> Landing | None:
    """Choose a descent, project its endpoint onto the court and check for a net fault.

    Search one camera view, incl search_start, excl end_frame. ``opts`` must
    already use the video frame rate. Disable net-fault inference for a
    returning camera view: the unseen flight may already have crossed the net.
    """
    landing = filtered_descending_landing(
        search_start, end_frame, track, kin, opts,
        constants.min_descend_samples, rejected_shuttle_frames,
    )
    if landing is None:
        return None
    landing_frame, landing_xy = landing
    px = np.array([[landing_xy[0] * resolution[0]], [landing_xy[1] * resolution[1]]])
    proj = project_pixels_to_court(px, resolution, court_info)
    court_xy = (float(proj[0, 0]), float(proj[1, 0]))
    half = CourtHalf.TOP if court_xy[1] < NET_COURT_Y else CourtHalf.BOT
    return Landing(
        frame=landing_frame, court_xy=court_xy, half=half,
        at_border=at_frame_border(landing_xy),
        net_ender=infer_net_fault and is_net_ender(search_start, end_frame, track, striker_half, net_band, resolution),
    )


def landing_window(
    search_start: int, search_limit: int, track: np.ndarray, dead: np.ndarray,
    sustained_loss_frames: int, rejected_shuttle_frames: np.ndarray,
) -> LandingWindow:
    """Find the landing search end, excluding the returned endpoint.

    Start after ``search_start``. Stop at the supplied limit, excluded footage
    or sustained shuttle loss. For visibility, ignore rejected shuttle grades.
    A gap after a top-edge exit may be a high shot; continue looking for its
    return instead of stopping at that gap.
    """
    video_end = len(track)
    boundaries: dict[str, int] = {'video_end': video_end}
    if search_limit < video_end:
        boundaries['next_rally'] = search_limit

    scan_start = search_start + 1
    scan_end = min(search_limit, video_end)
    seg_dead = dead[scan_start:scan_end]  # first masked frame after contact
    if seg_dead.any():
        boundaries['definitive_exclusion'] = scan_start + int(np.argmax(seg_dead))
    effective_visibility = (track[:, 2] == 1) & ~rejected_shuttle_frames
    invisible = ~effective_visibility[scan_start:scan_end]  # first sustained-loss run start
    for run_start, run_end in true_runs(invisible):
        if run_end - run_start >= sustained_loss_frames:
            if gap_after_top_exit(
                search_start, run_start, track, rejected_shuttle_frames,
            ):
                continue  # lob left the frame top; wait for the shuttle to re-enter
            boundaries['sustained_loss'] = scan_start + run_start
            break
    search_end = max(search_start + 1, min(boundaries.values()))
    reason_order = ('next_rally', 'definitive_exclusion', 'sustained_loss', 'video_end')
    reasons = tuple(
        reason for reason in reason_order
        if boundaries.get(reason) == search_end
    )
    end_without_exclusion = max(search_start + 1, min(
        end for reason, end in boundaries.items() if reason != 'definitive_exclusion'
    ))
    return LandingWindow(
        end_frame=search_end, search_stop_reasons=reasons,
        search_shortened_by_exclusion=search_end < end_without_exclusion,
    )


def gap_after_top_exit(
    search_start: int, run_start: int, track: np.ndarray,
    rejected_shuttle_frames: np.ndarray,
) -> bool:
    """Was the last visible, accepted position before this gap near the image top?

    The gap starts at search_start + 1 + run_start. A top-edge exit allows
    a longer search for a returning high shot.
    """
    last_vis = search_start + run_start
    visible = track[last_vis, 2] == 1
    visible = visible and not rejected_shuttle_frames[last_vis]
    return bool(visible and track[last_vis, 1] < TOP_EDGE_FRAC)


def build_landing_kinematics(
    track: np.ndarray, players: TrackedPlayers, kps: np.ndarray, resolution: tuple[float, float],
) -> LandingKinematics:
    """Measure speed and the nearest wrist/ankle distances at each frame.

    Divide each selected player's distances by their current box height.
    Invisible shuttle frames and frames with no usable player get NaN.

    :param resolution: Video width and height for converting shuttle xy to pixels.
    """
    width, height = resolution
    n_frames = len(track)
    wrist_distance_body_heights = np.full(n_frames, np.nan)
    ankle_distance_body_heights = np.full(n_frames, np.nan)
    for frame in np.flatnonzero(track[:, 2] == 1):
        shuttle_x, shuttle_y = track[frame, 0] * width, track[frame, 1] * height
        wrist_ratios = []
        ankle_ratios = []
        for player_row in range(2):
            detection_index = players.detection_indices[frame, player_row]
            box_height = players.box_heights_px[frame, player_row]
            if detection_index < 0 or not np.isfinite(box_height):
                continue
            wrists = kps[frame, detection_index, (WRIST_L, WRIST_R), :]
            wrist_px = np.hypot(wrists[:, 0] - shuttle_x, wrists[:, 1] - shuttle_y).min()
            # A missing joint distance becomes zero (near the player) for the minimum.
            wrist_ratio = wrist_px / box_height
            wrist_ratios.append(0.0 if np.isnan(wrist_ratio) else wrist_ratio)
            ankles = kps[frame, detection_index, (ANKLE_L, ANKLE_R), :]
            ankle_px = np.hypot(ankles[:, 0] - shuttle_x, ankles[:, 1] - shuttle_y).min()
            ankle_ratio_val = ankle_px / box_height
            ankle_ratios.append(0.0 if np.isnan(ankle_ratio_val) else ankle_ratio_val)
        if wrist_ratios:
            wrist_distance_body_heights[frame] = float(min(wrist_ratios))
            ankle_distance_body_heights[frame] = float(min(ankle_ratios))
    return LandingKinematics(
        wrist_distance_body_heights, ankle_distance_body_heights, compute_speed(track),
    )


def filtered_descending_landing(
    search_start: int, search_limit: int, track: np.ndarray,
    kin: LandingKinematics, opts: LandingFilterOptions, min_descend_samples: int,
    rejected_shuttle_frames: np.ndarray,
) -> tuple[int, np.ndarray] | None:
    """Choose the last uncarried descent, or the last accepted descent as fallback.

    Stop searching at the first ground settle. A descent needs at least
    min_descend_samples visible positions with increasing image y. Invisible
    frames between those samples are skipped. Reject a descent if any frame
    in its full range has a rejected shuttle grade.

    Prefer the last descent that does not look carried. If all accepted
    descents look carried, keep the last one. Return (frame, normalised xy),
    or None if no descent remains.
    """
    cap = landing_settle_cap(search_start, search_limit, kin, opts)
    search_end = max(cap, search_start + 1)
    frames = np.arange(search_start, search_end)
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
            if not rejected_shuttle_frames[start_frame:terminal + 1].any():
                last_terminal = terminal
                if not carried_terminal(search_start, terminal, kin, opts):
                    last_uncarried_terminal = terminal
        run_start = index
    # Keep the last observed descent if every accepted run looks carried.
    landing_frame = last_uncarried_terminal if last_uncarried_terminal is not None else last_terminal
    if landing_frame is None:
        return None
    return landing_frame, track[landing_frame, :2].copy()


def landing_settle_cap(search_start: int, search_limit: int, kin: LandingKinematics,
                opts: LandingFilterOptions) -> int:
    """First sustained ground-rest frame, or search_limit if none is found.

    Require low rolling-median speed and a shuttle away from the wrists.
    A shuttle closer to an ankle than a wrist also counts as grounded.
    Stop at the start of the first run lasting settle_min frames.
    """
    speed_seg = kin.speed[search_start:search_limit]
    # Missing speeds are skipped by the median. Nearby visible frames can still
    # establish rest; an entirely missing window cannot.
    static = rolling_nanmedian(speed_seg, opts.settle_win) <= opts.settle_thr
    held = np.nan_to_num(kin.wrist_distance_body_heights[search_start:search_limit], nan=np.inf) <= opts.carry_thr
    # A shuttle nearer an ankle than a wrist is grounded, even beside a player.
    seg_carry = kin.wrist_distance_body_heights[search_start:search_limit]
    seg_ankle = kin.ankle_distance_body_heights[search_start:search_limit]
    held = held & ~(seg_ankle < seg_carry)
    ground_static = static & ~held
    run = 0
    for offset, is_static in enumerate(ground_static):
        run = run + 1 if is_static else 0
        if run >= opts.settle_min:
            return search_start + offset - opts.settle_min + 1  # run onset
    return search_limit


def carried_terminal(
    search_start: int, terminal: int, kin: LandingKinematics, opts: LandingFilterOptions,
) -> bool:
    """Check whether a descent ends with the shuttle held near a wrist.

    Use the median wrist distance over the trailing carry_win frames, starting
    no earlier than the final contact. If the last position is closer to
    an ankle than a wrist, treat it as grounded instead.
    """
    lo = max(search_start, terminal - opts.carry_win + 1)
    window = kin.wrist_distance_body_heights[lo:terminal + 1]
    finite = window[np.isfinite(window)]
    carried = len(finite) > 0 and bool(np.median(finite) <= opts.carry_thr)
    if carried:
        ankle_r = kin.ankle_distance_body_heights[terminal]
        wrist_r = kin.wrist_distance_body_heights[terminal]
        if np.isfinite(ankle_r) and np.isfinite(wrist_r) and ankle_r < wrist_r:
            return False
    return carried


# Net faults and court boundaries


def is_net_ender(
    search_start: int, search_limit: int, track: np.ndarray,
    striker_half: CourtHalf, net_band: tuple[float, float], resolution: tuple[float, float],
) -> bool:
    """Check whether the flight stays on the hitter's side and ends in the net band.

    Use visible image-y positions in pixels. Top hitter: y never exceeds
    band_hi. Bot hitter: y never falls below band_lo. The final visible
    position must be inside the band. True means the hitter lost at the net.
    """
    frames = np.arange(search_start, search_limit)
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


def at_frame_border(xy: np.ndarray) -> bool:
    """Check whether a shuttle position is within 2% of an image edge."""
    x, y = float(xy[0]), float(xy[1])
    return x < TOP_EDGE_FRAC or x > 1.0 - TOP_EDGE_FRAC or y < TOP_EDGE_FRAC or y > 1.0 - TOP_EDGE_FRAC


class LandingMargins(NamedTuple):
    """Distances from the landing to the receiver's singles-court boundaries.

    :param margin_m: Signed metres to the nearest boundary: positive inside,
        zero on the line, negative outside. Outside a corner, use the diagonal
        distance to that corner.
    :param net_clear_m: Unsigned metres to the net line.
    :param line_clear_m: Unsigned metres to the nearest sideline or receiver
        baseline, excluding the net.
    """

    margin_m: float
    net_clear_m: float
    line_clear_m: float


def landing_margins(landing_norm: tuple[float, float], receiver_half: CourtHalf) -> LandingMargins:
    """Measure landing clearances in metres against the receiver's singles half."""
    x, y = float(landing_norm[0]), float(landing_norm[1])
    y_lo, y_hi = (NET_COURT_Y, 1.0) if receiver_half == CourtHalf.BOT else (0.0, NET_COURT_Y)
    baseline_y = y_hi if receiver_half == CourtHalf.BOT else y_lo

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


# Contact height code


def hit_height(
    track: np.ndarray, contact_frame: int, net_band: tuple[float, float],
    resolution: tuple[float, float],
) -> int:
    """ShuttleSet height code: 1 above the net-band centre, 2 at or below it.

    Compare the shuttle's image y at contact with the net-band centre, both in
    pixels. Require a visible shuttle position. This is an image-height code,
    not a measured height above the court.

    :param net_band: Lower/upper image-y bounds in video pixels.
    :param resolution: Video width and height.
    """
    if track[contact_frame, 2] != 1:
        raise ValueError(f'shuttle not visible at contact frame {contact_frame}: cannot read hit_height')
    _, height = resolution
    shuttle_y_px = track[contact_frame, 1] * height
    band_lo, band_hi = net_band
    centre = (band_lo + band_hi) / 2.0
    return 1 if shuttle_y_px < centre else 2
