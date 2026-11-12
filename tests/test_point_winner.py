"""Point-winner verdict chain tests: synthetic tracks/poses, one obvious answer each.

Covers the promoted D5 chain (src/scraper/point_winner.py): the alternation-rhythm fit,
the window fix, the ankle-refined landing filter, the net rule / in-out verdict, margins, hit
height, and the verdict assembly. No fixtures; each test builds its own small numpy arrays.
"""

import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.contacts.player_attribution import (
    contact_player_evidence,
    fit_alternation,
)
from annotator.courts.frame_geometry import SceneCourt
from annotator.rally_results.landing_and_winner import (
    Landing,
    LandingFilterOptions,
    LandingKinematics,
    LandingWindow,
    WinnerSource,
    build_landing_kinematics,
    carried_terminal,
    filtered_descending_landing,
    hit_height,
    is_net_ender,
    landing_margins,
    landing_window,
    pick_landing_to_end,
    resolve_rally_outcome,
)
from annotator.types import (
    ANKLE_L,
    ANKLE_R,
    WRIST_L,
    WRIST_R,
    CourtHalf,
    TrackedPlayers,
    WristGaps,
)

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40

def _mk_pose_arrays(
    n_frames: int,
    players: list[list[tuple[float, float, float, tuple[float, float], tuple[float, float]]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bboxes, scores, kps) for a scene of up to a few court-scale players per frame.

    ``players[frame]`` is a list of ``(foot_x, foot_y, height, wrist_xy, ankle_xy)`` tuples, one
    per player present that frame (empty = no detection). Each box is 60px wide with its foot at
    bottom-centre; both wrists (and both ankles) sit at the same given point, since the promoted
    code takes a min over the pair. Scores descend by slot.
    """
    bboxes = np.full((n_frames, 16, 4), np.nan)
    scores = np.full((n_frames, 16), np.nan)
    kps = np.full((n_frames, 16, 17, 2), np.nan)
    for frame, people in enumerate(players):
        for slot, (foot_x, foot_y, height, wrist_xy, ankle_xy) in enumerate(people):
            bboxes[frame, slot] = (foot_x - 30.0, foot_y - height, foot_x + 30.0, foot_y)
            scores[frame, slot] = 0.9 - 0.1 * slot
            kps[frame, slot, WRIST_L] = wrist_xy
            kps[frame, slot, WRIST_R] = wrist_xy
            kps[frame, slot, ANKLE_L] = ankle_xy
            kps[frame, slot, ANKLE_R] = ankle_xy
    return bboxes, scores, kps


def _sticky(n_frames: int, *, picks: np.ndarray, heights: np.ndarray, distances: np.ndarray) -> tuple[TrackedPlayers, WristGaps]:
    collapsed = np.min(np.where(np.isfinite(distances), distances, np.inf), axis=1)
    collapsed[np.isposinf(collapsed)] = np.nan
    players = TrackedPlayers(picks, np.full((n_frames, 2, 2), np.nan), heights, np.full(n_frames, np.nan))
    return players, WristGaps(collapsed, distances)


# ---------------------------------------------------------------------------
# Alternation-rhythm fit
# ---------------------------------------------------------------------------
def test_fit_alternation_picks_the_higher_scoring_phase():
    # Every guess matches the phase whose final stroke is Bot (phase(Bot,4) = [Top,Bot,Top,Bot]).
    guesses = [CourtHalf.TOP, CourtHalf.BOT, CourtHalf.TOP, CourtHalf.BOT]
    assert fit_alternation(guesses) == CourtHalf.BOT


@pytest.mark.parametrize(('guesses', 'final'), [
    ([CourtHalf.TOP, CourtHalf.TOP], CourtHalf.BOT),
    ([CourtHalf.BOT, CourtHalf.BOT], CourtHalf.TOP),
    ([None, CourtHalf.TOP, CourtHalf.TOP], CourtHalf.BOT),
    ([CourtHalf.TOP, CourtHalf.TOP, None], CourtHalf.TOP),
    ([None, None], None),
    ([], None),
])
def test_fit_alternation_breaks_ties_with_the_first_known_guess(guesses, final):
    assert fit_alternation(guesses) == final


def test_fit_alternation_ignores_none_guesses_when_scoring():
    # One ambiguous stroke in the middle; the other two both fit phase(final=Bot).
    guesses = [CourtHalf.TOP, None, CourtHalf.TOP, CourtHalf.BOT]
    assert fit_alternation(guesses) == CourtHalf.BOT


@pytest.mark.parametrize(('guesses', 'distances', 'final'), [
    ([CourtHalf.TOP, CourtHalf.TOP], [(0.1, 0.2), (0.1, 2.0)], CourtHalf.TOP),
    ([None, CourtHalf.TOP, CourtHalf.TOP], [None, (0.1, 0.2), (0.1, 2.0)], CourtHalf.TOP),
    ([CourtHalf.TOP, CourtHalf.TOP], [None, (0.1, 2.0)], CourtHalf.TOP),
    ([CourtHalf.TOP, CourtHalf.TOP], [(0.1, 0.2), (0.1, 0.2)], CourtHalf.BOT),
    ([CourtHalf.TOP, CourtHalf.TOP], [None, None], CourtHalf.BOT),
    ([None, None], [(0.1, 0.2), (0.1, 2.0)], None),
    ([CourtHalf.TOP, CourtHalf.BOT], [(2.0, 0.1), (0.1, 2.0)], CourtHalf.BOT),
])
def test_wrist_distance_breaks_only_known_vote_ties(guesses, distances, final):
    assert fit_alternation(guesses, distances) == final


@pytest.mark.parametrize(('feet', 'distances', 'visible', 'expected'), [
    ([300.0, 600.0], [0.2, 0.7], 1, (CourtHalf.TOP, (0.2, 0.7))),
    ([600.0, 300.0], [0.2, 0.7], 1, (CourtHalf.BOT, (0.7, 0.2))),
    ([300.0, 600.0], [0.7, 0.2], 1, (CourtHalf.BOT, (0.7, 0.2))),
    ([600.0, 300.0], [0.2, 0.2], 1, (CourtHalf.BOT, (0.2, 0.2))),
    ([300.0, 350.0], [0.2, 0.7], 1, (CourtHalf.TOP, None)),
    ([450.0, 600.0], [0.2, 0.7], 1, (None, None)),
    ([400.0, 600.0], [0.2, 0.7], 1, (None, None)),
    ([300.0, 500.0], [0.7, 0.2], 1, (None, None)),
    ([300.0, 600.0], [np.nan, 0.7], 1, (CourtHalf.BOT, None)),
    ([300.0, 600.0], [np.inf, 0.7], 1, (CourtHalf.BOT, None)),
    ([300.0, 600.0], [np.inf, np.inf], 1, (None, None)),
    ([300.0, 600.0], [0.2, 0.7], 0, (None, None)),
])
def test_contact_player_evidence_uses_scene_halves_for_hitter_and_wrist_costs(feet, distances, visible, expected):
    track = np.array([[0.5, 0.5, visible]])
    bboxes = np.zeros((1, 2, 4))
    bboxes[0, :, 3] = feet
    players, wrist_gaps = _sticky(
        1, picks=np.array([[0, 1]]), heights=np.ones((1, 2)), distances=np.array([distances]),
    )
    scenes = (SceneCourt(0, 1, {}, (400.0, 500.0), 0.1),)
    assert contact_player_evidence(0, track, players, wrist_gaps, bboxes, scenes) == expected


# ---------------------------------------------------------------------------
# Landing window: the top-exit wait vs a plain close
# ---------------------------------------------------------------------------
def test_window_end_waits_past_a_top_exit_gap():
    # Contact frame sits at the top edge (a lob leaving the frame); the sustained-loss gap that
    # follows is skipped, and nothing else closes the window before next_start.
    rows = [(0.5, 0.01, 1)]         # frame 0: at the top edge (< TOP_EDGE_FRAC)
    rows += [(0.0, 0.0, 0)] * 10    # frames 1-10: sustained loss at the 25 fps window
    rows += [(0.5, 0.5, 1)] * 5     # frames 11-15: back in view
    track = np.array(rows, dtype=float)
    dead = np.zeros(len(track), dtype=bool)
    assert landing_window(
        0, len(track), track, dead, scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds.sustained_loss_frames,
        np.zeros(len(track), dtype=bool),
    ).end_frame == len(track)


def test_window_end_closes_at_a_non_top_exit_gap():
    # Same shape, but the pre-gap sample is mid-frame, not the top edge: the sustained loss closes
    # the window right after the contact frame, exactly like a plain end would.
    rows = [(0.5, 0.5, 1)]
    rows += [(0.0, 0.0, 0)] * 10
    rows += [(0.5, 0.5, 1)] * 5
    track = np.array(rows, dtype=float)
    dead = np.zeros(len(track), dtype=bool)
    assert (
        landing_window(
            0, len(track), track, dead, scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds.sustained_loss_frames,
            np.zeros(len(track), dtype=bool),
        ).end_frame
        == 1
    )


def test_window_end_top_exit_wait_still_bounded_by_a_later_non_top_gap():
    # A top-exit gap is skipped, but a SECOND sustained-loss gap that did not follow a top exit
    # still closes the window.
    rows = [(0.5, 0.01, 1)]          # frame 0: top edge
    rows += [(0.0, 0.0, 0)] * 10     # frames 1-10: sustained loss after the top exit (skipped)
    rows += [(0.5, 0.5, 1)] * 3      # frames 11-13: back in view, mid-frame
    rows += [(0.0, 0.0, 0)] * 10     # frames 14-23: a second sustained loss, not top-exit-preceded
    rows += [(0.5, 0.5, 1)] * 3
    track = np.array(rows, dtype=float)
    dead = np.zeros(len(track), dtype=bool)
    assert (
        landing_window(
            0, len(track), track, dead, scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds.sustained_loss_frames,
            np.zeros(len(track), dtype=bool),
        ).end_frame
        == 14
    )


def test_window_end_masked_frame_closes_regardless_of_the_top_exit_wait():
    rows = [(0.5, 0.01, 1)] + [(0.5, 0.5, 1)] * 9
    track = np.array(rows, dtype=float)
    dead = np.zeros(len(track), dtype=bool)
    dead[3] = True
    assert (
        landing_window(
            0, len(track), track, dead, scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds.sustained_loss_frames,
            np.zeros(len(track), dtype=bool),
        ).end_frame
        == 3
    )


def test_window_end_event_mask_removes_top_exit_exemption() -> None:
    rows = [(0.5, 0.01, 1)] + [(0.0, 0.0, 0)] * 10 + [(0.5, 0.5, 1)] * 5
    track = np.array(rows, dtype=float)
    dead = np.zeros(len(track), dtype=bool)
    event_mask = np.zeros(len(track), dtype=bool)
    event_mask[0] = True

    assert landing_window(
        0, len(track), track, dead, scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds.sustained_loss_frames,
        event_mask,
    ).end_frame == 1


def test_window_end_masked_visible_frame_extends_sustained_loss_run() -> None:
    track = np.zeros((22, 3), dtype=float)
    track[:, 2] = 1
    track[:, 1] = 0.5
    track[11:, 2] = 0
    event_mask = np.zeros(len(track), dtype=bool)
    event_mask[10] = True
    dead = np.zeros(len(track), dtype=bool)

    assert landing_window(0, len(track), track, dead, 10, np.zeros(len(track), dtype=bool)).end_frame == 11
    assert landing_window(0, len(track), track, dead, 10, event_mask).end_frame == 10


def test_landing_window_stops_at_next_rally_before_mask_at_boundary():
    track = np.ones((12, 3), dtype=float)
    dead = np.zeros(len(track), dtype=bool)
    dead[5] = True

    window = landing_window(
        0, 5, track, dead, sustained_loss_frames=10, rejected_shuttle_frames=np.zeros(len(track), dtype=bool),
    )

    assert window == LandingWindow(5, ('next_rally',), search_shortened_by_exclusion=False)


def test_landing_window_labels_final_rally_video_end():
    track = np.ones((12, 3), dtype=float)
    dead = np.zeros(len(track), dtype=bool)

    assert landing_window(
        0, len(track), track, dead, 10, np.zeros(len(track), dtype=bool),
    ).search_stop_reasons == ('video_end',)


def test_landing_window_does_not_scan_loss_past_next_rally_boundary():
    track = np.ones((20, 3), dtype=float)
    track[1:14, 2] = 0
    dead = np.zeros(len(track), dtype=bool)

    window = landing_window(
        0, 5, track, dead, sustained_loss_frames=10, rejected_shuttle_frames=np.zeros(len(track), dtype=bool),
    )

    assert window == LandingWindow(5, ('next_rally',), search_shortened_by_exclusion=False)


def test_landing_window_reports_sustained_loss_and_exclusion_tie():
    track = np.ones((20, 3), dtype=float)
    track[5:, 2] = 0
    dead = np.zeros(len(track), dtype=bool)
    dead[5] = True

    window = landing_window(
        0, len(track), track, dead, sustained_loss_frames=3,
        rejected_shuttle_frames=np.zeros(len(track), dtype=bool),
    )

    assert window == LandingWindow(5, ('definitive_exclusion', 'sustained_loss'), search_shortened_by_exclusion=False)


def test_landing_window_keeps_one_frame_minimum_and_tied_boundary_reasons():
    track = np.ones((8, 3), dtype=float)
    dead = np.zeros(len(track), dtype=bool)
    dead[5] = True

    window = landing_window(
        4, len(track), track, dead, sustained_loss_frames=10,
        rejected_shuttle_frames=np.zeros(len(track), dtype=bool),
    )

    assert window == LandingWindow(5, ('definitive_exclusion',), search_shortened_by_exclusion=True)


# ---------------------------------------------------------------------------
# contact_player_evidence (the wrist_boxh arm, end to end)
# ---------------------------------------------------------------------------
def test_contact_player_evidence_reads_the_sticky_nearest_pick_half():
    # The bottom sticky pick owns the nearest cached wrist distance and its foot is below the band.
    resolution = (1920.0, 1080.0)
    n_frames = 30
    contact = 15
    shuttle_px = (500.0, 550.0)
    track = np.zeros((n_frames, 3))
    track[contact] = (shuttle_px[0] / resolution[0], shuttle_px[1] / resolution[1], 1.0)
    player_frame = [(500.0, 600.0, 100.0, shuttle_px, (500.0, 600.0))]
    bboxes, _, _ = _mk_pose_arrays(n_frames, [player_frame] * n_frames)
    picks = np.full((n_frames, 2), -1, dtype=int)
    picks[contact, 1] = 0
    heights = np.full((n_frames, 2), np.nan)
    heights[contact, 1] = 100.0
    distances = np.full((n_frames, 2), np.nan)
    distances[contact, 1] = 0.0
    players, wrist_gaps = _sticky(n_frames, picks=picks, heights=heights, distances=distances)

    scenes = (SceneCourt(0, len(track), {}, (400.0, 500.0), 0.1),)
    half, wrist_costs = contact_player_evidence(contact, track, players, wrist_gaps, bboxes, scenes)
    assert half == CourtHalf.BOT
    assert wrist_costs is None
    no_contact_scene = (SceneCourt(contact + 1, len(track), {}, (400.0, 500.0), 0.1),)
    with pytest.raises(ValueError, match='no court geometry'):
        contact_player_evidence(contact, track, players, wrist_gaps, bboxes, no_contact_scene)


def test_contact_player_evidence_none_when_shuttle_invisible_or_no_detection():
    track = np.zeros((5, 3))
    bboxes = np.full((5, 16, 4), np.nan)
    players, wrist_gaps = _sticky(
        5, picks=np.full((5, 2), -1), heights=np.full((5, 2), np.nan), distances=np.full((5, 2), np.nan),
    )

    assert contact_player_evidence(2, track, players, wrist_gaps, bboxes, ()) == (None, None)

    track[2, 2] = 1
    assert contact_player_evidence(2, track, players, wrist_gaps, bboxes, ()) == (None, None)


# ---------------------------------------------------------------------------
# Kinematic landing filter: the ankle rule flipping a pickup to a landing
# ---------------------------------------------------------------------------
def test_ankle_rule_flips_a_carried_terminal_to_a_landing():
    # Wrist-proximate at every frame (reads as carried); ankle even nearer, so the ankle rule
    # overturns the carry read to a genuine landing.
    kin = LandingKinematics(
        wrist_distance_body_heights=np.array([0.3, 0.3, 0.3]),
        ankle_distance_body_heights=np.array([0.1, 0.1, 0.1]),
        speed=np.zeros(3),
    )
    with_ankle = LandingFilterOptions(settle_win=3, settle_thr=0.01, settle_min=2, carry_win=3, carry_thr=0.5)

    assert carried_terminal(0, 2, kin, with_ankle) is False


def test_ankle_rule_does_not_overturn_when_the_wrist_is_nearer():
    kin = LandingKinematics(
        wrist_distance_body_heights=np.array([0.1, 0.1, 0.1]),
        ankle_distance_body_heights=np.array([0.3, 0.3, 0.3]),  # ankle FARTHER than the wrist
        speed=np.zeros(3),
    )
    opts = LandingFilterOptions(settle_win=3, settle_thr=0.01, settle_min=2, carry_win=3, carry_thr=0.5)
    assert carried_terminal(0, 2, kin, opts) is True


def test_carried_terminal_does_not_read_before_final_contact():
    # Unclamped, the window [0:5] has median 0.1 and calls this carried; clamping
    # to final_contact=2 leaves [0.1, 0.9, 0.9], median 0.9, not carried.
    kin = LandingKinematics(
        wrist_distance_body_heights=np.array([0.1, 0.1, 0.1, 0.9, 0.9]),
        ankle_distance_body_heights=np.full(5, np.nan),
        speed=np.zeros(5),
    )
    opts = LandingFilterOptions(settle_win=3, settle_thr=0.01, settle_min=2, carry_win=5, carry_thr=0.5)

    assert carried_terminal(2, 4, kin, opts) is False


# ---------------------------------------------------------------------------
# Keep the last descent when every run looks carried
# ---------------------------------------------------------------------------
def test_keep_last_drop_returns_the_last_run_when_every_run_is_carried():
    # One clean 3-sample descending run (image-y strictly increasing); its terminal reads carried
    # at every frame. Keep that observed descent as the fallback.
    track = np.array([
        [0.5, 0.10, 1],
        [0.5, 0.20, 1],
        [0.5, 0.30, 1],
    ])
    kin = LandingKinematics(
        wrist_distance_body_heights=np.full(3, 0.1),   # well under carry_thr at every frame: reads carried
        ankle_distance_body_heights=np.full(3, np.nan),  # ankle rule stays neutral (NaN never overturns)
        speed=np.zeros(3),
    )
    opts = LandingFilterOptions(settle_win=3, settle_thr=0.01, settle_min=10000, carry_win=3, carry_thr=0.5)

    min_samples = scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds.min_descend_samples
    landing = filtered_descending_landing(0, 3, track, kin, opts, min_samples, np.zeros(len(track), dtype=bool))
    assert landing is not None
    assert landing[0] == 2  # the only run's terminal, kept despite being carried



def test_landing_discards_a_masked_original_coordinate_interval_and_uses_a_later_run():
    track = np.array([
        [0.5, 0.10, 1], [0.5, 0.20, 1], [0.5, 0.00, 0],
        [0.5, 0.30, 1], [0.5, 0.20, 1], [0.5, 0.40, 1], [0.5, 0.60, 1],
    ])
    kin = LandingKinematics(
        wrist_distance_body_heights=np.full(7, np.nan), ankle_distance_body_heights=np.full(7, np.nan), speed=np.zeros(7),
    )
    opts = LandingFilterOptions(settle_win=3, settle_thr=0.01, settle_min=10000, carry_win=3, carry_thr=0.5)
    event_mask = np.zeros(7, dtype=bool)
    event_mask[2] = True

    landing = filtered_descending_landing(
        0, 7, track, kin, opts, scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds.min_descend_samples,
        rejected_shuttle_frames=event_mask,
    )

    assert landing is not None
    assert landing[0] == 6


def test_landing_returns_none_when_every_candidate_interval_is_masked():
    track = np.array([
        [0.5, 0.10, 1], [0.5, 0.20, 1], [0.5, 0.30, 1], [0.5, 0.20, 1],
        [0.5, 0.40, 1], [0.5, 0.60, 1],
    ])
    kin = LandingKinematics(
        wrist_distance_body_heights=np.full(6, np.nan), ankle_distance_body_heights=np.full(6, np.nan), speed=np.zeros(6),
    )
    opts = LandingFilterOptions(settle_win=3, settle_thr=0.01, settle_min=10000, carry_win=3, carry_thr=0.5)
    event_mask = np.ones(len(track), dtype=bool)

    assert filtered_descending_landing(
        0, len(track), track, kin, opts, scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds.min_descend_samples,
        rejected_shuttle_frames=event_mask,
    ) is None


# ---------------------------------------------------------------------------
# Net rule
# ---------------------------------------------------------------------------
def test_is_net_ender_true_when_flight_stays_in_band_and_dies_there():
    resolution = (1920.0, 1080.0)
    track = np.array([
        [0.5, 420.0 / resolution[1], 1],
        [0.5, 450.0 / resolution[1], 1],
        [0.5, 480.0 / resolution[1], 1],
    ])
    assert is_net_ender(0, 3, track, CourtHalf.TOP, (400.0, 500.0), resolution) is True


def test_is_net_ender_false_when_flight_crosses_the_band():
    resolution = (1920.0, 1080.0)
    track = np.array([
        [0.5, 420.0 / resolution[1], 1],
        [0.5, 600.0 / resolution[1], 1],  # crosses past band_hi: not a net-ender any more
        [0.5, 480.0 / resolution[1], 1],
    ])
    assert is_net_ender(0, 3, track, CourtHalf.TOP, (400.0, 500.0), resolution) is False


# ---------------------------------------------------------------------------
# inout_verdict at margin 0
# ---------------------------------------------------------------------------
def test_inout_verdict_at_margin_zero_resolves_exactly_on_the_line():
    # Landing exactly on the receiver's (Bot) baseline: clearance to that line is exactly 0, so at
    # margin 0 it is neither strictly inside (won) nor strictly outside (lost) -> ambiguous.
    assert landing_margins((0.5, 1.0), CourtHalf.BOT).margin_m == 0


def test_inout_verdict_at_margin_zero_still_resolves_a_clear_landing():
    assert landing_margins((0.5, 0.9), CourtHalf.BOT).margin_m > 0
    assert landing_margins((0.5, 0.01), CourtHalf.BOT).margin_m < 0


# ---------------------------------------------------------------------------
# landing_margins signs/values on hand-computed cases
# ---------------------------------------------------------------------------
def test_landing_margins_inside_receiver_half_is_positive():
    # x=0.5 is centred between the sidelines; y=0.75 sits 0.25 either side of the net/baseline
    # in the Bot half [0.5, 1.0], so the sideline clearance binds (the smaller of the two).
    margins = landing_margins((0.5, 0.75), CourtHalf.BOT)
    assert margins.net_clear_m == pytest.approx(0.25 * COURT_LENGTH_M)
    assert margins.margin_m > 0
    assert margins.margin_m == pytest.approx(margins.line_clear_m)


def test_landing_margins_outside_is_negative_point_to_rectangle_distance():
    # Past the far baseline (y=1.1) but inside x: only the y overshoot contributes.
    margins = landing_margins((0.5, 1.1), CourtHalf.BOT)
    assert margins.margin_m == pytest.approx(-(0.1 * COURT_LENGTH_M))


# ---------------------------------------------------------------------------
# hit_height: ShuttleSet coding (1 above, 2 at/below the net-band centre)
# ---------------------------------------------------------------------------
def test_hit_height_above_below_and_at_centre():
    net_band = (664.6, 703.7)
    resolution = (1920.0, 1080.0)
    centre_px = (net_band[0] + net_band[1]) / 2.0
    track = np.zeros((3, 3))
    track[:, 2] = 1
    track[0, 1] = (centre_px - 10.0) / resolution[1]  # above
    track[1, 1] = (centre_px + 10.0) / resolution[1]  # below
    track[2, 1] = centre_px / resolution[1]            # exactly at the centre

    assert hit_height(track, 0, net_band, resolution) == 1
    assert hit_height(track, 1, net_band, resolution) == 2
    assert hit_height(track, 2, net_band, resolution) == 2  # ties resolve to 2 (below)


def test_hit_height_raises_when_shuttle_not_visible():
    track = np.zeros((3, 3))  # visibility column stays 0
    with pytest.raises(ValueError, match='not visible'):
        hit_height(track, 1, (664.6, 703.7), (1920.0, 1080.0))


# ---------------------------------------------------------------------------
# Verdict assembly
# ---------------------------------------------------------------------------
def _landing(norm, net_ender=False, at_border=False, frame=5):
    half = CourtHalf.TOP if norm[1] < 0.5 else CourtHalf.BOT
    return Landing(frame=frame, court_xy=norm, half=half, at_border=at_border, net_ender=net_ender)


def test_net_fault_loses_for_both_landing_rules():
    outcome = resolve_rally_outcome(CourtHalf.TOP, None, _landing((0.5, 0.9), net_ender=True), 0.1)
    assert outcome.winner is CourtHalf.BOT
    assert outcome.conservative_landing_winner is CourtHalf.BOT
    assert outcome.winner_source is WinnerSource.NET_RULE


def test_best_estimate_calls_an_out_landing_at_the_image_border():
    outcome = resolve_rally_outcome(CourtHalf.TOP, None, _landing((0.5, 1.1), at_border=True), 0.1)
    assert outcome.winner is CourtHalf.BOT
    assert outcome.conservative_landing_winner is None


def test_conservative_estimate_withholds_an_exact_court_line():
    outcome = resolve_rally_outcome(CourtHalf.TOP, None, _landing((0.5, 1.0)), 0.1)
    assert outcome.winner is CourtHalf.TOP
    assert outcome.conservative_landing_winner is None


@pytest.mark.parametrize(('hitter', 'position'), [
    (CourtHalf.TOP, (0.5, -0.1)), (CourtHalf.BOT, (0.5, 1.1)),
    (CourtHalf.TOP, (-0.1, 0.25)), (CourtHalf.BOT, (1.1, 0.75)),
])
def test_out_landing_on_hitters_half_withholds_geometry_but_keeps_next_server(hitter, position):
    landing = Landing(20, position, hitter, False, False)
    outcome = resolve_rally_outcome(hitter, None, landing, 0.1)
    assert outcome.winner is None
    assert outcome.conservative_landing_winner is None
    assert outcome.winner_source is None
    next_serve = resolve_rally_outcome(hitter, hitter, landing, 0.1)
    assert next_serve.winner is hitter
    assert next_serve.winner_source is WinnerSource.NEXT_SERVER


def test_next_server_takes_priority_over_a_net_fault():
    outcome = resolve_rally_outcome(CourtHalf.TOP, CourtHalf.TOP, _landing((0.5, 0.9), net_ender=True), 0.1)
    assert outcome.winner is CourtHalf.TOP
    assert outcome.winner_source is WinnerSource.NEXT_SERVER
    assert outcome.conservative_landing_winner is CourtHalf.BOT


def test_landing_winner_and_margin_flags_are_independent():
    outcome = resolve_rally_outcome(CourtHalf.TOP, None, _landing((0.5, 0.9)), 10.0)
    assert outcome.winner is CourtHalf.TOP
    assert outcome.winner_source is WinnerSource.LANDING_GEOMETRY
    assert outcome.within_line_margin is True
    assert outcome.within_net_margin is True


def test_missing_landing_leaves_the_winner_unresolved():
    outcome = resolve_rally_outcome(CourtHalf.TOP, None, None, 0.1)
    assert outcome.winner is None
    assert outcome.conservative_landing_winner is None
    assert outcome.winner_source is None
    assert outcome.landing_margin_m is None
    assert outcome.within_line_margin is False
    assert outcome.within_net_margin is False


# ---------------------------------------------------------------------------
# Landing pick + kinematics build, end to end
# ---------------------------------------------------------------------------
def test_pick_landing_projects_the_filtered_terminal_and_flags_it():
    # Identity court geometry (H = I, border = the full working-res frame) makes court coords
    # exactly the image fraction, so a picked landing's norm should equal its raw track xy.
    n_frames = 10
    track = np.zeros((n_frames, 3))
    track[:, 2] = 1
    track[2] = (0.5, 0.5, 1)
    track[3] = (0.5, 0.7, 1)
    track[4] = (0.5, 0.9, 1)  # a clean 3-sample descending run, frames 2-4
    dead = np.zeros(n_frames, dtype=bool)
    kin = LandingKinematics(wrist_distance_body_heights=np.full(n_frames, np.nan), ankle_distance_body_heights=np.full(n_frames, np.nan),
                            speed=np.zeros(n_frames))
    opts = LandingFilterOptions(settle_win=3, settle_thr=0.01, settle_min=10000, carry_win=3, carry_thr=0.5)
    resolution = (1280.0, 720.0)
    court_info = {'H': np.eye(3), 'border_L': 0.0, 'border_R': resolution[0],
                  'border_U': 0.0, 'border_D': resolution[1]}
    net_band = (100.0, 200.0)  # well above the landing: irrelevant to net_ender here

    constants = scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds
    end_frame = landing_window(
        2, n_frames, track, dead, constants.sustained_loss_frames, np.zeros(len(track), dtype=bool),
    ).end_frame
    landing = pick_landing_to_end(
        2, end_frame, track, kin, opts, CourtHalf.TOP, net_band, resolution, court_info, constants,
        np.zeros(len(track), dtype=bool),
    )

    assert landing is not None
    assert landing.frame == 4
    assert landing.court_xy == pytest.approx((0.5, 0.9))
    assert landing.half == CourtHalf.BOT  # y=0.9 >= NET_COURT_Y(0.5)
    assert landing.at_border is False
    assert landing.net_ender is False


def test_landing_window_end_feeds_explicit_landing():
    n_frames = 10
    track = np.zeros((n_frames, 3))
    track[:, 2] = 1
    track[2:5, 1] = (0.5, 0.7, 0.9)
    dead = np.zeros(n_frames, dtype=bool)
    kin = LandingKinematics(
        wrist_distance_body_heights=np.full(n_frames, np.nan), ankle_distance_body_heights=np.full(n_frames, np.nan),
        speed=np.zeros(n_frames),
    )
    opts = LandingFilterOptions(7, 0.004, 10000, 7, 0.75)
    resolution = (1280.0, 720.0)
    court_info = {
        'H': np.eye(3), 'border_L': 0.0, 'border_R': resolution[0],
        'border_U': 0.0, 'border_D': resolution[1],
    }
    constants = scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds
    window = landing_window(2, n_frames, track, dead, constants.sustained_loss_frames, np.zeros(len(track), dtype=bool))
    landing = pick_landing_to_end(
        2, window.end_frame, track, kin, opts, CourtHalf.TOP, (100.0, 200.0), resolution, court_info, constants,
        np.zeros(len(track), dtype=bool),
    )
    assert window.end_frame == n_frames
    assert landing is not None and landing.frame == 4


def test_build_landing_kinematics_reads_nearer_wrist_and_ankle_in_body_heights():
    resolution = (1920.0, 1080.0)
    track = np.array([[100.0 / resolution[0], 150.0 / resolution[1], 1.0]])
    # foot (100, 200), box height 100; wrist right on the shuttle, ankle 50px away.
    _, _, kps = _mk_pose_arrays(1, [[(100.0, 200.0, 100.0, (100.0, 150.0), (100.0, 200.0))]])
    players, _ = _sticky(
        1, picks=np.array([[0, -1]]), heights=np.array([[100.0, np.nan]]), distances=np.array([[0.0, np.nan]]),
    )

    kin = build_landing_kinematics(track, players, kps, resolution)

    assert kin.wrist_distance_body_heights[0] == pytest.approx(0.0)  # wrist sits exactly on the shuttle
    assert kin.ankle_distance_body_heights[0] == pytest.approx(0.5)  # ankle 50px away / 100px box height


def test_tracker_failure_leaves_attribution_and_landing_unmeasured():
    resolution = (1920.0, 1080.0)
    track = np.array([[0.5, 0.5, 1.0]])
    bboxes = np.full((1, 1, 4), np.nan)
    kps = np.full((1, 1, 17, 2), np.nan)
    players, wrist_gaps = _sticky(
        1, picks=np.array([[-1, -1]]), heights=np.full((1, 2), np.nan), distances=np.full((1, 2), np.nan),
    )

    scenes = (SceneCourt(0, len(track), {}, (400.0, 500.0), 0.1),)
    assert contact_player_evidence(0, track, players, wrist_gaps, bboxes, scenes) == (None, None)
    kin = build_landing_kinematics(track, players, kps, resolution)
    assert np.isnan(kin.wrist_distance_body_heights[0])
    assert np.isnan(kin.ankle_distance_body_heights[0])
