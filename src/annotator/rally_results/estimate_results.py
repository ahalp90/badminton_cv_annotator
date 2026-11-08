"""Apply landing, winner and hit-height estimates to every rally in a video."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from annotator.courts.frame_geometry import SceneCourt, court_at_frame
from annotator.inference.prepare_annotation_inputs import PreparedAnnotationInputs
from annotator.rally_results import landing_and_winner
from annotator.types import FinishedRally
from shared.progress import progress_iter


def build_outcomes(
    inputs: PreparedAnnotationInputs,
    rallies: Sequence[FinishedRally],
    *,
    landing_options: landing_and_winner.LandingFilterOptions,
) -> dict[int, landing_and_winner.RallyOutcome]:
    """Estimate landings and winners for rallies whose final hitter is known.

    Measure video-wide motion once, then search each final flight. Reject a
    final contact on a flagged shuttle frame; the next server can still give
    a winner. Landing options must already use this video's frame rate.
    """
    video_kinematics = landing_and_winner.build_landing_kinematics(
        inputs.track, inputs.players, inputs.pose_kps, inputs.resolution,
    )
    outcomes = {}
    for rally_id, rally in enumerate(progress_iter(rallies, "resolving outcomes", unit="rallies")):
        striker = rally.final_half
        if striker is None:
            continue
        next_rally = rallies[rally_id + 1] if rally_id + 1 < len(rallies) else None
        next_server = next_rally.first_half if next_rally is not None else None
        next_start = next_rally.sequence.start_frame if next_rally is not None else len(inputs.track)
        final_contact = rally.sequence.events[-1].frame
        flight = None
        # An earlier hit may belong to the other player; it cannot replace this hit.
        if not inputs.rejected_shuttle_frames[final_contact]:
            flight = final_flight_range(final_contact, next_start, rally.sequence.end_frame, inputs)

        if flight is None:
            outcomes[rally_id] = landing_and_winner.resolve_rally_outcome(striker, next_server, None, None)
            continue

        scene, search_start, search_limit = flight
        starts_in_returning_view = search_start > final_contact
        kinematics = video_kinematics
        if starts_in_returning_view:
            # A camera cut is not shuttle motion in the returning view.
            speed = video_kinematics.speed.copy()
            speed[search_start] = np.nan
            kinematics = video_kinematics._replace(speed=speed)
        window = landing_and_winner.landing_window(
            search_start, search_limit, inputs.track, inputs.exclusion_mask,
            inputs.video_config.thresholds.sustained_loss_frames, inputs.rejected_shuttle_frames,
        )
        landing = landing_and_winner.pick_landing_to_end(
            search_start, window.end_frame, inputs.track, kinematics, landing_options,
            striker, scene.net_band, inputs.resolution, scene.court_info,
            inputs.video_config.thresholds,
            rejected_shuttle_frames=inputs.rejected_shuttle_frames,
            infer_net_fault=not starts_in_returning_view,
        )
        outcomes[rally_id] = landing_and_winner.resolve_rally_outcome(
            striker, next_server, landing, scene.landing_error_band_m,
            search_shortened_by_exclusion=window.search_shortened_by_exclusion,
        )
    return outcomes


def final_flight_range(
    final_contact: int, next_start: int, rally_end: int, inputs: PreparedAnnotationInputs,
) -> tuple[SceneCourt, int, int] | None:
    """Choose a camera view and landing search range, incl start, excl end.

    Prefer the view containing the final contact. If the view returns later,
    use it only within the rally's live time and when its first frame is usable.
    """
    scene = next((scene for scene in inputs.scene_courts
                  if scene.end_frame > final_contact and scene.start_frame < rally_end), None)
    if scene is None:
        return None
    if scene.start_frame <= final_contact:
        return scene, final_contact, min(next_start, scene.end_frame)
    # Recover a flight from a returning view only within this rally's live time.
    if inputs.exclusion_mask[scene.start_frame]:
        return None
    return scene, scene.start_frame, min(next_start, scene.end_frame, rally_end)


def build_hit_heights(
    inputs: PreparedAnnotationInputs, rallies: Sequence[FinishedRally],
) -> tuple[dict[int, int], dict[int, str]]:
    """Measure each contact height; record a reason when its position/court is unavailable."""
    hit_height_by_frame: dict[int, int] = {}
    unavailable_hit_height_reasons: dict[int, str] = {}
    for rally in rallies:
        for event in rally.sequence.events:
            contact_frame = event.frame
            try:
                frame_net_band = court_at_frame(inputs.scene_courts, contact_frame).net_band
                hit_height = landing_and_winner.hit_height(
                    inputs.track, contact_frame, frame_net_band, inputs.resolution,
                )
            except ValueError as exc:
                unavailable_hit_height_reasons[contact_frame] = str(exc)
                continue
            hit_height_by_frame[contact_frame] = hit_height
    return hit_height_by_frame, unavailable_hit_height_reasons
