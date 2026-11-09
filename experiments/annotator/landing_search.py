"""Compare the ordinary landing search with shorter trial time limits."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from annotator.inference.prepare_annotation_inputs import PreparedAnnotationInputs
from annotator.rally_results import landing_and_winner as point_winner
from annotator.rally_results.estimate_results import final_flight_range
from annotator.types import FinishedRally


@dataclass(frozen=True)
class LandingSearchComparison:
    """One ordinary outcome and its time-limited alternative."""

    rally_id: int
    search_seconds: float
    search_frames: int
    final_contact_frame: int
    time_limit_end_frame: int
    normal_end_frame: int
    trial_end_frame: int
    search_stop_reasons: tuple[str, ...]
    normal_outcome: point_winner.RallyOutcome
    trial_outcome: point_winner.RallyOutcome


def validate_search_limits(search_seconds: tuple[float, ...]) -> None:
    """Require positive, finite trial limits in increasing order."""
    if any(not math.isfinite(seconds) or seconds <= 0 for seconds in search_seconds):
        raise ValueError('landing search seconds must contain finite positive values')
    if any(later <= earlier for earlier, later in pairwise(search_seconds)):
        raise ValueError('landing search seconds must be strictly increasing')


def compare_landing_searches(
    evidence: PreparedAnnotationInputs,
    rallies: Sequence[FinishedRally],
    outcomes: dict[int, point_winner.RallyOutcome],
    search_seconds: tuple[float, ...],
    landing_options: point_winner.LandingFilterOptions,
) -> list[LandingSearchComparison]:
    """Run optional landing trials after ordinary inference, reusing prepared evidence."""
    validate_search_limits(search_seconds)
    options = point_winner.convert_landing_options(landing_options, evidence.fps)
    video_kinematics = point_winner.build_landing_kinematics(
        evidence.track, evidence.players, evidence.pose_kps, evidence.resolution,
    )
    rows = []
    for rally_id, rally in enumerate(rallies):
        striker = rally.final_half
        if striker is None:
            continue
        final_contact = rally.sequence.events[-1].frame
        if evidence.rejected_shuttle_frames[final_contact]:
            continue
        next_rally = rallies[rally_id + 1] if rally_id + 1 < len(rallies) else None
        next_server = next_rally.first_half if next_rally is not None else None
        next_start = next_rally.sequence.start_frame if next_rally is not None else len(evidence.track)
        flight = final_flight_range(final_contact, next_start, rally.sequence.end_frame, evidence)
        if flight is None:
            continue
        scene, search_start, search_limit = flight
        kinematics = video_kinematics
        if search_start > final_contact:
            # A returning camera view cannot supply motion across the unseen interval.
            speed = video_kinematics.speed.copy()
            speed[search_start] = np.nan
            kinematics = video_kinematics._replace(speed=speed)
        window = point_winner.landing_window(
            search_start, search_limit, evidence.track, evidence.exclusion_mask,
            evidence.video_config.thresholds.sustained_loss_frames, evidence.rejected_shuttle_frames,
        )
        normal = outcomes[rally_id]
        for seconds in search_seconds:
            frames = max(1, math.floor(seconds * evidence.fps + 0.5))
            time_limit_end = final_contact + frames
            trial_end = max(final_contact + 1, min(time_limit_end, window.end_frame))
            stop_reasons = []
            if time_limit_end == trial_end:
                stop_reasons.append('time_limit')
            if trial_end == window.end_frame:
                stop_reasons.extend(window.search_stop_reasons)
                trial = normal
            else:
                landing = None
                if trial_end > search_start:
                    landing = point_winner.pick_landing_to_end(
                        search_start, trial_end, evidence.track, kinematics, options,
                        striker, scene.net_band, evidence.resolution, scene.court_info,
                        evidence.video_config.thresholds, evidence.rejected_shuttle_frames,
                        infer_net_fault=search_start == final_contact,
                    )
                trial = point_winner.resolve_rally_outcome(
                    striker, next_server, landing, scene.landing_error_band_m,
                )
            rows.append(LandingSearchComparison(
                rally_id, seconds, frames, final_contact, time_limit_end,
                window.end_frame, trial_end, tuple(stop_reasons), normal, trial,
            ))
    return rows
