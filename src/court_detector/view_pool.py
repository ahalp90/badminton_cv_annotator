"""Pool scene courts across a video's returning camera views, then refit once per view.

Video-robust mode (run_video --court-mode video-robust) runs this after every scene
has its scene-robust court, so no court changes until the video is done. Fast-robust
mode (--court-mode fast-robust) runs the same steps on courts that each come from one
fresh search of the scene's middle frame, with no endpoint frames or composite.

1. Groups. Each scene with a court joins the first group whose fixed reference it
   matches, or becomes a new group's reference. A perceptual hash within
   court_views.MAX_HASH_DISTANCE only shortlists a group. composition.align must then
   give a usable warp. Camera movement is allowed because the warp carries donated
   samples into reference coordinates. Members align directly to the reference.
2. Donors. Fresh accepted individual frames donate their observed samples and
   stripe assignments, whether or not the scene's combined fit won. Reused courts
   donate nothing. In fast-robust mode each fresh middle-frame court donates. Donors
   move into reference pixels and turn to its court's orientation. Each marking
   keeps the donor with the most q_paint10 across
   the group; exact ties keep the earlier scene.
3. Fit. A group with MIN_DONOR_SCENES independent donor scenes gets one stripe fit to
   its kept donors' samples, starting from the reference's court. One scene may win
   every marking; the pool does not require a mixture of scenes.
4. Scores. In every member's middle frame, the pooled court, the scene's court and
   the middle frame's court before composition get the net choice's combined score.
   A missing score term stays missing. The summary's mean_combined_scores has a mean
   only when every member has that court's score; step 6 ranks by its own mean.
5. Candidates. Two donors can name the same painted stripe as different markings,
   so the one fit can land between them on blank floor. Each donor's finished court
   is therefore also a candidate for the whole group, beside the pooled court. Only
   its source, the member whose frame it was fitted in, can rule a candidate out.
   A donor's court passed its scene's own detection there. The pooled court's source
   is the reference: fit_in_reference checks the solver, rank, depth, convexity and
   hard validity, and the reference member's check adds the camera and, when on,
   the upright camera. A candidate without a combined score at its source is out.
6. Choice. Each candidate is carried into every other member and checked and
   scored there. Players can leave the court between points in an aligned view, so
   this check leaves their feet out. Candidates rank by their mean combined score
   over the members that could score them, including members whose check they
   fail. measured_members counts those members. The best scene candidate replaces
   the pool when its mean beats the pool's, or when the pool is out. An exact tie
   keeps the pool in video-robust mode and the scene court in fast-robust mode.
   Both modes also compare the scene candidates when the pooled fit fails.
7. Output. The chosen court replaces the court of every member that accepts it. A
   member whose check it fails, or that cannot score it, keeps its own court and
   records the rejection. When every candidate is out, all members keep theirs.
8. Receivers. A scene whose search found no court may still take its group's chosen
   court. At apply(), when every group exists, it joins the first group it matches as
   in step 1. With no court of its own, it aligns inside the reference's court at the
   same pixels. It never donates, scores a candidate or counts as a donor scene, so it
   cannot change the fit or the choice. The chosen court must pass a member's step 6
   checks and have a combined score there. Otherwise the scene stays without a court.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from annotator import court_views
from shared.court_model import CORNER_COURT_M

from . import composition, net_choice
from .detect import MAX_HORIZON_TILT_DEG, NET_OVERRUN_WORKING_PX
from .line_observations import MARKINGS

if TYPE_CHECKING:
    from .detect import LiveModules, PreparedView, SceneCourts, Switches
    from .measurements import ViewContext

logger = logging.getLogger(__name__)


class CourtMode(StrEnum):
    SCENE_ROBUST = "scene-robust"  # each scene keeps its own court
    VIDEO_ROBUST = "video-robust"  # scenes of one camera view may share a pooled court
    FAST_ROBUST = "fast-robust"  # video-robust pooling of fresh middle-frame courts, without endpoint frames


POOLED_KEY = "video_pool"  # chosen_key of a pooled court
GROUP_SCENE_KEY = "video_pool_scene"  # chosen_key of another scene's court carried into this scene
# The courts each member's scores compare
COMPARED_COURTS = ("pooled", "scene", "middle")
# Independent donor scenes a group needs before it pools
MIN_DONOR_SCENES = {CourtMode.VIDEO_ROBUST: 2, CourtMode.FAST_ROBUST: 3}
NO_POOLED_FIT = "no_valid_pooled_fit"  # a member's pooled_rejection when the group's fit failed
MISSING_SCORE = "missing_score"  # a member's rejection of a court that passes its checks without a combined score
# The parts of a group candidate that the group's summary keeps
CANDIDATE_SUMMARY = ("view_id", "mean_combined_score", "measured_members", "rejection", "transfer_rejections")


@dataclass(frozen=True)
class Member:
    """A scene in a view group. It keeps its middle frame's context, not its native image."""

    row: dict[str, Any]  # the scene's output row; apply() updates it
    context: ViewContext  # the middle frame's, for the final checks and scores
    to_reference: np.ndarray  # (3, 3) this middle frame's working px to the reference's
    alignment: dict[str, Any] | None  # composition.align's record; None for the reference itself
    scene_corners: np.ndarray  # (4, 2) native px
    middle_corners: np.ndarray | None  # (4, 2) native px, absent after endpoint-only recovery
    middle_score: dict[str, Any] | None  # middle_corners scored in this frame, when composition or join measured it
    chosen_measurement: dict[str, Any] | None  # the selected court's middle-frame measurement
    individual_courts: tuple[tuple[str, np.ndarray, dict[str, Any]], ...] = ()


@dataclass(frozen=True)
class Receiver:
    """A scene whose search found no court, placed in a view group. It keeps its middle frame's context."""

    row: dict[str, Any]  # the scene's output row; apply() updates it
    context: ViewContext
    to_reference: np.ndarray  # (3, 3) this middle frame's working px to the reference's
    alignment: dict[str, Any]  # composition.align's record


@dataclass(frozen=True)
class Donor:
    view_id: str  # the donor scene's middle view
    used: composition.UsedFrame  # to_reference and corners in the group reference's pixels and orientation


@dataclass
class ViewGroup:
    reference: composition.SearchedFrame  # the first member's middle frame and scene court; fixed
    image_hash: np.ndarray
    members: list[Member] = field(default_factory=list)
    donors: list[Donor | None] = field(default_factory=lambda: [None] * len(MARKINGS))  # one per marking
    # Independent source scenes considered, including those that win no markings.
    donor_view_ids: list[str] = field(default_factory=list)
    receivers: list[Receiver] = field(default_factory=list)  # placed by apply()


def image_hash(native_frame: np.ndarray) -> np.ndarray:
    return court_views.describe_court_view([native_frame]).hashes[0]


def q_paint(used: composition.UsedFrame, marking: int) -> float:
    return used.evidence["markings"][marking]["q_paint10"]


def score_row(paint: float | None, geometry: float | None, net_reward: float, geometry_weight: float) -> dict:
    """composition's combined score terms. A missing paint or geometry score leaves no combined score."""
    row: dict[str, Any] = {"paint_score": paint, "geometry_score": geometry, "net_reward": net_reward}
    if paint is not None and geometry is not None:
        row.update(composition.score_parts(paint, geometry, net_reward, geometry_weight))
    return row


def scene_donors(live: LiveModules, used_frames: tuple[composition.UsedFrame, ...], to_group: np.ndarray,
                 reference: composition.SearchedFrame,
                 middle_to_reference: np.ndarray | None = None) -> list[composition.UsedFrame | None]:
    """Accepted source frames carried into the group reference and turned to its court.

    :param to_group: (3, 3) the scene's middle frame working px to the group reference's.
    :return: Each marking's donor in the group's orientation, or None.
    """
    if middle_to_reference is None:
        middle = next(item for item in used_frames if item.frame.role == composition.MIDDLE)
        middle_to_reference = middle.to_reference
    # Scene-reference px back to the scene's middle frame, then on to the group reference.
    scene_to_group = to_group @ np.linalg.inv(middle_to_reference)
    reference_working = reference.corners_native / reference.native_per_working
    winners = composition.choose_donors(list(used_frames))
    turned = []
    for used in used_frames:
        if not any(used is winner for winner in winners):
            continue
        to_reference = scene_to_group @ used.to_reference
        own_working = used.corners_native / used.frame.native_per_working
        roll = composition.half_turn_roll(composition.carry(own_working, to_reference), reference_working)
        if roll == 0:
            turned.append(dataclasses.replace(used, to_reference=to_reference))
            continue
        # A turned court names each painted line by the opposite marking, so measure it again.
        corners = np.roll(used.corners_native, roll, axis=0)
        turned.append(composition.UsedFrame(used.frame, to_reference, corners,
                                            composition.measure(live, used.frame, corners)))
    return composition.choose_donors(turned)


def pooled_constraints(donors: list[Donor | None]) -> tuple[composition.stripe_fitting.Constraints, list[dict]]:
    """Every kept donor's samples on its markings, in the group reference's working px.

    :return: The joined constraints, and one row per marking naming its donor.
    """
    parts, summaries, seen = [], {}, []
    for donor in donors:
        if donor is None or any(donor.used is used for used in seen):
            continue
        seen.append(donor.used)
        markings = [marking for marking, other in enumerate(donors) if other is not None and other.used is donor.used]
        part, summary = composition.donated_constraints(donor.used, markings)
        parts.append(part)
        summaries.update(summary)
    rows = []
    for marking, name in enumerate(MARKINGS):
        donor = donors[marking]
        if donor is None:
            rows.append({"marking": name, "donor_view_id": None})
            continue
        rows.append({"marking": name, "donor_view_id": donor.view_id, "donor_role": donor.used.frame.role,
                     "q_paint10": q_paint(donor.used, marking), **summaries[marking]})
    return composition.joined(parts), rows


def carry_from_reference(reference_working: np.ndarray, frame: Member | Receiver) -> np.ndarray:
    """A court in the group reference's working px, in a member's or receiver's native px, in the same corner order."""
    carried = composition.carry(reference_working, np.linalg.inv(frame.to_reference))
    return carried * composition.native_per_working(frame.context)


def carry_to_member(reference_working: np.ndarray, member: Member) -> np.ndarray:
    """A court in the group reference's working px, in a member's native px and its scene court's corner order."""
    carried = carry_from_reference(reference_working, member)
    return np.roll(carried, composition.half_turn_roll(carried, member.scene_corners), axis=0)


def group_candidate(view_id: str | None, members: list[Member], source: Member, corners: list[np.ndarray | None],
                    scores: list[dict[str, Any]], rejections: list[str | None]) -> dict[str, Any]:
    """One court's standing as the whole group's court, from its corners, score and check in each member.

    A member with no combined score for a court that passes its checks rejects it as
    MISSING_SCORE. A rejection in the source, the member whose frame the court was
    fitted in, rules the candidate out. A rejection in any other member only keeps
    that member on its own court. The mean covers every member with a combined score,
    including members that reject the court.

    :param view_id: The source scene, or None for the pooled court.
    :param corners: One (4, 2) per member in its native px, or None without a court.
    :param rejections: Each member's first failed check, or None.
    :return: The CANDIDATE_SUMMARY fields, then corners, scores and rejections, one per member.
    """
    combined = [score.get("combined_score") for score in scores]
    rejections = [MISSING_SCORE if rejection is None and value is None else rejection
                  for rejection, value in zip(rejections, combined, strict=True)]
    measured = [value for value in combined if value is not None]
    source_rejection, transfer_rejections = None, []
    for member, rejection in zip(members, rejections, strict=True):
        if rejection is None:
            continue
        failure = {"view_id": member.row["view_id"], "reason": rejection}
        if member is source:
            source_rejection = failure
        else:
            transfer_rejections.append(failure)
    # A source that scores the court is always measured, so only a ruled-out candidate lacks a mean.
    mean = float(np.mean(measured)) if measured else None
    return {"view_id": view_id, "mean_combined_score": mean, "measured_members": len(measured),
            "rejection": source_rejection, "transfer_rejections": transfer_rejections, "corners": corners,
            "scores": scores, "rejections": rejections}


def replace_court(row: dict[str, Any], corners: np.ndarray, chosen_key: str, reused_from: str | None) -> None:
    """Replace a row's court and keep the scene's court beside it."""
    row.update(scene_corners_native_px=row["corners_native_px"], scene_chosen_key=row["chosen_key"],
               scene_reused_from=row["reused_from"], corners_native_px=np.asarray(corners).tolist(),
               chosen_key=chosen_key, reused_from=reused_from)


def receive_court(row: dict[str, Any], corners: np.ndarray, chosen_key: str) -> None:
    """Give a courtless row a shared court. Its own outcome stays beside it, as replace_court keeps a scene's court."""
    row.update(scene_status=row["status"], scene_no_court_reason=row["no_court_reason"], status="court",
               no_court_reason=None)
    replace_court(row, corners, chosen_key, None)


def recorded_corners(corners: np.ndarray | None) -> list[list[float]] | None:
    """Keep failed projections out of the JSON while retaining their rejection reason."""
    return corners.tolist() if corners is not None and np.isfinite(corners).all() else None


class VideoPool:
    """Group a video's scene courts by camera view as they finish, then apply pooled courts at the end."""

    def __init__(self, live: LiveModules, switches: Switches, court_mode: CourtMode = CourtMode.VIDEO_ROBUST) -> None:
        self.live = live
        self.switches = switches
        self.court_mode = court_mode
        self.groups: list[ViewGroup] = []
        # Scenes without a court, waiting for apply() to place them
        self.courtless: list[tuple[dict[str, Any], PreparedView]] = []

    def add(self, row: dict[str, Any], scene: SceneCourts) -> None:
        """Place one scene's court in a view group. Call it for each scene with a court, in video order.

        A failure leaves this scene out of every group, logs it and records it on the row.
        """
        try:
            self.join(row, scene)
        except (ValueError, ArithmeticError) as error:
            logger.exception("%s: could not join a view group; keeping the scene court", row["view_id"])
            row["view_pool"] = {"error": repr(error)}

    def add_receiver(self, row: dict[str, Any], prepared: PreparedView) -> None:
        """Keep a scene whose search found no court, so apply() can offer it its view group's chosen court.

        Its middle frame and context stay until apply() places it, in any order relative to the courts.
        """
        self.courtless.append((row, prepared))

    def shortlisted(self, hashed: np.ndarray) -> list[ViewGroup]:
        """The groups whose reference hash is within court_views.MAX_HASH_DISTANCE of this one, in group order."""
        return [group for group in self.groups if np.mean(hashed != group.image_hash) <= court_views.MAX_HASH_DISTANCE]

    def join(self, row: dict[str, Any], scene: SceneCourts) -> None:
        frame = composition.SearchedFrame(composition.MIDDLE, scene.native_frame, scene.context,
                                          np.asarray(scene.corners_native_px, dtype=float), None)
        hashed = image_hash(scene.native_frame)
        group, alignment, to_reference = None, None, np.eye(3)
        for candidate in self.shortlisted(hashed):
            record, warp = composition.align(frame, candidate.reference)
            if warp is not None:
                group, alignment, to_reference = candidate, record, warp
                break
        new_group = group is None
        if group is None:
            group = ViewGroup(frame, hashed)
        # Measure everything before changing the group, so a failure leaves it as it was.
        used_frames, middle_score = scene.used_frames, scene.middle_score
        if self.court_mode == CourtMode.FAST_ROBUST or (not used_frames and row["reused_from"] is None):
            # A fresh court without retained composition evidence can still donate its
            # own markings. One measurement gives both its samples and its score.
            evidence = composition.measure(self.live, frame, frame.corners_native)
            used_frames = (composition.UsedFrame(frame, np.eye(3), frame.corners_native, evidence),)
            middle_score = self.evidence_score(scene.context, frame.corners_native, evidence)
        winners = []
        if used_frames:
            winners = scene_donors(self.live, used_frames, to_reference, group.reference, scene.middle_to_reference)
        if new_group:
            self.groups.append(group)
        middle_corners = scene.middle_corners_native_px
        group.members.append(Member(row, scene.context, to_reference, alignment, frame.corners_native,
                                    None if middle_corners is None else np.asarray(middle_corners, dtype=float),
                                    middle_score, scene.chosen_measurement, scene.individual_courts))
        if not used_frames:
            return
        for marking, used in enumerate(winners):
            known = group.donors[marking]
            if used is not None and (known is None or q_paint(used, marking) > q_paint(known.used, marking)):
                group.donors[marking] = Donor(row["view_id"], used)
        group.donor_view_ids.append(row["view_id"])

    def apply(self) -> list[dict[str, Any]]:
        """Fit and apply one pooled court per group with enough donor scenes.

        :return: One summary per group. A group that fails logs it and keeps its scene courts.
        """
        self.place_receivers()
        summaries = []
        for group in self.groups:
            summary: dict[str, Any] = {
                "reference_view_id": group.members[0].row["view_id"],
                "member_view_ids": [member.row["view_id"] for member in group.members],
                "donor_view_ids": group.donor_view_ids, "pooled_view_ids": [], "reason": None,
                "chosen_court": None, "chosen_view_id": None,
                "receiver_view_ids": [receiver.row["view_id"] for receiver in group.receivers],
                "received_view_ids": [],
            }
            if len(group.donor_view_ids) < MIN_DONOR_SCENES[self.court_mode]:
                summary["reason"] = "too_few_donor_scenes"
            else:
                try:
                    self.pool_group(group, summary)
                except (ValueError, ArithmeticError) as error:
                    logger.exception("%s: pooled court failed; keeping the scene courts", summary["reference_view_id"])
                    summary.update(reason="pooling_failed", error=repr(error))
            logger.info("view group %s: %d members, %d of %d receivers took a court, reason %s, court %s from %s",
                        summary["reference_view_id"], len(group.members), len(summary["received_view_ids"]),
                        len(group.receivers), summary["reason"], summary["chosen_court"], summary["chosen_view_id"])
            summaries.append(summary)
        return summaries

    def place_receivers(self) -> None:
        """Place each courtless scene in the first view group it matches. Every group exists by now.

        A failure logs it, records it on the row and leaves the scene in no group.
        """
        for row, prepared in self.courtless:
            try:
                self.place_receiver(row, prepared)
            except (ValueError, ArithmeticError) as error:
                logger.exception("%s: could not join a view group; keeping the scene without a court", row["view_id"])
                row["view_pool"] = {"receiver": True, "error": repr(error)}
        # Placed receivers keep only their contexts; the native frames can go.
        self.courtless.clear()

    def place_receiver(self, row: dict[str, Any], prepared: PreparedView) -> None:
        """join()'s matching for a scene without a court, which aligns inside the reference's court instead."""
        for group in self.shortlisted(image_hash(prepared.native_frame)):
            # The reference court at the same native px masks this frame, as its own court masks a member's.
            frame = composition.SearchedFrame(composition.MIDDLE, prepared.native_frame, prepared.context,
                                              group.reference.corners_native, None)
            record, warp = composition.align(frame, group.reference)
            if warp is not None:
                group.receivers.append(Receiver(row, prepared.context, warp, record))
                return

    def share_with_receiver(self, receiver: Receiver, chosen_reference: np.ndarray, chosen_key: str,
                            summary: dict[str, Any]) -> None:
        """Give a receiver the group's chosen court when the court passes a member's transfer checks there.

        A rejection, a missing combined score or a failure leaves the scene without a court.
        A failure logs it and records it, without touching the members' outcome.

        :param chosen_reference: (4, 2) the chosen court in the reference's working px and corner order.
        """
        record: dict[str, Any] = {"reference_view_id": summary["reference_view_id"], "alignment": receiver.alignment,
                                  "receiver": True, "court": None}
        receiver.row["view_pool"] = record
        try:
            corners = carry_from_reference(chosen_reference, receiver)
            score, rejection = self.checked_score(receiver, corners)
        except (ValueError, ArithmeticError) as error:
            logger.exception("%s: could not check the shared court; keeping the scene without a court",
                             receiver.row["view_id"])
            record["error"] = repr(error)
            return
        if rejection is None and score.get("combined_score") is None:
            rejection = MISSING_SCORE
        record.update(rejection=rejection, corners_native_px=recorded_corners(corners), score=score)
        if rejection is not None:
            return
        record["court"] = summary["chosen_court"]
        receive_court(receiver.row, corners, chosen_key)
        summary["received_view_ids"].append(receiver.row["view_id"])

    def pool_group(self, group: ViewGroup, summary: dict[str, Any]) -> None:
        """Fit the group's pooled court, choose between it and each donor's scene court, and share the choice.

        Rows change only after every court is measured, so a failure changes none.
        """
        fast = self.court_mode == CourtMode.FAST_ROBUST
        reference = composition.UsedFrame(group.reference, np.eye(3), group.reference.corners_native, {})
        constraints, summary["markings"] = pooled_constraints(group.donors)
        fit = composition.fit_in_reference(self.live, reference, constraints)
        fit_corners = fit.get("corners_native_px")
        summary["fit"] = {"status": fit["status"], "sample_count": len(constraints.points), "valid": fit["valid"],
                          "validity_reason": fit["validity_reason"], "corners_reference_native_px": fit_corners}
        pooled_reference = None
        if fit["valid"]:
            pooled_reference = np.asarray(fit_corners, dtype=float)
        else:
            summary["reason"] = f"fit_{fit['validity_reason']}"
        outcomes = [self.member_outcome(group, member, pooled_reference) for member in group.members]
        means: dict[str, float | None] = {}
        for court in COMPARED_COURTS:
            scores = [outcome["scores"][court].get("combined_score") for outcome in outcomes]
            means[court] = None if None in scores else float(np.mean(scores))
        summary["mean_combined_scores"] = means
        # The pooled court's source is the reference, the group's first member. Without
        # a fit, every member's rejection is NO_POOLED_FIT, the reference's among them.
        pooled = group_candidate(None, group.members, group.members[0],
                                 [outcome["pooled_corners"] for outcome in outcomes],
                                 [outcome["scores"]["pooled"] for outcome in outcomes],
                                 [outcome["pooled_rejection"] for outcome in outcomes])
        summary["pooled_candidate"] = None
        if pooled_reference is not None:
            summary["pooled_candidate"] = {key: pooled[key] for key in CANDIDATE_SUMMARY}
            if pooled["rejection"] is not None:
                summary["reason"] = f"reference_{pooled['rejection']['reason']}"
        best_scene, summary["scene_candidates"] = self.best_scene_candidate(group, outcomes)
        pool_wins = pooled["rejection"] is None
        if pool_wins and best_scene is not None:
            # On an exact tie, video-robust keeps the pool and fast-robust keeps the scene court.
            pool_beats_scene = pooled["mean_combined_score"] > best_scene["mean_combined_score"]
            tie = pooled["mean_combined_score"] == best_scene["mean_combined_score"]
            pool_wins = pool_beats_scene or (tie and not fast)
        chosen = pooled if pool_wins else best_scene
        if chosen is None:
            return
        summary.update(chosen_court="pooled" if pool_wins else "group_scene", chosen_view_id=chosen["view_id"])
        if "role" in chosen:
            summary["chosen_frame_role"] = chosen["role"]
        for index, (member, outcome) in enumerate(zip(group.members, outcomes, strict=True)):
            pooled_corners = recorded_corners(outcome["pooled_corners"])
            record = {"reference_view_id": summary["reference_view_id"], "alignment": member.alignment,
                      "pooled_rejection": pooled["rejections"][index], "pooled_corners_native_px": pooled_corners,
                      "scores": outcome["scores"]}
            member.row["view_pool"] = record
            rejection = chosen["rejections"][index]
            if not pool_wins:
                record.update(group_scene_view_id=chosen["view_id"], group_scene_rejection=rejection,
                              group_scene_corners_native_px=recorded_corners(chosen["corners"][index]))
                record["scores"]["group_scene"] = chosen["scores"][index]
            # Only this member keeps its own court when it rejects the chosen one.
            record["court"] = summary["chosen_court"] if rejection is None else "scene"
            # The chosen scene already holds its own court.
            if rejection is not None or (member.row["view_id"] == chosen["view_id"]
                                          and np.array_equal(member.scene_corners, chosen["corners"][index])):
                continue
            replace_court(member.row, chosen["corners"][index], POOLED_KEY if pool_wins else GROUP_SCENE_KEY, None)
            if pool_wins:
                summary["pooled_view_ids"].append(member.row["view_id"])
        # Receivers come only after the choice and the members' rows, so they cannot change either.
        # The reference is the first member, at the identity, so its corners are the chosen
        # court in reference px and the reference's corner order.
        chosen_reference = chosen["corners"][0] / group.reference.native_per_working
        for receiver in group.receivers:
            self.share_with_receiver(receiver, chosen_reference, POOLED_KEY if pool_wins else GROUP_SCENE_KEY, summary)

    def best_scene_candidate(self, group: ViewGroup,
                             outcomes: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """The donor scene court with the highest mean combined score over the members that could score it.

        Only the best candidate so far keeps its corners and scores. An exact tie keeps the earlier scene.

        :param outcomes: member_outcome's, one per member, for the scene courts' own scores.
        :return: The best candidate, or None when every candidate is out at its source,
            and each candidate's CANDIDATE_SUMMARY fields.
        """
        best, rows = None, []
        for source in group.members:
            if source.row["view_id"] not in group.donor_view_ids:
                continue
            courts = [(None, source.scene_corners, None)]
            for role, corners, measurement in source.individual_courts:
                if np.array_equal(corners, source.scene_corners):
                    continue
                own_score = score_row(measurement["paint_score"], measurement["geometry_score"],
                                      self.net_reward(source.context, corners), self.switches.geometry_weight)
                courts.append((role, corners, own_score))
            for role, corners, own_score in courts:
                candidate = self.scene_candidate(group.members, outcomes, source, corners, own_score)
                if role is not None:
                    candidate["role"] = role
                row = {key: candidate[key] for key in CANDIDATE_SUMMARY}
                if "role" in candidate:
                    row["role"] = candidate["role"]
                rows.append(row)
                if candidate["rejection"] is None and (
                        best is None or candidate["mean_combined_score"] > best["mean_combined_score"]):
                    best = candidate
        return best, rows

    def scene_candidate(self, members: list[Member], outcomes: list[dict[str, Any]], source: Member,
                        source_corners: np.ndarray | None = None,
                        source_score: dict[str, Any] | None = None) -> dict[str, Any]:
        """One donor scene's finished court, carried into every other member and checked and scored there.

        :return: group_candidate's record of the court.
        """
        if source_corners is None:
            source_corners = source.scene_corners
        source_working = source_corners / composition.native_per_working(source.context)
        in_reference = composition.carry(source_working, source.to_reference)
        corners, scores, rejections = [], [], []
        for member, outcome in zip(members, outcomes, strict=True):
            if member is source:
                # The scene's own detection checked this court in this frame before accepting it.
                court = source_corners
                score = outcome["scores"]["scene"] if source_score is None else source_score
                rejection = None
            else:
                court = carry_to_member(in_reference, member)
                score, rejection = self.checked_score(member, court)
            corners.append(court)
            scores.append(score)
            rejections.append(rejection)
        return group_candidate(source.row["view_id"], members, source, corners, scores, rejections)

    def member_outcome(self, group: ViewGroup, member: Member, pooled_reference: np.ndarray | None) -> dict[str, Any]:
        """The pooled court in this member's middle frame and corner order, its checks, and all three scores.

        :param pooled_reference: (4, 2) the pooled court in the reference's native px; None
            when the fit failed, which leaves the member no pooled court or score.
        """
        pooled, pooled_score, rejection = None, {}, NO_POOLED_FIT
        if pooled_reference is not None:
            pooled = carry_to_member(pooled_reference / group.reference.native_per_working, member)
            pooled_score, rejection = self.checked_score(member, pooled)
        weight = self.switches.geometry_weight
        if member.middle_corners is None:
            middle_score = {}
        elif member.middle_score is None:
            middle_score = self.measured_score(member.context, member.middle_corners)
        else:
            known = member.middle_score
            middle_score = score_row(known["paint_score"], known["geometry_score"], known["net_reward"], weight)
        if member.chosen_measurement is not None:
            known = member.chosen_measurement
            scene_score = score_row(known["paint_score"], known["geometry_score"],
                                    self.net_reward(member.context, member.scene_corners), weight)
        elif np.array_equal(member.scene_corners, member.middle_corners):
            scene_score = middle_score
        else:
            scene_score = self.measured_score(member.context, member.scene_corners)
        return {"pooled_corners": pooled, "pooled_rejection": rejection,
                "scores": {"pooled": pooled_score, "scene": scene_score, "middle": middle_score}}

    def checked_score(self, member: Member | Receiver,
                      corners_native: np.ndarray) -> tuple[dict[str, Any], str | None]:
        """A court's score and first failed check in a member's or receiver's middle frame.

        An unmeasurable court has no score.
        """
        # An aligned view can show a break between points. Absent players do not
        # invalidate the court geometry shared with that view.
        measurement, rejection = composition.check_in_frame(
            self.live, member.context, corners_native, require_people=False,
            max_horizon_tilt_deg=MAX_HORIZON_TILT_DEG if self.switches.upright_camera else None,
        )
        if measurement is None:
            return {}, rejection
        score = score_row(measurement["paint_score"], measurement["geometry_score"],
                          self.net_reward(member.context, corners_native), self.switches.geometry_weight)
        return score, rejection

    def net_reward(self, context: ViewContext, corners_native: np.ndarray) -> float:
        net_state, posts = net_choice.net_posts(corners_native, context)
        return net_choice.net_reward(net_state, posts, NET_OVERRUN_WORKING_PX)

    def measured_score(self, context: ViewContext, corners_native: np.ndarray) -> dict[str, Any]:
        """A court's combined score in a frame, measured as composition.own_frame_score measures it."""
        live = self.live
        working = np.asarray(corners_native, dtype=float) / composition.native_per_working(context)
        homography = cv2.getPerspectiveTransform(CORNER_COURT_M, working.astype(np.float32)).astype(float)
        entry = {"homography_working": homography}
        with live.prepared_measurements(live.verifier):
            evidence, _ = live.verifier.measure_candidate(context, entry, {})
        return self.evidence_score(context, corners_native, evidence)

    def evidence_score(self, context: ViewContext, corners_native: np.ndarray, evidence: dict[str, Any]) -> dict[str, Any]:
        """A court's combined score from its measure_candidate evidence in the same frame."""
        return score_row(evidence["q_paint10_span_weighted"], evidence["q_geom_span_weighted"],
                         self.net_reward(context, corners_native), self.switches.geometry_weight)
