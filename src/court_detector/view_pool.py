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
2. Donors. In video-robust mode, a scene whose fresh search ended in an accepted
   composite donates the frames that won its markings, with their observed samples
   and stripe assignments. Reused, middle-frame and fallback courts donate nothing.
   In fast-robust mode every court is a fresh middle-frame fit, so its middle frame
   donates. The donors are carried into the reference's pixels and turned to its
   court's orientation. Each marking keeps the donor with the most q_paint10 across
   the group; exact ties keep the earlier scene.
3. Fit. A group with MIN_DONOR_SCENES independent donor scenes gets one stripe fit to
   its kept donors' samples, starting from the reference's court. One scene may win
   every marking; the pool does not require a mixture of scenes.
4. Scores. In every member's middle frame, the pooled court, the scene's court and
   the middle frame's court before composition get the net choice's combined score.
   A missing score term stays missing, and a mean over members exists only when
   every member has that court's score.
5. Scene fallback. Two donors can name the same painted stripe as different
   markings, so the one fit can land between them on blank floor. Each donor's
   finished composite is therefore also a whole-group candidate: it is carried into
   every member and checked and scored there. A candidate that fails a member's
   check or lacks a member's score is out. The best remaining candidate replaces the
   pool only when its mean over the same members beats the pool's mean, or when the
   pool has no mean. An exact tie keeps the pool in video-robust mode and the scene
   court in fast-robust mode. Fast-robust mode also compares the candidates when the
   pooled fit fails.
6. Output. A winning scene court replaces every member's court. Otherwise a valid
   pooled court replaces each member's court that passes that member's own checks
   (composition.check_in_frame, with its feet). A member that fails keeps its scene
   court, as does every member of a group that has neither a valid fit nor a winning
   scene court.
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
    from .detect import LiveModules, SceneCourts, Switches
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


@dataclass(frozen=True)
class Member:
    """A scene in a view group. It keeps its middle frame's context, not its native image."""

    row: dict[str, Any]  # the scene's output row; apply() updates it
    context: ViewContext  # the middle frame's, for the final checks and scores
    to_reference: np.ndarray  # (3, 3) this middle frame's working px to the reference's
    alignment: dict[str, Any] | None  # composition.align's record; None for the reference itself
    scene_corners: np.ndarray  # (4, 2) native px
    middle_corners: np.ndarray  # (4, 2) native px
    middle_score: dict[str, Any] | None  # middle_corners scored in this frame, when composition or join measured it
    composite_measurement: dict[str, Any] | None  # the scene composite's middle-frame measurement


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
    # Independent composites considered, including those that win no markings.
    donor_view_ids: list[str] = field(default_factory=list)


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
                 reference: composition.SearchedFrame) -> list[composition.UsedFrame | None]:
    """A composite's winning frames, carried into the group reference and turned to its court.

    :param to_group: (3, 3) the scene's middle frame working px to the group reference's.
    :return: Each marking's donor in the group's orientation, or None.
    """
    middle = next(item for item in used_frames if item.frame.role == composition.MIDDLE)
    # Scene-reference px back to the scene's middle frame, then on to the group reference.
    scene_to_group = to_group @ np.linalg.inv(middle.to_reference)
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


def carry_to_member(reference_working: np.ndarray, member: Member) -> np.ndarray:
    """A court in the group reference's working px, in a member's native px and its scene court's corner order."""
    carried = composition.carry(reference_working, np.linalg.inv(member.to_reference))
    carried = carried * composition.native_per_working(member.context)
    return np.roll(carried, composition.half_turn_roll(carried, member.scene_corners), axis=0)


def replace_court(row: dict[str, Any], corners: np.ndarray, chosen_key: str, reused_from: str | None) -> None:
    """Replace a row's court and keep the scene's court beside it."""
    row.update(scene_corners_native_px=row["corners_native_px"], scene_chosen_key=row["chosen_key"],
               scene_reused_from=row["reused_from"], corners_native_px=np.asarray(corners).tolist(),
               chosen_key=chosen_key, reused_from=reused_from)


class VideoPool:
    """Group a video's scene courts by camera view as they finish, then apply pooled courts at the end."""

    def __init__(self, live: LiveModules, switches: Switches, court_mode: CourtMode = CourtMode.VIDEO_ROBUST) -> None:
        self.live = live
        self.switches = switches
        self.court_mode = court_mode
        self.groups: list[ViewGroup] = []

    def add(self, row: dict[str, Any], scene: SceneCourts) -> None:
        """Place one scene's court in a view group. Call it for each scene with a court, in video order.

        A failure leaves this scene out of every group, logs it and records it on the row.
        """
        try:
            self.join(row, scene)
        except (ValueError, ArithmeticError) as error:
            logger.exception("%s: could not join a view group; keeping the scene court", row["view_id"])
            row["view_pool"] = {"error": repr(error)}

    def join(self, row: dict[str, Any], scene: SceneCourts) -> None:
        frame = composition.SearchedFrame(composition.MIDDLE, scene.native_frame, scene.context,
                                          np.asarray(scene.corners_native_px, dtype=float), None)
        hashed = image_hash(scene.native_frame)
        group, alignment, to_reference = None, None, np.eye(3)
        for candidate in self.groups:
            if np.mean(hashed != candidate.image_hash) > court_views.MAX_HASH_DISTANCE:
                continue
            record, warp = composition.align(frame, candidate.reference)
            if warp is not None:
                group, alignment, to_reference = candidate, record, warp
                break
        new_group = group is None
        if group is None:
            group = ViewGroup(frame, hashed)
        # Measure everything before changing the group, so a failure leaves it as it was.
        used_frames, middle_score = scene.used_frames, scene.middle_score
        if self.court_mode == CourtMode.FAST_ROBUST:
            # run_video forbids reuse in this mode, so this court is the middle frame's own
            # fresh fit. That one measurement gives both its donated samples and its score.
            evidence = composition.measure(self.live, frame, frame.corners_native)
            used_frames = (composition.UsedFrame(frame, np.eye(3), frame.corners_native, evidence),)
            middle_score = self.evidence_score(scene.context, frame.corners_native, evidence)
        winners = []
        if used_frames:
            winners = scene_donors(self.live, used_frames, to_reference, group.reference)
        if new_group:
            self.groups.append(group)
        group.members.append(Member(row, scene.context, to_reference, alignment, frame.corners_native,
                                    np.asarray(scene.middle_corners_native_px, dtype=float), middle_score,
                                    scene.composite_measurement))
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
        summaries = []
        for group in self.groups:
            summary: dict[str, Any] = {
                "reference_view_id": group.members[0].row["view_id"],
                "member_view_ids": [member.row["view_id"] for member in group.members],
                "donor_view_ids": group.donor_view_ids, "pooled_view_ids": [], "reason": None,
                "chosen_court": None, "chosen_view_id": None,
            }
            if len(group.donor_view_ids) < MIN_DONOR_SCENES[self.court_mode]:
                summary["reason"] = "too_few_donor_scenes"
            else:
                try:
                    self.pool_group(group, summary)
                except (ValueError, ArithmeticError) as error:
                    logger.exception("%s: pooled court failed; keeping the scene courts", summary["reference_view_id"])
                    summary.update(reason="pooling_failed", error=repr(error))
            logger.info("view group %s: %d members, reason %s, court %s from %s", summary["reference_view_id"],
                        len(group.members), summary["reason"], summary["chosen_court"], summary["chosen_view_id"])
            summaries.append(summary)
        return summaries

    def pool_group(self, group: ViewGroup, summary: dict[str, Any]) -> None:
        """Fit the group's pooled court, compare it with each donor's scene court, and apply the winner.

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
            # Video-robust keeps the scene courts. Fast-robust can still give the group one scene's court.
            if not fast:
                return
        outcomes = [self.member_outcome(group, member, pooled_reference) for member in group.members]
        means: dict[str, float | None] = {}
        for court in COMPARED_COURTS:
            scores = [outcome["scores"][court].get("combined_score") for outcome in outcomes]
            means[court] = None if None in scores else float(np.mean(scores))
        summary["mean_combined_scores"] = means
        winner, summary["scene_candidates"] = self.best_scene_candidate(group, outcomes)
        if winner is not None and means["pooled"] is not None:
            # On an exact tie, video-robust keeps the pool and fast-robust keeps the scene court.
            pool_beats_scene = means["pooled"] > winner["mean_combined_score"]
            tie = means["pooled"] == winner["mean_combined_score"]
            if pool_beats_scene or (tie and not fast):
                winner = None
        if winner is None and pooled_reference is None:
            return
        summary.update(chosen_court="pooled" if winner is None else "group_scene",
                       chosen_view_id=None if winner is None else winner["view_id"])
        for index, (member, outcome) in enumerate(zip(group.members, outcomes, strict=True)):
            pooled_corners = None if outcome["pooled_corners"] is None else outcome["pooled_corners"].tolist()
            record = {"reference_view_id": summary["reference_view_id"], "alignment": member.alignment,
                      "pooled_rejection": outcome["pooled_rejection"], "pooled_corners_native_px": pooled_corners,
                      "scores": outcome["scores"]}
            member.row["view_pool"] = record
            if winner is not None:
                corners = winner["corners"][index]
                record.update(court="group_scene", group_scene_view_id=winner["view_id"],
                              group_scene_corners_native_px=corners.tolist())
                record["scores"]["group_scene"] = winner["scores"][index]
                # The winning scene already holds its own court.
                if member.row["view_id"] != winner["view_id"]:
                    replace_court(member.row, corners, GROUP_SCENE_KEY, None)
                continue
            pooled = outcome["pooled_rejection"] is None
            record["court"] = "pooled" if pooled else "scene"
            if pooled:
                replace_court(member.row, outcome["pooled_corners"], POOLED_KEY, None)
                summary["pooled_view_ids"].append(member.row["view_id"])

    def best_scene_candidate(self, group: ViewGroup,
                             outcomes: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """The donor scene court with the highest mean combined score over every member.

        Only the best candidate so far keeps its corners and scores. An exact tie keeps the earlier scene.

        :param outcomes: member_outcome's, one per member, for the scene courts' own scores.
        :return: The winner, or None when every candidate is out, and each candidate's
            mean score and first rejection.
        """
        best, rows = None, []
        for source in group.members:
            if source.row["view_id"] not in group.donor_view_ids:
                continue
            candidate = self.scene_candidate(group.members, outcomes, source)
            rows.append({key: candidate[key] for key in ("view_id", "mean_combined_score", "rejection")})
            if candidate["rejection"] is None and (
                    best is None or candidate["mean_combined_score"] > best["mean_combined_score"]):
                best = candidate
        return best, rows

    def scene_candidate(self, members: list[Member], outcomes: list[dict[str, Any]], source: Member) -> dict[str, Any]:
        """One donor scene's finished court, carried into every member and checked and scored there.

        It stops at the first member whose check it fails or whose score it lacks, since it
        then cannot hold the whole group.

        :return: The source's view_id, the mean combined score and the first rejection.
            A candidate with no rejection also has its corners and scores, one per member.
        """
        source_working = source.scene_corners / composition.native_per_working(source.context)
        in_reference = composition.carry(source_working, source.to_reference)
        candidate: dict[str, Any] = {"view_id": source.row["view_id"], "mean_combined_score": None, "rejection": None}
        corners, scores = [], []
        for member, outcome in zip(members, outcomes, strict=True):
            if member is source:
                # The scene's own detection checked this court in this frame before accepting it.
                court, score, rejection = member.scene_corners, outcome["scores"]["scene"], None
            else:
                court = carry_to_member(in_reference, member)
                score, rejection = self.checked_score(member, court)
            if rejection is None and score.get("combined_score") is None:
                rejection = "missing_score"
            if rejection is not None:
                candidate["rejection"] = {"view_id": member.row["view_id"], "reason": rejection}
                return candidate
            corners.append(court)
            scores.append(score)
        candidate.update(mean_combined_score=float(np.mean([score["combined_score"] for score in scores])),
                         corners=corners, scores=scores)
        return candidate

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
        if member.middle_score is None:
            middle_score = self.measured_score(member.context, member.middle_corners)
        else:
            known = member.middle_score
            middle_score = score_row(known["paint_score"], known["geometry_score"], known["net_reward"], weight)
        if member.composite_measurement is not None:
            known = member.composite_measurement
            scene_score = score_row(known["paint_score"], known["geometry_score"],
                                    self.net_reward(member.context, member.scene_corners), weight)
        elif np.array_equal(member.scene_corners, member.middle_corners):
            scene_score = middle_score
        else:
            scene_score = self.measured_score(member.context, member.scene_corners)
        return {"pooled_corners": pooled, "pooled_rejection": rejection,
                "scores": {"pooled": pooled_score, "scene": scene_score, "middle": middle_score}}

    def checked_score(self, member: Member, corners_native: np.ndarray) -> tuple[dict[str, Any], str | None]:
        """A court's score and first failed check in a member's middle frame. An unmeasurable court has no score."""
        measurement, rejection = composition.check_in_frame(
            self.live, member.context, corners_native, require_people=self.switches.require_people,
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
