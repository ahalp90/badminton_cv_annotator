"""Pure prediction rules for the serve-identification follow-up."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """One frozen server and timing decision before ground-truth scoring."""

    branch: str
    predicted_server: str
    claimed_frame: str
    temporal_claim: str
    temporal_gt_label_at_10: str


def other_side(player: str) -> str:
    """Return the opposite court side."""

    try:
        return {"Top": "Bot", "Bot": "Top"}[player]
    except KeyError as exc:
        raise ValueError(f"unexpected player side: {player!r}") from exc


def temporal_slot_is_correct(claim: str, gt_label: str) -> bool:
    """Score serve claims against contact 1 and return claims against contact 2."""

    return (claim == "serve" and gt_label == "contact_1") or (
        claim == "return" and gt_label == "contact_2"
    )


def paired_outcome(new_correct: bool, old_correct: bool) -> str:
    """Describe a paired change from PR #82."""

    if new_correct and not old_correct:
        return "fix"
    if old_correct and not new_correct:
        return "damage"
    if new_correct:
        return "both_correct"
    return "both_wrong"


def preferred_decision(
    search: Mapping[str, str], baseline: Mapping[str, str]
) -> Decision:
    """Apply the preferred outgoing-selected rule with PR #82 fallback."""

    category = search["sequential_category"]
    if category == "first_visible_post_serve_contact":
        return Decision(
            branch="selected_outgoing_contact__incoming__other_side",
            predicted_server=other_side(search["sequential_selected_player"]),
            claimed_frame=search["sequential_selected_frame"],
            temporal_claim="return",
            temporal_gt_label_at_10=search[
                "tolerance_10_sequential_selected_label"
            ],
        )
    if category == "visible_serve":
        return Decision(
            branch="selected_outgoing_contact__not_incoming__selected_side",
            predicted_server=search["sequential_selected_player"],
            claimed_frame=search["sequential_selected_frame"],
            temporal_claim="serve",
            temporal_gt_label_at_10=search[
                "tolerance_10_sequential_selected_label"
            ],
        )

    return Decision(
        branch="pr82_fallback",
        predicted_server=baseline["baseline_server"],
        claimed_frame=baseline["baseline_frame"],
        temporal_claim=(
            "serve" if baseline["baseline_category"] == "visible_serve" else "return"
        ),
        temporal_gt_label_at_10=baseline["baseline_gt_label"],
    )


def rank1_sensitivity_decision(
    search: Mapping[str, str],
    baseline: Mapping[str, str],
    rank1: Mapping[str, str],
) -> Decision:
    """Apply the nearby 171-rule sensitivity with a rank-1 fallback."""

    category = search["sequential_category"]
    if category == "first_visible_post_serve_contact":
        return Decision(
            branch="selected_outgoing_contact__incoming__other_side",
            predicted_server=other_side(search["sequential_selected_player"]),
            claimed_frame=search["sequential_selected_frame"],
            temporal_claim="return",
            temporal_gt_label_at_10=search[
                "tolerance_10_sequential_selected_label"
            ],
        )
    if category == "visible_serve":
        return Decision(
            branch="selected_outgoing_contact__not_incoming__selected_side",
            predicted_server=search["sequential_selected_player"],
            claimed_frame=search["sequential_selected_frame"],
            temporal_claim="serve",
            temporal_gt_label_at_10=search[
                "tolerance_10_sequential_selected_label"
            ],
        )
    if rank1["pre_verdict"] == "incoming":
        return Decision(
            branch="fallback_rank1__incoming__other_side",
            predicted_server=other_side(rank1["player"]),
            claimed_frame=baseline["baseline_frame"],
            temporal_claim="return",
            temporal_gt_label_at_10=baseline["baseline_gt_label"],
        )

    return Decision(
        branch="fallback_rank1__otherwise__rank1_side",
        predicted_server=rank1["player"],
        claimed_frame=baseline["baseline_frame"],
        temporal_claim="serve",
        temporal_gt_label_at_10=baseline["baseline_gt_label"],
    )
