"""Plain prompts for the contact and broadcast cleanup trials."""

from __future__ import annotations

from .trial_schema import TrialArm, TrialCase, TrialKind

_EVENT_PROMPT = """You are checking a two-second badminton video around one automatically proposed contact.

The gold border marks the accepted timing window: eight native video frames before through eight frames after the proposed instant. This is about ±10 frames after normalising to 30 FPS. The cyan ring shows where the shuttle tracker says the shuttle is. Both markers can be wrong. Judge the underlying pixels and motion, not the marker or the proposal. A contact outside the gold-bordered window does not validate this proposal.

The labels define TOP as the far player and BOTTOM as the near player.

Answer these questions:
- Is there a real racket-shuttle contact anywhere in the gold-bordered window? Count an inferred off-screen serve only when serve preparation is visible before a cut and active rally play begins immediately after the cut. Players merely standing in standard court view are not enough.
- Is the support a visible contact, a logically inferred off-screen contact, no contact, or unclear?
- If yes, is the player on the top or bottom half of the court the actor?
- Is another clear racket-shuttle contact visible elsewhere in this short clip but outside the gold-bordered window?

Use "unclear" when the broadcast does not show enough visual or sequence evidence. Return a bare JSON object with exactly these keys: "contact_at_marker", "evidence_kind", "actor", "nearby_unmarked_contact", "visible_evidence", "uncertainty". contact_at_marker and nearby_unmarked_contact must be "yes", "no", or "unclear". evidence_kind must be "visible-contact", "inferred-contact", "no-contact", or "unclear". actor must be "top", "bottom", "no-contact", or "unclear". Give one short sentence for visible_evidence and uncertainty. Do not use a Markdown fence."""


_BROADCAST_PROMPT = """You are checking a badminton sequence around an automatically proposed rally span.

{sampling_note}

Decide whether most of the marked target is a coherent live rally sequence. A valid target can include serve preparation, an inferred off-screen serve, a camera cut, or a late final shuttle flight. A replay is not live play even when it shows real badminton action. Replays often repeat an earlier live action from a closer angle after the standard live view. Use the surrounding context to spot that sequence. Crowd or player cutaways, graphics, warm-up, and unrelated motion are also not live-rally evidence. Answer "no" when replay or cutaway footage occupies most of the marked target.

Before answering, compare the unmarked BEFORE context with the gold target, then compare the gold target with the unmarked AFTER context. Ask whether the target repeats or reframes action already shown, and whether the broadcast then returns to the standard live view or next point. A standard-view action followed by a closer repeat and then a return to standard view is a replay. Do not answer live-play merely because the target contains continuous badminton action.

Return a bare JSON object with exactly these keys: "valid_rally_evidence", "broadcast_content", "contains_camera_cut", "visible_evidence", "uncertainty". valid_rally_evidence and contains_camera_cut must be "yes", "no", or "unclear". broadcast_content must be "live-play", "mixed", "replay", "cutaway", "other", or "unclear". Give one short sentence for visible_evidence and uncertainty. Do not use a Markdown fence."""


def _broadcast_prompt(case: TrialCase) -> str:
    if case.pipeline_priors.get("sampling_layout") == "dense-four-second-target":
        sampling_note = (
            "The clip contains 50 ordered frames from a twenty-second source window. "
            "The 30 consecutive gold-bordered frames densely sample the four-second "
            "target. The unmarked frames sparsely show the surrounding broadcast order. "
            "Judge the gold-bordered target; use the unmarked frames only as context."
        )
    else:
        sampling_note = (
            "The gold border marks the ten-second target. Judge that marked target. "
            "The unmarked video before or after it is context for broadcast order only. "
            "Near the start or end of a recording, the marked target may sit at one end "
            "of this clip."
        )
    return _BROADCAST_PROMPT.format(sampling_note=sampling_note)


_TRACK_PROMPT = """You are checking whether a badminton shuttle tracker follows the real shuttle in a two-second video.

The cyan ring is the tracker's claim. The clip first gives unbordered full-view context, then repeats the short target interval slowly with a gold border.{zoom_note} Judge that repeated gold-bordered interval. The tracker can lock onto court text, a logo, a racket, a player, or empty background. Judge the underlying pixels and motion, not the ring. This is not a contact question.

Answer yes only when the cyan ring follows a visible real shuttle consistently through the gold-bordered interval. Answer no when it follows a different object, empty space, or a guessed path with no visible shuttle. Use unclear when compression, occlusion, or size makes the object genuinely impossible to identify.

Return a bare JSON object with exactly these keys: "tracked_object", "visible_evidence", "uncertainty". tracked_object names what the cyan ring follows and must be "real-shuttle", "text-or-logo", "player-or-racket", "empty-or-unrelated", or "unclear". Give one short sentence for visible_evidence and uncertainty. Do not use a Markdown fence."""


def _track_prompt(case: TrialCase) -> str:
    target_view = case.pipeline_priors.get("target_view")
    if target_view == "clean-then-marked-zoom":
        zoom_note = (
            " The gold-bordered target appears twice in a fixed enlarged view: first "
            "without the cyan marker, then with it. Use the clean replay to identify "
            "the actual pixels and the marked replay only to locate the tracker claim. "
            "Do not infer a shuttle from the marker."
        )
    elif target_view == "tracker-centred-zoom":
        zoom_note = (
            " The gold-bordered replay is a fixed enlarged view around the claimed "
            "track; it does not change the underlying frames."
        )
    else:
        zoom_note = ""
    return _TRACK_PROMPT.format(zoom_note=zoom_note)


def _support(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "not measured"


def _format_event_priors(case: TrialCase) -> list[str]:
    priors = case.pipeline_priors
    return [
        f"- court detector present: {_support(priors['court_present'])}",
        f"- shuttle tracker visible at the marked frame: {_support(priors['track_visible'])}",
        f"- pose wrist proximity supports this tracker point: {_support(priors['wrist_near'])}",
        f"- secondary proximity check supports it: {_support(priors['proximity_ok'])}",
        f"- candidate lost local duplicate suppression: {_support(priors['suppressed'])}",
        f"- broad exclusion mask covers the marked frame: {_support(priors['raw_masked'])}",
        f"- final exclusion mask covers the marked frame: {_support(priors['definitive_masked'])}",
        f"- seconds from previous raw candidate: {priors['seconds_from_previous_raw_candidate']}",
        f"- seconds to next raw candidate: {priors['seconds_to_next_raw_candidate']}",
        "A failed wrist or tracker check does not rule out a serve inferred across a broadcast cut.",
    ]


def _format_broadcast_priors(case: TrialCase) -> list[str]:
    return [
        f"- {name.replace('_', ' ')}: {value}"
        for name, value in sorted(case.pipeline_priors.items())
        if name != "sampling_layout"
    ]


def _format_priors(case: TrialCase) -> str:
    lines = ["\n\nFallible pipeline observations follow. They are hints, not labels:"]
    lines.extend(
        _format_event_priors(case)
        if case.kind is TrialKind.EVENT
        else _format_broadcast_priors(case)
    )
    lines.append("Resolve any conflict in favour of the visible video.")
    return "\n".join(lines)


def build_prompt(case: TrialCase, arm: TrialArm) -> str:
    """Build one prompt while keeping the video-only arm free of pipeline priors."""
    if case.kind is TrialKind.EVENT:
        base = _EVENT_PROMPT
    elif case.kind is TrialKind.BROADCAST:
        base = _broadcast_prompt(case)
    else:
        base = _track_prompt(case)
    if arm is TrialArm.VIDEO_ONLY:
        return base
    return base + _format_priors(case)
