# Decisions

All behavioural choices are fixed. Consecutive accepted contacts may represent one shot from positive endpoint evidence alone. Gap contents never veto the connection, and the maximum contact gap is 75 base-30fps frames.

## Settled continuity rule

For earlier accepted contact `A_j` and later accepted contact `A_i`, connect their paths when:

- `A_j` has a common-eligible outgoing trace with fitted distance decrease `<= -0.05` body heights
- `A_i` has a common-eligible incoming trace with fitted distance decrease `>= 0.05` body heights
- `A_j` and `A_i` are consecutive accepted contacts in the same rally span
- their contact gap is at most 75 base-30fps frames

Missing shuttle frames are neutral. This includes production `high_shot_oob` gaps and ordinary TrackNet dropouts. Do not require shared frames, adjacent frames, or a bbox-relative seam-distance check.

The current production `high_shot_oob` path supports this interpretation. `_gap_state_rest_mask` may hold a missing run open for `gap_state_demotion_bound = 75` base-30fps frames. `_gap_is_high_shot_oob` uses visible pre-gap motion, while `_gap_passes_reentry_guard` uses visible post-gap motion. Neither helper joins shuttle positions across the invisible section.

The local motion measurements keep their established rules:

```text
local window            30 base-30fps frames before or after the contact
minimum path            5 visible recurrence-clean frames
maximum local gap       2 base-30fps frames from the contact
gross-jump limit        largest_step_ratio <= 4.0 within each visible trace
incoming                fitted_decrease_bh >= 0.05
outgoing                fitted_decrease_bh <= -0.05
```

## Fixed gap rules

### Gap evidence

Only consecutive accepted contacts are candidates. For accepted frames `A_j` and `A_i`, require `i = j + 1`. There cannot be another accepted contact between them.

TrackNet evidence between the endpoint traces never vetoes the connection. Missing track, hallucinations, jumps, recurrence-guard failures, and any other gap contents are allowed. Do not add direction, trajectory-shape, speed, endpoint-distance, or spatial-continuity tests across the gap.

The connection uses only positive endpoint evidence: credible outgoing motion from `A_j` and credible incoming motion into `A_i`. The rule is deliberately permissive. Directional sanity may be reconsidered only after the fixed run is scored.

Each endpoint trace remains inside its own tracker scene. A scene boundary between the endpoint traces does not veto the connection.

### Contact timing

Require `A_i - A_j <= 75` base-30fps frames, scaled to source FPS. This is 2.5 seconds at 30 fps. The boundary is inclusive and matches the existing `high_shot_oob` demotion bound.

Do not tune the 75-frame limit against GT. Record the complete contact-gap distribution before GT scoring so the report shows whether the cap affects any candidate connection.

Store each consecutive accepted-contact gap in source frames and base-30fps units. The denominator is consecutive accepted-contact pairs, grouped over all 239 rallies. Save the exact values and a frequency table before GT joining. Report counts below 75, exactly 75, and above 75 base-30fps frames.

## Technical basis

```text
A_j < A_i             earlier and later accepted contact frames
S_k                   player slot attributed at accepted contact A_k
L                     scale(30 base-30fps frames, source fps)
G                     scale(2 base-30fps frames, source fps)
usable_k              recurrence-clean bool mask, shape (n_frames,), for S_k
track[:, :2]           normalised shuttle positions, shape (n_frames, 2)
distance[:, S_k]       shuttle-to-player distance in body heights, shape (n_frames,)
bbox_height[:, S_k]    player height in pixels, shape (n_frames,)
```

The existing pre-contact selection is:

```text
window                 [A_i - L, A_i)
P_i = [p0, p1)         latest maximal true run in usable_i and the same scene
pre_gap                 A_i - (p1 - 1); the immediately previous frame has gap 1
common eligible         len(P_i) >= 5 and pre_gap <= G and largest_step_ratio <= 4.0
incoming                eligible and fitted_decrease_bh >= 0.05
```

The safe post-contact mirror is:

```text
window                 (A_i, A_i + L]
O_i = [o0, o1)         earliest maximal true run in usable_i and the same scene
post_gap                o0 - A_i; the immediately next frame has gap 1
common eligible         len(O_i) >= 5 and post_gap <= G and largest_step_ratio <= 4.0
outgoing                eligible and fitted_decrease_bh <= -0.05
```

`largest_step_ratio` compares each bbox-normalised shuttle step with the median non-zero step in that visible path. The 0.25-BH historical movement and closure thresholds do not belong in the robust 0.05-BH rule. No measurement compares `O_j` and `P_i` across missing frames.

## Implied serve boundary

Infer an unshown or undetected serve only when the recurrence-clean trace reaches the earlier edge of the allowed interval, `max(span_start, scene_start)`, without an accepted origin. A trace stopped by any internal false mask component returns `not enough shuttle trajectory to tell`.

The new helper must retain an explicit reason instead of copying the current information loss. Available source evidence includes:

- half-open rally boundaries from `data.spans`
- half-open tracker scenes from `data.segments`
- missing player attribution or tracker segment at the contact
- shuttle track validity and non-zero coordinates
- court presence
- finite player distance and positive finite bbox height
- recurrence guard status
- the 30-base-30fps local measurement limit

The local 30-frame limit is a measurement window, not an observable rally boundary. Reaching it must continue the trace or return unknown. It cannot support an implied serve by itself.

This interpretation is settled. No behavioural choice remains before implementation.

## Resolved decisions

- Run the same search over all 239 one-to-one rallies; GT is scoring-only
- Keep the existing rally span as a coarse envelope
- Search accepted impulses only
- Move forward when usable post-contact evidence says an accepted impulse lacks outgoing motion
- Stop the main rule when there is not enough shuttle trajectory to tell
- Allow continue-past-unknown only as a labelled sensitivity check
- Treat outgoing motion plus measured absence of incoming motion as a visible serve
- Trace incoming motion backwards before inferring a missing serve
- Allow credible outgoing and incoming traces to connect across invisible TrackNet frames without temporal overlap or spatial joining
- Treat `high_shot_oob` and ordinary TrackNet dropouts the same way
- Restrict reconnection candidates to consecutive accepted contacts
- Let gap contents pass without direction, trajectory-shape, speed, jump, guard, or spatial checks
- Cap the accepted contact gap at 75 base-30fps frames and record the full GT-free gap distribution
- Require positive outgoing and incoming endpoint evidence; an earlier `no outgoing` verdict cannot reconnect
- Do not infer an exact frame for an unshown serve
- Keep the production serve-start condition out of the rule and diagnostics
- Keep raw and rejected impulses out of scope
- Keep all new work under `scratch/serve_id_by_lookback_followup/`
- Run focused tests and checks only; do not run the full repository test suite
- Commit messages are pre-authorised when they use plain language, a clear one-line summary, and at most two supporting paragraphs totalling 100 words
- Public repository material may be shared with Codex external delegates and agy
- Final conclusions require bounded agy Claude Opus and Gemini 3.1 audits
- Writing and voicing reviews must lead with the few main ideas, reveal technical evidence progressively, and use ordinary speech
