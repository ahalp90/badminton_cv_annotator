> ARCHIVED 2026-08-12: original findings. Current conclusion: `../../report.md`.

# Findings for the additive correction

## F1. Two measured changes help with no observed damage

The final policy changes 2 of 239 visible-start results. Both changes are right
at +/-10. None damage the baseline.

Visible-start correctness rises from 125 to 127. Separate server correctness
rises from 163 to 164.

## F2. The existing PR #82 result remains the baseline

PR #82 interprets its earliest accepted contact with the original 0.05-BH
incoming rule. Incoming motion means first visible return. Otherwise, the
result remains a visible serve. PySceneDetect bounds the measured path.

Every rally starts with this answer. Weak, absent, unavailable, or ambiguous
new evidence is a no-op.

## F3. Only the measured high-shot predecessor can intervene

The positive state requires all of these saved GT-free fields:

```text
incoming_admission = high_shot_oob
incoming_stop_reason = predecessor_admitted_by_high_shot_oob
incoming_category = visible_serve
incoming_predecessor_verdict = not_incoming
```

Five rallies meet the state. Three nominate the baseline. Two nominate a later
accepted contact. The two interventions fix two starts and damage none.

All five states occur in one set. The result needs a frozen holdout.

## F4. The old 97 slice remains mostly unresolved

The final correction fixes 2 of the 97 GT-unmatched first contacts. It leaves
95 wrong.

PR #82 gets 58 server answers right in this slice. The correction gets 59
right. The slice controls no decision.

## F5. A broad serve-setup differential is unsafe

The existing sticky serve-setup primitive was measured at accepted contacts.
The broad proposal changes to a later passing contact when the first contact
fails the gate.

It fires 138 times. It fixes 22 starts, damages 63, and changes 53 without
making them right. Net change is -41. Intervention precision is 15.9%.

All 22 fixes fall in the old 97 slice. The 63 damages fall outside it. GT cannot
be used to choose the favourable slice.

## F6. Continued same-player setup is still not safe

A stricter state requires the same player at both contacts, close wrist
evidence after the early impulse, a fixed short lookback, and measured
not-incoming motion at the later contact.

It fires twice. It fixes one start and damages one. It changes no server answer.

Wrist proximity can make a later contact plausible. It does not prove that the
early impulse was false.

## F7. Ordinary predecessor timing remains weak

The ordinary rule admits 196 predecessors. Thirty-nine have measured
not-incoming motion, but only 3 match GT contact 1 at +/-10.

Requiring the predecessor and anchor players to differ gives 26 candidate
changes. It fixes 3, damages 10, and leaves 13 wrong. Nineteen predecessors
also agree with the existing server.

## F8. The other combinations remain no-ops

Nine relaxed first-contact incoming calls fix 4 starts and damage 2. Their
server flips fix 5 and damage 4.

Six later incoming anchors with no admitted predecessor fix 4 starts, damage 1,
and leave 1 wrong.

Dropping rank 1 when the later H3 evidence, alternating fit, and existing
server agree changes 24 starts. It fixes 7, damages 9, and leaves 8 wrong.

These states can report conflicts. They cannot override the baseline safely.

## F9. Unavailable and outgoing evidence do not replace the result

`Unavailable` means the path cannot be measured. It is a no-op.

The outgoing-first replacement finds 26 fixes and 13 damages. Another 148
rallies have unavailable pre-contact evidence or no credible outgoing contact.
The outgoing gate remains unused.

## F10. GT scores frozen decisions

The final analysis projects GT-free inputs for all 239 rallies. It freezes each
action before loading stroke frames or server labels.

The final policy produces 237 `NO-OP`, 2 `CORRECT`, and 0 `CONFLICT` rows.

## F11. The wider state is not identifiable from the tested fields

TrackNet paths, accepted impulses, player attribution, alternating order,
sticky pose setup, and scene bounds do not give a safe general test for an
early false contact.

A plausible next measurement is direct physical-stroke evidence at a candidate
frame. Local RGB racket or hitting-arm motion is available. The frozen 288p
fixture videos have no audio stream.
