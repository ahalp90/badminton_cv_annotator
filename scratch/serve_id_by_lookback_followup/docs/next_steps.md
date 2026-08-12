# What to test next

## 1. Freeze the preferred server rule

Use the current rule without changing its path checks or fallback:

```text
find the earliest accepted contact with believable motion away afterwards

if its before-contact path shows motion towards the player:
    choose the other player as server
elif its before-contact path is usable but not incoming:
    choose the contact player as server
else:
    keep the PR #82 server answer
```

Do not replace the fallback with the 171 sensitivity before the first unseen
test. Do not add the curved-path proposal.

## 2. Report three answers

Report all three because each catches a different failure:

| Output | Correct when |
| --- | --- |
| Server side | Predicted side matches the annotated server side |
| Visible start | Serve claim matches contact 1, or return claim matches contact 2 |
| Joint | Both answers are correct |

Also report paired changes from PR #82:

- Fix: new rule correct, PR #82 wrong
- Damage: new rule wrong, PR #82 correct
- Both correct
- Both wrong

The development reference is 170/239 server sides, 132/239 visible starts and
117/239 joint. The paired server changes are 20 fixes and 13 damages.

## 3. Show the population narrowing

When the data allow, show both the one-to-one population and a broader
end-to-end population:

```text
all annotated rallies
    -> rallies covered by a prediction
        -> one predicted span to one annotated rally
            -> direct trajectory decisions
```

A gain on the last group may disappear when segmentation and coverage failures
are included.

## 4. Keep timing ideas separate

The high-shot predecessor rule remains a frozen timing hypothesis. All five
development examples came from one set. Report how often the state appears
before reporting its precision.

If the curved-path proposal is tested, apply it to every usable unseen path.
Do not select known errors first. Record the old and new timing and server calls
for each intervention.

## 5. Add a new observation for a larger timing gain

The existing work has already combined contact order, two-dimensional shuttle
distance, player alternation, wrist proximity and scene bounds.

If exact start timing remains inadequate, test local RGB evidence of a physical
stroke. Racket or hitting-arm motion directly addresses whether a hit happened.
The frozen fixture videos contain no audio stream, so contact sound is not
available here.
