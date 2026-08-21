> ARCHIVED 2026-08-12: original second-iteration report. Current narrative: `../../report.md`.

# Two measured corrections help, but no wider dependable rule was found

Trajectory evidence improves the existing result in two measured cases with no
observed damage. It cannot yet recover the wider set of GT-unmatched earliest
contacts safely.

The final correction changes 2 of 239 visible-start results. Both changes are
right at the main +/-10 tolerance. None damage the existing result. The
visible-start score rises from 125 to 127.

The separate PR #82 server score rises from 163 to 164. Only one corrected
start changes the predicted server.

The two fixes are useful stitching. They are not the missing general rule.

## Final correction result

The existing PR #82 answer remains unchanged unless the measured
`high_shot_oob` state supports a different accepted contact as the visible
serve.

| +/-10 visible-start outcome | Rallies |
| --- | ---: |
| Baseline correct, no change | 125 |
| Baseline wrong, no change | 112 |
| Fixed by correction | 2 |
| Damaged by correction | 0 |
| Conflict or unresolved | 0 |
| Total | 239 |

| Headline measure | Result |
| --- | ---: |
| Interventions | 2 |
| Fixes | 2 |
| Damage | 0 |
| Net fixes | +2 |
| Intervention precision | 100% (2/2) |

Two cases are not enough to establish a general precision rate. All five
supporting `high_shot_oob` states occur in `sset_21`, video 21, set 1.

## The baseline already uses the useful PR #82 motion rule

PR #82 starts from the earliest accepted contact. Its original 0.05-BH
incoming rule interprets that frame as the first visible return when the
shuttle moves into the player. Otherwise, it keeps the frame as the visible
serve. PySceneDetect court-scene bounds limit the measured path.

This visible-start interpretation gets 125 of 239 results right:

- a visible serve must align with GT contact 1
- a first visible return must align with GT contact 2

PR #82 also supplies the separate 163/239 server-attribution baseline. The new
layer starts from both results. It keeps the baseline rule and overrides two
outputs.

## Final correction policy

The correction receives the frozen PR #82 result and saved H3/R8 fields. It
does not receive GT.

```text
answer = existing PR #82 result

if a measured high_shot_oob state admits the predecessor
and that predecessor has measured not-incoming motion:
    candidate = predecessor as visible serve

    if candidate differs from the existing result:
        CORRECT
    else:
        NO-OP
else:
    NO-OP
```

Ordinary timing, unavailable evidence, and the outgoing-first result cannot
fire the correction. The saved schema produces no conflicts in this run.

The analysis projects only GT-free fields into 239 input rows. It freezes all
decisions before loading stroke frames or server labels. GT then scores the
saved decisions.

## The old 97 unmatched starts

All 97 rows have a first accepted frame that misses every GT contact at +/-10.
That does not mean all 97 server answers are wrong. PR #82 already gets the
server right in 58 of them.

| Measure within the old 97 | PR #82 baseline | After correction |
| --- | ---: | ---: |
| Correct visible starts | 0 | 2 |
| Wrong visible starts | 97 | 95 |
| Correct server attributions | 58 | 59 |
| Wrong server attributions | 39 | 38 |

The 97-row slice controls no decision. It only reports what happened after the
GT-free policy ran over all 239 rallies.

## Serve-setup evidence does not recover the 97 safely

The existing sticky pose data offered one plausible unused signal. The
serve-setup primitive checks player presence and shuttle-to-wrist proximity in
a fixed lookback. The follow-up measured it at every accepted contact.

`ServeStartConfig` has no default distance threshold. The check therefore used
the existing 1.4-BH accepted-contact wrist threshold. It used the fixed
serve-start lookback. It did not sweep thresholds or enable a new stillness
threshold.

### Broad gate differential

The broad proposal changes to the first later contact that passes the setup
gate when the first accepted contact fails it.

| Result | Rallies |
| --- | ---: |
| Interventions | 138 |
| Fixes | 22 |
| Damage | 63 |
| Changed but still wrong | 53 |
| Net fixes | -41 |
| Intervention precision | 15.9% (22/138) |

Within the old 97 slice, this proposal finds all 22 fixes. It also changes 46
rows without making them right. The 63 damaged starts sit outside that slice.
Using the GT-defined slice to avoid that damage would violate the experiment.

A failed gate at the early frame is absence of setup evidence. It is not proof
that the early impulse is false. A passing gate at a later rally contact also
does not prove that contact is a serve. Ordinary rally contacts can put the
shuttle near a wrist.

### Continued same-player setup

A stricter state tests the concrete early-impulse theory. It requires:

- the later contact to fall inside the fixed setup lookback
- the same player to own both accepted contacts
- the later contact to have measured not-incoming motion
- exactly one player to retain close wrist evidence after the early impulse
- that player to own the later contact

This state fires twice. It fixes one start and damages one. It changes no
server answer. The one fix lies in the old 97 slice. The damage lies outside
it.

The extra pose evidence makes the theory more specific. It does not make the
intervention dependable.

## Why the other evidence stays a no-op

### Ordinary predecessor timing

The ordinary 60-frame rule admits 196 predecessors. Of the 39 with measured
not-incoming motion, only 3 are GT contact 1 at +/-10.

Requiring the predecessor and anchor players to differ does not rescue the
rule. The resulting 26 candidate changes fix 3, damage 10, and leave 13 wrong.
Nineteen of the 26 predecessors agree with the existing server. The other seven
contradict it. Changing those seven server answers fixes 1 and damages 6.

### Relaxed incoming evidence at the same frame

H3/R8 makes nine first-contact paths newly incoming. Using them as overrides
fixes 4 visible starts, damages 2, and leaves 3 wrong. Their server flips fix 5
and damage 4.

The relaxed path check improves availability. It does not make the direction
call strong enough to override PR #82.

### Later incoming anchor with no admitted serve

Six later incoming anchors agree with the existing server and have no admitted
predecessor. Treating them as first visible returns fixes 4 starts, damages 1,
and leaves 1 wrong. This state remains an unresolved conflict, not a correction
rule.

### Dropping the first accepted contact

A one-step combination used the later H3 evidence, a better alternating-player
fit, and agreement with the PR #82 server. It changed 24 starts. It fixed 7,
damaged 9, and left 8 wrong.

Alternation can show that the accepted sequence is inconsistent. It cannot
prove which of the first two impulses is false.

### Unavailable evidence

`Unavailable` means the local shuttle path cannot be measured. It does not mean
the contact is wrong. The correction keeps the existing result.

### Outgoing-first search

The H3/R8 outgoing-first replacement finds 26 fixes and 13 damages. Another
148 rallies end with unavailable pre-contact evidence or no credible outgoing
contact. The outgoing gate remains excluded.

## The two supported corrections

| Rally | Existing start | Corrected start | Server effect |
| --- | --- | --- | --- |
| `sset_21:21:set1:17` | 26641, GT-unmatched | 26687, rank 3, GT contact 1 at -1 frame | Top to Bot; fixes server |
| `sset_21:21:set1:31` | 43041, GT-unmatched | 43078, rank 2, GT contact 1 at -1 frame | Bot stays Bot |

Three other `high_shot_oob` states nominate the existing first accepted frame.
They are evidence-backed no-ops.

## Answer

The current TrackNet, pose, accepted-contact, alternating-player, and scene
signals do not identify the wider early-false-contact state safely. They can
support the two high-shot corrections, but no tested combination gives the
precision needed to recover the 97.

PySceneDetect still helps by bounding trajectory evidence. A scene boundary
does not prove an exact contact. The pose setup gate also helps span opening,
but it does not verify a later serve contact.

The tested fields do not recover the wider 97 safely. A plausible next
measurement is direct physical-stroke evidence from local RGB motion, such as
racket or hitting-arm motion around the shuttle. The frozen 288p videos contain
no audio stream, so audio contact peaks are not available here.

That would be a new contact-verification experiment. It is outside this
correction pass. The present correction should remain high-shot-only and be
tested on a frozen holdout before production use.
