# Commit 11 and the landing collapse: the plain story

Written 2026-07-22. This is the human-readable companion to
findings.txt (the terse evidence ledger in this folder). Everything
here was verified this session; each section names its artefact.

## The one-paragraph version

Commit 11 made the landing score collapse, and that looked like a
regression. It is not. Nearly half of the old landing score came
from the shuttle tracker inventing a shuttle where it had lost the
real one. The old code turned that invention into a fixed guess
("the shuttle landed in the far half"), and the guess was lucky
often enough to inflate the score. Commit 11 removes the evidence
that kept the guess alive, so those rallies now honestly return no
answer. The counts below back each step.

## Background: how a landing gets called

The pipeline watches a badminton broadcast and, for each rally,
tries to say where the shuttle landed. It searches the shuttle's
track after the rally's last hit, looking for a falling motion. Two
rules bound that search:

- the settle cap: once the shuttle has clearly come to rest, stop
  searching. Anything that moves later is a person tidying up, not
  the rally. "At rest" is decided from the track alone. The
  shuttle's frame-to-frame movement is smoothed by a rolling median
  over 7 frames; at or below 0.004 of the picture per frame it
  reads as still. Five still frames in a row mark the resting
  point, and the search stops there. (Numbers are for 30 fps
  footage and rescale with frame rate)
- the held exception: a shuttle sitting still NEXT TO A WRIST is in
  someone's hand, not resting on the floor. Concretely, a still
  frame does not count toward the five when the nearest wrist sits
  within 0.75 of a body height of the shuttle. One refinement: a
  shuttle nearer an ankle than any wrist reads as on the floor by
  someone's feet, so the frame counts after all. Without that, a
  player standing over a fallen shuttle would block the cap forever

A third rule exists and is easy to confuse with these: the carry
filter. It discards a candidate fall whose endpoint stayed close to
a wrist over a trailing window, reading it as a hand lowering the
shuttle. It plays no part in this story; the controlled rerun below
showed it contributed nothing to the collapse.

The held exception needs to know where people's wrists are. Before
commit 11, it looked at every person detected inside the court
rectangle: players, but also ball kids and anyone wandering through.
That crowd of candidates is called the pool. Commit 11 switches the
evidence to the sticky tracker, which follows exactly one player per
court half. Fewer, cleaner candidates; that was the whole point of
the commit.

## What the scores did

How the landing score is measured, first. Each ground-truth rally
records where the shuttle actually landed. The scorer reduces both
sides to one binary question per rally: did the predicted landing
fall in the same court half (near side or far side) as the true
landing? There is no distance margin, and the exact position is
never scored. A rally with no prediction counts as wrong. Pilot has
113 scored rallies, so a score of 0.398 means 45 of 113 matched
halves.

The scored prediction is the chain's final answer. Every rule in
this report has already acted by then: the off-screen window rules,
the settle cap with its held and ankle exceptions, and the carry
filter. Two of those were checked separately for this movement. The
carry filter contributed nothing (the controlled rerun proved it),
and the top-exit check found no lob landings lost.

The capture is the recorded scoring run over three test videos; the
main one is called pilot. After commit 11 it showed (artefact:
c11_capture_repro_20260722.txt, reproduced exactly this session):

- pilot's landing score halved, 0.398 to 0.195 (45 correct to 22)
- the other two videos dipped slightly
- contact precision (the share of detected hits that are real hits)
  fell a few points on all three videos

## Finding one: the tracker invents shuttles

The shuttle tracks were produced by TrackNetV3 with a gap-filling
model called InpaintNet. When the real shuttle is untrackable,
InpaintNet fills the gap with an invented position. The filled
frames are flagged as if the shuttle were seen.

Those fills are not subtle. In each video the filler falls into one
fixed pattern: a fake shuttle bobbing through the same 16-frame loop
at the same screen spot. The identical loop repeats at hundreds of
separate places per video. A real shuttle cannot revisit identical
pixel positions on a fixed rhythm, so these spans are provably
invented. The loop alone covers 28 to 30 percent of the frames in
every one of the three test videos (census in findings.txt). That
is a floor, not a total: only the loop's repetition betrays a fill,
so quieter fills cannot be counted from this cache.

Where the loop sits, precisely. The track stores positions as
fractions of the video frame, so the loop lives in screen space,
not court space. Horizontally it sits mid-frame, near 0.475 of the
frame width, wandering by under two percent between spans.
Vertically it bobs between 0.233 and 0.292 of the frame height, in
the upper middle of the picture. The 16 vertical values are
bit-identical in every span, and all three videos share the same 16
values. One fixed pattern across three different broadcasts can
only come from the shared gap-filling model, not from the footage.

## Finding two: how invented shuttles earned points

At many rally ends the real shuttle becomes untrackable, and the
track jumps into an invented loop. Three things then happen.

First, the jump itself looks like a hit, so the pipeline often
records it as the rally's final contact. Second, the loop's slow
bobbing looks like a small fall, so the landing search finds a
"descent" inside it. Third, the loop's screen position maps through
the court geometry to a point 6.0 to 7.3 metres BEYOND the far
baseline (the loop's small horizontal wander makes three nearby
variants of that point). The scoring only asks which half the
shuttle landed in, so an impossible far-side point still counts as
an answer: TOP.

The result: on every such rally the old pipeline answered TOP. The
scored ground-truth comparison (artefact: gt_join_summary.txt)
proves the point. Of the 38 rallies commit 11 affected, the old
chain said TOP on all 38 and BOT on none. Ground truth happened to
be top-side on 21 of them, so 21 scored as correct. A fixed answer
that lucks into 21 points is a guess, not a measurement.

## Why the old code kept the guess and commit 11 drops it

The settle cap should have ended these searches almost immediately,
because an invented loop reads as a shuttle at rest. The held
exception blocked it. Under the pool, some bystander's wrist was
usually near the loop's screen spot. Frame after frame therefore
read as "held", and the cap stayed silent. The cap was effectively
dormant on the whole pilot video.

Under commit 11's sticky evidence there is no bystander to lean on.
The two tracked players are elsewhere, or the tracker has already
reset at the scene cut. No wrist evidence means not held. The cap
fires a few frames after the fake contact, and the search window
vanishes. The rally returns no landing. That is the entire
collapse: 26 no-answer rallies became 64, and all 38 new ones died
at the settle cap (artefact: bisect_per_rally.csv).

## The two off-screen guards still stand, with one caveat

The landing window carries two older guards, designed for shots
that leave the picture. If the shuttle disappears for a sustained
stretch, the window closes: a shuttle that never came back cannot
be searched. There is one exception: when the last visible sample
before the disappearance sat at the very top of the frame, the
shuttle was lobbed out of the picture and will fall back in, so the
window stays open for the re-entry. Commit 11 changes neither
guard.

Both guards read the shuttle's visibility flag, and that is the
caveat: the gap-filler overwrites true invisibility with invented
"visible" positions. On pilot, a fill span long enough to matter
appears before any real disappearance in 78 of the 102 windows.
There the old close never came, which is exactly where the invented
landings lived. Under commit 11 the settle cap closes those windows
instead, since an invented loop reads as still and unheld.

The re-entry case was checked directly. Across all 64 rallies that
now return no landing, not one has a fall from the frame top after
the settle cap fired. On pilot, the earlier cap cost no lob
landings. The guards also still see real events: 61 of the 102
windows contain a genuine disappearance, and 16 of those follow a
top-edge exit.

## How solid is this

What was measured where, first. The score movement and the
invented-fill census cover all three test videos. Everything
per-rally (the controlled rerun, the ground-truth comparison, and
the off-screen guard checks) ran on pilot only. The other two
videos show the same score direction and the same fill pattern, but
their rallies were not traced.

The chain was checked four ways this session:

- the capture rerun reproduced the overnight numbers exactly
- a controlled rerun reproduced the collapse. It scored the video
  twice, changing only the wrist-evidence source, with everything
  upstream bit-identical. The two runs are called arms below
- an external red-team (Sol, high effort) confirmed the mechanism
  and the fabrication census. It corrected two overclaims and
  refused the interpretation until real counts existed
  (sol_redteam.txt)
- the scored ground-truth comparison supplied the counts. Its
  built-in check passed: the commit-11 arm scored exactly the 22 of
  113 the capture scored

One number stays unattributed. The old full chain scored 45, and
the controlled rerun's old-evidence arm scored 42. That 3-call gap
belongs to commit 11's other changes: who gets credited with the
hit, and which detected hits count at all. Pinning it would need an
experiment that switches each change on and off separately.

## What this means for the ruling

The decision on the table is whether to accept commit 11 into the
branch (the options live in pickup.md). The counts change the
flavour of that decision:

- of the 23 correct calls the capture loses, 21 were the lucky
  fixed guess. Real lost capability is at most 2 calls
- nothing in the lost rallies is recoverable by tuning; there were
  no real landing calls there to save
- the deeper problem is upstream and untouched by any option: at
  least 28 to 30 percent of every track is provably invented
  filler, and it feeds all shuttle measurements, not just landings.
  It is cheap to detect, because the 16-frame rhythm is a
  signature. Nobody has yet scoped masking it
- the fake final contacts themselves also remain unmitigated. The
  jump into a fill passes the contact gate in both the old and new
  chains, so it can still be recorded as a rally's last hit. The
  last hit drives more than the landing search: it also drives who
  gets credited with the stroke, and through that the rally's
  winner call. No rule anywhere yet knows fabricated spans exist.
  This belongs to the same upstream scoping as the filler itself

Two genuine code defects in the commit also surfaced during review,
and they ride with any landing. One is an ordering bug in the new
landing evidence code: a missing measurement can hide a valid one,
depending on list order. The other is an unstated tie rule that
silently prefers one court half when both players sit equally close
to the shuttle. Both are listed in pickup.md's landing checklist.
