# Inpaint fabrications: invented shuttle positions in our tracks

Written 2026-07-22. Status: mechanism fully confirmed, from the saved
arrays down to the model weights. Fix proposed at the time; the sidecar
half of it has since shipped (commit `9475036`). See
`docs/tracknet/inpaint_sidecar.md` for the current producer contract and
`docs/tracknet/inpaint_sidecar_consumption.md` for consumer state.

- [The short version](#the-short-version)
- [Conventions](#conventions)
- [What is happening](#what-is-happening)
- [Why it matters](#why-it-matters)
- [How it was pinned down](#how-it-was-pinned-down)
- [Where it has been validated](#where-it-has-been-validated)
- [Proposed fix](#proposed-fix)
- [Weight-mode re-track](#weight-mode-re-track)
- [Open questions and unknowns](#open-questions-and-unknowns)
- [Artefact index](#artefact-index)

## The short version

About a third of the shuttle track in each of our three test videos
is invented. Wherever the shuttle was untrackable, the tracking
chain's gap-filler wrote the same fake bobbing pattern into the gap
and flagged it as genuinely seen. We have reproduced that exact
pattern from the filler model's own weights, so the mechanism is
closed end to end. The proposed fix has two parts. A signature filter
flags the fabrications in the tracks we already have. A fill-mask
sidecar, a second small file saved beside every future track, keeps
the honest fill flag alive from birth.

## Conventions

- fixtures: the three test videos the heuristics are scored on:
  pilot (video 1, Momota vs Chou, Fuzhou Open 2019 finals, 154,393
  frames), vid15 (149,487 frames), and sset21
- track: one array per video, one row per frame, holding x, y and a
  visibility flag (1 means "shuttle seen here"). Positions are stored
  as fractions of the frame
- TrackNetV3: the shuttle detector that produced the tracks.
  InpaintNet: its companion gap-filler, a small model that invents
  positions for frames the detector missed
- window: InpaintNet processes the track in 16-frame chunks. In the
  mode our fixtures used (nonoverlap), windows tile the video edge to
  edge starting at frame 0
- checkpoint: the saved file of learned weights and settings that a
  training run produces. The filler's window length lives inside it,
  not in the code
- the heuristics: our downstream rules that read the tracks to call
  rally events: the final contact (last hit), the landing position,
  and the point winner
- code citations (file:line) pin to the wt_annotator worktree at
  commit d04a789. Records and scripts live in
  inpaint_fabrications_investigation/c11_landing_bisect/ under this directory

## What is happening

InpaintNet sees no video. Its whole input is the detector's
coordinate sequence plus a mask saying which frames need filling
(model.py:113-128). When every frame of a 16-frame window is missing,
that input is all zeros plus an all-ones mask: zero evidence. A fixed
model given a fixed input returns a fixed output, so every fully
missing window, in every video, gets the same 16 invented positions.

Those 16 positions trace a small bobbing loop in the upper middle of
the frame: x near 0.475 of frame width, y spanning rows 67 to 84 of
the 288-row tracking grid. Because windows tile from frame 0, the
loop always lands on the same 16-frame lattice: the fabrications are
phase-locked. The filler then flags every fill as visible
(predict.py:69). The mask that knew which frames were fills is
dropped one line after its creation (predict.py:260-261). In the
saved arrays a fabricated position is therefore indistinguishable
from a real one.

The scale: repeating-loop spans cover 28 to 30% of frames in each
fixture, and that is a floor. Only the loop's repetition betrays a
fill, so gap-edge fills that mix in real evidence go uncounted.

## Why it matters

The heuristics read these tracks, and the fabrications sit exactly
where rallies end: that is when the shuttle becomes hard to track.
Three consequences are measured, all on pilot (findings.txt):

- fake final contacts. The track's jump into the loop reads as a hit.
  All 38 rallies flagged in the landing investigation had their final
  contact phase-locked to the fabrication lattice
- invented landings. The loop's screen position projects 6.0 to 7.3
  metres beyond the far baseline, so the landing rules answered "far
  half" on every one of those rallies. 21 of their 45 correct landing
  calls on pilot were that fixed guess coming up lucky
- a disabled safety guard. One guard abandons a landing search when
  the shuttle leaves the picture. It reads the visibility flag, and
  fills overwrite true invisibility. In 78 of pilot's 102 post-rally
  search windows a fill span pre-empted the guard

Beyond landings, the last hit decides stroke credit and, through
that, the point-winner call. No rule anywhere in the chain currently
knows fabricated spans exist.

## How it was pinned down

Four independent lines of evidence, each covering the previous one's
blind spot.

1. Track measurements (findings.txt, measured 2026-07-22). The
   repeating 16-value cycle was found in the saved arrays: pilot has
   375 repeating zones, vid15 446, sset21 250. Sorted, the 16 y
   values are bit-identical across every zone of all three videos.
   One pattern across three different broadcasts can only come from
   the shared model, not the footage

2. Code trace (inpaint_source_findings.md, commissioned 2026-07-22
   and audited line by line against the pinned tree). The mechanism
   above is cited to source: the mask's creation and loss, the
   visibility overwrite, the window tiling, the evidence-free input.
   The trace also closed three side questions: fills have no length
   cap, they happen before positions are rounded to whole pixels, and
   the batch entry point (the script that processes many clips in one
   run) uses the same inpaint code

3. Run-log arithmetic (same document, question 1). The window length
   is not in the code; it lives inside the checkpoint. The recorded
   rerun logs pin it anyway: the inpaint stage's batch counts (604 on
   pilot, 584 on vid15) fit a window length of exactly 16 and no
   other value

4. Checkpoint probe (probe_inpaint_cycle.py, run on bourbaki
   2026-07-22). The production checkpoint, fed the evidence-free
   input, printed window length 16 and the loop itself. Diffed
   against all three saved tracks (probe_vs_track.py): across every
   window carrying the loop (3,289 on pilot, 3,419 on vid15, 2,080
   on sset21), y matches at all 16 positions and x at 15 of 16. The
   one exception is the same single pixel in all three videos: the
   production runs (on a GPU) and the probe (on a CPU) round one
   value to neighbouring whole pixels

That last step closes the chain: the bytes in our saved tracks are
the weights' fixed answer to "no information", reproduced on demand.

## Where it has been validated

| Check | pilot | vid15 | sset21 |
|---|---|---|---|
| Repeating-cycle census | yes | yes | yes |
| Cross-video identical cycle | yes | yes | yes |
| Per-rally harm tracing | yes | no | no |
| Window length from run logs | yes | yes | no |
| Checkpoint probe diffed against track | yes | yes | yes |

The mechanism claims rest on all three videos. Per-rally harm
tracing means scoring each rally's final contact and landing call
against the human-recorded ground truth, and that was done on pilot
only. Vid15 and sset21 show the same cycle and the same drop in
landing accuracy, but their rallies were not traced individually.

## Proposed fix

Two parts, and the principle is flag, not delete. Short fills bridged
from nearby real detections are informed interpolation worth keeping;
fully missing windows contain zero evidence. Each downstream rule
should see the flag and decide for itself.

Part 1, retrofit for existing tracks. The loop's 32 numbers (16 x, 16
y) come from the weights, not the footage, so one signature covers
every video made with this checkpoint in nonoverlap mode. The rule
works per window: compare the window's 16 y rows against the probe
cycle position by position, and 14 or more matches marks the whole
window fabricated. A real shuttle cannot hit 14 exact integer rows in
the exact order on the exact lattice, so the rule cannot falsely flag
a real shuttle. On pilot this marks 3,289 of 9,649 windows
(34% of frames). Windows containing even one real detection produce
video-dependent fills, never the loop, so the rule cannot touch them:
the informed fills survive untouched.

Part 2, sidecar for all future tracks. The extraction chain holds a
correct fill-vs-detection mask for one line before discarding it
(predict.py:260-261). Saving that mask as a separate file beside the
track, one entry per frame, costs one write and changes nothing
else: the existing arrays stay byte-identical. Consumers then get
ground truth about provenance instead of signature reasoning.
Feasibility was checked in the code trace; the only care needed is
writing it with the same frame ordering as the track writer.

What consumers do with the flag is per-rule design work, not yet
done. Candidates:

- the contact gate (the rule that accepts a track jump as a hit)
  could refuse a final contact that lands on filled frames
- the landing search could require real detections for the falling
  motion it looks for
- the lost-shuttle guard could treat a filled run as the invisibility
  it really is
- speed and direction measurements can keep short informed fills

Sketches, assuming a per-frame boolean `fill_mask` (true means
invented), sourced from the sidecar or the signature filter:

```
# contact gate: a final contact on invented frames is no contact
if fill_mask[contact_frame]:
    reject_final_contact()

# landing search: a descent built on invented frames is no descent
if fill_mask[descent_start:descent_end].any():
    discard_candidate()

# lost-shuttle guard: invented frames are really invisible
effective_visibility = visibility & ~fill_mask

# speed and direction: keep only short, informed fills
run_length = fill_run_length_per_frame(fill_mask)
usable = ~fill_mask | (run_length < 16)
```

These are shapes, not designs; each rule's real threshold and window
belongs to its own change.

## Weight-mode re-track

A follow-up re-track ran pilot in weight mode to test the expectation
above. It confirmed the expectation, and the result is worse than it
first looked.

Two terms this section uses: **stride=8** is another name for nonoverlap
mode and **stride=1** for weight mode, after the step size. An
**attractor** is a position or sequence the track returns to far more
often than footage can explain; a **split-half check** derives the
attractors from each half of a video separately and confirms both halves
name the same ones.

Weight mode did not remove the invented positions; it flattened them.
The 16-frame bobbing loop is gone, but a single fixed point at 243 x 71
pixels took its place, holding 55,267 frames (35.8% of the video, up
from 0.0006% on the nonoverlap track). Three lines of evidence tie that
point to the filler (`stride1_retrack/summary.txt`):

- the blending weights (a symmetric tent) applied to the nonoverlap
  cycle predict it: the weighted mean is x=243.39, y=70.94 pixels, and
  the stored cycle values were already truncated, so the honest
  prediction is an interval (x of 243 or 244, y of 70 or 71) that
  contains the observed point
- the opening frames show the blending buffer filling: y runs 81, 77,
  75, 73, 72, 71 pixels, then locks onto the constant from frame 15,
  where the code switches from a partial to the full blend
  (`predict.py:312-315`)
- of the frames proven invented on the nonoverlap track, 81.5% now hold
  the constant; only 15.4% became an honest no-detection

Measured with one instrument, the invented share barely moved: 34.1% at
nonoverlap against 35.8% at weight mode. What changed is whether it can
be proven. A varying 16-position sequence recurring hundreds of times is
proof, because footage cannot reproduce it by chance. A flat constant is
not, because a stationary object, or TrackNet locking onto a static
image feature, can hold one pixel just as well. A per-track recurrence
detector (`rule_recurrence_v3.py`) that derives its own threshold and
passes a split-half check on both tracks found 34.9% of nonoverlap
frames provably invented against 0.0% of weight-mode frames; the
remaining 35.1% of weight-mode frames are merely suspect, not provable.
Weight mode converts proof into suspicion: a regression in
detectability, not an improvement.

Part 1 above (the signature retrofit) works, and is measured, at
nonoverlap. It cannot give the same certainty at weight mode, because no
rule reading only x, y and visibility can separate a filled frame from a
real detection that happens to land on the same pixel. That moves Part 2
(the fill-mask sidecar) from a good idea to the only sound option for
the mode production actually runs, and it earns its keep twice over: the
mask is derived from TrackNet's own gaps, so saving it also restores the
honest visibility the lost-shuttle guard needs. That sidecar has since
shipped; see the status note at the top of this document.

Two detectors were built from this re-track and are described in
`detector_options.md` with their measured performance and limits. Both
are safe to use now. The current one discovers the fabricated pattern
from each track by improbable exact recurrence, hardcodes no
coordinates, needs no checkpoint access, and works in both modes. A
per-frame sidecar, one code per frame, was also produced for both tracks
so downstream rules can grade rather than delete.

The re-track ran in 2h40m, roughly 8x nonoverlap's cost at this 512x288
resolution; an earlier 21-hour estimate assumed full-resolution decode
overhead that does not apply here.

## Open questions and unknowns

- weight mode. Our fixtures used nonoverlap mode, but production
  tracking uses weight mode. The difference is step size. Nonoverlap
  advances a whole window per step. Weight mode advances one frame,
  so windows overlap. Both stages step this way: TrackNetV3's
  8-frame detector windows and InpaintNet's 16-frame fill windows.

  The bobbing signature should not survive there, for a reason worth
  spelling out. With one-frame steps, every frame sits inside 16
  fill windows at once, at a different position in each. Deep inside
  a long gap all 16 of those windows are fully missing, so each
  emits the fixed cycle. The saved value becomes a weighted blend of
  all 16 cycle values, and that blend is the same for every deep-gap
  frame: one constant position, no rhythm. A constant position is
  also what a genuinely resting shuttle produces, so that signature
  would be ambiguous.

  ANSWERED 2026-07-22: a weight-mode re-track confirmed every part of
  this expectation. See
  [Weight-mode re-track](#weight-mode-re-track) below
- the uncounted fills. Gap-edge fills are video-dependent and carry
  no signature. The 28 to 30% is therefore a floor, and only the
  sidecar can mark the remainder. How much they matter is unmeasured
- consumer policy. Which rules ignore flagged frames, and which keep
  them, is undecided (see Part 2 above)
- the other two fixtures. The fabrications themselves are directly
  measured on vid15 and sset21: the census and the probe diff both
  cover them. What was never traced there is the harm accounting:
  how many of their contacts and landings sit on fills. Pilot's
  per-rally counts have no counterpart on the other two
- destroyed evidence. In some windows the filler may have overwritten
  the invisibility signal entirely. Only a re-track with the filler
  switched off could show what those windows looked like before
  filling (about 22 minutes per video at 288p, per the track cache
  README)
- a different checkpoint means a different loop. Retraining or
  swapping the checkpoint changes the invented positions. The probe
  script re-derives the signature in seconds, but it must be re-run
  per checkpoint
- fake final contacts are still live. When the track jumps from real
  detections into a fill, that jump can register as the rally's last
  hit, and no current rule rejects it. The flag makes rejection
  possible (the contact-gate sketch above); designing and building
  that rejection is future work
- the quiet fills, unmeasured. Re-running the tracker with the
  InpaintNet argument empty and diffing would give a ground-truth fill
  mask for one video, with no code change, at about 2h40m in weight
  mode. Until that runs, how much invented content escapes every
  signature method is unknown in both modes

## Artefact index

All under inpaint_fabrications_investigation/c11_landing_bisect/ unless noted:

- inpaint_source_findings.md: the full code-level investigation,
  eight questions answered with file:line citations
- findings.txt and c11_landing_report.md: the landing investigation
  where the fabrications were discovered
- probe_inpaint_cycle.py: the checkpoint probe (bourbaki), recorded
  output inline
- probe_vs_track.py: the probe-versus-track diff, recorded result
  inline
- ../pilot_track_npy/1.npy: the pilot track cache the measurements
  ran against, with its provenance README

Added 2026-07-22, under this directory:

- detector_options.md: the decision sheet for flagging invented content
  in tracks we already have
- stride1_retrack/: the weight-mode re-track and everything measured
  from it, including summary.txt
- briefs/: the two commissions that drove the work
