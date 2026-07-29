# Detecting invented shuttle positions: the options we have

Written 2026-07-22. This is the decision sheet for flagging invented
content in existing tracks, without waiting for the fill sidecar.

Short version: we have two working detectors for stride=8 tracks and one
for stride=1. All are safe to use now. None of them, and nothing that
reads only the saved track, can catch every invented frame. The sidecar
is the only thing that closes that gap.

## Conventions

- **TrackNet**: the model that locates the shuttle in each frame
- **InpaintNet**: a second model, run after TrackNet, that invents a
  position for any frame TrackNet missed. Its inventions are called fills
- **stride=8** (the code calls it `nonoverlap`): fill windows tile the
  video edge to edge, so each frame is handled once
- **stride=1** (`weight`): fill windows advance one frame at a time.
  Every frame therefore sits inside 16 overlapping windows, and their
  answers get blended. This is what production runs
- **split-half check**: derive the attractors from the first half of a
  video and from the second half separately, then confirm both halves
  name the same ones. A pattern found in only one half is noise rather
  than a fixed model output
- **the fill sidecar**: a proposed second file saved beside every track,
  recording which frames were invented. It does not exist yet. The case
  for it is in `inpaint_fabrications_investigation.md`, "Proposed fix"
- **the cycle**: given an entirely empty window, InpaintNet returns the
  same 16 positions every time. That output is a property of the weights,
  not the footage
- **attractor**: a position or sequence the track returns to far more
  often than footage can explain
- pilot: the test video. 154,393 frames, 512x288, 25 fps

## The situation

|                          | stride=8        | stride=1        |
|--------------------------|-----------------|-----------------|
| detectors available      | 2               | 1               |
| invented share found     | 34.1% to 34.9%  | 35.1%           |
| evidence grade           | proof           | strong suspicion|
| safe to use now          | yes             | yes             |

The grades differ because the two modes leave different traces. At
stride=8 the cycle survives intact. A varying 16-position sequence
recurring hundreds of times is something footage cannot produce. At
stride=1 the blending flattens the cycle to a single constant. A
stationary object can hold one pixel, so the same certainty is not
available.

That is the practical cost of the mode change. The invented share barely
moved, but it stopped being provable.

## Option 1: checkpoint cycle signature (stride=8 only)

Derive the 16 fixed positions by feeding the checkpoint an empty window,
then flag any tiled window whose rows match. The proposal and its
14-of-16 threshold come from `inpaint_fabrications_investigation.md`,
"Proposed fix", Part 1.

- catches 52,624 frames, 34.08%
- needs the checkpoint file, and re-derivation whenever it changes
- does not apply to stride=1, where the cycle never appears intact

Provenance: proposed in the fabrications investigation after the
checkpoint probe reproduced the cycle from the weights.

## Option 2: recurrence detector (both modes)

Find position sequences that repeat exactly at many unrelated moments,
then judge them on whether they move. Needs only the saved track, so no
checkpoint access and no hardcoded coordinates.

How it decides:

- group every 16-frame position sequence by exact equality
- merge occurrences closer together than 32 frames, so one long run or
  one loop zone counts once
- set the threshold at the largest ratio gap in the resulting counts,
  rather than picking a number
- an attractor that VARIES is proof of invention. An attractor that is
  FLAT is suspicion, since a stationary object can produce it

Measured on pilot:

|                              | stride=8    | stride=1    |
|------------------------------|-------------|-------------|
| threshold it derived         | 488         | 418         |
| margin at that threshold     | 122x        | 29.9x       |
| attractors that vary         | 16          | 0           |
| attractors that are flat     | 0           | 1           |
| frames flagged               | 34.90%      | 35.08%      |
| split-half agreement         | exact       | exact       |

On stride=8 it is a strict superset of Option 1: it catches all 52,624 of
those frames plus 1,256 more, because it scans every offset rather than
only the 16-frame lattice. It misses nothing Option 1 finds.

Provenance: built this session. Its structure owes two things to an
external red-team (Codex Sol, high effort, read-only): the split between
proof and suspicion, which an earlier version wrongly collapsed, and
split-half validation as a standing check.

## Option 3: coordinate matching (superseded)

The first sidecar built this session matched the fabricated coordinates
directly. It works, and its numbers agree with the others. But it hard
codes values from one checkpoint, and it offers no reason why those
coordinates and not others. Option 2 replaces it. Kept only because its
output is referenced in earlier console records.

## Why real detections survive

The reason to trust these on live data, stated as the failure each one
cannot produce:

- a **resting shuttle** makes one attractor episode at whatever pixel it
  landed on, never hundreds at the same pixel, so it stays far below any
  threshold
- a shuttle **passing through** an attractor pixel holds it for a frame
  or two, never the full 16, so it forms no occurrence
- **short informed fills**, bridged from nearby real detections, differ
  every time and so never become attractors
- output is **graded**, so a consumer can treat suspicion differently
  from proof rather than inheriting one policy

The loudest genuinely real pattern in the stride=8 track recurs 4 times,
against 488 for the quietest fabrication.

## What none of them can do

Every method here needs the invention to REPEAT. A fill bridged from real
detections is shaped by that evidence, so it differs every time and
leaves no trace to find. The fabrications investigation always described
its 28 to 30% as a floor for this reason.

So the honest shape is that these detectors are safe but incomplete, in
the same way, in both modes. The fill sidecar fixes the incompleteness
rather than the safety. Its case is set out in
`inpaint_fabrications_investigation.md`, "Proposed fix", Part 2.

One measurement would put a number on what is being missed: re-run the
tracker with the InpaintNet argument empty and diff. That needs no code
change, costs about 2h40m per video at stride=1, and yields a
ground-truth fill mask for that video. Not done.

## Scripts, and how far each got

All run on pilot. None is wired into the pipeline; every one is a
standalone script reading a saved track.

| Script | Does | State |
|---|---|---|
| `c11_landing_bisect/probe_inpaint_cycle.py` | derives the cycle from the checkpoint | run on bourbaki, output recorded inline |
| `c11_landing_bisect/probe_vs_track.py` | diffs the probe against saved tracks | run on all three fixtures |
| `stride1_retrack/make_fill_sidecar.py` | Option 3, coordinate matching | superseded by the two below |
| `stride1_retrack/rule_recurrence.py` | Option 2, first cut | superseded: its threshold was hand-set |
| `stride1_retrack/rule_recurrence_v2.py` | adds a derived threshold and recovers short runs | superseded by v3 |
| `stride1_retrack/rule_recurrence_v3.py` | **current**: adds the proof-versus-suspicion split and split-half validation | ready to use, not integrated |

Outputs sit beside the scripts as `*_fillmask.npy` and
`*_recurrence_v3.npy`, one code per frame, with a console transcript
beside each script. `fill_sidecar_manifest.json` carries the provenance
of the Option 3 run: source video, checkpoint md5s and the codes.

v3 writes these codes: 0 no flag (clean, and frames the track stores as
(0,0), which fold into 0), 1 fabricated and proven, 2 flat attractor and
suspect, 3 degraded. Code 2 (suspect flat) should stay usable by
default. Treating it as missing would destroy a genuinely resting
shuttle, which is the one thing these rules must not do.

Renumbered 2026-07-22: no-detection folds into 0 rather than keeping its
own code, and the codes shift down to a 0-3 range. The regenerated v3
console's counts are unchanged, with the old clean and no_detection
counts merged into the new code 0.

## Console transcripts carry pre-move paths

The `*_console.txt` files record runs made before this directory was
consolidated, so the paths they print are the old ones. They were left
verbatim on purpose, since the numbers in them are the evidence this
investigation rests on.
