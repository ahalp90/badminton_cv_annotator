# Validation overlay

Renders chosen frame spans of a video with an algorithm's per-frame output
drawn on top, so you can check the algorithm by eye instead of by number.

Point it at a video, an array of per-frame positions, and a CSV of frame spans.
It gives you back one mp4: each span with the tracked position boxed and
labelled, a couple of seconds of context either side, and a black gap between
spans.

## Why it exists

The shuttle tracker guesses a position for frames where it cannot see the
shuttle. That guessing stage was inventing a shuttle over roughly a quarter of
every video. Nobody noticed for weeks, because everyone was reading numbers
rather than looking at frames. Thirty seconds of rendered output settles
questions that days of number-work approach sideways.

## Quick start

The shuttle-track overlay, run from the repo root:

    PYTHONPATH=src python -m annotator.validation_overlay.overlays.shuttle_track \
      --video path/to/video.mp4 \
      --track path/to/track.npy \
      --segments path/to/spans.csv \
      --out src/annotator/validation_overlay/output_vids/check.mp4

`--track` is a `(n_frames, 3)` array: x as a fraction of frame width, y as a
fraction of frame height, and a visibility flag. Exactly `1` means tracked and
exactly `0` means not tracked. Other visibility values are invalid. Tracked x
and y values must be within `[0, 1]`. Untracked coordinates must remain finite,
but the bounds do not apply to them.

`--segments` is a CSV with a start column, an end column, and optionally a
label column. Frame numbers are inclusive and count from zero. If your CSV uses
different column names, say so rather than reshaping the file:

    --start-col final_contact --end-col window_end --label-col rally_id

Output lands wherever `--out` points. `output_vids/` next to this README is
gitignored, so it is the natural place to put renders.

## What you see

Each frame carries a small readout in the top-left corner:

- the source frame number, so you can go straight back to that row of the array
- `LEAD-IN`, `TARGET` or `LEAD-OUT`, telling you whether you are inside the
  span you asked for or in the context around it
- the segment's label, for the lead-in and the first second of the span
- the tracked position and visibility flag for that frame

A magenta box marks the tracked position. **Frames the tracker marked as not
tracked get no box.** That absence is the point: an empty frame tells you the
tracker lost the shuttle, and a box floating over nothing tells you the tracker
invented one.

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--render-width` | 1920 | minimum output width; a wider source keeps its own |
| `--hud-height` | 14 | text height in pixels, measured at 1920 wide |
| `--lead-in` | 2.5 | seconds of context before each span |
| `--lead-out` | 2.5 | seconds of context after each span |
| `--spacer` | 1.0 | seconds of black between spans |
| `--label` | shuttle | the name printed beside the box |
| `--verify` | off | check every frame is the frame it claims to be |

Small sources get upscaled without smoothing, so you see exactly the pixels the
tracker saw, just bigger. A 512x288 video renders at 1920x1080.

## Checking the render is honest

`--verify` checks that every rendered frame carries the pixels of the source
frame it claims. It decodes the video a second time from the start, hashing
every frame the render needs. Any hash that disagrees stops the render.

One limit is worth knowing. Where a run of consecutive source frames is
byte-identical, such as a freeze or a black gap, a shift inside that run passes
the check. That shift is equally invisible in the render, because the frames are
identical.

It costs one full decode up to the last span you asked for. On a 154,000-frame
video that is about a minute. Worth running the first time you point the tool at
a new video, and skippable after that.

One frame of seek error produces a render that looks completely plausible and is
wrong throughout. That is not hypothetical: the repo's older clip-slicing helper
does exactly that. See `DOCS.md`.

## Adding another algorithm

Each algorithm gets its own file under `overlays/`, owning its own inputs and
its own command line. There is no registry to update. See `DOCS.md`.
