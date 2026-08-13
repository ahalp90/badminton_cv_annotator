# Validation overlay: maintenance and extension

For someone changing this code or adding an algorithm to it. For running it,
read `README.md` first.

## Conventions used here

- **source frame index**: a frame's position in the source video, counting from
  zero. Every part of this tool speaks in these, and nothing else
- **span** or **target span**: the inclusive frame range one CSV row asks for
- **lead context**: the frames added either side of a span for context
- **spacer**: the run of black frames sitting between two rendered spans
- **overlay**: one algorithm's drawing code, one file under `overlays/`
- **the assembler**: the part of the core that pairs each decoded frame with its
  row of algorithm data and stacks the results into output order

## The shape

The core handles video and time. It knows nothing about badminton. Each overlay
knows about one algorithm and nothing about video.

```
core/decode.py     probe metadata, decode an exact span
core/timeline.py   read the CSV, build the ordered frame plan (pure, no video)
core/encode.py     pipe raw frames to one long-lived x264 process
core/hud.py        the corner readout and text scaling
core/cli.py        shared flags, frame composition, the identity gate, render()
overlays/          one file per algorithm
```

The plan in `timeline.py` is pure Python. Given segments, a frame count, a
frame rate and the timing settings, it returns the exact ordered list of source
frames and spacers. No video is touched, so it tests in milliseconds.

## Adding an algorithm

Write one file under `overlays/`. Nothing else changes: no registry, no
package initialiser, no core edits. Deleting an algorithm is deleting its file.

Your file needs three things.

**A parser** that inherits the shared flags and adds its own inputs:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="...", parents=[build_shared_parser()],
    )
    parser.add_argument("--picks", type=Path, required=True)
    return parser
```

**A draw function** matching the contract. It is called once per non-spacer
frame, with the frame already resized:

```python
def draw(image: np.ndarray, source_idx: int, in_target_span: bool) -> list[str] | None:
    """Draw this algorithm's marks in place; return extra readout lines."""
```

Index your own data with `source_idx` and nothing else. If your data is keyed
per scene or per rally rather than per frame, do that mapping yourself. Mutate
`image` in place; the return value is only for readout lines.

**A `main()`** that probes the video, reads the segments, loads and validates
your inputs, builds a plan, and calls `render()`. Copy the shape of
`overlays/shuttle_track.py`; shared input contracts belong in a lightweight
annotator module rather than in the overlay.

Use `style.scale` from the `HudStyle` for any pixel measurement, so your marks
size themselves the way the box does. Take new colours from
`~/Documents/protan_colour_scheme.md` rather than picking by eye.

The contract already covers the awkward cases: several marks on one frame, no
mark at all on a frame, and per-frame text. The end-to-end test exercises all
three with a synthetic overlay, so a new algorithm has a working reference.

## Why the decode command looks like that

`core/decode.py` builds one ffmpeg command. Its exact shape is load-bearing and
was settled by measurement, not preference.

**Never seek with a half-frame offset.** `slice_rallies.py:97` and
`clip_generator.py:93` both seek to `(frame + 0.5) / fps`, documented as a guard
against rounding onto the previous frame. On ffmpeg 6.1.1 it does the opposite
and lands one frame late, every time:

    target n=500     seek (n+0.5)/fps -> frame 501
    target n=500     seek  n/fps      -> frame 500
    target n=40000   seek (n+0.5)/fps -> frame 40001

Do not import either helper. `test_validation_overlay_core.py` pins this.

Four properties of the command each fix a specific failure:

- **`-t` sits before `-i`.** With `-copyts` an output-side `-t` measures against
  absolute timestamps, so it returns zero frames
- **`-frames:v` is present.** Without it the `select` filter keeps decoding to
  the end of the file: 0.13 seconds becomes 32
- **`-copyts` is present.** It keeps the source timestamps, which is what lets
  the filter select by absolute source time rather than by seek-relative time
- **timestamps come from `Fraction`.** At 25 fps floats happen to work. At
  30000/1001 the six-decimal rounding eats into the half-frame selection margin

The command seeks two seconds early on purpose and selects the window by
timestamp. That way the result never depends on the seek landing exactly.

## Traps

**ffmpeg reports success on a short read.** A decode running past the end of the
file exits zero and hands back fewer frames than asked for:

    past EOF [154391,154396]  want 6  got 2  rc=0

So `decode.py` compares the byte count itself and raises. Use an explicit
comparison and `raise`, never `assert`: assertions vanish under `python -O`, and
this check is the only thing standing between a truncated decode and a render
that silently stops early.

**Pipe reads split frames.** A size-limited read on a pipe returns whatever is
ready, so one read is not one frame. `_iter_decoded_frames` buffers to the exact
frame size and rejects a partial tail.

**Pixel shape decides the output's shape, never a mark's position.** A mark is
a fraction of the coded frame, so non-square pixels cannot move it. They only
decide what shape the render is written at, which is why `make_render_plan`
uses the sample aspect ratio for sizing rather than rejecting it.

Two details there are load-bearing. The output widens to at least
`coded_width * SAR` before the height is derived, because sizing off the coded
width alone would satisfy the ratio by *shrinking* the other axis and throwing
away real detail on a wide-pixel source. And three spellings all mean
unspecified and all become 1:1: the key absent, `"N/A"`, and ffmpeg's `"0:1"`.
Absent is the common case, since a plain encode records nothing, this tool's
own output included. Treating absent as an error refuses ordinary files.

**Frame rate must be constant, and `--verify` will not save you here.** The
whole timestamp model assumes frame `n` sits at `n/fps`. On variable-frame-rate
input that stops being true, and the identity gate rests on the same assumption,
so it can miss the resulting misalignment rather than catching it. The canonical
metadata probe first compares `r_frame_rate` against `avg_frame_rate` as exact
fractions. It then requires one presentation timestamp per counted frame and
proves that frame `n` lands exactly at `n/fps` in the stream time base. Missing,
duplicate, or irregular timestamps fail before rendering. Supporting VFR properly
would mean indexing by presentation timestamp instead of by ordinal, which
rawvideo does not carry through the pipe. Worth knowing before this meets phone
footage.

**The identity gate compares pixels, not indices.** `--verify` hashes each
planned frame and compares against a decode-from-zero reference. Any shift onto
visibly different pixels fails, which is every shift that could mislead a
viewer. A shift contained entirely within a run of byte-identical frames passes.
Strengthening that would mean carrying each frame's presentation timestamp
through the raw pipe, which rawvideo does not do. Do not describe `--verify` as
proving the source index.

**Pyrefly in a fresh worktree.** Pyrefly finds its environment from a `.venv` at
the tree root. That directory is gitignored, so a fresh worktree has none.
Pyrefly then falls back to an interpreter without jaxtyping, and invents errors
on shape annotations. Pass the interpreter:

    pyrefly check --python-interpreter-path ~/.venvs/badminton-cicd/bin/python

Seventeen `unknown-name` errors on symbols like `'batch'` mean the flag is
missing. The code is fine.

## Tests

    PYTHONPATH=src pytest tests/test_validation_overlay_*.py -v

Six tests, each pinning one thing that has already gone wrong somewhere:

| Test | Pins |
|---|---|
| seek regression | `n/fps` is exact and `(n+0.5)/fps` is one late |
| planner | the plan's frame order, clipping and spacer runs |
| frame-zero clamp | lead context at the file start clips without padding |
| assembler | every frame pairs with the right array row, before encoding |
| short read | a decode past the end raises rather than truncating |
| identity gate | every planned frame really is that source frame |

The assembler test asserts on the frame stream **before** encoding. At crf 18
the encode is lossy, so a pixel check on the finished mp4 would be fragile. The
identity gate is marked `slow`; run it with `-m slow`.

## Known gaps

Three things this tool does not do, each a deliberate stopping point rather
than an oversight.

**Track provenance is unchecked.** A `(n_frames, 3)` array from a different
video of the same length passes every guard, including `--verify`, and draws
confident nonsense. Verification covers video decoding rather than whether the
array belongs to the video. Detecting it would need a source hash the arrays do
not carry.

**The decoder reads stderr after `wait()`.** If ffmpeg ever filled its stderr
pipe it would block, and the stdout read loop would never see the pipe close.
With `-v error` that needs a pathological failure, so it is left as is. Drain
both pipes concurrently if you ever raise the log level.

## Extending the core

Three changes would need real thought rather than a new overlay file:

- **variable frame rate.** `probe_video` rejects it, because the whole
  timestamp mapping assumes one regular interval per frame
- **sources with a non-zero start time or rotation metadata.** Both rejected
  for the same reason: they break the assumption that frame `n` sits at time
  `n/fps` in the coded frame as stored
- **drawing two algorithms at once.** Write a third overlay that imports both
  draw functions and calls them in turn. Deliberately explicit rather than
  configurable
