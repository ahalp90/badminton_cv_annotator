# Our features

> **Where we’re time-normalising, normalise to frames (base-30 fps). Provide an fps column per rally.**

## Rally duration

- `(rally final contact + offset) – rally start`
- I slightly prefer the magic constant offset over something more normalised, like `(rally time / shots - 1)`. The normalised scales better, but it’s harder to reason around if trying to tie it to a broken auto-annotator.

## Posture variability

- Basically, how much did they scrunch up or end up crooked. But semi-robust to yaw in 2d perspective.
- Posture as `posture_t = abs(mean_eye_y - mean_ankle_y) / hip_width` per frame `t` per player.
- Median Absolute Deviation over rally:
  - Store every frame `t` value: `| posture_t – median(posture_t) |`
  - **Over rally, take the median of those values**
- Uses eye-to-ankle distance rather than bbox height to smooth out high arm movements. And hip width is the most population invariant and camera perspective invariant denominator I could think of. **Though you’ll need to record whether it was a man or woman [dedicated column]**
- Hips still get squished by yaw. But at least it implies they were working hard.

## Away from centre recovery

- At window opponent contact `t +/-5` base-30 frames (account for labelling noise):
  - Window mean of `(mean ankle)` distance from player’s court-half centre.
- Make sure you’re in homography-normalised space. Ankles can slightly project out. That’s fine, we’ll just end up slightly outside `(0,1)`. Don’t clip—keep it in that normalised space.
- Store per contact window **and** median over rally.

## Serve speed proxy

- `displacement / time`
- Measure from first contact to `(return | static | disappeared out viewport y=0 or y=max)`.
- Static will need to tolerate a tiny bit of detector noise.
- If we lose it during that trajectory due to inpaint masking, linearly interpolate.
- We’ve got a 2d single-vision camera, so it’s not true speed. We can only sometimes make even a bad guess at arc, so let’s not bother pretending.
- So long as these can be mapped back to the serve type by the next dataset user they’re actually pretty useful. The shot type, which can be reliably modelled, allows it to be scored meaningfully even without the arc.

## Shots per rally

- I’m not convinced this means anything, but it’s very cheap to calculate and store.

## Movement (in)efficiency

- `path – absolute displacement`
- Deception (or skill at reading opponent):
  - `Player_t -> Player_t+1`
- Record for each contact.
- Track mean ankle.
- Make sure these measures are taken within homography-normalised coordinates. Even if we end up outside the `(0,1)` homography, so long as it continues to scale it’s fine.

## Degradation

Measure degradation for all above features per player at a set and/or rally level:

- Progression over `{set,rally}`:
  - Least squares trend line, `tanh` `(-1, 1)` normalised to see if they get better or worse across the `{set,rally}` (i.e. how hard are they working).
  - Figure out a `tanh` scaling temperature so the values don’t get *tooo* compressed.

## Rally timestamps

- Rally timestamps in frame-range and seconds.

# Commentary

*Nearly forgot about this one!*

Only where the commentary communicates qualitative assessment.

Must occur either during a rally or within close succeeding proximity. As a magic number, perhaps the assessment phrase needs to occur within 5 seconds of rally end?

If we can pin the commentary directly to either of the two players by name and court slot (top/bot), great, give that its own column. If not, no worries.

The commentary needs to be cleaned of transcription glitches and verbal filler.

It needs its source time-range precisely identified by WhisperX to make sure we’re mapping to the right rally. YouTube ASR timestamps are too coarse.

Worth storing a global label per commentary chunk: *positive, negative, neutral*.

And perhaps an additional column with a single-word/concept descriptor of what aspect of play the commentator focused on. Whether this worthwhile really depends on the average length of the commentary chunks.

# Primitives

Provide the useful ShuttleSet cols:

- Contact-type
- Round within set
- Set number

*Since we’ve already got the extract running, it’d be great to do this for both ShuttleSet and ShuttleSet22.*

If feasible, provide all raw frame-level data in a separate bundle:

- Raw shuttle co-ords
- Raw keypoints + bbox stats

> [super useful to save people compute time, but potentially ends up huge?]

# How to deal with glitches

All features should be calculated with our inpaint hallucination masking.

Linearly interpolate wherever feasible.

Though I have no idea how to best deal with non-standard view data. Linearly interpolate if the signal drops during standard view. Extrapolate backwards if we’re in a match and it’s a non-standard-view start (measured either by PySceneDetect or CourtKeyNet? Or just totally discontinuous player bbox pos?). The non-standard stuff will never be in the same movement space...

But we absolutely must interpolate and extrapolate backwards, or we won’t have much of a dataset.

Hopefully these don’t happen mid-match, though I suspect occasionally they might. Those should be easier to deal with though.

These discontinuities should be easier to handle than with the core auto-annotator, since you can look at ground truth of the contact timings.

Store an **`interpolation_type`** column.

- I’d probably just store *linear* and *backward_extrapolated*.
- I’d do them as `IntEnum` to save space and speed.
- I’d leave empty all the standard visible rows to save space and readability.
- It’s a bit conceptually yuck, but `~np.isnan(col)` will give all the perfect rows.
