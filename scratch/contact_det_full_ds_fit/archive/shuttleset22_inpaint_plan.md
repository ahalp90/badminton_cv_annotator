# ShuttleSet22 inpaint run

## Goal

Run the original InpaintNet coordinate step on the saved TrackNet results for
the 47 non-overlapping ShuttleSet22 videos. Keep every existing prepared file
unchanged and save the new results beside linked copies of those files in a
writable mirror.

The test labels remain closed. This run only prepares shuttle evidence.

## Fixed videos

Use these ShuttleSet22 IDs:

```text
8 9 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
31 32 33 34 35 36 37 38 39 40 41 42 43 44 46 47 48 49 50 51 52
53 54 55 57
```

The eight known overlaps and the three videos without a frame-aligned source
stay outside the run.

## Fixed method

- Use the saved no-inpaint TrackNet CSV for each video.
- Use the checked InpaintNet checkpoint with SHA-256
  `5749b66b8002f3ad9e0af841604004706fc796df30599e6bf01952696009688c`.
- Use non-overlapping 16-frame windows.
- Use every saved visible TrackNet coordinate in the window as context.
- Build the normal inpaint mask with a height cut-off of 5% of 1080 pixels.
- Replace only selected frames in the model output. Then apply the normal
  coordinate threshold and integer conversion to the complete output.
- Use GPU inference with batch size 16. Keep one model loaded for the complete
  run.
- Derive the existing shuttle guard codes from each final inpainted track.
- Normalise the saved track by the checked 1920 by 1080 frame size.

The full original-video trial found at most a one-pixel GPU difference wherever
the earlier input could be reconstructed exactly. The fabricated guard result
was unchanged. Four extra frames were marked degraded among 154,393 frames.
The user accepted this difference and approved the GPU run.

## Output layout

The prepared extract is read-only to the experiment account. Make a writable
mirror with one directory per video. Link every unchanged prepared file into
that directory, then add:

- `<video>_ball_inpainted.csv.gz`
- `<video>_stride8_inpaint_mask.json.gz`
- `shuttle_track_inpainted.npy.xz`
- `shuttle_guard_codes_inpainted.npy.xz`
- `shuttle_guard_diagnostics_inpainted.json.gz`
- `inpaint_result.json.gz`

The receipt records input and output hashes, row counts, selected-frame counts,
the model identity and runtime. It stores basenames rather than machine paths.

## Restart and write rules

- Build one video in a temporary directory and rename the directory only after
  every check passes.
- A complete video directory is skipped only when its receipt and outputs pass
  their saved hash checks.
- Stop when a completed output has changed or a partial directory already
  exists.
- Never replace, rename or remove a prepared input.
- Never open a ShuttleSet22 contact label.

## Checks before the long run

1. Check the exact 47 directories, CSVs, saved tracks and source videos.
2. Check that no proposed output already exists.
3. Check the checkpoint and source-code hashes.
4. Run a dry run over all 47 videos.
5. Process video 8 and check every output by loading it again.
6. Check that the original CSV and track hashes did not change.
7. Measure the video-8 runtime and use it to set the run ETA.

## Long run

Run the complete command in a named `tmux` session. The video-8 result is
rechecked and skipped. Save progress after every video. The session log and
full results remain outside Git.

## Stop conditions

Stop before the remaining videos when:

- any fixed video is missing or duplicated;
- an input hash changes during the first-video check;
- the saved base track does not match the base CSV;
- a frame list has a gap, duplicate or non-binary visibility value;
- the checkpoint or checked model code differs;
- any output would replace an existing file;
- a written output fails its reload or hash check.

## Outside this run

- TrackNet inference or video decoding
- pose, court or annotation work
- contact features, model scores or labels
- changes to production or vendored TrackNet code
- any model, cut-off or nearby-contact choice

## Commits

- `Plan the ShuttleSet22 inpaint run`
- `Add the ShuttleSet22 inpaint runner`
