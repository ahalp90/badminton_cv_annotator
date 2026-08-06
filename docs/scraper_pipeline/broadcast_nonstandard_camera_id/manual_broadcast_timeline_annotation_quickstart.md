# Manual broadcast timeline annotation quick-start

The workflow is:

1. Prepare one complete 512x288, 25 FPS review video.
2. Optionally load a matching PySceneDetect `raw_cuts.csv` to tag cut-to-cut
   scenes instead of drawing every interval from scratch.
3. Optionally ask Gemini for rough proposals. Try the complete video first and
   shard it only if Gemini rejects or cannot cover it.
4. Launch the local annotation tool.
5. Label every class interval in the complete video.
6. Validate and hand over the completed CSV.

## 1. Prepare one complete review video

Run from the repository root. Choose a short, stable `VIDEO_ID`.

```bash
SOURCE=/absolute/path/to/source.mp4
VIDEO_ID=my_match
WORK="local_scratch/broadcast_timeline_annotation/$VIDEO_ID"
VIDEO="$WORK/${VIDEO_ID}_288p.mp4"

mkdir -p "$WORK"

ffmpeg -i "$SOURCE" \
  -vf "scale=512:288,fps=25" \
  -an -c:v libx264 -crf 18 -pix_fmt yuv420p \
  "$VIDEO"

ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,avg_frame_rate,nb_read_frames \
  -of default=noprint_wrappers=1 "$VIDEO"
```

Keep this converted video unchanged. The output labels refer to its exact
frames. Record the source, video ID, FPS, frame count, date, and annotator in
`$WORK/run_notes.txt`.

## 2. Optional: load PySceneDetect scenes

Use a `raw_cuts.csv` produced from the exact review video when one is
available. The file must be a complete, ordered, half-open partition:

```text
scene_index,start_frame,end_frame
0,0,723
1,723,3368
```

The
[ShuttleSet annotator heuristic reference release](https://github.com/ahalp90/badminton_cv_annotator/releases/tag/shuttleset-annotator-heuristic-reference-v1)
contains these files for `sset_01`, `sset_15`, and `sset_21`. For example,
`sset_01` uses:

```text
measurement/current_annotator_8config_288p/shared/
sset_01/tracknet-stride-8/raw_cuts.csv
```

Download the release and extract the matching file. This example selects
`sset_01` stride 8:

```bash
ARCHIVE="$WORK/shuttleset-current-annotator-reference-v1.tar.gz"
RELEASE_ROOT=shuttleset-current-annotator-reference-v1
RELEASE_CASE=sset_01/tracknet-stride-8
SCENES="$WORK/${VIDEO_ID}_raw_cuts.csv"

gh release download shuttleset-annotator-heuristic-reference-v1 \
  --repo ahalp90/badminton_cv_annotator \
  --pattern shuttleset-current-annotator-reference-v1.tar.gz \
  --dir "$WORK"

tar -xOzf "$ARCHIVE" \
  "$RELEASE_ROOT/measurement/current_annotator_8config_288p/shared/$RELEASE_CASE/raw_cuts.csv" \
  > "$SCENES"
```

Do not reuse a scene file from a different encode or video. Its frame numbers
must cover the review video's exact `[0, frame_count)` range. The release only
covers the three named ShuttleSet fixtures. Another video needs a fresh
PySceneDetect run from the pinned pipeline configuration.

PySceneDetect supplies camera-shot boundaries, not class labels. The reviewer
still assigns every class and corrects any semantic change that falls inside a
detected scene.

## 3. Optional: ask Gemini for rough proposals

Gemini is only a source of rough jump points. The annotator must still review
the complete video. Try uploading the complete review video first. Shard it
only if Gemini rejects the full upload or cannot cover it.

The Issue 29 fallback used 600-second shards with a two-second overlap:

```bash
mkdir -p "$WORK/shards" "$WORK/gemini_responses"

DURATION=$(ffprobe -v error -show_entries format=duration \
  -of csv=p=0 "$VIDEO")

for ((START=0; START<${DURATION%.*}; START+=598)); do
  ffmpeg -ss "$START" -i "$VIDEO" -t 600 \
    -an -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p \
    "$WORK/shards/${VIDEO_ID}_start_${START}s.mp4"
done
```

### Exact Issue 29 Gemini prompt

The following is the exact main prompt supplied for the `sset_01` run:

```text
Analyze the entire uploaded badminton broadcast to propose scene intervals for later human review.

Source metadata:
- video_id: sset_01
- fps: 25
- frame_count: 154393
- duration: 6175.72 seconds

Classify footage into exactly these classes:
- live: standard court-showing live footage
- live-non-standard: valid live action from an unusual camera view
- replay: repeated or slow-motion footage of earlier play
- cutaway: audience, player close-up, ceremony, or other non-play broadcast shot
- other: graphics, transitions, or footage outside those classes

Return JSON only. Use an ordered array with:
- start_s
- end_s
- event
- confidence
- note

Use absolute seconds from the start of the source video. Intervals must not overlap. Aim for continuous coverage of the full video. Flag rapid transitions and uncertain classifications in note.

These are approximate proposals. Do not claim exact frame boundaries.
```

For each shard, this exact suffix was appended. Replace `START` with the
shard's source start time:

```text
This shard begins at absolute source time START seconds.

Return start_s and end_s as absolute times in the original full video. Add START to every shard-relative timestamp. Include the overlapping edge footage. It will be deduplicated later.
```

For another video, change only the four source metadata values and each shard's
`START`. Omit the shard suffix when uploading the complete video.

Save each unchanged response as:

```text
$WORK/gemini_responses/<VIDEO_ID>_start_<START>s_gemini_raw.txt
```

Record the model, request time, prompt, and shard origins in `run_notes.txt`.
Gemini did not follow this prompt reliably during Issue 29. Later responses
used shard-relative timestamps and some described footage beyond the upload.
Treat every response as an optional proposal.

Gemini JSON cannot be passed directly to the annotation tool. A checked
proposal CSV needs `start_frame`, `end_frame`, `truth`, and `note` columns.
Issue 29's response converter supports only `sset_01`.

## 4. Launch the annotation tool

Install the development environment once:

```bash
uv sync --extra dev
```

Then launch the complete video without scene bootstrapping:

```bash
OUT="$WORK/${VIDEO_ID}_broadcast_timeline_labels.csv"

PYTHONPATH=src uv run python -m annotator.manual_broadcast_timeline_annotator \
  --video "$VIDEO" \
  --video-id "$VIDEO_ID" \
  --out-csv "$OUT"
```

If a matching PySceneDetect scene CSV exists, launch with:

```bash
PYTHONPATH=src uv run python -m annotator.manual_broadcast_timeline_annotator \
  --video "$VIDEO" \
  --video-id "$VIDEO_ID" \
  --out-csv "$OUT" \
  --scene-csv "$SCENES"
```

If a checked proposal CSV exists, add this option:

```bash
--proposal-csv "$WORK/${VIDEO_ID}_gemini_proposals.csv"
```

The `j` key only works when a proposal CSV or `--gt-csv` rally guide is loaded.
PySceneDetect scenes and Gemini proposals can be loaded together. The scene CSV
controls exact scene tagging. The proposal CSV remains an advisory overlay.
This is also the input seam for a future Issue 38 VLM: attach proposed classes
to the same scene intervals, then pass those rows through `--proposal-csv`.
Keep the mechanistic scene partition separate from the advisory VLM classes.

## 5. Label the complete video

Without `--scene-csv`, start at frame 0. Find the **last** frame before the
class changes and press its number. The next interval starts automatically.
Do not mark every camera edit, only class changes.

With `--scene-csv`, the tool starts at the midpoint of the first unlabelled
scene. Inspect the scene, then press its number anywhere inside it. The tool
labels the complete half-open scene and moves to the midpoint of the next
unlabelled scene. This avoids manually seeking to `end_frame - 1` and prevents
off-by-one scene bounds.

Scene cuts are a bootstrap, not semantic truth. If a class changes inside a
detected scene, press `s` on the exact first frame of the manual interval, move
to its last frame, and press the class number. An explicit `s` selection uses
the original frame-by-frame behaviour instead of tagging the full scene.

- `1`: standard court-showing live footage
- `2`: actual live action or warm-up from an unusual camera view
- `3`: replay
- `4`: player close-up, audience, ceremony, or another cutaway
- `5`: graphics, broadcast stings, graphics transitions, or other footage
- `,` and `.`: move one frame
- `<` and `>`: coarse jump
- `g`: first unlabelled frame
- `n`: add a note
- `d`: delete the current interval
- `v`: check complete coverage
- `q`: quit

The CSV is saved after every label. Run the same launch command to resume.

*Note:*  Some boundaries are very tight. A side-on shot at the start of a rally is a cutaway (4) if it only shows the player preparing to serve. If the player serves before the shot ends, label the entire shot live-non-standard (2).

## 6. Validate and hand over

```bash
PYTHONPATH=src uv run python -m annotator.manual_broadcast_timeline_annotator \
  --video "$VIDEO" \
  --video-id "$VIDEO_ID" \
  --out-csv "$OUT" \
  --validate-only
```

The command must report a valid partition. Hand over the output CSV and
`run_notes.txt`.

For the full label policy, read
[`broadcast_timeline_labelling_20260731-095201.md`](broadcast_timeline_labelling_20260731-095201.md).
