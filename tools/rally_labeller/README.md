# Rally labeller

Label contacts in badminton videos frame by frame: which player hit the shuttle (top or bottom of the image) and on which frame. Rallies are worked out from the contacts, so start and end marks are optional. The labels are for scoring and training the auto-annotator.

The labeller runs in a browser from a Vite dev server. Use it on the computer that holds the videos, or on a phone on the same home network.

## Set up

Run once, from this folder:

```bash
npm install
```

You also need `ffmpeg` and `ffprobe` on your PATH, and `uv` for looking up YouTube details.

## Start the labeller

From this folder:

```bash
npm run dev      # this computer only: open http://localhost:5173
npm run phone    # also on your home network: open the "Network" address on the phone
```

## Prepare part of a video

The browser plays a prepared copy of the part you plan to label. Press **Prepare…** at the top right, pick a source video, and enter a start and end time, such as `10:00` and `20:00`. Leave a time blank to begin at the start or run to the end. The new part opens when it's ready.

Or, from the repository root:

```bash
PYTHONPATH=src python3 tools/rally_labeller/prepare_video.py local_scratch/amateur_vids/E8WW8DFCnwk.mp4 --start 10:00 --end 20:00
```

Pass several files to prepare each one whole.

- Expect about 2.5× real time for 1080p60 sources, so a 10-minute part takes about 4 minutes.
- Each copy is at most 720 pixels high and has a keyframe every 15 frames, so stepping back one frame is as quick as stepping forward.
- Frame numbers follow the source video, so every part of one video shares one label file. The script stops with an error if the copy's frame count differs from what it asked for.
- The script records the source file's SHA-256 and frame size. It looks up the YouTube title, channel and upload date when the file name is a YouTube ID. Pass `--youtube-id` for a file with another name, or `--no-youtube` for a video from elsewhere.

Copies go to `local_scratch/rally_labeller/<video_id>/<first frame>-<end frame>/`, where the end frame is the first frame not included. The **Prepare…** panel lists the videos in `local_scratch/amateur_vids/`.

## Label

Choose a prepared part from the list at the top. The labeller saves after every change, to `data/amateur_contacts/<video_id>.json.gz`. Undo history lasts until you reload the page.

Right-click the video to turn the magnifier on: a box beside the pointer shows the area around it enlarged. Scroll to zoom in or out, and right-click again or press Esc to turn it off. On a phone, press and hold the picture, then drag. Set the zoom and box size in Settings, and tick **Sharpen the magnifier** for crisper edges; it also makes compression blocks more visible.

Under the video, the upper strip shows a few seconds either side of the playhead, with each contact as T (top player) or B (bottom player). The lower strip shows the whole part. Click either strip to go there.

### Four keys to start

| Key | Does |
| --- | --- |
| Space | Play / pause |
| V / B | Contact by the top (far) / bottom (near) player |
| F / G | Back / forward one frame. Shift+F / Shift+G jump 10 frames; change the jump in Settings. Hold to keep going |
| Z | Undo (Shift+Z redoes) |

Every action has a button with its key shown on it, so you can learn the rest as you go. Press the **?** button for every key and what each mark means. All keys sit on the left half of the keyboard, so your right hand can stay on the mouse.

### Fix a mistake

Every change shows a short note over the video, such as "▲ Top contact · 36210", so you can see what each key did.

- Press Z to undo the last change, and Shift+Z to redo it. The note says what was undone.
- Go to a mark by clicking it in the list or on the upper strip, or step through marks with R and T. Hold R or T to keep going.
- Wrong side: on the contact's frame, press the other key. A frame holds one contact, so the contact switches side.
- Wrong frame: press X on the mark's frame to delete it, then mark the right frame. The × beside a mark in the list also deletes it.
- Marking always adds a mark on the frame you're on. It never moves or changes another mark.

### Rallies

- Each contact joins the current rally.
- A pause of more than 4 seconds between contacts starts a new rally. Change the pause length in Settings; undo covers that too.
- **Break** (]) ends the current rally at its last contact; the next contact starts a new one.
- **Rally start** and **Rally end** are optional. Without them, a rally runs from its first contact to its last.
- **Unsure** (Tab) flags the mark on the current frame, or else the last one placed. Use it for a contact you can't see clearly.

### Labelled sections

Press **Section start** where you begin labelling and **Section end** where you stop. Frames outside every section count as unlabelled rather than rally-free, so you can label part of a long video and move on. A label file without sections doesn't record which stretches you finished, so the page warns until you mark one.

The list tags marks *outside sections*, and rally headings say when a rally crosses a section end. A mark tagged *no effect* changes nothing: a rally end or break with no rally open, or a section mark with no partner.

### On a phone

Run `npm run phone`, open the Network address on the phone, and hold it sideways. Press ⛶ for full screen, which also locks the screen sideways where the phone allows it.

- Tap the video to play or pause. A tap during playback stops on the frame you touched.
- Double-tap the upper half of the picture for a top contact, or the lower half for a bottom contact. The mark goes on the frame shown at your first tap, and playback carries on if it was playing. Double-taps right on the middle line don't count, because the far player often stands there.
- Drag sideways on the video to step through frames, one frame per 8 pixels. Drag along the lower strip to scrub through the whole part.
- Your left thumb has the step and play buttons; your right thumb has Top, Break, Undo and Bottom. Both thumbs can press at once.
- **List** opens the marks, with the suggested contacts below them; tap one to go to it. **More** opens every other button and the settings.
- "No section yet" in the top bar means nothing counts as labelled until you press **Section start**, under **More**.

## Suggestions

Show rough rallies and contacts from an auto-annotator run, or from a language model's first pass, as hints while you label. Each suggested rally shows on both strips as a sparse amber crosshatch, unlike the solid shading of rallies you've labelled. Each suggested contact shows on the upper strip as a short blue (top) or orange (bottom) tick. Suggestions are never saved as labels.

Find the suggested contacts in the lower half of the side panel, grouped by suggested rally, and click one to go to it. Press A and S to go to the previous or next suggested contact, and Shift+A and Shift+S for the previous or next suggested rally. To accept a suggested contact, go to it and press V or B. Once you've dealt with one, whether you marked it, found it wrong or skipped it, double-click it in the list or press D on its frame to strike it through. Do it again to undo. Done suggestions are saved per video in `local_scratch/rally_labeller/<video_id>/suggestions_done.json`, so the phone and the computer share them.

Convert an auto-annotator result, from the repository root:

```bash
python3 tools/rally_labeller/suggestions_from_annotator.py path/to/annotator_result.json.gz \
    --video-id E8WW8DFCnwk --name annotator-seed0
```

For other sources, write `local_scratch/rally_labeller/<video_id>/suggestions/<name>.json.gz` in this format, with frames numbered from the start of the source video and inclusive rally ends:

```json
{
  "schema": "rally-suggestions/1",
  "name": "gemini-rough",
  "rallies": [{ "start_frame": 600, "end_frame": 899 }],
  "contacts": [{ "frame": 620, "side": "top", "confidence": 0.9 }]
}
```

`side` may be `null` when unknown, and `confidence` may be `null`. Reload the page to pick up new files.

Untick **Show suggestions** when labelling videos for a test set, so the labels don't lean towards the system being scored.

## Label file

Each `data/amateur_contacts/<video_id>.json.gz` contains:

- `marks`: everything you placed, each with a frame, a kind and an unsure flag. The labeller loads these.
- `rallies` and `sections`: worked out from the marks on every save, with inclusive frame ranges. Each rally lists its contacts with their side (`top` or `bottom`).
- `source`: the source file name, SHA-256, frame width and height, and YouTube details.
- `fps`, `frame_count`, `rally_gap_seconds` and a `revision` count.

`top` is the far half of the court and `bottom` the near half, matching the auto-annotator's `Top` and `Bot`.

Each save is checked against the revision the page loaded. If the same video is open on two devices, the page that saves second merges its changes into the other device's copy, so neither overwrites the other's work. A page that comes back into view also picks up changes saved elsewhere.

## Other folders

Set these to use other folders. Relative paths count from the repository root.

| Variable | Folder for | Default |
| --- | --- | --- |
| `RALLY_LABELLER_MEDIA` | prepared copies and suggestions | `local_scratch/rally_labeller` |
| `RALLY_LABELLER_LABELS` | label files | `data/amateur_contacts` |
| `RALLY_LABELLER_SOURCES` | videos offered by **Prepare…** | `local_scratch/amateur_vids` |

`prepare_video.py` and `suggestions_from_annotator.py` read `RALLY_LABELLER_MEDIA` too, so set it once for both the labeller and the scripts.

## Checks

```bash
npm test             # mark and rally logic
npm run typecheck
```
