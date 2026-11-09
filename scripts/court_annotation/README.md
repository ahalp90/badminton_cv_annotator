# Mark courts and camera changes

Run the amateur collection from the repository root:

```bash
bash scripts/court_annotation/annotate_amateur.sh
```

The old `local_scratch/amateur_vids/annotate_courts.sh` command remains a wrapper for this launcher.

The workspace opens the first video with pending stops. Choose any other video from the video menu at the top left; each entry shows its pending stop count and the start of its note. **Next video** opens the next video in the list. Pass a filename to open that video first. The batch excludes `nTL8uUlb4fo` (heavily edited edutainment that is unsuitable for this annotation benchmark) and `i1KGN5cDoVA` (mostly a person talking). Their video files remain untouched; naming either explicitly adds it to the list. Set `INCLUDE_UNCERTAIN=1` to include the scan's uncertain transitions. The default queue includes an initial court frame and movement/drift candidates, merging flags within three seconds. Each candidate opens half a second before its detected end time. Scrub to the actual change or a clear court view; a scan flag does not establish a camera change.

Use the single desktop workspace: video on the left, saved courts and suspected movement on the right. During capture, the sidebar switches to point refinement and the timeline hides, keeping the video large. The buttons show what is available at each step.

1. Click any visible court intersection. Choose its identity by clicking the standard court diagram, either before or after aiming on the video. Filled markers show assigned points; a diamond marks the selected point. The dropdown remains available.
2. Refine the point in the zoom, then choose **Confirm point**. Right-drag the zoom to pan without moving the point. The zoom stays visible after confirmation. The selector advances to the next intersection from left to right, then the next row from far to near. Choose another label whenever needed. The final point does not wrap around. If the next identity is already marked, the prompt warns that confirming will replace it; Undo restores the previous points.
3. Choose **Save court** once all four outer corners are confirmed. Alternatively, mark at least five well-spread labelled intersections (inner lines are fine), choose **Fit court**, inspect the result, then **Save court**. Missing outer corners are derived from the fit; you do not need to mark them off-screen first.
4. Choose **Next pending** to inspect the next camera candidate. Use **Camera changed here** on the exact change frame, then scrub to a clear frame and mark its court. Choose **Unchanged** when the view did not move. For a momentary jolt (about one frame), go to that frame and choose **Shake**: it saves a one-frame jitter period there and marks the stop Unchanged. Mark longer jitter as a jitter period. Choose **Revisit later** for uncertain cases.

Both **Confirm point** and **Save court** are explicit. Confirming the fourth corner does not save or end the capture. To revise a confirmed point, select its label again; its existing position opens in the zoom. **Undo point** cancels a provisional aim first, then restores the previous confirmed point set, including the old position of a replaced point. Relabelling a reopened point moves its identity: the old label is removed, and an existing destination label is replaced. Undo restores both labels and their earlier positions.

**Copy court · frame N** copies actual labelled clicks from the nearest saved annotation. The button names the source frame; its tooltip gives the source time too. The nearest annotation can be before or after the displayed frame. Inspect or edit the points, fit again if corners must be derived, then choose **Save court**. Derived corners are never silently converted into manually clicked points.

**Previous** and the suspected-movement list revisit queue items. **Clear decision** resets the selected stop to pending. **Remove marker** removes a camera-change marker at the exact displayed frame. Changing a stop's decision detaches its associated marker; the marker remains if another stop uses it or it was also saved as standalone. Reflagging a stop replaces its old associated marker by the same rule. Standalone markers remain until explicitly removed. A saved court at or after the new change frame remains associated; an earlier court must be marked again.

The selected queue stop stays selected while you scrub. Decisions and a saved court therefore belong to that stop even if you choose a different frame. Saving court corners alone records an annotation; it does not confirm a camera change. A confirmed change remains pending until a court is saved for it. Uncertain stops remain pending. The next launch opens the first pending stop; if all are reviewed it opens the first stop for inspection.

Completed courts, decisions, periods and notes save immediately. Switching video or closing the workspace keeps all of them; only an unsaved court capture is discarded on close.

## Periods and notes

Mark play sections, jittery camera and visual transitions (fades, wipes, replays) as periods. Go to the first frame and choose **Start play**, **Start jitter** or **Start transition**. The button turns the colour of its timeline lane and shows the start time, for example **End play · 03:10**; go to the last frame and choose it. Start and end frames are both inclusive. Each kind can have any number of periods. **Shake** saves a one-frame jitter period at the displayed frame, with or without a stop selected. Different kinds may overlap, but periods of the same kind may not; the tool refuses a start or end that would overlap one. An open period is drawn faintly on the timeline up to the displayed frame.

The periods list shows every period. Click a start or end time to visit that frame; **Remove period** deletes the selected row. To move a boundary, remove the period and mark it again.

Type anything about the whole video in the notes field below the periods list, for example "doubles match: exclude from benchmarking" or "done". It saves as you type; Enter returns to the video. The video menu shows the start of each note.

## Files

Court corners and landmarks retain their existing CSV format. Each video's `<video_id>_review.json.gz`, beside the corner CSV, records:

- Candidate frame, reason and decision.
- Confirmed change frame and the separately chosen court annotation frame.
- All confirmed camera-change frames in `camera_changes`, including manual flags.
- `periods`: `{"kind": "play" | "jitter" | "transition", "start": frame, "end": frame}`, both inclusive. An unfinished period has `"end": null`.
- `notes`: the video's free-text note.

Stop decisions are `unchanged`, `changed`, `annotated`, `uncertain` or `pending`. A shake is a jitter period whose start equals its end. Earlier `unchanged` decisions may cover shakes recorded before the Shake button existed.

Queue timestamps are converted using the video's reported average FPS. For variable-frame-rate sources these are approximate starting points; manually confirmed change markers use the displayed decoded frame index. The queue does not automatically change downstream pipeline court selection.

For a video outside this collection:

```bash
~/.venvs/badminton-cicd/bin/python \
  scripts/court_annotation/annotate_court_corners_offframe.py \
  --video /path/to/video.mp4 \
  --out-csv /path/to/corners.csv \
  --candidate-dir /path/to/camera_scan
```

`--candidate-dir` names the camera-scan folder that holds `<video stem>/candidates.csv.gz`. Omit it for evenly spaced sample stops. Pass several `--video` files to get the video menu; `--start-video` names the one to open first, and `--start-frame` overrides its opening position.

## Two navigation lists

The workspace has separate scrollable lists for saved court frames (with TL, TR, BR, BL pixel coordinates) and suspected camera-movement points (with review decisions). Each list scrolls independently; the saved-court table has a horizontal scrollbar for all four coordinates. Clicking a saved court seeks to that frame and displays its saved court overlay without selecting a review stop; editing it does not complete a different candidate. Click a suspected movement or Next pending to resume queue work.

The court overlay menu beside **Copy court** draws a saved court over the displayed frame. Choose **Overlay previous saved court** to follow the most recent saved court at or before the displayed frame, or pick any saved frame to keep that court while you scrub the whole video. Clicking a saved court in the list picks its overlay. The overlay labels its source frame and time and changes no annotations. The copy button starts an editable capture from the saved frame named on the button.

While a saved court is selected, **Camera changed here** saves a standalone marker without deciding a queue item. Select a suspected movement first to associate the marker and subsequent court annotation with that review stop.

**Next confirmed** cycles through all confirmed markers, including standalone flags, and seeks their exact frames. It opens inspection without selecting a suspected movement. Use **Remove marker** to delete the displayed marker.

## Frame navigation and point refinement

Left/Right arrows step one frame while browsing, including when a list or button has focus. During refinement they nudge the provisional point. The frame field and dropdown keep their usual editing keys. Enter or Space confirms when the video or zoom has focus; every operation also has a labelled button. Navigation and queue decisions are disabled during a capture. Confirmed points remain unsaved until you choose Save court. The four outer corner identities are written in canonical TL, TR, BR, BL order regardless of click order.

The workspace uses PySide6 from `~/.venvs/badminton-cicd`. Video frames decode directly with OpenCV; no browser or transcoding service is involved. To install the GUI dependency in another copy of that environment:

```bash
uv pip install --python ~/.venvs/badminton-cicd/bin/python PySide6==6.12.0
```

The executable annotator uses that environment in its shebang.

## Timeline

The ruler shows time and separate lanes for saved courts, suspected movements, confirmed changes and the three period kinds. Click a marker to seek its exact frame. Zoom in around the current frame, zoom out, or choose Whole video. Seeking happens on release, so dragging does not repeatedly decode the video. Clicking a suspected-movement marker also selects its review stop. Saved-court markers open the saved annotation with its overlay; confirmed-change markers open independent inspection. Clicking the ruler or empty timeline space only seeks and keeps the selected review stop.

The default zoom dock displays a 40 × 40 source-pixel crop at 320 × 320 logical pixels (8×). `--loupe-size` changes the crop half-width; explicit larger crops are fitted into the same dock and therefore have lower visible magnification. Changing `--loupe-zoom` changes the intermediate nearest-neighbour rendering, while the dock remains 320 pixels wide.

Saving a court before a confirmed camera-change frame stores an independent annotation and leaves that stop pending. A court at or after the change can complete it. Enter in the frame field seeks and returns focus to the video. Negative numbers go to frame 0; numbers beyond the video go to its final frame. The −1/+1 frame buttons sit together, followed by the −1/+1 second buttons. Shift-click a frame button, or press Shift+Left/Right, to step 10 frames. Up/Down navigate the review tables while browsing.

Right-click or right-drag the video to zoom there without marking a point. While browsing, the zoom appears in a top corner of the video and stays on the same spot as you step frames, which makes small shifts and overlay misalignment easy to see. Escape closes it, and a left click inside it places a point at that spot. During point refinement, right-click moves the zoom panel instead. Escape cancels a provisional point or an empty capture; once points are confirmed, use **Cancel court** to discard them.

Arrow-key nudging uses `--nudge-step` (default 0.5 source pixels). There are no on-screen nudge buttons or step-size control.

## Derived corner provenance

The corner CSV header is unchanged. Manually labelled outer corners use `source=click`; corners derived from the fitted labelled intersections use `source=extrapolated`. Every clicked or derived corner has `visible=1` when its projected coordinates lie inside the image, and `visible=0` outside. This means in-image, not a judgement about player occlusion. The landmarks sidecar records the identities and pixel coordinates actually confirmed.

Saving checks the final combined clicked/derived court for crossed, mirrored or upside-down corner identities. A refused save keeps the points available to correct. Selecting an identity from the dropdown returns keyboard focus to the video, so arrows nudge instead of changing the label. When every review stop is decided, the status directs you to Next video or closing the workspace.

The projected court includes the outer boundary, singles sidelines, service lines and centre lines in bright pink in both the main view and the zoom. The centre lines stop at the short service lines. Both selection crosshairs use the same pink. Right-dragging the zoom changes only the area shown; it does not move a confirmed or provisional point.
