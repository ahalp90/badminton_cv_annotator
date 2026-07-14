# Amateur-video court corner ground truth

Hand-annotated court corners for four amateur badminton videos, for scoring
court detection on footage that has no recorded homography. Annotated by
Curtis Martin on 12 Jul 2026 with
`src/courtkeynet/validation_scripts/annotate_court_corners_offframe.py`.

Source videos are not in the repo; download by ID:

| video | frames annotated | notes |
| --- | --- | --- |
| [C6NrJyBwn6c](https://www.youtube.com/watch?v=C6NrJyBwn6c) | 4 | static camera; br, bl off-frame |
| [C1jR4vZmrkI](https://www.youtube.com/watch?v=C1jR4vZmrkI) | 3 | camera MOVES between frames; corners are per-frame only |
| [zYqgJo1L5uM](https://www.youtube.com/watch?v=zYqgJo1L5uM) | 2 | static camera; br, bl off-frame |
| [BkjErIAsZu4](https://www.youtube.com/watch?v=BkjErIAsZu4) | 2 | static camera; bl off-frame, tl and tr occluded (extrapolated positions land inside the image) |

`hand_corners.csv`: one row per corner, four per frame. Columns are in the
tool's docstring; key ones: `corner_label` (tl tr br bl), `visible` (0 means
not clickable) and `source` (extrapolated corners come from a homography
fitted to that frame's clicked landmarks). Extrapolated coordinates sit
outside the image where the corner really is off-frame.

`hand_corners_landmarks.csv`: the clicked crossings each fit used, with court
coordinates in metres. Refitting from these reproduces the extrapolated
corners exactly.

Quality at annotation time: fit rms 2.1 to 3.6 px over 18 to 25 landmarks per
frame; on the static-camera videos, independent per-frame fits agree within
2 to 4 px on every corner. No video shows the whole court, so every frame has
1 to 3 extrapolated corners.

`renders/` holds one visual audit per video: every painted line projected
through the frame's fit, drawn over the actual frame. Regenerate any frame
with `src/courtkeynet/validation_scripts/render_ground_truth.py`.
