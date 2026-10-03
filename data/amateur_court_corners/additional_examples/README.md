# Additional amateur court examples

These source images and manual labels are useful for checking partial courts
and unfamiliar camera views. They are development examples, not a held-out
accuracy benchmark. The main [amateur labels](../README.md) cover a separate set.

| Files | Use |
| --- | --- |
| [`../2026-09-08/hand_corners.csv.gz`](../2026-09-08/hand_corners.csv.gz) | Per-frame manual court corners, including extrapolated off-screen corners. |
| [`../2026-09-08/hand_corners_landmarks.csv.gz`](../2026-09-08/hand_corners_landmarks.csv.gz) | Clicked court crossings used to fit those corners. |
| `*_gameplay.png` | One saved source frame for each short clip. |
| `manifest.json.gz` | Source video IDs, download intervals and decoded frame indices. The corner and landmark CSVs contain the labels. |
| `E8WW8DFCnwk_sample.mp4`, `Cb-xs5rPyxI_gameplay.mp4`, `l-I_Di1Ad2Y_h264.mp4` | Three labelled source clips, checked in (2.5 MB combined). |
| `gxBQ_HwdgN4.mp4` (local only) | Full 419 MB source video for the seven GX frames. |

CSV video paths are relative to the repository root. The GX annotations are in
[`../2026-09-09/`](../2026-09-09/); their frame indices address the full
`gxBQ_HwdgN4.mp4`, not a trimmed clip. Obtain that video by its YouTube ID if the
local copy is absent. Keep its frame numbering when decoding it.

The three other IDs are `E8WW8DFCnwk`, `Cb-xs5rPyxI` and `l-I_Di1Ad2Y`.
The manifest records their excerpt times. These clips and source frames are
inputs; discarded detector outputs are not needed to use the manual labels.
