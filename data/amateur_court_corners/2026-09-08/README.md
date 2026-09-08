# Court annotations — 8 September 2026

Nine manually annotated frames cover three short, fixed-camera gameplay clips.
These annotations supplement Curtis Martin's four-video dataset in the parent
directory. Nearby frames check the same camera view; they do not establish
accuracy across the full source videos.

| YouTube ID | Local clip | Source section (seconds) | Annotated clip frames |
| --- | --- | --- | --- |
| E8WW8DFCnwk | E8WW8DFCnwk_sample.mp4 | 120–123 | 14, 90, 156 |
| Cb-xs5rPyxI | Cb-xs5rPyxI_gameplay.mp4 | 720–723 | 45, 58, 78 |
| l-I_Di1Ad2Y | l-I_Di1Ad2Y_h264.mp4 | 120–123 | 36, 64, 71 |

Section times describe the requested downloads. Frame numbers index the decoded
local clips, not the full YouTube videos. Source clips remain local under
`scratch/court_det_fix/worklog/checks/independent/examples_updated/`.

- [hand_corners.csv.gz](hand_corners.csv.gz): four corner rows per annotated frame.
- [hand_corners_landmarks.csv.gz](hand_corners_landmarks.csv.gz): clicked line
  intersections used to fit the court. Hidden corners are projected from these.
- [renders/](renders/): one overlay for each of the nine annotated frames.

Annotations were made with `annotate_court_corners_offframe.py`. The compressed
CSVs preserve the original tool output exactly. Published court detection
evaluations have not yet been rescored against these annotations.

The `video` column contains the local clip path shown above, rather than the
YouTube ID used by the parent dataset. Use the table to map clip names to IDs.
All four corners in each yellow-court (`E8WW8DFCnwk`) frame are extrapolated
from the clicked landmarks. Scores against those corners measure agreement
with the fitted projection, rather than directly clicked corner positions.

## View the overlays

| Clip | Saved frame overlays |
| --- | --- |
| Yellow court | [14](renders/E8WW8DFCnwk_sample_f14.jpg), [90](renders/E8WW8DFCnwk_sample_f90.jpg), [156](renders/E8WW8DFCnwk_sample_f156.jpg) |
| Letterboxed court | [45](renders/Cb-xs5rPyxI_gameplay_f45.jpg), [58](renders/Cb-xs5rPyxI_gameplay_f58.jpg), [78](renders/Cb-xs5rPyxI_gameplay_f78.jpg) |
| Centre court | [36](renders/l-I_Di1Ad2Y_h264_f36.jpg), [64](renders/l-I_Di1Ad2Y_h264_f64.jpg), [71](renders/l-I_Di1Ad2Y_h264_f71.jpg) |

Thin lines show the fitted court markings. The thick outline shows the outer
boundary; circles mark clicked landmarks and crosses mark visible corner
positions. Offscreen corners cannot be drawn within the image.

To regenerate an overlay, decompress both CSVs into the same temporary directory
with their original `.csv` names. Run
`src/courtkeynet/validation_scripts/render_ground_truth.py` with `--corners-csv`,
`--video`, `--frame` and `--out`. The renderer currently reads uncompressed CSVs.
