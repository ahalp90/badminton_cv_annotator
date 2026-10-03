# Evaluation of the released court detections

The reusable prediction files are in
[`data/court_detections/sset_and_sset22/extractions_20261003/`](../../../data/court_detections/sset_and_sset22/extractions_20261003/README.md).
This directory holds their accuracy measurements and visual review.

The [results report](../report.md) explains the changes. The
[gallery](gallery.md) shows the predicted courts, and the
[reproduction guide](../reproduce.md) rebuilds these measurements and images.

| File or directory | Use |
| --- | --- |
| `per_video.csv.gz`, `summary.json.gz` | Per-video measurements and full-corpus summary |
| `per_scene.csv.gz` | Final corners and reference errors for every scene |
| `per_rally.csv.gz` | Each labelled rally scored using its longest-overlapping scene |
| `rally_views/` | Each rally scored using the camera group occupying most of its duration; this supplies the report's rally-view results |
| `comparison_*.csv.gz` | Before/after measurements on the same videos and scenes |
| `pooling_diagnostic.csv.gz` | Cases where shared courts need closer investigation |
| `courts/`, `hard_samples/`, `rally_samples/` | One typical court per video, eight large reference disagreements, and stills from three rallies |
| `render_requests.json.gz`, `rally_samples.csv.gz` | Frame choices and captions for reproducing the images |

Errors use the references' 1280 × 720 resolution. Native corner fields retain
the source video resolution. Default-camera errors do not establish accuracy
for alternate views.
