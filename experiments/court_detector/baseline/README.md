# Original court detections: before the sharing repair

These are measurements and images from the original 86-video extraction.
They let the [main report](../report.md) compare the same scenes before and after
the court-sharing repair. The current reusable predictions live in the
[released dataset](../../../data/court_detections/sset_and_sset22/extractions_20261003/README.md).

| File or folder | Purpose |
| --- | --- |
| `per_video.csv.gz`, `summary.json.gz` | Per-video accuracy and corpus summary |
| `per_scene.csv.gz`, `per_rally.csv.gz` | Scene and rally measurements used to compare before and after sharing |
| [gallery.md](gallery.md), `courts/` | One original court from each video’s main camera view |
| [hard_samples/](hard_samples/README.md) | Large disagreements with the default-camera reference, for manual review |
| [rally_views/](rally_views/README.md) | Comparison of two ways to choose a court for a rally |
| [view_checks/](view_checks/README.md) | Targeted investigation of camera grouping and player-based rejection |
| `pooling_diagnostic.csv.gz`, `veto_scene.mp4` | Diagnostic scene records and one rejection example |
| `render_requests.json.gz`, `render_captions.csv.gz` | Frame choices and captions for reproducing the images |

The original prediction files are saved in [inputs/videos/](../inputs/videos/).
[reproduce.md](../reproduce.md) contains the commands for the numerical comparison.
