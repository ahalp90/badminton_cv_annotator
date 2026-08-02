# Classifier-shared utilities

This package contains code shared by BRIC and BST-X without assigning it to
either classifier.

| Module | Purpose |
| --- | --- |
| `player_mapping.py` | ShuttleSet A/B to Top/Bottom mapping and shot collection. |
| `eval_plots.py` | Precision and recall normalised confusion-matrix rendering. |
| `video_io.py` | Video metadata through `get_video_info`. |

Court geometry remains in [`shared`](../shared/) because the annotator also
uses it.
