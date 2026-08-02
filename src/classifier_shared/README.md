# Classifier-shared utilities

This package contains classifier utilities used by BRIC and BST-X.

| Module | Purpose |
| --- | --- |
| `taxonomy.py` | Classifier taxonomy registry, stroke mappings, and label derivation. |
| `dataset.py` | ShuttleSet paths, flaw parsing, split metadata, and clip bounds. |
| `player_mapping.py` | ShuttleSet A/B to Top/Bottom mapping and shot collection. |
| `eval_plots.py` | Precision and recall normalised confusion-matrix rendering. |
| `video_io.py` | Video metadata through `get_video_info`. |

Court geometry and TrackNetV3 are documented in [`shared`](../shared/).
