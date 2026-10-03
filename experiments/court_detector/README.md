# Court-detector research

**The extracted courts are in
[`data/court_detections/sset_and_sset22/extractions_20261003/`](../../data/court_detections/sset_and_sset22/extractions_20261003/README.md).**
It contains the 86 prediction files and a loading example. This directory
contains the comparisons and tools used to improve the detector.

| What you need | Open |
| --- | --- |
| Results, chosen method and remaining errors | [report.md](report.md) |
| Images of the released courts | [Final gallery](released_dataset_evaluation/gallery.md) |
| Commands to reproduce measurements or images | [reproduce.md](reproduce.md) |
| Detector code and CLI usage | [Source README](../../src/court_detector/README.md) |

## Folder map

| Folder | Why it is here |
| --- | --- |
| [released_dataset_evaluation/](released_dataset_evaluation/README.md) | Accuracy tables and galleries for the released dataset |
| [baseline/](baseline/README.md) | Original results for the same 86 videos, before court-sharing repairs; needed for before/after comparisons |
| [search_policy_trial/](search_policy_trial/README.md) | Eight-video tests of alternative search and selection rules, with reviewed failure examples; explains why those changes were rejected |
| [inputs/](inputs/README.md) | Saved predictions, choices and timings for reproducing those comparisons without rerunning detection |
| [saved_views/](saved_views/README.md) | Fixed-frame fixtures and a diagnostic runner used when changing detector logic |
