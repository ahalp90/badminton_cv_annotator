# Shared runtime code

Annotation, extraction and the classifiers reuse these helpers for court
geometry, video timing, shuttle tracks and progress reporting.

## Modules

| Module | Purpose |
| --- | --- |
| `court.py` | Convert image positions to court coordinates; load saved court geometry. |
| `court_model.py` | Court dimensions and painted lines in metres, with the far baseline at y=0. |
| `video_metadata.py` | Read exact video timing and check frame counts and timestamps. |
| `shuttle_track_validation.py` | Check shuttle-array shape, visibility and coordinate ranges. |
| `camera_cuts.py` | Find the first frame of each new camera scene. |
| `progress.py` | Report progress and redirect worker output while the terminal display runs. |
| `tracknetv3/` | TrackNetV3 and InpaintNet inference code. |

Classifier-only utilities live in [`classifier_shared`](../classifier_shared/).

## Court helpers

```python
from shared.court import (
    REF_COURT_M,
    REF_COURT_CORNERS_M,
    build_all_court_info,
    get_court_info,
    load_all_court_info,
    project,
)
```
