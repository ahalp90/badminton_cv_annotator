# Shared runtime code

This package holds code used outside a single classifier. Annotator and
classifier consumers share the court geometry. BRIC, BST-X, and the scrape
lane share the TrackNetV3 inference tree.

## Modules

| Module | Purpose |
| --- | --- |
| `court.py` | Court homography, projection, normalisation, and reference dimensions. |
| `tracknetv3/` | TrackNetV3 and InpaintNet inference code. |

Classifier-only utilities live in [`classifier_shared`](../classifier_shared/).

## Public court surface

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
