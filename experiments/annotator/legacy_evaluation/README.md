# Historical fixture evaluation

Use these helpers to rerun the earlier three-video ShuttleSet measurements:

- `fixtures.py`: fixture paths, metadata and expected input digests.
- `gt_scoring.py`: fixture loading, annotation execution and comparison with labels.
- `scoring.py`: rally matching and greedy contact-timing metrics.
- `static_courts.py`: supplied ShuttleSet calibration for the measurement runner.
- `perspective_shift.py`: the retired camera-change signal used in replay studies.

Current model fitting and evaluation use `src/annotator/training/`. Current
replay exclusion uses court absence and shuttle speed; camera changes alone do
not exclude frames.
