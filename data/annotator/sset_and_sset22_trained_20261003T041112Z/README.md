# Selected annotator model — 3 October 2026

This is the selected new-court **base** model. Pass this directory to the
annotator's `--models` argument, keeping `models.joblib` and `metadata.json`
together. It requires **scikit-learn 1.9.1** and was fitted and load-checked with
Python 3.12.13.

The contact tree was trained on 40 original-ShuttleSet development videos.
Sequence and review models were trained on the 32 videos in groups A–D.
ShuttleSet22 supplied evaluation data: 46 videos, excluding video 15 because
its labels are misaligned. These videos had also informed earlier investigations.
The directory name covers both datasets; only original ShuttleSet supplied
fitting examples. Its timestamp identifies the recorded refit invocation start,
**2026-10-03 04:11:12 UTC**.

The saved configuration uses contact cutoff **0.9**, keeps
`reject_masked_without_player=False`, and uses `side_geometry=video` for
contact-side assignment. On the 46-video ShuttleSet22 comparison, it recovered
**1,744 of 3,327 fully correct rallies** and **34,200 of 37,184 labelled hits**.

## Provenance

- Annotator code: `97b5e4de524fe2698492c7f72aceb2a5fddee6a3`
- Court release: `e0151791dd1c525311b497d460eff65b4c455021`,
  `experiments/court_detector/fast_robust_20261002/court_sharing_patched`
- Original fitted directory: `base/bundle`
- Contact training preserved the historical sampling order, then fitted the
  selected examples in video-ID order

The files were moved from the evaluated bundle without modification.
SHA-256 checksums:

```text
960cb9fb08a75742d6265fc3dcc8a3ebe8eed7974b86c02116c6f562bd6b9cba  models.joblib
da572d147af1664837a86430a6d2ba448a60d024c926e0f9dcbaf5f7740726f8  metadata.json
```
