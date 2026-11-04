# Three-model annotator

The default bundle contains a contact classifier, one rally selector and a
separate review model. Training fits one selector on all training videos and
one per held-out group to provide review-training predictions.

The simplified selector matched the previous implementation on the eight
validation videos. Both runs used the same 32 training videos, vision inputs,
option pools, targets, weights, tree settings and seed 0. Validation includes
all 668 rallies, including the 314 marked with flaws.

| Validation measure | Previous implementation | One-selector implementation |
| --- | ---: | ---: |
| Fully correct rallies | 298/668 | 298/668 |
| Contact F1 | 90.209% | 90.209% |
| Correct-player F1 | 85.391% | 85.391% |
| Fully correct unflagged rallies | 227/354 | 227/354 |

These are validation results. Confirmation on the 46-video test set is pending.

## Runtime

Use scikit-learn 1.9.1 and the published XGBoost 3.4.1 package to load
[the default seed 0 bundle](models/raw_baseline-seed-0/). The project dependency
pins remain unchanged.

Fitting used XGBoost 3.4.2 on a V100. The three trees were exported through
XGBoost’s native UBJSON format, loaded in 3.4.1, and saved as the deployment
bundle in that runtime. The model metadata therefore records the 3.4.1 loading environment;
the fitting version remains 3.4.2. The annotator settings and tree roles are
unchanged. Keep the historical bundles in their corresponding source checkout.

As of 4 November 2026, XGBoost 3.4.2 is available as a source release but has
no matching PyPI wheel.
The [release notes](https://github.com/dmlc/xgboost/releases/tag/v3.4.2) describe
the Python 3.11 support change. Use the
[native model API](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html)
when exporting trees between runtimes. Retain the fitted bundle in its training
environment until the exported bundle has passed CPU annotation.

## Rebuild

Use the existing staged trainer with the raw-baseline preprocessing candidate,
seed 0, five-frame-preferred targets and the current tree settings. Reuse prepared
vision evidence and compatible contact fits and option pools. The
`--skip-replay-parity` switch skips the optional historical equality replay;
score the exported validation streams and run a fresh CPU annotation instead.

Production manifest-based fitting exposes contact radii and independent
selector/review tolerances. The staged sidecar experiment retains its fixed
5/10-frame policy. See [retuning](../../../../../docs/annotator/retuning.md)
for the fitting entry points and saved settings.
