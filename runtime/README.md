# `runtime/`

This directory retains the non-API runtime files used by BRIC tooling and
classifier comparisons.

## `checkpoints/`

`src/bric/` uses these locations as its defaults for optional perception
weights:

- `checkpoints/tracknetv3/` for TrackNetV3 and InpaintNet
- `checkpoints/yolo11/` for the YOLO player detector

The weight files are gitignored. The tracked `.gitkeep` files preserve the
expected directory layout.

## `deployed/bric/`

The tracked BRIC run contains its manifest, metrics, predictions and evaluation
results. These files remain because classifier comparison tooling reads them,
including `scripts/plots/bar_chart_overall_shuttleset_comparison.py`.

The retired web API's upload, job, clip and database directories are no longer
part of this tree. Its former deployment instructions are preserved in
`docs/archive/runtime_api_deployment.md`.
