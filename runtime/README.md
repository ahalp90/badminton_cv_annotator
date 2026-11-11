# `runtime/`

Use this directory for deployed annotator and classifier bundles and optional
perception weights.

## `checkpoints/`

`src/bric/` uses these locations as its defaults for optional perception
weights:

- `checkpoints/tracknetv3/` for TrackNetV3 and InpaintNet
- `checkpoints/yolo11/` for the YOLO player detector

The dataset builder's court stage reads `checkpoints/deeplsd/DeepLSD`: a DeepLSD
checkout with its `weights/deeplsd_md.tar`. Place or link that checkout there.

The weight files are gitignored. The tracked `.gitkeep` files preserve the
expected directory layout.

## `deployed/bric/`

The tracked BRIC run contains its manifest, metrics, predictions and evaluation
results. These files remain because classifier comparison tooling reads them,
including `scripts/plots/bar_chart_overall_shuttleset_comparison.py`.

The retired web API's upload, job, clip and database directories are no longer
part of this tree. Its former deployment instructions are preserved in
`docs/archive/runtime_api_deployment.md`.

## `deployed/annotator/`

The default three-model annotator bundle is saved here in the normal model
format. Its contact model uses standing-person counts. See
[annotator model settings](../docs/annotator/tree_stack.md) for training and
inference behaviour.
