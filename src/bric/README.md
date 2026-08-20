# BRIC — Badminton RGB Inference Classifier

Alternative stroke classifier to BST. R(2+1)D-18 over RGB player crops,
fused with shuttle-trajectory features and player coordinates.

## Layout

```
src/bric/
├── network.py                # R(2+1)D-18 + fusion head
├── dataset.py                # ShuttleSetDataset, reads NPZ caches
├── train.py                  # CLI training entry point
├── infer.py                  # Inference handler (planned, PR 4)
├── eval.py                   # Test-set evaluation (planned, PR 3)
├── smoke_test.py             # `uv run python -m bric.smoke_test`
├── perception/               # BRIC's perception stack
│   ├── players.py            # YOLO11 + ByteTrack
│   ├── shuttle.py            # TrackNetV3 wrapper
├── preprocessing/            # Cache producers (run once per source video)
│   ├── slice_rallies.py      # Rally-level mp4 extraction
│   ├── preprocess_videos.py  # YOLO+ByteTrack → players cache + RGB tensors
│   └── extract_shuttle.py    # TrackNetV3 → shuttle cache
└── diagnostics/              # Cache + data quality checks
    ├── evaluate_players.py   # Player cache visibility / outliers
    ├── evaluate_shuttle.py   # Shuttle cache visibility / verdict
    ├── validate_court_positions.py
    ├── validate_rgb.py       # Contact-sheet JPGs from RGB cache
    └── debug_court_bias.py   # Per-stroke court coord diagnostics
```

Court geometry and TrackNetV3 live in `src/shared/`. Taxonomy, dataset helpers,
player mapping, plotting, and video metadata live in
`src/classifier_shared/`. A
per-stroke clip slicer (`shared/slicer.py`) is planned for PR 4 as
the cross-arch primitive for the live-upload inference path.

## Import rules

- `bric.*` modules **never** `import bst_x.*`.
- They import cross-pipeline court geometry from `shared.*`.
- They import classifier-only utilities from `classifier_shared.*`.
- They import their own perception layer from `bric.perception.*`.

## What's trained by us

**BRIC (R(2+1)D-18 + fusion head)** — fine-tuned on ShuttleSet
single-stroke clips with ground-truth stroke timing. Everything else
(TrackNetV3, YOLO11) is pretrained and frozen. Heuristic stroke
localisation (SRA / swing detector) is deferred to v2.

## Rally clip preprocessing

`shots_master.csv` stores one rally context interval on every stroke row. The
half-open interval starts two seconds before the first ShuttleSet contact and
ends three seconds after the final contact. Both bounds are clamped to the
canonical source video. Regeneration therefore requires the raw videos under
`training/data/shuttleset/raw_video/`:

```bash
python -m scripts.build_shots_master
uv run python -m bric.preprocessing.slice_rallies
uv run python -m bric.preprocessing.extract_shuttle
```

Run the stages in that order after changing rally bounds or canonical source
videos. The slicer writes a `.bounds.json` sidecar beside each rally clip. A
clip without matching bounds metadata is regenerated instead of skipped. The
TrackNet cache stores the same rally bounds and source frame count, so an old
cache is also regenerated. `--force` remains available when the stored
metadata matches but a manual rebuild is needed.
