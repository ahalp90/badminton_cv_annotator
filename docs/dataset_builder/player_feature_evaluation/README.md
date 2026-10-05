# Player feature evaluation: report material

The report uses [evaluation.md](evaluation.md) and the six figures in [appendix.md](appendix.md). Keep Figures A12–A17 together after A11. The first three show the linear fits discussed in the evaluation; the next three show the logistic and Spearman follow-up. The PNGs are the current report-insertion versions from the 5 October 2026 handover.

## Saved plotting inputs

The `data/` directory contains five compressed CSVs:

- `player-lookup.csv.gz`: the 43 player identities, profile measurements, groups and pooled rally outcomes.
- `player-match-results.csv.gz`: the 172 player-match records behind those outcomes and the logistic fits; each match has two player perspectives.
- `linear-fit-results.csv.gz`: the six linear fits and their statistics.
- `model-results.csv.gz`: the two logistic weighting choices and Spearman results for six features, including raw and adjusted p-values.
- `logistic-curve-values.csv.gz`: the saved coordinates for the 12 displayed logistic curves.

Outcomes were checked using point-winner labels and score progression in the original ShuttleSet and ShuttleSet22 set annotations. The match table identifies the source dataset, video and player pair. Profiles use the previously derived timing, court-position and shot-type measurements; tracking and court calibration were not independently revalidated for this analysis. The captions explain the denominators and method limitations. These are frozen plotting inputs, not a replacement for the upstream dataset.

## Reproduce the six PNGs

From the repository root:

```bash
MPLCONFIGDIR=/tmp/player-feature-figures-mpl uv run --no-project --with-requirements docs/dataset_builder/player_feature_evaluation/requirements.txt python docs/dataset_builder/player_feature_evaluation/render_figures.py
```

The renderer reads the bundled tables and rewrites only the six PNGs in `figures/`. It does not refit models, recluster players or change the report text. Figure numbering and captions are maintained in `appendix.md`.
