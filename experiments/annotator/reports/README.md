# Experiment reports

These reports contain the measurements, methods and rerun commands behind the
[development narrative](../development.md). They are grouped by the question
they address. Results and reproduction instructions share one document.

## Contact detection and rally structure

| Question | Report |
|---|---|
| Did searching further around rally boundaries recover useful hits? | [Boundary search](boundary_search.md) |
| Did a different wrist-distance measure improve player assignment? | [Player assignment](player_assignment.md) |
| How did learned contact detection and sequence repair develop? | [Development narrative](../development.md#late-august-learn-which-candidate-frames-are-hits), with links to the original studies |

## Court evidence

| Question | Report |
|---|---|
| How much did repaired courts help annotation? | [Court repair](court_repair.md) · [reproduction](court_repair.md#reproduction) |
| Which alternative court-fitting methods were tried? | [Court fitting](court_fitting.md) |
| What made plausible courts difficult to rank correctly? | [Court selection](court_selection.md), a historical assessment of saved candidates |
| What evidence supported sharing a court across frames? | [Temporal court evidence](court_temporal.md), including the earlier three-frame comparison |
| What changed when the detector sampled or combined several frames? | [Court sampling](court_sampling.md) |

## Model fitting and selection

| Question | Report |
|---|---|
| Why did freshly fitted models differ after the refactor? | [Refit investigation](refit_regression.md) · [reproduction](refit_regression.md#reproduction) |
| How was the current model selected, and how does it compare? | [Model selection](model_selection.md) · [reproduction](model_selection.md#reproduction) |

Saved CSVs, arrays, images and exact implementation excerpts remain with their
experiment outputs. The reports link to them where they support a finding.
