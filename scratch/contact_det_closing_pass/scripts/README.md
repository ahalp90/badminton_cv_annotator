# Report writers

Four scripts rebuild the closing-pass report from saved predictions. They never
retrain or rerun vision. Each one writes under `--report-dir` in the same layout
as `scratch/contact_det_closing_pass/`, so the reports' relative links keep working.

| Script | Reads | Writes under `--report-dir` |
|---|---|---|
| `write_acceptance_tables` | `--serve-results`: `chosen_acceptance_{development,broader}.json.gz`, `{development,broader}_serves.json.gz` | `results/serve_followups/acceptance_{breakdown.json,per_video.csv}.gz`, `figures/chosen_acceptance.png` |
| `write_serve_tables` | `--serve-results`: `{development,broader}_serves.json.gz` | `serve_discovery.md`, `results/serve_followups/serve_per_video.csv.gz`, `figures/serve_timing.png` |
| `summarise_metrics` | `--annotations` (ShuttleSet22 root), `--results` (stage predictions and `serve_followups/chosen_acceptance_broader.json.gz`) | `serve_tables.md`, `results/metric_summary.json.gz`, `figures/` |
| `regenerate_figures` | `--summary`, `--results` | `--output-dir` |

`--serve-results` defaults to the original `results/serve_followups/`, and
`--results` defaults to the original `results/`. Use `--trusted-labels` on
`summarise_metrics` to supply another saved clean-label file. Its default is the
original 47-video label set, including ShuttleSet22 video 15.

## Run order

From the repository root:

```bash
export PYTHONPATH="$PWD/src:$PWD"
module=scratch.contact_det_closing_pass.scripts
report=local_scratch/my_report   # any fresh folder
python -m $module.write_acceptance_tables --report-dir "$report"
python -m $module.write_serve_tables --report-dir "$report"
python -m $module.summarise_metrics --annotations /path/to/ShuttleSet22 --report-dir "$report" --reference-check
```

`--reference-check` asserts the original closing-pass counts. Leave it off for
a fresh run. `summarise_metrics` refuses to write into the original report
folder unless the reference check is on and every stage is scored.

## Fresh selected-model run

`--selected-only` scores only the chosen chain: original, serve repair,
whole-sequence, later contact, local insertion and the final detector. It skips
the boundary-only and wider-shortlist stages and the figures that read other
experiment files. The results root then needs only:

- `broader_predictions.json.gz` with `baseline`, `opening_only` and `combined` outputs
- `later/later_broader_predictions.json.gz`
- `followups/local_broader_predictions.json.gz`
- `followups/local_boundary_broader_predictions_fixed_membership.json.gz`
- `serve_followups/chosen_acceptance_broader.json.gz` with `frozen_policies.gap.comparison.threshold`
  and per-row `gap_score` and `judgements`

Each prediction file holds `videos` rows with `fixture`, `fps` and either
`output` or `outputs[<stage>]` in the format read by `run_later_broader.restore_stream`.
