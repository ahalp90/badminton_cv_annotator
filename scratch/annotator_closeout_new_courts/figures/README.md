# Figures

These figures summarise the saved new-court results. All use the 46 ShuttleSet22 videos outside video 15, cleaned labels and ±10 frames at 30 fps unless the figure says otherwise. Earlier-model numbers appear in the report text, not the plots, except in the two court-comparison figures.

**Contents**  
[Exploring the results](#exploring-the-results)  
[Old versus new courts](#old-versus-new-courts)  
[Summary figures](#summary-figures)  
[Footage and court evidence](#footage-and-court-evidence)  
[Generate the figures](#generate-the-figures)

## Exploring the results

These plots show the spread and kinds of errors behind the headline scores. They are embedded in the [main report](../README.md).

| Figure | What it shows |
|---|---|
| [Video variation](video_variation.png) | Contact recovery against fully correct rally rate, one point per video |
| [Video outcomes, first half](video_outcome_breakdown_1.png), [second half](video_outcome_breakdown_2.png) | Correct-player matches, other matches and missed labels for every video |
| [Rally coverage](rally_coverage.png) | Complete, incomplete and unreached rallies before selection |
| [Selection](selection.png) | Correct, wrong and unknown clips kept and discarded |
| [Error combinations](selected_errors.png) | Which errors occur together in wrong selected clips |
| [Timing offsets](timing_offsets.png) | How close matched contacts are to labels; misses excluded |
| [Contact position](contact_position.png) | Serve, middle and final-contact miss rates at two tolerances |
| [Input conditions](upstream_context.png) | Court and player availability within missed and matched contacts |

The [interactive video breakdown](../VIDEO_BREAKDOWN.html) adds per-video player confusion, extra predictions, input conditions and the old-versus-new court result; open it locally.

## Old versus new courts

- [`court_comparison.png`](court_comparison.png) — fully correct totals for the historical model, the fresh old-court refit and the new model, against the 33-rally gap between two training seeds; rallies gained and lost in each video.
- [`court_change_groups.png`](court_change_groups.png) — matched hits and rallies gained and lost, grouped by whether the old and new courts accepted each scene.

## Summary figures

- [`rally_correctness.png`](rally_correctness.png) — exact rally-sequence and fully-correct-rally rates, with and without video 53.
- [`contact_correctness.png`](contact_correctness.png) — contact timing and timing+player recovery for the same populations.
- [`review_queue.png`](review_queue.png) — correct, wrong and unjudgeable selected clips for each population.
- [`misses_by_input_state.png`](misses_by_input_state.png) — where the 2,984 misses occur in the pipeline.
- [`original_comparison.png`](original_comparison.png) — original-ShuttleSet validation and held-out development videos against ShuttleSet22.

Removing a video changes the denominator. It does not repair saved output. The version without video 53 is a sensitivity view, not a better benchmark.

## Footage and court evidence

- [`video53_court_fixed.png`](video53_court_fixed.png) — the earlier video 53 frame with the old OpenCV-broken outline and the new outline.
- [`video17_court_fixed.png`](video17_court_fixed.png) — the earlier video 17 frame with the old shared outline and the new outline.
- [`contact_sample_results.png`](contact_sample_results.png) — direct hit/player checks of 24 randomly sampled missed contacts, split by court decision.
- [`label_check_results.png`](label_check_results.png) — label verdicts for 28 cases drawn from the largest failure groups.

Both court figures draw on the same source frames as the [earlier court evidence](../../annotator_wrapup_evaluation/figures/README.md#source-frame-evidence).

## Generate the figures

From the repository root, run:

```bash
python scratch/annotator_closeout_new_courts/scripts/plot_evaluation.py
python scratch/annotator_closeout_new_courts/scripts/plot_court_cases.py
```

The first script reads the tables in `results/` and writes every summary figure here. `plot_court_cases.py` makes the two court overlays. `summarise_judgements.py` makes `contact_sample_results.png`, and `summarise_label_checks.py` makes `label_check_results.png`. The court overlays need the earlier evaluation's saved frames and the pulled court records in the ignored `raw/` directory. [Methods and reproduction](../evaluation_reproduction.md) covers the steps that produce those inputs.
