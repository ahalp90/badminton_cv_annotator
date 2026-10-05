# Report figures

The [main report](../README.md) uses ten figures to explain practical use, the court change and the remaining errors. The other figures support the detailed evaluation and development history.

## Main report

| Figure | Question answered |
|---|---|
| [Performance overview](performance_overview_rallies.png) | How accurate is contact detection overall, how many whole rallies are fully correct, and what quality and coverage does confidence selection provide? |
| [High-confidence clips](high_confidence_selection.png) | How many selected clips are exact, need corrections, cut off a rally or cannot be assessed? |
| [Selected-clip errors](selected_errors.png) | Which errors occur together in the 117 incorrect high-confidence annotations? |
| [Court-decision groups](court_change_groups.png) | How do contact recovery and rally gains/losses change with court acceptance? Each panel identifies its old-model baseline. |
| [Video variation](video_variation.png) | How much do contact recovery and complete-rally correctness vary between videos? |
| [Video outcomes, first half](video_outcome_breakdown_1.png), [second half](video_outcome_breakdown_2.png) | Which videos lose contacts or correct player assignments, and how many complete rallies remain? |
| [Court and player availability](upstream_context.png) | Which inputs were available at each missed contact? |
| [Shuttle guard](final_hit_guard.png) | How often does the guard flag false final contacts and the real contacts used for comparison? |
| [Position within a rally](contact_position.png) | How often are serves, middle contacts and final contacts missed in court-accepted frames? |

Counts and percentages are printed on the plots. The main report uses the repository's blue and sand palette, with grey for additional groups. This avoids relying on red–green contrast. Labels, legends and annotations have space around them. The video scatter plot uses 0–100% on both axes.

## Supporting figures

| Figure | Purpose |
|---|---|
| [Contact metrics](results_at_a_glance.png) | Timing and timing-plus-player precision, recall and F1 |
| [Rally gains and losses](rally_change_decomposition.png) | Net changes by court decision, supporting the court comparison |
| [Historical development](system_progression_trusted.png) | Cumulative development stages on the earlier 3,422-rally, 47-video benchmark |
| [Timing offsets](timing_offsets.png) | Cumulative distance from labels among matched contacts; missed contacts are excluded |
| [Rally coverage](rally_coverage.png) | Complete, incomplete and unreached rallies before confidence selection |
| [Selection](selection.png) | Correct, wrong and unassessed clips kept or discarded |
| [Court comparison](court_comparison.png) | Aggregate rally totals, training-seed variation and per-video gains/losses |
| [Court-input comparison](court_input_contact_recall.png) | Missed contacts under the old-court refit and new-court annotator |
| [Rally correctness](rally_correctness.png) | Exact contact sequences and fully correct rallies, with and without video 53 |
| [Contact correctness](contact_correctness.png) | Timing and timing-plus-player recovery for those populations |
| [Review queue](review_queue.png) | Correct, wrong and unassessed selected clips for those populations |
| [Misses by input state](misses_by_input_state.png) | Court and player state at missed contacts |
| [Original ShuttleSet comparison](original_comparison.png) | Validation and held-out development results alongside ShuttleSet22 |

The [interactive video breakdown](../VIDEO_BREAKDOWN.html) provides more detail for each video and can be opened locally. Removing video 53 is a sensitivity check; it changes the evaluated population without changing the saved predictions.

## Footage and court evidence

- [Video 53 court](video53_court_fixed.png) — the old and new outlines on the checked frame.
- [Video 17 court](video17_court_fixed.png) — the corrected outline retains the visible far player.
- [Random contact sample](contact_sample_results.png) — judgements for 24 randomly sampled misses.
- [Targeted label checks](label_check_results.png) — judgements for 28 cases from the largest error groups.

The court overlays use the same frames as the [earlier court evidence](../../../../scratch/annotator_wrapup_evaluation/figures/README.md#source-frame-evidence).

## Regeneration

From the repository root, after the report is placed in its experiment directory:

```bash
python experiments/annotator/annotator_closeout_new_courts/scripts/plot_readme_summary.py
python experiments/annotator/annotator_closeout_new_courts/scripts/plot_closeout_story.py
python experiments/annotator/annotator_closeout_new_courts/scripts/plot_readme_details.py
python experiments/annotator/annotator_closeout_new_courts/scripts/plot_video_variation.py
```

The four scripts write fourteen figures beside the report. While working in a nested draft folder, run the scripts from that folder's `scripts/` directory instead; each script writes to its own parent directory's `figures/` folder.

- `plot_readme_summary.py` draws contact metrics, selected clips and historical development. It checks current counts against `results/` when all required tables are present. Otherwise it uses the published snapshot embedded in the script.
- `plot_closeout_story.py` draws the combined court-decision comparison, missed contacts by court input, rally gains/losses and shuttle-guard rates from the published counts.
- `plot_readme_details.py` draws selected-clip errors, cumulative timing accuracy, accepted-frame miss rates and input availability from the evaluation tables.
- `plot_video_variation.py` draws the scatter plot and two per-video panels from `results/per_video.csv.gz`. Use `--results <directory>` to read that table from outside a nested draft folder.

The remaining figures use `plot_evaluation.py`, `plot_court_cases.py`, `summarise_judgements.py` and `summarise_label_checks.py`. Run `plot_evaluation.py` before the four scripts above: it also writes some of the same filenames. Court overlays need the earlier saved frames and local court records. The judgement-summary scripts need their original local JSON inputs. [Methods and reproduction](../evaluation_reproduction.md) records those dependencies.
