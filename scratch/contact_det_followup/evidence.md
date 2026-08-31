# Evidence and reproduction

The numbers in [the report](report.md) come from the files below. The commands rerun each experiment without requiring a reader to study the scripts first.

## Three kinds of result

The report keeps these result types separate:

- **Frozen test:** a fixed rule was scored once on 47 test videos after its choices were made on development data
- **Held-out prediction:** a model acted without labels on videos excluded from that model's training
- **Best-case check:** labels chose the best allowed edit for each rally; this measures available room, not deployable performance

Labels were allowed to train models and score saved predictions. They were not allowed to choose an action for a rally when a label-free rule ran.

## Video groups

| Name used in the report | Videos | Purpose |
| --- | ---: | --- |
| A–D | 32 | Develop and compare the small follow-up models |
| V | 8 | Final untouched check for the chosen first-contact model |
| Development total | 40 | Choose the side rule and inspect global settings |
| Frozen test | 47 | Score the fixed baseline and side rule once |

The development contact predictions are group-held-out. Each video was scored by a contact model that did not train on it. The small follow-up models use their own held-out or nested checks where the report says so.

## Timing tolerances

Frame distances use a 30 fps clock. The main score accepts a predicted contact within five frames of its label. The ±10 result repeats the same calculation with a wider allowance.

All success decisions use the ±5 result. The ±10 result provides context only.

## Result records

| Question | Saved evidence |
| --- | --- |
| What was the frozen baseline? | `results/baseline_recount.json` |
| Did the whole-rally side vote help? | `results/side_development.json`, `results/side_audit.json`, `configs/side_rule.json` |
| Were close opposite-side duplicates present? | `results/opposite_side_duplicate_audit.json` |
| Did another cut-off or merge distance help? | `results/setting_sweep.json` |
| How many first-contact repairs existed? | `results/start_best_case.json` |
| Could a model choose those first-contact repairs? | `results/start_model_development.json`, `results/start_model_validation.json` |
| How much room did one start edit and one deletion provide? | `results/combined_best_case.json` |
| Could a model choose safe deletions? | `results/delete_model_development.json` |
| Could a model accept only trustworthy rallies? | `results/keep_review_development.json` |

The older one-rally error counts used in figure 2 are recorded in the committed `scratch/contact_det_full_ds_fit/shuttleset22_test_report.md`. The strict 483-of-3,982 baseline comes from the follow-up recount.

## Rebuild the records

Run these commands from the repository root. Prediction commands do not load labels. Scoring commands load labels after predictions have been written.

### Baseline and player sides

```bash
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.recount_baseline

~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.write_development_side_predictions
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.score_development_sides

~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.write_side_predictions
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.score_side_audit
```

### Contact-list checks

```bash
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.audit_opposite_side_duplicates
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.score_setting_sweep
```

### First-contact checks

```bash
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.score_start_best_case
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.score_start_model
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.write_start_validation_predictions
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.score_start_validation
```

### Whole-rally edits and review model

```bash
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.audit_combined_best_case
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.score_delete_model
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.score_keep_review
```

## Rebuild the figures

```bash
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.plot_visual_guide
```

The plotting script reads the committed result records and writes matching PNG and SVG files to `figures/`.
