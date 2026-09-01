# Evidence and reproduction

The numbers in [the report](report.md) come from the files below. The commands rerun each experiment without requiring a reader to study the scripts first.

## Datasets and result types

The contact detector was developed on 40 ShuttleSet videos. The split used 32 training videos and eight validation videos. The final detector was trained on all 40 before it ran on ShuttleSet22.

The ShuttleSet pool contains videos 1–44, apart from 9, 10, 12, and 27. Its fixed validation videos are `sset_18`, `sset_22`, `sset_24`, `sset_25`, `sset_30`, `sset_31`, `sset_39`, and `sset_40`. The other 32 videos form the training set.

The held-out ShuttleSet22 test uses video IDs 8–13, 15–44, 46–55, and 57 in ShuttleSet22's own numbering. None of those 47 videos was used to develop the detector. The predictions and settings were saved before the ShuttleSet22 labels were opened.

The report keeps these result types separate:

- **Held-out ShuttleSet22 test:** a chosen detector or rule was scored once on 47 ShuttleSet22 videos after its predictions and settings had been saved
- **Held-out model result:** a model acted on videos excluded from the relevant training or model-selection choice
Some checks tried every permitted edit and compared each result with the ground truth. These checks show whether a useful repair was present among the saved options. They do not show that a running annotator can choose it.

Labels were allowed to train models and score saved predictions. A runnable rule or model did not use labels to choose an action for a rally.

## How the follow-up used the videos

| Video set | Videos | Purpose |
| --- | ---: | --- |
| ShuttleSet training set | 32 | Develop and compare the small follow-up models using four held-out groups |
| ShuttleSet validation set | 8 | Score the fixed first-contact model on videos that were not used to develop it |
| ShuttleSet development total | 40 | Choose the side rule and inspect global settings |
| Held-out ShuttleSet22 test | 47 | Score the baseline and chosen side rule once |

The development contact predictions are group-held-out. Each video was scored by a contact model that did not train on it. The small follow-up models use their own held-out or nested checks where the report says so.

## Timing tolerances

Frame distances use a 30 fps clock. The main score accepts a predicted contact within five frames of its label. The ±10 result repeats the same calculation with a wider allowance.

All success decisions use the ±5 result. The ±10 result provides context only.

## Result records

| Question | Saved evidence |
| --- | --- |
| What was the ShuttleSet22 baseline? | `results/baseline_recount.json` |
| Did the whole-rally side vote help? | `results/side_development.json`, `results/side_audit.json`, `configs/side_rule.json` |
| Were close opposite-side duplicates present? | `results/opposite_side_duplicate_audit.json` |
| Did another score cut-off or window for removing nearby copies help? | `results/setting_sweep.json` |
| How often could any allowed first-contact edit repair a rally? | `results/start_best_case.json` |
| Could a model choose those first-contact repairs? | `results/start_model_development.json`, `results/start_model_validation.json` |
| How often could any allowed start or deletion edit repair a rally? | `results/combined_best_case.json` |
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

The pipeline overview and first-contact flowchart have their own Mermaid sources. Rebuild them with:

```bash
/home/ariel/.venvs/skill-utils/bin/mermaidx \
  -i scratch/contact_det_followup/figures/00_pipeline_overview.mmd \
  -o scratch/contact_det_followup/figures/00_pipeline_overview.svg \
  --embed-font

/home/ariel/.venvs/skill-utils/bin/mermaidx \
  -i scratch/contact_det_followup/figures/00_pipeline_overview.mmd \
  -o scratch/contact_det_followup/figures/00_pipeline_overview.png \
  -w 1800 \
  -b white

/home/ariel/.venvs/skill-utils/bin/mermaidx \
  -i scratch/contact_det_followup/figures/05_first_contact_flow.mmd \
  -o scratch/contact_det_followup/figures/05_first_contact_flow.svg \
  --embed-font

/home/ariel/.venvs/skill-utils/bin/mermaidx \
  -i scratch/contact_det_followup/figures/05_first_contact_flow.mmd \
  -o scratch/contact_det_followup/figures/05_first_contact_flow.png \
  -w 1600 \
  -b white
```
