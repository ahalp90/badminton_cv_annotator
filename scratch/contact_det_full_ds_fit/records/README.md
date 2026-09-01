# Experiment JSON records

These small records keep fixed experiment definitions, input receipts and headline results together. The main reports link here when a result needs its saved evidence.

**Jump to:** [Fixed definitions](#fixed-definitions) · [Input and provenance receipts](#input-and-provenance-receipts) · [Result summaries](#result-summaries) · [Other evidence](#other-evidence)

## Fixed definitions

| Record | Purpose | Main reader or producer |
| --- | --- | --- |
| [`shuttleset_development_split.json`](shuttleset_development_split.json) | The 40 development videos and fixed 32/8 split | Shared experiment configuration |
| [`baseline_runs.json`](baseline_runs.json) | The nine RF/HGB runs and event-setting search space | [`baseline_config.py`](../scripts/baseline_config.py) and the baseline runners |
| [`training_video_score_groups.json`](training_video_score_groups.json) | Four groups for out-of-fold scores on the 32 training videos | [`score_training_videos.py`](../scripts/score_training_videos.py) |
| [`rally_start_model_runs.json`](rally_start_model_runs.json) | The six rally-start model and cut-off choices | [`run_rally_start_model.py`](../scripts/run_rally_start_model.py) |
| [`final_video_score_groups.json`](final_video_score_groups.json) | Five groups for out-of-fold scores on all 40 development videos | [`score_final_contact_groups.py`](../scripts/score_final_contact_groups.py) |

## Input and provenance receipts

| Record | Purpose | Producer or rebuild path |
| --- | --- | --- |
| [`pilot_feature_check.json`](pilot_feature_check.json) | Exact replay check for the three pilot videos | [`freeze_contact_features.py`](../scripts/freeze_contact_features.py) |
| [`training_video_score_inputs.json`](training_video_score_inputs.json) | Expected training-score inputs, row counts and hashes | [`score_training_videos.py`](../scripts/score_training_videos.py) |
| [`training_rally_start_input_summary.json`](training_rally_start_input_summary.json) | Receipt for label-free rally-start inputs from training videos | [`save_training_rally_start_inputs.py`](../scripts/save_training_rally_start_inputs.py) |
| [`validation_rally_start_input_summary.json`](validation_rally_start_input_summary.json) | Receipt for the corresponding validation inputs | [`save_validation_rally_start_inputs.py`](../scripts/save_validation_rally_start_inputs.py) |
| [`final_contact_score_inputs.json`](final_contact_score_inputs.json) | Expected all-40 scoring inputs, row counts and hashes | [`score_final_contact_groups.py`](../scripts/score_final_contact_groups.py) |

## Result summaries

| Record | Purpose | Producer or rebuild path |
| --- | --- | --- |
| [`baseline_summary.json`](baseline_summary.json) | Nine-run comparison and chosen eight-video result | Baseline scoring scripts, including [`score_contact_baseline.py`](../scripts/score_contact_baseline.py) |
| [`missed_contact_summary.json`](missed_contact_summary.json) | Missed first/later contacts and one-short sections | [`check_missed_contacts.py`](../scripts/check_missed_contacts.py) |
| [`rally_start_candidate_summary.json`](rally_start_candidate_summary.json) | Candidate-list size, coverage and gate result | [`check_rally_start_candidates.py`](../scripts/check_rally_start_candidates.py) |
| [`training_video_score_summary.json`](training_video_score_summary.json) | Out-of-fold contact scores across the 32 training videos | [`score_training_videos.py`](../scripts/score_training_videos.py) |
| [`rally_start_model_summary.json`](rally_start_model_summary.json) | Six small-model results and the stopped training gate | [`run_rally_start_model.py`](../scripts/run_rally_start_model.py) |
| [`shuttleset22_test_summary.json`](shuttleset22_test_summary.json) | Frozen test result, independent contact recount and rally-section recount | [`score_shuttleset22_test.py`](../scripts/score_shuttleset22_test.py) and [`summarise_shuttleset22_sections.py`](../scripts/summarise_shuttleset22_sections.py) |

## Other evidence

The ignored `../raw/` directory holds the large score arrays, predictions, fitted model and detailed results behind these records. The `../archive/` directory holds historical plans, stage reports and the worklog.

Some JSON fields name their source files by basename. Those basenames are preserved as provenance from the original run; moving this directory does not change the recorded evidence.
