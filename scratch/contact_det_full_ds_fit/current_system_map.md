# How the contact detector works

This page describes the contact detector and points to its code and saved results.

**Pipeline:** [System picture](#the-system-in-one-picture) · [Inputs](#what-goes-into-the-detector) · [Sections](#how-rally-sections-are-found) · [Events](#how-a-contact-becomes-an-event) · [Section scoring](#how-complete-sections-are-judged)

**Reference:** [Video use](#which-videos-were-used-at-each-step) · [Limits](#what-each-part-contributes-and-where-it-stops) · [Code](#code-map) · [JSON records](#results-saved-as-json) · [Ignored files](#files-kept-out-of-git) · [History](#old-plans-and-worklog)

## The system in one picture

```text
shuttle track + player poses + rally sections
                    │
                    ▼
       possible contact frames chosen without labels
                    │
                    ▼
        85 physical and validity features
                    │
                    ▼
          HGB contact probability score
                    │
                    ▼
         keep scores of at least 0.9
                    │
                    ▼
     merge predictions within six base-30 frames
                    │
                    ▼
            Top or Bottom player side
                    │
                    ▼
          contacts inside detected sections
                    │
                    ▼
        contact timing + player side
        + is the whole section right?
```

“Base-30 frames” means that the frame count uses a 30 fps clock. The code changes that count to fit the real frame rate of each video.

## What goes into the detector

The contact detector works from prepared tracking and pose data rather than directly from RGB frames.

It uses data made by earlier parts of the annotator:

- shuttle coordinates and visibility;
- player pose points;
- player-side estimates;
- court-view and rally-section information; and
- saved candidate frames where a contact is plausible.

Earlier pipeline stages made those inputs, and this experiment treated them as given. Errors in the shuttle track, pose, court view or section edges can cause wrong contact results. Those earlier stages need their own tests.

The model uses 85 fields for each candidate frame. Some hold measured values. Others say whether the needed tracking or pose data was present.

The measured values include shuttle speed, changes in direction, impulse, distance to the nearest wrist and player ankle motion.

## How rally sections are found

The section finder looks for long periods when the shuttle is at rest or cannot be tracked.

At 30 fps, a rest must last at least 90 frames, or three seconds, to split two active parts of a video. A section begins at the start of an active part that contains a strong burst of shuttle speed. It ends when the next long rest begins.

The three-second rule finds a break between rallies. It does not add three seconds before and after each section.

The saved sections therefore have different amounts of room around the labelled contacts. Among the 2,515 clean ShuttleSet22 sections, the median was 1.0 second before the first contact and 2.9 seconds after the last contact. One quarter began within five frames of the first contact.

ShuttleSet22 has contact labels rather than exact rally-edge labels. The test can say whether all rally contacts are inside one section. It cannot say whether the visual clip edges are ideal.

## How a contact becomes an event

HGB gives every candidate frame a score from 0 to 1. A higher score means that the frame looks more like a contact.

The final setup keeps scores of 0.9 or more. It then joins nearby frames into one contact. “Nearby” means six frames on a 30 fps clock.

The merge step matters because one real contact can create several high-scoring nearby frames. Keeping them all would create extra events.

The chosen model uses:

- HGB rather than RF;
- original per-frame motion values;
- balanced class weights;
- up to 24 non-contact training rows for each real contact row; and
- the model details recorded in [`records/baseline_runs.json`](records/baseline_runs.json).

Real contacts are rare in the training data. Balanced class weights stop the many non-contact rows from dominating training.

A contact row marks a real contact. A non-contact row is a candidate frame that is not a contact. The model saw up to 24 non-contact rows for each contact row. This gave it more examples of frames it should reject.

## How complete sections are judged

Each predicted contact can match only one labelled contact. Each label can also match only once.

The main contact result reports matches within five frames on the 30 fps clock. The reports also show results for one, two and ten frames.

A **detected section** is a stretch of video found by the existing annotator.

The old development scorer called the 609 sections it kept **accepted sections**. The saved result does not say why it removed the other 68.

For the ShuttleSet22 cut-off check, a section is scored only if it remains after the filter, matches one labelled rally and has enough human player labels.

A detected section is fully correct only when:

- it maps to exactly one labelled rally;
- every labelled contact has one timing match;
- there are no extra predicted contacts;
- every predicted contact has a player-side answer; and
- every player-side answer is correct.

A section cannot be fully correct if it joins several real rallies. Sections that match no labelled rally are not part of this one-rally score.

That last rule made the old 16.60% whole-rally result look better than an all-output score. It counted 493 correct sections among the 2,969 sections that touched one rally. It left out 943 sections with no labelled rally and 70 with parts of several rallies.

The later recount uses every predicted section. It found 2,515 clean rally sections from 3,982 predictions. Rally-section precision was 63.16%. Those sections covered 2,515 of 3,422 usable labelled rallies, so recall was 73.50%.

Only 483 of all 3,982 sections were both clean and fully correct for contacts and player side. That is 12.13%.

## Which videos were used at each step

| Step | Videos | What it was for | How labels were kept separate |
| --- | ---: | --- | --- |
| Pilot replay | 3 | Check that the larger feature build matched the earlier work | Labels were read only after the features had been saved |
| Main model comparison | 32 train + 8 validation | Choose the RF/HGB design | Validation videos never trained their own model |
| Rally-start training check | 32 | Test six ways to choose an earlier contact | The first model had not trained on the video it scored |
| Final cut-off and join rule | 40 in five groups | Choose the 0.9 cut-off and six-frame distance | Each video was scored by a model trained on the other 32 |
| Final model fit | 40 | Train the chosen HGB model once | All settings had already been chosen |
| ShuttleSet22 test | 47 | See how the final model worked on new videos | All predictions were saved before any test labels were read |

Eight ShuttleSet22 matches also appear in the development data, so the test left them out. The test also left out three matches whose public videos could not be lined up with the official frame numbers.

## What each part contributes and where it stops

| Part | What it contributes | Where it stops |
| --- | --- | --- |
| HGB contact score | Finds useful contact candidates in new videos | Precision falls to 80.62% on ShuttleSet22 |
| Nearby-contact merge | Reduces duplicate events | Missing events remain missing |
| First-contact handling | Often finds a possible contact near the missed first contact | The best tested choice was right only 51.7% of the time |
| Player side | 92.02% accuracy on answered five-frame timing matches | One wrong side spoils a whole section |
| Detected sections | Found 73.50% of labelled rallies | Only 63.16% of predicted sections held one whole rally and no part of another |
| Contact score | Ranks single contacts | Whole-rally correctness also depends on section edges, missing contacts, extra contacts and player sides |
| Check of the whole section | Tests the complete output that the project needs | Only 12.13% of all ShuttleSet22 sections are clean and fully correct at five frames |

## Code map

| Area | Main files |
| --- | --- |
| Shared experiment rules and split | `scripts/experiment_config.py`, `scripts/baseline_config.py` |
| Feature freeze and loading | `scripts/freeze_contact_features.py`, `scripts/feature_dataset.py` |
| Nine-run comparison | `scripts/run_baseline_menu.py`, `scripts/score_contact_baseline.py`, `scripts/baseline_results.py` |
| Missed-contact and rally-start checks | `scripts/check_missed_contacts.py`, `scripts/check_rally_start_candidates.py` |
| Held-out training scores | `scripts/score_training_videos.py` |
| Rally-start inputs and the model that stopped | `scripts/save_training_rally_start_inputs.py`, `scripts/save_validation_rally_start_inputs.py`, `scripts/rally_start_model.py`, `scripts/run_rally_start_model.py` |
| Out-of-fold development result used to choose the event rules, then final fit | `scripts/score_final_contact_groups.py`, `scripts/fit_final_contact_model.py` |
| ShuttleSet22 preparation and scoring | `scripts/inpaint_shuttleset22_tracks.py`, `scripts/prepare_shuttleset22_predictions.py`, `scripts/score_shuttleset22_test.py` |
| ShuttleSet22 rally-section recount | `scripts/summarise_shuttleset22_sections.py` |
| Report figures | `scripts/plot_report_figures.py` |

Each experiment script has tests under `tests/`. The final group-scoring tests also cover the model-fitting code.

From the repository root, rebuild all 14 report figures with:

```bash
MPLCONFIGDIR=/tmp/contact-det-matplotlib \
  ~/.venvs/badminton-cicd/bin/python -m \
  scratch.contact_det_full_ds_fit.scripts.plot_report_figures
```

This needs the local ignored input at `raw/final_contact_scores/combined_first/final_contact_setting_result.json` as well as the tracked records.

## Results saved as JSON

| File | What it records |
| --- | --- |
| [`records/shuttleset_development_split.json`](records/shuttleset_development_split.json) | The 40 development videos and the 32/8 split |
| [`records/pilot_feature_check.json`](records/pilot_feature_check.json) | The replay check for the three pilot feature files |
| [`records/baseline_runs.json`](records/baseline_runs.json) | The nine model runs and the event settings they could use |
| [`records/baseline_summary.json`](records/baseline_summary.json) | Eight-video comparison and chosen-run error counts |
| [`records/missed_contact_summary.json`](records/missed_contact_summary.json) | First/later miss diagnosis and one-short sections |
| [`records/rally_start_candidate_summary.json`](records/rally_start_candidate_summary.json) | Candidate-list size and target coverage |
| [`records/training_video_score_groups.json`](records/training_video_score_groups.json) | Four groups used so each training video was scored by a model trained on other videos |
| [`records/training_video_score_inputs.json`](records/training_video_score_inputs.json) | The training-score input files and their hashes |
| [`records/training_video_score_summary.json`](records/training_video_score_summary.json) | Scores across all 32 training videos, with each video kept out of its model |
| [`records/training_rally_start_input_summary.json`](records/training_rally_start_input_summary.json) | Training candidate lists made without reading labels |
| [`records/validation_rally_start_input_summary.json`](records/validation_rally_start_input_summary.json) | Validation candidate lists made without reading labels |
| [`records/rally_start_model_runs.json`](records/rally_start_model_runs.json) | The six tested ways to choose an earlier contact |
| [`records/rally_start_model_summary.json`](records/rally_start_model_summary.json) | All six results and why the work stopped there |
| [`records/final_video_score_groups.json`](records/final_video_score_groups.json) | The five groups used to score all 40 videos without training on them |
| [`records/final_contact_score_inputs.json`](records/final_contact_score_inputs.json) | The all-40 scoring files and their hashes |
| [`records/shuttleset22_test_summary.json`](records/shuttleset22_test_summary.json) | The final ShuttleSet22 result, the separate contact recount and the later rally-section recount |

[`records/README.md`](records/README.md) lists all 16 JSON records and their rebuild paths.

## Files kept out of Git

The ignored `raw/` folder holds the larger files behind these results:

- saved feature arrays;
- all nine model-run results and score arrays;
- validation predictions and detailed rally scoring;
- missed-contact and rally-start candidate details;
- training and final score arrays made without training on the video being scored;
- the fitted HGB model and reload record; and
- run logs and files from repeat checks.

The folder now contains 581 files and is about 363 MiB. It includes the full ShuttleSet22 prediction files and detailed test result recovered from the remote run.

The full raw folder is kept in the [RF/HGB contact study raw evidence release](https://github.com/ahalp90/badminton_cv_annotator/releases/tag/contact-det-rf-hgb-series-v1). The release has a file list and checksum. It leaves out local paths, remote setup details and `HANDOVER.md`.

## Old plans and worklog

The original plans, reports from each step, decisions and full worklog are under [`archive/`](archive/).

Those files show the work in the order it happened. Their “next step” sections are old. The reports in the main folder give the finished result.
