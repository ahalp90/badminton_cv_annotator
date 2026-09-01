# Closed experiment notes

These are the old notes from the RF and HGB contact tests.

They show how the work reached each choice. Some end with a next step that has since been finished. The current reports are one folder up.

## Table of contents

- [Where to start now](#where-to-start-now)
- [What is in this archive](#what-is-in-this-archive)
- [Experiment history](#experiment-history)
- [A note about old links](#a-note-about-old-links)

## Where to start now

The current reports are:

- [`../README.md`](../README.md) for a short summary;
- [`../VISUAL_QUICK_GUIDE.md`](../VISUAL_QUICK_GUIDE.md) for a visual summary;
- [`../baseline_report.md`](../baseline_report.md) for the development experiments;
- [`../shuttleset22_test_report.md`](../shuttleset22_test_report.md) for the final test on new videos; and
- [`../current_system_map.md`](../current_system_map.md) for how the detector works and where the results live.

## What is in this archive

| Group | Files | What they show |
| --- | --- | --- |
| Running state | `RESUME.md`, `campaign.yaml`, `plan.md` | What had been done, and what was next, at each point |
| Rules and decisions | `contract.md`, `decisions.md` | The data split, when labels could be read, and choices made before each test |
| Development comparison | `baseline_runs.md`, `feature_preparation_audit.md` and the baseline report in Git history | The nine RF/HGB runs and the check that rebuilt features matched the pilot |
| Missed-contact follow-up | `missed_contact_check_plan.md`, `missed_contact_report.md` | They show why the work turned towards rally starts |
| Rally-start follow-up | `rally_start_candidate_*`, `training_rally_start_input_*`, `validation_rally_start_input_report.md` and `rally_start_model_*` | Why the small model for adding a first contact stopped at 51.7% right |
| Final fit and test setup | `final_contact_fit_plan.md`, `final_contact_fit_report.md`, `shuttleset22_inpaint_plan.md`, `shuttleset22_test_plan.md` | How the final model was fitted and how all 47 predictions were saved before labels were read |
| Full history | `worklog.md` | The untouched record in date order |

## Experiment history

```text
experiment rules and split
    → nine RF/HGB runs
    → missed-contact check
    → rally-start candidate list
    → rally-start model stops below the 80% rule
    → fair scores for all 40 development videos
    → final HGB fit
    → 47-video ShuttleSet22 test
```

The saved JSON results remain one folder up. The scripts still read them from there.

## A note about old links

The old files have not been rewritten. Some links in the old worklog point to where files lived at the time. The filenames have not changed. This page points to the current reports.
