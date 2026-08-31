# Contact detector follow-up

This folder checks whether small choices after frame scoring can improve complete rally outputs.

The scripts read the existing baseline records from `scratch/contact_det_full_ds_fit/`. They do not copy the large feature arrays.

## Contents

- [Current state](#current-state)
- [Data rule](#data-rule)
- [Commands](#commands)

## Current state

The shared loader and baseline recount are in place. The next check chooses player sides across each rally.

## Data rule

Prediction commands run without labels. Scoring commands load the saved clean labels after predictions exist.

The 47-video test labels only score a final frozen choice. Development videos are used to choose settings and train small follow-up models.

## Commands

Run the baseline recount from the repository root:

```bash
~/.venvs/badminton-cicd/bin/python -m scratch.contact_det_followup.scripts.recount_baseline
```
