# Contact detector follow-up

This folder checks whether small choices after frame scoring can improve complete rally outputs.

The scripts read the existing baseline records from `scratch/contact_det_full_ds_fit/`. They do not copy the large feature arrays.

## Contents

- [Current state](#current-state)
- [Data rule](#data-rule)
- [Commands](#commands)

## Current state

The shared loader and baseline recount are in place.

The first follow-up applies the existing alternation vote to each complete contact list. The vote chooses between the two possible Top/Bot patterns. Its minimum vote gap was chosen on the 40 development videos.

On development, the vote raised fully correct sections from 478 to 903. It repaired 426 sections and broke one.

On the frozen 47-video test set, fully correct sections rose from 483 to 901. It repaired 418 sections and broke none. This captures 418 of the 434 sections in the descriptive side-only ceiling.

The result has a trade-off. Test-set side accuracy across individual matched contacts fell from 92.0% to 91.1%. Contact-and-side F1 fell from 75.7% to 75.1%. The vote is useful when the whole alternating rally is the output that matters.

## Data rule

Prediction commands run without labels. Scoring commands load the saved clean labels after predictions exist.

The 47-video test labels only score a final frozen choice. Development videos are used to choose settings and train small follow-up models.

## Commands

Run the baseline recount from the repository root:

```bash
~/.venvs/badminton-cicd/bin/python -m scratch.contact_det_followup.scripts.recount_baseline
```

Rebuild and score the development side choices:

```bash
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.write_development_side_predictions
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.score_development_sides
```

The development scorer writes the chosen vote gap. Use it to rebuild and score the frozen test choices:

```bash
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.write_side_predictions
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.score_side_audit
```
