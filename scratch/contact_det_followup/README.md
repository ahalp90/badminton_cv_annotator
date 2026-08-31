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

The opposite-side duplicate check found no adjacent events within two frames in either saved post-merge prediction set. A cleanup rule would have no pair to change in the current outputs, so this line stops at the count.

The 57-setting sweep found a small timing gain from lowering the contact cut-off from 0.9 to 0.85. Timing-complete sections rose from 940 to 953 across the 40 development videos. The change repaired 139 sections and broke 126, for a net gain of 13. This is below the planned 25-section signal, so the global setting change stops here.

Five leave-one-group sensitivity checks also chose 0.85 with the existing six-frame merge distance. These are not clean outer held-out tests. The saved model for one of the other groups may have trained on the omitted group. The stop decision uses the more generous descriptive result above: even the setting chosen on all 40 videos adds only 13 net sections.

The lower cut-off raised first-contact recall from 49.4% to 53.5%. Contact F1 fell slightly from 88.49% to 88.38%. The compact raw scores do not contain player sides for most alternative frames, so this sweep measures complete timing only. It does not count fully correct contact-and-side outputs.

The first-contact best-case check uses the 32 A-D development videos. It leaves the eight V videos aside for the final model check. Labels may choose keep, add, or replace for this ceiling only.

The current predictions have 745 sections with complete contact timing. The best allowed timing choice raises this to 1,063. It repairs 318 sections, spread across all four video groups. Adding an earlier event can repair 259 sections. Replacing the current first event can repair another 59.

The old timing-and-current-side target finds only 74 repairs. Choosing contact timing first and then applying the whole-rally side vote finds 300. This clears the planned 40-section signal, so the next step is a small label-free action model.

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

Run the opposite-side duplicate count:

```bash
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.audit_opposite_side_duplicates
```

Run the 57-setting timing sweep:

```bash
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.score_setting_sweep
```

Run the first-contact best-case check:

```bash
~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_followup.scripts.score_start_best_case
```
