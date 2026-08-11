# Current handover

The corrected investigation is complete on `investigation/serve-start-trajectory`. No production file under `src/**` changed.

Read [README.md](README.md), then [report.md](report.md). Those two files are enough to understand the question, result and next step. The short [decisions.md](decisions.md) and [findings.md](findings.md) are the working reference for a follow-up investigation. Do not load `ARCHIVE/` unless the history is actually needed.

## What is settled

- Use 239 one-to-one rallies for analyses that require one contact sequence per ground-truth rally.
- Use 249 covered rallies only as merge sensitivity. Use all 292 ground-truth rallies for the end-to-end view.
- Use ±10 base-30fps frames as the main contact-alignment tolerance. Keep ±5 and ±30 as sensitivity checks.
- The simple motion rule calls a path incoming when its robust fitted decrease reaches 0.05 apparent player body heights. The threshold was chosen before corrected scoring and was not swept.
- Keep residual scatter and trend-to-jitter as explanations, not extra decision thresholds.
- Compare direct motion correction with prepend/refit using the same earliest-contact fallback. The fair checked results are 163/239 and 159/239.

## Current interpretation

Motion helps when a usable path exists, but usable evidence exists in only 24/239 primary rallies. Anchor selection is the larger problem: 97/239 earliest contacts are unmatched at ±10, while later contacts often recover the serve or return.

The next investigation should improve which accepted contact starts the sequence, then increase path availability. Test the unchanged 0.05-BH rule on new videos before considering a more complicated classifier.

## Reproducing the result

The commands are in [README.md](README.md). Expected final checks are:

- input preparation verifies all three fixtures;
- the validator reports 292 rallies, 344 spans, 1,012 path points, 16 fixed-rule rows, all metrics, the report and six plots;
- the focused test file passes 55 tests;
- focused Ruff and `git diff --check` pass.

Generated inputs and outputs remain ignored. The required frozen release asset lives under `assets/`. Pose links still point to the separate local autograder fixture named in `prepare_inputs.py`.

## History

[ARCHIVE/ARCHIVE_MAP.md](ARCHIVE/ARCHIVE_MAP.md) describes the untouched full plan, worklog, findings, decisions and three useful review records from 10-11 August 2026. Superseded presentation reviews sit in a separate archive subfolder instead of the live tree.
