# Earliest-contact serve trajectory investigation

This investigation asks whether the shuttle clearly approaches the player at the earliest accepted contact. Incoming motion would mean that the contact is probably the first return, so the other player probably served.

The earliest accepted contact comes from the ordinary contact detector. It is not a serve detector. The corrected analysis uses the contact player's geometry at that frame and never uses the released server label to choose that player.

Read `report.md` first. Its Bottom line gives the decision and action before the optional extended summary and technical sections.

## Result

- 292 GT rallies provide the end-to-end view.
- 249 rallies satisfy the current `COVERED` definition, including ten GT rallies in five merged predicted spans.
- 239 one-to-one rallies form the primary downstream set.
- At the main ±10 baseline, the earliest anchor is nearest 119 serves, 19 first returns, 4 later strokes and no GT stroke in 97 primary rallies. Five windows contain more than one GT stroke.
- Later contacts recover the serve in 49 of the 97 unmatched sequences. Another 36 recover the first return without recovering the serve.
- Only 24/239 primary rallies have usable recurrence-mask motion evidence. Nineteen of the 135 unique ±10 serve/return anchors have such evidence.
- The fixed 0.05-BH trend rule finds 9 of 17 returns and makes 4 false return calls. The unchanged historical rule finds the same 9 returns and makes 3 false calls.
- Applying the same 0.05-BH rule after excluding producer-marked inpainted points finds 7 returns and makes no false return calls, but usable paths fall from 19 to 10.
- Direct motion-based server attribution is correct in 163/239 primary rallies. Prepending a hypothetical contact and rerunning the alternating fit reaches only 125/239 or 127/239.

The 0.05-BH threshold is an engineering judgement fixed before corrected scoring. Residual scatter and trend-to-jitter are diagnostics only.

## Files

- `plan.md`: approved scope, exclusions and checkpoint messages.
- `decisions.md`: fixed population, alignment and motion-rule decisions.
- `findings.md`: verified repository and result findings.
- `experiment_data.py`: frozen input and GT loading.
- `trajectory_features.py`: alignment, sequence and trajectory arithmetic.
- `analyse_serve_trajectory.py`: row-level analysis and checked result bundle.
- `report_outputs.py`: final report and plots.
- `validate_outputs.py`: independent source-backed recalculation and output checks.
- `report.md`: standalone result.
- `review_feedback.md`: preserved fresh-reader WebUI audit and required revisions.
- `serve_trajectory_progressive_disclosure_feedback.md`: requested whole-report progressive-disclosure revision.
- `progressive_disclosure_feedback.md`: independent WebUI audit of that revision.
- `worklog.md`: compaction-safe session record and review history.

Generated inputs and outputs remain ignored. NumPy arrays use `.npy.xz`; JSON and CSV use `.json.gz` and `.csv.gz`.

The progressive-disclosure WebUI audit and final native cold read found no blocking defect. The source-backed validator binds the report, six plots and all checked numerical outputs. The final fourth-batch repository gates are recorded in `worklog.md`.
