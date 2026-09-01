# Contact detector follow-up: what matters

One change is worth keeping. The whole-rally side rule chooses player sides across the rally instead of judging each contact alone. On the held-out 47-video ShuttleSet22 test, it raised fully correct sections from **483 to 901 of 3,982** at the main ±5-frame tolerance. It repaired 418 sections and did not make any previously correct section wrong.

Here, a section is one predicted rally. It is fully correct only when every contact time and player side is right and no extra contacts remain. The ±5 tolerance allows a predicted contact to fall up to five frames from its label on a 30 fps clock.

The rule improves player-side assignments. It does not recover missing contacts or remove false ones. Most ShuttleSet22 sections therefore remain wrong: **22.63%** are fully correct after the rule.

The test results were not separated by broadcast convention, so they do not yet establish that the gain will generalise to unfamiliar broadcasts.

![On the held-out 47-video ShuttleSet22 test, the whole-rally side rule nearly doubled fully correct sections. It repaired 418 sections and made no previously correct section wrong.](figures/01_complete_rallies.png)

## What to keep, investigate, or leave alone

| Part of the follow-up | What happened | What to do |
| --- | --- | --- |
| Whole-rally side rule | Repaired 418 ShuttleSet22 test sections and made no previously correct section wrong at ±5 frames | Use it for complete rallies. Keep the old assignments when contacts must stand alone. |
| Global contact cut-off | On 40 ShuttleSet development videos, 0.85 corrected the contact count and timing in 139 sections but made 126 other contact lists incorrect. Player sides were not tested. | Keep 0.90 for the main ±5 measure. |
| First-contact model | Repaired six sections and made no previously correct section wrong on eight ShuttleSet validation videos | Treat this as evidence of a weak signal. Trace missing, excluded, and badly ranked first contacts before training again. |
| Close-duplicate cleanup | The audit found no opposite-side contacts within two frames of each other in the 40 ShuttleSet development videos or 47 ShuttleSet22 test videos. | Leave this path alone. |
| Deletion model | On 32 ShuttleSet training videos, it repaired 42 sections and made 88 previously correct sections wrong. | Do not add it. Revisit only with new evidence that can separate false contacts from real ones. |
| Keep-or-review model | Reached 40.87% precision while accepting 16.14% of sections on ShuttleSet development data | Do not use it for automatic acceptance. Find inputs that distinguish complete rallies from incomplete ones, then test any replacement across whole broadcast families. |

## The standalone annotator is still a distant goal

A standalone tool could reject most rallies and keep only the ones it expects to be completely correct. The tested keep-or-review model could not find that dependable subset. Stricter cut-offs did not help: precision stayed near 50% even when the model accepted less than 1% of the ShuttleSet development sections. It was not tested on ShuttleSet22.

![On 32 ShuttleSet training videos, the rally-level acceptance model remained far below the 90% precision target at every tested coverage. This model was not tested on ShuttleSet22.](figures/04_keep_review_curve.png)

The ShuttleSet22 test videos were separate from the 40 ShuttleSet development videos. Predictions and settings were saved before the test labels were opened. However, the results were not separated by camera layout, tournament, graphics package, or broadcast style. They do not yet show that the gain will hold across unfamiliar broadcast conventions.

All experiments used saved outputs from the existing contact model. They did not retrain the vision model or change the upstream tracker.

## Choose what to read next

- Read the [full report](report.md) to understand the pipeline, each experiment and what the results mean.
- Use [evidence and reproduction](evidence.md) for the exact video splits, saved result files and rebuild commands.
- Use [useful next work](next_steps.md) to continue the first-contact and broadcast-generalisation investigations.
