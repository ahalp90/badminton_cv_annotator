# Follow-up results

This directory holds results produced after the completed PR 80 investigation
was frozen.

Each follow-up gets its own report. New evidence belongs here and does not
rewrite the completed findings in the parent directory. A historical finding
changes only when the user explicitly requests a correction.

Follow [`../../WRITEUP_PRINCIPLES.md`](../../WRITEUP_PRINCIPLES.md) for every
report. After documenting one follow-up experiment, stop at that step's
boundary before starting the next experiment.

## Completed reports

- [`1_scene_comparison.md`](1_scene_comparison.md): Qwen and Intern on the same
  463 short scene clips. Intern is the provisional preference.
- [`2_final_model_gate.md`](2_final_model_gate.md): paired rally-start testing
  on 32 reviewed cases. Intern is the clean-interface choice and starting model
  for later follow-ups, but neither model handled serve state or contact timing
  reliably.
- [`3_precision_first_dataset.md`](3_precision_first_dataset.md): the frozen
  deterministic ladder could not select a non-empty zero-error rally subset on
  held-out fixture data.
- [`4_serve_reconstruction.md`](4_serve_reconstruction.md): compact automatic
  observations led Intern to repeat the supplied inspection point. Adding
  fallible server and contact proposals reduced server correctness. Neither arm
  passed the gate for a Qwen confirmation or wider run.
