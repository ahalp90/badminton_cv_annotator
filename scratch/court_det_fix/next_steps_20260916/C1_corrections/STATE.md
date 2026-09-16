# C1 state

- Inputs: `worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/{EVALUATION.md,status.md,SCHEDULE.md}`; `worklog/webui_evaluation_returns_15092026/camera_calibration_assessment.md`; `line_identity/{results.md,worklog.md,evidence.md}`; the GX, distortion and identity notes; `frozen_helpers_20260914/automatic_axes/run_automatic.py`.
- Output: this state file, `changes.md`, and in-place corrections to the current summaries and supporting notes under `scratch/court_det_fix/`.
- Completed: recorded HEAD `872ea806c5b69a195f32e9f60b5840294eefb6cd`; confirmed the worktree was clean; verified that `set_bound` is a lower bound and that observation-only inputs reach both generation and `evaluate_pool` scoring; patched the named summaries and supporting notes.
- Current result: summaries now distinguish incidence lower bounds from fits, leave distortion inconclusive without an upper bound, keep temporal scoring untested, treat rank/spacing as diagnostics, and keep GX-row attribution visual and bounded.
- Unresolved: the modern independent-versus-shared score matrix on one verified view remains unmeasured; no historical result will be recomputed.
- Next command: no further C1 edit; C2 may verify detailed raw traces, and the modern same-union temporal score test remains a separate open route.
