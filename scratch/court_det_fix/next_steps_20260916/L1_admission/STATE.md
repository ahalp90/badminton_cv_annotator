# L1 state

- Inputs: `line_identity/axis_replay.py`, its frozen `projective_seed.py` and `run_given.py` helpers, the retention reference helper, C2 witnesses, and the saved axis table. Current HEAD is `1376886`; the worktree was clean at start.
- Outputs: `next_steps_20260916/L1_admission/` holds the executable, comparison CSV, witness JSON, result note, runtime receipt in the witness, and six overlays. The WebUI seed received only the table and witness JSON.
- Completed: read the L1 brief and repository rules; started Serena/Pyrefly; confirmed the six source packs, baseline directions, direction-fit records, and existing mask/retention functions are present. The helper self-test, syntax check and Ruff check pass. The six-pair screen completed in one process; saved original axis and shortlist checks passed where records exist.
- Current result: Am2-B pair 15 improves 61.6207/81.2626 px to 7.5291 px across raw/mask and shortlist stages. Am3-B pair 43 regresses 4.2708/4.2709 px to 23.2484 px. The other four pairs are unchanged. Runtime was 558.066 s and peak RSS 880.738 MiB.
- Unresolved: the Am2 gain is selected-pair evidence only; no full-pool result or all-pair trial is justified. This exact variant stops after the six-pair screen.
- Next command: run output checks, review the result note and witnesses, then commit the final L1 package.
