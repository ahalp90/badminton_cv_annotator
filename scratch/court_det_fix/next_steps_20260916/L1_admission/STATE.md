# L1 state

- Inputs: `line_identity/axis_replay.py`, its frozen `projective_seed.py` and `run_given.py` helpers, the retention reference helper, C2 witnesses, and the saved axis table. Current HEAD is `1376886`; the worktree was clean at start.
- Outputs: `next_steps_20260916/L1_admission/` will hold the executable, comparison CSV, witness JSON, result note, runtime receipt, and available overlays. The WebUI seed will receive only the table and witness JSON.
- Completed: read the L1 brief and repository rules; started Serena/Pyrefly; confirmed the six source packs, baseline directions, direction-fit records, and existing mask/retention functions are present. The helper self-test, syntax check and Ruff check pass. An Am3-R smoke replay matches the saved original axis IDs, parameters, and per-pair shortlist.
- Current result: the selector uses uncapped original `match_axis` output, preserves `distinct[:256]`, admits 256 more rows by signed centre/log-span cells, and ranks the unchanged shortlist with the frozen finite-support score. The Am3-R A smoke is 46.0998 px before/after masks and 48.5665 px after the cap.
- Unresolved: whether the broader admission survives the unchanged geometry/player masks and per-pair shortlist cap on the six fixed pairs.
- Next command: commit this executable checkpoint, run the six-pair screen, then inspect the compact witnesses and overlays.
