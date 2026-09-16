# Note on the follow-ups folder

`R` is the remote experiment root.

Written 2026-09-15 by the Claude Code session that evaluated the WebUI returns and ran the follow-up checks. Nothing in the repository's tracked files changed; everything below is under this gitignored scratch tree.

## What to read

- `CLAUDE_FOLLOWUPS/HANDOVER.md`: read first in the next session; a run on the compute host was in flight when the session closed.

- `CLAUDE_EVALUATION.md`, one level up from the follow-ups: how the three WebUI assessments relate to the direction experiment, and the six follow-ups they implied.
- `CLAUDE_FOLLOWUPS/EVALUATION.md`: what the checks found and what to do next. Read this if you read one thing.
- `CLAUDE_FOLLOWUPS/SCHEDULE.md`: the plan, including the per-view audit that stays on hold, with paths, tools and remote conventions for a fresh session.
- `CLAUDE_FOLLOWUPS/status.md`: one row per check with its outcome.
- `CLAUDE_FOLLOWUPS/manifest.json`: MD5 and size of every input the checks read, every output they wrote, the gate outcome per check, and the compute host job receipts.

## What each check folder holds

Each folder has the script that produced its numbers, its `table.csv`, its `note.md` written by the agent that ran it, and any pictures. The briefs the agents followed are in `briefs/`. Six folders: `gx0_rows`, `gx_directions`, `identity_diagnostics`, `distortion`, `distortion_ridge`, `cap_loss`.

## Cruft

- `CLAUDE_FOLLOWUPS/cap_loss/records/new/` (117 MB) and `records/saved/` (25 MB) are the instrumented generation records, the per-pair proposal arrays and the saved records they were gated against. They are re-derivable from the input packs, the E2 records and the scripts in `cap_loss/remote_src/`. Delete when space matters; `cap_loss/table.csv` and `note.md` carry the result.
- `CLAUDE_FOLLOWUPS/_bundles/` (1.6 MB) is the five WebUI zips extracted. Delete freely.
- `CLAUDE_FOLLOWUPS/distortion/` is the first distortion attempt, inconclusive by its own gate. Kept as a record of a method that does not work; small.
- Everything else is small and worth keeping.

## Remote state

The compute host holds `R/cap_loss/` with the instrumented copies, the finished run `cap_loss_20260915_110406` and a second run `pregate_<timestamp>` that was still running at session close; its two jobs finish on their own and leave receipts. Nothing under `direction_agreement/` or `automatic_axes_20260914/` on the compute host was written. The folder can stay for the next instrumented run the evaluation proposes, or be removed.
