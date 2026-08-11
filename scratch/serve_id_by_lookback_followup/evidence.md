# Evidence

## Verified facts

- Branch base: `4f9703f339e2f9821d986d376dbfca9d6fd18ad7`, the tip of PR #82's investigation branch when this follow-up branch was created
- Primary population: 239 one-to-one rallies, keyed by `(fixture, video_id, set_id, rally)`
- Current ±10 labels from PR #82 output: 119 contact 1, 19 contact 2, 4 later, 97 unmatched
- `experiment_data.py` sorts `filtered_by_rally` frames and validates them against raw acceptance fields
- `analyse_serve_trajectory.py` chooses the earliest accepted frame as the current anchor
- PR #82's recurrence-clean motion result uses `closest_pre_contact_run`, `measure_incoming_motion`, `fit_robust_distance_trend`, and a fixed 0.05-BH fitted decrease
- The main Codex session could not call Serena MCP directly; a read-only Serena-enabled trace confirmed the accepted-contact ordering path

## Provenance

- Source investigation: `scratch/serve_start_trajectory_exploration/`
- Original follow-up request: `Scope.md`
- Corrected user rulings: `00_SHARED_CONTRACT.md` and `01_LAUNCH_ACCEPTED_CONTACT_TRACE.md`
- Full ignored worker artefacts: `delegates/`

## Not yet verified

- A symmetric post-contact path definition
- A fixed outgoing-motion rule
- A GT-free continuity rule between an earlier outgoing path and a later incoming path
- Observable trace-end reasons that distinguish no accepted origin from unavailable evidence
