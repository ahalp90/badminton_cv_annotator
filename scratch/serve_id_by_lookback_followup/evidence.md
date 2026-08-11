# Evidence

## Verified facts

- Branch base: `4f9703f339e2f9821d986d376dbfca9d6fd18ad7`, the tip of PR #82's investigation branch when this follow-up branch was created
- Working tip at the corrected sweep: `ba3d401d2ed4f995f91371421c78dd769c848ea3`; the worktree was clean before the sweep
- Primary population: 239 one-to-one rallies, keyed by `(fixture, video_id, set_id, rally)`
- Current ±10 labels from PR #82 output: 119 contact 1, 19 contact 2, 4 later, 97 unmatched
- `experiment_data.py` sorts `filtered_by_rally` frames and validates them against raw acceptance fields
- `analyse_serve_trajectory.py` chooses the earliest accepted frame as the current anchor
- PR #82's recurrence-clean motion result uses `closest_pre_contact_run`, `measure_incoming_motion`, `fit_robust_distance_trend`, and a fixed 0.05-BH fitted decrease
- Serena/Pyrefly was visible and active during the corrected sweep. Text search covered dynamic row-field use that semantic references did not resolve
- `closest_pre_contact_run` searches `[contact - lookback, contact)`, keeps the latest maximal true run, and excludes the contact frame. It returns only `(start, end, frames_to_contact)`
- A direct post-contact mirror can search `(contact, contact + lookahead]`, keep the earliest maximal true run, and define `frames_from_contact = start - contact`. The immediate next frame has gap 1, matching the pre-contact convention
- The symmetric common eligibility checks are at least 5 frames, at most 2 base-30fps frames from the contact, and `largest_step_ratio <= 4.0`. The 30-base-30fps path window and all frame gaps are scaled to source FPS
- The robust incoming call is fitted distance decrease `>= 0.05` body heights. The direct outgoing mirror is fitted distance decrease `<= -0.05` body heights
- The recurrence-clean mask combines valid non-zero shuttle coordinates, a positive track flag, court presence, finite player distance, positive finite player bbox height, and `guard_codes == NO_FLAG`. A false component splits the run
- The current helper returns `None` when the local incoming check lacks usable evidence. The selected contact's pre-contact result must retain that unavailable state
- `data.spans` and `data.segments` retain half-open rally and scene boundaries. The current path mask uses the scene boundary but not the rally span boundary
- The final ruling removes all contact reconnection. The search stops at the first credible outgoing contact and uses only that contact's existing pre-contact incoming check for classification
- The outgoing search is binary. Missing or unusable post-contact evidence and measured absence of outgoing motion both fail the credible-outgoing predicate
- The experiment has no outgoing-unavailable reporting state or continue-past-unknown sensitivity run
- A later contact never overrides an earlier `no outgoing` verdict
- Cross-gap evidence, contact-gap caps, and contact-gap distributions have no role in the final experiment
- The focused baseline passed: `55 passed` in `0.67s`, exit 0
- The fixed 239 population crosswalk is derived from GT rally-to-span mapping. Search actions remain GT-free through an explicit `SearchInputs` projection with no stroke frames, truth labels, or boundary fields
- The follow-up local path is bounded by both tracker scene and rally span. The span bound is an intentional addition to PR #82's scene-only path
- The checked run froze 104, 84, and 51 search rows for `sset_01`, `sset_15`, and `sset_21`. Check mode rebuilt and matched all 239 rows
- The binary outgoing predicate accepts the first contact in 21 rallies. It skips the first contact in the other 218
- The search selects some contact in 212 rallies. The selected accepted rank has median 3 and maximum 24; 27 rallies have no credible accepted contact
- The selected contact's pre check is unavailable in 100 rallies. It returns incoming in 94 and not incoming in 18
- At +/-10, the final categories contain 16 fixed, 34 damaged, 62 unchanged, 100 pre-contact unknown, and 27 no-credible-contact results
- Within the 97 baseline-unmatched starts, +/-10 results contain 15 fixed, 38 unchanged wrong, 35 pre-contact unknown, and 9 no-credible-contact results
- Of the 119 baseline-correct starts at +/-10, 18 remain correct, 34 become classified wrong, and 67 end in a terminal unknown or no-credible-contact state
- The +/-5 and +/-30 checks do not reverse the result. They produce 16/23 and 15/44 fixed/damaged counts respectively; +/-30 has 167 selected-frame multiple matches and is only a coarse check
- The declared Claude Opus 4.6 Thinking and Gemini 3.1 Pro High read-only audits both returned `PASS`. Their launch tripwires confirmed that neither changed the repository

## Provenance

- Source investigation: `scratch/serve_start_trajectory_exploration/`
- Original follow-up request: `Scope.md`
- Corrected user rulings: `00_SHARED_CONTRACT.md` and `01_LAUNCH_ACCEPTED_CONTACT_TRACE.md`
- Read-only Luna run-ending trace: `delegates/20260811-luna-run-end-reasons/`; its material claims above were checked locally against source and tests

## Not yet verified

- Counts from applying the approved rule to all 239 rallies
- Whether the fixed sequential rule improves the 239-rally GT comparison
