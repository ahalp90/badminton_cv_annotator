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
- The current helper does not retain why a run starts, ends, or is absent. `None` can mean no usable sample, scene exclusion, missing player attribution, or missing tracker segment
- `data.spans` and `data.segments` retain half-open rally and scene boundaries. The current path mask uses the scene boundary but not the rally span boundary
- `path_reaches_scene_start` is the only stored boundary reason. It does not distinguish a coincident 30-frame lookback limit
- Under the declared forward scan, every earlier skipped contact has a usable no-outgoing verdict. The final positive-endpoint ruling makes those contacts ineligible for reconnection
- The user ruled that credible outgoing and incoming traces may connect across invisible TrackNet frames without overlap, adjacency, or spatial joining
- Production `_gap_state_rest_mask` already keeps `high_shot_oob` gaps open without joining missing shuttle positions. The current `gap_state_demotion_bound` is 75 base-30fps frames
- The source has no fixed maximum accepted-contact-to-contact interval
- The user fixed reconnection candidates to consecutive accepted contacts and ruled that all gap contents are non-vetoing
- The accepted contact gap limit is an inclusive 75 base-30fps frames. It is fixed before GT scoring and matches `high_shot_oob`
- The experiment must record the full consecutive contact-gap distribution before joining GT
- The focused baseline passed: `55 passed` in `0.67s`, exit 0

## Provenance

- Source investigation: `scratch/serve_start_trajectory_exploration/`
- Original follow-up request: `Scope.md`
- Corrected user rulings: `00_SHARED_CONTRACT.md` and `01_LAUNCH_ACCEPTED_CONTACT_TRACE.md`
- Read-only Luna run-ending trace: `delegates/20260811-luna-run-end-reasons/`; its material claims above were checked locally against source and tests

## Not yet verified

- Counts from applying the approved rule to all 239 rallies
