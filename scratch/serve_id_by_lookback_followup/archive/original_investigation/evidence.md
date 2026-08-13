> ARCHIVED 2026-08-12: original evidence ledger. Current evidence: `../../docs/results.md`.

# Evidence

## H3/R8 extension evidence fixed before the run

- Code 3 is not synonymous with halo. It combines halo frames and repeated
  attractor positions outside the recurrence core.
- Across all three fixtures, 53,471 of 55,937 code-3 frames are halo-only. In
  saved rally spans, 12,207 of 12,879 are halo-only.
- Reconstructing the guard with a 15-frame halo exactly reproduces production
  `grade_track` codes for all three fixtures.
- A three-frame halo changes 13,549 frames in `sset_01`, 15,161 in `sset_15`,
  and 8,429 in `sset_21` relative to the current 15-frame halo.
- The current path extractor measures one contiguous usable run. It never
  calculates a step across a missing or guarded frame.
- Overlay sample 07 has 25 consecutive usable post-contact frames, a median
  non-zero step of 0.084648 body heights, a largest step of 0.577997 body
  heights, and a largest-step ratio of 6.828283. Its failure at 4.0 was not a
  missing-frame denominator error.
- The frozen release spans were generated with active high-shot-out-of-bounds
  handling, a 75-base-30fps demotion bound, and two-sided re-entry checks.
- Across the fixed population, 146 consecutive accepted-contact pairs bracket a
  measured high-shot state. Forty-two exceed the ordinary scaled 60-frame cap.
- Inclusive endpoint buffers of 6, 9, 10, 12, 15, and 30 base-30fps frames admit
  0, 5, 6, 9, 18, and 21 of those 42 pairs. The approved fixed buffer is 12,
  chosen to match the existing impulse half-window rather than GT.
- In clip 13, `sset_21:21:set1:6`, accepted contacts 16509 and 16574 bracket the
  measured high-shot state `[16515, 16564)`. Their endpoint distances are 6 and
  10 frames, so the fixed exception admits the predecessor.
- These measurements used accepted contacts and production state only. GT did
  not select the halo, ratio, 60-frame cap, or 12-frame buffer.

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

- No numerical H3/R8 item remains unverified. Independent review returned
  `CLEAN`; focused checks, whole-project Pyrefly, and all tests pass. The
  repository-wide Ruff gate still has 661 pre-existing findings outside this
  extension, while focused Ruff passes.

## H3/R8 checked results

- The write pass saved 3,200 accepted-contact rows and 239 result rows. The
  check pass rebuilt and directly matched every contact row, GT-free result
  field, score, and summary.
- Pre paths are eligible for 2,329 contacts. Verdicts are 1,963 incoming, 366
  not incoming, and 871 unavailable.
- The H3/R8 sequential search selects a contact in 234 rallies. Categories are
  68 first-post, 23 visible-serve, 143 pre-unavailable, and 5 no-credible.
- The sequential search is correct in 43 rallies at +/-10, compared with 34 in
  the completed H15/R4 search. It fixes 26 and damages 13 classified results,
  while 148 rallies end unavailable or no-credible.
- The incoming-only search has 234 anchors. Categories are 44 visible-serve, 33
  first-post, 157 predecessor-unavailable, and 5 no-anchor-with-unavailable.
- The incoming-only search is correct in 26 rallies at +/-10. Ordinary timing
  admits 196 predecessors but yields 3 correct visible serves. The high-shot
  exception admits 5 and all 5 are correct visible serves.
- Search B recovers clip 13 through the fixed measured state `[16515, 16564)`.
  Search A also classifies accepted frame 16509 as the visible serve after the
  ratio-8 relaxation makes its pre trace eligible.
- First-contact outgoing failures under H3/R8 are 103 of 168 Top contacts and
  20 of 71 Bottom contacts. The bottom-occlusion hypothesis is unsupported.

## Additive correction evidence

- The PR #82 visible-start interpretation is correct in 125 of 239 rallies at
  +/-10. Its separate server attribution is correct in 163.
- The five `high_shot_oob` states nominate the baseline in three rallies and a
  later accepted contact in two. The two changes fix two starts and damage none.
- Only one high-shot correction changes the server. It is a fix, taking server
  correctness from 163 to 164.
- Within the old 97 unmatched slice, the final policy fixes two starts and one
  server error. It leaves 95 starts and 38 server answers wrong.
- A broad accepted-contact serve-setup differential fires 138 times. It fixes
  22 starts, damages 63, and changes 53 without making them right.
- All 22 broad serve-setup fixes lie in the old 97 slice. Its 63 damages lie
  outside the slice, so the GT-defined slice cannot supply a legal trigger.
- A stricter continued same-player setup state fires twice. It fixes one start,
  damages one, and changes no server answer.
- The frozen 288p source videos contain video only. `ffprobe` found no audio
  stream, so contact-sound evidence is unavailable in this fixture set.
