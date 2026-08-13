# Issue 90 worklog

## Resume

- Branch: `issue-90-held-out-serve-rule`
- Worktree: `worktrees/issue-90-held-out-serve-rule`
- Base: `c0b6ee66ead471164ed11c7848d6023490abeed8`
- Carmack checkout: `/home/cmarti56/issue90-gpu`
- Carmack data: `/home/cmarti56/issue90-data`
- Active Carmack job: none.
- Next: review and merge PR #97. Production integration remains out of scope.

## Decisions

- `sset_20`: men, downcourt mapping 0, 81,650 frames at 25 fps.
- `sset_22`: women, downcourt mapping 1, 100,896 frames at 30 fps.
- These videos are outside the three development videos.
- GPU extraction uses Carmack's CUDA 13.2 module and the isolated RTMLib venv.

## Work completed

- Created the worktree from merged PR #88.
- Downloaded both canonical YouTube videos to Carmack.
- Verified reported resolution, frame rate, duration, and frame count.
- Verified all required model checkpoints.
- Completed pose extraction for both held-out videos.
- Completed TrackNet and InpaintNet for both held-out videos.
- Completed court detection for both held-out videos.
- Completed annotation with 74 spans for `sset_20` and 79 for `sset_22`.
- Kept all ShuttleSet labels unopened while preparing the prediction freeze.
- Froze all 153 label-blind predictions before opening ShuttleSet labels.
- Verified the local and remote prediction checksum:
  `6b946a6422681936472019f32965a2a9ba32a4d88ef8a15dc0e015ec72873e09`.
- Scored 85 strict one-to-one held-out rallies.
- PR #82 and PR #88 both got 61 server sides correct.
- PR #88 improved visible starts from 31 to 41 and joint answers from 27 to 38.
- PR #88 had eight server fixes and eight damages.
- PR #88 regressed on `sset_22`, so the production promotion gate failed.
- The independent review found one span that overlapped a split rally and a
  covered rally. Excluding it changed the population from 86 to 85 but did not
  change the decision.
- The follow-up review verified the corrected denominator and report.
- Final gates: seven focused tests passed, Ruff passed, Pyrefly reported zero
  errors on the changed Python files, and `git diff --check` passed.
- Whole-project Pyrefly remains red on 17 existing Jaxtyping dimension-name
  errors outside this change.
- A second scoring run reproduced the scored CSV and metrics byte for byte.
- Published the evaluation as draft PR #97.

## Incidents

- First launch lacked `src/bst_x` on `PYTHONPATH` and exited before inference.
- Second launch lacked the system CUDA toolkit libraries and exited before
  inference.
- The corrected launch passed a direct CUDA model load and used the L40.
- Court detection initially lacked PySceneDetect. Installed the pinned
  `scenedetect==0.7.1` dependency in the isolated Carmack environment.
