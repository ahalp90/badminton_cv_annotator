# Line identity: worklog

## Resume

- Stage: diagnostics. The courts-before-the-gate run is analysed and the axis-matching replay has located the loss on Amateur-3. Next: the paint-colour and person-box fragment filters, replayed at the direction and axis-matching stages on all nine views.
- Active jobs: none on the compute host.
- Last passed gate: axis replay reproduces every kept matching and every record nearest court (six pairs, exit 0).
- Paths: local `scratch/court_det_fix/line_identity/`; the earlier follow-up checks live in the gitignored `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/`; remote mirror `R/line_identity/` once matcher runs start (R is the remote experiment root, mapped in the gitignored `paths.local.sh`).

## Terms used in this folder

- Direction selection: the automatic choice of 16 vanishing directions from merged line groups (`vp_pruning.estimate`, coverage rule). The baseline selection is arm B; the direction experiment's two changes are the midpoint-anchor selection (M) and the precise-representative selection (R).
- Direction fit: the least-squares court fitted to the frozen control from one ordered pair of selected directions with both vanishing points fixed (stage E3 of the direction experiment). It says what the directions could support; it is not a generated court.
- Axis matching: how the matcher turns one direction pair into courts (`projective_seed.match_axis`): enumerate a scale and offset from every pair of direction-compatible observed line groups against every pair of court markings, keep enumerations with at least three supported markings that the necessary player rule accepts, drop duplicates by marking assignment, keep the 512 best by score (the per-direction cap), then combine every kept horizontal matching with every kept vertical matching.
- Kept matching: one of the 512 survivors per direction. Combined court: one horizontal kept matching paired with one vertical one.
- Control: the frozen reference court per view from the direction experiment (`evidence.md` there); six visually approved, three manual references. Distances are maximum corner distance in working pixels (960 by 540) with the 180-degree relabelling allowed.

## Log

- 2026-09-16 07:20. Read the handover from the follow-up checks. Both detached jobs of the courts-before-the-gate run had exited 0 on the compute host (GX0 under M 35.9 min, Amateur-3 under R 20.9 min). Pulled the records (530 MB, gitignored cruft in the follow-ups folder).
- 2026-09-16 07:30. Gate 1 passed: both new generation records equal the saved ones in every pair status, shortlist, winner and pooled count. Gate 2 passed after one script fix: the earlier cross-check compared an index into the post-mask array with an index into the pre-mask array; it now maps through the mask. Result: the nearest court before the masks is the nearest usable court on both views (33.8 px GX0, 40.1 px Amateur-3); nothing within 20 px among 22 and 17 million combined courts. Note, table and evaluation updated in the follow-ups folder.
- 2026-09-16 07:45. Read `match_axis`: each combined court is the product of 512 kept matchings per direction, and on the best-fit pairs the per-direction cap excluded 1,500 to 26,000 distinct matchings. Wrote `axis_replay.py` to separate enumeration, the support and player rules, duplicate removal and the cap, using the direction-fit homography decomposed in the pair's basis as the ideal per-direction scale and offset.
- 2026-09-16 08:05. Axis replay on six pairs (GX0 under M pair 23; Amateur-3 under R pairs 43 and 30; Amateur-3 under B pair 43; GX0 under B pairs 22 and 143). Gates: the 512 kept matchings per direction equal the record by ID and parameters, the enumerated and distinct counts equal the record's diagnostics, and the kept-by-kept nearest court equals the record's nearest for the pair (33.822, 46.100, 40.140, 4.271, 7.662 and 34.327 px). Finding: on Amateur-3 under R pair 43 the horizontal matchings that reach the 4.3 px fit are enumerated and pass every rule, but rank about 11,500 of 23,864 by the axis score; all 512 kept horizontal matchings support 5 of 5 markings with scores above 0.885 while the near-ideal one scores 0.58. The vertical matching survives (rank 103). Under B the same pair keeps a 5.4 px horizontal matching at rank 367. The Amateur-3 frame has parallel floor-plank seams along the sideline direction, which is where the 82 direction-compatible groups come from.
