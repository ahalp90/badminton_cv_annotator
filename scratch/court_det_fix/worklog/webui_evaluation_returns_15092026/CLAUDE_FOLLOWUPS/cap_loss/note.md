# Cap loss: where the close courts vanish inside the matcher

`R` is the remote experiment root.

The per-pair cap of 256 is not where the close courts vanish. For GX0 arm M and Amateur-2 frame 28019 arm B, the nearest court that passed the player checks is the same court the cap retained, so the loss lies before the cap, inside `propose_role` or the player checks. For Amateur-3 frame 0 arm R the cap discarded 8 courts in 2 pairs that sit up to 4.07 px closer than the pool's nearest (40.14 px against 44.21 px); that leaves the court 36 px short of the 4.3 px direction fit, so most of that loss also lies before the cap.

## Run

- Remote folder: `R/cap_loss/` on the compute host. It holds copies of `run_matcher.py`, `common.py`, `run_remote.sh` and `run_automatic.py`; the copies are in `remote_src/` here.
- Run name: `cap_loss_20260915_110406`, outputs under `cap_loss/runs/cap_loss_20260915_110406/e4/<arm>/results/`. Nothing under `direction_agreement/` or `automatic_axes_20260914/` was written.
- Three detached jobs launched together (`nice -n 10`, single-threaded), all exit code 0: GX0 M 35.6 min, Am3-0 R 21.8 min, Am2-28019 B 23.8 min.
- Instrumentation: in the copied `generate`, after `propose_role` every `candidate.corners_px` (working pixels, float32, n by 4 by 2) is recorded; after `select_pool` the retained indices into that array are recorded. One compressed `.npz` per case-arm sits beside the generation record with `pair_<id>_corners` and `pair_<id>_retained_index`. The copied `run_matcher.py` always regenerates (arm B's directions equal the saved estimator's, which the original would have short-cut through identity reuse) and skips both rescoring stages.
- The recordings and generation records are in `records/new/<arm>/results/`; the saved records they were gated against are in `records/saved/`; the job logs and receipts are in `records/run_logs/`.

## Gate 1: the instrumentation changed nothing

Every pair status, every shortlist candidate ID list in order, every shortlist `corners_px`, the final entries, `line_winner_id` and `paint_winner_id` equal the saved records. GX0 M and Am3-0 R are compared with the saved E4 generation records; Am2-28019 B with the saved baseline record (`automatic_axes_20260914/results`).

```
gxBQ_window_00_frame_0 arm M: new run cap_loss_20260915_110406 vs saved direction_agreement_20260915_144900
  PASS pair count 240 vs 240
  PASS every pair status 240 of 240 equal
  PASS every shortlist candidate_id list in order 134 of 134 matched pairs equal
  PASS every shortlist corners_px 134 of 134 matched pairs exactly equal; largest absolute gap 0.000e+00 px
  PASS final entries candidate IDs 256 vs 256 entries
  PASS line_winner_id 106:1732 vs 106:1732
  PASS paint_winner_id 211:19062 vs 211:19062
  PASS pooled_candidates 29696 vs 29696
am3_window_00_frame_0 arm R: new run cap_loss_20260915_110406 vs saved direction_agreement_20260915_144900
  PASS pair count 240 vs 240
  PASS every pair status 240 of 240 equal
  PASS every shortlist candidate_id list in order 100 of 100 matched pairs equal
  PASS every shortlist corners_px 100 of 100 matched pairs exactly equal; largest absolute gap 0.000e+00 px
  PASS final entries candidate IDs 256 vs 256 entries
  PASS line_winner_id 32:2640 vs 32:2640
  PASS paint_winner_id 43:13964 vs 43:13964
  PASS pooled_candidates 22593 vs 22593
am2_window_01_frame_28019 arm B: new run cap_loss_20260915_110406 vs saved automatic_axes_20260914 baseline
  PASS pair count 240 vs 240
  PASS every pair status 240 of 240 equal
  PASS every shortlist candidate_id list in order 104 of 104 matched pairs equal
  PASS every shortlist corners_px 104 of 104 matched pairs exactly equal; largest absolute gap 0.000e+00 px
  PASS final entries candidate IDs 256 vs 256 entries
  PASS line_winner_id None vs None
  PASS paint_winner_id None vs None
  PASS pooled_candidates 22762 vs 22762
GATE 1 PASS
```

## Gate 2: the nearest retained court equals the accounting's nearest pooled

Distances are maximum corner distance in working pixels with the 180-degree relabelling allowed (a copy of `projective_seed.corner_errors`), against the frozen control from the E3 record. Recomputed in float64 from the record shortlists, the nearest retained court matches the accounting table exactly in value and candidate ID for all three case-arms (33.822 for GX0 M, 44.208 for Am3-0 R, 15.321 for Am2-28019 B). The float32 recording matches the values within 1e-4 px. One caveat: on GX0 M, candidates 23:10986 and 23:10987 share their two worst corners, so their errors tie exactly and the argmin picks whichever comes first in the ordering (the recording names 23:10986, the accounting 23:10987; both are retained).

```

gxBQ_window_00_frame_0 arm M: 134 matched pairs, 3810512 proposed, 29696 retained
  nearest proposed court   33.822475 px  candidate 23:10986  (pair 23)
  nearest retained court   33.822475 px  candidate 23:10986  (pair 23); float64 from the record shortlists 33.822484574 px candidate 23:10987
  cap gain (nearest retained minus nearest proposed): 0.000009 px; within the rounding margin, so no loss at the cap
  proposed courts more than 0.001 px closer than the pool's nearest retained court: 0 in 0 pairs []
  pairs where the cap discarded a court closer than that pair's own nearest retained court: 107 of 134
  GATE 2 accounting nearest pooled 33.822 px candidate 23:10987: float64 from record PASS; float32 recording value PASS (gap 9.37e-06 px); float32 argmin candidate 23:10986, a tie on the worst corner broken by ordering

am3_window_00_frame_0 arm R: 100 matched pairs, 2449344 proposed, 22593 retained
  nearest proposed court   40.139707 px  candidate 30:21176  (pair 30)
  nearest retained court   44.207806 px  candidate 30:16137  (pair 30); float64 from the record shortlists 44.207753298 px candidate 30:16137
  cap gain (nearest retained minus nearest proposed): 4.068046 px; real
  proposed courts more than 0.001 px closer than the pool's nearest retained court: 8 in 2 pairs [30, 44]
    candidate 30:21176  40.139707 px  discarded by the cap
    candidate 30:21266  40.139811 px  discarded by the cap
    candidate 44:14115  40.337172 px  discarded by the cap
    candidate 44:14028  40.337286 px  discarded by the cap
    candidate 30:21178  40.869208 px  discarded by the cap
    candidate 30:21268  40.869312 px  discarded by the cap
    candidate 44:14134  42.993793 px  discarded by the cap
    candidate 44:14047  42.993898 px  discarded by the cap
  pairs where the cap discarded a court closer than that pair's own nearest retained court: 79 of 100
  GATE 2 accounting nearest pooled 44.208 px candidate 30:16137: float64 from record PASS; float32 recording value PASS (gap 5.31e-05 px); float32 argmin candidate same

am2_window_01_frame_28019 arm B: 104 matched pairs, 812540 proposed, 22762 retained
  nearest proposed court   15.321411 px  candidate 16:501  (pair 16)
  nearest retained court   15.321477 px  candidate 16:381  (pair 16); float64 from the record shortlists 15.321459062 px candidate 16:381
  cap gain (nearest retained minus nearest proposed): 0.000048 px; within the rounding margin, so no loss at the cap
  proposed courts more than 0.001 px closer than the pool's nearest retained court: 0 in 0 pairs []
  pairs where the cap discarded a court closer than that pair's own nearest retained court: 64 of 104
  GATE 2 accounting nearest pooled 15.321 px candidate 16:381: float64 from record PASS; float32 recording value PASS (gap 1.76e-05 px); float32 argmin candidate same

wrote <repo>/scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/cap_loss/table.csv with 338 rows
GATE 2 PASS
```

## Per case-arm

Counts are courts that passed the player checks (proposed) and courts the per-pair cap kept (retained), summed over matched pairs. Per-pair values are in `table.csv`.

| case-arm | matched pairs | proposed | retained | nearest proposed court | nearest retained court | cap gain |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| GX0 M | 134 | 3,810,512 | 29,696 | 33.822 px, candidate 23:10986, pair 23 (pencils 1 and 9; 12,622 proposed, 256 retained) | 33.822 px, candidate 23:10986 / 23:10987 (tie), pair 23 | 0.00001 px |
| Am3-0 R | 100 | 2,449,344 | 22,593 | 40.140 px, candidate 30:21176, pair 30 (pencils 2 and 0; 31,452 proposed, 256 retained), discarded by the cap | 44.208 px, candidate 30:16137, pair 30 | 4.068 px |
| Am2-28019 B | 104 | 812,540 | 22,762 | 15.321 px, candidate 16:501, pair 16 (pencils 1 and 2; 7,064 proposed, 256 retained), discarded by the cap | 15.321 px, candidate 16:381, pair 16 | 0.00005 px |

Am2-28019 B's nearest proposed court (16:501) lies 0.0002 px from the retained 16:381; the 2 px distinctness rule dropped it as a duplicate, so the 0.00005 px gain is not a loss.

Courts more than 0.001 px closer than the pool's nearest retained court, across every pair: GX0 M none; Am3-0 R eight, all discarded by the cap, four in pair 30 (nearest 40.140 px) and four in pair 44 (pencils 2 and 15; nearest 40.337 px); Am2-28019 B none.

The cap does discard courts closer than each pair's own nearest retained court in most pairs (107 of 134 for GX0 M, 79 of 100 for Am3-0 R, 64 of 104 for Am2-28019 B), but in only the two Am3-0 R pairs above is any of them closer than the pool's overall nearest.

## Answer

For GX0 M and Am2-28019 B the per-pair cap discarded no court closer than the pool's nearest: the nearest proposed court is the retained one to within float32 rounding, so the loss lies before the cap, inside `propose_role` or the player checks. For Am3-0 R the cap discarded courts closer than the pool's nearest in 2 of 100 pairs (pairs 30 and 44), 8 courts in all, the best 4.07 px closer (40.14 px against 44.21 px); the direction fit for that arm is 4.3 px, so the cap accounts for about 4 px of a 40 px gap and the rest lies before the cap. No fix proposed.

## Files

- `note.md` (this file), `table.csv` (338 rows, one per matched pair per case-arm)
- `gate_records.py`, `gate1_output.txt`; `analyse.py`, `gate2_analysis_output.txt`
- `remote_src/` (the four files copied to `cap_loss/` on the compute host), `run_name.txt`
- `records/new/`, `records/saved/`, `records/run_logs/`
