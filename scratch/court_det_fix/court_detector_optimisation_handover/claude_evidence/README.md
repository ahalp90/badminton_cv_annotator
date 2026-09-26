# Speed-up evidence

These scripts produced or checked the numbers in the
[speed-up README](../README.md) and in the archived records it links to. Many
need run folders or saved inputs on Carmack, the shared compute server, which
are not in git.

Keep this folder and `fresh_feet/` where they are. The joined detector, its
check and its tests read files from `fresh_feet/`:
`court_detector/run_views.py:58`, `court_detector/feet.py:4`,
`court_detector/check_20260925/run_all.sh:33` and
`tests/test_court_detector_feet.py:21`.

## What each folder backs

Item numbers refer to the archived
[speed-up list](../../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md).
The evaluation is the archived
[evaluation of the web-UI packet](../../archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md).

| Files | What they measured | Numbers they back |
| --- | --- | --- |
| `d17/` | The research chain on all 28 views on 24 September at commit f72ed1c6, full search against a screen that drops some direction pairs (SVD12). Then the rerun after items 3 and 5 | The joined detector's accuracy: median reference error 1.2–3.6 working pixels on the 18 court views with reference points (10 with landmark sets, 8 with four clicked corners) (`d17_results.txt`). Items 3 and 5: 20,647 → 13,942 s over the 20 court views, with 168 of 168 files identical (`players_rerun_results.txt`) |
| `exact_rewrites/` | `compare_exact_runs.py` checks two research-chain run folders bit for bit. `compare_stage_times.py` compares their stage times | The exactness of items 4, 12, 13 and 14 |
| `shortlist_bound/` | An upper bound on court scores, batch sizes, and the x and y split: timing and bit-identity | Item 12: the bound costs 53–60% of full scoring. The raw rows are here, and the summary table is in item 12 |
| `axis_batch/` | `score_axes` time and peak memory at batch sizes from 64 to 16,384, on synthetic inputs shaped like one direction, on a laptop (`axis_batch.py`) | Larger CPU batches (F4) are no faster than 256 (`axis_batch.txt`) |
| `axis_duplicates/` | Bit-identical axis hypotheses among those scored | Item 12: 0 duplicates among 321,224 (`summary.txt`) |
| `w5_savings/` | The camera-check rewrite, how much of the scoring stage can still change the choice, and a profile of its measurement step. Also the 28-view Carmack check after items 13 and 14 | Item 13: 2.71 → 0.39 s on 80 saved batches. Items 13 and 14: 8,931 → 8,250 s with 168 files identical. Item 15's decisions |
| `prefilter/` | How deep a coarse score must reach to keep each shortlist. Then, on 26 September, how deep the courts that matter sit (`k_depth.py`, `summarise_k_depth.py`) and which K leaves every overall shortlist unchanged (`dropped_courts.py`). Also where each chosen court sits in its pair's build order (`build_order.py`), and, with the upright-camera filter, where the chosen, top-five and overall-shortlist courts sit (`build_order_upright.py`, on the final Carmack run) | Item 16 (`summary.txt`). The cascade's K = 2,048 default (`k_depth_summary.txt`, `dropped_courts.txt`) |
| `shortlist_cap_replay/` | A 128 cap on each search's overall shortlist, replayed from the saved 25 September artefacts: chosen courts (`replay_cap.py`), places 2–5 (`substitutes.py`) and one view's top courts (`top_courts.py`). File suffixes give the G0 and G1 caps. The `upright_*` files repeat the chosen-court replay on the 26 September Carmack run, with the upright-camera filter, and count each view's shortlist sizes (`shortlist_sizes.py`) | Keeping both caps at 256, before and after the filter ([web-UI page](../../webui_final_opt_handover/README.md#shortlist-caps-of-128)) |
| `upright_camera/` | The 26 September saved search outputs and the filtered check run | What the upright-camera tests would remove from the saved outputs: the pair test 1,770 of 2,914 direction pairs, 65% of G0 pair time and 54% of G1's; the court test 44–48% of the remaining pairs' shortlisted courts; gxBQ_window_00_frame_0's right court moves from G0 #96 to #1; steep pairs held 253–256 of G0's 256 overall places on the three views whose court changed (`saved_outputs.txt`). Hand-mark errors of both runs' final courts in working px (`reference_errors.txt`) and in floor metres, with where each changed winner sat on 25 September (`floor_errors.txt`). Every scored candidate on am3_window_02_frame_17174 by floor error: 189 eligible ones beat the winner (`candidates_am3_window_02_frame_17174.txt`) |
| `built_courts/` | Every court each direction pair builds before full scoring, rebuilt on Carmack from the final 26 September run's inputs with full scoring switched off (`local_built_courts.py`, run by `run_built_courts.sh`; one record per view in `filtered/`, and `unfiltered/` repeats it without the upright-camera filter). `summarise_built_courts.py` joins them, and `cascade_cost.py` turns each pair's court count into a cost for the cascade. The rebuild matches the run's 10,595 shortlisted courts exactly | Each court's horizon distance, its players' implied width and its rank by line-guess average (`summary.txt`). The line-guess average ranks the courts that matter about as deep as build order: at K = 2,048 it keeps the chosen court on 10 of 18 views, against 12 by build order ([web-UI page](../../webui_final_opt_handover/README.md#ruled-out-choosing-the-k-courts-by-their-line-guess-average)). 3,303,856 of the 12,438,584 courts built with the filter put the horizon at least 2.84 image widths out, 3,097,308 of them on the shuttleset_03 views ([web-UI page](../../webui_final_opt_handover/README.md#skipping-courts-that-imply-a-camera-looking-straight-down)). A 0.2–3 m player width would skip 13.6% of built courts ([player-size check](../../court_detector/check_20260926_player_size/README.md)). With the filter, the cascade at K = 2,048 would save about 5–6% of the run, estimated from court counts (`cascade_cost.txt`, [web-UI page](../../webui_final_opt_handover/README.md#the-cascade-item-16-parked)) |
| `precision/` | float16, float32 and float64 in the camera check | Item 17 (`precision_result.txt`) |
| `fresh_feet/` | Fresh person detections around each view, the shot check, the feet variants, the player-test screen and the research-chain runs per variant | Items 10 and 11, and the fresh-detection finding on control 100347. Also the test set: `views.json` lists the 28 views |
| `compare_runs.py`, `score_drift.py`, `timings.py`, `pair_times.py`, `exact_patches.diff`, `reproduce_exactness.py`, `CODEX_REDTEAM.md` | Patches 1 and 2: the six-view Carmack comparison, its timings, and a red team of both patches | The evaluation's six-view result, 6,255 → 2,572 s, and its score-drift table |
| `alignment_probe.py`, `history_probe.py`, `stage_probe.py`, `ipp_probe.py`, `ipp_timing.py`, `ipp_bench.py` | Where the score drift came from: OpenCV's IPP distance transform rounds differently at different memory addresses | The evaluation's "Score drift and its cause", and item 2 |
| `profile_pair.py`, `joint_players_probe.py`, `profile_evidence.py`, `compare_outputs.py` | Profiles of single direction pairs, of pool evidence and of the refit stage. Also the matrix-product player test against the old one | The evaluation's profiles and its F5 section |
| `feet_census.py`, `count_f1.py` | People per frame, and axis hypotheses that pass the player test | The evaluation's "Finding 1" and its F1 verdict |

## Before rerunning a script

- Its docstring gives the command. Four of them are wrong:
  `history_probe.py`, `ipp_probe.py` and `stage_probe.py` carry
  `alignment_probe.py`'s docstring, and `joint_players_probe.py` carries
  `profile_pair.py`'s. The table above says what each one does.
- `CODEX_REDTEAM.md` runs `redteam/reproduce_exactness.py`. That script is now
  `reproduce_exactness.py` in this folder.
- `precision/precision_probe.py` needs `w5_savings/camera_batches.npz`, which
  is not committed. `w5_savings/camera_probe.py` saves those batches from a
  research-chain run.
- `fresh_feet/uncommitted_888cf999.diff` is the `run_d17.py --feet` change that
  the item 10 runs used before commit 6178fda5 added it.
- `fresh_feet/run_feet_variants.sh` also launches the research chain for the
  joined detector's check. [WIRING.md](../../d17_timing/WIRING.md#how-to-check-an-integrated-detector)
  gives its arguments.
