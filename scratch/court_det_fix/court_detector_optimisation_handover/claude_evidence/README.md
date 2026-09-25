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
| `axis_duplicates/` | Bit-identical axis hypotheses among those scored | Item 12: 0 duplicates among 321,224 (`summary.txt`) |
| `w5_savings/` | The camera-check rewrite, how much of the scoring stage can still change the choice, and a profile of its measurement step. Also the 28-view Carmack check after items 13 and 14 | Item 13: 2.71 → 0.39 s on 80 saved batches. Items 13 and 14: 8,931 → 8,250 s with 168 files identical. Item 15's decisions |
| `prefilter/` | How deep a coarse score must reach to keep each shortlist | Item 16 (`summary.txt`) |
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
