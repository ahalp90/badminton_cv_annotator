# Run record

Exact commands, hashes, timings and exit codes. `R` is the remote experiment root and `L` the local source tree; `sync.sh` resolves both from the gitignored `paths.local.sh`. Every remote stage runs through `run_remote.sh`, which sets `PYTHONDONTWRITEBYTECODE=1`, single-thread BLAS and OpenMP, a run-local cache directory, and the packet's `PYTHONPATH` plus `automatic_axes_20260914/svd_fixed` and this folder.

## Local checks

Run from this folder; `<repo>` is the repository root and `L` the alias above. Rerun after every code change; the last run of each is at the final commit.

```text
~/.venvs/badminton-cicd/bin/ruff check .                                  # exit 0
PYTHONDONTWRITEBYTECODE=1 ~/.venvs/badminton-cicd/bin/pytest tests -q -p no:cacheprovider
                                                                          # 41 passed; exit 0
~/.venvs/badminton-cicd/bin/pyrefly check *.py tests/*.py -c <repo>/pyrefly.toml \
  --search-path . --search-path L/automatic_axes/svd_fixed --search-path L/automatic_axes \
  --search-path L/axis_matching --search-path L/marking_diagnosis --search-path L/vp_pruning \
  --search-path <repo>/src --search-path <repo>                           # 0 errors
```

Local stages (AEST), run with the local Python from `paths.local.sh`:

| Stage | Command | When | Exit | Notes |
| --- | --- | --- | ---: | --- |
| summarise | `summarise.py --run-dir runs/<run>` | 17:58, rerun 19:58 after adding the pair-count columns | 0 | 81 matcher rows; writes `summary.md` and `summary.json.gz` |
| gallery | `PYTHONPATH=<repo>:<repo>/src:. render_gallery.py --run-dir runs/<run>` | 17:59 | 0 | nine sections, candidate joins valid; first attempt without `<repo>/src` on the path failed on the `courtkeynet` import |
| report audit | `~/.venvs/write-experiment-report/bin/python <skill>/scripts/audit_report.py results.md --allow-terms <cruft>/report/allow_terms.txt --format text` | 16:55 onwards | 0 | mechanical score 68.0 on the first draft, 89.2 after the E0-E3 tidy, 79.8 with the E4 sections |

## Remote stages

Run name: see `run_name.txt`. Log, PID and exit receipts: `runs/<run>/{logs,receipts}/`. Remote timestamps are UTC. Launch form: `bash sync.sh launch <label> <script> [args]`, which runs `run_remote.sh <run> <label> <script> --root . --run <run> [args]` detached on the remote.

| Label | Script and args | Started (UTC) | Finished | Exit | Notes |
| --- | --- | --- | --- | ---: | --- |
| manifest | `manifest --git-head d0c9a12 --git-branch fix/court-det --git-dirty-files 0 --git-checkpoint 9fcec94 --session-model claude-fable-5-1` | 05:05:36 | 05:05:41 | 0 | run synchronously; thread count later pinned to 1 in the script |
| arms (first) | `run_arms` | 05:03 | 05:03 | 1 | ImportError (`ARMS` from the wrong module); no records written |
| arms | `run_arms` | 05:06:47 | 05:07:40 | 0 | nine cases; per case about 0.2-0.8 s prepare, 0.1-0.2 s merge replay, 0.2-0.9 s estimate replay, 0.02-0.04 s per residual matrix, 0.04 s per coverage allocation |
| fits | `run_fits` | 05:10:03 | 05:10:46 | 0 | 72 sets, 17,280 fits, 39.4 s of fitting; SVD refits 0.6-2.0 ms per set |
| diag_dry | `diagnose_matrix` | 05:18:05 | 05:18:39 | 0 | dry run on the baseline records only; outputs deleted afterwards |
| pool_replay | `../automatic_axes_20260914/check_pool_replays` (packet E0 step 5) | 05:21:10 | 05:21:22 | 0 | nine "exact unfiltered selection replay passed" messages in `logs/pool_replay.log` |
| manifest (rerun) | as above with `--git-head afc23ff`, then again with `--git-head 0caae0a` after the commit | 05:23:18 | 05:23:24 | 0 | after the audit-1 fixes; threads now recorded as requested 1, reported 1. The first rerun passed a dirty-file count of 0 while the audit fixes were still uncommitted; the second rerun, after commit `0caae0a`, computed it live (0) |
| arms (rerun) | `run_arms` | 05:23:24 | 05:24:13 | 0 | identical results under the audited code; records now carry the final `code_md5` |
| fits (rerun) | `run_fits` | 05:24:13 | 05:24:51 | 0 | identical results; replays exact |
| e4_M_gx0 | `run_matcher --arm M --ids gxBQ_window_00_frame_0` | 05:25:19 | 06:07:58 | 0 | E4 smoke: generation 2145 s, camera-first 52 s, all-camera 340 s; final-evaluation and selection replays exact at both rescoring stages; pooled 29,696, 256 final, 3,265 all-camera |
| e4_M_a | `run_matcher --arm M --ids gxBQ_window_00_frame_5 am2_window_00_frame_150` | 06:10:44 | 06:59:06 | 0 | generation 1733 s (GX5), 857 s (Am2-150); 8 exact replays |
| e4_M_b | `run_matcher --arm M --ids am2_window_01_frame_28019 am3_window_00_frame_0 shuttleset_03_scene_0017 shuttleset_03_scene_0019 shuttleset_03_scene_0016 shuttleset_21_scene_0020` | 06:16:03 | 07:33:41 | 0 | generation 1543, 1024, 240, 170, 229, 187 s in that order; 24 exact replays |
| e4_R_a | `run_matcher --arm R --ids gxBQ_window_00_frame_0 gxBQ_window_00_frame_5` | 06:16:04 | 07:36:36 | 0 | generation 2311 s (GX0), 1842 s (GX5); 8 exact replays |
| e4_R_b | `run_matcher --arm R --ids am2_window_00_frame_150 am2_window_01_frame_28019 am3_window_00_frame_0 shuttleset_03_scene_0017 shuttleset_03_scene_0019 shuttleset_03_scene_0016 shuttleset_21_scene_0020` | 06:16:06 | 07:53:26 | 0 | generation 1154, 1367, 1231, 216, 200, 216, 146 s in that order; 28 exact replays |
| diag_partial | `diagnose_matrix --allow-missing --ids gxBQ_window_00_frame_0` | 06:17 | 06:17 | 0 | GX0 only, B and M, R missing; used to test the accounting and gallery on real records; its `e4/diagnosis.json.gz` and `accounting.csv.gz` were deleted locally and are overwritten by the final `diag` run |
| diag | `diagnose_matrix` | 07:54:20 | 07:55:56 | 0 | full matrix: 81 diagnosed rows (nine cases, B/M/R, three stages), none missing or empty; ran under the pushed `d761d7a` code, so its `code_md5` differs from the E4 records' `experiment_code_md5` (`0caae0a`) in `diagnose_matrix.py`, `summarise.py`, `render_gallery.py` and the tests only (`sync.sh` also changed, but `code_md5` hashes `.py` files only); the stage scripts are unchanged between the two |

The launches at 06:11 and 06:16 differ because the first `sync.sh launch` form bound `&` to the whole `cd && nohup` list, so the remote shell waited on the job and the chained launches stalled; the local chain was killed before it could fire them and the three streams were launched with the corrected helper.

Exact UTC times for the reruns are in `runs/<run>/receipts/*_times.txt`.

Parallelism: the packet asks for sequential remote computation. After the smoke, the remaining 17 case-arms ran as four concurrent single-thread processes (one OpenCV thread, single-thread BLAS each), each handling its cases in the packet's order, on an otherwise idle 32-core host. Cases are independent and each process is single-threaded, so results do not depend on the split; the earlier automatic-axes session used the same arrangement with three groups. This trades the packet's strict serial order for wall-clock time.

Code identity: every stage record carries `code_md5` (MD5 of each module in this folder at run time). After the audit-1 fixes, E0-E3 were regenerated so their records and the E4 records share one code set. The manifest records the shared-module hashes, which did not change.

Audits (Opus 5, medium, read-only, four to six minutes each): prompts, digests and reports are in the session's untracked cruft folder `scratch/court_det_fix/worklog/claude_session_15092026_14h32m/` under `audit1/`, `audit2/` and `audit3/`; the worklog summarises each audit's findings and the fixes. Audit 3's first two launches produced nothing (a low-memory kill at 18:06 and a session usage-limit refusal at 18:07); the third ran 19:47-19:53.
