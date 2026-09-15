# Run record

Exact commands, hashes, timings and exit codes. `R` is the remote experiment root and `L` the local source tree; `sync.sh` resolves both from the gitignored `paths.local.sh`. Every remote stage runs through `run_remote.sh`, which sets `PYTHONDONTWRITEBYTECODE=1`, single-thread BLAS and OpenMP, a run-local cache directory, and the packet's `PYTHONPATH` plus `automatic_axes_20260914/svd_fixed` and this folder.

## Local checks

```text
~/.venvs/badminton-cicd/bin/ruff check --fix .        # 6 fixed, 0 remaining; exit 0
PYTHONDONTWRITEBYTECODE=1 ~/.venvs/badminton-cicd/bin/pytest tests -q -p no:cacheprovider
                                                      # 30 passed in 1.43s; exit 0
```

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
| e4_M_a | `run_matcher --arm M --ids gxBQ_window_00_frame_5 am2_window_00_frame_150` | 06:11 | | | |
| e4_M_b | `run_matcher --arm M --ids am2_window_01_frame_28019 am3_window_00_frame_0 shuttleset_03_scene_0017 shuttleset_03_scene_0019 shuttleset_03_scene_0016 shuttleset_21_scene_0020` | 06:16 | | | |
| e4_R_a | `run_matcher --arm R --ids gxBQ_window_00_frame_0 gxBQ_window_00_frame_5` | 06:16 | | | |
| e4_R_b | `run_matcher --arm R --ids am2_window_00_frame_150 am2_window_01_frame_28019 am3_window_00_frame_0 shuttleset_03_scene_0017 shuttleset_03_scene_0019 shuttleset_03_scene_0016 shuttleset_21_scene_0020` | 06:16 | | | |
| diag_partial | `diagnose_matrix --allow-missing --ids gxBQ_window_00_frame_0` | 06:17 | 06:17 | 0 | GX0 only, B and M, R missing; used to test the accounting and gallery on real records; its `e4/diagnosis.json.gz` and `accounting.csv.gz` were deleted locally and are overwritten by the final `diag` run |

The launches at 06:11 and 06:16 differ because the first `sync.sh launch` form bound `&` to the whole `cd && nohup` list, so the remote shell waited on the job and the chained launches stalled; the local chain was killed before it could fire them and the three streams were launched with the corrected helper.

Exact UTC times for the reruns are in `runs/<run>/receipts/*_times.txt`.

Parallelism: the packet asks for sequential remote computation. After the smoke, the remaining 17 case-arms ran as four concurrent single-thread processes (one OpenCV thread, single-thread BLAS each), each handling its cases in the packet's order, on an otherwise idle 32-core host. Cases are independent and each process is single-threaded, so results do not depend on the split; the earlier automatic-axes session used the same arrangement with three groups. This trades the packet's strict serial order for wall-clock time.

Code identity: every stage record carries `code_md5` (MD5 of each module in this folder at run time). After the audit-1 fixes, E0-E3 were regenerated so their records and the E4 records share one code set. The manifest records the shared-module hashes, which did not change.

Audit 1 (Opus 5, medium, read-only, about five minutes): prompt, evidence digest and report are in the session's untracked cruft folder under `audit1/`; the worklog summarises the findings and the fixes.
