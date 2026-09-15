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
| diag_dry | `diagnose_matrix` | 05:4x | 05:4x | 0 | dry run on the baseline records only; outputs deleted afterwards |
| pool_replay | `../automatic_axes_20260914/check_pool_replays` (packet E0 step 5) | 05:5x | 05:5x | 0 | nine "exact unfiltered selection replay passed" messages in `logs/pool_replay.log` |
| manifest (rerun) | as above with `--git-head afc23ff` | 05:5x | 05:5x | 0 | after the audit-1 fixes; threads now recorded as requested 1, reported 1. The dirty-file count was passed as 0 while the audit fixes were still uncommitted; the manifest is rerun after the next commit with the live count |
| arms (rerun) | `run_arms` | 05:5x | 05:5x | 0 | identical results under the audited code; records now carry the final `code_md5` |
| fits (rerun) | `run_fits` | 05:5x | 05:5x | 0 | identical results; replays exact |
| e4_M_gx0 | `run_matcher --arm M --ids gxBQ_window_00_frame_0` | 06:00 | | | E4 smoke test |

Exact UTC times for the reruns are in `runs/<run>/receipts/*_times.txt`.

Code identity: every stage record carries `code_md5` (MD5 of each module in this folder at run time). After the audit-1 fixes, E0-E3 were regenerated so their records and the E4 records share one code set. The manifest records the shared-module hashes, which did not change.

Audit 1 (Opus 5, medium, read-only, about 30 minutes): prompt, evidence digest and report are in the session's untracked cruft folder under `audit1/`; the worklog summarises the findings and the fixes.
