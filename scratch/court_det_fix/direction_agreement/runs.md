# Run record

Exact commands, hashes, timings and exit codes. `R` is the remote experiment root and `L` the local source tree; `sync.sh` resolves both from the gitignored `paths.local.sh`. Every remote stage runs through `run_remote.sh`, which sets `PYTHONDONTWRITEBYTECODE=1`, single-thread BLAS and OpenMP, a run-local cache directory, and the packet's `PYTHONPATH` plus `automatic_axes_20260914/svd_fixed` and this folder.

## Local checks

```text
~/.venvs/badminton-cicd/bin/ruff check --fix .        # 6 fixed, 0 remaining; exit 0
PYTHONDONTWRITEBYTECODE=1 ~/.venvs/badminton-cicd/bin/pytest tests -q -p no:cacheprovider
                                                      # 30 passed in 1.43s; exit 0
```

## Remote stages

Run name: see `run_name.txt`. Log, PID and exit receipts: `runs/<run>/{logs,receipts}/`.
