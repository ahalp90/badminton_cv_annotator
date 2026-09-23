# SVD search records: tracked and local-only

The six-case, three-arm experiment completed on Carmack on 23 September 2026 at
`14310f7`. [WORKLOG.md](WORKLOG.md) records scope, timings and visual rulings.

| Path | Storage and purpose |
| --- | --- |
| `run_20260923/cases/` (18 records, 22.66 MiB) | Tracked complete case records: candidates, fits, selections and stage timings; also input to `colour_consistency/floor_trial.py` |
| `run_20260923/generation/` (18 records, 96.49 MiB) | Local-only raw generation evidence produced by `run.py` |
| `run_20260923/completion.json.gz`, `receipts/` | Tracked completion and run receipts |
| `gallery/` | Tracked final gallery and compact comparison |
| `gallery_preview/` (3.98 MiB) | Local-only preliminary output from `build_gallery.py`; not the final result |

Case records contain `generation_record` paths relative to the run root. Those
provenance targets require the local-only generation directory. The tracked
case records alone suffice for their stage timings and the existing floor
trial. Do not assume a fresh clone includes every linked raw record.

All paths remain stable. Raw output is retained and not declared cheaply or
identically regenerable. The original remote root was
`/scratch/ahalperi/court_det_fix/svd_search_run_20260923`; transfers completed.
Use explicit copies for a new checkout and follow `~/.codex/remote_hpc.md` for
remote access. Do not relaunch the completed experiment for orientation.

`run.py --output <fresh-directory>` produces the cases/generation/completion
layout; use its `--case`, `--arm` and `--workers` controls for a deliberate new
trial. `build_gallery.py --input <run-root> --output <gallery-directory>` renders
saved results. Gallery work remains assigned to Sol with shared-template reuse.

The [23 September inventory](../archive/20260923_top_level/untracked_before.json.gz)
records the retained files, sizes and hashes before the storage-policy refresh.
