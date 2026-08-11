# Current worklog

## 2026-08-11: close-out and tidy

- Made the plain-language rewrite of `report.md` and `README.md` authoritative.
- Kept the contents below “What should we try next?” and clarified the `multiple` count beside the anchor-alignment plot.
- Corrected the server comparison to use the same earliest-contact fallback for direct motion and prepend/refit: 163/239 versus 159/239.
- Moved the full two-day worklog, plan, findings, decisions and retained review records to `ARCHIVE/` without shortening them.
- Replaced the live handover set with short current documents.
- Moved the investigation from its timestamped run directory to the stable `scratch/serve_start_trajectory_exploration/` path.
- Kept the frozen release asset, prepared inputs and generated outputs local and ignored.

Relocation checks:

- frozen input preparation: exit 0; all three fixtures and recorded MD5 values verified;
- full result regeneration: exit 0;
- source-backed validator: exit 0; checked 292 rallies, 344 spans, 1,012 path points, 16 fixed-rule rows, all metrics, the report and six plots;
- focused tests: exit 0; 55 passed;
- focused Ruff: exit 0.

The close-out commit and local WebUI-pack path are reported with the final handover rather than extending this log again.

For all work before this cleanup, read [ARCHIVE/worklog_20260810-11.md](ARCHIVE/worklog_20260810-11.md).
