I’ll inspect only the three named source files, map the batch and per-video control flow, and cite exact line ranges for failure, skip, summary, and shared-state behaviour. No files will be changed.
**Question**

Per-video failure, skip, resume, and pre-existing-output mechanics in the fixed e2e batch path.

**Scope inspected**

- [e2e_court_annotator.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/e2e_court_annotator.py)
- [batch_report.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/batch_report.py)
- [experiment_records.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/experiment_records.py)

No files changed. Scoped `git diff` was empty. I ran line-numbered reads and `rg` inventories for exceptions, statuses, existence checks, and report writers. No runtime tests were run.

**Evidence**

1. **Failure handling**

   - Shared case loading runs once per fixed case. `Exception` is caught at [e2e_court_annotator.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/e2e_court_annotator.py:1172) lines 1172-1197, a failed `CaseData` and `shared/.../failure.json` are recorded, and the loop continues to the next case. Dependent configurations later become failed.
   - Court-evidence failures are caught separately for `CourtConsensusError` and general `Exception` at [e2e_court_annotator.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/e2e_court_annotator.py:940) lines 940-973. Partial evidence where available, `failure.json`, status `failed`, and a terminal configuration manifest are written; the function returns, so the outer configuration loop continues.
   - Inference and output-writing failures catch `Exception` at [e2e_court_annotator.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/e2e_court_annotator.py:975) lines 975-1021. They are recorded as `inference` failures and the configuration returns as `failed`.
   - GT verification failure catches `Exception` at [e2e_court_annotator.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/e2e_court_annotator.py:1213) lines 1213-1223. It writes `scoring_failure.json`, leaves inference-only states as `inference_only`, writes their manifests, and returns. Per-configuration scoring failures catch `Exception` at lines 1232-1246, record failure, mark the configuration failed, and continue.
   - Setup failure catches `Exception` at [e2e_court_annotator.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/e2e_court_annotator.py:1283) lines 1283-1291. It writes `setup_failure.json`, skips configuration execution, and sets exit code 1. Terminal run-manifest failure is caught at lines 1299-1303 and logged to stderr.
   - Reporting or cleaning failures catch the listed exceptions at [e2e_court_annotator.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/e2e_court_annotator.py:1316) lines 1316-1321, log to stderr, and return 1. CLI `OSError`/`ValueError` is converted to a parser error at lines 1342-1345.
   - Parsing and pin-validation catches re-raise as `ValueError` at lines 347-350, 362-365, and 416-419. They therefore reach setup or per-case handling. Package-metadata absence is handled as `None` at lines 1063-1066.
   - [experiment_records.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/experiment_records.py:280) converts unreadable scanner JSON to `RuntimeError`; `clean_run` wraps and re-raises all exceptions at lines 326-340. The e2e caller catches that at lines 1316-1321. There are no `try`/`except` blocks in `batch_report.py`.

   Per-video/configuration failures reach the e2e `manifest.json` through status and failure artefact records, but prevent `summary.json` and `report.md`: `_run_cli_measurement` returns immediately when the measurement exit code is non-zero at lines 1313-1315.

2. **Existence, skip, and resume handling**

   No existence-based skip or resume path was found.

   - Existing output roots are rejected at [e2e_court_annotator.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/e2e_court_annotator.py:679) lines 679-681 and again for timestamp collisions at lines 1310-1311. Root creation also uses `exist_ok=False` at line 1267.
   - Configuration directories use `exist_ok=True` at line 937, but no work is skipped or reused. Files are written afresh.
   - `_artifact_record` requires an output file to exist and hashes it at lines 254-260. This validates closed artefacts; it is not a reuse check.
   - `experiment_records.py` validates existing manifest paths as non-symlink regular files inside the run root at lines 66-71, and validates the run directory at lines 166-172. `build_summary` also requires eight configuration records and readable JSON/metrics files at lines 74-95.
   - Those post-run readers do not recompute or compare recorded artefact hashes. Input pins are separately hash-verified at [e2e_court_annotator.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/e2e_court_annotator.py:405) lines 405-409 and 767.

3. **Summary/report distinctions**

   - `batch_report.py` counts only `processed`, `excluded`, and `skipped` statuses at [batch_report.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/batch_report.py:39) lines 39-44. A processed-empty video is still `processed` with zero rally/contact counts. `failed` has no dedicated count and is rendered only as `status; reason` at lines 55-66. `skipped` gets a count at line 76. The report is written and printed at lines 93-100.
   - The e2e manifest distinguishes `failed`, `partial_failure`, and `succeeded` at [e2e_court_annotator.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/e2e_court_annotator.py:1086) lines 1086-1096. Cases expose `not_run`, `failed`, or `succeeded`; configurations expose `not_run`, `failed`, `inference_only`, or `succeeded` at lines 1097-1109. There is no `processed-empty` or `skipped` status.
   - `experiment_records.py` copies leaf configuration statuses into `summary.json` at lines 85-93, but the Markdown report shows only the overall outcome and metrics at lines 140-153. `summary.json` and `report.md` are written at lines 156-163, only after the e2e measurement returns zero.

4. **Silent absence with exit code 0**

   No concrete in-scope e2e path was found.

   The fixed `CASES` and `PARENTS` produce all eight configuration states. Each state is visited at [e2e_court_annotator.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/e2e_court_annotator.py:1200) lines 1200-1207 and run at lines 1293-1298. Any failed or non-`succeeded` state forces exit code 3. Exit code 0 additionally requires successful summary/report generation and cleaning at lines 1313-1329.

   `batch_report.py` trusts the caller’s `outcomes` sequence and performs no cardinality check, so an external caller could omit a video. That caller is outside the inspected scope, and the module itself does not set a process exit code.

5. **Shared mutable state**

   - Confirmed mutable global state: `_setup` temporarily reassigns `gt_scoring_module.SHARED_FILES` at [e2e_court_annotator.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/src/annotator/e2e_court_annotator.py:1157) lines 1157-1163, with an explicit `NOT THREADSAFE` note. It is restored in `finally`.
   - `LANDING_OPTIONS` is a module-level options object at lines 79-85 and is passed to every `run_video` call at line 988. No mutation is visible in the inspected files.
   - The detector is shared per run through `RunDriver`, but is not module-level. `batch_report.py` and `experiment_records.py` have no mutable per-video module state.

**Counterevidence**

`e2e_court_annotator.py` imports `experiment_records`, not `batch_report`; the generic `VideoOutcome` report is not part of the e2e call chain. The e2e final manifest is configuration-oriented rather than a direct per-video outcome report.

**Unresolved/dynamic surfaces**

- Callers that construct `VideoOutcome` are outside scope.
- Failure-record and terminal-manifest writes inside exception handlers are not themselves protected.
- `run_video` internals and the mutability of `LandingFilterOptions` are outside scope.
