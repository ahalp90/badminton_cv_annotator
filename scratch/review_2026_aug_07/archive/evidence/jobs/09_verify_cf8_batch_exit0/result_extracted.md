1. Claim under test

Claim B concerns batches with at least one input track where every retained video is skipped because of an FPS-ID mismatch or a caught per-video exception.

2. Evidence SUPPORTING the claim

- B1: FPS mismatch sets `VideoOutcome(..., 'skipped')`, logs `skipping <id>: absent from fps CSV`, then continues: `src/annotator/rally/cli.py:201-206`.
- B1: The per-video `Exception` handler stores a flattened reason, logs `skipping <id>: <exception>`, and continues: `src/annotator/rally/cli.py:228-234`.
- B1: The skip reason is included in the final report: `src/annotator/batch_report.py:64-66`.
- B1: The CLI comment describes the exception path as “log-and-skip per video”: `src/annotator/rally/cli.py:228`.
- B2: `format_batch_report` computes processed and skipped counts but only branches on `video_count` and `all_excluded`; `processed_count == 0` does not raise: `src/annotator/batch_report.py:39-53`.
- B2: The CLI only checks `all_excluded_error`, not processed or skipped counts: `src/annotator/rally/cli.py:248-263`.
- B2: The compatibility facade calls the CLI and returns normally without an explicit exit code: `src/annotator/rally_segmentation.py:218-224`.
- B3: The two CSV writers always write their headers and then any rows supplied: `src/annotator/rally/cli.py:149-170`.
- B3: The writer is called whenever `all_excluded_error is None`, independently of the processed count: `src/annotator/rally/cli.py:248-252`.
- B3: The output files are the configured rally-spans CSV and contact-frames CSV: `src/annotator/rally/cli.py:100-101`, `src/annotator/config.py:27-28`.
- B3: Header columns are `video_id,rally_id,start_frame,end_frame` and `video_id,rally_id,contact_frame,proximity_ok,wrist_near,suppressed`: `src/annotator/rally/cli.py:156-168`.
- B4: A test makes `run_video` raise `ValueError()` and directly calls the CLI without expecting an exception. It checks `batch completed: 0 of 1` and a skipped outcome: `tests/test_batch_report.py:129-145`.
- B4: The FPS-mismatch test directly calls the CLI, checks that no video was processed, and checks the skip warning: `tests/test_fps_cli_and_tracknet_modes.py:58-103`.
- B5: The report has distinct all-excluded and ordinary-completion branches: `src/annotator/batch_report.py:46-53`.
- B6: Missing-FPS videos are skipped, then the final CSV is still opened and written: `src/scraper/commentary_pairing.py:340-364`.
- B6: The final pairing write has no `all_rows`/paired-count guard. It writes the header, writes zero or more rows, and logs the count: `src/scraper/commentary_pairing.py:359-364`.
- B6: `PAIRS_CSV` defaults to `rally_commentary_pairs.csv`: `src/scraper/config.py:35`.
- B6: The all-skipped missing-FPS test calls `main()` normally and confirms the resulting CSV has no data rows: `tests/test_scraper_commentary_pairing.py:682-710`.

3. Evidence REFUTING or weakening the claim

- The all-excluded path is explicitly guarded before CSV writing. It creates an error when every track is filtered out: `src/annotator/rally/cli.py:141-145`.
- The all-excluded path skips `_write_segmentation_csvs`, publishes a failed report, and raises the stored error: `src/annotator/rally/cli.py:248-263`.
- The all-excluded test confirms non-success behaviour and confirms both segmentation CSVs are absent: `tests/test_batch_report.py:147-179`.
- The exception handler catches `Exception`, not `BaseException`; `SystemExit` and `KeyboardInterrupt` would not become skip records: `src/annotator/rally/cli.py:207-234`.
- Report publication or CSV I/O failures still propagate: `src/annotator/rally/cli.py:254-261`.
- The success tests call `main()` directly. No focused test asserts an OS-level subprocess `returncode == 0`.
- Commentary pairing can fail before the final write during manifest validation. Tests confirm that such failures leave the existing pairs CSV unchanged: `tests/test_scraper_commentary_pairing.py:381-403`, `tests/test_scraper_commentary_pairing.py:455-486`.

4. Verdict on each factual sub-claim

- B: PARTLY. The stated headline, header-only segmentation CSVs, and normal process return hold for ordinary FPS-mismatch and `Exception` skip paths when report and file writes succeed. The literal “every exception” wording excludes `BaseException` and I/O failures.
- B1: CONFIRMED. Both skip assignments are in `src/annotator/rally/cli.py:201-234`. The CLI emits a warning and later prints the skip reason in the report. It does not call traceback or `log.exception`; the caught exception becomes a skip record.
- B2: CONFIRMED. No generic processed-count or skipped-count failure check was found. The only zero-processed guard is the doubles all-excluded condition.
- B3: CONFIRMED. Both segmentation CSVs receive headers with zero rows when the writer is reached.
- B4: CONFIRMED, with qualification. The comment at `src/annotator/rally/cli.py:228` documents deliberate per-video continuation, and direct-call tests pin normal completion. No subprocess exit-code assertion was found.
- B5: CONFIRMED. All-excluded batches use the explicit guard and raise. `git log -S'all_excluded_error'` identifies commit `db33b85`, whose message is “Split rally segmentation responsibilities”. Commit `ba4e750` states that the all-excluded report precedes the “existing refusal to write empty outputs”.
- B6: PARTLY. The final pairing stage has no empty-output or mass-failure guard and writes an empty/header-only CSV after normal all-skip processing. Earlier validation exceptions can prevent reaching that stage.

5. Unresolved or dynamic surfaces

- `src/annotator/rally/cli.py:191-192` imports `publish_batch_report` and `run_video` inside `main`; tests patch the `annotator.run_video.run_video` target: `tests/test_batch_report.py:78-90`.
- The facade re-exports the CLI entry point through `_cli_main`: `src/annotator/rally_segmentation.py:52-54`, `src/annotator/rally_segmentation.py:218-224`.
- Scoped searches found no additional target wrapper, Makefile target, project script, or CI count check. CI runs the test suite only: `.github/workflows/ci.yml:39-60`.
- A focused pytest run could not execute in this read-only environment: system `pytest` was unavailable (exit 127), and the pinned pytest command failed because no writable temporary directory was available (exit 1).

Files changed: none. Diff location: none; access was read-only.