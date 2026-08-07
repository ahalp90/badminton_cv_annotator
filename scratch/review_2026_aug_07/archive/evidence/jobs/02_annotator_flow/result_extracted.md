## 1. Question investigated

Traced the current `main` branch (`8c675695...`) annotator flow, including entry points, FPS/configuration propagation, masks, rally segmentation, outcomes, artefacts, readers, and batch failure handling.

## 2. Files/symbols inspected

Used Serena `get_symbols_overview`, `find_symbol`, `find_referencing_symbols`, `search_for_pattern`, and `find_declaration`, plus targeted `rg`/line reads.

Inspected:

- `run_video.py`, `e2e_court_annotator.py`, `rally_segmentation.py`
- `rally/{cli,spans,contacts,evidence,serve,trajectory}.py`
- `config.py`, `fps_constants.py`, `resolve.py`
- `court_evidence.py`, `dead_mask.py`, `replay_mask.py`, `composition_mask.py`, `inpaint_guard.py`
- `types.py`, `point_winner.py`, `video_outcomes.py`
- `batch_report.py`, `experiment_records.py`
- `doubles_flag.py`, `broadcast_timeline_labels.py`, `manual_broadcast_timeline_annotator.py`
- `src/scraper/commentary_pairing.py`
- relevant tests for monkeypatch and CLI surfaces

Excluded stale directories were not read.

## 3. Concrete evidence

### Entry points and call chains

- `pyproject.toml` has no `[project.scripts]` section. Its sections are `[project]`, `[project.optional-dependencies]`, and tool sections only: `pyproject.toml:1-179`.
- No `src/annotator/__main__.py` exists. `run_video.py` defines `run_video`, but no `main` or module guard: `src/annotator/run_video.py:430-605`.
- Fixed measurement CLI: `e2e_court_annotator.main` -> `_run_cli_measurement` -> `run_annotator_measurement`: `src/annotator/e2e_court_annotator.py:1332-1349`, `1307-1329`, `1260-1304`.
- Measurement setup loads cases, validates arrays/video metadata, creates raw cuts, and creates eight parent/case states: `src/annotator/e2e_court_annotator.py:1144-1206`.
- Each configuration builds static or detected court evidence, calls `run_video`, writes inference artefacts, then records terminal state: `src/annotator/e2e_court_annotator.py:925-1022`.
- Scoring writes metrics and updates configuration manifests: `src/annotator/e2e_court_annotator.py:1210-1246`.
- Successful measurement then calls `write_summary_and_report` and `clean_run`: `src/annotator/e2e_court_annotator.py:1313-1321`.
- Batch CLI: `rally_segmentation.main` delegates directly to `rally.cli.main`: `src/annotator/rally_segmentation.py:218-224`.
- Batch CLI loads tracks, calls `run_video` in court-optional segmentation-only mode, writes CSVs, and publishes the report: `src/annotator/rally/cli.py:173-263`.
- `run_video` callers found by Serena: `src/annotator/e2e_court_annotator.py:983`, `src/annotator/rally/cli.py:214`, `src/annotator/calibration/gt_scoring.py:734`, and `src/annotator/calibration/sweep.py:417`.

### Configuration and FPS

- Default scrape paths are `masks/`, `rally_spans.csv`, and `contact_frames.csv` below `BADMINTON_SCRAPE_DIR` or `data/scrape_output`: `src/annotator/config.py:24-28`.
- `BaseAnnotatorConfig` carries the threshold preset, dead-mask mode, smoothing mode, FPS overrides, span-opening policy, re-entry policy, and rejected inpaint grades: `src/annotator/config.py:112-139`.
- `scale_for_fps` converts base-30 frame counts and per-frame speeds into final `FpsConstants`: `src/annotator/fps_constants.py:77-107`.
- `resolve` validates overrides, calls `scale_for_fps`, calls `scale_thresholds`, and returns `ResolvedAnnotatorConfig`: `src/annotator/resolve.py:16-38`.
- `run_video` resolves the supplied `fps` internally, then creates shared span options: `src/annotator/run_video.py:516-517`.
- E2E also resolves the same fixed case FPS before calling `run_video`; `run_video` resolves it again because no resolved-config argument is passed: `src/annotator/e2e_court_annotator.py:977-978`, `src/annotator/run_video.py:516-517`.
- The batch CLI obtains FPS from `--fps` or an `id,fps` CSV and passes the value to `run_video`: `src/annotator/rally/cli.py:80-101`, `180-218`.
- E2E fixed cases declare 25 or 30 FPS, while `probe_video` validates the actual video metadata against that value: `src/annotator/e2e_court_annotator.py:121-148`, `488-505`.
- `probe_fps` is used by the composition-mask CLI when `--fps` is omitted: `src/annotator/fps_constants.py:110-126`, `src/annotator/composition_mask.py:147-160`.
- Resolved thresholds/constants are passed into `segment_video` through `_span_options`: `src/annotator/run_video.py:165-175`, `416-426`.
- Resolved `replay_mask_min_frames` is passed to exclusion-mask finalisation: `src/annotator/run_video.py:406-415`.
- Replay signal functions re-derive FPS constants from the raw `fps` argument: `src/annotator/replay_mask.py:51-74`, `161-217`.
- Raw cut detection also re-derives `composition_min_scene_len` from FPS: `src/annotator/court_evidence.py:183-200`.

### Court evidence and masks

- E2E loads raw video cuts and writes `shared/<case_id>/raw_cuts.csv` with `scene_index,start_frame,end_frame`: `src/annotator/e2e_court_annotator.py:761-786`.
- Static and detected court builders return `CourtEvidenceResult`, containing `CourtInputs`, scene records, `keep_vote`, `court_present`, and optional consensus data: `src/annotator/court_evidence.py:52-81`, `131-145`, `533-580`, `583-701`.
- Court evidence writes `court_scenes.csv`, `scene_rows.csv`, `keep_vote.npy`, and `court_present.npy`: `src/annotator/e2e_court_annotator.py:789-802`.
- `build_dead_mask` dispatches `REPLAY`, `COMPOSITION`, or `UNION`; it returns a frame-aligned boolean mask: `src/annotator/dead_mask.py:44-89`.
- Replay mode combines court absence, perspective shift, velocity drop, and optional non-evidence signals: `src/annotator/replay_mask.py:223-242`.
- Composition mode returns a boolean mask plus `CompositionSegment` records: `src/annotator/composition_mask.py:93-129`.
- The composition CLI writes `masks/<video_id>_dead_mask.npy`: `src/annotator/composition_mask.py:136-170`.
- The replay CLI writes `masks/<video_id>_replay.npy`: `src/annotator/replay_mask.py:302-346`.
- `run_video` builds a dead mask when no raw mask is injected, then filters short runs and optionally unions invalid-court frames: `src/annotator/run_video.py:378-415`.
- `inpaint_guard.grade_track` returns per-frame `uint8` codes and an info dictionary: `src/annotator/inpaint_guard.py:266-282`.
- Calibration grades tracks and passes `inpaint_codes` into `run_video`: `src/annotator/calibration/gt_scoring.py:431-484`.
- Replay-mask CLI grades tracks and converts rejected grades into a non-evidence mask: `src/annotator/replay_mask.py:292-298`.

### Rally segmentation and contacts

- `rally_segmentation.py` re-exports names imported from `types`, `config`, `fps_constants`, `rally.contacts`, `rally.evidence`, `rally.serve`, `rally.spans`, and `rally.trajectory`: `src/annotator/rally_segmentation.py:39-108`.
- Serena indexes three locally defined public functions: `scale_thresholds`, `segment_video`, and `main`.
- `scale_thresholds` contains local FPS-threshold composition logic and is referenced by `resolve`: `src/annotator/rally_segmentation.py:111-128`, `src/annotator/resolve.py:23-24`.
- `segment_video` is local orchestration. It applies `apply_replay_mask`, calls `find_rally_spans`, then calls `assemble_contacts`: `src/annotator/rally_segmentation.py:131-215`.
- `find_rally_spans` is implemented in `rally/spans.py`: `src/annotator/rally/spans.py:340-368`.
- `assemble_contacts` is implemented in `rally/contacts.py` and returns `list[ContactCandidate]`: `src/annotator/rally/contacts.py:186-207`.
- `tracker_segments` and `build_sticky_result` are implemented in `rally/evidence.py`: `src/annotator/rally/evidence.py:63-111`, `248-278`.
- `apply_replay_mask` is implemented in `rally/trajectory.py`: `src/annotator/rally/trajectory.py:40-72`.
- `segment_video` returns `(list[tuple[int, int]], list[ContactCandidate])`: `src/annotator/rally_segmentation.py:131-146`, `169-175`.
- `ContactCandidate` fields are `rally_id`, `contact_frame`, `proximity_ok`, `wrist_near`, and `suppressed`: `src/annotator/types.py:58-65`.
- `StickyResult` stores frame-aligned distances, picks, standing counts, ankle positions, box heights, per-slot distances, pixel wrist distances, and analysed flags: `src/annotator/types.py:161-192`.

### Attribution, verdict, landing, and hit-height

- Court-dependent segmentation builds tracker segments, sticky evidence, dead masks, spans, and contacts: `src/annotator/run_video.py:342-427`.
- Contact filtering removes `wrist_near=False`, `suppressed=True`, and definitive-mask contacts; it then fits striker halves and next servers: `src/annotator/video_outcomes.py:101-161`.
- `attribute_half`, `fit_alternation`, and `next_server_half` are implemented in `point_winner.py`: `src/annotator/point_winner.py:126-150`, `167-196`.
- Landing selection calls `landing_window` and `pick_landing_to_end`: `src/annotator/video_outcomes.py:256-332`.
- Verdict construction calls `rally_verdict` and `geometric_verdict`: `src/annotator/video_outcomes.py:397-439`.
- Hit-height construction calls `build_hit_height_rows`; `ValueError` rows are collected as failures: `src/annotator/video_outcomes.py:517-537`.
- `run_video` returns `AnnotatorResult` containing spans, raw/filtered contacts, attribution fields, verdict rows, landings, geometric verdicts, hit heights, and hit-height failures: `src/annotator/run_video.py:69-107`, `568-605`.
- The related in-memory schemas are `VerdictRow`, `GeometricVerdictRow`, `Landing`, and `HitHeightRow`: `src/annotator/point_winner.py:675-779`, `847-853`.

### On-disk artefacts

- `run_video` itself writes no files. E2E serialises its `AnnotatorResult` to `annotations.json`: `src/annotator/e2e_court_annotator.py:805-806`.
- NamedTuple/dataclass/NumPy values become JSON objects, lists, and enum values: `src/annotator/e2e_court_annotator.py:263-290`, `301-305`.
- E2E writes `raw_replay_mask.npy`, `definitive_exclusion_mask.npy`, `annotations.json`, and `landing_horizons.csv`: `src/annotator/e2e_court_annotator.py:1010-1013`.
- `landing_horizons.csv` uses `LANDING_HORIZON_COLUMNS`: `src/annotator/e2e_court_annotator.py:101-110`.
- Scoring writes `strict_contacts.csv`, `wide_edge_contacts.csv`, and `metrics.json`: `src/annotator/e2e_court_annotator.py:818-850`.
- Their headers are `STRICT_CONTACT_COLUMNS` and `WIDE_CONTACT_COLUMNS`: `src/annotator/e2e_court_annotator.py:112-118`.
- `metrics.json` contains `schema_version`, configuration ID, calibration metrics, strict-contact metrics, court-valid fraction, exclusion fraction, and landing-horizon metrics: `src/annotator/e2e_court_annotator.py:832-850`.
- Leaf `manifest.json` contains status, case/configuration identity, source/command/device/timing, resolved config, inputs, artefact records, and failure: `src/annotator/e2e_court_annotator.py:860-921`.
- Root `manifest.json` contains run status, timing, environment, cases, configurations, setup/scoring failures, and exit code: `src/annotator/e2e_court_annotator.py:1080-1132`.
- Failure JSON fields are `schema_version`, scope, case, parent, stage, exception type, message, traceback, and timestamp: `src/annotator/e2e_court_annotator.py:688-718`.
- `experiment_records.build_summary` reads root `manifest.json`, each referenced leaf `manifest.json`, and each leaf `metrics.json`; it requires eight configurations: `src/annotator/experiment_records.py:74-111`.
- `write_summary_and_report` writes `summary.json` and `report.md`: `src/annotator/experiment_records.py:156-163`.
- `clean_run` scans non-NPY files and may create `experiments/annotator` backup archives named `<run>_cleaned_<timestamp>.tar.gz`: `src/annotator/experiment_records.py:175-180`, `243-251`, `321-343`.
- Batch `rally_spans.csv` header: `video_id,rally_id,start_frame,end_frame`: `src/annotator/rally/cli.py:149-159`.
- Batch `contact_frames.csv` header: `video_id,rally_id,contact_frame,proximity_ok,wrist_near,suppressed`: `src/annotator/rally/cli.py:160-169`.
- Batch report path is `<rally_spans_stem>_batch_report.txt`; its sections and counts are plain text: `src/annotator/batch_report.py:30-85`, `88-101`.

## 4. Callers/consumers found

- `resolve.resolve`: called by `run_video` and E2E configuration setup: `src/annotator/run_video.py:516`, `src/annotator/e2e_court_annotator.py:977`.
- `read_whole_video_flags`: called by the batch CLI before track processing: `src/annotator/rally/cli.py:114-135`.
- `doubles_flag` CLI reads `rally_spans.csv` and writes `doubles_flags.csv` with `video_id,rally_id,doubles_flag`: `src/annotator/doubles_flag.py:31-34`, `104-145`, `148-175`.
- `src/scraper/commentary_pairing.py` reads `rally_spans.csv` and `<video_id>_replay.npy`: `src/scraper/commentary_pairing.py:206-236`, `332-354`.
- No reader for `contact_frames.csv` was found in `src`.
- `broadcast_timeline_labels` is consumed by the manual GUI only. It reads an existing label CSV and writes it after edits: `src/annotator/manual_broadcast_timeline_annotator.py:475-480`, `520-545`, `574-625`.
- Manual label CSV schema is `video_id,fps,frame_count,start_frame,end_frame,truth,note`: `src/annotator/broadcast_timeline_labels.py:17-25`, `220-246`, `261-295`.
- No caller from `run_video`, E2E, or batch segmentation was found for the broadcast timeline modules.

## 5. Counterevidence / surprises

- The expected `pyproject.toml` script-entry-point surface is absent.
- The expected package-level `__main__` and `run_video.main` are absent.
- `rally_segmentation.main` is a wrapper; batch parsing and output writing live in `rally/cli.py`.
- E2E does not pass `inpaint_codes` or `shuttle_hallucination_mask` to `run_video`; those inputs are present in the callable API but are supplied by calibration/replay-mask paths.
- The batch report explicitly states that rally-level exclusion reasons are not recorded in the report: `src/annotator/batch_report.py:81-84`.
- The E2E scorer reads `court_present.npy` back to compute `court_valid_fraction`: `src/annotator/e2e_court_annotator.py:847-850`.
- A source search found no E2E-run-root readers for `annotations.json`, landing horizons, strict/wide contacts, or saved exclusion masks. Calibration fixtures separately use files with some of the same names under an external fixture root: `src/annotator/calibration/fixtures.py:89-97`.

## 6. Unresolved or dynamic surfaces

- Imported aliases in `rally_segmentation.py` are visible through text imports but are not independently indexed by Serena as local symbols. Serena reference queries succeeded for local `scale_thresholds`, `segment_video`, and `main`; imported aliases returned no matching local symbol.
- `BADMINTON_SCRAPE_DIR` changes batch artefact paths at import time: `src/annotator/config.py:24-28`.
- Calibration fixture paths depend on `ANNOTATOR_FIXTURES_ROOT`: `src/annotator/calibration/fixtures.py:115-122`.
- `getattr` is used for JSON/dataclass serialisation and detector-device capture, not for a discovered pipeline dispatch: `src/annotator/e2e_court_annotator.py:263-290`, `571-589`, `1140-1142`.
- Test-only monkeypatch targets include `run_video`, `segment_video`, `build_dead_mask`, sticky builders, point-winner functions, and `grade_track`: `tests/test_annotator_run_video.py:127-131`, `278-281`, `469-503`, `tests/test_annotator_measurement.py:337-338`, `tests/test_annotator_replay_mask.py:489-490`.
- No lint or test suite was run because this was a read-only evidence collection task. Repository status checks exited 0.
- Files changed: none. Diff location: none.