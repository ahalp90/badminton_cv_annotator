I’ll inspect the repository rules, `fps_constants.py`, and all in-scope production references. I’ll use semantic reference search if Serena is available, then cross-check every candidate text hit with line-level evidence.
The shared module currently exposes four module constants, two public classes (`ScalingKind`, `FpsConstants`), one public field-name set, and two public functions. Direct production imports span 14 files; I’m now checking whether any non-importing code still embeds frame-rate conversions or table values.
**Question**

The shipping fps policy is centralised. No independent production formula was found that duplicates a `fps_constants` policy row.

**Scope inspected**

All Python files under `src/annotator/**` and `src/scraper/**` were searched. Targeted reads covered `fps_constants.py`, its importers, `rally_segmentation.py`, `resolve.py`, calibration code, video outcomes, scraper pairing, and validation-overlay timing code. Tests were counted only.

**Evidence**

Public project-defined symbols in `fps_constants.py`:

| Symbol | Production importers / uses |
|---|---|
| `BASE_FPS` | No external importer; internal scaling at `fps_constants.py:35,37` |
| `REST_SPEED_BASE30` | No external importer; `scale_for_fps()` at `fps_constants.py:88` |
| `START_SPEED_BASE30` | No external importer; `scale_for_fps()` at `fps_constants.py:89` |
| `ScalingKind`, `.scale` | Directly used by `calibration/{scoring.py:13,269, gt_scoring.py:15,404}`, `point_winner.py:55,369-373`, `resolve.py:9,26`; re-exported by `types.py:15` and consumed by `calibration/sweep.py:50,358` |
| `FpsConstants` | Imported by `config.py:16`, `point_winner.py:55`, `rally_segmentation.py:53`, `rally/spans.py:13` |
| `FPS_CONSTANT_FIELD_NAMES` | `resolve.py:9,13` |
| `scale_for_fps` | `config.py:16,36`; `composition_mask.py:35,158-159`; `court_evidence.py:23,190`; `rally/contacts.py:6,13-18`; `rally_segmentation.py:53,118`; `replay_mask.py:32,72,212`; `resolve.py:9,23`; `calibration/sweep.py:51,292`; `scraper/commentary_pairing.py:26,150` |
| `probe_fps` | `composition_mask.py:34,158` |

`FpsConstants` contains 23 public final-value fields. They are consumed through `resolve`, `run_video`, `rally/spans.py`, `rally/serve.py`, `point_winner.py`, `replay_mask.py`, and `video_outcomes.py`.

`scale_thresholds()` reads `scale_for_fps(fps)` or a passed `FpsConstants` at `rally_segmentation.py:118`, then copies shared fields at `:120-127`. It does not restate table values. Only dimensionless `contact_impulse_multiple` remains caller/override supplied.

`resolve()` derives its allowed rows from `FPS_CONSTANT_FIELD_NAMES` at `resolve.py:13`, calls `scale_for_fps()` at `:23`, reuses the resulting constants at `:24`, and applies shared `ScalingKind.FRAME_COUNT` at `:26`.

| Class | Classification |
|---|---|
| (a) Shared policy | `fps_constants.py:18-106`; all `scale_for_fps()` and `ScalingKind.scale()` production call sites above; base-30 caller inputs in calibration and landing options are still converted by the shared scaler |
| (b) Independent duplicate | None found |
| (c) Benign local conversion/literal | `scraper/commentary_pairing.py:93,172` seconds/frame conversion; `video_outcomes.py:349` requested-horizon conversion; `manual_broadcast_timeline_annotator.py:435` display conversion; `e2e_court_annotator.py:616` reporting conversion; validation-overlay conversions at `core/timeline.py:115`, `core/decode.py:190-193`; fixture rates at `calibration/fixtures.py:299,319,339` and `e2e_court_annotator.py:122-125` |

No production `* 30`, `/ 30`, or `29.97` hits were found outside the shared implementation. The remaining `25.0`/`30` hits are shared calls, fixture metadata, UI/pixel constants, comments, counts, or timeouts.

**Production references**

The main flow is:

`resolve()` → final `FpsConstants` and scaled thresholds → `run_video()` → replay masking, rally spans, serve setup, contact detection, landing logic, and outcome scoring.

The scraper has one shared-policy consumer, `commentary_pairing.py:150`, plus independent seconds/frame unit conversions.

**Test references**

Textual counts only:

- 8 test files mention a public fps symbol.
- `scale_for_fps`: 53 hits across 7 files.
- `ScalingKind`: 33 hits across 2 files.
- `FpsConstants`: 4 hits across 1 file.
- `FPS_CONSTANT_FIELD_NAMES`: 2 hits across 1 file.
- `probe_fps`: 5 hits across 2 files.
- `BASE_FPS`, `REST_SPEED_BASE30`, `START_SPEED_BASE30`: 0 hits.

**Counterevidence**

None requiring classification as (b). Potential-looking base-30 literals in calibration grids and landing options are inputs later routed through the shared scaler.

**Unresolved/dynamic surfaces**

- `calibration/sweep.py:295-299` uses dynamic `hasattr/getattr` over override names.
- `resolve.py` accepts caller-supplied `overrides_base30` and `FpsConstants`.
- Several modules independently acquire fps metadata via OpenCV or ffprobe. These are fps-source surfaces, not scaling-policy formulas.

Checks: Serena symbol overview and reference searches succeeded; all `rg` inventory searches exited 0; `git diff --name-only -- src/annotator src/scraper` was empty. No files changed, so there is no diff location. Lint, type checks, and tests were not run for this read-only evidence pass.
