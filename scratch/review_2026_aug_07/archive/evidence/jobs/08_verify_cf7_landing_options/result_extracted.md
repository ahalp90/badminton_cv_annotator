1. Claim under test

Claim A: the landing-filter tuple `(7, 0.004, 5, 7, 0.75)` is independently duplicated in maintained source, and the manifest does not derive it from the executed options.

2. Evidence SUPPORTING the claim

- `src/annotator/e2e_court_annotator.py:79-85` constructs `LANDING_OPTIONS` with the five literals.
- `src/annotator/calibration/gt_scoring.py:464-472` constructs another `LandingFilterOptions` with the same five literals.
- `src/annotator/e2e_court_annotator.py:637-640` writes the five literals again in `_configuration_values()`.
- The executed e2e call passes `LANDING_OPTIONS` at `src/annotator/e2e_court_annotator.py:984-997`.
- `_configuration_values()` does not reference `LANDING_OPTIONS`; it is inserted into manifests at `src/annotator/e2e_court_annotator.py:909` and `:1124`.
- `LandingFilterOptions` has required fields for these five values at `src/annotator/point_winner.py:355-359`; only the boolean fields have defaults at `:360-363`.
- The scoped test search found no assertion comparing manifest `landing_filter_options` with the executed `LANDING_OPTIONS`. The direct configuration test checks only court-policy fields at `tests/test_annotator_measurement.py:58-62`.
- Synthetic `run_video` monkeypatches accept `**_kwargs` without inspecting landing options at `tests/test_annotator_measurement.py:472-485` and `:572-580`.

3. Evidence REFUTING or weakening the claim

- The e2e construction is keyword-based, not positional: `src/annotator/e2e_court_annotator.py:80-84`.
- `LandingFilterOptions` is a shared type, so the field shape is centralised even though the five values are not: `src/annotator/point_winner.py:335-363`.
- `_configuration_values()` does derive `ref_err_px` from `REF_ERR_PX` and `landing_horizons_s` from `LANDING_HORIZONS`: `src/annotator/e2e_court_annotator.py:641` and `:650`.
- The manifest also records `state.resolved_config` separately at `src/annotator/e2e_court_annotator.py:909-910`. Its dataclass serialisation is generic at `:263-274`, but `ResolvedAnnotatorConfig` contains no landing-option fields: `src/annotator/config.py:165-175`.
- The current raw `"replay"` string equals the enum value `DeadMaskMode.REPLAY.value`: `src/annotator/types.py:21-26`, and the default is `DeadMaskMode.REPLAY` at `src/annotator/config.py:123-124`.
- Tests pin the resolved default mode to `DeadMaskMode.REPLAY` at `tests/test_annotator_types.py:71-86`, but do not compare it with the manifest string.

4. Verdicts

- A1: PARTLY. Literal duplication and absence of a shared named value constant are CONFIRMED. The description “positional construction” is REFUTED; both source constructors use keyword arguments.
- A2: CONFIRMED. `_configuration_values()` does not reference `LANDING_OPTIONS`, and the scoped tests contain no manifest-to-executed-options equality assertion.
- A3: PARTLY. The manifest’s `"replay"` is a raw string and is not derived from the execution variable. The current e2e path nevertheless resolves `BaseAnnotatorConfig()` to `DeadMaskMode.REPLAY`: `src/annotator/e2e_court_annotator.py:978`, `src/annotator/run_video.py:516`, `src/annotator/resolve.py:27-32`. The builder consumes that resolved mode at `src/annotator/run_video.py:392-397` and `src/annotator/dead_mask.py:71-78`. No equality assertion links the two representations.
- A4: CONFIRMED. `REF_ERR_PX = 3.5` is defined independently at `src/annotator/e2e_court_annotator.py:78`; `run_video` has an independent default literal at `src/annotator/run_video.py:447`. The e2e path passes `REF_ERR_PX` explicitly at `:996`, so it does not use the `run_video` default.

Commit a6fafb9

- `a6fafb9` is an ancestor of `HEAD`.
- Its patch changes only `src/annotator/point_winner.py` and `tests/test_fps_constants.py`.
- It centralises FPS conversion mechanics in `convert_landing_options()` at `src/annotator/point_winner.py:366-376`.
- The helper scales `settle_win`, `settle_thr`, `settle_min`, and `carry_win`; it does not source the base values or transform `carry_thr`.
- Runtime use is centralised through `pick_landing_to_end()` at `src/annotator/point_winner.py:705-707`.
- The added tests pin those scaling rules at `tests/test_fps_constants.py:180-208`.
- The commit does not centralise the literals in either constructor or in `_configuration_values()`.

5. Unresolved or dynamic surfaces

- Calibration inputs pass a committed mask through `raw_exclusion_mask` at `src/annotator/calibration/gt_scoring.py:439-480` and `:726-735`; that path can bypass `build_dead_mask()`. The e2e path explicitly passes `None` at `src/annotator/e2e_court_annotator.py:996-998`.
- The e2e CLI invokes the module directly via `python -m annotator.e2e_court_annotator` at `src/annotator/e2e_court_annotator.py:1270-1278`; no dynamic registry or `getattr` surface was found for these values.
- Additional equivalent literals occur in tests at `tests/test_annotator_run_video.py:49`, `:800`, and `tests/test_fps_constants.py:181`, `:190-195`. These tests check execution or FPS conversion, not cross-site manifest equality.
- No files were changed. Diff location: none.

Checks: Serena structural tools were used for symbol overviews, symbol bodies, references, declarations, and pattern searches. Scoped `rg`, numbered source reads, `git show`, `git status`, and ancestry checks completed. The Serena launcher could not acquire its read-only lock (`Errno 30`); the active Serena MCP connection remained usable. No tests or lint checks were run because this was a read-only evidence task.