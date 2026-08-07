I’ll inspect the repository instructions first, then trace the two test files through `serve.py` and their production callers. I’ll also check the requested collection count and the relevant git history, without changing files.
The worktree already has unrelated user changes (`.gitignore`, `.serena/`, and `scratch/review_2026_aug_07/`); I’ll leave them untouched. Serena is available and activated for this checkout, so I’ll use it for symbol references and direct reads for the two test files.
The production path is now clear: `run_video._run_court_segmentation` builds sticky evidence, `build_serve_options` converts it into `ServeSetupInputs`, and `rally.spans._serve_start_find_rally_spans` resolves the gate before `_ServeGate.qualifies` reaches `_sticky_serve_setup_before`. I’m checking the upstream evidence shapes and resolution handling next, because those determine whether each validator case is merely synthetic or can enter through the public run path.
**Question**

The cluster protects B1 stillness primitives, B2 sticky serve-gate routing, sentinel handling, builder conversion, and fail-loud option validation. On the normal production path, malformed setup arrays and gate arguments are synthetic misuse. Malformed `resolution` can enter through the public `run_video` path because only non-`None` is checked before the serve builder.

**Scope inspected**

`A` = [test_annotator_serve_setup.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/tests/test_annotator_serve_setup.py); `B` = [test_annotator_serve_setup_b2.py](/home/ariel/Documents/COSC594/badminton_cv_annotator/tests/test_annotator_serve_setup_b2.py).

Also inspected `serve.py`, `run_video.py`, `rally/spans.py`, sticky evidence construction, type definitions, facade exports, and relevant history.

Changed files: none. Diff location: none.

Assumption: “normal production path” means `run_video` using `build_sticky_result`, resolved FPS constants, and court-generated resolution. Callers violating runtime annotations count as direct garbage callers.

**Evidence**

Collection: A has 38 cases; B has 24; combined total is 62. There are 31 test functions.

(a) Behavioural rules

| Ref | Test | Cases | Assertion |
|---|---|---:|---|
| A:38 | `test_series_drift_sentinel_and_partial_zero_detection` | 1 | Ignores paired-zero/NaN rows but counts partial-zero coordinates; returns count 2 and drift √13. |
| A:45 | `test_series_drift_odd_split_assigns_extra_sample_to_first_half` | 1 | Odd detected samples use a ceiling split in the first half, producing drift 25. |
| A:80 | `test_serve_setup_still_claimed_frame_is_inclusive` | 1 item, 2 calls | The claimed frame is included; movement at that frame causes rejection. |
| A:97 | `test_serve_setup_still_fails_when_one_player_is_over_threshold` | 1 | Every requested player must satisfy the stillness threshold. |
| B:30 | `test_sticky_lanes_route_each_bound_median` | 5 | Median standing counts below 1 reject; counts from 1 upward route to a passing lane. |
| B:44 | `test_partial_lane_rejects_alternating_cross_slot_minimum` | 1 | Alternating slot evidence cannot be pooled into a coherent partial lane. |
| B:51 | `test_standard_lane_accepts_either_slot_when_its_ratio_passes` | 1 | The standard lane passes when either slot’s distance/height ratio passes. |
| B:56 | `test_standard_lane_pairs_each_distance_with_its_own_height` | 1 | Each slot’s distance is paired with that slot’s height. |
| B:80 | `test_sticky_distance_window_excludes_burst_frame` | 1 | The burst frame is excluded from the setup-distance window. |
| B:89 | `test_burst_frame_count_cannot_change_lane_selection` | 1 | Lane selection uses pre-burst counts, not the burst row. |
| B:123 | `test_standard_lane_stillness_rejects_a_player_even_when_distance_passes` | 1 | Passing distance evidence cannot override a required player’s failed stillness check. |

(b) Fail-closed semantics

| Ref | Test | Cases | Assertion |
|---|---|---:|---|
| A:61 | `test_series_drift_below_two_detected_returns_nan_and_count` | 2 | Fewer than two detected points returns NaN with the detected count. |
| A:75 | `test_serve_setup_still_requires_each_player_and_fails_closed_on_nan` | 1 | Missing ankle evidence for one required player rejects the gate. |
| A:88 | `test_serve_setup_still_clips_window_at_frame_zero` | 1 | A clipped window with insufficient evidence rejects. |
| A:92 | `test_serve_setup_still_nonpositive_body_unit_fails_closed` | 1 | A zero body-height unit rejects. |
| B:37 | `test_sticky_coverage_fails_closed_and_stillness_can_be_off` | 1 item, 2 calls | Missing analysed coverage rejects; complete coverage can pass when stillness is disabled. |
| B:66 | `test_sticky_gate_ignores_invisible_corner_garbage` | 1 | NaN/unanalysed evidence rejects; fabricated finite invisible evidence would incorrectly pass. |
| B:99 | `test_burst_analysed_row_is_ignored_without_stillness_and_required_with_it` | 1 item, 2 calls | Burst-row coverage is ignored without stillness and required when stillness is enabled. |
| B:118 | `test_standard_lane_presence_floor_keeps_absent_slot_from_being_rescued` | 1 | An absent standard-lane slot cannot be rescued by the other slot. |
| B:132 | `test_builder_converts_and_preserves_sentinels` | 1 | Builder converts units, removes the retired field, and preserves NaN/+inf/analysed sentinels. |
| B:198 | `test_one_row_clipped_window_fails_closed_in_both_lanes` | 2 | A frame-zero setup window fails in both count lanes. |

(c) Input-validation rejection

| Ref | Test | Cases | Rejected states |
|---|---|---:|---|
| A:70 | `test_series_drift_rejects_shape_rank_and_dtype` | 4 | Rank-1, `(n,3)`, rank-3, and string arrays. |
| A:104 | `test_serve_setup_still_rejects_invalid_window` | 4 | `0`, `-1`, `1.5`, and `True`. |
| A:110 | `test_serve_setup_still_rejects_invalid_claimed_frame` | 4 | `-1`, out-of-range `4`, `1.5`, and `True`. |
| A:116 | `test_serve_setup_still_rejects_invalid_threshold` | 3 | Negative, NaN, and infinity. |
| A:122 | `test_serve_setup_still_rejects_invalid_slots` | 6 | Empty tuple, duplicate, non-`Slot`, mixed type, list, and unhashable member. |
| A:139 | `test_serve_setup_inputs_validate_rejects_bad_dtypes_and_counts` | 7 | Boolean/NaN/infinite/negative/fractional counts, integer `analysed`, integer height. |
| A:148 | `test_serve_setup_inputs_validate_rejects_wrong_shapes_and_lengths` | 1 item, 2 checks | One-dimensional wrist distances and mismatched height length. |
| B:162 | `test_builder_rejects_bad_resolution` | 4 | Zero component, NaN component, one-element tuple, and list. |
| B:172 | `test_dispatch_validates_options_cross_fields` | 1 item, 6 checks | Missing setup, legacy `dist`, missing lookback, missing stillness window, negative stillness window, and negative threshold. |

Reachability of every (c) family:

- `series_drift` malformed shapes/dtype: synthetic only on the normal path. Its only production call is `serve_setup_still` at `serve.py:121`; the builder creates floating `(t, 2)` ankle arrays and validates them at `serve.py:146-166`.

- `serve_setup_still` invalid window, frame, threshold, and slots: synthetic direct misuse. `_sticky_serve_setup_before` receives burst indices from span detection and passes literal slot tuples at `serve.py:287-288` and `serve.py:316`. `_resolve_serve_gate` validates the window and thresholds at `serve.py:369-386`.

- `ServeSetupInputs` bad count/dtypes/shapes: the concrete test states are not emitted by the normal sticky producer. Sticky evidence initialises fixed shapes and dtypes at `evidence.py:153-163`; count is an integer array and receives an integer analysis count at `evidence.py:155` and `evidence.py:172`. The builder casts or allocates the other fields and validates the result at `serve.py:143-166`. A malformed `StickyResult` or direct `ServeSetupInputs` can still reach this production boundary and be rejected.

- Bad builder resolution: partly production-reachable. `run_video.build_serve_options` passes `resolution` to the builder at `run_video.py:62`, and `_run_court_segmentation` supplies `court.resolution` at `run_video.py:399-403`. `_validate_run_inputs` only checks that resolution is present at `run_video.py:216-230`. A public caller can therefore supply zero, NaN, wrong length, or a list at runtime. A one-element tuple may fail earlier when sticky distances unpack `width, height` at `evidence.py:208`.

- Cross-field dispatch cases: setup missing, legacy `dist`, and missing windows are synthetic direct `ServeStartOptions` misuse because `build_serve_options` always supplies `dist=None`, setup, and both FPS-derived windows at `run_video.py:59-65`. A negative threshold can flow from a malformed `ServeStartConfig`, whose fields are only typed at `types.py:69-75`; `_resolve_serve_gate` then rejects it.

**Production references**

- Sticky evidence construction: `evidence.py:153-195` and `evidence.py:249-278`.
- Serve builder: `serve.py:133-167`.
- Production builder call: `run_video.py:48-66` and `run_video.py:399-403`.
- Gate resolution: `spans.py:276-299`.
- Gate execution: `serve.py:337-345`, `_sticky_serve_setup_before` at `serve.py:247-317`.
- Ratio helper: `serve.py:232-244`, called from `_sticky_serve_setup_before` at `serve.py:282-296`.

Serena reference scans found no additional production callers for the scoped symbols beyond these paths and the facade exports.

**Test references**

`git log --oneline --follow`:

- A: `51b2977`, `cc02b62`, `305b3ad`.
- B: `db33b85`, `51b2977`, `0f2c090`, `cc02b62`, `82689a5`.

Introducing commits:

- `305b3ad` introduced A and changed missing-player stillness from fail-open to fail-closed.
- `82689a5` introduced B for sticky-sourced evidence and three-lane routing.
- `db33b85` moved the implementation into `rally/serve.py` and retargeted B’s private import.

Bug-fix evidence:

- `305b3ad` says an unmeasurable player could previously be dropped, allowing the other player to pass.
- `cc02b62` changed both tests and implementation after identifying double normalisation and wrong-player distance/height pairing. B:56 directly pins that fix.
- The history records the bug-fix intent but not an explicit failing-test transcript.

Checks run:

- Combined collection command: exit 0, 62 tests collected.
- Per-file collection: exit 0, A 38; B 24.
- Git history commands: exit 0.
- No runtime test, lint, or type check was run.

**Counterevidence**

- The resolution validator is not purely defensive unit coverage: malformed `run_video` inputs can reach it because runtime resolution validation is incomplete.
- `ServeSetupInputs.validate` is production-called by both the builder and `_resolve_serve_gate`, even though the specific malformed arrays in A are synthetic.
- NaN, missing-player, and analysed-coverage states in group (b) are production-shaped. Sticky evidence initialises and writes those sentinels in `evidence.py:153-172`.

**Unresolved/dynamic surfaces**

- Public callers can bypass `run_video.build_serve_options` and construct `ServeStartOptions` or `ServeSetupInputs` directly.
- Runtime annotations do not enforce resolution, config, or custom `FpsConstants` shapes and values.
- Dynamic imports outside repository text/Serena references were not audited.
