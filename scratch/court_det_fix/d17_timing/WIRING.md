# How the D17 court detector is wired

The accepted court detector (the D17 chain) runs end to end in one place only:
`run_d17.py` in this folder, a timing script. This note says what that script
calls, which runtime swaps change the chosen court, and which inputs are
frozen. It also lists what a single detector function must keep and what it
can drop. Read it before joining the stages into one unit, or before
optimising across stage boundaries.

Written against branch `exp/court-det-opt` at commit 44254b42. Paths are
relative to `scratch/court_det_fix/` unless they start with `src/` or
`experiments/`.

## Terms

- **View**: one video frame to find a court in. The test set has 28, listed in
  `court_detector_optimisation_handover/claude_evidence/fresh_feet/run_feet_variants.sh`
- **Source record**: one view's input, a dict with the frame's size, its line
  fragments (`segments_px`, from DeepLSD), person boxes (`bbox_px`), player
  feet (`all_feet_px`), image path and provenance. All in native pixels
- **Context**: the `verifier.ViewContext` that `prepare_view` builds from a
  source record and its frame. It holds the working-size fragments, line
  families, grouped observations and the resized frame
- **G0, G1, line templates**: the three candidate sources. G0 searches all
  fragments. G1 searches only fragments that look like paint. Line templates
  build courts from rectangles of crossing lines
- **Parent, child**: a parent is a candidate court as proposed. Its child is
  the same court refitted to the painted stripes
- **W5**: the stage that measures every parent, refits it, and ranks parents
  and children together
- **Gated**: a candidate whose `historical.historical_fullcourt` flag is true,
  meaning it passed the full-court gates. Only gated candidates can be chosen
- **Runtime swap**: code that replaces a module's function while the program
  runs (monkey-patching). The chain depends on several

## The chain at a glance

One call per stage, in the order `run_d17.main` makes them. The stripe refit
runs only when the net choice returns a court. Times are wall-clock seconds,
summed over the 28 one-view processes of the Carmack run for commit 44254b42
(`run_wall_s` total 8,250 s; the rest is imports, runtime loading and the
summary).

| Stage | Call | Takes | Produces | Disk | Time |
| --- | --- | --- | --- | --- | ---: |
| Load runtime | `wider_evaluation/run_cases.load_runtime(ROOT, control pack)` | Nothing per view | `run_w5`, the `verifier` module, a runtime dict; registers every case (see "Runtime swaps") | Reads the view packs | Start-up |
| Prepare view | `verifier.prepare_view(ROOT, case_id)` | Source record and frame | Context | Reads the pack and frame | 27 s |
| Search | `wider_evaluation/generation.ensure_populations(ROOT, context, runtime, arm_dir, 16)` | Context | G0 and G1 population records, up to 256 courts each | Writes an input audit and both records; the script reads them back | 5,319 s |
| W5 | `w5_holistic/run_w5.process_case(ROOT, case_id, arm_dir, min_visible_lengthwise=4, min_visible_cross_court=3)` | Fresh populations and seeded templates, both through swaps | Case record: `parents`, `valid_children`, `fit_attempts`, `rankings` | Writes the record and an arrays `.npz` | 2,754 s |
| Net choice | `run_d17.net_rows(...)`, then `net_recovery/bounded_trial.choose(rows, 0.04, 4.0)` | Case record, read back, and the context's fragments and size | The chosen court's `origin_key` or None, and the scored rows | Reads the record | 47 s |
| Stripe refit | `net_recovery/refit_selected.refit_selection(case_id, str(record_path), chosen, label, verifier, runtime, manifest)` | The chosen `origin_key`, the record path and the manifest | A dict; the court is its `corrected` entry | Reads the record and frame again | 48 s |

Selection happens before the stripe refit, as in the accepted 20-case gallery.
Keep that order.

### Search

- **Directions.** `vp_pruning.estimate` finds the view's line directions. Its
  settings come from `frozen_views/baseline_directions/gxBQ_window_00_frame_0.json.gz`;
  only that file's `settings` entry is read
- **G1 fragments.** `line_identity/filter_replay.paint_masks(source, native frame, scale)["paint"]`
  marks paint-like fragments, and `filtered_source` keeps them.
  `generation._filter_module` loads the module by file path, unless a module
  named `filter_replay` with both functions is already imported. Then it uses
  that one. Pin the intended file when joining the stages
- **Search per population.** `w5_holistic/automatic_generation.generate` runs
  once for G0 and once for G1. It tries each ordered pair of directions. It
  skips pairs outside the direction budget and pairs that fail the camera
  bound, then keeps 256 courts per pair and 256 overall. Budget 16 means every
  pair is eligible, which is the current baseline
- **Pool evidence.** `run_automatic.evaluate_pool` then measures up to 256
  kept courts. It re-reads the frame when any survive

### W5 (`process_case`)

In order: load the runtime again, prepare the view again, load the three
populations, check them for reference fields, merge duplicates
(`canonicalise_populations`), build a record for every parent
(`make_parent_record`), and attempt a refit for each (`attempt_refit`). Only
hard-valid parents are measured and refitted; the rest are recorded as
invalid. Then it ranks. The ranking the detector uses is `rankings["C"]`,
over hard-valid parents and valid children together. The rest of
`process_case` is research: see "What a single detector keeps and drops".

### Net choice

`net_rows` walks `rankings["C"]["provisional_rank"]` and keeps gated
candidates only. For each, `am1_net_selection_trial.project_pieces` projects
the net. When that works (state `measured`), `bounded_trial.post_features`
tests pieces 2 and 3, the two posts, against the fragments.

`choose` scores each row as its evidence under `rankings["C"]["r2_criterion"]`,
plus 0.04 × the post reward. `net_rows` stores that evidence as
`paint_score`, but the criterion is not always paint. It is normally
`q_paint10_span_weighted`, and falls back to `q_geom_span_weighted` when no
camera-eligible candidate has a paint score (`verifier.rank_candidates`). A
post counts when a fragment covers one of its six lowest samples, and no
covering fragment reaches more than 4 working pixels below its base. The
reward is 0, 0.5 or 1 for zero, one or two posts. The first row in rank order
wins exact ties. No gated rows means no court: the view abstains.

`net_rows` copies how `bounded_trial.measure_case` builds rows. It leaves out
that function's checks against frozen results and the fields `choose` never
reads. The script also checks that weight 0 picks the top gated court.

### Stripe refit

`refit_selection` reads the case record from disk and prepares the view a
third time. It checks the frame's path and MD5 against
`wider_evaluation/runs/20260922/manifest.json.gz`. It then finds the chosen
court's parent and saved fit attempt, rebuilds the fit constraints
(`fixed_stripe_refit.prepare`) and replays the W5 fit. The replay must match
the saved fit within 1e-4 native pixels.

Next it corrects which stripe edge each fit fragment sits on. For each fit
fragment:

- `edge_polarity/run_probe.relabel` measures masked grey profiles on either
  side of it, on the working frame
- `observed_colour.sample_fragment` samples colour along it on the native
  frame, re-read from disk. `edge_auto_trial.infer_polarity` turns that into
  a polarity, or 0 without 8 usable samples and an 80% majority
- `edge_auto_trial.automatic_position` combines the old position, the grey
  contrast and pair count, the expected bright side and the polarity. It
  keeps the old position unless the polarity is known, there are at least 7
  side pairs and the contrast is at least 10 grey levels

Only the constraint positions change. It then refits
(`fixed_stripe_refit.refine`), and `fit_geometry` validates the result.
The output's `corrected` entry holds `corners_native_px`,
`homography_working`, `valid` and `validity_reason`.

## Runtime swaps

These swaps and table changes carry the detector's behaviour or inputs. The
last column says what dropping each one does. The first three change the
chosen court with no error.

| Swap | Made in | Active | Without it |
| --- | --- | --- | --- |
| `run_w5.load_g0` and `load_g1` return the fresh populations | `run_d17.py` | For `process_case` | W5 scores old saved pools: `frozen_views/baseline_generation/` for G0, a 15 September remote record for G1 |
| `run_w5.import_runtime` adds `seeded_line_templates` as `runtime["line_template"]` | `run_d17.py` | Whenever W5 loads its runtime | Templates are built without the three seeds, which loses the Am1 recovery |
| `vp_pruning.estimate` adds three seed points: where each pair of the three longest lengthwise lines meets | `colour_consistency/am1_recovery_trial.generate_seeded` | Only while templates are built | No seeded templates. The search's own directions never see the seeds |
| `verifier.PACK_OF` and `verifier.CASE_LABELS` gain every case in the three packs and the control pack | `run_cases.load_runtime` | Always, in this script | 15 of the 28 views are missing from the verifier's own tables. Six pack views fail to load; the nine control views fail the label lookup |
| `verifier.load_source`, `load_case_provenance` and `frame_path` serve the control views | `run_cases.load_runtime` | Always, in this script | The 24 `sset_21_…` control views are not in the packs, so they fail |
| `verifier.load_source` replaces each view's `all_feet_px` from the `--feet` file | `run_d17.replace_feet` | With `--feet` | The pack's feet are used. The baseline uses the standing-only file |
| `run_automatic.frame_path` reads frames through `verifier.frame_path` | `generation.ensure_populations`, only during the search | Search | The search looks for frames in older folder layouts |
| `verifier.grayscale_sample` caches the greyscale frame; `verifier.raw_junctions` returns a stub | `wider_evaluation/measurement.prepared_measurements` | W5 | Same measurements, slower. Junctions reach only `candidate_review`, which is research |

These swaps are timing only and change nothing:

- `generation.generate` is wrapped to label the G0 and G1 stages, and to apply
  the smoke-test option `--max-matched-pairs`
- About 50 `clock.wrap` calls time individual functions

### Traps

- **Swaps made after the runtime loads can miss.** `run_w5.load_verifier`
  copies the verifier's functions into `runtime["verifier"]` when the runtime
  loads. A later swap reaches only code that looks the function up on the
  module, or that loads the runtime again. Today that is harmless:
  `prepare_view` looks `load_source` up on the module, and `process_case`
  reloads the runtime. In a single detector, pass inputs in rather than
  swapping functions
- **The seeded templates must patch the modules W5 uses.** `run_d17.py`
  asserts that `am1_recovery_trial` holds the same `vp_pruning` and
  `line_template_source` modules. Which copy imports depends on
  `run_w5.add_helper_paths`; the web UI's `SOURCE_MAP.md` and
  `court_detector_optimisation_handover/tools/resolve_hotpath.py` show which
  copy is live
- **One trial module is imported twice.** `run_d17.py` imports
  `am1_recovery_trial` by its bare name. `refit_selected.py` imports it as
  `scratch.court_det_fix.colour_consistency.am1_recovery_trial`, only for
  `resolve_saved_path`. Python treats these as two modules, and each runs its
  import-time code
- **Thread settings come first.** `run_d17.py` sets six thread variables
  (`OPENBLAS_`, `MKL_`, `OMP_`, `NUMEXPR_`, `VECLIB_MAXIMUM_` and
  `BLIS_NUM_THREADS`) to 1 before numpy and OpenCV load. It then calls
  `cv2.setNumThreads(1)`. Keep the same set-up for a bit-for-bit comparison
- **Imports have side effects.** `am1_recovery_trial`, `bounded_trial` and
  `refit_selected` edit `sys.path` when imported, and `am1_recovery_trial`
  sets three of the thread variables again. `run_w5.import_runtime` imports
  `run_population` before `run_diagnosis` on purpose; its comment explains
  why

## Frozen inputs and their live replacements

| Input | Path | In git | Used by | Live replacement |
| --- | --- | --- | --- | --- |
| View packs (source records) | `frozen_views/packs/{gx_extension,marking_refit,broadcast_extension}_inputs.json.gz` | Yes | `prepare_view`. `experiments/annotator/independent_court/case_provenance.py` pins each pack's MD5 and the provenance file's, so any other pack fails loudly | A source record built from the video: DeepLSD fragments, person boxes, feet, frame size |
| Case provenance | `frozen_views/case_provenance.json.gz` | Yes | `prepare_view`, for views outside the control pack. Says which frames the image and person boxes come from, and so whether boxes can mask the photometry | Equivalent frame and box provenance for the live frame |
| Frames | `frozen_views/frames/{gx,original,amateur}/` | Yes | Prepare view, pool evidence, W5 and the stripe refit | The decoded frame |
| Control views | `wider_evaluation/runs/20260922/control_inputs.json.gz`; frames in `evidence/independent_proposals/development/inputs/controls/frames/` | Pack yes; frames **no** (git-ignored, copy by hand) | The control swap | As for view packs |
| Standing-only feet | The `--feet` file, `feet_standing.json.gz` (138 KB) | **No**. Use the baseline's copy on Carmack: `fresh_feet_20260924/feet_standing.json.gz` in the court-detector run root. Rebuilding it needs the videos and the pose model | The feet swap | See "Feet" below |
| Direction settings | `frozen_views/baseline_directions/gxBQ_window_00_frame_0.json.gz` | Yes | Search | Constants |
| Manifest | `wider_evaluation/runs/20260922/manifest.json.gz` | Yes | Refit's frame check | None; drop the check |
| Saved control candidates | `automatic_axes_20260914/all_camera/` | **No**, and not in this checkout | W5's known-control diagnostic, on 3 views | None; drop the diagnostic |

**Feet.** The baseline's feet come from the throwaway scripts in
`court_detector_optimisation_handover/claude_evidence/fresh_feet/`
(`extract_window_people.py`, `shot_check.py`, `build_feet_variants.py`):

- Detect people and poses in a 3 s window at 10 fps around the frame
- Keep only samples in the same shot as the frame
- Take each person box's bottom centre as a foot
- Drop seated people with `is_sitting` from
  `src/bst_x/preparing_data/heuristics/base.py`, threshold −0.3

Follow-up item 10 in `CLAUDE_FOLLOWUPS.md` covers this choice.

## What a single detector keeps and drops

**Keep**, as it changes the chosen court:

- The search on G0 and G1, with the budget-16 direction screen
- Seeded line templates, with the (4, 3) visibility floor
- W5's merge, measurement, refit and C ranking
- The net choice at weight 0.04 and overrun 4 px
- The stripe refit, after the choice, including `fixed_stripe_refit.prepare`
  on the chosen court's parent
- Every behaviour swap above, as a plain function argument or call

**Drop**, as it is research only:

- The search's input audit file (`inputs/<case>.json.gz`)
- Writing the population records and reading them back
- In `process_case`: the reference-field check, the B ranking, the
  permutation check, the known controls, the candidate reviews and the rank
  sensitivity
- Writing the case record and arrays, and reading the record back twice
- In the refit: the manifest and MD5 check, and the replay of the saved W5
  fit with its tolerance check
- The weight-0 check and the timers

`CLAUDE_FOLLOWUPS.md` estimates the research files at about 580 s over the 28
views (6.5%). Legacy pool evidence (its item 7) could go too. Gate evidence
comes from the same `evaluate_pool` call, so that change needs care.

**Repeated work.** A single detector can do each of these once:

- `prepare_view` runs three times per view: in the script, in
  `process_case`, and in the refit. Commit 7c74b015 already caches its
  `prepare_observations` step
- The frame is read from disk by each `prepare_view`, and again by the G1
  paint mask, each `evaluate_pool` call and the refit's colour sampling
- The runtime loads three times: twice in `run_cases.load_runtime`, once in
  `process_case`

## Decisions the code leaves open

- **The final court when the refit fails.** The script saves both the chosen
  court and the refit result, with no rule for which is the answer. On the 28
  views, 21 end with a valid corrected court. One,
  `sset_21_gloiZ_gTJaE_frame_00014336`, ends with an invalid corrected fit
  (`rank_deficient`). The script also catches refit errors; none occurred in
  this run
- **Abstention.** Six of the 28 views have no gated court, all `sset_21_…`
  control views. The detector should report "no court" for them
- **Where the unit lives**: `src/` or `experiments/`
- **Which frames feed the feet** for a new video

## How to check an integrated detector

Run the new detector and `run_d17.py` on the same 28 views with the same feet
file. For each view, compare `selection.bounded` and
`selection.polarity_refit.corrected` from the script's summary with the new
detector's output, bit for bit.

The baseline run lives on Carmack, in the court-detector run root:

- **Summaries**: `d17_camera_20260924/standing/budget16/d17/`, one
  `.json.gz` per view, 28 in all
- **Feet**: `fresh_feet_20260924/feet_standing.json.gz`. Pass the folder
  `fresh_feet_20260924` as the launcher's `FEET_DIR`
- **Launcher**: `court_detector_optimisation_handover/claude_evidence/fresh_feet/run_feet_variants.sh CHECKOUT OUT FEET_DIR PYTHON JOBS standing`.
  It writes each view's summary to `OUT/standing/budget16/d17/`
- **Git-ignored inputs**: nine views need the control frames, and three need
  `automatic_axes_20260914/` unless the known-control diagnostic is
  dropped. The baseline's checkout, `d17_camera_checkout_20260924`, has
  both under its `scratch/court_det_fix/`. Copy them into any new checkout

## Other documents

- `court_detector_optimisation_handover/SOURCE_MAP.md`: the web UI's map of
  hot-path files. Its "No single runner" line predates `run_d17.py`
- `court_detector_optimisation_handover/CLAUDE_FOLLOWUPS.md`: every speed-up
  tried, and the deployment-mode estimate
- `handover_20260923/01_LAUNCH_OPTIMISATION.md`, step 3: the recipe
  `run_d17.py` follows
