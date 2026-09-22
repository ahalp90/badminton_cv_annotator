# Wider court evaluation

## Resume

The 44 fresh cases are dispatched on Carmack with six workers: 20 added
detector views and 24 separate controls. The original 27 W5 records are reused.
Seven fresh cases are complete and reviewed locally; no failure records have
appeared. All 27 previous cases have been re-reviewed with thin overlays and
far-end crops. Next: finish the remaining fresh cases and their visual review,
then combine the comparison and report the groups separately.

Code: `dc7872d`, committed and pushed. Remote checkout:
`/scratch/ahalperi/court_det_fix/wider_eval_checkout_20260922`.
Investigation root beneath it: `scratch/court_det_fix/`.
Run: `wider_evaluation/runs/20260922/measured/`.
Dispatch PID: `2521822`; `logs/full.log` and `receipts/full.*` record progress.
The launcher uses `~/.venvs/venv-rtmlib/bin/python` with six workers and one
native thread per worker. Do not launch overlapping retries for the same case.

Compare full-source W5 `(4,3)` with G1 plus line templates from the same measured
candidate pool. Preserve the 27 previous cases as regression controls. Report
all 47 frozen cases and the separate 24 broadcast controls in their own groups.

## Concerns and observations

- The 20 added detector cases and 24 controls need new G0/G1 proposal records.
  Each control has one real RTMDet sample. Two samples contain no detections;
  those are measured empty frames. Control boxes use the original >0.3 cutoff.
- Cached broadcast composites include unverified views. Keep their outcomes
  visible and qualify reference-based geometry scores.
- Existing runner helpers contain historical case lists and paths. Check those
  before treating the wider run as a routine replay.

## Review standard

The user clarified the standard on 22 September: imperceptible misalignment is
the ideal, assessed visually. Lines should ideally hug the outer edges of the
white court markings; that is a preference, not a minimum deployment threshold.
Review full images and far-end crops as clean, tolerable fallback or unacceptable.
Record the affected court region. Numerical errors support visual judgement;
this run does not impose a new pixel cutoff or acceptable fallback percentage.
Borderline visual cases remain open for user review.

## Scope and runbook

1. Inventory the frozen packs, images, source records and player observations.
   Freeze case membership, previous use and settings in a compressed manifest.
2. Adapt only the evaluation runner where needed. Reuse the existing scoring,
   refinement, candidate identity and player rule. Check retained measurement
   equality before omitting unused diagnostics or repeated conversions.
3. Smoke-test three representative inputs. Preserve raw frames, source identity,
   all candidate evidence, both gated and ungated winners and observation counts.
4. Run the wider comparison with at most six workers. Inspect selected overlays
   and far-end crops. Count empty pools, unavailable gates and rejected useful
   proposals separately.
5. Obtain one bounded Claude Code Opus correctness review. Verify material leads
   against source or executable checks. File results and update pickup.

Production integration, CourtKeyNet removal, new search rules, SVD optimisation
and scene-level implementation are outside this evaluation batch. Existing
evidence remains intact. The user authorised commits and pushes for code sync,
rsync of large packages, and sharing project information with Anthropic for
red-team review. The Claude review has no elapsed-time limit.

## Module state

- `w5_holistic/`: existing 27-case runner and frozen source loaders; unchanged.
- `wider_evaluation/measurement.py`: per-worker, single-image greyscale reuse;
  unused junction diagnostics omitted inside a restoring context manager.
- `wider_evaluation/smoke.py`: original versus prepared measurement equality
  on SS03-16, GX5 and SS21-10. All retained evidence and arrays match exactly.
- `wider_evaluation/freeze_inputs.py`: records 47 detector and 24 control cases,
  previous W5 membership, raw image hashes and available observation counts.
- `wider_evaluation/run_cases.py`: six-worker adapter for added frozen views;
  validated and running on the 44 fresh cases.

## Execution log

- Preparation: read pickup, findings, previous evaluation and team rules.
  Working tree was clean on `fix/court-det`. Serena tools are visible.
- Two Luna Max readers are mapping the runner and local input inventory.
  Their results are leads requiring direct verification.
- Confirmed Carmack access after retrying the sandbox-blocked connection with
  network permission. Existing remote jobs were idle; no host was changed.
- Verified the 20 missing frozen paths. Copied 14 images from existing local
  exports with byte equality and dimension checks. Decoded six short-clip
  images from `evidence/independent_proposals/development/examples_updated/`.
  Each clip's existing anchor matched decoded pixels exactly, including the
  letterboxed clip's native 960×720 dimensions. All new PNG round trips match.
- Three-case prepared-measurement smoke: exit 0, exact retained evidence and
  array equality. Greyscale conversions fell from 30–36 sampling calls to one
  per view. This selected-candidate check is not a deployment speed estimate.
- Initial helper Ruff: exit 0 after import formatting. Whole-project Pyrefly:
  exit 0, zero errors (39 suppressed). Later adapter edits need final checks.
- Canonical control labels come from `recorded/controls.json.gz`: eight
  non-court and 16 unlabelled. The old export manifest's all-unlabelled values
  are not used. The frozen inventory has 71 cases and no missing raw images.
- Verified all 24 control images against their DeepLSD cache MD5s and person
  dimensions. Prepared bottom-centre feet without changing box order or scores.
- Parent verified native paint-filter equality and automatic direction equality
  on GX0; a new 960×720 letterboxed view also prepared successfully.
- Comparator exactly reproduces 54 prior gated winners and eligible counts:
  full-source and G1-plus-template access across all 27 regression cases.
- Existing runner tests: 8 passed, exit 0. New adapter boundary tests: 2 passed,
  exit 0. Ruff and whole-project Pyrefly passed (0 errors, 39 suppressed).
- Initial Claude launch was rejected by automatic approval review. After the
  user's explicit payload authorisation, Opus launched successfully without a
  timeout. Review record: `local_scratch/external_delegate/20260922-wider-runner-audit/`.
- The old Carmack checkout contains changes from earlier runs. Use a fresh
  worktree for the synced commit; preserve the old checkout and its data.
- Snapshot `abbe46b` was committed and pushed. Fresh remote worktree:
  `/scratch/ahalperi/court_det_fix/wider_eval_checkout_20260922`.
  All 71 raw image hashes match the manifest there. The documented court
  environment is `~/.venvs/venv-rtmlib/bin/python` (NumPy 2.4.6, SciPy 1.17.1);
  the general pipeline and torch environments lack SciPy. Remote imports and
  input preparation pass in the court environment.
- Claude Opus 5 completed one independent review with no verified defect in
  its coverage. Parent verified the model ID, replay equality, source filtering,
  and control preparation. Small pre-dispatch improvements add strict result
  serialisation, atomic population writes, explicit omitted-junction markers,
  and a diagnostic for restricted-pool ranking fallback differences.
- Parent checked the control comparability caveat. Both broadcast preparation
  and controls use box bottom-centres. All 20 broadcast line arrays equal the
  default DeepLSD cache exactly; `line_population: original` is not a different
  detector. Sample counts, score cutoffs and selected detections still differ,
  so the controls remain a separate arm.
- Four control feet fall outside the image. They now remain unavailable, as
  in the frozen broadcast preparation; their boxes/scores remain unchanged.
  Parent verified all prepared feet and all four exclusions directly.
- Pre-dispatch correction `dc7872d` was committed, pushed and checked out in
  the fresh remote worktree. The three-case measurement equality smoke,
  two adapter boundary tests, Ruff and whole-project Pyrefly passed again.
  Existing eight runner tests remain valid. The 44-case dispatch started with
  ordinary broadcast, letterboxed, empty-person, close-up and yellow cases.

## Control input review

The raw contact sheet is `wider_evaluation/runs/20260922/control_raw_contact.png`.
The original eight non-court labels and 16 unlabelled statuses are preserved.
Among the unlabelled raw images, visual inspection distinguishes:

- Wide venue/multiple courts: frames 0 and 1.
- Ordinary full-court broadcast view: 19115, 33450, 43006, 76455 and 86012.
- Tighter elevated court views: 47785, 52563 and 81233.
- Sideline close-ups or partial-court views: 4779, 28671, 57342, 66898,
  71677 and 95569.

These describe input framing, not detector success or new ground-truth labels.

## Initial completed-case review

Seven fresh cases have completed without recorded failures. Their comparisons
and thin-line overlays are local under `wider_evaluation/runs/20260922/`.
`visual_rulings.json.gz` records inspected images and provisional visual rulings.
The run is incomplete; these are case observations, not aggregate results.

- Control 300: the ungated winner overlays the exterior building. The player
  rule rejects every candidate, so both arms correctly return no court.
- Control 4779: both arms accept a badly misplaced court in a sideline close-up.
  Player support does not make this selection safe.
- Control 0: both arms select the foreground court, but extend its far/left end
  visibly beyond the painted baseline. The near/right end looks much better.
- Yellow 90: both arms agree. The near cross-court lines look plausible, but
  the far sidelines diverge and the backcourt is shortened. Unacceptable fit.
- Letterboxed 58: the arms choose slightly different, close fits. Small shifts
  remain at the far baseline/service line and centre line. Tolerable fallbacks;
  the near baseline lies outside the image.
- Centre 64 and SS03-1: both arms agree on plausible fits with visible far
  baseline/service offsets. Both are recorded as tolerable fallbacks.

The first two completed cases each use one greyscale conversion for 15,429 and
41,406 sampling calls respectively. This confirms reuse across a whole view;
it is not an overall runtime speed-up measurement.

## Regression visual review

The parent reviewed all 27 prior cases, including both arms where their selected
geometry differs. Provisional rulings for each arm are seven clean fits,
16 tolerable fallbacks, three unacceptable fits and one requiring user review.
These visual categories do not introduce a numerical deployment threshold.
Clean means the visible projection follows the paint closely; exact placement
on the paint's outer edge is not asserted.

The three unacceptable cases remain Yellow14, Am1-54 and SS21-10. SS21-39
requires review because its cached composite mixes camera views. The selected
geometry follows the dominant court, but cannot validate scene-level behaviour.
Many fallback cases show a much better near end than far backcourt. Reinspection
also confirms three GX ungated winners on the wall; the player rule replaces
them with plausible court fits.

The gallery helper initially assumed 960×540 for every case. Parent comparison
against saved corners exposed a 296-pixel mapping error for the letterboxed
view. The helper now preserves aspect ratio and checks every projection against
the saved native corners. All 38 selected geometries pass within 0.018 pixels;
this checks rendering fidelity, not detector accuracy. Missing compact-review
winners were recovered from their unchanged full records. Gallery and metadata:
`wider_evaluation/runs/20260922/baseline_gallery/`.

The renderer smoke completes all 27 cases with zero missing roles (exit 0).
Ruff and whole-project Pyrefly pass after the correction (exit 0 each;
zero type errors, 39 suppressed). Earlier runner tests remain applicable.
