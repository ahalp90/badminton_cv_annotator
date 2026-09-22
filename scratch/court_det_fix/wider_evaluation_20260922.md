# Wider court evaluation

## Resume

Preparation is active. The 71-case manifest and three-view measurement smoke
are complete. All 47 detector raw images are local. Next: finish the fresh
proposal adapter and launch the 20 added views on Carmack with six workers.
All 24 controls now have prepared line/player observations from existing caches.

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
  preparation and validation are in progress.

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
