# Dead-code audit implementation worklog

## Resume

- **Next action:** Prepare and validate the B2 commit and PR handoff script.
- **Current batch:** B2 implementation, independent review, corrections, and
  local gates are complete on the continuing `cleanup-dedup` branch.
- **Verified so far:** B1 commit `f589f55` is contained in merged `origin/main`
  through PR #41 (`a4eddca`). B2's focused suite passes 161 tests with 6 skips.
  Ruff and focused Pyrefly pass.
- **Runbook:** `docs/dead_code_clean/decisions.md`, rulings R0 through R9.

## Concerns and observations

- **B0:** The audit predates the web-demo retirement. The API consumer in R1,
  API imports in R2, and every API-specific deletion in R8 are now obsolete.
- **B1:** The scrape lane is a documented hand-run consumer. Its stride 8 and
  `--large_video` contract must survive the TrackNet move.
- **B2/B3:** The deployed BRIC taxonomy key, ordered 14-class list, and
  checkpoint output shape are hard compatibility gates.
- **All batches:** Historical evidence may describe paths that were correct at
  the time. Update active commands and runbooks, while preserving historical
  claims where a path is part of the evidence.
- **B1:** The available local test environment lacks `positional_encodings` and
  `moviepy`. This blocks two broader collection/import checks, while the B1
  targeted tests and source-level gates pass.
- **B1:** Whole-project Pyrefly reports 17 existing jaxtyping shape-name errors
  in untouched BST-X files. Pyrefly reports zero errors on the B1 files.
- **B1 review:** A fresh Codex session found one medium documentation defect.
  The shared setup path, single-clip command, CLI flag descriptions, and BRIC
  layout still named deleted locations. All cited active instructions were
  corrected and rechecked. No runtime finding was reported.
- **B2:** The available venv does not contain `matplotlib`, although it is
  declared in both root and BST-X dependencies. Plotting execution needs an
  environment with the declared evaluation dependencies installed.
- **B2 review:** A fresh Codex session found three confirmed path defects:
  incomplete BST-X `PYTHONPATH` commands and self-bootstrap roots, a notebook
  setup cell missing `src`, and one stale active player-mapping recipe. All
  were corrected. A follow-up active-path search is clean outside explicitly
  historical pre-phase documents.

## Original and intended shape

| Area | Original shape | Intended shape |
| --- | --- | --- |
| TrackNetV3 | BST-X and BRIC each carry a vendor tree. | One authoritative tree at `src/shared/tracknetv3/`. |
| Shared classifier code | Classifier-only and cross-pipeline helpers are mixed under `src/shared/` and `src/bst_x/pipeline/`. | Cross-pipeline helpers remain in `src/shared/`; classifier-only helpers move to `src/classifier_shared/`. |
| Downloading | BST-X and scraper have separate yt-dlp implementations. | The scraper downloader is canonical, with a ShuttleSet adapter and separate metadata module. |
| Annotator and CourtKeyNet | Test-only wrappers and stale fixed-25fps aliases remain in production modules. | Tests use the live production surfaces and dead wrappers are removed. |
| BST-X and BRIC | A small set of verified helper implementations are duplicated or unreachable. | The approved R7 and BRIC R8 helpers are consolidated or removed. |
| API | The audit expected a live web API. | The web-demo PR removed it, so API-specific audit actions are dropped. |

## Module state

### TrackNetV3

One tree now lives at `src/shared/tracknetv3/`. It uses the authoritative BST-X
content and retains the required `--large_video` forwarding. BRIC runs its
`predict.py`; BST-X and the scrape profile run its `batch_predict.py`. The old
BST-X and BRIC trees are absent. No model weights were moved or changed.

### Shared and classifier helpers

`src/shared/` contains the cross-pipeline court implementation, dataset and
taxonomy code reserved for B3, and TrackNetV3. `src/classifier_shared/` now
contains player mapping, evaluation plotting, and video metadata. The old BST
court and player-mapping modules are absent. The test-only temporal module and
unused frame/thumbnail video helpers are absent.

### Downloaders

`src/bst_x/pipeline/download_videos.py` and
`src/scraper/download_scraped_videos.py` remain separate. R3 has not been
implemented.

### Annotator and CourtKeyNet

The R5 wrappers and stale constants remain. Their named live replacements also
remain, so the planned test retargets are still possible.

### BST-X and BRIC

Both R7 dedups remain. BRIC still has the dead player-tracking chain and the
duplicate `_select_device`. API-specific R8 targets are absent.

## Execution batches

1. **B0 revalidation:** Recheck R1 through R8 against merged main and remove
   obsolete API touch-points from the implementation scope.
2. **B1 TrackNetV3:** Move the BST-X tree to the shared location, retarget the
   BRIC and BST-X wrappers, update active commands and exclusions, and delete
   the old trees.
3. **B2 shared foundations:** Consolidate court, player mapping, evaluation
   plots, video I/O, and temporal helpers. Retarget all D2-listed consumers.
4. **B3 taxonomy and dataset:** Consolidate taxonomy, flaw parsing, and clip
   bounds. Preserve the deployed BRIC contract and verify checkpoint loading.
5. **B4 downloader:** Add the ShuttleSet adapter and video-only mode, move
   resolution metadata handling, and retire the BST-X downloader.
6. **B5 annotator and tests:** Apply R5 and R6 deletions and dedups, including
   the approved test fixture consolidation.
7. **B6 BST-X and BRIC:** Apply R7 and the remaining BRIC-only R8 changes.
8. **B7 final gates:** Run targeted tests, full pytest, Ruff, Pyrefly, BRIC
   smoke, TrackNet command smokes, and the approved adversarial review.

R4 and R9 require no implementation changes.

## Readiness and execution log

### B0 revalidation

- **Files:** Read-only review of `docs/dead_code_clean/`, `src/`, `tests/`,
  `pyproject.toml`, and active TrackNet documentation.
- **Change:** Created this current execution record and removed retired API
  work from the planned scope.
- **Gate:** Green. Worktree starts at `a555159`; Git was clean before this
  worklog. TrackNet baseline: 62 passed in 0.53 seconds.
- **Commit:** Not yet committed.

### B1 TrackNetV3

- **Readiness:** The authoritative source and consumers were mapped. The API
  consumer named by R1 is gone. Remaining consumers are BRIC's subprocess,
  BST-X's batch subprocess, and the documented scrape profile.
- **Files:** Moved the authoritative tree to `src/shared/tracknetv3/`; removed
  both old trees; updated the BRIC wrapper, BST-X pipeline defaults, tests,
  lint/type exclusions, dependency comments, and active TrackNet runbooks.
- **Change:** Both classifiers and the scrape profile now use one TrackNetV3
  tree. The canonical path is the CLI default. Local checkpoint files under
  its `ckpts/` directory are ignored to prevent accidental commits.
- **Gate:** Green for B1 scope. Post-review focused suite: 64 passed in 2.19
  seconds. Ruff: passed.
  Pyrefly on touched files: 0 errors. `predict.py --help`,
  `batch_predict.py --help`, and the pipeline CLI help passed. Temporary-index
  staged diff check passed, with no weight or model files present. Broader
  namespace/integration collection was blocked by a missing
  `positional_encodings` dependency in the available venv. Independent review:
  one confirmed medium documentation finding, fixed and rechecked.
- **Commit:** `f589f55 Consolidate TrackNetV3 under shared`; merged by PR #41
  as `a4eddca`.

### B2 shared foundations

- **Readiness:** R2 and the D2 consumer table governed the scope. Taxonomy,
  flaw parsing, and clip bounds remain reserved for B3.
- **Files:** `src/shared/court.py`, new `src/classifier_shared/`, all D2-listed
  court and player-mapping consumers, BRIC plotting and video consumers,
  operational runbooks, and focused tests.
- **Change:** Added BST's resolution-indexed court builder to the shared union
  surface and retired `pipeline.court_utils`. Consolidated player mapping in
  `classifier_shared` and retired both old copies. Moved the plot renderer and
  video metadata there. The presentation script is now a thin renderer CLI.
  Removed the unused video frame/thumbnail helpers and the test-only temporal
  module.
- **Gate:** Green for the B2 scope. Independent review suite: 248 passed, 6
  skipped. Post-fix focused suite: 201 passed, 6 skipped. Ruff: passed.
  Focused Pyrefly: 0 errors. Raw extraction, the equivalence failsafe, BST
  preparation, training augmentations, and the notebook setup resolve with
  both package roots. Plotting execution is blocked in the available venv
  because its declared `matplotlib` dependency is not installed; source
  compilation and static checks pass.
- **Commit:** Pending.
