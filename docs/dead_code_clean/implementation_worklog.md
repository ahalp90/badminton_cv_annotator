# Dead-code audit implementation worklog

## Resume

- **Next action:** Run the B1 commit script, push the branch, and open the first
  focused PR.
- **Current batch:** B1 implementation, local gates, and independent review are
  complete.
- **Verified so far:** The branch starts at merged `origin/main` commit
  `a555159`. `src/api/`, its API tests, and the web frontend are absent. Both
  TrackNetV3 trees and the remaining R2, R3, R5, R7, and BRIC R8 targets still
  exist. B1 now has one shared TrackNet tree. Its direct suite passes 63 tests,
  and the shuttle-array compatibility test also passes.
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

`src/shared/` still contains court, dataset, taxonomy, player mapping,
evaluation plots, temporal helpers, and video I/O. `src/classifier_shared/`
does not exist. No part of R2 has been implemented.

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
- **Commit:** Pending.
