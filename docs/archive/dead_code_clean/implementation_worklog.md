# Dead-code audit implementation worklog

## Resume

- **Current state:** B1 through B6 are merged. B7 gates, final independent
  review, and documentation archive are complete on `cleanup-dedup-b7` from
  merge commit `182b1f1`.
- **Final handoff:** Review this archive diff, commit it on
  `cleanup-dedup-b7`, and merge it through a pull request.
- **Verified so far:** B1 commit `f589f55` is contained in merged `origin/main`
  through PR #41 (`a4eddca`). B2 commit `03f8b26` is contained in merged PR #42
  (`ef77e71`). B3 and its corrections are contained in merged PR #43
  (`4a52929`). B4 commit `a8fc6b9` is contained in merged PR #44 (`9f21cc3`).
  B5 commit `51b2977` is contained in merged PR #45 (`4b13ea4`).
  B6 commit `dc50e8b` is contained in merged PR #46 (`182b1f1`).
- **Runbook:** `docs/archive/dead_code_clean/decisions.md`, rulings R0 through R9.

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
- **B1 follow-up:** Corrected the shared setup path, single-clip command, CLI
  flag descriptions, and BRIC layout. The active instructions were rechecked.
- **B2:** The available venv does not contain `matplotlib`, although it is
  declared in both root and BST-X dependencies. Plotting execution needs an
  environment with the declared evaluation dependencies installed.
- **B2 follow-up:** Corrected incomplete BST-X `PYTHONPATH` commands and
  self-bootstrap roots, a notebook setup cell missing `src`, and one stale
  player-mapping recipe. The active-path search is clean outside historical
  pre-phase documents.
- **B3:** No deployed BRIC checkpoint file is present under `runtime/` or
  `training/`. The manifest contract and 14-output head can be verified here,
  but loading `best.pt` needs the external checkpoint.
- **B3 follow-up:** Validation helpers use the frame-zero-clamped clip-bound
  function. Active taxonomy commands and imports use the current API. Validation
  examples distinguish the current Phase-2 sticky-anchor input, the legacy
  Phase-1 merged-25 input, and the UNE v1.14 collated training data. BST-X
  commands use a six-entry taxonomy registry.
- **B3 path evidence:** The local workstation does not mount `/scratch/comp320a`.
  Tracked 29 April output records the current Phase-2 sticky-anchor input, and
  tracked 21 April output records the legacy merged-25 baseline.
- **B1-B3 provenance review:** Corrected stale taxonomy imports and ownership,
  validation paths, unknown handling, plotting descriptions, missing-curation
  behavior, and TrackNet large-video wording. Restored the original court and
  player-mapping comments at their corresponding statements.
- **B4:** New downloads use `{id}.mp4`. Resume checks, clip generation, and
  resolution metadata also accept legacy `{id} {title}.mp4` files. A directory
  containing both names for one ID raises an error.
- **B4 gate:** `pipeline.build_dataset --help` remains blocked by the missing
  declared `moviepy` dependency. The new adapter and metadata CLIs load.
- **B4 review:** Added rename collision preflight, preserved existing resolution
  metadata when every video is unreadable, and placed the package-root setup
  before Stage 1 commands.
- **B5 baseline:** The focused annotator and test suite reached 391 passes and
  3 skips. `tests/test_environment.py` failed because this venv lacks the
  declared `matplotlib` dependency; R6 approves deleting that environment-only
  test. CourtKeyNet wrapper collection is blocked by missing `safetensors`.
- **B5 gate:** The complete focused B5 suite passes with 384 tests and 3 skips.
  Full collection finds 1,087 tests, then stops on nine missing-dependency
  imports. The missing packages are `safetensors`, `positional_encodings`,
  `tensorboard`, and `torcheval`.
- **B7:** GPU validation ran on Bourbaki's A100 in an isolated Python 3.12
  environment. The remote worktree remained clean.
- **B7 observation:** TrackNet's inpaint sidecar fails on a six-frame clip with
  a mask-length error. The relevant source blobs are identical to the
  pre-cleanup BST-X tree. An 84-frame clip completed normally.

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

`src/shared/` contains the cross-pipeline court implementation and TrackNetV3.
`src/classifier_shared/` contains classifier taxonomy, dataset helpers, player
mapping, evaluation plotting, and video metadata. The old BST court and
player-mapping modules are absent. The test-only temporal module and unused
frame/thumbnail video helpers are absent.

### Downloaders

`src/scraper/download_scraped_videos.py` owns yt-dlp execution. The ShuttleSet
adapter enables its H.264 video-only mode. Resolution scanning and its missing-
video report live in `src/bst_x/pipeline/video_metadata.py`. The old BST-X
downloader is absent.

### Annotator and CourtKeyNet

The approved R5 wrappers, fixed-25-fps aliases, and reported-only helpers are
absent. Tests use the retained batch, iterator, evidence, scorer, selector,
landing, and fps-resolved surfaces directly. The protected R5 mirrors remain.

### BST-X and BRIC

Raw extraction uses the pipeline clip index. Fixed-five sweep readers use the
shared reducers after checking completeness. BRIC retains the YOLO weights path,
while the dead tracker chain is absent. Evaluation uses the training device
selector.

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
   smoke, TrackNet command smokes, and the adversarial review.

R4 and R9 require no implementation changes.

## Readiness and execution log

### B0 revalidation

- **Files:** Read-only review of `docs/archive/dead_code_clean/`, `src/`, `tests/`,
  `pyproject.toml`, and active TrackNet documentation.
- **Change:** Created this current execution record and removed retired API
  work from the planned scope.
- **Gate:** Green. Worktree starts at `a555159`; Git was clean before this
  worklog. TrackNet baseline: 62 passed in 0.53 seconds.
- **Commit:** First committed in `f589f55 Consolidate TrackNetV3 under shared`.

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
- **Gate:** Green for B1 scope. Focused suite: 64 passed in 2.19
  seconds. Ruff: passed.
  Pyrefly on touched files: 0 errors. `predict.py --help`,
  `batch_predict.py --help`, and the pipeline CLI help passed. Temporary-index
  staged diff check passed, with no weight or model files present. Broader
  namespace/integration collection was blocked by a missing
  `positional_encodings` dependency in the available venv. The active
  documentation paths were rechecked after correction.
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
- **Gate:** Green for the B2 scope. Focused suite: 248 passed, 6
  skipped. Post-fix focused suite: 201 passed, 6 skipped. Ruff: passed.
  Focused Pyrefly: 0 errors. Raw extraction, the equivalence failsafe, BST
  preparation, training augmentations, and the notebook setup resolve with
  both package roots. Plotting execution is blocked in the available venv
  because its declared `matplotlib` dependency is not installed; source
  compilation and static checks pass.
- **Commit:** `03f8b26 Consolidate shared classifier foundations`; merged by
  PR #42 as `ef77e71`.

### B3 taxonomy and dataset

- **Readiness:** R2 and every surviving D2 taxonomy/dataset consumer were
  rechecked. The deployed BRIC manifest pins the legacy key, exact ordered
  14-class trainable list, and 14-output head.
- **Files:** New `classifier_shared/taxonomy.py` and `dataset.py`; moved split
  CSV; removed `shared/taxonomy.py` and `dataset.py`; trimmed
  `pipeline/config.py` and `clip_generator.py`; retargeted BRIC, BST-X,
  scripts, validation commands, tests, and active runbooks.
- **Change:** The six BST-X taxonomies are unchanged. Retained BRIC keys keep
  their old names and ordered full/trainable lists. Every merge map now sends
  `driven_flight` to `drive`. Flaw parsing requires an explicit path. Both
  classifiers use the BST-X clip-bound implementation with the frame-zero
  clamp. The unused computed `SPLITS_V2` export is gone.
- **Gate:** Green for the available B3 scope. Post-review focused suite: 95
  passed, 13 skipped. Ruff:
  passed. Focused Pyrefly: 0
  errors. Source compilation passed. Direct comparisons against the B2 branch
  confirm all six BST-X taxonomy contracts and both retained BRIC ordered class
  lists are unchanged. The deployed manifest matches the 14-class trainable
  view. Clean-shell help checks passed for the retargeted fail-rate and busted-
  clip validation commands. The inference smoke reaches its model import, then
  stops on the missing `positional_encodings` dependency. Broader BST-X
  collection is blocked by missing `torcheval` and `positional_encodings`.
  The deployed checkpoint is not available locally, so its state dictionary
  has not been loaded. Both validation helpers now call the clip-bound function,
  with early-frame tests, and active documentation uses the current taxonomy
  API and CLI. Follow-up checks: 67 focused tests
  passed, whole-project Ruff passed, and focused Pyrefly reported zero errors.
  The registry assertion passed, and four affected CLI help commands showed
  only the six BST-X choices. Two additional help commands remain blocked by
  the documented missing `moviepy` and `matplotlib` dependencies. Active
  package descriptions state the current module ownership and behavior. The
  notebook JSON and setup cell passed from the documented working directory.
  Active-document sweeps also corrected the BST-X overview and both X3D plan
  copies. `tests/test_taxonomy.py` remains blocked at collection by the missing
  `positional_encodings` dependency; direct registry assertions passed. The
  final path sweep separated current Phase-2 pose input, legacy Phase-1 pose
  input, and current UNE v1.14 collated training output. The same 67 focused
  tests, Ruff, focused Pyrefly, source compilation, and path-provenance checks
  passed after that correction.
- **Commit:** `6014b21 Consolidate classifier taxonomy and dataset helpers` and
  `72f1438 Correct B3 paths and preserve source documentation`; merged by PR
  #43 as `4a52929`.

### B4 downloader

- **Readiness:** R3 and the D1 compatibility report governed the scope. The
  existing downloader, both filename forms, resolution report, clip lookup,
  raw-video guard, config constants, and active runbooks were mapped.
- **Files:** Added `pipeline/download_adapter.py` and `video_metadata.py`;
  removed `pipeline/download_videos.py`; updated the scraper downloader,
  pipeline consumers, retained video rename tool, tests, and active runbooks.
- **Change:** ShuttleSet match rows now enter the scraper-owned downloader
  through a fixed-schema adapter and explicit video-only mode. Metadata and
  clip readers accept new and legacy filenames and reject duplicate ID matches.
  The raw-video guard ignores `sources.toml`.
- **Gate:** Focused downloader and scraper suite: 222 passed. Whole-project
  Ruff passed. Focused Pyrefly reported zero errors. The adapter, metadata, and
  scraper downloader CLI help commands passed. `pipeline.build_dataset --help`
  stopped at the documented missing `moviepy` dependency. The independent B4
  review found three defects; all three corrections passed their targeted checks.
- **Commit:** `a8fc6b9 Consolidate ShuttleSet video downloading`; merged by PR
  #44 as `9f21cc3`.

### B5 annotator and tests

- **Readiness:** R5 and R6 govern the scope. The listed compatibility wrappers,
  aliases, calibration helpers, and direct test consumers were rechecked on
  merged main. No live production callers remain for the approved deletions.
- **Files:** Planned changes are limited to the R5 annotator and CourtKeyNet
  targets, their direct tests, the R6 fixtures and raw-schema assertions, and
  this worklog.
- **Change:** Removed the approved compatibility wrappers, stale aliases, and
  reported-only helpers. Consolidated rally-region geometry, landing verdict
  geometry, and repeated `run_video` span options. Moved the scoring floor into
  its test, shared serve-setup and doubles-CSV builders through conftest, removed
  repeated int8 assertions, and deleted the environment-only import test.
- **Gate:** Focused suite: 384 passed, 3 skipped. Whole-project Ruff and
  `git diff --check` pass. Focused Pyrefly reports zero errors. Project-wide
  Pyrefly reports the same 17 errors in untouched BST-X shape annotations. Full
  collection finds 1,087 tests before nine dependency-related collection
  errors. CourtKeyNet execution remains blocked by missing `safetensors`. The
  independent review found two low documentation issues; both are corrected.
- **Commit:** `51b2977 Remove dead annotator and test helpers`; merged by PR #45
  as `4b13ea4`.

### B6 BST-X and BRIC

- **Readiness:** R7 and the remaining BRIC-only R8 items govern the scope. The
  focused baseline passed 106 tests. The player-tracking chain has no live
  caller, while `DEFAULT_YOLO_WEIGHTS` remains in use by BRIC preprocessing.
- **Files:** `preparing_data/raw_extract.py`, `pipeline/clip_index.py` through its
  existing public helper, `hparam_sweep.py`, `bric/perception/players.py`,
  `bric/eval.py`, the focused sweep test, and this worklog.
- **Change:** Raw extraction now uses `build_clip_path_index`. Fixed-five run
  readers use the shared metric reducers and retain their completeness check.
  Removed the dead BRIC tracker chain. BRIC evaluation now uses the training
  device selector.
- **Gate:** Focused suite: 108 passed. Whole-project Ruff and `git diff --check`
  pass. Focused Pyrefly reports zero errors. Import checks confirm the retained
  weights path and canonical clip-index function. The BRIC evaluation CLI is
  blocked by the previously documented missing `torcheval` dependency. The
  independent review found no issues and matched the baseline clip index,
  sweep arithmetic, completeness errors, and device selection.
- **Commit:** `dc50e8b Deduplicate BST-X helpers and trim BRIC`; merged by PR
  #46 as `182b1f1`.

### B7 final gates

- **Readiness:** B1 through B6 are merged. GitHub CI and PR-quality checks pass
  on the B6 merge.
- **Files:** The completed `docs/archive/dead_code_clean/` record set only.
- **Change:** No production change. Record the final checks and archive the
  completed cleanup documents.
- **Gate:** Whole-project Ruff passes. Project Pyrefly reports the same 17
  existing BST-X jaxtyping shape-name errors. Full local pytest stops at the
  same nine missing-dependency collection errors. The complete suite in the
  isolated Bourbaki environment passes 1,302 tests with 26 skips. The focused
  TrackNet, BRIC, and shared-helper suite passes 98 tests. TrackNet command
  smokes pass outside the repository and both inference entry points expose
  `--large_video`. The BRIC shared-module probe reports the 14-class deployed
  taxonomy contract.
  On Bourbaki's A100, BRIC passed tensor, YOLO11, and R(2+1)D-18 forward
  passes. BST-X validation passed on CUDA. Two seeded CUDA training runs
  produced the same checkpoint hash and metrics. TrackNet `--large_video`
  inference with the official checkpoints produced an 84-row contiguous CSV
  and its inpaint sidecar from an 84-frame 1080p clip. The final independent
  review found no issues across R1 through R9, protected contracts, active
  paths, documentation, or the B7 record.
- **Follow-up gates:** The released deployed BRIC checkpoint loads strictly on
  CUDA with no missing or unexpected keys. Its classifier is `(14, 576)`, its
  manifest matches the 14-class taxonomy, and a CUDA forward returns `(1, 14)`
  logits. The real Bourbaki data checks pass 15 BST-X preflight cases and one
  end-to-end data-loader/model-forward case. The RTMLib provider guard passes
  all three cases in the separate pose environment.
- **Remaining external checks:** Nine annotator calibration cases require a
  complete external fixture bundle. The available shared-scratch copy lacks
  the three pinned `kp_scores.npy` arrays; the other 41 tests in those modules
  pass. The retired Docker namespace case remains skipped because its web-demo
  condition no longer exists. The opt-in legacy venv-name scan finds old
  `venv-bst` wording in architecture notes and its own test pattern. Neither
  item is a GPU check or a production cleanup gate.
- **Archive:** Moved all 16 cleanup records to
  `docs/archive/dead_code_clean/`, preserving the directory structure. Updated
  25 internal absolute path prefixes to the archive location and added the
  archive README. No audit prose was rewritten.
- **Commit:** Not yet committed.
