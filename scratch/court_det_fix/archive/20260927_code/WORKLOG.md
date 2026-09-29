# Put the court detector's code together

## Outcome

Completed on 27 September 2026. The detector's implementation now lives in
court_detector/. Consumers and tests use that package; superseded source is
archived. Numerical comparisons matched. The document tidy and code move are
uncommitted, as requested. See pickup.md for subsequent work; this record is frozen.

## Concerns and observations

- Keep the measurements module shared with its temporary greyscale cache patch
- Keep one class identity for image/box source information and fitting constraints
- Keep both JSON round trips, stable tie order, float precision and readonly arrays
- Full video checks require videos on Carmack. Local checks cover saved court
  selection/refit, a bounded real search pair, scoring and template support
- Whole-repo lint already fails with 1,080 errors; baseline log is beside this file
- Whole-project Pyrefly passes before the move (0 errors, 39 suppressed)

## Plan and boundaries

Goal: the prepared-view detector loads its mathematical implementation from one
normal Python package, with no research sys.path injection or bare module copies.

1. Capture the accepted calculations twice and compare exact outputs. Snapshot
   source files and current dirty changes. Map direct, transitive and dynamic imports
2. Move coherent low-level geometry/observation/fitting modules into court_detector.
   Extract live functions from research runners using source-preserving AST selection.
   Use relative imports. Update the detector loader and frozen-view adapter
3. Update live consumers and tests. Move superseded scratch scripts into the existing
   archive with a note that their paths are historical; keep all data at current paths
4. Compare outputs against the pre-change capture, exercise package-only loading,
   migrate bounded matcher/camera checks, and run full lint/types/tests. Distinguish
   inherited failures from new ones. Obtain a bounded independent review
5. Update the existing guide, file map, speed-up code references and pickup. Preserve
   a final dated worklog in the archive. No new live state document

OUT: scoring/search defaults; maths and precision; GPU/Numba/parallel features;
large raw data or recovery files; src pipeline behaviour; commit/push/merge.

The review/tests must cover module identity, patched globals, caches, frozen input
paths, alias imports, numerical outputs and dependency isolation. No broad style
rewrite of migrated function bodies. Historical scripts may keep old internal paths.

## Module shape before the move

- The detector imported fifteen named research modules and further experiment
  helpers. It changed sys.path to select between duplicate bare module names
- Search already receives images/lines/feet in memory. It does not need the old
  filter_replay loader or population-file caching
- Scoring already has a pure in-memory entry. Its old driver also contains CLI,
  gallery and dataset functions that should stay archived
- src has no consumers of the eight independent-court mathematical modules.
  Experiment/test consumers will import their new canonical locations

## Execution log

- Read repository rules, plan-and-execute, worklog, style-py and Serena instructions
- Three independent traces covered search, scoring and runnable regression checks
- Baseline full Ruff: exit 1, 1,080 inherited errors; full log baseline-ruff.log
- Baseline whole-project Pyrefly: exit 0, 0 errors (39 suppressed)
- Commit: none; not authorised

## Completed move

- Moved 24 coherent implementation modules into court_detector. Extracted live
  functions from old experiment drivers without changing numerical bodies.
  Normal package imports replace bare-name/path loading. Runtime measurement
  functions and the temporary sampling patch share one module; image-source
  classes have one identity. Both JSON round trips remain.
- Updated 29 test/experiment consumers, including the direct-file test loader.
  Added bounded search tests and checks for package-only imports, shared types,
  unchanged sys.path, bare-name interference and sampling-patch restoration.
- Archived 184 superseded Python/shell files plus two historical documents.
  The 186 files match the pre-move snapshot byte-for-byte. The dated check
  drivers are archived too; their reports and saved outputs stay in place.
- Updated the existing code guide, pickup, file map, speed-up design and
  archive links. The cheap-score proposal now names proposals.propose_role
  and generation.generate. Current state still belongs only in pickup.
- Preserved all 8,348 inventoried data/config files at their existing paths,
  with unchanged size and modification time. No remote jobs or data changes.

## Verification

- Numerical baseline: two pre-move captures matched exactly after timing fields
  were omitted. The post-move capture and a second capture after archiving both
  matched all 33,858,811 bytes (exit 0). The check covers a saved view's 1,037 net
  rows and final choice/refit at geometry weights 0 and 0.1; one real search pair
  with 256 combinations; parent/child scoring; camera filtering off/on; template
  support and camera outputs. This is not a full video or all-view rerun.
- Full pytest: exit 1, 2,359 passed, 29 skipped, one failure. The sole failure was
  an unrelated interpreter lookup: bare python was absent from shell PATH.
  With the project venv on PATH, that test plus all module/search tests passed:
  19 passed, exit 0. No code change was needed for that environment issue.
- Whole-project Pyrefly: exit 0, 0 errors (39 suppressed), unchanged from baseline.
- Ruff over maintained package and changed test/experiment consumers: exit 0.
  Full-repo Ruff: exit 1, 1,096 issues versus 1,080 before the move. Comparing
  old/new paths found 17 added import-order warnings in byte-identical archived
  files and one removed warning. Moving files changes Ruff's import grouping;
  archived originals were preserved rather than reformatted.
- All 1,932 non-archive document links resolve. Canonical links and heading
  anchors, previous archival preservation, and git diff --check pass (exit 0).

## Independent review

One fresh read-only reviewer traced saved inputs through generation, scoring,
net choice and stripe refit. The reviewer separately checked patched globals
and class identities. Random sample: image_sources.py, seed 20260927, drawn
from the remaining top-level modules. No verified live-path defect was found.

The reviewer's independent numerical capture matched the same baseline (exit 0).
All 31 top-level modules imported with unchanged sys.path when src was on the
Python path. Eighteen identity/provenance tests passed (exit 0). Retained
function/class ASTs across the 24 moved modules matched originals apart from
import adapters. A lead about stale imports in dated check drivers was resolved:
those reports already require historical commits; their scripts are now archived.

The review has the same local-data limits as the numerical check. Full video
runs still require the source videos and people records held on Carmack.

## Recovery

.recovery/before-code-move-20260927.tar.gz is the verified source snapshot.
.recovery/code-move-checks-20260927.tar.gz preserves the numerical harness,
baseline, move maps and check logs. Both paths are relative to court_det_fix.
Old script paths were deliberately left unchanged inside archived files.
No commit, push or merge was made. GPU, Numba and parallel work remain untouched.

## Fresh-reader check

A fresh reader started at INDEX.md and recovered the sole current handover,
maintained package, next work, cheap-score implementation location, plain-language
meaning of G0/G1 and historical-code boundary. No conflicting live guidance was
found. Link checks across the seven reviewed documents passed (exit 0).
