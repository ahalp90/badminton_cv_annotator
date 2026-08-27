# Full-dataset contact experiment plan

## Completed stage: ORIENT

Objective: confirm the experiment contract, validation split, OUT-list and first implementation batches.

Entry facts:

- The three-video pilot is complete through its Phase 3 stop rule.
- The branch is clean at `5f6da72` before this campaign scaffolding.
- The 40-video extraction is complete and mapped in the existing local-only data map.
- The pilot freezer and scorer hard-code three fixtures and leave-one-fixture-out scoring.

Outcome: the user confirmed `contract.md`, the recommended validation split and the planned local commit approach on 2026-08-27.

## Current stage: MAP

Objective: trace the pilot freezer, manifest, ground-truth loader, model split, retained-score and downstream rally contracts.

Outputs:

- a source-backed current-system map;
- explicit reusable and replaced boundaries;
- the silent-failure cases each implementation batch must test.

Exit gate: B1–B4 can be executed without rediscovering their data flow, invariants or tests.

## Planned stages

1. **MAP:** trace the pilot input, manifest, ground-truth, split, scoring and downstream rally contracts. In progress.
2. **PLAN:** turn the map into executable batches and add focused contract tests.
3. **BUILD:** generalise the scripts, freeze full-corpus features and run the fixed baselines.
4. **VERIFY:** run focused local gates, project gates and independent diff/result review.
5. **CLOSE:** retain portable results and leave the exact ShuttleSet22 next action.

## Initial batches and commit messages

### B0 — portable experiment record

- Files: the campaign records in this directory.
- Change: record the agreed scope, split, OUT-list and resume point.
- Gate: manual check for machine paths, hostnames and access details.
- Proposed commit: `Set up the full-dataset contact experiment`

### B1 — roster and split contract

- Files: new experiment modules and tests in this directory.
- Change: load a portable video roster and enforce disjoint train, validation and test identities.
- Gate: focused unit tests plus lint and types.
- Proposed commit: `Add the full-dataset contact split`

### B2 — arbitrary-video feature freeze

- Files: new experiment freezer and tests in this directory.
- Change: adapt the pilot's per-video feature construction to a manifest-defined roster without changing its feature equations.
- Gate: pilot-fixture equivalence, focused tests, lint and types.
- Proposed commit: `Freeze contact features for any video roster`

### B3 — fixed train/validation scorer

- Files: new experiment scorer and tests in this directory.
- Change: fit HGB and RF on the training videos, select decisions on the fixed validation videos and retain per-video scores.
- Gate: synthetic split tests, deterministic repeat, focused tests, lint and types.
- Proposed commit: `Score fixed contact train and validation splits`

### B4 — strict rally evaluation

- Files: new experiment evaluation modules and tests in this directory.
- Change: report timing, side and fully-correct retained-rally results for the validation stream.
- Gate: pilot-record reproduction, focused tests, lint and types.
- Proposed commit: `Add full-dataset rally scoring`

### B5 — baseline run and report

- Files: portable manifests, compact result artefacts, figures and report in this directory.
- Change: record the 32/8 HGB and RF baseline and the evidence-backed next experiment.
- Gate: repeated result verification, independent result audit and Git disclosure check.
- Proposed commit: `Record the full-dataset contact baseline`

Later experiment commits will be planned only after the baseline errors justify them.

## Review roster

- Primary integration and experiment judgement: root agent.
- Mechanical pilot-to-full-data map: Luna xhigh, read-only; completed and subject to primary audit.
- Per-batch review: fresh bounded reviewer declared before each launch.
- Result-level adversarial review: DeepSeek V4 Flash and agy Opus, read-only, after a complete baseline exists.

## Audit check-ins

### A1 — before the first full-corpus freeze

Check the roster, split, path portability, label-blind boundary and pilot-feature equivalence. The full freeze does not start until this review is clear.

### A2 — before the first HGB/RF baseline launch

Check split isolation, ground-truth loading order, negative sampling, threshold selection, deterministic seeds and result-manifest binding. The baseline does not start until this review is clear.

### A3 — after baseline scoring

Independently recompute the main timing and strict-rally totals from retained per-video predictions. Reviewers then challenge the interpretation and proposed next experiment. No follow-up model run starts from an unaudited baseline.

### A4 — before the ShuttleSet22 test

Check the frozen settings, the overlap exclusions, the final 40-video refit record and the rule that test labels only score the result. The test runs once after this check-in.
