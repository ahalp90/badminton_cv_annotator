# Prompt for the local frontier model

You are taking over performance engineering of the accepted court detector in
`badminton_cv_annotator`.

Read this packet in order:

1. `README.md`
2. `STATIC_AUDIT.md`
3. `PATCH_QUEUE.md`
4. `EXPERIMENT_PROTOCOL.md`
5. `SOURCE_MAP.md`
6. repository `scratch/court_det_fix/pickup.md`
7. repository `handover_20260923/{00_SHARED_CONTRACT,01_LAUNCH_OPTIMISATION}.md`
8. repository instructions.

## Mission

Make the detector deployable without losing the accepted recall, precision,
source coverage, ranking, abstention, or final geometry behaviour. The target is
roughly 30 seconds for a five-minute video, with 90 seconds an upper-end
tolerance; report startup separately. Expensive search should eventually scale
with distinct compatible views.

There is no annotation budget. Do not ask for new held-out samples, new labels,
or manual review campaigns. You have the existing frozen development data,
references, controls, user rulings, uncommitted local data, and HPC.

## Safety and repository rules

- Begin by recording actual Git HEAD, branch, and status. Never reset, clean, or
  delete untracked data.
- Resolve active module paths at runtime. Multiple byte-identical and historical
  snapshots exist.
- Use fresh output directories. Preserve accepted historical results.
- Keep G0, G1, templates, SVD12, existing caps, gates, ranking, tie order,
  bounded net rule, abstention, and selection-then-stripe-correction.
- Do not use reference labels in runtime pruning.
- Keep one numerical/OpenCV thread per worker and avoid nested process pools.
- Separate exact optimisations from any semantic search-space experiment.
- Commit coherent accepted patches on the feature branch only after checks pass.

## Required first pass

1. Run the packet bootstrap and import resolver.
2. Locate the locally fastest existing broadcast case and a slow GX case from
   saved timings.
3. Capture cold/warm one-worker baselines with stage wall/CPU time, peak RSS,
   concrete work counters, and complete output records.
4. Profile the actual imported code.
5. Quantify:
   - total axis interpretations;
   - player-compatible interpretations;
   - `score_axes` calls;
   - basis inversions;
   - combined axis pairs;
   - geometry/player-valid pairs;
   - full-evidence candidates;
   - refit attempts/residual evaluations.

Do not launch the full corpus first.

## Implement in this order

### 1. Prepared geometry and inverse lifetime

Prepare frame-invariant group geometry/template tables once and pair-invariant
basis inverse/rectified feet once. Pass them explicitly. Prove the same retained
axis/candidate output.

### 2. Existing player gate before axis scoring

Calculate `necessary_players()` before `score_axes()` and avoid scoring rejected
parameters in lean mode. Keep a full-diagnostics oracle mode. This gate already
exists; you are changing sequence, not policy.

### 3. Adaptive scoring blocks and prepared endpoint algebra

Increase batch size under a memory budget. Then replace repeated general line
evaluation with the affine endpoint-coefficient form. Use scalar baseline
fallback at support/tie frontiers.

### 4. Lean combined-pair path

Avoid generic inversion of every combined court. Derive joint player predicates
from rectified feet and axis scale/shift; block the Cartesian product; project
and validate corners once; retain from compact arrays; materialise only
survivors.

### 5. Batch evidence and gates

Prepare feet once. Batch player fractions, detector `_score`, camera diagnostics,
profiles, and stripe evidence. Preserve order and frontier rechecks.

### 6. Refit

Add complete-key duplicate caching and reuse stripe nearest-distance work.
Implement an analytic Jacobian only with the current solver as a fallback.

### 7. Parallelism and scene reuse

Use deterministic case/pair/refit parallelism at one level. After first-search
cost is affordable, move compatible-view matching before full inference with
validate-or-full-search fallback.

## Validation

For each patch:

- create a separate output directory;
- compare against a freshly measured control on the same revision/hardware;
- use `tools/compare_records.py`;
- compare stable candidate identities, pair status/order, retained indexes,
  winner, acceptance/abstention, gates, and final corners;
- report numerical frontiers explicitly;
- run one fast and one slow full case before the frozen corpus;
- stop a patch when it has no useful saving or an unexplained output change.

Use existing references and controls only after equivalence passes to confirm no
quality regression. Preserve contrary cases and unlabelled rows in reports.

## Deliverables

Maintain a single optimisation session directory with:

```text
worklog.md
evidence.md
mechanisms.md
runs.md
```

For each accepted patch, record mechanism, files, commit, commands, hardware,
threads, before/after stage and total timing, peak memory, output comparison,
and remaining risk.

Finish with:

1. accepted/rejected patch table;
2. complete fresh detector cost per distinct view;
3. five-minute video cost split into startup, decoding/cuts, first searches,
   validation/reuse, fallbacks, and output;
4. paired quality/control results;
5. unresolved failures;
6. exact Git/worktree/check state.

Do the work now in the current session. Do not promise background work or ask for
new annotations.
