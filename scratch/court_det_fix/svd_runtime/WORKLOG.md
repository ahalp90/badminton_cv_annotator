# SVD matcher runtime comparison

## Resume

Preparing a bounded nine-case comparison of 12 versus 16 direction families.
User approved proceeding and permits up to six remote workers. The cached
check preserves all eight historically approved fits and all nine best
reference-agreement candidates. Actual runtime is the remaining question.

## Scope and concerns

- Measure the unchanged matcher and camera-direction gate, with original
  family and pair IDs. Keep direction normalisation, scoring and caps fixed.
- Measure SVD ranking overhead separately. This is matcher-stage timing;
  image loading, full evidence scoring and scene processing are excluded.
- Compare shared-pair outputs across arms. Keep the 16-family comparator and
  G0 fallback. No production detector change or accuracy tuning in this pass.
- Cached winner substitutions are mixed: GX5 line-reference error improves
  530 to 352 working pixels; GX5 paint error worsens 531 to 1,652; Am2-28019
  paint error improves 892 to 17. These are not fresh visual judgements.
- Run case pairs sequentially within each worker, alternating arm order.
  Up to six workers, one native numerical thread each. Record CPU time as
  well as elapsed time because shared-host scheduling may affect timing.

## Modules and execution

- `run_benchmark.py`: new diagnostic runner only; delegated to Sol medium,
  default service tier. Reuses frozen matcher functions.
- Existing fitting and detector modules remain unchanged.
- Carmack connection succeeded with network access enabled after sandbox DNS
  resolution failed. No alternate host was used.
- Existing branch authority allows a checkpoint commit and push for remote
  execution. Planned header: `Measure SVD pruning before changing search`.
- Sol's syntax/import, nine-case SVD rank and Ruff checks pass. Whole-project
  Pyrefly passes (0 errors, 39 suppressed). The coordinator's SS03-16 smoke
  completes matching and fresh-arm comparison, then fails the historical
  cache check at pair 1, horizontal axis entry 156 (axis ID 2283 versus 2293).
  This ordering difference needs diagnosis; the comparison tolerance is unchanged.
- Opus is auditing runner correctness and, at the user's additional request,
  simple compute-efficiency opportunities in the called matcher. Both are
  read-only and use `claude-opus-5-5` at high effort.
- Carmack's two-worker smoke on SS03-16 and SS21-20 passes, exit 0, 37.1 s
  batch wall time. All four full-arm pairs also match historical cache records
  in a separate comparison, exit 0. The local SS03-16 difference does not
  reproduce in the Carmack environment. Native thread counts are one.
- Opus runner audit found no verified defect. It independently replayed three
  SS03-19 pairs against the local cache. Its useful limitations: host contention
  can affect elapsed-time ratios, historical mismatch before writing loses a
  result, and matcher-only timing excludes later scoring. The full Carmack run
  will preserve its results first and compare the historical cache afterwards.
  Record shared-pair timing ratios alongside overall savings.
- Runtime runner committed and pushed as `394b9ed`. Remote checkout is
  `/scratch/ahalperi/court_det_fix/svd_runtime_checkout_20260923`; run outputs
  are under `/scratch/ahalperi/court_det_fix/svd_runtime_20260923`.
