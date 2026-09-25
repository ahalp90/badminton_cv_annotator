> Historical record from the web-UI handover packet of 23 September 2026,
> filed on 25 September 2026. The
> [speed-up README](../../../court_detector_optimisation_handover/README.md)
> owns current status; the [archive map](../../README.md) records the original
> paths. Content is unchanged. The packet's `tools/`, `task_manifest.json`,
> `LOCAL_MODEL_PROMPT.md` and single-file copy are in git history at commit
> 92535b6e.

# Source and data map

## Reviewed revision

- Branch: `fix/court-det`
- GitHub revision: `2a00f77172a59374136e19246ede1d854aaaff01`
- Review date: `2026-09-23`

The local checkout may contain later commits and uncommitted modules/data.
Record actual state; do not reset to this revision.

## Resolve the live imports

Run:

```bash
python tools/resolve_hotpath.py --court-root scratch/court_det_fix
```

The reviewed `w5_holistic/run_w5.py::add_helper_paths()` establishes this
precedence:

1. `next_steps_20260916/webui_seed/source`
2. `frozen_helpers_20260914/marking_diagnosis`
3. `frozen_helpers_20260914/vp_pruning`
4. `frozen_helpers_20260914/axis_matching`
5. `frozen_helpers_20260914/legacy`
6. `src`
7. repository/court root

At the reviewed commit, the hot path is primarily:

| Concern | Reviewed path / function |
| --- | --- |
| Fresh SVD12 generation | `w5_holistic/automatic_generation.py::generate` |
| Pair proposal orchestration | `next_steps_20260916/webui_seed/source/run_given.py::propose_role` |
| Axis matching | `.../source/projective_seed.py::{basis_for, offsets, score_axes, match_axis, combine}` |
| Geometry and continuous support | `frozen_helpers_20260914/marking_diagnosis/scan_population.py::{geometry, continuous_support, retain}` |
| Player test | `frozen_helpers_20260914/legacy/zone_net.py::player_fractions` |
| Observation grouping | `experiments/annotator/independent_court/assignment.py` |
| Pool evidence | `.../source/run_automatic.py::evaluate_pool` |
| Gates | `frozen_helpers_20260914/marking_diagnosis/run_diagnosis.py::gate_evidence` |
| Stripe evidence | `experiments/annotator/independent_court/stripe_observations.py` |
| Refit | `experiments/annotator/independent_court/fixed_stripe_refit.py` |
| W5 orchestration/ranking | `w5_holistic/run_w5.py` and `w5_holistic/verifier.py` |
| Runtime benchmark runner | `svd_search/run.py` |
| Line-template vector camera precedent | `w5_holistic/line_template_source.py` |

Do not edit a frozen snapshot until the import resolver confirms it is the active
copy or the change deliberately creates a new integrated module.

## Existing tests and useful checks

- `frozen_helpers_20260914/axis_matching/test_projective_seed.py`
  - synthetic recovery;
  - degenerate directions;
  - horizon handling;
  - proof that per-axis player rules are necessary for the joint rule;
  - finite-score rotation invariance.
- `net_recovery/test_bounded_trial.py`
- W5 verifier/ranking tests named in the repository handover.
- Scoped Ruff, Pyrefly, and syntax checks described by repository instructions.

Add focused tests beside the active module. Do not build an unrelated test
framework.

## Existing timing/evidence artefacts

- `svd_runtime/RESULTS.md` and sibling result records:
  full16 versus SVD12 matcher timing.
- `svd_search/run_20260923/cases/`:
  stage timings and counts for baseline/deeper/shortlist arms.
- `svd_search/run_20260923/generation/`:
  per-pair generation records.
- `svd_search/WORKLOG.md`:
  accepted/rejected quality observations.
- `net_recovery/statistics/paired_reference_report.md` and siblings:
  paired quality results.
- `w5_holistic/verifier.py`:
  fixed cohorts, references, hard validity, ranking, and record conventions.

Use existing records rather than rerunning completed historical experiments.

## Critical local-only inputs named by the repository handover

Verify presence and checksums before a new worktree or HPC transfer:

```text
local_scratch/net_recovery/20260923/bounded/bounded_trial.json.gz
local_scratch/net_recovery/20260923/selected_polarity/bounded_results.json.gz
local_scratch/net_recovery/20260923/selected_polarity/bounded_requests.json.gz
local_scratch/external_delegate/20260923-am1-recovery-trial/seeded_pool.json.gz
local_scratch/external_delegate/20260923-net-bounded-audit/result.md
```

Also inventory all untracked development packs and frozen run outputs referenced
by `scratch/court_det_fix/FP_INDEX.md`.

Never substitute a similarly named tracked pool. Cache identity must include
image hash/dimensions, geometry, evidence source, settings, and relevant code
revision.

## Documentation drift warning

Some older handovers refer to modules such as
`w5_holistic/court_detection.py`, `candidate_cache.py`, or `scoring.py`. They are
not present in the reviewed Git tree. Treat those names as historical design
language unless the local uncommitted checkout actually contains them.

## Production integration boundary

The current accepted quality result is assembled experimentally. No single
runner at the reviewed revision combines fresh SVD12 G0/G1 generation, seeded
templates, bounded net selection, and stripe refitting. First build a fresh,
reproducible baseline adapter or use the current local integrated runner if one
now exists; do not compare a fresh patch with historical paint scores.
