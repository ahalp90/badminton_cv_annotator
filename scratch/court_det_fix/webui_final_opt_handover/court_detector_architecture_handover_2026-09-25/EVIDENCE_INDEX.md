# Evidence index

## Repository revision

- Repository: `ahalp90/badminton_cv_annotator`
- Branch: `exp/court-det-opt`
- Head reviewed: `a53b8a0e318a59c4d262f01d24d5653fe3be1296`
- Main timing/evaluation documents dated 24 September 2026

## Primary committed evidence

| Claim used in this pack | Repository path |
|---|---|
| Current optimisation history, remaining/open/rejected ideas | `scratch/court_det_fix/court_detector_optimisation_handover/CLAUDE_FOLLOWUPS.md` |
| Profiling, full16 versus SVD12 quality, player/stripe fixes | `scratch/court_det_fix/court_detector_optimisation_handover/CLAUDE_EVALUATION.md` |
| Current 8,931 → 8,250 stage comparison; slowest 772 → 666 s | `scratch/court_det_fix/court_detector_optimisation_handover/claude_evidence/w5_savings/carmack_stage_times_vs_item12.txt` |
| Coarse 4/8/16-sample proposal replay over 3,369 calls | `scratch/court_det_fix/court_detector_optimisation_handover/claude_evidence/prefilter/summary.txt` |
| Coarse-proposal measurement implementation and stable-order argument | `scratch/court_det_fix/court_detector_optimisation_handover/claude_evidence/prefilter/measure_prefilter.py` |
| Exact Lipschitz bound rows | `scratch/court_det_fix/court_detector_optimisation_handover/claude_evidence/shortlist_bound/gap_bound_rows.jsonl` |
| No exact duplicate scored axis hypotheses in sampled calls | `scratch/court_det_fix/court_detector_optimisation_handover/claude_evidence/axis_duplicates/summary.txt` |
| End-to-end stage wiring and deployment/research boundaries | `scratch/court_det_fix/d17_timing/WIRING.md` |
| Current pair loop and caps | `scratch/court_det_fix/w5_holistic/automatic_generation.py` |
| Axis matching, combination, and joint player implementation | `scratch/court_det_fix/next_steps_20260916/webui_seed/source/projective_seed.py` |
| Current role proposal and finite scoring | `scratch/court_det_fix/next_steps_20260916/webui_seed/source/run_given.py` |
| 64-sample continuous support and greedy retention | `scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis/scan_population.py` |
| Current serial parent/refit loops | `scratch/court_det_fix/w5_holistic/run_w5.py` |
| Full16 versus SVD12 per-view/stage report | `scratch/court_det_fix/court_detector_optimisation_handover/claude_evidence/d17/d17_results.txt` |

## Data reproduced in this pack

| Pack file | Origin |
|---|---|
| `data/current_top_level_stage_times.csv` | current Carmack stage comparison |
| `data/current_detailed_stage_times.csv` | current Carmack stage comparison |
| `data/committed_prefilter_tradeoff.csv` | committed prefilter summary |
| `data/committed_shortlist_bound_rows.csv` | committed bound JSON rows |
| `data/cascade_scenario_model.csv` | arithmetic projection from current 8,250 s and 2,054 s finite score stage |
| `data/parallel_latency_model.csv` | Amdahl target-sizing model |
| `data/combined_target_sizing_model.csv` | algorithmic reduction plus optimistic parallel model |

## Synthetic evidence generated for this pack

- `benchmarks/bench_response_map_reuse.py`
  - 960×540, two float32 maps, 12 markings, 64 samples, 256-court batches
  - line-coherent synthetic sample pixels
  - verifies byte-identical scores on local NumPy 2.3.5
- `benchmarks/bench_cartesian_topk.py`
  - 512 retained hypotheses per axis
  - compares full NumPy outer-sum/argpartition with a heap best-first enumerator
- Environment: `data/synthetic_benchmark_environment.json`

The synthetic benchmarks test implementation mechanics, not detector accuracy. Accuracy conclusions in the report are based on committed repository evaluations.
