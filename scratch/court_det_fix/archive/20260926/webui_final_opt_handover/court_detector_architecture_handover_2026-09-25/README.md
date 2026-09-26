# Court detector optimisation: architectural handover pack

**Repository reviewed:** `ahalp90/badminton_cv_annotator`  
**Branch:** `exp/court-det-opt`  
**Branch head reviewed:** `a53b8a0e318a59c4d262f01d24d5653fe3be1296`  
**Prepared:** 25 September 2026

## Headline

The branch has already removed most of the obvious waste. The remaining exact, single-threaded clean-ups are worth doing, but they are not enough to turn the current slowest scene from **666 s to 90 s**.

The recommended path is a partial repackaging of the existing detector, not a replacement:

1. Build one **in-memory exact deployment path** and remove research I/O.
2. Turn search and W5 into a **deterministic parallel task graph**.
3. Add a Faster-R-CNN-like **coarse proposal → exact ROI scoring cascade** while retaining all 16 direction groups and the existing exact score/refit/ranker.
4. For repeated camera views, inject the previous court as the first proposal, validate/refit it cheaply, and fall back to the full cascade when validation fails.

The committed 28-view coarse-score replay is encouraging: a 16-sample proposal score followed by exact scoring of the top **8,192** courts per pair reproduced **all 3,369 pair shortlists** while costing about **66% of the current court-scoring stage**. That projects to roughly **8.5% off the whole current run**, before parallelism. A budget of 4,096 projects to 10.7% off, but missed one internal pair shortlist and therefore needs an adaptive fallback.

Parallelism is the main route to the 90 s *maximum* target. A target-sizing model says the current 666 s slowest scene reaches roughly 105 s with 16 workers only when about 92% of work is parallelisable; combining that with a measured/validated 15–20% algorithmic reduction reaches about 86–91 s. This is a model, not an end-to-end measurement, so the first implementation milestone should be an exact parallel prototype on the fixed deployment machine.

## Start here

- [Full evaluation](COURT_DETECTOR_ARCHITECTURE_EVALUATION.md)
- [Implementation blueprint](IMPLEMENTATION_BLUEPRINT.md)
- [Acceptance and benchmark runbook](ACCEPTANCE_AND_BENCHMARK_RUNBOOK.md)
- [Evidence index](EVIDENCE_INDEX.md)
- [Assumptions and limits](ASSUMPTIONS_AND_LIMITS.md)

## Pack contents

- `data/`: committed-data extracts, projections, and synthetic benchmark output
- `benchmarks/`: runnable synthetic benchmarks and model builder
- `figures/`: report figures
- `patch_sketches/`: pseudocode and code-seam sketches

## Reproduce the synthetic evaluation

```bash
python benchmarks/bench_response_map_reuse.py \
  --output data/synthetic_response_map_results.csv

python benchmarks/bench_cartesian_topk.py \
  --output data/synthetic_cartesian_results.csv

python benchmarks/build_models.py --root .
```

The synthetic timings are diagnostic only. The repository's committed Carmack measurements remain the primary evidence.
