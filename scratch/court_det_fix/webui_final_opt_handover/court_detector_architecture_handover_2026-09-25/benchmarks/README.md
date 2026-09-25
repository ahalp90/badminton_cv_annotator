# Benchmarks and models

## `bench_response_map_reuse.py`

Tests an exact local feature-reuse idea for the Gaussian distance response in `continuous_support`.

- Baseline: gather float32 distances, apply Gaussian for every sampled court point.
- Reuse: apply the same Gaussian once to the two 960×540 distance maps, then gather responses.
- Includes dense-map precomputation in the primary timing.
- Verifies byte-identical final scores on the local NumPy build.

Interpretation: use only for large court populations and validate on Carmack. This is not an end-to-end detector benchmark.

## `bench_cartesian_topk.py`

Tests whether best-first enumeration of separable horizontal/vertical axis scores is itself a major speed lever.

Interpretation: it helps only at very small K and the current pair-loop own time is already small. Use it to avoid materialisation in a proposal implementation, not as the core optimisation.

## `build_models.py`

Writes CSVs and figures from committed repo figures plus transparent arithmetic assumptions. It does not claim measured speed for parallel execution or the proposed cascade.
