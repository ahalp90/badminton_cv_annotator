# SVD12 search-depth run

Run from the Carmack checkout at the exact path below. The old
`/scratch/ahalperi/court_det_fix/svd_runtime_20260923` run is unrelated and must stay untouched.

```bash
cd /scratch/ahalperi/court_det_fix/svd_search_checkout_20260923
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
/home/ahalperi/.venvs/venv-rtmlib/bin/python scratch/court_det_fix/svd_search/run.py \
  --workers 6 --output /scratch/ahalperi/court_det_fix/svd_search_run_20260923
```

Use `--workers 5` if the original benchmark is still running; otherwise use
`--workers 6`. The parent will choose at launch time.

`--case ID` and `--arm baseline|deeper|shortlist` are
repeatable filters. For a local one-pair smoke, add `--max-matched-pairs 1` and use a
fresh output directory. Omit that flag for results. The output path must be absent or
empty; the runner refuses cache reuse. Each process uses one numerical thread.

Six cases, three independent arms, automatic G0 only. Axes/courts caps are 512/256/256,
640/256/256 and 512/512/512. Fitting and ranking call the W5 functions unchanged.
The W5 4,3 visibility floor applies to its line-template population, which this G0-only
run does not generate. Reference corners are used only after selection.

Files to sync from this change:

- `scratch/court_det_fix/w5_holistic/automatic_generation.py`
- `scratch/court_det_fix/svd_search/` (runner and postprocessor)

Required existing code is `src/`, `scratch/court_det_fix/w5_holistic/`,
`scratch/court_det_fix/wider_evaluation/{run_cases.py,generation.py,measurement.py}`,
`scratch/court_det_fix/next_steps_20260916/webui_seed/source/`,
`scratch/court_det_fix/frozen_helpers_20260914/` and their usual dependencies.
Input packs and frames are listed in `input_manifest.json.gz`. The GX0 baseline direction
file in the manifest supplies unchanged estimator settings. Other directions are
estimated afresh from each frozen source. The input frames have no extra frame work.

After results return, run:

```bash
~/.venvs/badminton-cicd/bin/python scratch/court_det_fix/svd_search/build_gallery.py \
  --input /path/to/svd_search_run_20260923 \
  --output scratch/court_det_fix/svd_search/gallery
```

The gallery and `comparison.json.gz` are local output. Candidate IDs may differ
between arms; comparison uses reference corner geometry. Each oracle is restricted
to candidates in the ranker's selectable `provisional_rank`. Reference corners
are used for retrospective diagnosis, never detector selection. The saved W5
selection may come from G0, G1 or line templates; the gallery shows its population.
