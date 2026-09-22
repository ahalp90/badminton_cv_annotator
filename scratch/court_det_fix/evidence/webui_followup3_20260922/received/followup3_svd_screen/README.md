# Follow-up 3: fixed-B SVD pruning screen

Start with **[return.md](return.md)** for the decision, measurements, coordinate caveat and witnesses.

The analysis reads the supplied frozen E2, E3 and baseline-direction records. It reuses only the two SVD diagnostic functions from the frozen producer. It does not import or run the detector, matcher, OpenCV, a fitting optimiser, or the rest of the producer module. It does not replace B directions with B_svd directions.

## Reproduce

Requires Python 3.10+ and a compatible NumPy installation; the retained run used Python 3.13.5 and NumPy 2.3.5. The original `task3_inputs.tar.gz` is not duplicated in this results archive. After unpacking it beside this directory:

```sh
cd followup3_svd_screen
python run_screen.py --inputs ../task3_inputs --output rerun_results
TASK3_INPUT_ROOT=../task3_inputs python -m unittest -v test_screen
```

`--output` must name a new or empty directory. Input paths are read-only. The input root is the directory containing `.github`, `scratch` and `PINNED_REVISION.txt`, not the `scratch/court_det_fix` subdirectory. A pinned checkout with the same source/data layout also works; absence of a revision marker is recorded explicitly.

## Outputs

| File | Contents |
| --- | --- |
| `return.md` | Scientific result, go/no-go boundaries, all-view errors and selected groups |
| `run_screen.py` | Complete analysis; stdlib and NumPy only |
| `test_screen.py` | Eight focused tests, including complete nine-view replay and meaningful failures |
| `verification.txt` | Retained successful test log |
| `results/summary.csv.gz` | 45 case/budget rows, best finite/converged fits, original selected IDs, counts and deltas |
| `results/groups.csv.gz` | 144 groups with support IDs/counts, singular values, gap, residual and rank |
| `results/aggregate.csv.gz` | Five aggregate budget rows, including approved/manual populations |
| `results/results.json.gz` | Full provenance, diagnostics, all selections, coordinate checks and exact cached regression witnesses |

All JSON/CSV outputs use gzip. The analysis checks the saved E3-to-E2 and estimator MD5 cross-links and reproduces the B-support SVD diagnostics before interpreting retention. It never treats an unconverged finite fit as a failed fit.

The result is a screen of cached direction-fit potential, not an end-to-end court detector result. K=12 was selected after examining these nine development views; this is not held-out validation or a runtime benchmark.
