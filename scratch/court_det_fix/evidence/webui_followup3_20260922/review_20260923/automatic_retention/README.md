# Saved automatic candidates survive the 12-family screen

The cached check supports an actual 12-versus-16 matcher comparison. **All eight
historically approved automatic fits survive.** The screen drops three current
score-winner roles, so it changes the available selection outcomes. Keep the
full 16-family comparator and G0; production pruning is not enabled.

This extends [the earlier assessment](../../ASSESSMENT.md). It uses the same
nine views and the frozen family ranking from the returned SVD screen.
No matcher search, new fit, changed direction or threshold sweep was run.

## Retention

| Cached measure | Full 16 families | Retained 12 families |
| --- | ---: | ---: |
| Ordered direction-family pairs | 2,160 | 1,188 |
| Automatic candidates | 19,286 | 8,657 |
| Historical approved witnesses | 8 | 8 |
| Best automatic reference-agreement candidate per view | 9 | 9 |
| Candidates within one working pixel of each pool's best error | 34 | 34 |
| Line-score winners | 9 | 8 |
| Paint-score winners | 9 | 7 |

The historical witnesses span six cases. Their candidate IDs and native corner
arrays match the automatic cache exactly. They include no GX witness.

The six approved E3 reference geometries are a different set. None matches an
automatic candidate within 0.001 working pixels. Do not describe the nearest
automatic candidate as approved. The 34-candidate neighbourhood measures
reference agreement, not usability. GX5's best maximum corner error is
89.94 working pixels, so retaining it does not establish a useful GX5 fit.

Three winner roles disappear:

| Case | Winner | Candidate | Original family IDs |
| --- | --- | --- | --- |
| GX5 | Line score | `181:29836` | 12, 1 |
| GX5 | Paint score | `10:1274` | 0, 11 |
| Am2-28019 | Paint score | `184:4123` | 12, 4 |

The earlier user preference for centre-to-edge and the Am1 net-band failure
concern fitting and semantic acceptance. They do not validate these scores or
turn this pruning check into a detector-quality result.

## Identity and numerical checks

All nine automatic caches match baseline B's points, masks, direction IDs,
line arrays, normalisation and settings. Each contains the same 240 ordered
pair identities. The mask uses original family IDs, without reranking them.

Projection reconstruction agrees exactly with every saved candidate after
importing the producer's canonical float32 `CORNER_COURT_M`. Duplicating those
coordinates as float64 decimal literals caused large errors near the projective
horizon. The final check uses an absolute tolerance of 1e-7 working pixels and
zero relative tolerance; the observed maximum discrepancy is zero.

## Reproduction and next step

[run_retention.py](run_retention.py) writes [results.json.gz](results.json.gz).
[check_approved_witnesses.py](check_approved_witnesses.py) writes
[approved_witnesses.json.gz](approved_witnesses.json.gz). Run either from the
repository root with `~/.venvs/badminton-cicd/bin/python` and set
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
Both runs and scoped Ruff pass, exit 0. Whole-project Pyrefly passes, exit 0
(0 errors, 39 suppressed). No production code changed.

Measure actual matcher work and runtime with 12 versus 16 families next.
The retained pairs account for 4,254.21 of 8,839.07 cached elapsed seconds.
Those sums are historical accounting, not a measured speedup. Keep direction
generation, assignment fitting and candidate caps fixed. Assess broader search
at equal cost separately, after measuring any savings.
