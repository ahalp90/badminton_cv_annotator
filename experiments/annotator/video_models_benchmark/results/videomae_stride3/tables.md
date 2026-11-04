| Run | Status | Selected (updates) | Last check (updates) | Validation loss | Fully correct rallies (%) | Correct-player F1 (%) | Contact F1 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| LR 3.125e-5, seed 0 | complete | 55,000 | 70,000 | 0.2072 | 29.19 | 81.26 | 92.37 |
| LR 3.125e-5, seed 1 | complete | 65,000 | 80,000 | 0.2129 | 24.25 | 80.10 | 91.86 |
| LR 1e-4, seed 0 | complete | 55,000 | 70,000 | 0.1767 | 35.33 | 86.70 | 93.72 |
| LR 1e-4, seed 1 | still training | 35,000 | 35,000 | 0.1993 | 31.44 | 84.38 | 93.14 |

Learning-rate ranking (provisional):

| Rank | Head LR | Fully correct rallies (%), seed 0 / 1 / mean | Correct-player F1 (%), seed 0 / 1 / mean | Contact F1 (%), mean | Selected checkpoints |
|---:|---|---:|---:|---:|---|
| 1 | 1e-4 | 35.33 / 31.44 / **33.38** | 86.70 / 84.38 / **85.54** | 93.43 | 55k / 35k |
| 2 | 3.125e-5 | 29.19 / 24.25 / **26.72** | 81.26 / 80.10 / **80.68** | 92.11 | 55k / 65k |

Dense and stride-three scores (%) at ±10 frames:

| Run | Sampling | Contact recall | Contact precision | Contact F1 | Raw player F1 | Alternating player F1 | Raw fully correct | Alternating fully correct |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LR 3.125e-5, seed 0 | Dense | 96.17 | 88.86 | 92.37 | 91.38 | 81.26 | 27.69 | 29.19 |
| LR 3.125e-5, seed 0 | Stride 3, offset 0 | 94.96 | 92.86 | 93.90 | 93.00 | 85.27 | 35.33 | 36.68 |
| LR 3.125e-5, seed 0 | Stride 3, offset 1 | 94.87 | 92.72 | 93.79 | 92.88 | 86.39 | 35.48 | 36.83 |
| LR 3.125e-5, seed 0 | Stride 3, offset 2 | 94.86 | 93.09 | 93.97 | 93.01 | 86.37 | 34.73 | 36.08 |
| LR 3.125e-5, seed 1 | Dense | 96.38 | 87.74 | 91.86 | 90.82 | 80.10 | 23.65 | 24.25 |
| LR 3.125e-5, seed 1 | Stride 3, offset 0 | 95.07 | 92.44 | 93.73 | 92.82 | 85.60 | 33.23 | 34.13 |
| LR 3.125e-5, seed 1 | Stride 3, offset 1 | 95.28 | 92.06 | 93.64 | 92.68 | 86.00 | 30.69 | 31.44 |
| LR 3.125e-5, seed 1 | Stride 3, offset 2 | 95.37 | 92.15 | 93.73 | 92.81 | 85.46 | 31.89 | 32.63 |
| LR 1e-4, seed 0 | Dense | 97.07 | 90.59 | 93.72 | 92.57 | 86.70 | 33.68 | 35.33 |
| LR 1e-4, seed 0 | Stride 3, offset 0 | 96.35 | 93.97 | 95.15 | 94.16 | 90.43 | 42.66 | 44.46 |
| LR 1e-4, seed 0 | Stride 3, offset 1 | 96.23 | 93.21 | 94.70 | 93.71 | 89.15 | 39.82 | 41.92 |
| LR 1e-4, seed 0 | Stride 3, offset 2 | 96.03 | 93.62 | 94.81 | 93.92 | 90.18 | 41.32 | 42.96 |
| LR 1e-4, seed 1 | Dense | 96.96 | 89.60 | 93.14 | 92.02 | 84.38 | 30.54 | 31.44 |
| LR 1e-4, seed 1 | Stride 3, offset 0 | 96.23 | 93.06 | 94.61 | 93.51 | 88.69 | 39.52 | 40.42 |
| LR 1e-4, seed 1 | Stride 3, offset 1 | 96.24 | 93.03 | 94.61 | 93.47 | 88.07 | 42.37 | 43.56 |
| LR 1e-4, seed 1 | Stride 3, offset 2 | 96.23 | 93.25 | 94.71 | 93.69 | 88.96 | 41.02 | 41.62 |

Dense and stride-three scores (%) at ±5 frames:

| Run | Sampling | Contact recall | Contact precision | Contact F1 | Raw player F1 | Alternating player F1 | Raw fully correct | Alternating fully correct |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LR 3.125e-5, seed 0 | Dense | 94.91 | 87.69 | 91.16 | 90.33 | 80.30 | 25.75 | 26.80 |
| LR 3.125e-5, seed 0 | Stride 3, offset 0 | 93.36 | 91.30 | 92.32 | 91.62 | 83.95 | 31.74 | 32.49 |
| LR 3.125e-5, seed 0 | Stride 3, offset 1 | 93.59 | 91.47 | 92.52 | 91.70 | 85.30 | 31.89 | 32.93 |
| LR 3.125e-5, seed 0 | Stride 3, offset 2 | 93.40 | 91.66 | 92.52 | 91.70 | 85.13 | 30.84 | 32.04 |
| LR 3.125e-5, seed 1 | Dense | 94.94 | 86.43 | 90.49 | 89.62 | 79.03 | 21.11 | 21.41 |
| LR 3.125e-5, seed 1 | Stride 3, offset 0 | 93.59 | 91.00 | 92.28 | 91.57 | 84.35 | 29.49 | 29.94 |
| LR 3.125e-5, seed 1 | Stride 3, offset 1 | 93.89 | 90.72 | 92.28 | 91.52 | 84.84 | 27.10 | 27.54 |
| LR 3.125e-5, seed 1 | Stride 3, offset 2 | 93.93 | 90.75 | 92.31 | 91.61 | 84.17 | 28.74 | 29.34 |
| LR 1e-4, seed 0 | Dense | 95.80 | 89.42 | 92.50 | 91.53 | 85.63 | 30.54 | 31.29 |
| LR 1e-4, seed 0 | Stride 3, offset 0 | 94.80 | 92.47 | 93.62 | 92.84 | 89.04 | 37.72 | 39.07 |
| LR 1e-4, seed 0 | Stride 3, offset 1 | 94.93 | 91.96 | 93.42 | 92.59 | 88.10 | 35.78 | 36.98 |
| LR 1e-4, seed 0 | Stride 3, offset 2 | 94.68 | 92.30 | 93.47 | 92.80 | 89.11 | 36.83 | 37.87 |
| LR 1e-4, seed 1 | Dense | 95.68 | 88.42 | 91.91 | 90.96 | 83.44 | 28.14 | 28.89 |
| LR 1e-4, seed 1 | Stride 3, offset 0 | 94.58 | 91.46 | 92.99 | 92.15 | 87.42 | 35.63 | 36.08 |
| LR 1e-4, seed 1 | Stride 3, offset 1 | 94.96 | 91.79 | 93.35 | 92.40 | 86.96 | 37.72 | 38.47 |
| LR 1e-4, seed 1 | Stride 3, offset 2 | 94.70 | 91.77 | 93.21 | 92.40 | 87.78 | 36.38 | 36.83 |

Test results: pending. Run rescore_videomae.py with --test-index once both test seeds finish.
