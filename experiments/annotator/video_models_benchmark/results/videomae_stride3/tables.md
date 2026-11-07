| Run | Status | Selected (updates) | Last check (updates) | Validation loss | Fully correct rallies (%) | Correct-player F1 (%) | Contact F1 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| LR 3.125e-5, seed 0 | complete | 55,000 | 70,000 | 0.2072 | 29.19 | 81.26 | 92.37 |
| LR 3.125e-5, seed 1 | complete | 65,000 | 80,000 | 0.2129 | 24.25 | 80.10 | 91.86 |
| LR 1e-4, seed 0 | complete | 55,000 | 70,000 | 0.1767 | 35.33 | 86.70 | 93.72 |
| LR 1e-4, seed 1 | complete | 40,000 | 55,000 | 0.1882 | 30.84 | 83.48 | 92.95 |

Learning-rate ranking (selected by the queue):

| Rank | LR | Fully correct rallies (%), seed 0 / 1 / mean | Correct-player F1 (%), seed 0 / 1 / mean | Contact F1 (%), mean | Selected checkpoints |
|---:|---|---:|---:|---:|---|
| 1 | 1e-4 | 35.33 / 30.84 / **33.08** | 86.70 / 83.48 / **85.09** | 93.33 | 55k / 40k |
| 2 | 3.125e-5 | 29.19 / 24.25 / **26.72** | 81.26 / 80.10 / **80.68** | 92.11 | 55k / 65k |

Available sampling scores (%) at ±10 frames: Dense, Stride 3, offset 0, Stride 3, offset 1, Stride 3, offset 2

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
| LR 1e-4, seed 1 | Dense | 97.21 | 89.05 | 92.95 | 91.76 | 83.48 | 30.24 | 30.84 |
| LR 1e-4, seed 1 | Stride 3, offset 0 | 96.45 | 92.54 | 94.46 | 93.32 | 87.78 | 38.47 | 38.77 |
| LR 1e-4, seed 1 | Stride 3, offset 1 | 96.42 | 92.72 | 94.53 | 93.33 | 88.05 | 40.87 | 42.07 |
| LR 1e-4, seed 1 | Stride 3, offset 2 | 96.19 | 92.43 | 94.27 | 93.17 | 87.15 | 37.87 | 38.62 |

Available sampling scores (%) at ±5 frames: Dense, Stride 3, offset 0, Stride 3, offset 1, Stride 3, offset 2

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
| LR 1e-4, seed 1 | Dense | 95.86 | 87.81 | 91.66 | 90.80 | 82.59 | 27.10 | 27.54 |
| LR 1e-4, seed 1 | Stride 3, offset 0 | 94.79 | 90.94 | 92.82 | 92.00 | 86.53 | 33.98 | 33.98 |
| LR 1e-4, seed 1 | Stride 3, offset 1 | 94.93 | 91.29 | 93.07 | 92.11 | 86.84 | 35.18 | 35.63 |
| LR 1e-4, seed 1 | Stride 3, offset 2 | 94.66 | 90.96 | 92.77 | 91.96 | 85.96 | 33.23 | 33.68 |

| Tolerance | System | Contact F1 (%) | Correct-player F1 (%) | Fully correct rallies (%) |
|---|---|---:|---:|---:|
| ±10 frames | VideoMAE (base) | 95.35 (95.00–95.70) | 85.17 (84.15–86.19) | 39.50 (36.73–42.26) |
| ±10 frames | X3D-M | 96.90 (96.52–97.28) | 89.95 (88.36–91.54) | 55.95 (52.42–59.48) |
| ±10 frames | Our annotator | 96.85 (96.75–96.92) | 94.63 (94.32–94.78) | 60.18 (59.06–60.96) |
| ±10 frames | MonoTrack | 96.04 (95.95–96.18) | 95.84 (95.69–95.99) | 47.83 (46.44–49.17) |
| ±5 frames | VideoMAE (base) | 93.97 (93.61–94.33) | 84.01 (83.00–85.01) | 33.81 (31.71–35.92) |
| ±5 frames | X3D-M | 95.72 (95.19–96.26) | 88.96 (87.27–90.65) | 49.46 (45.60–53.32) |
| ±5 frames | Our annotator | 95.59 (95.51–95.64) | 93.50 (93.19–93.63) | 52.47 (51.67–52.90) |
| ±5 frames | MonoTrack | 94.86 (94.75–95.06) | 94.75 (94.59–94.97) | 40.26 (38.71–41.93) |

Available sampling scores (%) at ±10 frames: Stride 3, offset 0

| Run | Sampling | Contact recall | Contact precision | Contact F1 | Raw player F1 | Alternating player F1 | Raw fully correct | Alternating fully correct |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LR 1e-4, seed 0 | Stride 3, offset 0 | 96.86 | 94.57 | 95.70 | 95.13 | 86.19 | 40.79 | 42.26 |
| LR 1e-4, seed 1 | Stride 3, offset 0 | 96.92 | 93.15 | 95.00 | 94.31 | 84.15 | 35.29 | 36.73 |

Available sampling scores (%) at ±5 frames: Stride 3, offset 0

| Run | Sampling | Contact recall | Contact precision | Contact F1 | Raw player F1 | Alternating player F1 | Raw fully correct | Alternating fully correct |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LR 1e-4, seed 0 | Stride 3, offset 0 | 95.47 | 93.22 | 94.33 | 93.94 | 85.01 | 35.11 | 35.92 |
| LR 1e-4, seed 1 | Stride 3, offset 0 | 95.51 | 91.79 | 93.61 | 93.16 | 83.00 | 30.87 | 31.71 |

Validation at stride 3, offset 0, each run at its selected checkpoint, ±10 frames:

| Run | Selected (updates) | Validation loss | Macro F1 | Contact recall (%) | Contact precision (%) | Contact F1 (%) | Raw player F1 (%) | Alternating player F1 (%) | Fully correct rallies (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR 3.125e-5, seed 0 | 55,000 | 0.2053 | 0.7768 | 94.96 | 92.86 | 93.90 | 93.00 | 85.27 | 36.68 |
| LR 3.125e-5, seed 1 | 65,000 | 0.2101 | 0.7770 | 95.07 | 92.44 | 93.73 | 92.82 | 85.60 | 34.13 |
| LR 1e-4, seed 0 | 55,000 | 0.1759 | 0.8038 | 96.35 | 93.97 | 95.15 | 94.16 | 90.43 | 44.46 |
| LR 1e-4, seed 1 | 40,000 | 0.1874 | 0.7976 | 96.45 | 92.54 | 94.46 | 93.32 | 87.78 | 38.77 |
| LR 5e-4, seed 0 | 30,000 | 0.2000 | 0.7884 | 96.31 | 92.26 | 94.24 | 93.13 | 87.94 | 36.38 |

Training time. Recorded hours cover training, validation and event scoring at each check:

| Run | Validation sampling | Selected (updates) | Last check (updates) | Recorded hours to selected | Recorded hours to last check | Training updates per second | Estimated validation minutes per check |
|---|---|---:|---:|---:|---:|---:|---:|
| LR 1e-4, seed 0 | dense (every frame) | 55,000 | 70,000 | 7.55 | 9.55 | 3.60 | 16.6 |
| LR 1e-4, seed 1 | dense (every frame) | 40,000 | 55,000 | 5.59 | 7.58 | 3.60 | 16.6 |
| LR 5e-4, seed 0 | stride 3, offset 0 | 30,000 | 45,000 | 2.94 | 4.37 | 3.63 | 5.6 |
