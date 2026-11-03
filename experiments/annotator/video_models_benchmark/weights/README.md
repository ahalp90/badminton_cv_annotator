# Selected X3D-M checkpoints

Both models use scheme A, 50% contact sampling and peak learning rate 0.02.
Each checkpoint was selected by minimum validation cross-entropy across its
original training and continuation.

| File | Seed | Selected update | Validation CE |
|---|---:|---:|---:|
| `x3d_m_A_contact50_lr0.02_seed0.pt` | 0 | 25,000 | 0.133007 |
| `x3d_m_A_contact50_lr0.02_seed1.pt` | 1 | 35,000 | 0.135657 |

Each file is 11.67 MiB and contains the model state, training settings, label
scheme, selected update and validation loss. Optimiser state is omitted.
Use either file with the [inference command](../README.md#running-the-pilot).
The [training records](../results/x3d_pilot/) include all winning and non-winning runs.
