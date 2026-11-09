# Selected video-model checkpoints

Download [video-contact-models-20261110.zip](https://github.com/ahalp90/badminton_cv_annotator/releases/download/video-contact-models-20261110/video-contact-models-20261110.zip)
from the [release](https://github.com/ahalp90/badminton_cv_annotator/releases/tag/video-contact-models-20261110).
It contains the five X3D-M and two VideoMAE Base checkpoints tested in the
[completed benchmark](../results/stride3_20261105/).

Both models use positive centres within ±1 frame and 50% contact clips per
batch. X3D uses learning rate 0.02; VideoMAE uses 0.0001. Each checkpoint is
the earliest minimum-validation-cross-entropy checkpoint from its run.
The seeds are evaluated separately.

| File | Selected update |
|---|---:|
| `x3d_m_seed0.pt` | 25,000 |
| `x3d_m_seed1.pt` | 35,000 |
| `x3d_m_seed2.pt` | 30,000 |
| `x3d_m_seed3.pt` | 20,000 |
| `x3d_m_seed4.pt` | 40,000 |
| `videomae_base_seed0.pt` | 30,000 |
| `videomae_base_seed1.pt` | 65,000 |

Each checkpoint contains model weights, training settings, label scheme,
selected update and validation loss. Optimiser state is omitted. Use these
with the [inference command](../README.md#running-the-pilot), a matching RGB
cache and supplied rally bounds.

Removing the old bundled weights changed the original training commit IDs;
the released checkpoint files are unchanged.
