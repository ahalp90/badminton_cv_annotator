# Draft GitHub issue for qaz812345/TrackNetV3

Paste-ready below the line. Everything quoted was verified against
upstream master 2026-07-22: the inference path (first-hand fetch),
the full no-mask-output sweep and the training-regime reads (two
independent reviews, with load-bearing `train.py` quotes re-fetched
first-hand).

---

Title: InpaintNet fills gaps longer than its window with a fixed
invented pattern, and the output gives no way to filter it

Hi, thanks for TrackNetV3! We've been using it with the released
checkpoints to track shuttles in long broadcast videos, and it has
worked well. One thing seems worth reporting.

**The short version.** When a tracking gap spans a whole InpaintNet
window, the model gets no evidence at all: its input is all-zero
coordinates plus an all-ones mask. Its output is then fixed by the
trained weights alone, so every such window, in every video, gets
the same 16 invented positions. `predict()` marks them
`Visibility = 1`, and the inference path never outputs the
`Inpaint_Mask`, so downstream code cannot tell them from real
detections.

**Why it happens.** InpaintNet is trained as an interpolator. The
training masks are random per-frame dropout: each frame is masked
independently with probability 0.3 (`get_random_mask`,
`np.random.binomial(1, mask_ratio, ...)` in train.py), and the loss
is computed only at masked positions. Under that recipe a fully
masked 16-frame window almost never occurs (0.3^16, about 4 in a
billion), so the model never learns a "no evidence" behaviour. At
inference, `generate_inpaint_mask` marks whole gaps for filling
with no length cap; it only checks the y-coordinate of the two
frames flanking the gap. A long gap therefore hands the model
exactly the input it never saw in training, and the output is
whatever the weights default to.

**You can see it without a video:**

```python
import torch
from model import InpaintNet

ckpt = torch.load('ckpts/InpaintNet_best.pt', map_location='cpu')
seq_len = ckpt['param_dict']['seq_len']

net = InpaintNet()
net.load_state_dict(ckpt['model'])
net.eval()

coords = torch.zeros(1, seq_len, 2)   # a window inside a long gap
mask = torch.ones(1, seq_len, 1)
with torch.no_grad():
    out = net(coords, mask)[0]

print('x px:', [int(v * 512) for v in out[:, 0]])
print('y rows:', [int(v * 288) for v in out[:, 1]])
```

With the released checkpoint this prints, give or take a pixel of
CPU-vs-GPU rounding:

```
x px:   [240, 244, 244, 245, 242, 245, 244, 244, 245, 246, 242, 240, 241, 241, 241, 247]
y rows: [81, 73, 70, 69, 67, 68, 69, 70, 68, 68, 71, 73, 77, 80, 83, 84]
```

That little bobbing loop is exactly what our saved tracks contain
wherever a whole window had no detection. In our three broadcast
videos it covers about a third of all frames, much of it inter-rally
footage where there is genuinely no shuttle to track, yet the CSV
reports a visible shuttle bobbing mid-frame.

One mode note. We first ran `--eval_mode nonoverlap`. There windows
tile, so the loop repeats verbatim. We have since measured the default
`weight` mode on the same video, and the expectation from the code
held: the overlapping blend collapses the cycle to a single
near-constant position.

Concretely, on a 154,393-frame broadcast that constant sits at x=243,
y=71 in the 512x288 tracking grid, and it occupies 35.8% of frames
across 591 separate runs. It is predictable from the checkpoint alone.
Applying the `weight` ensemble's tent weights to the 16 cycle values
above gives a weighted mean of x=243.4, y=70.9 pixels. Truncation puts
the saved value inside a two-pixel interval per axis, and the observed
point falls inside it.

That makes suggestion 2 below matter more in the default mode than in
ours. Under `nonoverlap` the repeating loop is self-evident, so a user
can find and filter it from the CSV alone. Under `weight` the same
invented content becomes one constant position. That is also what a
genuinely stationary detection looks like, so no post-hoc filter can
separate the two. The snippet above is mode-independent.

**Three suggestions.** Any of them would help:

1. Skip filling where there is nothing to interpolate from: for
   example, leave a gap unfilled once it spans a full window, or
   skip windows containing no real detection. Interpolation is not
   defined there anyway.
2. Emit the inpaint mask at inference, as a CSV column or a sidecar
   file, so users can apply their own policy. As shipped it never
   reaches the output: the only `save_inpaint_mask` path is the
   ground-truth-dependent training-data writer.
3. If retraining is ever on the table, the realistic masks already
   exist in the pipeline: `generate_mask_data.py` builds gap-shaped
   masks from real TrackNet misses, and the dataset loads them, but
   `train_inpaintnet` binds them to `_` and draws the random masks
   instead (they are only used at validation, for checkpoint
   selection). Mixing the real masks into training would show the
   model contiguous gaps of the shape inference actually asks
   about.

Happy to open a PR for the first two. And to be clear, the fill behaviour
on short occlusion gaps is genuinely useful; this is only about the
gaps the model has no evidence for. Thanks again for the project!
