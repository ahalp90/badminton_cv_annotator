# W3 compact packet — L3 temporal pilot

## Finding

The seven cached GX packs are complete and visually support a common sampled pixel view.
The fixed 30-court panel scored validly on every frame. Cross-frame line rankings change,
so observations supply different evidence. The shared median line winner is
`gxBQ_window_00_frame_0::106:93818`, but its locked annotated-corner error median is
799.36 working pixels. Known low-error origin-0 courts are repeatedly displaced by
origin-0 `106:*` and origin-5 `181:*` alternatives. The formal matrix has eight
non-dominated courts. Temporal reuse is therefore conditional on better discrimination.

## Evidence packet

- [result.md](result.md) — bounded interpretation and classification.
- [source_manifest.json](source_manifest.json) — video, frame, image, pack and origin provenance.
- [score_matrix.json](score_matrix.json) — complete line and separate paint vectors.
- [score_matrix.csv](score_matrix.csv) — small tabular view of the same scores.
- [view_alignment.json](view_alignment.json) and `alignment_0_to_*.png` — sampled registration.
- [run_l3_temporal.py](run_l3_temporal.py) — one-process reproducible executor.

Producing basis: `820597ffcc02dd143d16a60448ad4ce53391338b`, with the runner and packet
files dirty at execution time. The final Git commit records the packet.

## One next experiment

Run a reference-blind, shared rank-sum ablation on the existing 30×7 matrix. For every
frame, rank all common courts by `stripe.exclusive.score` and by the existing paint
profile score, using ascending `court_id` for ties. Sum the two ranks per court and
aggregate the seven sums by their mean. Select one shared court. Only then join the
existing annotated references for that court and the four recorded origin winners.

Success means a low-error origin-0 `22:*` court wins without choosing a different court
per frame. Failure means the 106/181 alternatives still win, which would make candidate
generation or cue discrimination the immediate route. This trial uses no new detector
run, no new labels and no continuous reuse guard.

### Local executor prompt

From the repository root, load `score_matrix.json`. Assert that its common subset is the
same 30 courts on all seven declared frames. Recompute per-frame line and paint ranks from
the stored scores only. Apply the fixed rank-sum rule above and write one small JSON/CSV
witness under this directory. Lock that winner before reading reference fields. Report its
seven line/paint vectors, the four origin-winner vectors, and their exact existing
corner-error diagnostics. Do not add candidates, tune weights, select frames, or claim
continuous temporal safety.
