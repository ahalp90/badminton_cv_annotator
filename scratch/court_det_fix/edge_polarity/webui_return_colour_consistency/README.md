# Court boundary versus net tape: bounded handover

This pack hands over the Am1 frame 54 / GX frame 5 audit at repository revision
`13f02fdbf8954ead15dbc7241a696a314473f9c5`.

## Bottom line

The existing court evidence does **not** identify the physical object behind a
bright line response. It can establish a finite, convex, camera-plausible court
whose assigned fragments fit the badminton template, while still interpreting
Am1's net-bottom tape as the far baseline.

The useful new lead is an exploratory **same-paint chroma veto**. It compares the
local Lab colour increment of fragments already assigned to the far baseline
with other transverse court-paint fragments in the same image. With frozen
membership it rejects Am1 strongly, while GX remains admissible because GX has
no retained far-baseline fragments and the rule treats that as untestable rather
than wrong.

The previously proposed outward attached-mesh test was run after the images
became available and failed to distinguish the cases. It is retained in the
pack as a documented negative result, not a recommended feature.

## Recommended reading order

1. [`docs/01_SCOPE_AND_EXECUTIVE_SUMMARY.md`](docs/01_SCOPE_AND_EXECUTIVE_SUMMARY.md)
2. [`docs/02_TRIED_AND_DISCARDED.md`](docs/02_TRIED_AND_DISCARDED.md)
3. [`docs/03_ACCEPTANCE_PATH_AND_IDENTIFIABILITY.md`](docs/03_ACCEPTANCE_PATH_AND_IDENTIFIABILITY.md)
4. [`docs/04_MEANINGFUL_FINDING_SAME_PAINT_CHROMA.md`](docs/04_MEANINGFUL_FINDING_SAME_PAINT_CHROMA.md)
5. [`docs/05_REPRODUCTION.md`](docs/05_REPRODUCTION.md)
6. [`docs/06_NEXT_STEPS.md`](docs/06_NEXT_STEPS.md)
7. [`docs/07_SOURCE_ANCHORS.md`](docs/07_SOURCE_ANCHORS.md)

## One-command rerun

From the pack root:

```bash
bash run_all.sh
```

The scripts need only Python, NumPy and OpenCV. They do not perform candidate
search, alter the frozen parents, change fragment membership, or use reference
corners in any decision.

## Status labels used in the write-up

- **Verified:** reproduced from the frozen packet or directly measured from the
  supplied images.
- **Exploratory:** useful result on these two views, not calibrated as a general
  threshold.
- **Discarded:** tested and shown not to distinguish these cases.
- **Unvalidated:** plausible follow-up that has not been demonstrated here.

No production source was edited.
