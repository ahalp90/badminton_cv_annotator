# One bounded job — finish the two-pair pixel ownership audit

> **Execution record:** This job completed on 17 September 2026. The committed
> evidence is the unpacked `w2_pair_atlas/` directory. The runner's ZIP is a
> transient convenience file and is not part of the commit. Read `../pickup.md`
> for the current status.

## Decision and scope

Run **one diagnostic export**, not a new detector or score. The fresh CSV defeats a
simple connected-run replacement: it gives the false Am2 court 0.5 and the approved
court 5/11. Do not tune around that result.

Resolve only this question: do the false court's near baseline/long-service tests
borrow one physical ridge, and what supplies the approved court's weaker far-pair
tests? The program exports the pixels needed to answer; it does not assign ownership.

Fixed inputs are at `bb6787adeca9a19b5f1d3c3324d54cfcb2ea6e68`. Exactly two
ORIGINAL `automatic_all_camera` candidates and four intervals receive pixel replay:

| Case | Candidate | Interval indices | Existing ruling |
| --- | --- | --- | --- |
| `am2_window_01_frame_28019` | `184:4123` | 10, 11: near long service / near baseline | False |
| `am2_window_00_frame_150` | `30:33` | 6, 7: far baseline / far long service | Approved |

The script also checks the uploaded 48-row CSV against the already-saved four-candidate
trace. This is arithmetic validation, not an expanded image/candidate experiment.

## Execute without interactive checkpoints

Use the **existing approved host and execution rules**, and the existing NumPy/OpenCV
environment. L2's pinned `STATE.md` records its run on Carmack; use the currently approved
connection, launcher and policy, not an invented host or a bypass. This job needs only CPU.
There is no need to modify the matcher, use a GPU, schedule a large sweep, or install a model.

Use a new output directory under this handover for the committed evidence. From the
checkout, with the approved environment active, run:

```bash
bash /absolute/path/to/w2_fresh_review/run_followup.sh
```

For a launcher whose working directory is elsewhere, set the actual paths once:

```bash
REPO=/absolute/path/to/existing/checkout \
PYTHON=/absolute/path/to/existing/environment/bin/python \
OUT=/absolute/path/to/w2_fresh_review/w2_pair_atlas \
bash /absolute/path/to/w2_fresh_review/run_followup.sh
```

The output directory must be new. Do not reset or checkout repository files. `git show`
reads the pinned objects. A local-file fallback is allowed only when its bytes have the
exact required Git blob hash. The job uses three existing input files: the L2 witness
JSON and the two native Am2 frames. It neither scans history nor reads the giant
candidate pools. All helpers used by this job are in this handover; there are no
runtime imports of repository code.

## Immutable identity and checks

`N = scratch/court_det_fix/next_steps_20260916`.

| Input | Required identity |
| --- | --- |
| `N/webui_seed/L2_scoring/witnesses.json` | Git blob `739af3d1dacc7f3fedb551eab01a2b4bad94fd06` |
| `N/webui_seed/frames/amateur/am2/frame_00028019.png` | Git blob `a6c1ace2fad3a6f350b7b91fad991840955fd866` |
| `N/webui_seed/frames/amateur/am2/frame_00000150.png` | Git blob `08ab1d43d219320bc18d2bfb25fbde5f5f3cf456` |
| Included `inputs/real_interval_probe.csv` | SHA-256 `41fd27c0387d5dc9864bde9cd0031a04b8669c37698fca7e5173aaaa15644e61` |

The job records HEAD, tracked dirty-patch identity, actual input and helper hashes,
local worktree input identities, and dependency versions. It checks all 48 table rows,
then reproduces availability and both-side contrasts for the four target intervals.
The numerical contrast-replay tolerance is `1e-4`; passing-offset masks must agree
exactly. This is a reproduction check, not a court-quality threshold.

A failed input or numerical check writes `manifest.json` with `BLOCKED`; return that
file and the command error. Do not silently substitute inputs, widen tolerances, rerun
extraction, or invent a scientific conclusion. No further interactive approval is
needed to execute the specified job.

## Outputs and readout

The runner creates **`return_pack.zip`** as a transient convenience bundle. The committed
evidence is the unpacked directory, which contains one `atlas.html` with two sections,
full working-view context, native raw/overlay crops, shared-image-chart strips, the exact
paired traces, small coordinate/intensity arrays, and a manifest.
All available tests remain distinct from unavailable ones. Shared strips align image
coordinates, not station indices; interpolation is display only, not added resolution.

No semantic interpretation is required of Luna Max to complete the job: run the script
and return the bundle. If image inspection is available, add at most 250 words covering
both pairs: which physical structure supplies the tests; whether two nominal identities
have distinct ridges, one shared ridge, or unresolved support; and what can actually be
seen at the approved pair. Use the unannotated pixels before the overlays. Unknown is a
valid answer. Do not claim that a passing point, projected track, or diagram label proves
court-paint ownership.

The single changed component is the **diagnostic export**. No H, extraction, assignments,
population, score, winner, acceptance rule, or visual ruling changes. No new cases or
full-pool extension belong to this job. Stop after returning the bundle; do not auto-launch
another experiment. Both GX0 controls and comparator remain outside the automatic pools.
