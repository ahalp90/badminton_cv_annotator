# C2: two decisive traces

## Verdict

Both checks are supported by the saved raw records and the frozen producing
helper. Distances are maximum corner errors in working pixels at 960 by 540;
the corresponding source frames are 1920 by 1080. The control corners and
their approval/source metadata are in `witnesses.json`.

### Check A: GX0 proxy and full-pool directions disagree

The pair-143 proxy is the product of its 512 kept horizontal and 512 kept
vertical assignments. The full-pool row scans every saved per-pair shortlist
before the global cap. The proxy improves after the paint-observation filter,
but the full pool worsens. Both full-pool winners come from pair 22.

| arm / stage | scope | error | pair / candidate | source record |
| --- | --- | ---: | --- | --- |
| B / baseline_generation | pair-143 proxy; axes 2135, 6223 | 34.3268 | 143 / no saved shortlist candidate | `frozen_views/baseline_generation/gxBQ_window_00_frame_0.json.gz` |
| B / baseline_generation | full pre-global pool; axes 2072, 3064 | 7.6621 | 22 / `22:4588` | same as above |
| paint_observations / results | pair-143 proxy; axes 530, 1274 | 12.5775 | 143 / `143:228` | `line_identity/runs/line_identity_20260915_222437/matcher/paint_observations/results/gxBQ_window_00_frame_0.json.gz` |
| paint_observations / results | full pre-global pool; axes 113, 1664 | 9.0360 | 22 / `22:232` | same as above |

The exact corners, scores, shortlist fields, pool counts and input hashes are
in the A witnesses. The changes are therefore `34.3268 -> 12.5775` for the
proxy and `7.6621 -> 9.0360` for the full pre-global pool.

The local filtered replay swaps two adjacent equal-score retained horizontal
IDs (308 and 311) relative to the saved JSON order. The retained ID set,
parameters by ID and enumerated/distinct diagnostics match, so this tie-order
detail does not change the proxy result and is recorded explicitly in the
witness gate.

### Check B: the actual Amateur-3 R pair-43 survivor chain

The horizontal replay checks the saved first 512 IDs, parameters and
diagnostics. Its control-error diagnostic holds the vertical direction at the
saved direction-fit homography. Ranks are zero-based and use descending score
order.

| row | enumeration | assignment tuple | error | score | support / player | ranks (eligible / distinct / kept) |
| --- | ---: | --- | ---: | ---: | --- | --- |
| closest eligible enumerated | 29686 | `(7, 197, 32, 3, 19)` | 4.2783 | 0.657914 | 5 / true | 18256 / — / — |
| its actual representative | 29665 | `(7, 197, 32, 3, 19)` | 8.5405 | 0.784450 | 5 / true | 6798 / 1936 / — |
| closest distinct assignment | 30886 | `(6, 10, 32, 3, 19)` | 6.2409 | 0.581071 | 5 / true | 27748 / 11518 / — |
| closest kept assignment | 18981 | `(7, 132, 41, 254, 172)` | 49.6472 | 0.927052 | 5 / true | 914 / 228 / 228 |

The 4.2783 row is a duplicate of enumeration 29665, which has the higher
score and is consequently the representative. The 6.2409 row is a different
assignment, not that representative. It is also outside the first 512
distinct rows. The closest retained horizontal row is 49.6472 px. The
earlier explanation that duplicate removal retained a lower-scoring 6.2409 px
row is therefore corrected.

## Corrections and handoff

The corrected descriptions are in:

- `line_identity/results.md`
- `line_identity/evidence.md`
- `line_identity/worklog.md`

The runnable check is `check_traces.py`. `witnesses.json` contains the raw
trace identities, reference corners, native/working scale, source paths and
MD5s. `ranking_panel.json` exports 17 distinct automatic winners, the two
observed-bank controls (1864 and 5144), and candidate 89 as a separately
labelled comparator while preserving the original visual rulings.

The uploadable small seed is `../webui_seed/`, with `INDEX.md` and
`../webui_seed.zip`. It contains the witnesses, compact axis/filter tables,
the current reports and the directly relevant frozen/replay source files.
