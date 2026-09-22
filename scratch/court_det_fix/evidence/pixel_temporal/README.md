# Pixel evidence and temporal scoring

A bright-line match can have the wrong physical source. The completed pixel
review demonstrated this on a net band, but also found borrowed support on
an approved court. Simple coherence or shared-pixel vetoes therefore failed
their contrary-example test. The temporal rank-sum calculation is also
complete; a broader independent-versus-shared scoring comparison remains open.

## What the pixels established

The W2 review examined false Amateur-2 frame-28019 candidate `184:4123`
and approved frame-150 candidate `30:33`. Its recorded visual readout
attributes the false pair's long shared support to a raised pale net band,
not two floor markings. The approved pair also borrows support from a white
sock and crossing sideline. Its far-boundary strip remains unresolved.

Two passing tests on the approved pair use the same twelve source pixels
with different interpolation weights. That is a direct counterexample to
rejecting a court merely because expected markings share pixels.

| Candidate and interval | Passing stations | Longest connected run |
| --- | ---: | ---: |
| False `184:4123`, interval 10 | 18/24 | 16 |
| False `184:4123`, interval 11 | 18/24 | 12 |
| Approved `30:33`, interval 6 | 13/24 | 4 |
| Approved `30:33`, interval 7 | 10/24 | 3 |

The false pair is more coherent because the net band is coherent. The
review closes this four-interval diagnostic, not court acceptance in
general. It does not support another full-pool connected-run experiment
without a new reason to expect useful discrimination.

Open the [pixel atlas](w2/evidence/atlas.html), [visual readout](w2/results/visual_readout.json)
and [numerical checks](w2/results/verification.json). Raw crops, overlays,
coordinates and interpolation footprints remain beside them.

Reproduction has two important boundaries:

- Direct bilinear sampling reproduces the 480 saved offset pass flags.
  The review's installed OpenCV remap changed one borderline flag; that local
  call must not silently replace the recorded-runtime result
- The returned manifest records executed atlas-builder SHA-256
  `80c0e9c0b97135e08bef4d9f03da2b38b73ddc5b59526d3e2e44800e1a091252`.
  The available [source version](w2/producer_source/build_pair_atlas.py)
  hashes to `7200d56010292d15f176fc96d2a9f004b38ea4d52941556456b1499ef1cb848b`.
  The executed source has not been recovered in this local check. Preserve
  that gap; it does not invalidate the returned pixels or their verifier

The atlas-builder checksum mismatch predates the cleanup. Treat it as a
low-priority provenance footnote, not an open research task or W5 blocker.
Do not spend a fresh session hunting the matching script unless exact export
reproduction is needed or a concrete discrepancy appears in the saved checks.

## Temporal result: complete, still not a usable selector

The L3 pilot scored 30 saved courts on seven GX frames, giving 210
measurements. Shared median-line scoring selected `106:93818` with median
saved corner error 799.356692 working pixels.

W4 ranked line and paint scores separately per frame, added those ranks and
selected the lowest average. It chose frame-0 `22:4579`: total 119,
average 17, median saved corner error 14.068941 pixels. That court was
already visually judged skewed. The calculation found a better saved option;
it did not establish a correct temporal selector.

Keep the [original score matrix](../../next_steps_20260916/L3_temporal/score_matrix.csv),
[rank-sum result](w4/results/l3_rank_sum.json), [ranks](w4/results/l3_ranks.csv)
and [selection lock](w4/results/selection_lock.json). The lock separates
score-based choice from the later reference-error comparison.

## Other completed diagnostics

| Evidence | Finding and limit |
| --- | --- |
| [GX0 row crops](diagnostics/gx0_rows/) | Rows 31, 51 and 61 were visually attributed to wall cladding, a floor seam and a spectator's face. Attribution alone does not prove why allocation chose them. |
| [Person-mask replay](diagnostics/person_mask_replay/) | Two masks both remove the face fragment. Only the midpoint-in-box rule substantially improves the control fit, from 6.798 to 1.498 working pixels. This shows changed allocation, not the isolated cause. |
| [Later GX directions](diagnostics/gx_directions/) | Some later frames have low incidence bounds against GX0's control. A bound is not an attained fit; camera equivalence and generated courts were not checked. |
| [Identity diagnostics](diagnostics/identity_diagnostics/) | Wrong courts can retain full constraint rank. The false Am2 pair is separated by less than half a working pixel. Rank and spacing are useful warnings, not universal rejection thresholds. |
| [Fragment bow](diagnostics/distortion/) and [ridge check](diagnostics/distortion_ridge/) | Both failed to establish a validated GX-specific distortion bound. The ridge method jumped between features on Amateur-3 and exceeded its clean-control threshold on ShuttleSet. Distortion is deferred, not disproved. |

Full diagnostic tables, crops, inputs and scripts remain available. The
[direction-search note](../direction_search/README.md) owns the separate
cap and pre-gate results and raw arrays.

## Unfinished comparisons

Independent versus shared scoring still needs comparison on the same
verified automatic candidate union. Separate native independent selection,
independent scoring on the union, and shared scoring. Keep candidate-access
arms distinct and retain Amateur-3 as a regression control. W4's rank sum
does not answer that question.

Multi-frame generation, camera-change detection and safe court reuse remain
separate. Nearby cached frames are not independent quality evidence.
The expanded [G0/G1 assessment](../g0_g1/README.md) and corrected five-case
person-observation comparison also remain unfinished.

## Rerun the recorded checks

From the repository root, write fresh outputs outside the retained evidence:

```bash
~/.venvs/badminton-cicd/bin/python \
  scratch/court_det_fix/evidence/pixel_temporal/w2/verify_return.py \
  --evidence scratch/court_det_fix/evidence/pixel_temporal/w2/evidence \
  --csv scratch/court_det_fix/evidence/pixel_temporal/w2/inputs/real_interval_probe.csv \
  --out /tmp/court-w2-check

python3 scratch/court_det_fix/evidence/pixel_temporal/w4/review_checks.py \
  --l3-csv scratch/court_det_fix/next_steps_20260916/L3_temporal/score_matrix.csv \
  --output /tmp/court-w4-check
```

W2 needs NumPy and OpenCV; W4 uses the standard library. Neither runs a model
or contacts Carmack. Original review wording is preserved in the tidy backup
under `worklog/webui_evaluation_returns_15092026/` and
`worklog/webui_further_followups_16092026/`.
