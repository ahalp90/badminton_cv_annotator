# Critical read of the SVD returns — 23 September 2026

**The 12-family screen merits a bounded automatic-candidate retention check
after the fitting decision. It is not yet a validated detector optimisation.**
The first return supplies a reproducible screening result. The supplement
supplies useful negative comparisons and an untested fitting proposal.
Neither establishes a broader meaningful candidate pool or a wall-clock gain.

## What the two packages establish

| Package | Checked result | Consequence |
| --- | --- | --- |
| [Fixed-B screen](received/followup3_svd_screen/return.md) | Keeping 12 of 16 direction families preserves the exact best cached reference-fit pair in 9/9 development views. Pair launches fall from 240 to 132, a 45% reduction. Keeping 8 or 10 preserves only 2/9 or 3/9 | Test the 12-family screen against automatic candidates; reject the two aggressive budgets on this evidence |
| [Supplement](received/svd_court_evaluation/README.md) | At 12 families, support-only ranking retains 8/9 best pairs, singular-gap ranking 2/9, and the smallest-to-middle singular-value ratio 4/9 | These alternatives do not improve the preservation result. Do not substitute one as an assumed remedy |

Both packages use the same nine frozen views and cached direction-fit records.
The supplement is further analysis of those inputs, not independent validation.
The main prompt prescribed 12 as the primary budget with sensitivity checks.
The result also happens to make 12 the smallest successful tested budget.
There are six visually approved controls and three manual references; all are
previously used development views.

## Scientific interpretation

**The strongest limitation is what “best fit” means.** The producer's
[`control_fit`](../../frozen_helpers_20260914/automatic_axes/diagnose_directions.py)
fixes two vanishing directions, then fits the remaining parameters directly to
reference corners. Its docstring explicitly says it does not use observed
lines. The screen preserves the best such pair. It has not demonstrated that
the automatic matcher finds a good court using that pair, retains all useful
alternatives, or ranks one correctly. GX5's retained best diagnostic error is
still 23.800169 working pixels. A zero degradation here is not a clean fit.

**The screen is tied to the current normalisation.** Group ranking uses SVD
residuals, support counts and original group IDs, without reference errors.
Equivalent coordinate descriptions preserve the ranking when the saved
normalisation map is transformed consistently. Changing the metric itself
can change admissions: translating the normalised frame loses two best pairs
at budget 12. On SS03-19, the best surviving diagnostic error then rises from
2.268884 to 142.287745 pixels. This is sensitivity to the chosen algebraic
metric, not a discovered coordinate-transform bug. Preserve the existing
centred, image-diagonal normalisation in any trial; do not describe this as a
coordinate-independent measure of court relevance.

**The supplement's preferred engineering approach remains a hypothesis.** It
proposes fitting each distinct assignment from all assigned observations before
choosing its representative and applying the cap. That targets a real loss
mechanism: the retained
[`match_axis`](../../frozen_helpers_20260914/axis_matching/projective_seed.py)
sorts by score, keeps the first occurrence of each assignment, then caps the
list. The documented Amateur-3 R example contains a 4.2783-pixel enumeration,
an 8.5405-pixel duplicate representative and a 49.6472-pixel closest retained
result. Those are a different arm from the baseline-B screen.

The proposal could change both geometry and ordering. The returned code does
not implement it or establish safe assignment equivalence, useful-candidate
retention, or runtime. Its later joint homography refit cannot recover an
assignment discarded earlier. Keep this as a separate conditional lead;
do not turn the supplementary recommendation into a detector redesign now.

The SVD-direction refit figures in the supplement are also old cached
reference-fit potentials. Eight improvements in nine views are not eight
new automatic detection improvements. The roughly 0.7 ms median SVD cost is
an old producer timing, not a new end-to-end benchmark.

## Next bounded check

**Later update, 23 September:** the saved automatic check below has completed.
The [retention findings](review_20260923/automatic_retention/README.md) preserve
this assessment and add today's visual context. All eight historically
approved automatic fits survive the 12-family mask. Three score-winner roles
are lost. Proceed to an actual 12-versus-16 matcher comparison, retaining the
full-source comparator and G0. Runtime remains unmeasured.

Keep fitting first, then SVD/search coverage as recorded in pickup. Begin with
saved automatic candidate records for the same baseline directions. Check
input identity and original group IDs before mapping the 12-family mask onto
those records. Account for useful candidates and full-pool selections, not
only the best reference-fit pair or a one-pair proxy. Keep the 16-family
comparator and the existing G0 fallback.

If that check survives, compare actual matcher work and runtime for 12 versus
16 families. Some omitted pairs may already be cheap or rejected, so 45% fewer
possible pair launches does not imply 45% less runtime. Only then assess
whether spending the saved work on broader search improves coverage at equal
cost. Keep screen pruning, changed directions, assignment fitting and relaxed
caps as separate changes. No matcher trial or detector change was made here.

## Verification and limits

- All 27 numerical inputs and the SVD producer match their Git blobs at
  `b47eee0ff43ec7cbebcc2d84eb9e5e98075d4eea` byte-for-byte
- The unchanged main runner exits 0 locally. All group orders, budget outcomes
  and aggregate rows match the return exactly
- Both unchanged supplementary scripts exit 0 in their documented 27-file
  bundle layout. Both JSON outputs match exactly, including 5,746 numeric values
- A direct full-checkout layout first failed with `KeyError: sets`: its E3
  directory also contains an aggregate record. Repeating with the documented
  nine-case input layout resolves this without editing the supplied scripts
- The original prompt requested retention within one pixel of the best fit.
  The main script instead counts numerical equivalence within 1e-8 pixels.
  A direct local check fills that omission: every case has only one converged
  pair within one pixel, and the 12-family screen retains it. This adds no
  evidence of preserving a diverse collection of useful pairs
- Actual support sizes range from 3 to 42 lines. The two-line zero-residual
  example is a synthetic warning, not an observed two-line failure in these inputs
- The frozen cap-loss note in the supplement matches the local source note.
  Its representative/cap ordering was checked against the retained matcher
- No images, new fits, matcher search or production runtime were evaluated.
  The full eight-test suite was not rerun; the main numerical path and both
  supplementary calculations were replayed directly. A bounded Luna Max
  mechanics check also passed the synthetic failure/convergence accounting
  test and inspected SS03-17, drawn with seed 23 from the other eight cases.
  The scientific interpretation and source checks above were assessed directly
  by the coordinator

The [check receipt](review_20260923/checks.json.gz) records provenance,
comparisons and the extra one-pixel counts. The [main replay](review_20260923/replay/)
and supplementary outputs beside it preserve the local results. Original
archives and all 17 extracted files remain unchanged.

From the repository root, rerun the main calculation into a new output folder:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ~/.venvs/badminton-cicd/bin/python \
  scratch/court_det_fix/evidence/webui_followup3_20260922/received/followup3_svd_screen/run_screen.py \
  --inputs . --output /tmp/court-svd-screen-new
```

For the supplement, copy its two scripts into a fresh working folder and
provide only the 27 named inputs under its `task3_inputs/` layout. Follow the
supplied README there so its output writes do not overwrite the filed return.
