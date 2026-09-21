# Cached replay for directional admission experiments

## Bottom line

Build the line-template hypothesis bank once per case, immediately before the
directional visibility floor. Save every valid hypothesis, not only the 256 that
survive selection. Each experiment arm can then apply its own floor, reorder and
refill the pool, and run the ordinary W5 stages.

This removes the repeated vanishing-point, rectangle, projection, support,
visibility and admission-camera work. It does not reuse arm-specific candidate
selection or evidence. The current four-arm sweep remains the reference run;
the cache should be implemented and checked against it before it drives another
decision.

## Safe split

The split sits inside
[`line_template_source.generate`](line_template_source.py), just before
`select_with_visibility_floor` applies the floor and the 256-candidate cap.

```text
cached once per case
    line families and vanishing points
      -> rectangles and projected court templates
      -> geometry, support and visible-piece counts
      -> admission-camera checks
      -> complete pre-floor hypothesis bank

replayed for every arm
    visibility floor
      -> score order, 256 cap, diversity and refill
      -> selected records and W5 gates
      -> source union and duplicate handling
      -> W5 measurement, refit and children
      -> final ranking and report
```

The bank must include hypotheses rejected by each tested floor. A saved W5 run
cannot serve as this cache because it contains only the selected population.

## Cache contents

For every valid pre-floor hypothesis, preserve the original row order and save:

- working-image corners;
- lengthwise and cross-court support means;
- the derived admission score;
- lengthwise and cross-court visible-piece counts;
- final admission-camera error and eligibility after frontier rechecks;
- rectangle ID, rectangle order and template index;
- the stable proposal ID, or enough ordered identity data to reproduce it.

Keep raw working-image corners and let the existing materialisation path rebuild
native corners and the working homography. Do not introduce a second homography
implementation at the cache boundary.

Store arrays without rounding, downcasting or other conversion, and reject an
unexpected data type on load. Duplicate handling later compares exact
homography bytes, so a seemingly harmless float conversion could change which
G0, G1 and line-template records collapse together.

The builder must also return the complete floor-independent source metadata
that the comparator already treats as stable. This includes rectangle counts,
camera frontier recheck counts, estimator details, retained-point counts and
the number of rectangle-template hypotheses. Store that dictionary verbatim as
JSON beside the arrays. Its ordering block includes the selected rectangle-ID
sequence. Empty banks still need their source metadata and exact reason, such as
`no_coverage_rectangles`.

## Provenance and invalidation

Write the bank and one small manifest atomically. Derive one cache key from:

- the schema version and case ID;
- one source-pack digest;
- one producer digest covering the pre-floor builder, `detector.py`,
  `assignment.py`, the imported `vp_pruning.py` and the legacy camera helper;
- external runtime settings, NumPy and OpenCV versions, and the single-thread
  setting.

The manifest also records working and native dimensions, array shapes and data
types so the loader can reject a malformed bank. Keep the component paths and
digests in this one manifest for diagnosis. Arm packets and reports record only
the resulting cache key. Do not repeat component hashes or add per-array hashes;
the behavioural equivalence gate is the evidence that the cache preserved the
experiment.

Store the bank outside individual arm run directories. Each arm must record the
same cache key in its line-template metadata. The comparator must reject arms
with different keys.

Invalidate the cache after any change to fragments, line-family preparation,
vanishing-point or rectangle ordering, court templates, support sampling,
geometry validity or admission-camera logic. A floor-only change should reuse
it. The cache does not apply to the older direction-agreement arms because
those arms change anchors and line allocation before this boundary.

The admission-camera value in the bank is not the later W5 camera gate.
Preserve it under an explicit name such as
`line_template.camera_error_before_w5_gates`; the ordinary W5 path may compute
and attach its own gate result for selected candidates.

## Small implementation

1. Split `line_template_source.generate` into a pure pre-floor builder and a
   selector/materialiser that consumes its result. Keep `generate` as the
   compatibility wrapper.
2. Add atomic save and load helpers for the pre-floor bank and manifest.
3. Let `run_w5.load_populations` and the case runner accept an optional cache
   root. Pass the same root through preflight, so preflight does not rebuild the
   bank once per arm. Record the resolved cache key in every run.
4. Make `compare_directional_runs.py` require one cache key across compared
   arms when cached mode is used.

The focused unit test should show that the compatibility wrapper returns the
same entries and floor-independent metadata as the split builder and selector
on a small synthetic context. The existing refill test already covers the
central selection rule. Add one comparator test that deliberately supplies
different cache keys and expects a clear failure.

Before using the cache for a scientific result, run current code both directly
and through the cache on the same real case. Test `(0,0)` and one non-zero floor
that actually refills the pool. Diff the full case record and its
`line_template_sources` entry, ignoring only elapsed time and cache metadata.
This same-code comparison avoids mistaking later code drift for a cache defect.
