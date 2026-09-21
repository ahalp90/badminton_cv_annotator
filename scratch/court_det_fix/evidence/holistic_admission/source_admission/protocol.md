# Candidate-admission audit

This is a read-only proposal audit. It does not change W5 or use references during
automatic generation.

## Frozen input and ordering

- Use each W5 source pack's cached `segments_px`, scaled to the W5 working size
  exactly as `w5_holistic/verifier.py::prepare_segments`.
- Build wide families with `detector._wide_line_families` and the inherited
  `max_family_lines=32`, `merge_distance=8`, `merge_angle_deg=3` settings.
- Estimate directions with frozen `vp_pruning.py`, using
  `pencil_selection="coverage"`, `angle_deg=1.5`, 128 directions, 16 pencils,
  overlap 0.8, and the frozen seed.
- Build the complete deterministic coverage order before applying the global
  rectangle cap. Keep the first 4096 selected rectangle identities. No target
  index or reference geometry may affect this order.
- Expand every retained rectangle by all 150
  `detector.TEMPLATE_TRANSFORMS`. Preserve the rectangle identity and template
  index in every proposal ID.
- Apply only inherited positive-depth, finite, convex, visible-span, and area
  validity checks. Reject no proposal because of the detector's old family
  support or supported-line floors.

## Automatic variants

For each proposal, compute two label-free soft-support maps:

1. `family_map`: the existing two wide-family distance maps. Each template
   direction is measured against its inherited family map, but the old hard
   support and line-count floors are removed.
2. `union_map`: one distance map made from every cached working fragment. Every
   template direction is measured against this map, removing broad-family
   assignment as a failure mode.

Rank each map with both cheap summaries:

- `mean_directions`: mean support over the two template direction families.
- `minimum_directions`: minimum support over those two family means.

The four combinations are fixed before any reference or prior-control read.
Within each combination, sort by descending support and stable proposal ID.
Apply deterministic greedy corner diversity with the inherited 12-working-pixel
maximum corner separation. Save retained caps at 32, 64, 128 and 256 from this
single full order.

Before each support ordering, apply the unchanged W5 camera eligibility rule:
`camera_diagnostic.camera` error `<= 0.1`. This is inherited eligibility, not a
new ranking score. Evaluate the frozen camera calculation over its 200-focal
grid in vectorised form. Recheck candidates within `1e-3` of the cutoff with
the scalar helper using float32 native corners, matching W5's homography
reconstruction. Use the scalar value for those frontier candidates in the
saved error and eligibility mask. Record the recheck count and maximum
vector/scalar difference.

## Stop rule and post-hoc inspection

Run GX5 first. Before loading references, test whether any automatic variant's
first 256 diverse proposals contains the approved geometry or a close plausible
proposal. If none does, stop and record admission failure. If one does, run the
same fixed variants on the other eight W5 views, then inspect references and W5
anchors only after all automatic outputs are saved.

Record exact settings, input hashes, counts, timings, proposal provenance,
coordinate semantics, contamination checks, and command exit codes.
