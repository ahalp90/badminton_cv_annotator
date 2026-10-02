# Meaningful finding: same-paint chroma veto

## Status

**Exploratory but directly effective on the frozen failure.** It is suitable as
a diagnostic or veto candidate for broader validation, not yet as a calibrated
production acceptance threshold.

## Idea

Once the fitter claims specific fragments as a far baseline, require those
fragments to resemble the same physical paint system as other transverse court
markings in that image.

This is different from the existing greyscale ridge measurement. The ridge test
asks whether a centre is brighter than both sides. The chroma test asks whether
the bright material has the same local colour increment as known court paint.

## Fixed inputs

The check changes none of the following:

- source parent;
- fitted court;
- retained raw fragments;
- fragment-to-marking membership;
- sample membership;
- reference or approved corners.

It does not use case IDs in its decision.

## Per-fragment measurement

For each retained fragment:

1. Sample 24 positions along the fragment.
2. At each position, search normal offsets `-4, -2, 0, 2, 4` working pixels.
3. Choose the offset maximizing the existing two-sided greyscale ridge response.
4. Keep positions with ridge contrast at least 10.
5. At the winning centre, convert centre and two side samples to CIE Lab.
6. Compute

```text
ΔLab = Lab(centre) - Lab(mean(side_minus, side_plus)).
```

7. Use the fragment median `(Δa, Δb)` as its opponent-colour signature.

Subtracting the local side colour reduces sensitivity to the underlying floor
and broad illumination changes.

## Groups and one-sided decision

- Boundary group: usable fragments already assigned to interval 6,
  `far_baseline`.
- Reference group: usable fragments assigned to other transverse intervals
  7–11.
- A fragment is usable when at least eight of 24 samples pass the ridge floor.
- Require at least three fragments in both groups.

If the boundary group is absent, return `untestable_no_supported_far_boundary`
and **do not reject**. This preserves partial/cropped courts such as GX.

Let `b` be the median boundary `(Δa, Δb)` and `r` the median reference value.
Define

```text
D = ||b - r||₂
s = 1.4826 * median_i ||r_i - r||₂
```

The bounded demonstration rejects when

```text
D > 20  and  D > 4 * s.
```

These constants are engineering probes, not calibrated thresholds.

## Measured result

| Quantity | Am1 | GX |
|---|---:|---:|
| Usable far-baseline fragments | 5 | 0 |
| Usable transverse references | 4 | 14 |
| Boundary fragment IDs | `158, 303, 88, 173, 117` | none |
| Reference fragment IDs | `334, 474, 389, 503` | not needed |
| Boundary median `(Δa, Δb)` | `(6.0, 0.0)` | — |
| Reference median `(Δa, Δb)` | `(12.75, 64.0)` | — |
| `D` | `64.35497` | — |
| `s` | `10.77952` | — |
| `D / s` | `5.97011` | — |
| Decision | `reject_paint_mismatch` | `no_rejection`, untestable boundary |

The Am1 boundary fragments are almost neutral relative to their sides, matching
white/grey net tape. The transverse court references have a strong yellow
increment. GX has no retained fragments assigned to the far baseline, so the
check correctly makes no claim about that cropped boundary.

## Why this is more meaningful than the discarded tests

- It tests material consistency, not just line geometry.
- It does not require all corners or the complete baseline to be visible.
- It operates only when the candidate itself claims boundary evidence.
- It cannot turn absence of evidence into evidence of failure.
- It is local, inexpensive and post-fit.

## Suggested integration point

The least invasive integration is after stripe assignments are available in
`verifier.measure_candidate` / `physical_marking_evidence`:

1. compute per-fragment signatures from `context.frame` and
   `context.observations.segments`;
2. retain the assignment indices from `stripe_score["assignments"]`;
3. emit a structured diagnostic such as:

```text
boundary_paint_consistency:
  status: evaluated | untestable_no_supported_far_boundary |
          untestable_no_reference | no_contradiction | reject_paint_mismatch
  boundary_fragment_count: ...
  reference_fragment_count: ...
  boundary_median_delta_ab: ...
  reference_median_delta_ab: ...
  separation: ...
  robust_scale: ...
```

Initially keep it out of ranking and use it for gallery measurement. If it
survives calibration, use only the explicit `reject_paint_mismatch` state as a
veto; never reject an untestable partial view.

## Known limitations

- White court paint and white net tape may be indistinguishable by chroma.
- Courts can use multiple paint colours or have repairs and severe fading.
- Specular highlights and coloured lighting can distort Lab increments.
- The current reference uses only other transverse markings to reduce
  orientation/illumination confounds; some views may not have enough of them.
- Two cases cannot calibrate the ridge floor, fragment minimum, colour distance
  or robust-scale multiplier.
- The measurement was formulated after seeing these images, so the result is
  not held-out validation.

The correct claim is therefore: **promising bounded veto, not solved general
classification.**
