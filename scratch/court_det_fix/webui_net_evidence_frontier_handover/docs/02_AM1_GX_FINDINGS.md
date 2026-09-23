# Am1 and GX findings

## Cases

- **Am1 frame 54**: false far baseline follows the lower white band of the net.
- **GX frame 5**: approved partial view with an off-frame court corner and no
  retained fragment assigned to the far baseline.

The frozen two-case packet is included at `data/cases.json.gz` and records exact
parents, fits, assignments, finite intervals, fragment geometry, dimensions,
and completed refits at revision
`13f02fdbf8954ead15dbc7241a696a314473f9c5`.

## Acceptance semantics

The audited path establishes finite projection, positive corner denominators,
convexity, camera plausibility, player containment, and measured image support.
It does not establish that a bright stripe is floor paint rather than net tape.
A full-rank, converged fit can therefore be semantically wrong.

Am1 illustrates both numerical failure and semantic failure:

| Fit | Status | Rank | Condition | Minimum denominator | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| Original labels | Invalid projection | 8 | about 7.48e5 | -0.00135 | geometric failure |
| Polarity-only | Invalid projection | 8 | about 5.00e5 | -0.00648 | geometric failure |
| Strong centres | Converged | 8 | 941.27 | +0.06005 | finite but semantically wrong |

GX demonstrates why frame containment is not a safe replacement. Its strong-
centre fit has a legitimate corner below the 540-pixel working frame.

## Negative result: retained-fragment support for the physical net

A bounded control projected the regulation net implied by each selected court
and measured support using only retained court fragments.

Pre-analysis expectation:

- Am1: low support;
- GX: high support.

Measured mean visible support:

- Am1: approximately `1.94e-12`;
- GX: `0.05418`, with zero support on both projected top-tape halves and only
  weak post support.

This control down-ranks Am1 but cannot certify GX. Retained court fragments are
not a net detector, and legitimate partial views can contain little usable net
support.

## Negative result: outward attached-mesh test

The image-aware attached-mesh formulation looked for repeated short strands
emerging from the outward side of the alleged far baseline.

| Case | Visible span | Attachments | Outcome |
| --- | ---: | ---: | --- |
| Am1 | 344.63 working px | 1 | no contradiction |
| GX | 213.92 working px | 1 | no contradiction |

The compressed and occluded mesh was too weak for this LSD-based formulation.
GX also produced a similar isolated attachment. This is negative evidence for
that particular detector, not for all mesh or texture approaches.

## Exploratory positive result: same-paint chroma

The locally useful distinction came from material consistency rather than net
shape. Each retained fragment received a local Lab colour increment:

```text
ΔLab = Lab(bright ridge centre) - Lab(mean of the two local sides)
```

The far-baseline fragments were compared with other transverse court-marking
fragments while keeping the parent, fit, assignments, and sample membership
fixed.

| Quantity | Am1 | GX |
| --- | ---: | ---: |
| Usable far-baseline fragments | 5 | 0 |
| Usable transverse references | 4 | 14 |
| Boundary median (Δa, Δb) | (6.0, 0.0) | — |
| Reference median (Δa, Δb) | (12.75, 64.0) | — |
| Opponent-colour separation | 64.35 | — |
| Robust reference scale | 10.78 | — |
| Separation / scale | 5.97 | — |
| Local outcome | paint mismatch | abstention |

The Am1 band behaves as nearly neutral white/grey material, while the local
court markings have a strong yellow increment. GX contains no claimed far-
baseline fragments, so the measurement says nothing about that cropped edge.

This is evidence for a broader principle:

> Evidence claimed as an outside boundary can be compared with the material
> system of the court before it is allowed to function as paint.

The same principle could be realised by colour, learned appearance embeddings,
semantic segmentation, temporal texture, or explicit net ownership. The
current numeric thresholds are only an exploratory two-case probe.

## Complementarity with net detection

The chroma result and an independent net detector address different parts of
the ambiguity:

- material consistency can say “this does not resemble the other court paint”;
- net evidence can say “this resembles or belongs to a net”;
- ownership can prevent one fragment from rewarding both explanations.

Am1 is unusually favourable for this complement because its court paint is
yellow and its net band is neutral. White-painted courts remain a harder case.
