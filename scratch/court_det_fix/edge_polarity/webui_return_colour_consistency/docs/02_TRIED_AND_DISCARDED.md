# Tried and discarded, or useful only as diagnosis

This file is intentionally concise. These ideas should not be rediscovered as
if they had not already been checked.

## 1. Require every court corner to be inside the image — discarded

GX's strong-centre fit has a legitimate corner at working `y = 580.55` in a
540-pixel-high image. The detector clips finite projected markings to their
visible image portions. Cropping is therefore not evidence of a wrong court.

## 2. Optimiser convergence, rank or condition number as the semantic gate — discarded

Am1 shows two distinct phenomena:

- original-label fit: invalid projection, rank 8, condition about `7.48e5`;
- polarity-only fit: invalid projection, rank 8, condition about `5.00e5`;
- strong-centre fit: converged, rank 8, condition `941.27`, positive minimum
  denominator `0.06005`, finite convex corners, but the far baseline is net tape.

Conditioning is useful for diagnosing unstable geometry. It does not decide what
physical object generated a well-fitted fragment.

## 3. Greyscale polarity or ridge contrast alone — insufficient by construction

Polarity identifies which side of a fragment is brighter. The existing ridge
measurement identifies a bright centre with darker pixels on both sides. A
paint stripe and white net tape can share both measurements.

## 4. Project support for the camera model's physical net — not a usable gate

Pre-run prediction:

- Am1: low retained-fragment support;
- GX: high retained-fragment support.

Measured mean visible support:

- Am1: approximately `1.94e-12`;
- GX: `0.05418`, with zero support on both projected top-tape halves and only
  weak post support.

This does not certify GX. It asks whether a separately predicted physical net is
represented by the retained court fragments, which is not the same as deciding
whether the disputed line is net tape.

## 5. Outward attached-mesh contradiction — tested and discarded

Predeclared rule:

- test the outward side of the fitted far boundary;
- ignore endpoint 5%;
- qualifying strand touches within 5 working pixels, extends 12–45 pixels
  outward and lies within 15 degrees of the outward normal;
- reject only for at least three attachments spanning at least 25% of the line.

Measured:

| Case | Visible span | Attachments | Decision |
|---|---:|---:|---|
| Am1 | `344.63 px` | 1 | no contradiction |
| GX | `213.92 px` | 1 | no contradiction |

The compressed/occluded mesh is too weak for this LSD measurement, and GX
produces a similar isolated attachment. Do not carry this forward as a gate.

## 6. Missing Am1 assignments as a rejection rule — only a lead

Am1's retained membership contains no right sideline or near baseline. That is
consistent with weak depth leverage, but it is not a proved causal explanation
and cannot safely reject partial courts.

## 7. Paired 40 mm edges as object identity — diagnostic only

Paint-edge consistency helps localise a stripe, but a net band may also have two
visible edges. It does not by itself establish that the stripe lies on the floor.
