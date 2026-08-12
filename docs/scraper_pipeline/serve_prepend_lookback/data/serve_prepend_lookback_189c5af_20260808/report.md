# Serve-lookback candidate measurement

Generated: 2026-08-08T14:20:33.953385+00:00

Fixture profile: `une-189c5af-static-stride8` from source commit `189c5af58e45d23ae827dde516924194eb238e18`.

This is a recording-only measurement. It does not change production output.

## Results

| Pose band | Arm | Target misses | Selected | Target recovered | False positives | Precision | Recall | All unmatched recovered |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| middle-half | evidence_only | 136 | 14 | 0 | 14 | 0.000 | 0.000 | 0 |
| middle-half | current_mask_policy | 136 | 0 | 0 | 0 | unavailable | 0.000 | 0 |
| middle-two-thirds | evidence_only | 136 | 14 | 0 | 14 | 0.000 | 0.000 | 0 |
| middle-two-thirds | current_mask_policy | 136 | 0 | 0 | 0 | unavailable | 0.000 | 0 |

## Per-video results

| Video | Pose band | Arm | Target misses | Target recovered | False positives | All unmatched recovered |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| sset_01 | middle-half | evidence_only | 63 | 0 | 3 | 0 |
| sset_01 | middle-half | current_mask_policy | 63 | 0 | 0 | 0 |
| sset_01 | middle-two-thirds | evidence_only | 63 | 0 | 3 | 0 |
| sset_01 | middle-two-thirds | current_mask_policy | 63 | 0 | 0 | 0 |
| sset_15 | middle-half | evidence_only | 39 | 0 | 6 | 0 |
| sset_15 | middle-half | current_mask_policy | 39 | 0 | 0 | 0 |
| sset_15 | middle-two-thirds | evidence_only | 39 | 0 | 6 | 0 |
| sset_15 | middle-two-thirds | current_mask_policy | 39 | 0 | 0 | 0 |
| sset_21 | middle-half | evidence_only | 34 | 0 | 5 | 0 |
| sset_21 | middle-half | current_mask_policy | 34 | 0 | 0 | 0 |
| sset_21 | middle-two-thirds | evidence_only | 34 | 0 | 5 | 0 |
| sset_21 | middle-two-thirds | current_mask_policy | 34 | 0 | 0 | 0 |

## Interpretation guardrails

- A true positive is a selected candidate within the canonical tolerance of a target missed serve.
- A false positive is a selected candidate that does not match a target missed serve.
- Contact injection copies accepted contacts, adds selected candidates, and keeps spans fixed.
- The evidence-only injection clears the raw mask only at selected candidate frames.
- `live-non-standard` labels identify unusual live views. They do not by themselves prove a serve contact.
- The two pose bands are sensitivity variants. Neither is a production configuration.
