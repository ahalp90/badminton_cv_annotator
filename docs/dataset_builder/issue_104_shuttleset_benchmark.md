# Issue 104 ShuttleSet benchmark, Run 1

## Run 1 disposition

Keep rally timestamps, posture variability, linear-interpolation provenance,
and direct ShuttleSet source fields. Keep raw pose, court, and shuttle
primitives only in a separate bundle with their masks and reliability notes.

The current evidence cuts shots per rally, away-from-centre recovery, and
movement inefficiency. Their formulas run reliably, but the frozen rally,
contact, and attribution outputs are not accurate enough to support them. Rally
duration, serve speed, degradation, commentary, player sex, and backward
extrapolation remain unresolved.

These are Run 1 dispositions, not the final issue #104 decision. Keep issue
#104 open. A later run can use the same scorer after Ari's upstream work changes
the relevant production outputs. Issue #18 should treat cut and unresolved
fields as deferred until that comparison.

## Frozen evidence

| Input | Exact identity |
|---|---|
| Production source | `ad8da4f297e9278a9cc39bf216026545a7bbab05` |
| Final task 2.5 configuration | `external/shuttleset-full.toml`, SHA-256 `6e2a15ea3c44c4bc3cf8b38c461cdfd55c359178b49854080521949c07e93b20` |
| Issue #103 artifact run | run ID `a5d37677def443469f6b83d8ee838e7b` |
| Issue #103 run manifest | SHA-256 `84f91c139decdc4fe29957b8dd56cdd400491ba2b5aa190684fd3aa0e84a55db` |
| Rally projections | SHA-256 `71c54a7a7521871c152acedd46b399c86e78969b24949b35f6f4bda59567409c` |
| ShuttleSet ground truth | `training/data/shuttleset/annotations`, tree SHA-256 `cd81737c72d45036b4068065ffc43d21a8b61db40da0259f1c08471d7c427899` |
| `shots_master.csv` | SHA-256 `569dc74bbbb5d015a1e0be93b2c9a0885603eb320555028f11b9d259c79ee79f` |
| `homography.csv` | SHA-256 `b10f9f14a56ed499ded1805337e1d30d80aa0b3a72b6821dd76694c6a45b8035` |
| Issue #104 evaluator base | `f7571e60e439230346e4ed3449d56dd3929e7eb6` |
| Detailed external result | SHA-256 `45be6824bd7bfe6db20d465d031e489801e6e066cb1a83eb84c8c024839ad645` |

The task 2.5 configuration selects 40 fixed ShuttleSet videos. It uses
TrackNet stride 8, the large-video path, eight pose shards, CourtKeyNet pad
resize, and no commentary. Issue #96 kept this production configuration
unchanged. Later merged VLM and contact-detector experiments do not change this
input.

The supported replay restore validated the fixed source identity, artifact
indexes, model identities, hashes, frame counts, FPS, and array shapes before
feature scoring. It loaded the pinned shuttle, pose, court, and projection
artifacts without running vision inference. All 40 ground-truth reconciliations
used frame offset zero. This rules out a constant frame-index correction as the
source of the results.

The detailed report remains outside Git. The tracked
[`issue_104_per_video.json.gz`](data/issue_104_per_video.json.gz) contains the
exact per-video counts and summaries for all 40 videos. Its SHA-256 is
`98b431cbbb3098cfe022d416823002fbbb97dbc3447902aa317498bda3ec25db`.

## Matching and populations

- A ground-truth rally is covered only when every authoritative contact frame
  falls inside one half-open predicted span. Contacts crossing spans are split.
  Contacts outside all spans are missed.
- Canonical strict contact credit uses deterministic greedy one-to-one nearest
  frame matching within 5 base-30 frames, scaled to source FPS. The strict
  score gives credit only inside covered rallies. The tolerance curve also
  reports all overlapping-span candidates.
- Shuttle and player coordinates are compared at the exact detailed-set contact
  frame. Error is Euclidean distance in normalized doubles-court coordinates.
- Each accepted production court scene is compared with ShuttleSet's static
  four-corner quad at the 1280 by 720 reference resolution.
- A landing prediction is paired through its covered rally span. Landing frames
  are not independently matched.

The corpus contains 40 videos, 4,442,098 frames, 44.695 hours, 3,359
ground-truth rallies, 33,267 authoritative master contacts, and 3,527 predicted
rallies. The detailed set tables contain 33,486 contact rows. There are 161
reconciled duplicate rally labels and 20 mismatched rallies across five videos.
Twenty-eight detailed contact rows have no exact master `player_side` row, so
player attribution is unusable there. Shuttle coordinate scoring remains valid
for those rows.

## Production benchmark

### Rally and contact detection

| Measure | Result |
|---|---:|
| Covered rallies | 2,225 / 3,359, 66.24% |
| Split rallies | 823 / 3,359, 24.50% |
| Missed rallies | 311 / 3,359, 9.26% |
| Merged predicted spans | 36 |
| Spurious predicted spans | 523 |
| Strict contact precision | 18,023 / 40,962, 43.99% |
| Strict contact recall | 18,023 / 33,267, 54.18% |
| Strict contact F1 | 48.56% |

The all-overlapping-span contact curve is:

| Tolerance, base-30 frames | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 1 | 31.82% | 37.25% | 34.32% |
| 2 | 54.37% | 63.67% | 58.65% |
| 5 | 61.52% | 72.04% | 66.37% |
| 10 | 65.33% | 76.49% | 70.47% |

The 25 FPS group covers 77.80% of 1,482 rallies and has strict contact F1
57.93%. The 30 FPS group covers 57.11% of 1,877 rallies and has strict contact
F1 40.58%. The gap is real in this corpus, but it is not a frame offset. All 40
reconciliations use offset zero.

Per-video rally coverage ranges from 8.11% on `sset_11` to 97.33% on
`sset_02`. Strict contact F1 ranges from 6.23% to 76.38% on the same videos.
Leaving out any one video moves aggregate coverage only from 65.22% to 68.23%
and contact F1 from 47.69% to 49.96%. No aggregate conclusion depends on one
fixture.

### Court, player, shuttle, landing, and attribution

| Output | Correct or eligible population | Result |
|---|---:|---:|
| Court corners | 15,096 / 15,096 corners | median 4.34 px, p90 9.52 px |
| Shuttle at GT contacts | 27,502 / 33,486 rows | median 0.459, p90 1.031 |
| Striker position at GT contacts | 30,586 / 33,486 rows | median 0.078, p90 0.132 |
| Opponent position at GT contacts | 30,582 / 33,486 rows | median 0.061, p90 0.105 |
| Landing coordinates | 1,162 / 3,280 GT-available rallies | median 0.076, p90 0.702 |
| Exact shot count | 298 / 3,359 rallies | 8.87% |
| Final striker attribution | 1,016 / 3,359 rallies | 30.25% |
| Server attribution | 1,140 / 3,359 rallies | 33.94% |
| Hit height | 7,659 / 33,265 labels | 23.02% |
| Landing half | 929 / 3,280 labels | 28.32% |
| Winner | 1,004 / 3,159 labels | 31.78% |

Coordinate exclusions are explicit. Shuttle scoring excludes 4,303 rows with
missing ground truth and 1,681 with no eligible prediction. Striker scoring
excludes 1,069 ground-truth or attribution cases and 1,831 missing predictions.
Opponent scoring excludes 1,072 ground-truth or attribution cases and 1,832
missing predictions. Landing scoring excludes 79 rallies without a ground-truth
coordinate and 2,118 with no paired prediction.

Leaving out any video keeps the shuttle median between 0.454 and 0.464, the
striker median between 0.077 and 0.080, the opponent median between 0.060 and
0.063, and the court median between 4.27 and 4.39 px. The strong court and
player results, and the weak shuttle result, are not single-video effects.

## Feature evaluation and provisional decisions

The prototypes evaluated all 3,527 predicted rally rows. Posture variability is
available for 7,024 of 7,054 player-rallies, or 99.57%. Recovery is available
for 38,155 of 40,962 contact windows, or 93.15%. Movement inefficiency is
available for 74,056 of 74,914 player intervals, or 98.85%. Linear interpolation
fills 72,756 player-signal frames. These are coverage results, not independent
feature-accuracy claims.

Leave-one-video-out medians are stable. Posture MAD stays between 1.022 and
1.029, recovery distance between 0.144 and 0.145, and movement inefficiency
between 0.0595 and 0.0605. This proves broad population support, but it does not
repair weak rally, contact, or attribution inputs.

| Trial field | Run 1 decision | Reason |
|---|---|---|
| Rally frame and second timestamps, with FPS | **Keep** | Exact conversion, complete population, and required row identity. Reliability follows the reported rally boundary quality. |
| Rally duration from final contact plus offset | **Unresolved** | Issue #22 does not define the end offset. Zero eligible values were emitted rather than inventing it. |
| Posture variability MAD | **Keep** | Formula is complete, coverage is 99.57%, player coordinates are accurate enough, and leave-one-video-out results are stable. There is no independent posture ground truth, so issue #18 must label it as derived rather than validated biomechanics. |
| Player sex metadata for posture interpretation | **Unresolved** | The frozen source has no authoritative field. Names or tournament folders must not be guessed. |
| Away-from-centre recovery | **Cut** | Although coverage is 93.15%, the contact and server attribution inputs are too weak for trustworthy player-specific windows. |
| Serve speed proxy | **Unresolved** | Return, static, and viewport endpoint policy is incomplete. Exact-frame shuttle error is also too large to support a keep decision. |
| Shots per rally | **Cut** | Only 298 of 3,359 ground-truth rallies have the exact predicted count. |
| Movement inefficiency | **Cut** | Coverage is high, but missing and spurious contacts change interval boundaries. The result would silently measure the wrong intervals. |
| Raw degradation slope | **Unresolved** | Upstream retained-feature set and player identity are not complete enough for a meaningful progression. |
| Tanh-normalized degradation | **Unresolved** | Issue #22 does not define the temperature. |
| Commentary sentiment, concept, timing, and player link | **Unresolved** | Issue #103 disabled commentary, so there is no valid population. |
| ShuttleSet contact type, round, and set fields | **Keep** | They are direct human-source fields. They must remain source-scoped rather than presented as annotator predictions. |
| Linear interpolation and `interpolation_type` provenance | **Keep** | Internal gaps are bounded by observations inside one court scene. The provenance is explicit and broadly exercised. |
| Backward extrapolation | **Unresolved** | Issue #22 does not define a safe scene or match-start policy. No extrapolated values were emitted. |
| Raw shuttle, pose, bbox, and court primitives | **Keep in a separate bundle** | The existing compressed inputs are feasible: 72,272,724 bytes shuttle, 3,523,620,168 bytes pose, and 2,859,648 bytes court. Keep visibility, guard, and interpolation provenance. Do not describe raw shuttle positions as accurate. |

## Comparison contract for later runs

This report and its per-video summary are the first comparable baseline. A later
run must record its exact source commit, configuration digest, artifact run,
rally-record digest, ground-truth digest, and evaluator revision. It should use
the same matching rules and report any changed populations or exclusions.

Compare aggregate, per-video, FPS-stratified, and leave-one-video-out results.
A changed disposition must not depend on one video or a nonzero reconciliation
offset. New vision inference is needed only when the later run intentionally
changes a primitive producer or the pinned artifacts fail integrity checks.

For now, issue #18 can use the keep rows as provisional inputs. It should carry
FPS, frame ranges, exact source identity, missing values, and interpolation
provenance. Source-provided contact type, round, and set must be distinguishable
from predicted fields. Raw primitives belong in a linked bundle rather than the
rally table.

Do not add replacement heuristics for the cut or unresolved fields. Revisit
them with the same benchmark after the relevant upstream work is ready.
