# Trial feature derivation definitions

| Item | Value |
| --- | --- |
| Status | Proposal for Ari's review |
| Scope | Definition only; no implementation or frozen schema |
| Owner | Issue [#22](https://github.com/ahalp90/badminton_cv_annotator/issues/22) |

This document proposes measurable definitions for the current trial-feature
shortlist. It does not record Ari's approval. Thresholds, reliability cutoffs,
and keep or cut decisions remain provisional.

## Workflow boundary

- [#102](https://github.com/ahalp90/badminton_cv_annotator/issues/102)
  defines the supported fixed-input path and reusable primitive artifacts.
- [#103](https://github.com/ahalp90/badminton_cv_annotator/issues/103)
  runs and verifies that path across the eligible ShuttleSet corpus.
- #22 defines candidate derivations over those primitives.
- [#104](https://github.com/ahalp90/badminton_cv_annotator/issues/104)
  benchmarks the provisional calculations and records keep, cut, or unresolved
  decisions.
- [#18](https://github.com/ahalp90/badminton_cv_annotator/issues/18)
  productionises retained features and freezes their names, types, and schema.

Issue [#17](https://github.com/ahalp90/badminton_cv_annotator/issues/17)
remains the umbrella for extraction and benchmarking. The #102 interface is in
progress, so this proposal does not assume its final artifact names or layout.

## Current evidence boundary

The production path already persists shuttle, raw pose, court, mask,
annotation, and rally-record artifacts with run provenance. Rally duration is
already stored as `(end_frame - start_frame) / fps` under half-open rally
intervals.

The sticky player's raw pose-slot picks and their court-projected positions are
used inside the annotator, but they are not fields of the persisted
`AnnotatorResult`. Raw accepted contacts also lack per-contact player
assignment. These values must be exposed by a reusable primitive contract or
reproduced under a recorded configuration before player-position features are
derived. The stroke classifier is not connected to the dataset builder.

For formulas below, `K_r = (run_id, source_dataset, video_id, rally_id)`.
Court position `(x, y)` is the existing normalised doubles-court coordinate,
with `x` spanning the 6.10 m width and `y` spanning the 13.40 m length.
`player_half` is a `Top` or `Bot` rally entity, not a durable player identity.
Cross-rally player aggregation requires a separate identity mapping.

Every proposed output should retain `valid` and a nullable `missing_reason`.
An unavailable input is null, never zero or `false`. Reliability fields below
are diagnostics and provenance. They are not calibrated probabilities.

## 1. Rally duration

**Purpose.** Measure the live-play length of one detected rally for timing and
workload analysis.

**Output grain and entity.** One value per detected rally.

**Keys.** `K_r`.

**Formula.** Keep the implemented definition:

```text
duration_frames = end_frame - start_frame
duration_seconds = duration_frames / fps
```

The rally interval is `[start_frame, end_frame)`.

**Units.** Frames and seconds.

**Required primitives.** Rally start and end frames, canonical video `fps`,
canonical `frame_count`, and the annotation configuration that produced the
span.

**Validity and missing values.** Structural validity requires
`0 <= start_frame < end_frame <= frame_count` and consistent positive `fps`.
The current record validator rejects violations. Analytical eligibility also
requires the provisional complete-rally rule, which #104 must benchmark. Keep
the measured duration when structurally valid and flag ineligible rallies;
do not silently remove them from the source record.

**Permitted aggregation.** Count, sum, mean, median, quantiles, and
distributions over explicitly eligible rallies. Report the included rally
count. A sum is not match effective playing time unless the match envelope and
rally coverage are also valid.

**Provenance and reliability fields.** Rally-span stage and configuration,
annotation artifact reference, `fps` source, boundary or completeness status,
`valid`, and `missing_reason`.

**Main limitations.** Missed rallies and early or late boundaries bias both
the value and any downstream timing ratio. A valid frame interval is not proof
that the rally is complete.

**Validation approach.** In #104, use ShuttleSet's first and last annotated
contact frames only as contact-envelope diagnostics. Report automatic-start
minus first-contact and automatic-end minus last-contact offsets, per video and
by 25/30 FPS input. Do not treat contact frames as true rally boundaries. For
duration accuracy, manually mark service contact and the point where the
shuttle lands, enters the net, or a fault ends the rally on a benchmark sample.
Report start, end, and duration error in frames and seconds, and count
unmatched and unusable rallies separately.

**Decision required from Ari.** Define the complete-rally eligibility rule and
decide whether ineligible but structurally valid durations remain available
with a flag.

## 2. Out-of-position states: Dive, Off-balance, and Stretch

**Purpose.** Describe compromised posture that may reduce a player's ability
to recover during a rally.

**Output grain and entity.** Recommended measurement shape: one observation
per player at each accepted contact, keyed to the pose at that contact. Derive
one per-rally, per-player share for each state. A second candidate is a share
over every valid in-rally frame. The choice remains open because the source
paper is contact-aligned while issue #13 proposed proportion of time.

**Keys.** Event observation: `(K_r, stroke_idx, contact_frame, player_half)`.
Rally summary: `(K_r, player_half, state)`.

**Candidate formulas.** Let `S` be the shoulder midpoint, `H` the hip
midpoint, `A_L/A_R` the ankles, and `S_L/S_R` the shoulders in image pixels.
Define:

```text
torso_deviation_deg = degrees(atan2(abs(S.x - H.x), abs(S.y - H.y)))
ankle_horizontal_span_px = abs(A_L.x - A_R.x)
shoulder_width_px = distance(S_L, S_R)
stance_ratio_horizontal = ankle_horizontal_span_px / shoulder_width_px
```

The [Court to Conversation](research/skill_assessment/bharadwaj_2026_court_to_conversation_cv_badminton_and_rag_llms.pdf)
paper reports these candidate rules at shuttle impact:

```text
Dive        = torso_deviation_deg > 75
Off-balance = 40 <= torso_deviation_deg <= 75
Stretch     = stance_ratio_horizontal > 1.8
```

These are paper-reported trial candidates, not adopted thresholds. The paper
selected them qualitatively on elite footage and did not perform the required
large-scale multi-expert state annotation. #104 should compare them with other
candidate thresholds before Ari selects any rule. Dive, Off-balance, and
Stretch remain separate flags; whether Stretch may overlap another state is
also unresolved. Euclidean shoulder distance is a provisional interpretation
of shoulder width. #104 should also test horizontal shoulder separation and
record the selected denominator in the definition version.

For a set `V_contact` of valid accepted-contact observations, the recommended
summary is:

```text
contact_state_share =
    count(state is true in V_contact) / count(V_contact)
```

For valid in-rally frames `V_frame` from a source with fixed `fps`, the
time-share candidate is:

```text
state_time_seconds = count(state is true in V_frame) / fps
valid_state_time_seconds = count(V_frame) / fps
state_time_share = state_time_seconds / valid_state_time_seconds
```

The contact and time denominators must have distinct names and must not be
mixed.

**Units.** Torso deviation in degrees; ankle span and shoulder width in pixels;
stance ratios and state shares as dimensionless values; time-share totals in
seconds.

**Required primitives.** Rally span, accepted contact frame for the
contact-aligned candidate, stable selected pose per player, relevant keypoint
coordinates and scores, player half, exclusion mask, and pose/player-picker
configuration. The time-share candidate also requires canonical `fps`.
Contact-aligned player attribution is required if the output must distinguish
striker and opponent.

**Validity and missing values.** Validity is per state. Dive and Off-balance
are null when the player pick or torso joints are unavailable, or torso length
is zero. Stretch is null when the pick, ankle, or shoulder joints are
unavailable, or shoulder width is zero. All states are null when the frame is
excluded or required player attribution is unresolved. Do not impute from
adjacent frames until an interpolation rule is proposed and benchmarked. A
rally share requires at least one valid observation. Contact shares must
include `valid_observation_count` and `eligible_observation_count`. Time shares
must include the corresponding valid and eligible seconds.

**Permitted aggregation.** Per-rally state count and share by `player_half`.
Across rallies, pool valid contact-observation counts for
`contact_state_share`. Pool `state_time_seconds` and
`valid_state_time_seconds` for `state_time_share`; do not pool raw frame counts
across different FPS values. The distribution of rally shares may also be
reported. Do not treat `Top` or `Bot` as the same person across rallies without
a stable identity mapping.

**Provenance and reliability fields.** Pose artifact and model, player-picker
method and configuration, contact source, definition and threshold set,
required-joint validity, observation counts or seconds, `fps` source for time
shares, mask status, `valid`, and `missing_reason`.

**Main limitations.** Monocular pose, occlusion, motion blur, perspective, and
airborne movements can distort 2D angles and ratios. Contact-only sampling is
not proportion of time. Frame sampling overweights long rallies and depends
less directly on the cited definition.

**Validation approach.** Have multiple reviewers label Dive, Off-balance, and
Stretch independently on a sampled set of contact frames and in-rally frames.
Report reviewer agreement, per-state precision/recall, coverage, and results
by view and pose quality. #104 should compare the event and time-share shapes
before making a keep or cut decision.

**Decisions required from Ari.** Choose contact share or time share, decide
whether to trial the paper thresholds, choose the Stretch shoulder-width
definition, define state overlap or precedence, and approve the minimum
evidence needed for a valid pose observation.

## 3. Away-from-centre recovery position

**Purpose.** Measure how far a player has recovered toward the centre of their
own half when the opponent contacts the shuttle.

**Output grain and entity.** One event value for the non-hitting player at each
accepted and attributed opponent contact. Keep event values, then optionally
derive a per-rally player summary.

**Keys.** Event: `(K_r, stroke_idx, contact_frame, player_half)`. Rally summary:
`(K_r, player_half)`.

**Formula.** For a projected ground position `(x, y)`, use half-court centre
`c_top = (0.5, 0.25)` or `c_bot = (0.5, 0.75)`:

```text
distance_from_half_centre_m = sqrt(
    (6.10 * (x - centre_x))^2
    + (13.40 * (y - centre_y))^2
)
```

Candidate ground anchors are the selected bbox bottom-centre, which matches
the sticky picker's selection geometry, or the selected pose's mean ankle
position. Ari must choose one. Recommended rally summaries are median and
valid-event count. Mean and standard deviation are candidates. A player ratio
is deferred because asymmetric missing contacts can make it misleading.

**Units.** Metres for distance. Normalised court `(x, y)` may be retained as
diagnostic evidence.

**Required primitives.** Accepted contact frames, per-contact hitter half,
stable player picks, the selected ground anchor, scene homography and court
validity, court-projected position, rally span, and exclusion mask. Current
persisted contacts do not include per-contact hitter half, and current
annotation persistence does not include stable picks or projected positions.

**Validity and missing values.** An event is null when the contact or hitter is
unresolved, the opponent pick or ground anchor is unavailable, the court is
invalid, the projection is non-finite, or the frame is excluded. Do not carry
forward a prior player position. A rally summary requires at least one valid
opponent-contact event and must report valid and eligible event counts.

**Permitted aggregation.** Median, mean, spread, and distribution over valid
events, with counts. Cross-rally or head-to-head aggregation requires durable
player identity and comparable event coverage. Do not aggregate missing
events as zero distance.

**Provenance and reliability fields.** Contact and attribution source,
player-picker method and configuration, position-anchor choice, pose artifact,
homography scene and artifact, court-presence flag, event coverage, `valid`,
and `missing_reason`.

**Main limitations.** This is a 2D ground-position proxy. Bbox and ankle
anchors behave differently during jumps and lunges. Contact misses sample
recovery unevenly, and camera or homography error is location dependent.

**Validation approach.** In #104, compare the non-hitting player's projected
position with ShuttleSet `opponent_location_x/y` after documenting coordinate
alignment, event matching, and striker/opponent role alignment. Report 2D
position error and centre-distance error in metres, coverage, and results by
video, court scene, and anchor candidate. Manually review mismatches and
unusable annotations.

**Decisions required from Ari.** Choose the ground anchor, the event-matching
rule, the per-rally summary fields, and whether any player ratio remains in
scope.

## 4. Rest time, work density, and effective playing time

**Purpose.** Describe the balance between live play and observed recovery time
within a verified game or match interval.

**Output grain and entity.** Rest time is one gap between consecutive rallies.
Work density and effective playing time are one aggregate per verified game or
match envelope. Until game and match identities are available, any aggregate
is a source-video interval and must be labelled as such.

**Keys.** Gap: `(run_id, source_dataset, video_id, preceding_rally_id,
following_rally_id)`. Aggregate: the stable game, match, or explicit source
interval key selected by #18.

**Candidate formulas.** For adjacent rallies `i` and `i+1`:

```text
observed_gap_seconds_i = (start_frame_(i+1) - end_frame_i) / fps
```

Call this `rest_time_seconds` only when source continuity and rally coverage
show that the gap represents elapsed player rest rather than a non-contiguous
edit. A replay or cutaway may occur during real player rest, but source timing
alone does not prove that equivalence.

For a verified envelope containing rally durations `D_i` and eligible
inter-rally rests `R_i`, two work-density conventions are candidates:

```text
work_rest_ratio_pct = 100 * sum(D_i) / sum(R_i)
active_interval_pct = 100 * sum(D_i) / (sum(D_i) + sum(R_i))
```

They answer different questions and must not share one field name. Do not
average per-gap ratios. Ari must select the convention meant by work density.
When the denominator is exactly the first-rally-start to last-rally-end
interval, `active_interval_pct` equals effective playing time and should not be
stored as a duplicate feature.

For an envelope `[envelope_start, envelope_end)`:

```text
effective_playing_time_pct =
    100 * sum(D_i) / ((envelope_end - envelope_start) / fps)
```

The first detected rally start to last detected rally end is a candidate
envelope, not an accepted match boundary.

**Units.** Seconds for gaps and durations; percent for ratios.

**Required primitives.** Ordered rally spans, `fps`, complete-rally status,
source continuity or edit evidence, definitive exclusion mask, stable game or
match envelope, and coverage/exclusion outcomes.

**Validity and missing values.** A gap is structurally invalid if rallies
overlap or timing bases differ. Preserve `observed_gap_seconds` when timing is
valid, but set `rest_time_seconds` null when continuity or rally coverage is
unknown. Ratios are null for zero denominators, fewer than two rallies where a
rest denominator is needed, an unverified envelope, or incomplete coverage.
Do not subtract replay or cutaway time without naming and validating a
separate edited-timeline measure.

**Permitted aggregation.** Sum valid rally and rest seconds within one verified
envelope, then calculate the ratio from sums. Report rally count, gap count,
excluded count, and uncovered time. Do not combine videos or matches without
explicit population and weighting rules.

**Provenance and reliability fields.** Rally-span stage and configuration,
envelope kind and bounds, continuity or edit status, mask artifact, included
and excluded counts, uncovered duration, `valid`, and `missing_reason`.

**Main limitations.** Broadcast gaps may contain edits, replays, or omitted
play. Missed rallies inflate rest and reduce effective playing time. The
current rally key has no game or match entity, so source-video aggregates may
not represent complete matches.

**Validation approach.** In #104, use consecutive ShuttleSet contact frames
only for event alignment, not as rest-time truth. On a benchmark sample,
manually mark rally-end and next-service-contact boundaries and classify gaps
as continuous rest, replay/cutaway, edit, or unknown. Measure gap error only on
usable continuous cases. Benchmark aggregate formulas only on envelopes with
reviewed coverage.

**Decisions required from Ari.** Select the work-density convention, define
the envelope entity and boundary, decide when a gap may be called rest, and
set the required continuity and coverage evidence.

## 5. Smash shuttle speed

**Purpose.** Estimate the initial pace of a detected smash for event-level
comparison.

**Output grain and entity.** One value per detected, attributed smash event.
Absence of an event row means no smash was detected; it must not be stored as
zero speed.

**Keys.** `(K_r, stroke_idx, contact_frame, player_half)` plus the stroke
classifier's event identity if it differs from the accepted contact identity.

**Candidate formulas.** Project visible shuttle image coordinates to the
normalised court plane, then convert each point to metres:

```text
q_t_m = (6.10 * x_t, 13.40 * y_t)
```

Candidate A is the first post-contact step:

```text
projected_speed_step_mps = fps * distance(q_(c+1)_m, q_c_m)
```

Candidate B fits a contiguous valid prefix of court-plane coordinates starting
at `c`. Let `W = {c, ..., c + k}` for `k <= K`, where every sample in `W` is
valid and `W` stops before the first missing or rejected sample. With time
`tau_t = (t - c) / fps`, fit ordinary least-squares slopes `beta_x` and
`beta_y` over `W`, then:

```text
projected_speed_fit_mps = sqrt(beta_x^2 + beta_y^2)
```

`K` and the minimum contiguous sample count are unresolved parameters for
#104. A separate maximum-gap candidate requires its own benchmark and may not
share this estimator's definition. Both values are apparent 2D ground-plane
speed proxies. They are not true 3D shuttle speed because a homography cannot
recover shuttle height.

**Units.** Metres per second. Kilometres per hour may be derived as
`3.6 * metres_per_second`, but the field name must retain `projected_2d` or an
equivalent proxy label.

**Required primitives.** Connected smash classification and confidence,
accepted contact and per-contact striker attribution, frame-aligned shuttle
track and visibility, shuttle fill and guard provenance, scene homography and
court validity, exclusion mask, and `fps`. Stroke classification is not
currently connected to the dataset builder.

**Validity and missing values.** A value is null when classification did not
run, the smash or contact is unresolved, the shuttle samples are unavailable
or rejected, the court or homography is invalid, the window crosses a scene or
excluded interval, or the estimator lacks enough valid samples. Do not bridge
missing samples or use inpainted samples without a separately benchmarked
rule. Candidate B requires enough contiguous valid samples from contact. A
rally with no detected smash has no speed event; stage failure remains an
explicit unavailable outcome.

**Permitted aggregation.** Event count, median, quantiles, and distribution
over valid detected smashes. Report valid events and classifier/contact
coverage. Player or match aggregation requires durable identity and must not
assume undetected smashes have zero speed.

**Provenance and reliability fields.** Stroke model, class and raw confidence;
contact and attribution source; shuttle model and artifact; visibility,
inpaint, and guard status for the window; homography scene and artifact;
estimator and `K`; `fps`; `valid`; and `missing_reason`.

**Main limitations.** The proxy mixes horizontal shuttle motion with
height-dependent perspective error. Smashes decelerate rapidly, so the result
depends on frame rate, contact timing, window length, missed detections, and
inpainting. ShuttleSet has no direct speed ground truth.

**Validation approach.** In #104, first benchmark smash-event and contact
matching. For matched events, report track coverage, projection continuity,
step-versus-fit agreement, and sensitivity to `K`, FPS, guard grade, and
inpainting. Use trajectory and landing checks only as indirect diagnostics.
Absolute validation as true 3D speed would require an independent calibrated
measurement source outside ShuttleSet.

**Decisions required from Ari.** Decide whether the 2D proxy is useful enough
to trial, select the estimator and window, choose permitted shuttle samples,
and define the classifier and contact evidence required before a value is
valid.

## Decision list for Ari

1. Approve or revise the complete-rally eligibility shape.
2. Choose contact-share or time-share OOP measurement, Stretch formula,
   threshold candidates, and state overlap rules.
3. Choose the recovery-position ground anchor and rally summaries.
4. Select the work-density convention and the valid rest/envelope rules.
5. Decide whether to benchmark the smash 2D proxy and, if so, select its
   estimator inputs.
6. Confirm which player-pick, per-contact attribution, projected-position, and
   stroke-class primitives #22 may expect from the in-progress reusable
   artifact interface.
