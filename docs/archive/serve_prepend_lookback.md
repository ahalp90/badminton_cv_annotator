> ## ARCHIVED: superseded serve-prepend handover
>
> Archived: **20260731-070505 UTC**.
>
> This document is retained only as a record of earlier design reasoning. It is **not authoritative**
> for current code, current measurements, current processing semantics or implementation decisions.
> Use the [current build orientation](../scraper_pipeline/serve_prepend_lookback/serve_prepend_lookback_20260731-091227.md)
> and the measurement pack beside it instead.
>
> Its limited ongoing utility is provenance. It records the original problem framing, the earlier
> design alternatives, the three decisions that shaped the measurement, and the risks identified
> by the earlier review. It also preserves a few implementation cautions that may still be useful
> if the feature is revisited: the body-unit wrist-gate semantics, the no-court sticky-evidence
> sentinels, the parity cost of a false contact, and the fact that GT endpoints are contact frames.
> These are small facts, not a reason to use this document as an active specification.
>
> The current orientation supersedes the old handover's pre-W2.9 figures, earlier prominent-person
> and serve-setup proposals, `None` half-guess assumptions, old commit anchors, rerun instructions,
> historical record pointers and definition of done. The current orientation already carries the
> relevant current call sites, processing order, build decisions and YAGNI constraints.
>
> **Do not use this archive to make current build decisions.**

# Serve-prepend lookback: handover

This is the tracked handover for the serve-prepend lookback feature. It
carries the problem, the recorded design, the build shape, the three
decisions owed, the binding constraints, and current code seams. Every
number came from the 2026-07-23 measurement on the pre-W2.9 sticky chain;
labelled sections say so.

The handover was moved here on 2026-07-31 because replay-event labelling is
the first dependency in its replay family. The current replay-labelling
assessment is at
[`non_play_manual_labelling_20260731-095201.md`](../scraper_pipeline/broadcast_nonstandard_camera_id/non_play_manual_labelling_20260731-095201.md);
that note supersedes stale claims about current measurements, but preserves
this handover as historical design context.

## Conventions

- "the annotator" is the GT-free pipeline that turns a video into rally
  spans, contacts, and point-winner verdicts. Its per-video chain is
  `annotator.run_video.run_video` (`src/annotator/run_video.py`).
- "GT" is ShuttleSet's per-rally ground truth. The first GT stroke of a
  rally IS the serve frame.
- The three fixtures are the calibration videos: `sset_01`
  (formerly `pilot`, 113 GT rallies, 25 fps), `sset_15` (formerly `vid15`,
  104 rallies, 25 fps), `sset_21` (formerly `sset21`, 75 rallies, 30 fps).
- "the replay mask" is the per-frame boolean saying a frame belongs to a
  replay or cutaway. Masks on disk are raw detector output. Consumers apply
  belief at their existing consuming boundary: only a flagged run at least
  `replay_mask_min_frames` long (13 at 25 fps, fps-scaled) is believed, and
  believed frames produce no contact events (shipped 2026-07-27).
  `annotator.replay_mask.filter_short_exclusion_runs` is the only belief
  implementation.
- "the external review" is the 2026-07-23 design red-team; its record is
  `records/sol_redteam_round2_20260723.txt` (gitignored). Its constraints
  bind this build.

## 1. The problem, with numbers

The annotator misses serves. Rally detection opens on shuttle motion, and
a serve often happens before the detector commits to a rally, so the first
contact recorded is frequently the return, not the serve. A missing serve
shifts every later stroke's parity, and parity is how the point winner
infers who served next and who won the previous rally.

The 2026-07-23 serve-miss scope run (**pre-W2.9, pre-W3.1 rename**;
see `docs/scraper_pipeline/evidence/serve_prepend/historical_20260723/`):

- 136 GT serves across the three fixtures are missed by the pre-W2.9
  chain — `sset_01` 64 (formerly `pilot`), `sset_15` 38 (formerly `vid15`),
  `sset_21` 34 (retained stem).
- 113 of the 136 have a clean visible shuttle-track run within one second
  of the GT serve frame (53 / 35 / 25). Recoverable from track evidence
  alone.
- 23 of the 136 have no clean track run nearby (11 / 3 / 9). Recovering
  these needs something other than the shuttle track (decision 1).
- On sset_21, 17 of its 34 missed serves sit inside the believed replay
  mask, so under the shipped no-contacts-on-believed-frames rule they are
  unrecoverable until the replay-mask redesign (decision 3).

**These numbers must be rerun on the current chain before they drive any
build decision.** See the "Rerun after W2.9" pointer at §5 and the live
`local_scratch/autograder_architecture/TODO.md`. Do not treat the 136 as
current behaviour.

The external review also measured mask-edge proximity: 35 of 113 sset_01
rally starts sit within 10 frames of a mask end, 75 within 50. A lookback
that crosses masked frames is the common case, not an edge case.

One more fact for scoring: GT start and end frames are contact frames. The
recorded rally start is the first contact, so a recovered serve
legitimately sits before the current GT-scored rally extent. Scoring that
treats GT extents as the rally's true visual extent under-runs both ends;
this is a known flaw for clip design and measurement (recorded 2026-07-27).

## 2. The recorded design

The design as ruled (2026-07-23, re-scoped 2026-07-27): a lookback that
runs when a rally's first accepted contact looks like a return rather than
a serve, searches a bounded window before the rally start, and proposes a
serve contact from:

- **The prominent-person pick.** Person evidence for the candidate server.
  Ruled shape from an earlier decision: rank people by bbox area with an
  image-centre tie-break, surface the pick as an unassigned
  "prominent person", never fill a court-half slot with it.
- **Raw contact impulses.** Shuttle-track direction/speed changes in the
  lookback window, taken from the raw track before any gate.
- **A serve-setup gate.** The candidate frame must look like serve setup
  (server roughly still, shuttle near the racket) before a trigger fires.

Delivery shape: an integrated pipeline feature behind a config switch,
built to ship-quality on the assumption it lands. The only conditional
part is the default: whether the switch starts on or off is decided from
the measured numbers, and that decision is the owner's. The switch name and
config home are picker's choice within house naming.

Every trigger gets GT-scored, including false positives on replay
stretches. The feature's value is decided by measurement, not by
inspection.

## 3. What to build

Two pieces, in this order.

**First the measurement harness**, because it de-risks everything else.
The external review specified the smallest version. "Recording-only" means
the trigger logic runs inside the recorder (it must, to produce verdicts),
but nothing it does alters spans, contacts, or any output the pipeline
writes:

- A recorder that walks the three fixtures using existing raw pose arrays,
  shuttle tracks, and inpaint-quality codes. No pipeline behaviour changes.
- For every rally (not just known misses), it records: derivable
  serve-miss status, the prominent pick's raw detection slot, bbox
  area/centre/confidence, wrist keypoint validity, track visibility,
  inpaint code, mask state, pixel distance, bbox-height-normalised
  distance, and the trigger verdict.
- Misses are derived from GT with the existing matcher
  (`annotator.calibration.scoring.greedy_match`; its tolerance argument is
  in frames). Tolerances are fps-scaled: use
  `annotator.calibration.gt_scoring.canonical_tolerance(fps)`, which
  encodes the 2026-07-08 recall-first ruling's base-30 "5" band (4 frames
  at 25 fps for sset_01 and sset_15, 5 at 30 fps for sset_21). Report the
  curve the calibration sweep reports, which scales its (1, 2, 5, 10)
  base-30 bands the same way. A trigger with no matching GT serve is a
  false positive by definition, on replay stretches and junk intervals
  included.
- Insertion is simulated in memory: push the proposed contact through the
  point-winner's alternation fit
  (`annotator.point_winner.fit_alternation`,
  `annotator.point_winner.next_server_half`) and record the
  counterfactual contact count, server, previous-rally winner, and
  commentary-pairing eligibility. The prepended contact enters with a
  `None` half guess, because the prominent pick is unassigned by ruling;
  this None arm is the required counterfactual. A second arm with an
  attributed half is optional and only if decision 1 lands an attribution
  rule. Record pairing eligibility for both prepend variants
  (span-expanded and injected). Do not alter spans or returned contacts in
  this phase.
- Report trigger precision and recall, and report track availability and
  masked-scene support as separate columns. The fixtures contain few
  close-up serves, so a good overall number must not be read as close-up
  generalisation.

**Then the pipeline feature** behind its config switch, shaped by what the
harness shows and by the three decisions below. The `--no-replay-mask`
switch on the calibration CLI is the house precedent for an opt-in/opt-out
flag: the substitute value is the domain identity (an all-False mask),
never a sentinel, and the run's config echo records the choice.

A scoring definition to lift verbatim from the review record: a miss is a
GT first stroke unmatched within the canonical tolerance while the first
accepted detected contact matches a later GT stroke. True positive:
trigger matches the missed GT serve. False positive: trigger on an
already-covered serve, a replay or junk interval, or a wrong frame. False
negative: a derivable miss with no trigger. This needs no new hand labels.

## 4. The three decisions still owed

### Decision 1: the evidence path for a triggered contact

When the wrist gate runs, a contact candidate needs a finite
wrist-to-shuttle distance within
`annotator.rally_segmentation.BODY_UNIT_WRIST_THRESHOLD` (1.4 bbox
heights); the scoring filter drops candidates that fail. When the gate
never ran (no gate inputs), the verdict is `None` and the filter keeps the
candidate. The evidence path decides which arm a lookback contact lives
in. In the lookback window there may be no usable court, and sometimes no
shuttle point at all.

Owner's lean, for the harness to confirm or kill: the usual shuttle/wrist
serve-setup checks should work in the lookback window, except that there
is no known court. Player bboxes, keypoints, and a shuttle point remain.
The candidate server is likely the most salient bbox in the centre of the
video's Y-axis. It covers the 113 track-bearing misses; the 23 no-track
misses are the residual class the lean deliberately does not reach.

What supports the lean:

- Sticky cannot help here: without court corners it produces no picks
  (verified 2026-07-27 in the homography-failure audit). The observable
  cache contract for no-court frames is `picks = -1`, `analysed = False`,
  `distances = +inf`; select such frames by the analysed flag or infinite
  distance, never by NaN checks. Raw pose arrays are the person evidence.
- RTMPose always outputs all 17 COCO keypoints when it detects a person;
  confidence may be low, but the wrists exist.
- 113 of 136 misses have a clean track run within a second, so
  track-anchored impulses cover the bulk.

What cuts against it, from the external review:

- Bbox-height normalisation has no measured transfer to frame-filling
  views; a huge bbox divisor can compress unrelated or fabricated track
  points into apparently close distances.
- Serve-setup appearance cannot separate live serves from replayed serves;
  a replayed serve is the replay class most likely to look exactly like
  serve setup.
- Ordinary serve distance populations already overlap between videos.

The 23 no-track misses need their own evidence path and measurement, or a
record-and-skip floor.

### Decision 2: prepend semantics

When a trigger is accepted, does the rally span expand to include the
serve frame, or is the contact injected with the span left alone? The
external review flagged both directions:

- Expansion changes the span table every consumer reads, and changes
  GT-scoring denominators (see §1 on contact frames).
- Injection without expansion puts a contact outside its stated rally
  interval, which is inconsistent for any consumer that assumes contacts
  live inside spans.

One thing has improved since the review: commentary pairing no longer
discards a rally for any mask overlap. The shipped rule is interior grace
(`src/scraper/stage11_pairing.py::_believed_replay_in_rally_interior`): a
rally's asserted start and end each get
`scale_for_fps(fps).replay_mask_min_frames` of grace, and only believed replay
deeper than that grace makes it unpairable. The belief threshold and pairing
grace are the same fps-scaled constant, not two independent knobs. So expanding
a span slightly into a masked lead-in is no longer the automatic pairing
death it was. Expansion into believed replay deeper than the grace still
is.

Whichever is chosen, the choice must be visible in the output schema (a
flag or provenance column) so downstream consumers and the GT scorer can
tell a prepended rally from a natural one.

### Decision 3: interaction with the replay-mask trust rule

The shipped rule enforces itself at two points. First,
`annotator.rally_segmentation.apply_replay_mask` freezes the shuttle
track on believed replay frames before segmentation, so detection never
sees motion there. Second, `run_video` filters any surviving contact on a
believed frame after the scoring gate
(`src/annotator/run_video.py::run_video`, using
`annotator.replay_mask.filter_short_exclusion_runs`). Applied strictly to lookback
triggers, the 17 sset_21 misses inside the believed mask stay
unrecoverable.

Decision: does the lookback obey the rule as-is (clean, loses the 17), or
get a narrow, measured exemption? If an exemption is explored, the
review's warning stands: replayed serves maximally satisfy any serve-setup
gate, so an exemption without a strong live-vs-replay discriminator will
manufacture false serves from replays. The ownership boundary is
deliberate: this decision covers only whether the lookback obeys the rule.
Recovering in-mask serves in general belongs to the replay-mask redesign,
and an exemption here must not quietly grow into that redesign. The
harness's false-positive columns on replay stretches are the evidence
either way.

Replay-event labelling on one video was ruled the first task in the
replay family, and one of its four named consumers is exactly this
feature's false-positive rate. The GT-derived scoring above is sufficient
for go/no-go without those labels; the labels make the false-positive
audit stronger.

## 5. Rerun after W2.9

Every 2026-07-23 count in §1 was measured against the pre-W2.9 sticky
build. W2.9 changed contact behaviour on all three fixtures (contact_f1
lifted; server/getpoint moved in mixed directions; see
`docs/architecture_notes/completed_general_refactors/annotator_cleanup/w2_9_delta.diff`).
The miss ledger and the 113 / 23 track split must be re-measured on the
current chain before either lookback design or replay-mask redesign
proceeds.

The rerun specification is in
`local_scratch/autograder_architecture/TODO.md`. It uses
`annotator.calibration.fixtures.FIXTURES`,
`annotator.calibration.gt_scoring.build_run_video_inputs`, and
`annotator.calibration.gt_scoring.canonical_tolerance` to reproduce the
2026-07-23 harness shape against the current tip, and emits three
canonical-stem CSVs plus a comparison paragraph against the historical
136. It is deliberately separate from feature implementation.

## 6. Binding constraints

- Score every trigger, not only triggers inside known miss cases.
- A false trigger is worse than a miss: one wrong added contact flips
  alternation parity and can corrupt both that rally's fit and the
  previous rally's winner. This asymmetry should shape every threshold.
- No span or contact change ships before the in-memory counterfactual
  checks pass on all three fixtures.
- The prominent pick must stay unassigned: it never fills a court-half
  slot, per the earlier ruling that keeps slot identity clean.
- Report close-up and masked-scene support separately; do not let
  fixture-wide averages claim generalisation the fixtures cannot show.
- The feature ships behind its config switch either way; the default is
  decided from the numbers by the owner.

## 7. Current code seams

Symbol names against `feature/commentary-scraper` at the current tip.
Commit `ebae2b3` is the anchor for the 2026-07-27 wording; the module
layout has been tidied since (W2.6–W2.8). Use `git log -S "symbol_name"`
or `git grep "symbol_name" ebae2b3` to follow anything that moved.

- `src/annotator/rally_segmentation.py`: per-video segmentation entry;
  `detect_contacts` returns the hit-detection surface the review says to
  leave untouched; `BODY_UNIT_WRIST_THRESHOLD = 1.4` (line 102) is the
  wrist gate; `ServeStartOptions` is the serve-start options type;
  `apply_replay_mask` is the track-freeze half of the replay rule;
  `true_runs` is the shared run-finding helper.
- `src/annotator/run_video.py`: the one-video chain. `build_serve_options`
  lives here (not in the segmentation module) — read it before inventing
  new gates. The contact filter that drops believed-replay contacts also
  lives here, after the scoring gate. Serve options are currently derived
  from the unmasked sticky cache; the review flagged that ordering, so
  check it before relying on it.
- `src/annotator/replay_mask.py`: `filter_short_exclusion_runs` is the only belief
  implementation. Raw masks on disk, belief at every consuming entrance.
  Do not add a second belief path.
- `src/annotator/point_winner.py`: `fit_alternation` and
  `next_server_half` are the parity machinery the counterfactual
  simulation drives.
- `src/annotator/calibration/`: `fixtures.FIXTURES` is the three-fixture
  substrate; `scoring.greedy_match` is the GT matcher (tolerance in
  frames); `gt_scoring.score_video` is the per-video scorer;
  `gt_scoring.build_run_video_inputs` returns the invariant `run_video`
  inputs the current live capture uses; `gt_scoring.canonical_tolerance`
  is the fps-scaled tolerance helper.
- `src/scraper/stage11_pairing.py`: `_believed_replay_in_rally_interior`
  is the interior-grace pairing rule described in decision 2.

## 8. Data and records

- **Historical evidence:**
  `docs/scraper_pipeline/evidence/serve_prepend/historical_20260723/`
  carries the three 2026-07-23 CSVs (`pilot_missed_serves.csv`,
  `vid15_missed_serves.csv`, `sset21_missed_serves.csv`, retained under
  their pre-W3.1 names) with one row per missed serve and per-frame
  track-visibility context.
- **Related historical producer:**
  `local_scratch/autograder_architecture/now_tracked/serve_prepend/serve_miss_scope.py`
  (gitignored) is the pre-W3.1 producer; the current-chain rerun
  specification does not port it as-is. See the rerun TODO for the
  substitute using `build_run_video_inputs`.
- **Producer contract (inpaint):** `docs/tracknet/inpaint_sidecar.md`;
  consumer state and open work at
  `docs/tracknet/inpaint_sidecar_consumption.md`.
- **External review record:** `records/sol_redteam_round2_20260723.txt`
  (gitignored) carries the binding verdict and the full measurement
  column list.
- **Homography-fail audit:**
  `records/homography_fail_verification_sol_20260727.txt` (gitignored)
  shows which consumers survive a scene with no court and why sticky
  cannot serve the lookback window.

## 9. Definition of done

- The recorder harness runs on all three fixtures and its CSV carries
  every column in §3, with trigger precision / recall, track availability,
  and masked-scene support reported separately.
- The three decisions above are made and written down, each with the
  harness numbers that justified it.
- The pipeline feature sits behind its config switch; with the switch off,
  the full test suite passes and its calibration metrics match
  `tests/data/annotator_calibration/reference/`. Any score movement is
  reviewed as a behavioural change.
- With the switch on, the three-fixture GT scores and the counterfactual
  columns are the evidence pack for the default-on versus default-off
  call.
- A false-trigger analysis on replay stretches exists, whatever decision
  3 lands on.
