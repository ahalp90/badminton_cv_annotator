> ARCHIVED 2026-08-12: historical launch brief. Current position: `../../HANDOVER.md`.

# Launch: H3/R8 dual opener experiment

This launch supersedes the original runbook for the next scratch-only extension.
It preserves the completed outgoing-first baseline and writes separate H3/R8
evidence and results.

## Resume state

- Repository: the repository root
- Branch: `investigation/serve-id-by-lookback-followup`
- Tip when this launch was written: `24ac3a4`
- Population: the same 239 one-to-one rallies from `sset_01`, `sset_15`, and
  `sset_21`
- Stable rally key: `(fixture, video_id, set_id, rally)`
- State: H1 through H4 are complete. The checked outputs and report are listed
  below.
- Next action: resume from the checked result. The user authorised one final
  state commit on 2026-08-11; further commits need fresh approval.
- The original 20 overlays have also been rerendered at 1920x1080 with the
  readable HUD. Preserve them with the existing stride comparisons.
- The user approved the final state commit on 2026-08-11. Do not make further
  commits without fresh approval.

Read this file first after compaction. Then read the Resume section in
`worklog.md`, followed by `h3_r8_report.md`, `.github/AGENTS.md`, and
`.codex/context.md`.

## Question

Run two opener searches over the same frozen accepted contacts after relaxing
the recurrence halo and gross-jump limit:

1. rerun the completed sequential outgoing-first search; and
2. test an incoming-only search that starts from the earliest accepted contact
   with positive incoming evidence and inspects its nearest earlier accepted
   contact.

Measure every accepted contact before either search. Save that GT-free evidence
to disk. Derive both searches only from the saved evidence, then join GT for
scoring.

## Fixed local trajectory settings

Use the same settings for pre-contact and post-contact traces:

```text
recurrence halo              3 source frames on each side
local trace window           30 base-30fps frames
minimum path                 5 usable frames
maximum local contact gap    2 base-30fps frames
largest_step_ratio           <= 8.0
incoming threshold           fitted_decrease_bh >= 0.05
outgoing threshold           fitted_decrease_bh <= -0.05
```

The three-frame halo is literal source frames. The recurrence detector's core
is not FPS-scaled today.

Keep recurrence core frames, flat core frames, and repeated attractor positions
rejected. Shrink only the halo around a core recurrence event.

The ratio limit remains a gross-jump guard. It does not require a smooth path.
The current trace extractor uses one contiguous usable run, so it never measures
a step across a missing or rejected frame.

## Reconstructing the three-frame guard

Build the experimental guard from production `grade_track` codes without
changing production code:

1. Treat `FABRICATED` and `SUSPECT_FLAT` frames as the recurrence core.
2. Recover the exact `(x, y)` positions present at core frames.
3. Mark any frame at one of those positions as `on_attractor`.
4. Add three frames before and after each contiguous core run.
5. Mark `(halo3 | on_attractor) & ~core` as `DEGRADED`.
6. Restore the original core codes.
7. Restore stored `(0, 0)` shuttle frames to `NO_FLAG`, matching production.

As a required invariant, the same reconstruction with a 15-frame halo must
exactly reproduce production `grade_track` codes for all three fixtures.

## Freeze all accepted-contact evidence first

For every accepted contact in every primary rally, save one GT-free evidence
row before running either opener search.

Suggested file:

`h3_r8_contact_evidence.csv.gz`

Each row should contain:

- stable rally identity, FPS, span ID, accepted rank, accepted frame, and player;
- pre-contact run bounds, contact gap, frame count, largest-step ratio, fitted
  decrease, eligibility state, and verdict;
- post-contact equivalents and the binary credible-outgoing result; and
- any measured high-shot-out-of-bounds state needed for the predecessor rule.

Pre-contact verdicts remain three-way:

```text
incoming       eligible path and fitted_decrease_bh >= 0.05
not incoming   eligible path and fitted_decrease_bh < 0.05
unavailable    no usable path or failed common path eligibility
```

Post-contact outgoing remains binary. Missing or ineligible post evidence is
`false`, the same as measured non-outgoing motion.

The evidence builder may receive only a GT-free projection. It must not receive
GT stroke frames, labels, rally boundaries, or truth tables.

## Search A: sequential outgoing-first

Rerun the existing search using only the frozen H3/R8 evidence:

```text
for accepted contacts from earliest to latest:
    credible outgoing -> select and stop
    otherwise         -> skip

no selected contact -> no credible accepted contact
```

Classify the selected contact from its frozen pre-contact verdict:

```text
incoming       -> first visible post-serve contact; imply an unshown serve
not incoming   -> visible serve at the selected accepted frame
unavailable    -> not enough pre-contact trajectory to classify
```

This is a clean rerun of the completed baseline with only halo 3 and ratio 8.

## Search B: incoming-only predecessor check

Choose the earliest accepted contact whose frozen pre-contact verdict is
`incoming`. This is the first evidence-backed real rally contact. The search
does not use outgoing evidence.

Because this is the earliest `incoming` contact, every earlier accepted contact
is necessarily `not incoming` or `unavailable`. The backward search therefore
has at most one hop:

1. Let the anchor be the earliest `incoming` contact.
2. Find the immediately preceding accepted contact.
3. Decide whether that predecessor is eligible for inspection using the rules
   below.
4. If it is eligible and `not incoming`, classify the predecessor as the
   visible serve.
5. If it is eligible and `unavailable`, stop with insufficient predecessor
   trajectory. Do not collapse this into `not incoming`.
6. If no predecessor exists, or the predecessor is outside both admission
   rules, classify the anchor as the first visible post-serve contact and imply
   a missing or unshown serve. Do not invent a serve frame.

The predecessor's time eligibility supplies a candidate only. It is not proof
that the predecessor caused the incoming shot.

If there is no incoming anchor, keep these terminal outcomes separate:

- `no_accepted_contact`;
- `no_measured_incoming` when every accepted contact is `not incoming`; and
- `no_incoming_anchor_with_unavailable_evidence` when at least one accepted
  contact is unavailable but none is incoming.

Suggested successful and terminal categories are:

- `visible_serve`;
- `first_visible_post_serve_contact`;
- `predecessor_evidence_unavailable`;
- `no_measured_incoming`;
- `no_incoming_anchor_with_unavailable_evidence`; and
- `no_accepted_contact`.

Also record the exact stop reason, such as no predecessor, beyond the ordinary
window, or admitted by the high-shot exception.

## Predecessor admission rules

The nearest earlier accepted contact is eligible when either rule passes.

### Ordinary timing rule

The accepted-contact gap is at most 60 base-30fps frames, inclusive. Scale the
cap with `ScalingKind.FRAME_COUNT`.

### Measured high-shot-out-of-bounds exception

Allow a gap beyond 60 when the two accepted contacts bracket a measured
production `high_shot_oob` state and each contact is within 12 base-30fps frames
of its respective state endpoint, inclusive.

The state must satisfy the existing outbound and two-sided re-entry checks. Use
the frozen release configuration. The high-shot state is:

```text
[gap_start, min(gap_start + gap_state_demotion_bound, gap_end))
```

The 12-frame endpoint buffer matches the existing base-30 impulse half-window.
It was chosen before GT scoring. The exception permits inspection of the
predecessor but does not establish causation or continuity through the gap.

Clip 13 is the motivating measured case. In `sset_21:21:set1:6`, accepted
contacts 16509 and 16574 have a 65-frame gap. The measured high-shot state is
`[16515, 16564)`, placing the contacts 6 and 10 frames from its endpoints.

## GT boundary and scoring

The fixed population crosswalk is GT-derived. Keep that crosswalk separate from
the search interface.

The order of work is strict:

1. build and save all accepted-contact evidence with no GT search input;
2. derive and save both GT-free search results from that evidence;
3. freeze the GT-free columns;
4. join GT by stable rally key; and
5. append scoring fields and summaries.

GT must not choose an anchor, predecessor, high-shot state, threshold, category,
or stop reason.

Do not run threshold sweeps. In particular, do not compare 10, 30, 60, and 75
frame predecessor rules.

## Outputs

Keep the completed baseline files unchanged. Write separate files such as:

- `h3_r8_contact_evidence.csv.gz`;
- `h3_r8_search_results.csv.gz`; and
- `h3_r8_summary.json.gz`.

All three outputs now exist. Check mode rebuilt and exactly matched 3,200
contact rows, 239 GT-free search prefixes, the scored rows, and the summary.
The result is explained in `h3_r8_report.md`.

The result row should contain both searches, their selected frames and ranks,
the incoming anchor, the predecessor candidate, the accepted-contact gap, the
admission source, the final category, and the stop reason. Append GT and scoring
columns only after the GT-free prefix is complete.

Use explicit `--write` and `--check` modes. Check mode must rebuild and compare
the decompressed evidence, the GT-free result prefix, the scored rows, and the
summary.

## Implementation batches

### Batch H1: helper and focused tests

Add separate scratch-only helper and tests. Do not modify the completed baseline
helper unless sharing a genuinely identical primitive avoids correctness drift.

Cover:

- halo 3 clears distant halo-only frames while retaining recurrence core, flat
  core, repeated attractor positions, and stored blank points;
- halo 15 exactly reconstructs production codes;
- ratio 8.0 is inclusive and 8.01 fails;
- five-frame and two-frame local eligibility boundaries;
- three-way pre-contact verdicts and binary outgoing evidence;
- earliest incoming anchor ignores outgoing evidence;
- the 60-frame predecessor cap is inclusive;
- the high-shot endpoint buffer is inclusive and one frame beyond fails;
- the high-shot exception admits a candidate without claiming causation;
- predecessor `unavailable` remains distinct from `not incoming`; and
- no-anchor terminal states remain distinct.

### Batch H2: checked evidence builder

Build all accepted-contact evidence eagerly. Require 239 stable rally keys,
chronological frames, contiguous accepted ranks, and one evidence row per
accepted frame. Keep the GT-free input boundary structural.

### Batch H3: run and score

Run `--write`, then `--check`. Report both searches over all 239 rallies and the
baseline transition slices. Record high-shot admissions separately.

### Batch H4: adversarial review and gates

Ask a fresh read-only reviewer to check GT leakage, the guard reconstruction,
one-hop semantics, high-shot endpoint arithmetic, category denominators, and
saved-row reproducibility.

Run the focused scratch tests, Ruff, Pyrefly 1.1.1, and diff checks. Record every
exit code. Do not run repository-wide gates for this scratch-only extension.

## OUT-list

- production source and PR #82 source remain read-only;
- completed baseline code, result files, and report remain unchanged;
- no stride-1 TrackNet input in this experiment;
- no raw or rejected impulse candidates;
- no outgoing gate in Search B;
- no recursive backward chain;
- no cross-gap direction, trajectory-shape, or spatial-continuity test;
- no veto based on missing tracks, hallucinations, jumps, or guard failures in
  the gap;
- no claim that timing or high-shot state proves shot causation;
- no threshold tuning against GT;
- no production rally-start or rally-end change;
- no commit, push, merge, or PR without fresh user authority; and
- no overlay re-render as part of the numerical experiment.

## Verified measurements that motivated the fixed settings

- Code 3 currently includes both the recurrence halo and repeated attractor
  positions. Across the three fixtures, 53,471 of 55,937 code-3 frames are
  halo-only. Within saved rally spans, 12,207 of 12,879 are halo-only.
- Reconstructing halo 15 from saved guard codes exactly matched production on
  all three fixtures.
- Halo 3 changes 13,549 frames in `sset_01`, 15,161 in `sset_15`, and 8,429 in
  `sset_21` relative to halo 15.
- Clip 07's post trace contains 25 consecutive usable frames. Its largest-step
  ratio is 6.828, so missing detections did not inflate its denominator.
- Across the fixed population, 146 consecutive accepted-contact pairs bracket a
  measured high-shot state. Forty-two are beyond the ordinary 60-frame cap.
  A 12-frame endpoint buffer admits 9 of those 42.

These are descriptive checks, not GT-tuned parameter searches.

## Halt conditions

Stop and report if implementation reveals a conflict with these rules, if a
search action requires GT, if the halo-15 reconstruction fails, or if saved
evidence cannot be rebuilt exactly.
