# Issue 32 update draft: Rally-start visibility and replay-sting feasibility

## Issue

Add this update to issue 32. Do not create a separate replay-sting issue yet.

## GitHub body

Issue #28 found that the tested raw-impulse plus central-pose serve-lookback
prototype recovered zero target first strokes. Human review also found a
different case: a broadcast can remain in replay while a physical serve occurs,
then return after the rally has started. There is no serve frame in the video
for a prepend feature to recover.

This issue measures whether recurring, match-local broadcast stings can bracket
replay intervals accurately enough to exclude replay footage and preserve the
visible remainder as a partial rally.

Issue #32 owns the rally-start visibility truth, replay-sting feasibility, and
partial-rally continuation measurement. This work does not replace the complete
`live`, `live-non-standard`, `replay`, `cutaway`, and `other` timeline.

### Current evidence

- The current issue 28 target is an unmatched first ShuttleSet stroke with at
  least one later matched stroke. It is not yet a verified missed visible
  serve.
- 26 of 136 target first rows carry ShuttleSet `flaw=1`. Twenty-five have an
  unknown stroke type.
- In `sset_21`, 24 of 34 target rows carry both signals.
- Removing all 26 flagged rows would leave zero recovery on 110 targets, so
  this does not revive the tested serve-prepend rule.
- In `sset_01`, 57 of 62 human replay intervals have `other` immediately on
  both sides. Sixty-one have `other` on at least one side.
- The same adjacency is not established in `sset_15` or `sset_21`. Their
  scene-assisted labels may have merged brief stings into cutaway intervals,
  or their broadcast packages may differ.
- The completed 32-row visibility pilot contains 19 visible contacts, 4
  broadcast-omitted starts, 8 off-frame contacts, and 1 uncertain contact.
- The pilot contains all 26 flaw-marked targets and six deterministic controls.
  It does not estimate visibility or omission prevalence for all 136 targets.
- `off-frame` is now separate from uncertainty. It means current-rally service
  action is present while physical contact falls outside the camera image.
- The accepted full-audit seeds contain all 136 rally-start keys. They preserve
  the 32 reviewed pilot rows and leave 104 rows pending.

Full evidence and methods:
`docs/scraper_pipeline/serve_prepend_lookback/broadcast_omitted_start_and_sting_evidence_20260809.md`.

### Measurement questions

1. How many human replay intervals have an entry sting, exit sting, and a
   visually matching pair?
2. How often does the same sting pattern enclose a break, montage, cutaway, or
   other non-replay interval?
3. What boundary error results from sting pairing?
4. What incremental replay coverage does it add over the current mask?
5. After a replay exit, does the broadcast return to setup, active rally,
   cutaway, or other footage?
6. Can a partial rally safely begin at the first supported live frame without
   claiming that an omitted serve was detected?

### Annotation boundary

Use separate event-level truth for rally-start visibility and sting pairs.
Keep the canonical broadcast timeline unchanged.

The completed first pilot used the existing tool against disposable timeline
copies and recorded event decisions separately. It justified the full
136-start audit, so a rally-start companion now provides atomic compact-row
saves, first-pending resume, one-step undo, exact markers, and validation. It
reads the canonical timeline without importing or calling its writer.

The pilot did not justify a full 179-replay audit. Replay-sting event state and
sampling remain a separate later decision.

Issue #73 remains closed while this separation holds. If the implementation
requires timeline splitting or range replacement, stop and reopen or replace
#73 before annotation.

### Deliverables

- A pinned event-truth schema and reload-checked gzip files.
- A completed 32-row quality/control pilot covering all 26 flaw-marked issue 28
  targets plus two unflagged transition controls per video.
- The existing 17-window candidate and practice audit retained in the human
  review package.
- A recording-only sting-pair measurement stratified by video and negative
  case.
- Visibility composition for the 136 issue-28 targets, plus prototype recovery
  on the visible subset.
- A partial-rally continuation measurement.
- One written go or no-go decision for a production sting signal.

### Acceptance criteria

- Every source video, timeline, GT table, mask, and generated file is pinned by
  metadata and checksum where available.
- Rally joins are one-to-one and select complete first rows. Pandas
  `GroupBy.first()` is prohibited because it skips nulls by column.
- Uncertain visibility and sting cases remain explicit.
- Sting precision, recall, boundary error, and denominators are reported per
  video and pooled.
- Existing replay-mask performance is the comparison baseline.
- Replays without matching stings and matching stings around non-replay
  footage are included.
- Canonical timeline files are byte-identical before and after event auditing.
- No production mask, segmentation, serve, attribution, winner, or pairing
  behaviour changes in this issue.

### Related issues

- #28: serve-lookback measurement
- #30: conditional serve-prepend implementation, not justified by current
  evidence
- #32: real rally-start and end semantics
- #38: later VLM scene filtering
- #73: deferred atomic timeline splitting, only a dependency if event auditing
  edits timeline intervals

### Implementation plan

`docs/scraper_pipeline/serve_prepend_lookback/broadcast_start_and_replay_sting_plan_20260809.md`

## Posting boundary

This text is a proposed update to issue 32. Posting it is outside the current
read-only GitHub scope and requires a separate explicit request from Curtis.
