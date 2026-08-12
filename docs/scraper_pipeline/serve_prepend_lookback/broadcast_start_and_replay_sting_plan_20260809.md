# Broadcast-start and replay-sting work plan

## Resume block

- **Next action:** Curtis and Ari review the
  [`issue_28_serve_lookback_decision_20260809.md`](issue_28_serve_lookback_decision_20260809.md)
  closeout recommendation. Do not begin Phase 3 until Curtis records whether
  the optional Issue 32 audit should continue.
- **Current state:** Curtis accepted the completed Phase 1 pilot and chose the
  full 136-row rally-start audit. Phase 2 Batches 4 and 5 are implemented. Each
  independent review found one low-severity stale resume instruction and no
  code or data-contract defect. Both stale instructions are corrected. No
  production behaviour, replay-sting audit, or GitHub issue changed. The
  evidence-backed Issue 28 decision brief is ready for Ari's review.
- **Verified:** complete source joins reproduce 136 targets, 26 flaw rows, and
  25 unknown first types. The 32 reviewed decisions contain 19 visible, 4
  broadcast-omitted, 8 off-frame, and 1 uncertain row. The related suite passes
  with 142 tests and 6 skips. The three decision seeds contain 136 rows, with
  32 reviewed and 104 pending, while generated files reproduce byte-for-byte.
- **Runbook in play:** Phase 2 below and
  [`rally_start_visibility_audit_runbook_20260809.md`](rally_start_visibility_audit_runbook_20260809.md).
- **Working tree:** `worktrees/issue-32-rally-start-replay-sting`.
- **External worklog:**
  `cosc595/issue-32-rally-start-replay-sting/phase1_plan_and_worklog_20260809.md`.

## Goal

Measure which ShuttleSet rally starts contain a visible serve, and determine
whether repeated match-local broadcast stings can bracket replay footage while
preserving a partial rally after the broadcast returns.

## Concerns and observations

- `sset_21` supplies 24 unknown and flaw-marked first strokes among 34 current
  target rows. The meaning of those rows must be visually adjudicated.
- `sset_15` contains observed broadcast-omitted starts even though none of its
  39 target first rows carries the upstream flaw flag. The flag cannot be the
  only exclusion rule.
- The current broad timeline cannot represent `serve visible` or `broadcast
  omitted`. Those are event attributes, not mutually exclusive frame classes.
- Issue 73 describes a real interval-splitting defect. This work must not rely
  on editing or splitting the canonical timeline. If an implementation needs
  timeline surgery, stop and reopen or replace issue 73 before annotation.
- `sset_01` strongly exhibits `other -> replay -> other`. The same adjacency is
  not established in the other two timelines.
- A replay exit can lead to active rally footage. Excluding the replay must not
  discard the visible remainder of that rally.

## Issue routing decision

| Issue | Decision |
| --- | --- |
| [#28](https://github.com/ahalp90/badminton_cv_annotator/issues/28) | Finish the current measurement with precise wording: unmatched first ShuttleSet stroke, plus the visible-serve limitation. |
| [#30](https://github.com/ahalp90/badminton_cv_annotator/issues/30) | Do not implement. Close after the issue 28 result and adversarial review are accepted. |
| [#32](https://github.com/ahalp90/badminton_cv_annotator/issues/32) | Own real rally-start semantics, broadcast-omitted starts, rally-start event truth, replay-sting feasibility, and partial-rally continuation. |
| [#38](https://github.com/ahalp90/badminton_cv_annotator/issues/38) | Link as a possible downstream consumer. The sting feasibility pass is deterministic and should precede VLM work. |
| [#73](https://github.com/ahalp90/badminton_cv_annotator/issues/73) | Keep closed while event annotations are written separately. Reopen only if timeline intervals must be edited. |
| New issue | Do not create one now. Keep the accepted work under issue 32. |

## Existing 17-window audit

Keep the existing
[`serve_prepend_annotation_audit_runbook_20260809.md`](serve_prepend_annotation_audit_runbook_20260809.md)
in the handoff. It checks 19 evidence-passing candidate frames grouped into 15
regions, plus two `sset_15` practice intervals.

This narrow audit answers:

- whether selected candidates are live, replay, cutaway, or non-standard live;
- whether two pre-rally practice intervals follow one class policy; and
- whether the mask interpretation in the issue 28 report needs correction.

It does not answer whether the physical serve is visible. Run it as one part
of the larger audit rather than discarding it.

## Annotation data contracts

Do not add event columns to the complete broadcast timeline CSV. Use separate
tables so their grains remain explicit.

### Rally-start visibility table

One row per audited ShuttleSet rally:

```text
video_id
fps
frame_count
set_id
rally
gt_first_frame
gt_first_type
gt_first_flaw
review_status             pending | reviewed
serve_visibility          visible | broadcast-omitted | off-frame | uncertain
visible_serve_frame       nullable
first_visible_rally_frame nullable
broadcast_return_frame    nullable
preceding_truth
confidence                certain | uncertain
review_note
```

Required rules:

- `visible` requires a visually observable service contact.
- `broadcast-omitted` requires evidence that live rally footage begins after
  the physical service action.
- `off-frame` requires evidence that current-rally footage includes the service
  action while the physical contact falls outside the camera image.
- `uncertain` is retained in the output and excluded from prototype recovery
  on visible targets.
- Frame fields use the review video's zero-based frame numbering.
- The source video metadata must match the canonical timeline metadata.
- Pending templates use `review_status=pending` with blank decision fields.
  Reviewed rows use `review_status=reviewed` and must satisfy the conditional
  frame-field rules above.
- `broadcast_return_frame` is the first frame of the shot returning from
  non-live footage to the current rally. `first_visible_rally_frame` is the
  first frame at or after that return where current-rally play is supported.
- `visible` requires only `visible_serve_frame`; both omitted-start markers
  remain blank.
- `broadcast-omitted` requires both omitted-start markers, leaves
  `visible_serve_frame` blank, and requires
  `broadcast_return_frame <= first_visible_rally_frame`.
- `off-frame` leaves all three frame markers blank, uses certain confidence,
  and requires a note explaining the camera boundary.
- `uncertain` leaves all three frame markers blank, uses uncertain confidence,
  and requires a note.
- Every recorded marker must lie inside its row's review window and video
  bounds.

### Replay-sting table

One row per audited human replay interval:

```text
video_id
fps
frame_count
replay_start_frame
replay_end_frame
entry_sting_start_frame   nullable
entry_sting_end_frame     nullable
exit_sting_start_frame    nullable
exit_sting_end_frame      nullable
entry_sting_present       yes | no | uncertain
exit_sting_present        yes | no | uncertain
same_sting_template       yes | no | uncertain | not-applicable
post_replay_state         setup | active-rally | cutaway | other | uncertain
confidence                certain | uncertain
note
```

## Annotation-tool decision

Do not extend the existing broad timeline editor before the pilot. Use its
current video, timeline overlay, trackbar, and proposal navigation against a
disposable copy of each canonical timeline. Record decisions in the separate
rally-start table. The GUI has no enforced read-only mode, so its `--out-csv`
must never point at a canonical timeline during this audit.

The pilot justifies reviewing all 136 target starts. It does not yet justify
reviewing all 179 replay intervals. Build a small rally-start event companion
first. It may reuse read-only video and drawing helpers, but it owns an
event-row state model rather than `TimelineSession` interval surgery.

The rally-start companion must provide:

- proposal-row navigation and resume at the first unreviewed row;
- keys for visible, omitted, off-frame, and uncertain serve states;
- explicit cursor capture for visible serve, first visible rally, and broadcast
  return;
- atomic row save and one-step undo;
- a status overlay showing all pending exact bounds;
- validation-only mode; and
- no write path to the canonical broadcast timeline.

### Issue 73 gate

Before implementing the companion, trace every write call. If it calls
`TimelineSession.commit_interval`, `replace_range`, or another operation that
edits the complete frame partition, stop. Reopen issue 73 or implement its
atomic replacement contract first. A proposal overlay or read-only timeline
draw does not trigger this gate.

## Phased work

### Phase 0: independent adversarial review, complete

Review the current worktree read-only against `origin/main`. Required review
questions:

1. Does manual broadcast truth affect candidate selection, matching, or
   injection anywhere outside the recorded `manual_truth` field?
2. Does the 136-row target really mean first ShuttleSet stroke unmatched while
   later strokes matched?
3. Reproduce the 26 flaw and 25 unknown counts from complete source rows.
4. Confirm that excluding any target subset leaves zero recovery for the tested
   prototype.
5. Check whether `flaw=1` has a stronger documented meaning than source quality
   concern. Do not equate it with broadcast omission without evidence.
6. Reproduce replay adjacency counts and inspect canonicalization effects.
7. Confirm that the proposed event tables cannot corrupt the complete timeline.
8. Check issue routing for duplication with issues 30, 32, 38, and 73.

The review must record initial and final Git tree-state digests and make no
project changes.

The accepted review found that the 26 flaw rows are a quality stratum rather
than a prevalence sample. It also found two `sset_01` rallies whose first two
strokes share one frame. Phase 1 therefore selects exactly one
`ball_round == 1` row and verifies that its frame equals `gt_serve_frame`.

### Phase 1: target extraction and quality/control pilot

- Add a deterministic extractor for the target rally-start guide.
- Select exactly one `ball_round == 1` source row per target and verify its
  frame against the committed `gt_serve_frame`. Frame-only sorting is
  insufficient because two source rallies have tied first and second frames.
- Include current human class, prior class, live-transition frame, raw type,
  and flaw flag in each guide row.
- Validate keys, counts, bounds, metadata, and source hashes.
- Review the 26 flaw-marked target starts as a source-quality stratum.
- Add two deterministic unflagged transition controls per video. These controls
  validate the workflow and do not estimate omission prevalence.
- Keep all 136 target rows available. The previously observed `sset_15`
  omitted-start row is not pinned in repository files.
- Run the existing 17-window candidate and practice audit during the same human
  review cycle.

Pilot exit decision:

- Use the pilot to validate the decision contract, review procedure, and human
  time per row.
- Do not report omitted-start prevalence from the 32 pilot rows.
- Decide whether to review all 136 targets from the value of a complete target
  composition and the measured review cost.

Completed pilot result:

- The 32 reviewed rows contain 19 visible contacts, 4 broadcast-omitted starts,
  8 off-frame contacts, and 1 uncertain transition-obscured contact.
- These counts describe the 26-row quality stratum and six workflow controls.
  They do not estimate omission prevalence among all 136 targets.
- The pilot justifies keeping `off-frame` separate from uncertainty. It also
  confirms that the complete 136-row audit is required before reporting target
  visibility composition.
- Human review time was not recorded, so this pilot cannot support a reliable
  estimate for the remaining 104 rows.

### Phase 2: rally-start event-audit companion, accepted

- Treat the full target rows and canonical timeline as immutable inputs. Write
  only compact decision rows keyed by `(video_id, set_id, rally)`.
- Add deterministic per-video decision seeds that preserve the 32 reviewed
  pilot rows and expose the remaining 104 rows as pending.
- Extract one shared four-state decision contract for the guide builder and
  companion. It must validate pending blanks, reviewed notes, confidence,
  nullable markers, marker ordering, review-window bounds, and video bounds.
- Use a strict atomic writer: write a sibling candidate, reload and compare it,
  then replace the destination. A failed save must not change the destination,
  in-memory rows, cursor, or undo state.
- Resume at the first pending key. Keep proposal navigation independent of
  frame position because review windows overlap.
- Reject any decision output that aliases an input or uses a canonical
  broadcast-timeline filename.
- Validation-only must create no output and open no OpenCV window.
- Cover resume, atomic save, one-step saved-row undo, invalid bounds, nullable
  markers, metadata mismatch, path protection, and validation in unit tests.
- Prove the canonical timeline files are byte-identical before and after
  initialization, save, undo, resume, and validation-only paths.
- Run a fresh adversarial review after the state/persistence batch and again
  after the GUI batch. Report detailed verified findings before Phase 3.

Replay-sting state remains outside Phase 2. It needs a separate pilot because
its interval key, four sting bounds, paired-presence states, template decision,
and post-replay state differ from rally-start decisions.

### Phase 3: complete human audit

- Review all 136 issue 28 target starts.
- Review the 15 candidate regions and two practice intervals from the existing
  runbook.
- Audit all 179 human replay intervals for entry and exit stings if the pilot
  supports full sting measurement. Otherwise use a pre-declared stratified
  sample and report the sample grain.
- Retain uncertain rows. Do not force a binary decision.

### Phase 4: recording-only measurement

Report per video and pooled:

- visible service starts, omitted starts, and uncertain starts;
- prototype recovery on the visible subset of the 136 issue-28 targets;
- partial-rally capture after omitted starts;
- replay intervals with entry sting, exit sting, and matching sting pair;
- paired-sting precision against human replay intervals;
- replay-pair recall;
- boundary error in frames;
- negative cases such as stings around breaks, montages, or non-replay
  cutaways; and
- incremental value over the existing replay mask.

Keep the broad timeline, event truth, current mask, and proposed sting signal
as separate arrays or tables.

### Phase 5: implementation decision

Only propose a production change if the recording-only result supports it.
The likely production shape is:

1. detect or cluster a transition template within one match;
2. pair plausible entry and exit occurrences;
3. confirm the enclosed interval with existing replay evidence;
4. keep stings and confirmed replay frames excluded; and
5. open a partial rally at the first supported live frame after the exit.

Do not use one global sting template across tournaments or broadcasters.

## Verification ladder

| Risk | Gate |
| --- | --- |
| Source-row join | Exact one-to-one `(video_id, set_id, rally)` join and independent row-count reproduction |
| Null first-stroke fields | Regression proving whole-row selection does not borrow later non-null values |
| Event CSV integrity | Allowed enums, nullable-field rules, unique keys, ordered bounds, metadata equality, gzip round trip |
| Timeline safety | Hash or byte equality of all three canonical timeline files before and after audit sessions |
| Measurement drift | Existing issue 28 summary reproduces before the new event truth is consumed |
| Sting feasibility | Per-match precision, recall, boundary error, and negative strata with explicit denominators |
| Partial-rally semantics | Cases returning to setup, active rally, cutaway, and other are tested separately |

## OUT list

- Production serve prepend from issue 30. The tested design recovered no target.
- Changes to the canonical broad-timeline CSV schema.
- Reopening issue 73 unless timeline interval edits become necessary.
- VLM implementation from issue 38.
- A tournament-wide or broadcaster-wide fixed sting template.
- Automatic removal of a whole rally because its prefix was not broadcast.
- Forced labels for uncertain human cases.
- Threshold tuning on the same rows used for final evaluation without a
  declared split.
- Replay-sting GUI or detector work before a separate replay pilot is accepted.
- Refactoring the mutable timeline editor into a generic event framework.
- Automatically inferred human decisions, copied notes, or event prefill from
  nearby rows.

## Proposed commit batches

These messages are drafts for Curtis to approve before execution:

1. `Clarify observable serve targets in the issue 28 measurement`
2. `Add deterministic rally-start visibility audit guides`
3. `Record rally-start visibility pilot decisions`
4. `Add safe rally-start event audit state`
5. `Add rally-start event review companion`
6. `Measure replay-sting pairing and partial-rally continuation`, only after a
   separate replay pilot and scope decision.

Do not combine production replay-mask changes with these recording-only
batches.

## Handoff to the next agent

The next agent should read, in order:

1. repository `AGENTS.md`, `.github/AGENTS.md`, and
   `.codex-rules/badminton.md`;
2. the external Phase 1 plan and worklog named in the resume block;
3. this plan;
4. [`broadcast_omitted_start_and_sting_evidence_20260809.md`](broadcast_omitted_start_and_sting_evidence_20260809.md);
5. [`serve_prepend_lookback_20260808_measurement.md`](serve_prepend_lookback_20260808_measurement.md);
6. [`serve_prepend_annotation_audit_runbook_20260809.md`](serve_prepend_annotation_audit_runbook_20260809.md); and
7. [`issue_32_rally_start_replay_sting_update_draft_20260809.md`](issue_32_rally_start_replay_sting_update_draft_20260809.md).
8. [`issue_28_serve_lookback_decision_20260809.md`](issue_28_serve_lookback_decision_20260809.md).

Phase 0 and Phase 1 are accepted. Phase 2 implementation and both independent
reviews are complete. The Issue 28 decision brief is ready for Curtis and Ari.
Do not begin Phase 3, create a new issue, edit a canonical timeline, or begin
replay-sting annotation or detector work without a new recorded decision.
