# Accepted-contact trace runbook

The experiment is ready to implement. The runbook keeps search decisions separate from ground-truth scoring and limits all code, evidence, and reports to this investigation folder.

## Planning gate

- Consecutive accepted contacts connect from credible endpoint traces alone
- Gap contents never veto a connection and receive no cross-gap motion or spatial test
- The maximum contact gap is 75 base-30fps frames, inclusive
- The complete contact-gap distribution is recorded before GT scoring
- The user approved all behavioural choices; Python implementation may start

## OUT-list

- Production code and all PR #82 source files remain read-only because this pass tests a scratch hypothesis
- Raw and rejected impulse candidates remain out because the accepted sequence is the declared search space
- Production serve-start conditions, scorer edits, and production annotations remain out because they would mix diagnosis with redesign
- GT stroke frames, ordinals, and server labels never enter a search function or select a threshold because GT is scoring-only
- Threshold sweeps remain out because the experiment tests one fixed rule
- Learned models, dynamic programming, and a general contact graph remain out because they add machinery before the narrow rule is measured
- An implied serve has no invented frame because the trace only supports a before-contact event
- Existing experiment outputs remain unread and unchanged because the checked PR #82 counts are the fixed baseline
- Repository-wide checks remain out because the shared contract calls for focused scratch checks
- Pushes, merges, PR creation, and external write authority remain out because the current authority covers local investigation commits only

## Fixed source conventions

- Accepted frames come from sorted `VideoData.accepted_by_span[span_id]`
- Each local endpoint trace stays inside the half-open `VideoData.spans[span_id]` envelope and its own tracker scene; a pair may cross a scene boundary
- Recurrence-clean eligibility reuses the mask components recorded in `decisions.md`
- Local pre- and post-contact windows are 30 base-30fps frames
- A path needs 5 frames, a contact gap of at most 2 base-30fps frames, and `largest_step_ratio <= 4.0`
- Incoming is fitted decrease `>= 0.05` body heights; outgoing is fitted decrease `<= -0.05` body heights
- Credible outgoing and incoming traces may connect across missing shuttle frames without shared frames, adjacency, or a spatial seam check
- Production `high_shot_oob` and ordinary TrackNet dropouts use the same continuity rule
- Only consecutive accepted contacts are connection candidates
- Gap contents never veto a connection, including hallucinations, jumps, guard failures, and visible directional weirdness
- The consecutive contact gap must be at most 75 base-30fps frames, inclusive
- Missing evidence returns `not enough shuttle trajectory to tell`
- The continue-past-unknown result is stored separately as sensitivity-only
- A forward no-outgoing verdict lacks positive endpoint evidence and cannot reconnect

## Batch 0: record the approved rule

Files:

- `decisions.md`
- `evidence.md`
- `mechanisms.md`
- `runs.md`
- `worklog.md`
- `plan.md`
- `audit_index.md`

Change: fold the user's permissive consecutive-contact rule, positive endpoint requirement, 75-frame timing ceiling, and gap-distribution requirement into the source conventions.

Gate:

```bash
git diff --cached --check -- \
  scratch/serve_id_by_lookback_followup/decisions.md \
  scratch/serve_id_by_lookback_followup/evidence.md \
  scratch/serve_id_by_lookback_followup/mechanisms.md \
  scratch/serve_id_by_lookback_followup/runs.md \
  scratch/serve_id_by_lookback_followup/worklog.md \
  scratch/serve_id_by_lookback_followup/plan.md \
  scratch/serve_id_by_lookback_followup/audit_index.md
```

Exact commit message:

```text
Pin the accepted-contact trace experiment

Record the fixed path, continuity and state rules before implementation. Keep the GT boundary and unknown outcomes explicit so the analysis cannot tune itself against labels.
```

## Batch 1: implement and test the search

Files:

- `accepted_contact_trace.py`
- `test_accepted_contact_trace.py`

Change: add small pure helpers for post-contact runs, retained end reasons, direction checks, continuity, backwards trace state, and the main versus sensitivity outcomes. Search functions accept accepted frames and trajectory evidence only. They accept no GT field.

Focused cases:

- strict pre/post off-by-one and FPS-scaled gap boundaries
- 5-frame, 2-base-30fps, 4.0-ratio, and ±0.05-BH inclusive edges
- scene and span boundaries versus internal unusable gaps
- missing player and missing segment outcomes
- visible serve, forward junk skip, unknown stop, and implied serve
- missing intervals, including `high_shot_oob` and ordinary TrackNet dropouts, without seam-position tests
- gap contents that remain deliberately ignored and the inclusive 75-frame contact boundary
- source-frame and base-30fps contact-gap distribution records
- no-outgoing endpoints that correctly remain disconnected
- main and continue-past-unknown sensitivity separation
- chronological accepted contacts and stable final accepted rank

Gate:

```bash
~/.venvs/badminton-cicd/bin/pytest -q \
  scratch/serve_id_by_lookback_followup/test_accepted_contact_trace.py

~/.venvs/badminton-cicd/bin/ruff check \
  scratch/serve_id_by_lookback_followup/accepted_contact_trace.py \
  scratch/serve_id_by_lookback_followup/test_accepted_contact_trace.py

~/.local/bin/uvx --from pyrefly==1.1.1 --with jaxtyping==0.3.11 \
  pyrefly check \
  scratch/serve_id_by_lookback_followup/accepted_contact_trace.py \
  scratch/serve_id_by_lookback_followup/test_accepted_contact_trace.py
```

Reference checks:

- Serena references plus text search confirm the new helper call flow
- `accepted_contact_trace.py` contains no GT, truth, stroke-frame, production serve-start, raw-contact, or rejected-contact input
- a fresh read-only reviewer checks the diff against this runbook and OUT-list

Exact commit message:

```text
Add the accepted-contact trace search

Mirror the established trajectory measurements after each accepted contact and retain why a trace stops. Keep the primary unknown result separate from the continue-past-unknown sensitivity run.
```

## Batch 2: build the checked analysis

Files:

- `analyse_accepted_contact_trace.py`
- `test_accepted_contact_trace.py`

Change: run the GT-free search over every primary one-to-one rally, freeze all 239 search outcomes, then join GT by `(fixture, video_id, set_id, rally)` for scoring. Add a check mode that rebuilds search rows and compares them directly with the saved decompressed rows.

Saved evidence:

- `accepted_contact_trace_rows.csv.gz`
- `accepted_contact_trace_summary.json.gz`

Each row records the accepted sequence, every consecutive contact gap in source and base-30fps units, per-contact path verdicts and reasons, junk skips, backwards origins, implied serves, final accepted rank, main outcome, and sensitivity outcome. GT columns are appended only after the search result is complete.

Gate: repeat Batch 1 checks with `analyse_accepted_contact_trace.py` included, then run its synthetic I/O and GT-separation tests. A fresh read-only reviewer checks the diff and saved schema.

Exact commit message:

```text
Build the accepted-contact trace analysis

Run the fixed search before joining ground truth and save one checked row for every primary rally. Keep the 97 unmatched starts as a reporting slice rather than a search population.
```

## Batch 3: run, score, and report

Files:

- `accepted_contact_trace_rows.csv.gz`
- `accepted_contact_trace_summary.json.gz`
- `report.md`
- `runs.md`
- `evidence.md`
- `mechanisms.md`
- `worklog.md`
- `audit_index.md`

Change: run the rule once on all 239 rows before reading GT labels. Check the saved rows, calculate the fixed transition table, and write a short conclusion-first report with roughly five representative cases.

Required report slices:

- fixed, damaged, unchanged, and unknown over all 239
- the same relevant counts within the 97 currently unmatched starts
- junk skips, backwards origins, implied serves, selected accepted ranks, and final visible-contact GT ordinals
- ±10 primary scoring with compact ±5 and ±30 sanity checks
- main and continue-past-unknown sensitivity results kept separate
- the complete GT-free consecutive contact-gap distribution and counts below 75, exactly 75, and above 75 base-30fps frames

Gate:

- analysis command exits 0 and writes 239 unique stable keys
- check mode rebuilds and directly matches every decompressed search row
- all Batch 2 focused tests, Ruff, and Pyrefly checks pass
- `git diff --check` passes for the report and living records
- declared Claude Opus and Gemini 3.1 auditors review compact evidence for GT leakage, ordinal interpretation, threshold tuning, denominators, indirect truth use, and overclaiming
- genuine audit findings are fixed and recorded before the report is final

Exact commit message:

```text
Record the accepted-contact trace results

Save the checked 239-rally evidence and report the fixed rule's gains, damage and unknowns. Keep the unmatched-start slice, sensitivity run and audit findings explicit.
```

## Halt conditions

- Stop if code or data contradicts an approved decision
- Stop if the search needs GT to choose an action or number
- Stop if the full trace cannot distinguish an observable boundary from unavailable evidence
- Stop if a required check needs an unavailable external environment
- Stop after the short audited report; do not propose production architecture
