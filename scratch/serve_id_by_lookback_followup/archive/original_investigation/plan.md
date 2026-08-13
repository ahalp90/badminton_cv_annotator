> ARCHIVED 2026-08-12: completed runbook. Current next step: `../../docs/next_steps.md`.

# Sequential accepted-contact opener runbook

## Approved extension

The original batches below document the completed baseline. The next run is
governed by `02_LAUNCH_H3_R8_DUAL_SEARCH.md`.

Its order is:

1. completed: implement and test the halo-3, ratio-8 evidence helpers;
2. completed: save evidence for every accepted contact;
3. completed: derive the outgoing-first and incoming-only searches from that
   frozen evidence;
4. completed: freeze GT-free results, then score them; and
5. completed: run a fresh adversarial review and final gates.

The extension OUT-list and halt conditions in the new launch file supersede the
baseline prohibition on backwards tracing only for the one-hop predecessor
check. The completed baseline files remain unchanged.

The experiment asks whether post-contact outgoing motion removes false early impulse openers. It then reuses the fixed PR #82 incoming-motion check to classify the first credible contact.

## Planning gate

- Scan accepted contacts from earliest to latest
- Skip every contact without credible outgoing motion
- Stop at the first contact with credible outgoing motion
- Classify that contact from its existing pre-contact incoming verdict
- Never reconnect contacts or override an earlier `no outgoing` verdict
- Keep unavailable pre-contact evidence as the selected contact's three-way classification result

## OUT-list

- Production code and all PR #82 source files remain read-only
- Raw and rejected impulse candidates remain out
- GT never enters a search function or selects a threshold
- Threshold sweeps, learned models, dynamic programming, backwards tracing, contact chains, and contact reconnection remain out
- Cross-gap direction, shape, spatial continuity, gap contents, and contact-gap caps remain out
- An implied serve has no invented frame
- Existing experiment outputs remain unread and unchanged
- Repository-wide checks remain out because the shared contract calls for focused scratch checks
- Pushes, merges, PR creation, and external write authority remain out

## Fixed source conventions

- Accepted frames come from sorted `VideoData.accepted_by_span[span_id]`
- Each local path stays inside the half-open rally span and its tracker scene
- Recurrence-clean eligibility reuses the mask components in `decisions.md`
- Local pre- and post-contact windows are 30 base-30fps frames
- A path needs 5 frames, a contact gap of at most 2 base-30fps frames, and `largest_step_ratio <= 4.0`
- Incoming is fitted decrease `>= 0.05` body heights
- Outgoing is fitted decrease `<= -0.05` body heights
- Missing post-contact evidence fails the binary credible-outgoing predicate
- Missing pre-contact evidence returns `not enough shuttle trajectory to tell`

## Batch 0: record the first approved rule

Commit `caa8207` records the superseded contact-reconnection planning state. The living documents retain that history in `worklog.md` and now follow the simpler final ruling.

## Batch 0b: simplify the approved rule

Files:

- `decisions.md`
- `evidence.md`
- `findings.md`
- `mechanisms.md`
- `runs.md`
- `worklog.md`
- `plan.md`
- `audit_index.md`

Change: remove backwards tracing, reconnection, cross-gap checks, the 75-frame cap, and the contact-gap distribution. Pin the chronological outgoing-motion search and the existing incoming-motion classification.

Gate:

```bash
git diff --cached --check -- \
  scratch/serve_id_by_lookback_followup/decisions.md \
  scratch/serve_id_by_lookback_followup/evidence.md \
  scratch/serve_id_by_lookback_followup/findings.md \
  scratch/serve_id_by_lookback_followup/mechanisms.md \
  scratch/serve_id_by_lookback_followup/runs.md \
  scratch/serve_id_by_lookback_followup/worklog.md \
  scratch/serve_id_by_lookback_followup/plan.md \
  scratch/serve_id_by_lookback_followup/audit_index.md
```

Exact commit message:

```text
Simplify the accepted-contact opener rule

Remove backwards origin tracing and all gap-connection machinery. Stop at the first accepted contact with credible outgoing motion, then reuse the fixed PR #82 incoming check to classify it as a visible serve or first visible post-serve contact.
```

## Batch 1: implement and test the search

Files:

- `accepted_contact_trace.py`
- `test_accepted_contact_trace.py`

Change: add small pure helpers for the post-contact run, shared path eligibility, the binary credible-outgoing predicate, the three-way pre-contact verdict, and the chronological search. Search functions accept accepted frames and trajectory evidence only. They accept no GT field.

Focused cases:

- strict pre/post off-by-one and FPS-scaled local gap boundaries
- 5-frame, 2-base-30fps, 4.0-ratio, and +/-0.05-BH inclusive edges
- missing player and missing segment outcomes
- non-credible outgoing skip and first credible outgoing stop
- incoming classification as first visible post-serve contact
- measured no-incoming classification as visible serve
- unavailable post evidence treated exactly like any other non-credible outgoing result
- unavailable pre evidence returning `not enough shuttle trajectory to tell`
- all non-credible outgoing contacts ending as no credible accepted contact
- chronological accepted contacts and stable final accepted rank
- absence of GT fields and all contact-connection machinery

Gate:

```bash
~/.venvs/badminton-cicd/bin/pytest -q \
  scratch/serve_id_by_lookback_followup/test_accepted_contact_trace.py

~/.venvs/badminton-cicd/bin/ruff check \
  scratch/serve_id_by_lookback_followup/accepted_contact_trace.py \
  scratch/serve_id_by_lookback_followup/test_accepted_contact_trace.py

~/.local/bin/uvx --from pyrefly==1.1.1 --with jaxtyping==0.3.11 \
  pyrefly check --search-path . \
  scratch/serve_id_by_lookback_followup/accepted_contact_trace.py \
  scratch/serve_id_by_lookback_followup/test_accepted_contact_trace.py
```

Reference checks:

- Serena references plus text search confirm the new helper call flow
- `accepted_contact_trace.py` contains no GT, truth, stroke-frame, production serve-start, raw-contact, or rejected-contact input
- a fresh read-only reviewer checks the diff against this runbook and OUT-list

Exact commit message:

```text
Add the sequential accepted-contact opener search

Skip accepted impulses without credible outgoing motion. Stop at the first credible outgoing contact and classify it from the existing three-way pre-contact incoming result.
```

## Batch 2: build the checked analysis

Files:

- `analyse_accepted_contact_trace.py`
- `test_accepted_contact_trace.py`

Change: run the GT-free search over every primary one-to-one rally, freeze all 239 search outcomes, then join GT by `(fixture, video_id, set_id, rally)` for scoring. Add a check mode that rebuilds search rows and compares them directly with the saved decompressed rows.

Saved evidence:

- `accepted_contact_trace_rows.csv.gz`
- `accepted_contact_trace_summary.json.gz`

Each row records the accepted sequence, each contact's binary credible-outgoing result, the selected contact's pre verdict, skipped contacts, final accepted rank, and search outcome. GT columns are appended only after the search result is complete.

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
- `plan.md`

Change: run the rule once on all 239 rows before reading GT labels. Check the saved rows, calculate the fixed transition table, and write a short conclusion-first report with roughly five representative cases.

Required report slices:

- fixed, damaged, unchanged, pre-contact-unknown, and no-credible-contact outcomes over all 239
- the same relevant counts within the 97 currently unmatched starts
- accepted contacts skipped for non-credible outgoing motion, selected accepted ranks, visible serves, implied serves, and final visible-contact GT ordinals
- +/-10 primary scoring with compact +/-5 and +/-30 checks

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

Save the checked 239-rally evidence and report the fixed rule's gains, damage and unknowns. Keep the unmatched-start slice and audit findings explicit.
```

## Halt conditions

- Stop if code or data contradicts an approved decision
- Stop if the search needs GT to choose an action or number
- Stop if a required check needs an unavailable external environment
- Stop after the short audited report; do not propose production architecture
