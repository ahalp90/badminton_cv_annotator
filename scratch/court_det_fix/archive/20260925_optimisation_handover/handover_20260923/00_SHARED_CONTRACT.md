> Historical record: the 23 September 2026 launch of the speed-up work, filed on
> 25 September 2026. The [speed-up README](../../../court_detector_optimisation_handover/README.md)
> owns current status; the [archive map](../../README.md) records the original
> paths. Content is unchanged.

# Court detector: shared contract for the next session

## Start here

Read this file and [the launch prompt](01_LAUNCH_OPTIMISATION.md), plus mandatory
repository instructions (`AGENTS.md`, `.github/AGENTS.md`, `.codex/context.md`).
Other investigation records are on-demand evidence. Do not reread the archive.
This packet supersedes historical next-step wording in those records.

## Aim and runtime requirement

Build a deployable, CourtKeyNet-free automatic court finder. Preserve the usable
court geometry established by the completed experiments while substantially
reducing compute. No manual paint, polarity, floor or player choices.

The user's latest expectation is roughly **30 seconds for a five-minute video**,
with **90 seconds a tolerable upper end**. Separate startup/warmup from processing
when reporting this target. These are approximate expectations, not established
service limits. They replace the rejected 5%-of-duration / 15-second proposal.
For longer videos, expensive court search should roughly follow the number of
distinct scene/view groups. The user calls this PySceneDetect histogram bagging;
the actual code uses content cuts and later perceptual-hash/alignment grouping.
The user expects few groups; measure this. Target hardware is unspecified.

## Working state and authority

- Checkout: `/home/ariel/Documents/COSC594/badminton_cv_annotator`.
- Branch: `fix/court-det`. Last pre-packet tip:
  `ce886198f96326cfc56b10c36dfb5bcab9a83a1c`; this packet's commit follows it.
  Verify actual HEAD/status; never reset to the recorded hash automatically.
- Behaviour checkpoint: `82c8a4d`; statistical checkpoint: `94b242c`.
- The user authorised ongoing experimental work, relevant tests, delegation and
  suitable checkpoint commits on this feature branch. No main commits or deploys.
- This handover drafts the next session; it launches no optimisation campaign.
  Opening the launch prompt starts scoped investigation and bounded trials.
- Preserve unrelated changes and untracked data. Stage explicit paths. Do not
  delete caches, rewrite history, bypass hooks, or overwrite historical outputs.
- Use separate trial output directories. Keep semantic optimisations apart from
  changes to candidate coverage, ranking or acceptance.

## Delegation and communication

Use the `external-delegate` skill. The coordinator owns conclusions and final
quality; delegate reports are leads until checked against source/results.

- Sol **high** for substantial implementation, experiment runners and galleries;
  prefer default/non-priority service where configurable. Assign separate files.
- Luna **max**, priority where supported, for tightly bounded mechanical reads,
  keyed comparisons and inventories. Do not delegate open causal judgement to it.
- Claude Code **Opus 5-5 xhigh** for bounded independent audit; high is also
  authorised. Native agent tools, no memory, no time limit. Anthropic may receive
  relevant project data. Do not silently substitute a model.
- Carmack compute and project-data transfer are authorised. Read
  `~/.codex/remote_hpc.md` first; one remote session at a time. Stop/report a failed
  connection rather than switching hosts. Git for code, rsync for large data.
- Serena/Pyrefly was used at `http://127.0.0.1:9121/mcp`. Check/reuse through the
  `serena-pyrefly` skill; provide access to delegates where useful. Report absence.
- Conserve Astra tokens through bounded delegation, without outsourcing judgement.
  Give concise progress updates; avoid provenance theatre and approval repetition.

## Evaluation and record keeping

Use paired statistics first. Reserve visual checks for meaningful or ambiguous
cases. The user accepts frequent tiny imperfections in amateur footage. Its
references mix paint centres and outer edges, so small signed drift is ambiguous.
The user judges the displayed net-selected, stripe-corrected gallery very usable;
this is gallery-level feedback, not 20 independent per-case labels.

Sol builds any needed gallery by literally reusing
`../svd_search/gallery_template.html`; no exhaustive review request. Limited
coordinator image inspection is authorised when it informs a decision.

Use the least complex suitable worklog/plan skill. Before launching workers,
create a single new optimisation session directory with `worklog.md` (Resume
first), `evidence.md`, `mechanisms.md`, and `runs.md` when commands run. Add an
`audit_index.md` only for delegated audit packets. Link these from pickup.
Update after material results and before/after long runs. Keep one final report.
