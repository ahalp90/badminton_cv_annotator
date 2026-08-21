> ARCHIVED 2026-08-12: historical rule sheet. Current position: `../../HANDOVER.md`.

# Shared contract for accepted-contact rally openers

Run this as one fresh investigation session.

## Latest extension

The completed sequential experiment now has an approved scratch-only H3/R8
extension. After compaction, read `02_LAUNCH_H3_R8_DUAL_SEARCH.md` instead of the
historical `01_LAUNCH_ACCEPTED_CONTACT_TRACE.md`.

The extension keeps the completed baseline unchanged. It freezes trajectory
evidence for every accepted contact, reruns the outgoing-first search with a
three-frame recurrence halo and an 8.0 step-ratio limit, then runs the approved
incoming-only predecessor check. Its fixed predecessor rules are 60
base-30fps frames or the measured high-shot-out-of-bounds exception specified in
the new launch file.

## Verified baseline

- Repository: the repository root
- Branch: `investigation/serve-id-by-lookback-followup`
- Base tip: `4f9703f339e2f9821d986d376dbfca9d6fd18ad7`
- Population: all 239 one-to-one rallies across `sset_01`, `sset_15`, and `sset_21`
- Stable row key: `(fixture, video_id, set_id, rally)`
- Current ±10 first-impulse labels: 119 contact 1, 19 contact 2, 4 later, 97 unmatched
- Evidence root: `scratch/serve_id_by_lookback_followup/`

## Context policy

At startup read only this contract, `02_LAUNCH_H3_R8_DUAL_SEARCH.md`, the Resume block in `worklog.md`, `.github/AGENTS.md`, and `.codex/context.md`.

Open `findings.md`, `decisions.md`, PR #82 files, prior delegate output, and the original `Scope.md` only when the launch prompt names them. Later user rulings in this contract and the launch prompt override conflicting experiment text in `Scope.md`.

Never read `.env`, credentials, or `.claude/`. The extension may read the
completed baseline evidence named in its launch file and the new H3/R8 outputs
that it creates. Do not browse unrelated experiment outputs.

## Writing and voicing gate

The user is cognitively overloaded, not inexperienced. Make the work easy to
take in without hiding technical facts.

Every report, review, decision note, and final summary must:

- open with the few main ideas or choices the user needs to consider
- reveal supporting detail progressively, with technical evidence below the plain summary
- sound like a normal person speaking and use the simplest word that stays precise
- avoid project-management language, inflated framing, and technical presentation that makes a simple idea feel complicated
- treat corrections as ordinary parts of the settled account unless the correction will still matter in two weeks

Editing and voicing reviews use this section as their main standard. A document
that is technically complete but cognitively heavy has not passed.

## Exact roster

- Coordinator: fresh Codex `gpt-5.6-sol`, high effort, integration and final judgement
- Source worker: headless Codex `gpt-5.6-luna`, max effort, read-only, one narrow question at a time, 20,000 tokens, 30 minutes
- Final auditor: direct-host agy `claude-opus-4-6-thinking`, model-defined effort, read-only compact evidence, 20,000 tokens, 30 minutes
- Final auditor: direct-host agy `gemini-3.1-pro-high`, high effort, read-only compact evidence, 20,000 tokens, 30 minutes
- Writing cold read when needed: direct-host agy `gemini-3.6-flash-high`, high effort, one document only, 12,000 tokens, 20 minutes
- Fallback: coordinator verifies locally with Serena/Pyrefly and text search; do not broaden or silently change models

The user authorised sharing this public repository's project work with Codex external delegates and agy. No external worker has write, commit, push, or credential authority.

## Boundaries

- Keep production code and PR #82 files read-only
- Keep all new files under `scratch/serve_id_by_lookback_followup/`
- Use accepted impulses only
- GT scores results after the search; GT never selects search actions or thresholds
- Keep delegation linear and stop workers that expand beyond their named question
- Run Ruff and Pyrefly only on files added or edited under this investigation folder. Run focused tests where the scratch experiment code warrants them. Do not run repository-wide gates
- The user authorised the final H3/R8 state commit on 2026-08-11. Do not commit
  further changes, commit to `main`, push, merge, or open a PR without separate
  authority

## Persistent state

Keep `evidence.md`, `mechanisms.md`, `runs.md`, `worklog.md`, and `audit_index.md` current. Update the Resume block before long commands, delegation, compaction, or handoff.
