# Historical documents

[pickup.md](../pickup.md) is the current handover. These records preserve
completed work and the decisions at that time. Their dated resume and next-run
sections are historical. Each archived document points back here and to pickup.

The [23 September top-level refresh](20260923_top_level/README.md) preserves the
previous index, pickup, decision record and filepath map. Its disposition record
also documents every newly tracked input and explicitly local-only run output.

## Path map

| Former path under `court_det_fix/` | Current location and role |
| --- | --- |
| `evaluation_20260922.md` | [Four-comparison worklog](20260922/evaluation_20260922.md), preserved with rebased links |
| `evaluation_results_20260922.md` | [Four-comparison results](20260922/evaluation_results_20260922.md), preserved with rebased links |
| `wider_evaluation_20260922.md` | [71-case worklog and findings](20260922/wider_evaluation_20260922.md), preserved with rebased links |
| `DETECTOR_DECISIONS.md` before close-out | [Historical copy](20260922/DETECTOR_DECISIONS.md), including the completed wider-run plan; the [live decisions](../DETECTOR_DECISIONS.md) retain lasting findings |
| `WEBUI_FOLLOWUPS.md` | [Original task prompts](../evidence/webui_followups_20260922/PROMPTS.md); returns are linked from pickup |
| `TIDY_PLAN.md` | Retired pointer, removed at the user's request after the re-entry check. Its complete text remains in the close-out snapshot below |
| `followup3_svd_screen_completed.tar.gz` | [Unreviewed SVD returns](../evidence/webui_followup3_20260922/README.md) |
| `svd_court_evaluation.tar.gz` | [Unreviewed SVD returns](../evidence/webui_followup3_20260922/README.md) |
| `court_detector_optimisation_handover/CLAUDE_FOLLOWUPS.md` | [Speed-up list](20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md): every speed-up tried to 25 September, with measurements and reasons. The [speed-up README](../court_detector_optimisation_handover/README.md) keeps current status |
| `court_detector_optimisation_handover/CLAUDE_EVALUATION.md` | [Evaluation of the web-UI packet](20260925_optimisation_handover/CLAUDE_EVALUATION.md), 24 September profiles and verdicts on F1–F12 |
| `court_detector_optimisation_handover/README.md` before 25 September | [Web-UI packet overview](20260925_optimisation_handover/webui_packet/README.md). The folder's README is now the speed-up README |
| `court_detector_optimisation_handover/STATIC_AUDIT.md` | [Static audit](20260925_optimisation_handover/webui_packet/STATIC_AUDIT.md), which defines F1–F12 |
| `court_detector_optimisation_handover/PATCH_QUEUE.md` | [Patch queue](20260925_optimisation_handover/webui_packet/PATCH_QUEUE.md), the packet's ordered plan P0–P11 |
| `court_detector_optimisation_handover/EXPERIMENT_PROTOCOL.md` | [Experiment protocol](20260925_optimisation_handover/webui_packet/EXPERIMENT_PROTOCOL.md) |
| `court_detector_optimisation_handover/SOURCE_MAP.md` | [Source map](20260925_optimisation_handover/webui_packet/SOURCE_MAP.md) of the 23 September hot path |
| `court_detector_optimisation_handover/court_detector_optimisation_handover.md`, `LOCAL_MODEL_PROMPT.md`, `task_manifest.json`, `tools/` | Deleted on 25 September: a copy of the packet, and the prompt, task list and helpers for a session that has run. Git history has them at commit 92535b6e |
| `handover_20260923/` | [Shared contract](20260925_optimisation_handover/handover_20260923/00_SHARED_CONTRACT.md) and [launch prompt](20260925_optimisation_handover/handover_20260923/01_LAUNCH_OPTIMISATION.md) that started the speed-up work on 23 September. The [speed-up README](../court_detector_optimisation_handover/README.md) keeps current status |

For D06's 24/27 development judgement, follow the four-comparison results.
For D09's 47-case plus 24-control result, follow the wider worklog's
[completed numeric comparison](20260922/wider_evaluation_20260922.md#completed-numeric-comparison).
Current priorities have moved to pickup; measurements and historical reasoning
remain here intact. Candidate pools, source inputs and experiment scripts did
not move during this pass.

## Recovery and close-out checks

A fresh reader starting only at pickup recovered the next test, controls,
stopping rule, D06 evidence, rerun command and return status within two link
hops. No competing current handover was found. All five historical documents
were compared with the snapshot: only notices and relative links changed.
Both supplied archive byte sequences match. The local link/anchor check and
`git diff --check` passed with exit 0. No runtime tests were needed for filing.

Local `.recovery/session-closeout-20260922.tar.gz` preserves all original
top-level Markdown, `.gitignore` and the two received archives. Its 12 members
were checked byte-for-byte before filing. The separate
`.recovery/session-closeout-links-20260922.tar.gz` preserves documents before
link updates. Both paths are relative to `court_det_fix/`. The compressed
`session-closeout-20260922-manifest.json.gz` beside them records the disposition
and original SHA-256 values. Existing large data/recovery trees were excluded
from the snapshot because this pass did not alter them.

Recover one original file from the repository root without unpacking a tree:

```bash
tar -xOf scratch/court_det_fix/.recovery/session-closeout-20260922.tar.gz \
  scratch/court_det_fix/evaluation_20260922.md
```

Historical literal paths inside code blocks and frozen worklogs retain their
original meaning; use this map for renamed top-level documents. The original
21 September tidy archives remain documented in [INDEX](../INDEX.md#recovery).

## Earlier sealed recovery

Paths in this section are relative to `court_det_fix/`.

The local `.recovery/disposition.csv.gz` ledger maps original paths to
retained evidence or the sealed backup. The archive
`.recovery/court-before-cleanup-20260921.tar.gz` preserves 7,746 original
files under repository-relative names. Every member was checksum-verified
before working copies were removed. Original worklogs are intact inside it.

To read the original source for decision D01 without restoring a session tree,
run this from the repository root:

```bash
tar -xOf scratch/court_det_fix/.recovery/court-before-cleanup-20260921.tar.gz \
  docs/courtkeynet/fallback_evaluation/README.md
```

Use the same command with any original path in the ledger. The original
pre-tidy plan is separately sealed in `.recovery/planning-input.tar.gz`.
Keep both archives. The later `.recovery/w5-preparation-20260922.tar.gz`
preserves the completed W5 campaign, its worktree, handovers, check-ins,
earlier monitoring records and the filing-time launcher/comparator sources.
Its retired members were compared against the working copies before removal.
Use the same `tar -xOf` command with this archive to recover a preparation file.
The separate `.recovery/w5-monitor-20260922.tar.gz` preserves the visible
Luna launch brief and launcher from `local_scratch/external_delegate/`.
`.recovery/w5-compatibility-links-20260922.tar.gz` preserves the former
symlink names and targets; their evidence targets remain in place.
Model assets remain in `local_scratch/court_line_models/`.
