# Court-detector tidy plan

## Decision requested

Approve, amend or reject the disposition rows below. This document is the only
tidy plan. Nothing has moved or been deleted.

Before execution, create one recoverable `tar.gz` snapshot under
`local_scratch`. Worklogs remain intact. Every deletion requires approval of
the named row.

## Inventory summary

The whole scoped workspace occupies 7.6 GB. The tracked
`scratch/court_det_fix` subtree accounts for 6.0 GB, 6,640 files, 267 Markdown
files and 358 Python files. The rest is split across repository documentation,
reusable experiment code and four ignored working areas created on 20--21
September.

| Area | Size | Files | Main contents |
| --- | ---: | ---: | --- |
| `worklog/` | 3.3 GB | 5,068 | historical checks, Web UI sessions, reviews, source media and worklogs |
| `w5_holistic/` | 1.2 GB | 726 | active W5 code plus six run directories |
| `line_identity/` | 826 MB | 457 | active matcher/replay code plus one large run |
| `direction_agreement/` | 745 MB | 200 | completed experiment and its run |
| `next_steps_20260916/` | 43 MB | 93 | completed C1/C2/L1/L2/L3 work and the W5 seed source |
| `frozen_views/` | 33 MB | 25 | live frozen packs, frames and provenance |
| `frozen_helpers_20260914/` | 796 KB | 70 | live helper snapshot, currently missing one dependency |
| `docs/courtkeynet/fallback_evaluation/` | 5.8 MB | 14 | five result narratives, one worklog, a check script, recorded inputs and figures |
| `experiments/annotator/independent_court/` | 171 MB | — | reusable experiment code and recorded evidence |
| `local_scratch/add_missing_court_candidates_20260920/` | 78 MB | 42 | yesterday's admission audit, scripts and raw arrays |
| `local_scratch/campaigns/w5-line-admission/` | 1.4 GB | — | today's live campaign; 1.3 GB is a clean nested Git worktree |
| `local_scratch/runs/broadcast_junction_box_repair_20260921/` | 40 KB | 1 | byte-identical copy of a tracked repair result |
| relevant `local_scratch/external_delegate/2026092{0,1}-*/` | 1.4 MB | — | launch wrappers and review returns from yesterday and today |

Thirteen files under `experiments/annotator/independent_court/` changed on this
branch. The fallback-evaluation documentation predates the branch, but it is
part of the same investigation history and contributes to the current
navigation problem. It is therefore in scope. Other repository documentation
and older unrelated `local_scratch` work remain out of scope.

## Target structure

Keep the two active code paths stable. Remove run data and historical session
material from their live surface.

```text
scratch/court_det_fix/
├── INDEX.md                       one thin entry point
├── pickup.md                      the only live state note
├── w5_holistic/                   active code, tests and current decisions only
├── line_identity/                 active code and tests only
├── frozen_views/                  immutable inputs
├── frozen_helpers_20260914/       immutable helper snapshot
├── data/
│   ├── w5_runs/
│   ├── line_identity_runs/
│   ├── direction_agreement_run/
│   ├── historical_checks/
│   └── webui_followup_records/
├── archive/
│   ├── direction_agreement/
│   ├── next_steps_20260916/
│   ├── worklogs/
│   ├── sessions/
│   └── handovers/
└── scripts/                       reusable measurement and migration tools

docs/courtkeynet/fallback_evaluation/
├── README.md                     short durable overview and sole entry point
├── evidence/                     the five existing result records, intact
├── archive/                      the historical worklog
└── figures/                      figures referenced by the evidence

local_scratch/campaigns/w5-line-admission/
├── RESUME.md                     live pickup note until campaign close
├── PROJECT_STATE.md              compact state and deployment account
├── W5_GO.md                      current remote authority
├── w5_launch_packet.md           current launch contract
├── evidence/                     unique ignored inputs and compact results
└── archive/                      closed remote handoffs and review records
```

Every `data/` subdirectory gets a short README naming its producer, date,
machine, source inputs, rerun command where known, and whether it is safely
regenerable. `archive/ARCHIVE_MAP.md` maps every old path to its new home.

## Disposition manifest

### Active code and current evidence

| ID | Current path | Disposition | Target or consolidation | Gate |
| --- | --- | --- | --- | --- |
| A01 | `w5_holistic/*.py`, `*.sh`, tests | Keep | Keep at the current path while W5 remains active | focused tests and reference grep |
| A02 | `w5_holistic/steering_record.md` | Keep | Current W5 decision record | W5 result incorporated |
| A03 | `w5_holistic/box_provenance_impact.md` | Keep | Historical-impact record | repaired player matcher and final Opus ruling incorporated |
| A04 | `w5_holistic/directional_admission_cache_design.md` | Archive | `archive/handovers/w5/` as an unimplemented design | confirm no open task depends on it |
| A05 | `line_identity/*.py`, tests | Keep | Keep at the current path until the repaired five-case matcher closes | focused tests and reference grep |
| A06 | `line_identity/results.md`, `evidence.md` | Consolidate | one `line_identity/README.md` for current result and interface; originals move intact to `archive/line_identity/` | claim-by-claim extraction check |
| A07 | `line_identity/worklog.md`, `runs.md` | Archive intact | `archive/worklogs/line_identity/` | extract every open item first |
| A08 | `line_identity/prior_checks/` | Archive | `archive/line_identity/prior_checks/` | inbound-reference rewrite |
| A09 | `line_identity/inputs/` | Move | `data/line_identity_inputs/` | update configured paths and smoke input loading |
| A10 | `frozen_views/` | Keep | Current path | exact pack/frame/provenance tests |
| A11 | `frozen_helpers_20260914/` | Keep then freeze | Current path; add the missing camera helper before declaring complete | import smoke and helper inventory |

### Run data

| ID | Current path | Size | Disposition | Target | Gate |
| --- | --- | ---: | --- | --- | --- |
| D01 | `w5_holistic/runs/line_template_regression_20260920/` | 514 MB | Move, retain | `data/w5_runs/` | result, manifest, rulings and gallery open after move |
| D02 | `w5_holistic/runs/w5_stage2_20260920/` | 314 MB | Move, retain | `data/w5_runs/legacy_stages/` | record its surviving claim in data README |
| D03 | `w5_holistic/runs/w5_stage3_20260920/` | 115 MB | Move, retain | `data/w5_runs/legacy_stages/` | same |
| D04 | `w5_holistic/runs/w5_stage4_20260920/` | 41 MB | Move, retain | `data/w5_runs/legacy_stages/` | same |
| D05 | `w5_holistic/runs/w5_stage5_20260920/` | 233 MB | Move, retain | `data/w5_runs/legacy_stages/` | same |
| D06 | `w5_holistic/runs/broadcast_junction_box_repair_20260921/` | 764 KB | Move, retain | `data/w5_runs/repairs/` | link from historical-impact record |
| D07 | future three-arm W5 runs | unknown | Move after interpretation | `data/w5_runs/directional_admission/` | final comparison and receipt complete |
| D08 | `line_identity/runs/line_identity_20260915_222437/` | 821 MB | Move, retain | `data/line_identity_runs/` | repaired five-case comparison complete |
| D09 | `line_identity/runs/{paint_profiles,filter_replay,axis_replay}/` | 208 KB | Move, retain | `data/line_identity_runs/diagnostics/` | result links updated |
| D10 | `direction_agreement/runs/direction_agreement_20260915_144900/` | 744 MB | Move, retain and compress | `data/direction_agreement_run/` | preserve manifest, summaries and rerun notes uncompressed |

### Completed experiments and coordination records

| ID | Current path | Disposition | Target or consolidation | Gate |
| --- | --- | --- | --- | --- |
| C01 | `direction_agreement/` excluding `runs/` | Archive intact | `archive/direction_agreement/` | extract its settled conclusion into `INDEX.md`; rewrite 18 inbound references |
| C02 | `next_steps_20260916/{C1_corrections,C2_traces,L1_admission,L2_scoring,L3_temporal}/` | Archive intact | `archive/next_steps_20260916/` | confirm each `STATE.md` has no live item |
| C03 | `next_steps_20260916/webui_seed/source/` | Consolidate after W5 | copy the exact live helper subset into `frozen_helpers_20260914/`; archive the source tree | helper-content comparison and W5 smoke |
| C04 | remaining `next_steps_20260916/webui_seed/` | Archive intact | `archive/next_steps_20260916/webui_seed/` | extract any still-live report claim |
| C05 | `worklog/WORKLOG.md` and all other worklogs | Archive intact | `archive/worklogs/` | never trim or merge |
| C06 | `worklog/{START_HERE.md,HANDOVER.md,RUNBOOK.md,viable_followups*.md,EVIDENCE_INDEX.md,CLAUDE_DIRECTION_EXPERIMENTS.md}` | Archive intact | `archive/handovers/early_campaign/` | all live tasks must exist in `pickup.md` first |
| C07 | `worklog/claude_session_*` and `worklog/w5_seances/` | Archive intact | `archive/sessions/` | record model/date and retain outputs |
| C08 | `worklog/webui_further_followups_16092026/` | Archive by packet | `archive/handovers/webui_followups/` | current W5 handover closes and open items move to `pickup.md` |
| C09 | `worklog/webui_evaluation_returns_15092026/` documentation | Archive intact | `archive/sessions/webui_evaluation_15092026/` | heavy records split under E02 below |
| C10 | `worklog/archive/` | Relocate intact | `archive/worklogs/prior_archives/` | preserve existing tombstones and snapshot notes |
| C11 | `worklog/tools/` | Promote useful scripts | `scripts/` | each retained script gets a one-line rerun instruction; obsolete scripts become deletion candidates |
| C12 | new `INDEX.md` and `pickup.md` | Create last | root | fresh-agent re-entry test, two hops maximum |

### Historical bulk evidence

| ID | Current path | Size | Disposition | Target | Gate |
| --- | --- | ---: | --- | --- | --- |
| E01 | `worklog/checks/independent/player_guided/` | 762 MB | Move, retain | `data/historical_checks/player_guided/` | map every cited result and frozen helper source |
| E02 | `worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/{pregate_loss,cap_loss}/` | 665 MB | Move, retain and compress raw pools | `data/webui_followup_records/` | keep compact assessments and producer commands visible |
| E03 | `worklog/checks/independent/examples_updated/` | 409 MB | Move, retain pending ruling | `data/historical_checks/examples_updated/` | user rules whether the 419 MB video is irreplaceable |
| E04 | `worklog/checks/independent/source_videos/` | 149 MB | Split | completed sources to `data/source_media/`; incomplete `.part` files become deletion row X02 | source provenance README |
| E05 | `worklog/checks/independent/inputs/` | 53 MB | Move, retain | `data/historical_checks/inputs/` | producer and consumers recorded |
| E06 | `worklog/checks/independent/control_checkpoints/` | 50 MB | Move, retain | `data/historical_checks/control_checkpoints/` | consumer grep |
| E07 | remaining named `worklog/checks/*` result folders | about 1.1 GB | Move by experiment family | `data/historical_checks/` | per-family README and claim pointer |
| E08 | `worklog/checks/README.md` | Consolidate | content becomes the `data/historical_checks/README.md`; original archives intact | path verification |

### Related repository material

| ID | Current path | Disposition | Target or consolidation | Gate |
| --- | --- | --- | --- | --- |
| R01 | `experiments/annotator/independent_court/*.py` | Keep | This is reusable experiment code, not scratch material | tests and import references |
| R02 | `case_provenance.py`, `export_people.py` | Keep as live | provenance boundary used by the repaired pipeline | focused tests |
| R03 | `check_paint_control.py`, `render_paint_refit.py`, `run_assignment.py`, `run_junction_selection.py`, `run_junctions.py`, `run_paint_refit.py`, `run_refit_selection.py`, `run_stripes.py` | Consolidate navigation | keep files; index them by pipeline stage in the experiment README | CLI smoke and inbound-reference check |
| R04 | `recorded/player_guided/{README.md,extension_results.md,paint_geometry.md}` | Consolidate navigation | keep evidence docs; make `README.md` the sole entry and label the other two as evidence records | link check |
| R05 | `recorded/player_guided/` binary archives | Keep | already in the correct recorded-data location | provenance and consumer check |
| R06 | `docs/courtkeynet/fallback_evaluation/README.md` | Rewrite and keep | sole durable entry point; summarise the settled result and point to current `scratch/court_det_fix/INDEX.md` | every surviving claim links to evidence |
| R07 | `independent_detector.md`, `neural_lines.md`, `scene_geometry_repair.md`, `scene_grouping.md` | Move intact | `docs/courtkeynet/fallback_evaluation/evidence/` | rewrite inbound links; do not merge away experimental limits |
| R08 | `scene_geometry_repair_worklog.md` | Archive intact | `docs/courtkeynet/fallback_evaluation/archive/` | open items copied to `pickup.md` or marked closed |
| R09 | `check_consensus_repair.py` and `recorded_inputs/` | Move together | `experiments/annotator/independent_court/recorded/fallback_evaluation/` | reproduce the recorded check from its new path |
| R10 | `figures/independent/` | Keep | figures remain beside the docs that use them | all image links resolve |
| R11 | all other repository `docs/` | Out | unrelated to this detector investigation | none |

### Ignored work from 20--21 September

The live campaign directory stays at its current path until the visible Carmack
worker and the three-arm W5 run have finished. This is the only part of the tidy
that must wait for remote work.

| ID | Current path | Disposition | Target or consolidation | Gate |
| --- | --- | --- | --- | --- |
| L01 | `local_scratch/campaigns/w5-line-admission/{RESUME.md,PROJECT_STATE.md,W5_GO.md,w5_launch_packet.md}` | Keep while live | keep as the four-file campaign entry; archive together at campaign close | final G1 and W5 results recorded |
| L02 | campaign `contract.md`, `plan.md`, `worklog.md` and `campaign.yaml` | Archive intact | `local_scratch/campaigns/w5-line-admission/archive/campaign/` after close | all unfinished work appears in tracked `pickup.md` |
| L03 | `REMOTE_*.md`, `G1_VALIDATOR_READY.md`, `stages/` and `workers/` | Archive intact | `local_scratch/campaigns/w5-line-admission/archive/remote/` | no local or Carmack process still reads them |
| L04 | `reviews/` and the six review briefs at campaign root | Consolidate | retain final `result.md` files under `archive/reviews/`; move their accepted findings into tracked evidence; remove duplicated launch wrappers under X11 | claim-by-claim review map |
| L05 | `repairs/` | Consolidate | retain the six-case detector packet, marking replay and `person_observations_v3/` under `evidence/repairs/` | G1 result and historical-impact ruling complete |
| L06 | `b2_staging/` | Split | retain the accepted r3 input receipt and logs under `evidence/remote_receipts/`; failed and superseded staging becomes X10 | canonical G1 result promoted |
| L07 | `worktrees/g1-layout-r1/` | Remove after use | no target: it is a clean 1.3 GB checkout; its only tip commit is patch-equivalent to tracked commit `c298203` | remote process ended, clean status rechecked and patch equivalence rechecked |
| L08 | `local_scratch/add_missing_court_candidates_20260920/` | Consolidate | plans, findings, decisions, reusable scripts and compact results to campaign `archive/admission_audit_20260920/`; raw arrays become X12 | every cited result remains reproducible or recorded |
| L09 | `local_scratch/runs/broadcast_junction_box_repair_20260921/result.json.gz` | Remove after snapshot | no target: it is byte-identical to the tracked result under `w5_holistic/runs/` | equality rechecked immediately before removal |
| L10 | named `local_scratch/external_delegate/20260920-*` and `20260921-*` jobs listed under X11 | Consolidate then remove | final review results to campaign `archive/reviews/`; discard launch scaffolding | every successful return is mapped; failed empty returns are named |
| L11 | future G1 and W5 remote returns | Import once | compact results and receipts to campaign `evidence/remote_results/`; final conclusions to tracked reports | accounting and content validation pass |

## Proposed deletion list

These rows are candidates only. The snapshot and applicable extraction gate
must pass first.

| ID | Exact class | Approximate size | Reason | Approval condition |
| --- | --- | ---: | --- | --- |
| X01 | every `__pycache__/`, `.ruff_cache/`, `.pytest_cache/`, `.pyrefly_cache/` under the scoped trees | 3.7 MB | regenerated caches | approve as one class after snapshot |
| X02 | nine `worklog/checks/independent/source_videos/*.part` files | about 155 MB | incomplete downloads, not valid source media | user confirms no partial file is the only surviving source |
| X03 | duplicate `worklog/WEBUI_EVALUATION_PROMPTS.md` after retaining the byte-identical copy under the archived Web UI session | 22 KB | exact duplicate | archive map names retained copy |
| X04 | `worklog/archive/2026-09-14-published-checkpoint/pre-tidy-snapshot.tar.gz` | 35 MB | superseded snapshot | new full snapshot exists and user approves old snapshot removal |
| X05 | stale temporary outputs identified by `.tmp`, failed staging or zero-byte receipt naming during execution | unknown | incomplete generated output | list every exact path before approval |
| X06 | large W5/line-identity raw arrays and case records duplicated by a validated compact result | unknown | potentially regenerable bulk | per-run proof that the compact result preserves every cited claim; exact paths listed |
| X07 | obsolete one-off scripts under `worklog/tools/` | unknown | no remaining caller or rerun value | static/dynamic reference check and exact filename list |
| X08 | clean nested worktree `local_scratch/campaigns/w5-line-admission/worktrees/g1-layout-r1/` | 1.3 GB | full duplicate checkout | L07 gate passes; remove with `git worktree remove`, not a filesystem delete |
| X09 | `local_scratch/runs/broadcast_junction_box_repair_20260921/result.json.gz` | 32 KB | exact duplicate of the tracked result | L09 equality gate passes |
| X10 | campaign `repairs/person_observations/`, `repairs/person_observations_v2/`, five-case detector packet and superseded or failed `b2_staging/` content | about 3 MB | replaced by the validated six-case packet, v3 inputs and r3 staging | remote result proves which generation was used; exact staging paths listed before removal |
| X11 | the 18 delegate job directories listed below | 1.4 MB | duplicated briefs, launch wrappers and returned text | L04 and L10 review map passes |
| X12 | `local_scratch/add_missing_court_candidates_20260920/admission_audit/results_{gx5,gx5_camera,w5_camera}/*.npz` | about 69 MB | bulky intermediate arrays | compact JSON, decision record and rerun command preserve every cited result |

No unique historical run, source video, worklog, review result, manifest,
visual ruling or frozen input is currently proposed for deletion. X11 removes
delegate wrappers only after their unique review results have been retained.

X11 covers these exact directories:

```text
20260920-w5-collision-audit
20260920-w5-collision-audit-extended
20260920-w5-collision-opus-review
20260920-w5-final-gallery-review
20260920-w5-final-packet-closeout
20260920-w5-final-packet-postfix
20260920-w5-final-packet-postreport
20260920-w5-identity-code
20260920-w5-identity-code-followup
20260920-w5-identity-final-review
20260921-141203-carmack-visible-luna
20260921-box-repair-code-opus
20260921-broadcast-result-opus
20260921-directional-cache-opus
20260921-historical-repair-opus
20260921-line-template-regression-red-team
20260921-marking-am4-correction-opus
20260921-marking-repair-result-opus
```

## Execution order

1. Approve or amend this manifest
2. Create `local_scratch/court-detector-tidy-backup-20260921.tar.gz` from all
   unique ignored material plus the tracked diff and commit record. Exclude the
   clean nested worktree because Git can reproduce it exactly.
3. List the backup and open a sample of its files before changing paths.
4. Tidy the closed documentation, completed experiments and historical data.
   This work does not wait for Carmack.
5. Delete only approved cache, duplicate and incomplete-output rows whose gates
   have passed.
6. After G1 and W5 finish, close and consolidate the live campaign directory.
7. Create `data/`, move run data and write provenance READMEs.
8. Move completed experiments and worklogs into `archive/` with tombstones.
9. Consolidate live line-identity, experiment and fallback-doc navigation.
10. Update every inbound and outbound reference.
11. Create `INDEX.md`, `pickup.md` and `archive/ARCHIVE_MAP.md` last.
12. Run reference-integrity checks across the whole repository.
13. Give a fresh agent only `INDEX.md` and require correct orientation within
    5,000 tokens, with evidence and rerun answers reachable in two hops
14. Present any further exact deletion candidates for a separate ruling.

## House rules for the finished tree

- `pickup.md` is the only live state note
- update state notes by rewriting them, not appending session history
- worklogs archive intact
- heavy outputs go under `data/<experiment>/` with a provenance README
- completed session material moves to `archive/` in the same session
- `INDEX.md` stays a thin map, not another history document
- no document points into a section of the rewritable pickup note
- new deletion candidates are listed explicitly and require approval
