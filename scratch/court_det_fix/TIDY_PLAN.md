# Court-detector consolidation and deletion plan

## Bottom line

The court-detector investigation is unfinished. Its history contains several
competing approaches whose useful results are scattered across reports, run
directories and session records. Moving all of that into an archive would keep
the evidence hidden and preserve the same navigation problem.

This tidy will first compile the competing approaches into one decision ledger.
It will retain the smallest evidence pack that supports each live approach and
the data that would avoid an expensive rerun. Everything else will leave the
current workspace after a recoverable backup.

Nothing has moved or been deleted. Every deletion still needs approval.

## Scope

The scoped workspace occupies 7.6 GB.

| Area | Size | What is mixed together |
| --- | ---: | --- |
| `scratch/court_det_fix/` | 6.0 GB | active code, results, raw runs, worklogs, reviews and source media |
| `local_scratch/campaigns/w5-line-admission/` | 1.4 GB | the live campaign and a 1.3 GB nested worktree |
| `experiments/annotator/independent_court/` | 171 MB | reusable experiment code and recorded evidence |
| `local_scratch/add_missing_court_candidates_20260920/` | 78 MB | proposal-source audit code, results and reusable arrays |
| `docs/courtkeynet/fallback_evaluation/` | 5.8 MB | five overlapping result narratives, a worklog, figures and a runnable check |
| relevant `local_scratch/external_delegate/2026092{0,1}-*/` | 1.4 MB | review results mixed with launch wrappers |
| `local_scratch/runs/broadcast_junction_box_repair_20260921/` | 40 KB | duplicate of a tracked repair result |

Other repository documentation and older unrelated `local_scratch` work are
out of scope.

## The record this tidy must produce

Create `scratch/court_det_fix/DETECTOR_DECISIONS.md` as the central account.
It will compare these approach families:

| Approach family | Main question | Evidence currently scattered across |
| --- | --- | --- |
| Existing model and scene repair | How far can CourtKeyNet, consensus repair and repeated-camera sharing carry the current pipeline? | `docs/courtkeynet/fallback_evaluation/` |
| Independent line proposals | Can OpenCV, DeepLSD, LINEA and projective/player-guided proposals recover partly visible courts without CourtKeyNet? | fallback docs, `experiments/annotator/independent_court/`, historical checks |
| Direction recovery | Which direction grouping, representative and anchor rules preserve viable court lines? | `direction_agreement/`, C1/L1 records |
| Line identity and search allocation | Which fragments should survive, and where do ordering and per-direction caps discard good courts? | `line_identity/`, C2/L2 records |
| Holistic scoring and admission | Can whole-court scoring, line-template proposals and visibility floors choose usable courts with sane compute? | `w5_holistic/`, the W5 campaign and the admission audit |

Player-box provenance, camera assumptions and compute cost are cross-cutting
columns rather than separate approaches.

For every approach, the ledger must state:

- the change being tested and the pipeline stage it affects;
- the strongest result and known failure cases on the shared corpus;
- whether the player-box wiring error weakens the result;
- quality, search cost and expected deployment cost;
- which code or data remains reusable;
- the next experiment that would separate it from the competing approaches;
- a link to one compact evidence pack;
- a verdict: active contender, useful component, superseded, rejected or
  unresolved.

The ledger is a synthesis, not another chronological worklog. A context-naive
reader should be able to see what has been learned, what still competes and why
the next experiment is worth running.

## Retention rule

A file remains visible only when it satisfies at least one of these tests:

1. It is active detector code or a test of active behaviour.
2. It is needed to compare or finish an unresolved approach.
3. It is an expensive-to-recreate input or intermediate that a likely next
   experiment can reuse.
4. It is the smallest surviving evidence for a settled claim.
5. It is the current pickup or remote-run contract.

Being historically interesting is not enough. Git preserves tracked history.
The pre-tidy backup preserves ignored material. Worklogs remain intact in that
backup; they do not need to remain in the current checkout.

## Target surface

Keep active code paths stable until the detector decision is resolved. Remove
the bulky and chronological material around them.

```text
scratch/court_det_fix/
├── INDEX.md                       short entry point
├── pickup.md                      current state and next action
├── DETECTOR_DECISIONS.md          comparison of every live approach
├── w5_holistic/                   active code, tests and current decision only
├── line_identity/                 active code and tests while still needed
├── frozen_views/                  one shared case corpus
├── frozen_helpers_20260914/       one runnable helper snapshot
├── evidence/
│   ├── model_scene_repair/
│   ├── independent_proposals/
│   ├── direction_recovery/
│   ├── line_identity/
│   └── holistic_admission/
└── scripts/                       reusable measurement and migration tools
```

Each evidence pack contains only:

- one plain-language result note;
- one manifest naming inputs and code;
- compact metrics and candidate records needed for comparison;
- a few representative success and failure images;
- expensive reusable intermediates, when retaining them is cheaper than
  regenerating them;
- the shortest known rerun or rescore command.

There will be no general `archive/` dumping ground in the tracked tree.

## Disposition plan

### Central record and live code

| ID | Current material | Action | Gate |
| --- | --- | --- | --- |
| A01 | all result notes and decision records | Extract claims into `DETECTOR_DECISIONS.md`, with one evidence link per claim | each approach has a result, limit and next decision |
| A02 | `w5_holistic/` source and tests | Keep at the current path while W5 is live; remove old run directories under W rows below | focused tests and W5 close-out |
| A03 | `line_identity/` source and tests | Keep while the repaired player matcher and allocation question remain live | repaired matcher and ledger verdict |
| A04 | `direction_agreement/` source | Keep only code reused by a live branch; otherwise remove after its result is compiled | import and caller check |
| A05 | `frozen_views/` | Keep and make it the single shared case source | cases, frames, controls and provenance reconcile |
| A06 | `frozen_helpers_20260914/` | Complete the missing camera helper, then freeze | import smoke and helper inventory |
| A07 | reusable scripts under `worklog/tools/` and completed experiment folders | Promote only scripts named by an evidence pack | caller or rerun instruction exists |

### Competing experiment evidence

| ID | Current material | Keep visible | Remove from the current tree after extraction |
| --- | --- | --- | --- |
| B01 | `docs/courtkeynet/fallback_evaluation/` | one model/scene-repair evidence note, one independent-proposals note, compact recorded inputs and referenced figures | the five overlapping narratives, chronological worklog and standalone script at their old paths |
| B02 | `experiments/annotator/independent_court/` | reusable detector/evaluator code, tests and unique recorded inputs used by a live approach | duplicate archives, generated views and evidence copied elsewhere |
| B03 | `direction_agreement/` | compact metrics, representative cases and the surviving lesson for direction rules | the 744 MB raw run and chronology once no likely rescore needs it |
| B04 | `line_identity/` | pair traces, candidate identifiers and any pools needed to test ordering or caps without rerunning generation | duplicated full-pool results and obsolete diagnostics after repaired comparison |
| B05 | `next_steps_20260916/` | unresolved decisions and reusable scripts folded into the relevant evidence packs | state files, handoffs and duplicated seed tree |
| B06 | `w5_holistic/` | current code, final comparison packet, manifest and representative gallery | superseded stages, duplicate galleries and raw outputs with no planned consumer |
| B07 | `worklog/checks/` | unique shared inputs and expensive intermediates with a named likely consumer | generated outputs, repeated copies and method variants that teach no surviving lesson |
| B08 | Web UI and Claude session records | accepted findings and final independent-review rulings | prompts, launch scaffolding, repeated context and conversational returns |

### Large tracked material

| ID | Current path | Size | Decision |
| --- | --- | ---: | --- |
| W01 | `w5_holistic/runs/w5_stage{2,3,4,5}_20260920/` | 703 MB | compile the progression and final lesson, retain only evidence cited by the ledger, then remove the raw runs |
| W02 | `w5_holistic/runs/line_template_regression_20260920/` | 514 MB | retain the result, manifest, selected gallery and candidate records needed by W5; remove uncited bulk |
| W03 | `direction_agreement/runs/direction_agreement_20260915_144900/` | 744 MB | retain reusable direction inputs only if a named next experiment consumes them; otherwise remove after B03 |
| W04 | `line_identity/runs/line_identity_20260915_222437/` | 821 MB | retain the smallest pools needed for cap/order rescoring; remove redundant matcher outputs after the repaired comparison |
| W05 | `worklog/checks/independent/` | 2.1 GB | deduplicate shared frames, controls, people and candidate inputs into the corpus; remove output copies and dead method branches |
| W06 | `worklog/webui_evaluation_returns_15092026/` | 679 MB | retain compact assessments and any unique reusable pools; remove the session-shaped record |
| W07 | remaining `worklog/` bulk | about 500 MB | retain only unique live inputs; remove chronology and generated output after ledger extraction |

W03--W06 need a consumer-and-regeneration audit before deletion. The question is
practical: would a likely next comparison use this file, and would deleting it
force hours of recomputation? If neither answer is yes, the file goes.

### Ignored work from 20--21 September

The live campaign stays at its current path until the visible Carmack job and
the three-arm W5 run finish. No other part of the tidy waits for them.

| ID | Current path | Action | Gate |
| --- | --- | --- | --- |
| L01 | campaign `RESUME.md`, `PROJECT_STATE.md`, `W5_GO.md` and `w5_launch_packet.md` | keep as the live entry; compile the result into tracked records at close | G1 and W5 complete |
| L02 | campaign plans, worklog, remote go files, stages and workers | preserve in the backup, then remove after open work reaches `pickup.md` | no process reads them |
| L03 | campaign `reviews/` and root review briefs | retain each final ruling once; remove briefs and launch wrappers | accepted findings mapped to tracked claims |
| L04 | campaign `repairs/` | keep the six-case detector packet, marking replay and v3 person observations until historical accounting closes; remove superseded versions | repaired G1 result and audit complete |
| L05 | campaign `b2_staging/` | retain the accepted r3 receipt and any costly reusable inputs; remove failed and superseded staging | canonical G1 promoted |
| L06 | `worktrees/g1-layout-r1/` | remove with `git worktree remove`; it is a clean 1.3 GB duplicate whose tip is patch-equivalent to tracked code | remote process ended and clean/equivalent state rechecked |
| L07 | `add_missing_court_candidates_20260920/` | fold findings and reusable scripts into the proposal evidence pack; retain its 69 MB arrays if they avoid a likely generation rerun | W5 evidence pack names consumers |
| L08 | duplicate broadcast repair result | remove | equality with tracked result rechecked |
| L09 | 18 relevant external-delegate directories | retain final rulings once, then remove every wrapper directory | review map complete |
| L10 | future G1 and W5 returns | import compact results and reusable inputs once | accounting and content validation pass |

## Proposed deletion classes

These are candidates, not approval to delete them.

| ID | Material | Approximate reclaim | Required proof |
| --- | --- | ---: | --- |
| X01 | caches under every scoped tree | 4 MB | generated cache class only |
| X02 | nine incomplete `source_videos/*.part` files | 155 MB | no partial file is the only surviving source |
| X03 | byte-identical prompt and result copies | under 1 MB | one canonical copy named |
| X04 | superseded 35 MB pre-tidy snapshot | 35 MB | new backup opens correctly |
| X05 | nested `g1-layout-r1` worktree | 1.3 GB | L06 gate passes |
| X06 | W5 stages 2--5 bulk | up to 703 MB | W01 extraction passes |
| X07 | uncited line-template bulk | likely hundreds of MB | W02 retained packet reproduces the comparison |
| X08 | direction-agreement raw run | up to 744 MB | no named next experiment consumes it |
| X09 | redundant line-identity matcher outputs | likely hundreds of MB | repaired comparison and retained rescore pools pass |
| X10 | historical-check and Web UI outputs with no consumer | likely 1--3 GB | W05--W07 consumer audit passes |
| X11 | failed and superseded campaign staging and repair versions | about 3 MB | canonical G1 result names the accepted generation |
| X12 | 18 named 20--21 September delegate directories | 1.4 MB | final rulings retained once |
| X13 | duplicate local broadcast repair result | 32 KB | byte equality rechecked |
| X14 | obsolete one-off scripts | unknown | no caller, rerun instruction or unique method remains |

X12 covers these exact directories under `local_scratch/external_delegate/`:

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

The likely reduction is 5.5--6.5 GB. The exact amount depends on which candidate
pools and source inputs would save a likely future HPC run. Those are useful
working data, not posterity, and should be retained once rather than regenerated.

## Execution order

1. Approve or amend this plan and its deletion classes.
2. Create one `local_scratch/court-detector-tidy-backup-20260921.tar.gz` from
   unique ignored material, the tracked diff and a Git commit record. Do not
   duplicate the clean nested worktree in the backup.
3. List the backup and open representative files.
4. Draft `DETECTOR_DECISIONS.md` from the current records before moving or
   deleting their sources.
5. Build the five compact evidence packs and the deduplicated shared corpus.
6. Check every large input and intermediate for a named likely consumer and
   record its regeneration cost.
7. Present the resulting exact path-level delete list for approval.
8. Delete approved tracked material from the current branch. Git remains its
   historical archive.
9. Delete approved ignored material from `local_scratch` after its backup gate.
10. Close the live campaign after G1 and W5 finish.
11. Rewrite all surviving links and add short tombstones only where an external
    reference would otherwise break.
12. Create `INDEX.md` and `pickup.md` last.
13. Give a fresh agent only `INDEX.md` and require it to identify the competing
    approaches, evidence, current blocker and next experiment within 5,000
    tokens and two link hops.

## Finished-tree rules

- `DETECTOR_DECISIONS.md` owns the experiment comparison.
- `pickup.md` owns live state and the next action.
- `INDEX.md` only routes readers to those records, active code and evidence.
- Each result exists in one evidence pack.
- Expensive reusable intermediates name their next likely consumer.
- Session transcripts, launch scaffolding and chronological worklogs stay in
  the backup or Git history, not the current tree.
- Completed runs are reduced to their decision-bearing packet in the same
  session.
- New deletion candidates require approval.
