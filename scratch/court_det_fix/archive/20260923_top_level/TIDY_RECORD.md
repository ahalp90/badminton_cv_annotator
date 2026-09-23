# Top-level and untracked-file disposition

## Authority and scope

The user explicitly authorised refreshing the whole top level, assigning
non-overlapping document purposes, deliberately tracking or retaining all
untracked files locally, then committing and pushing. The dispositions below
implement that request. There is no deletion class and no new runtime experiment.

The live entry is INDEX; pickup owns current operational state. Detailed
historical findings belong in DETECTOR_DECISIONS. FP_INDEX maps code and data.
The launch packet carries the next phase's contract and bounded starting work.

Out of scope: experiment algorithms, frozen results, galleries, model files,
already ignored bulk data, and archived worklog contents. Existing run data
keeps its path because scripts and records refer to those paths. Adding exact
ignore rules changes storage policy, not experiment inputs.

## Snapshot and reference checks

Before rewriting or relocating, `local_scratch/court_det_fix_tidy/20260923/before.tar`
preserved all court_det_fix Markdown, the local ignore file and 91 untracked
files (245.87 MiB). Untouched tracked binary data and previously ignored data
were excluded. The snapshot is 259,184,640 bytes and is retained locally.
The compressed inventory records exact paths, sizes and SHA-256 values.

Inbound references were searched across court_det_fix, repository docs and
project context. INDEX's `recovery` and `recovery-not-another-reading-path`
anchors remain valid. The detailed decision history retains its named section
anchors. The moved net report's Markdown links and inbound links are rewritten.

## Disposition

| Item or class | Action | Reason |
| --- | --- | --- |
| INDEX.md | Archive previous prose; rewrite as thin entry map | One obvious starting point, no duplicate history |
| pickup.md | Archive previous prose; rewrite as current resume | State, next action, constraints and readiness only |
| DETECTOR_DECISIONS.md | Archive previous prose; refresh detailed history | Preserve IDs, measured evidence, decisions and contrary cases |
| FP_INDEX.md | Archive previous prose; refresh filepath map | Source/data navigation without parallel experiment summaries |
| NET_EVIDENCE_ASSESSMENT.md | Relocate to net_recovery/ASSESSMENT.md | Specialised evidence belongs with its experiment |
| handover_20260923/*.md | Track | Compact next-session contract and launch instructions |
| 20 frozen native-frame PNGs, 15.40 MiB | Track at existing paths | Required replay inputs and image-hash checks |
| WebUI Am1/GX PNGs | Track at existing paths | Retained audit inputs |
| WebUI ZIP | Explicit local-only rule, existing path | Duplicate assembled transfer packet |
| colour_consistency/smoke.json.gz | Explicit local-only rule, existing path | Generated smoke output; producing command remains in PLAN.md |
| svd_runtime/full/ | Explicit local-only rule, existing path | 95.68 MiB retained raw timing evidence; compact results already tracked |
| svd_runtime/integration_smoke_raw/ and smoke/ | Explicit local-only rules | Retain raw smoke evidence without pushing bulk data |
| svd_runtime/partial_history.json.gz and saved_input_check.json.gz | Track | Small comparison/verification results |
| svd_runtime/receipts/ | Track | Small completed-run logs and exit receipts |
| svd_search/run_20260923/cases/ | Track 18 case records, 22.66 MiB | Consumed by follow-up experiments; preserve exact candidates and stage timings |
| svd_search/run_20260923/generation/ | Explicit local-only rule | 96.49 MiB raw generation evidence retained for deeper profiling/rebuilds |
| svd_search/gallery_preview/ | Explicit local-only rule | Generated preliminary gallery; final gallery is tracked |
| New raw-data notes, archive map and inventory | Track | Make storage decisions and recovery reproducible |

Generated directories remain at their original paths under scoped ignore rules.
They are preserved inputs for future work, not declared disposable or cheap to
regenerate. A fresh clone will need an explicit copy of the local-only records.

## Verification

- The original inventory resolves to 47 deliberately tracked files and 44
  explicitly local-only files. All 89 non-Markdown data files retain their
  recorded SHA-256 values; the two launch documents were intentionally completed
- Local Markdown targets and section anchors pass, including the preserved
  recovery and filepath-map anchors. Newly tracked JSON and PNG formats pass
- The shared contract is 77 lines and the launch prompt is 170 lines, within the
  250-line combined startup budget
- A fresh Sol high reader started from INDEX alone. Orientation took about 2,350
  tokens. Current state, pending decisions, D09's archive route, the statistics
  rerun, closed mechanisms and scene-grouping dependency were found within two
  links. No broken route or contradictory document role was reported
- Material prior findings survive in the live decisions or intact archived
  sources: D01–D18, 24/27, 7/7 versus 5/7, 52.9%, 26.6%, 10.0%, the net regressions,
  accepted stripe correction and the 30/90-second clarified runtime expectation
- Documentation/storage checks exit 0. No runtime code or experiment output was
  changed; no new pytest, lint or type-check run is required for this tidy
- The final close-out commit and push are recorded in Git; the working tree and
  remaining untracked files are checked after the push
