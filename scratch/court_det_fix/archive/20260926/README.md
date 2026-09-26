# Files archived on 26 September 2026

These are dated records and scripts worth keeping for a specific future question.
[pickup.md](../../pickup.md) is the only live project state. The
[speed-up design](../../court_detector/PERFORMANCE.md) carries the useful open work
from the former handovers. The worklogs and archived script bodies are intact.

The project owner authorised judgement calls on archiving and deletion, asked to
keep large raw data, and accepted that archived scripts may have outdated paths.

## What moved

| Former path under `court_det_fix/` | Where to read it now |
| --- | --- |
| `INDEX.md` before the tidy | [Original account](originals/INDEX.md); the live file now has one defined role |
| `pickup.md` before the tidy | [Original account](originals/pickup.md); the live file now has one defined role |
| `DETECTOR_DECISIONS.md` before the tidy | [Original account](originals/DETECTOR_DECISIONS.md); the live file now has one defined role |
| `FP_INDEX.md` before the tidy | [Original account](originals/FP_INDEX.md); the live file now has one defined role |
| `court_detector/README.md` before the tidy | [Original account](originals/court_detector/README.md); the live file now has one defined role |
| `court_detector_optimisation_handover/README.md` before the tidy | [Original account](originals/court_detector_optimisation_handover/README.md); the live file now has one defined role |
| `colour_consistency/AM1_RECOVERY.md` | [AM1_RECOVERY.md](experiments/colour_consistency/AM1_RECOVERY.md) |
| `colour_consistency/PLAN.md` | [PLAN.md](experiments/colour_consistency/PLAN.md) |
| `direction_agreement/common.py` | [common.py](experiments/direction_agreement/common.py) |
| `direction_agreement/diagnose_matrix.py` | [diagnose_matrix.py](experiments/direction_agreement/diagnose_matrix.py) |
| `direction_agreement/manifest.py` | [manifest.py](experiments/direction_agreement/manifest.py) |
| `direction_agreement/membership.py` | [membership.py](experiments/direction_agreement/membership.py) |
| `direction_agreement/render_gallery.py` | [render_gallery.py](experiments/direction_agreement/render_gallery.py) |
| `direction_agreement/run_arms.py` | [run_arms.py](experiments/direction_agreement/run_arms.py) |
| `direction_agreement/run_fits.py` | [run_fits.py](experiments/direction_agreement/run_fits.py) |
| `direction_agreement/run_matcher.py` | [run_matcher.py](experiments/direction_agreement/run_matcher.py) |
| `direction_agreement/run_remote.sh` | [run_remote.sh](experiments/direction_agreement/run_remote.sh) |
| `direction_agreement/selection.py` | [selection.py](experiments/direction_agreement/selection.py) |
| `direction_agreement/summarise.py` | [summarise.py](experiments/direction_agreement/summarise.py) |
| `direction_agreement/sync.sh` | [sync.sh](experiments/direction_agreement/sync.sh) |
| `direction_agreement/tests/conftest.py` | [conftest.py](experiments/direction_agreement/tests/conftest.py) |
| `direction_agreement/tests/test_imports.py` | [test_imports.py](experiments/direction_agreement/tests/test_imports.py) |
| `direction_agreement/tests/test_matcher_adapter.py` | [test_matcher_adapter.py](experiments/direction_agreement/tests/test_matcher_adapter.py) |
| `direction_agreement/tests/test_membership.py` | [test_membership.py](experiments/direction_agreement/tests/test_membership.py) |
| `direction_agreement/tests/test_selection.py` | [test_selection.py](experiments/direction_agreement/tests/test_selection.py) |
| `direction_agreement/tests/test_summarise.py` | [test_summarise.py](experiments/direction_agreement/tests/test_summarise.py) |
| `edge_polarity/README.md` | [README.md](experiments/edge_polarity/README.md) |
| `edge_polarity/WEBUI_OBJECTIVE_AUDIT.md` | [WEBUI_OBJECTIVE_AUDIT.md](experiments/edge_polarity/WEBUI_OBJECTIVE_AUDIT.md) |
| `edge_polarity/local_audit/ASSESSMENT.md` | [ASSESSMENT.md](experiments/edge_polarity/local_audit/ASSESSMENT.md) |
| `edge_polarity/local_audit/WORKLOG.md` | [WORKLOG.md](experiments/edge_polarity/local_audit/WORKLOG.md) |
| `net_recovery/ASSESSMENT.md` | [ASSESSMENT.md](experiments/net_recovery/ASSESSMENT.md) |
| `net_recovery/WORKLOG.md` | [WORKLOG.md](experiments/net_recovery/WORKLOG.md) |
| `svd_runtime/COMPUTE_AUDIT.md` | [COMPUTE_AUDIT.md](experiments/svd_runtime/COMPUTE_AUDIT.md) |
| `svd_runtime/README.md` | [README.md](experiments/svd_runtime/README.md) |
| `svd_runtime/RESULTS.md` | [RESULTS.md](experiments/svd_runtime/RESULTS.md) |
| `svd_runtime/WORKLOG.md` | [WORKLOG.md](experiments/svd_runtime/WORKLOG.md) |
| `svd_runtime/check_history.py` | [check_history.py](experiments/svd_runtime/check_history.py) |
| `svd_runtime/run_benchmark.py` | [run_benchmark.py](experiments/svd_runtime/run_benchmark.py) |
| `svd_runtime/run_carmack.sh` | [run_carmack.sh](experiments/svd_runtime/run_carmack.sh) |
| `svd_runtime/smoke_integration.py` | [smoke_integration.py](experiments/svd_runtime/smoke_integration.py) |
| `svd_runtime/summarise.py` | [summarise.py](experiments/svd_runtime/summarise.py) |
| `svd_search/README.md` | [README.md](experiments/svd_search/README.md) |
| `svd_search/RUN_READY.md` | [RUN_READY.md](experiments/svd_search/RUN_READY.md) |
| `svd_search/WORKLOG.md` | [WORKLOG.md](experiments/svd_search/WORKLOG.md) |
| `svd_search/build_gallery.py` | [build_gallery.py](experiments/svd_search/build_gallery.py) |
| `svd_search/run.py` | [run.py](experiments/svd_search/run.py) |
| `svd_search/test_search.py` | [test_search.py](experiments/svd_search/test_search.py) |
| `w5_holistic/steering_record.md` | [steering_record.md](experiments/w5_holistic/steering_record.md) |
| `webui_final_opt_handover/` | [Whole packet](webui_final_opt_handover/README.md), including GPU/Numba proposals and synthetic benchmarks |
| `webui_net_evidence_frontier_handover/` | [Whole packet](webui_net_evidence_frontier_handover/README.md), including the bounded net audit |

## Where the live content went

| Source | Live home |
| --- | --- |
| Old pickup, detector README and speed-up handovers | [pickup](../../pickup.md): current position, next work and unresolved blockers |
| Decisions through 26 September, including rejection of the 12-direction shortcut | [D12 and D19–D25](../../DETECTOR_DECISIONS.md#d12): dated decisions and evidence |
| Agreed cheap-score trial, camera reuse plan, GPU/CPU constraints and parallel work | [Speed-up design](../../court_detector/PERFORMANCE.md) |
| Detector API and switches | [Detector guide](../../court_detector/README.md) |
| Code, rerun entry points and data required locally or remotely | [File map](../../FP_INDEX.md) |

The archived originals keep commit IDs, detailed measurements, earlier rulings and
conditional ideas that do not belong in the current queue. Their old “next” and
“pending approval” sections are historical. In particular, the newer decisions
replace the earlier acceptance of the 12-direction shortcut and the earlier
claim that the combined detector does not exist.

## Rerunning an archived experiment

Archived scripts retain their original imports, paths and commands. They are
**not runnable from this archive without checking those assumptions**. Restore a
trial in a separate checkout at its recorded commit, or adapt its paths for the
specific question. Its raw data remains at its former path unless the whole small
handover packet moved together. Do not launch old remote scripts as current jobs.

This archive map is the notice for all records below it. The original packet
READMEs remain byte-identical so their supplied checksums stay valid.

Archived Markdown also keeps its original relative and absolute paths. Interpret
them from the former location in the table above; then apply this map. This keeps
the worklogs unchanged. New archive notices and this map are the navigation layer.

For D12, the [benchmark results](experiments/svd_runtime/RESULTS.md) preserve the
52.9% matcher-time finding. The later
[full-detector comparison](../20260925_optimisation_handover/CLAUDE_EVALUATION.md#svd12-screen-against-full-search)
explains why the joined detector no longer uses that shortcut.

## What stayed and what was removed

Large raw data and existing recovery archives stayed in place. Live modules,
their helper imports, the feet fixtures and gallery templates stayed at the paths
that running code uses. Moving those needs a separate code change.

Removed 6 empty “Retired source” pointers and 144 generated
Python cache files (2,560,856 bytes). The pointers contained no unique
findings. Their destinations are below; source text also survives in the snapshot.

| Removed pointer | Read instead |
| --- | --- |
| `w5_holistic/box_provenance_impact.md` | [evidence/holistic_admission/box_provenance.md](../../evidence/holistic_admission/box_provenance.md) |
| `worklog/viable_followups.md` | [INDEX.md](../../INDEX.md) |
| `line_identity/results.md` | [evidence/g0_g1/README.md](../../evidence/g0_g1/README.md) |
| `next_steps_20260916/L2_scoring/result.md` | [evidence/g0_g1/README.md](../../evidence/g0_g1/README.md) |
| `next_steps_20260916/L3_temporal/result.md` | [evidence/pixel_temporal/README.md](../../evidence/pixel_temporal/README.md) |
| `next_steps_20260916/webui_seed/reports/line_identity_results.md` | [evidence/g0_g1/README.md](../../evidence/g0_g1/README.md) |

Also removed `evidence/webui_followup3_20260922/received/svd_court_evaluation/frozen_direction_search_README.md` after confirming it was byte-identical to the retained [direction-search note](../../evidence/direction_search/README.md). The received packet now links to that source; its 14 broken copied links disappear with the duplicate.

## Recovery and checks

The local snapshot is `.recovery/before-tidy-20260926.tar.gz`, relative to
`court_det_fix/`. Its 696 files were compared byte-for-byte before any moves.
It covers Markdown, Python, shell/config files and the two small handover
packets. Untouched large data was excluded. Snapshot removal was not part of
this tidy.

[disposition.csv.gz](disposition.csv.gz) records every original file, its size,
date, role, link counts and disposition. The complete original inventory and
move map also live under `.recovery/tidy-20260926-*.json.gz`.

Completed 27 September after starting on 26 September. No commit or push was
made. All checks below passed with exit code 0:

- All 89 moved files and six original live documents match the pre-tidy
  snapshot byte-for-byte. The archived worklogs and script bodies are intact
- 8,348 retained data/config files have unchanged sizes and modification times
- Eight sampled decisions, results and commit references survive in the
  mapped archive records
- All 1,904 Markdown links and anchors outside the archive resolve. The new
  archive map's links and anchors also resolve. Archived bodies retain their
  former path assumptions as documented above
- The net-frontier packet's supplied SHA-256 checksums pass
- `pytest -q tests/test_court_detector_modules.py tests/test_court_detector_feet.py`:
  37 passed. These check the detector's imports, helper paths and feet behaviour
- `git diff --check` passes

A fresh reader given only INDEX recovered the state, next work, unresolved
choices, D12's archived evidence, the prepared-view command and its missing
inputs within two links. INDEX plus pickup total 952 words, below the roughly
5,000-token orientation budget. The reader found the document roles and
plain descriptions clear.

No detector code was changed. The broad lint, type and test suites were not
needed for the document edits and byte-preserved archival moves.
