# W2 handover index

Read `pickup.md` first. It records the completed two-pair pixel export and its limits.

## Live packet

- `pickup.md`: current status, result, scope and rerun command
- `w2_fresh_review/w2_pair_atlas/`: unpacked atlas, paired traces, crops, strips and manifest
- `w2_fresh_review/REVIEW.md`: pre-export analysis, retained as historical reasoning
- `w2_fresh_review/results/`: fresh CSV analysis, interval table, tests and execution status
- `w2_fresh_review/inputs/`: supplied CSV and pinned provenance inputs
- `w2_fresh_review/*.py`, `run_followup.sh`, `tests/`: bounded reproduction helpers

## Scope

The export covers two ORIGINAL automatic-all-camera Am2 candidates and four intervals.
It checks all 48 uploaded table rows and changes no detector, matcher, score, gate,
population or acceptance rule. Physical pixel ownership remains unresolved.

The runner creates a ZIP as a transient convenience output. The committed packet keeps
the unpacked files so a reviewer can inspect them directly.

## House rules

- `pickup.md` is the current entry point
- New measurements get a dated result directory and an execution status
- Generated caches and packaging duplicates stay out of the commit
- Historical handovers remain recoverable in the dated pre-tidy snapshot
