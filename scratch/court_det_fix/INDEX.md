# Court-detector investigation

The branch is building a CourtKeyNet-free detector. Good courts exist in the
saved development pools; reliable automatic selection and practical runtime
remain unresolved. Start with [pickup.md](pickup.md), then read only the
evidence needed for the question at hand.

## Research and evidence

| Question | Read |
| --- | --- |
| What has the branch established? | [Findings and decisions](DETECTOR_DECISIONS.md) |
| What remains unassessed in G0/G1? | [Population inventory and assessment brief](evidence/g0_g1/README.md) |
| What does W5 test, and why? | [Whole-court scoring and admission](evidence/holistic_admission/README.md) |
| Which results survive the person-box error? | [Corrected provenance account](evidence/holistic_admission/box_provenance.md) |
| Why did direction and cap changes fail? | [Direction/search evidence](evidence/direction_search/README.md) |
| What did pixel and temporal tests establish? | [Pixel/temporal evidence](evidence/pixel_temporal/README.md) |
| Which earlier ideas and data remain useful? | [Independent proposals](evidence/independent_proposals/README.md) |
| Why is CourtKeyNet being removed? | [Retirement evidence](evidence/retirement/README.md) |

The [remote packet](worklog/remote_records_20260921/README.md) owns preserved
Carmack data and launch receipts. The separate W5 campaign owns the running
job. Neither belongs to this tidy's removal list.

## Code and saved data

Production and current experiment code stays in its existing directories.
The line-identity matcher is `line_identity/line_run_matcher.py`; its distinct
name prevents an import clash with the direction matcher. The evidence notes
link directly to recorded comparisons, complete candidate populations,
original images and runnable checks. Compatibility links preserve old replay
paths; those paths are not alternative reading routes. Two relocated audit
helpers have only their repository-root lookup adjusted.

## Recovery, not another reading path

The [disposition ledger](.recovery/disposition.csv.gz) maps original paths to
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
Keep both archives. Closed court-related local-scratch and delegate trees
have been removed; only the active W5 campaign/monitor and model assets stay.
Short retired-source redirects remain where protected W5 records need them.

## Keep this usable

- This index and pickup are the short orientation path; stay below 5,000 tokens
- Update a result beside its evidence, not in a new session narrative
- Preserve candidate origin, evaluation stage, units and visual-review status
- Keep unfinished assessment prominent; completed generation is not assessment
- Keep original inputs and useful candidate data, with producer and source paths
- Retire superseded prose after its findings and open questions have a home
- Keep recovery copies sealed; do not create another browsable session tree
- Keep operational W5 records unchanged until the job and result transfer finish
- CourtKeyNet is retired as an approach and must be removed before branch end
