# Court-detector investigation

The branch is building a CourtKeyNet-free detector. G1 paint-filtered proposals
plus line templates are the leading next design. Court selection has improved;
far-end fit, rejection of bad results and practical runtime still need work.
The four September pickup comparisons are complete.
The target is a scene-level detector using a few sampled frames, shared search
and agreement across a stable camera view, not full search on every frame.

Start with [pickup.md](pickup.md) for what to do next. Use
[FP_INDEX.md](FP_INDEX.md) to find files by idea rather than experiment name.
For an independent review from GitHub, open the
[raw evidence packet](evidence/review_20260922/README.md).

## Research and evidence

| Question | Read |
| --- | --- |
| What has the branch established? | [Findings and decisions](DETECTOR_DECISIONS.md) |
| What did the completed comparisons show? | [22 September evaluation](evaluation_results_20260922.md) |
| Where do proposal access and scoring differ? | [G0/G1 comparison](evidence/g0_g1/README.md) |
| Which older ideas should return, and when do we test more views? | [Next design and test plan](DETECTOR_DECISIONS.md#next-design-and-wider-test) |
| What does W5 test, and why? | [Whole-court scoring and admission](evidence/holistic_admission/README.md) |
| Which results survive the person-box error? | [Corrected provenance account](evidence/holistic_admission/box_provenance.md) |
| Why did direction and cap changes fail? | [Direction/search evidence](evidence/direction_search/README.md) |
| What did pixel and temporal tests establish? | [Pixel/temporal evidence](evidence/pixel_temporal/README.md) |
| Which earlier ideas and data remain useful? | [Independent proposals](evidence/independent_proposals/README.md) |
| Why is CourtKeyNet being removed? | [Retirement evidence](evidence/retirement/README.md) |

The completed [W5 directional packet](evidence/holistic_admission/directional_20260921_r5/README.md)
contains all three 27-case arms, receipts and the comparison. Full-frame review
and source/player-gate comparisons are complete. The local-only
`worklog/remote_records_20260921/README.md` indexes the preserved full G0/G1
input and candidate data. Use the published review packet when reading on GitHub.

## Code and saved data

Production and current experiment code stays in its existing directories.
The line-identity matcher is `line_identity/line_run_matcher.py`; its distinct
name prevents an import clash with the direction matcher. The evidence notes
link directly to recorded comparisons, complete candidate populations,
original images and runnable checks. Replay consumers use the canonical
evidence locations after retirement of the temporary compatibility paths.
The one internal `case_records` link in the older line-template result packet
shares its retained measurements without duplicating them.

## Recovery, not another reading path

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

## Keep this usable

- This index and pickup are the short orientation path; stay below 5,000 tokens
- FP_INDEX maps ideas to files; it does not duplicate the results or move files
- Update a result beside its evidence, not in a new session narrative
- Preserve candidate origin, evaluation stage, units and visual-review status
- Keep unfinished assessment prominent; completed generation is not assessment
- Keep original inputs and useful candidate data, with producer and source paths
- Retire superseded prose after its findings and open questions have a home
- Keep recovery copies sealed; do not create another browsable session tree
- Keep W5 manifests, receipts and original recorded paths intact as provenance
- CourtKeyNet is retired as an approach and must be removed before branch end
