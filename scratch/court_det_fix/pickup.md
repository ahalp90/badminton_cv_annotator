# Court-detector pickup

W5 stopped after producing an incomplete first `(3,3)` arm. Several views were
rejected because identical G0/G1 geometry carried different saved gates. The
other two arms did not run. The user has a separate Luna monitoring session.

The runner has been repaired locally and synced to Carmack's existing
`/scratch/ahalperi/court_det_fix/w5_directional_20260921_r1_checkout`.
The five replacements and their rollback copies were hash-verified. These are
uncommitted replacements: use this checkout as synced, without resetting it.
Launch and monitoring are left to the user's Luna session; no job was started.
The runner now keeps conflicting candidates separately, runs without a
preflight stage, calculates sensitivity in the case workers from in-memory
arrays, and streams progress. Use `w5_holistic/run_three_arm_sweep.sh` with a
new run prefix. Do not reuse the archived launcher's synthetic
preflight receipt. Preserve the first arm's existing evidence.

The local [remote packet index](worklog/remote_records_20260921/README.md)
records the actual `w5_directional_20260921_r4` launch. It supersedes the
pre-launch wording in the older campaign resume and W5 handover. Do not start
a second job from those stale instructions.

## When W5 returns

Check the three directional floors `(3,3)`, `(4,3)` and `(5,3)` against the
fixed 27-case population. Preserve the actual launcher and receipts. Compare
changed winners with the saved visual rulings, including useful GX5 and the
weak GX0/Am2 admissions. The saved nine-view post-hoc result does not establish
that pre-cap admission works on the expanded pool.

Each floor is a minimum count of visible projected court markings, ordered
lengthwise then cross-court. It controls entry before the 256-candidate limit.

Use the [W5 evidence account](evidence/holistic_admission/README.md) for the
reasoning and the protected
[operational handover](worklog/webui_further_followups_16092026/w5_handover/README.md#after-remote-results-return)
for the required result reviews. Its launch and tidy-status text is historical:
the original job stopped and this cleanup is complete. Do not repeat its queued
tidy instructions. Update this pickup after interpreting W5.

## Separate unfinished research

- [G0/G1 assessment](evidence/g0_g1/README.md): the 27-case ledger preserves
  available stages, inputs and gaps. Generation validation and W5 consumption
  do not assess comparative quality
- [Person-mask comparison](evidence/holistic_admission/box_provenance.md):
  five corrected `person_observations` cases still need the matcher comparison
- [Independent versus shared scoring](evidence/pixel_temporal/README.md):
  use the same verified automatic candidate union. The W4 rank sum is complete

CourtKeyNet is not a contender. It must be removed before branch completion.
Representative quality, practical runtime and integration are also unfinished;
see [findings and decisions](DETECTOR_DECISIONS.md).

## Protected during this tidy

Keep `worklog/remote_records_20260921/` intact with its index. Keep the
repository-root `local_scratch/campaigns/w5-line-admission/`, its worktree,
repairs and monitoring records unchanged. Do not clean the remote checkout or
preserved Carmack data until W5 finishes and its outputs are pulled and checked.

## Consolidation status

The approved cleanup is installed. Six subject accounts replace the old
session narratives. Useful data and producer sources are retained; complete
G0/G1 populations remain available for assessment. Closed court-related
local-scratch and delegate trees have left the working tree.

Original reports, worklogs and removed outputs are recoverable through the
[index's sealed-backup route](INDEX.md#recovery-not-another-reading-path).
Compatibility links preserve existing replay consumers without duplicating
the evidence. W5's campaign, monitor and complete remote packet are unchanged.
The runner repair synced only five code files to Carmack. Original code is in
`/scratch/ahalperi/court_det_fix/w5_runner_backup_20260921.7yUIBf`.
Run data and the archived launcher remain unchanged. The repair did not start
a job. Retire W5's temporary operational material
only after the job finishes and its outputs are pulled and checked.
