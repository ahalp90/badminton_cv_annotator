# Court-detector pickup

W5 `w5_directional_20260921_r5` has finished and is local. All three
directional floors `(3,3)`, `(4,3)` and `(5,3)` contain the fixed 27 cases.
The [filed packet](evidence/holistic_admission/directional_20260921_r5/README.md)
records completion checks, the comparison and the remaining result review.
Start there. No new remote sweep is needed.

The supervisor finished the three pilots and galleries, then stopped at an
obsolete comparator requirement for ten workers. The arms used six workers.
The local comparison validates with the existing worker-count repair and
exactly matches the saved comparison. Visual rulings remain pending.

## Next research step

Interpret the three-arm comparison against the saved visual rulings. Inspect
wrong or unclear views and any prior-ruling conflicts at full-frame scale.
Include useful GX5 and the weak GX0/Am2 admissions. Choose one global floor
only if the evidence supports it; otherwise reject the rule.

Each floor counts visible projected court markings, ordered lengthwise then
cross-court, before the 256-candidate limit. The earlier nine-view post-hoc
result does not establish that admission works on the expanded pool.
The [W5 evidence account](evidence/holistic_admission/README.md) preserves
that distinction and the remaining review requirements.

## Separate unfinished research

- [G0/G1 assessment](evidence/g0_g1/README.md): generation validation and
  W5 consumption do not assess comparative quality
- [Person-mask comparison](evidence/holistic_admission/box_provenance.md):
  five corrected `person_observations` cases still need the matcher comparison
- [Independent versus shared scoring](evidence/pixel_temporal/README.md):
  use the same verified automatic candidate union; the W4 rank sum is complete

CourtKeyNet is not a contender. Remove it before branch completion.
Representative quality, practical runtime and integration are also unfinished;
see [findings and decisions](DETECTOR_DECISIONS.md).

## Known gaps: no search needed at pickup

- G1 scene 0029 has saved candidates but no original input/estimator pair in
  the local packet. Copies may remain on Carmack; this is unverified. Check
  there only if exact regeneration is needed for the G0/G1 assessment
- The W2 atlas-builder checksum mismatch is a recorded provenance caveat.
  Do not hunt the exact script without a concrete need

## Filed evidence and recovery

The W5 packet belongs to `evidence/holistic_admission/`. The
[remote packet index](worklog/remote_records_20260921/README.md) still owns
the complete preserved G0/G1 data. Keep those inputs and candidate populations.

Temporary W5 preparation, monitoring records and the campaign worktree are
sealed in `.recovery/w5-preparation-20260922.tar.gz`. The archive preserves
their original repository-relative paths, including the previous pickup.
The [index](INDEX.md#recovery-not-another-reading-path) describes recovery.
Remote files have not been cleaned by this local filing pass.
