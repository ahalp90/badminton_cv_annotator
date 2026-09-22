# Court-detector pickup

The [wider evaluation](wider_evaluation_20260922.md) is now running on Carmack.
It adds the remaining 20 frozen views and 24 separate broadcast controls with
six workers, and reuses the completed 27-case W5 baseline. Resume from that
worklog for the live run, checks and review state. Code snapshot: `dc7872d`.

Start with the [22 September evaluation](evaluation_results_20260922.md).
W5, expanded G0/G1, the corrected five-case person-mask comparison and
independent/shared temporal scoring are complete. Checks and run details
are in the [evaluation worklog](evaluation_20260922.md).

The practical next design is G1 plus line-template proposals with player
evidence. It preserves 24/27 selections judged usable for development, not
24 clean fits. Yellow14 and Am1 need better proposals; SS21-10 needs better
input/gate handling. Far-end clipping can remain despite a convincing near
end. No tested rule is a safe automatic acceptance criterion.

Use [FP_INDEX](FP_INDEX.md) to find experiments by idea. The
[GitHub evidence packet](evidence/review_20260922/README.md) provides raw
frames, selected overlays and numerical records for an independent review.

## Filed W5 packet

W5 `w5_directional_20260921_r5` has finished and is local. All three
directional floors `(3,3)`, `(4,3)` and `(5,3)` contain the fixed 27 cases.
The [filed packet](evidence/holistic_admission/directional_20260921_r5/README.md)
records completion checks and the comparison. No new remote sweep is needed.

The supervisor finished the three pilots and galleries, then stopped at an
obsolete comparator requirement for ten workers. The arms used six workers.
The local comparison validates with the existing worker-count repair and
exactly matches the saved comparison. The full-frame review is complete.

## Next research step

Run the broader committed sample set next, after a short frozen-runner smoke
check. Compare the current full-source W5 `(4,3)` baseline with G1 plus line
templates using the same scoring and player rule. Keep the current 27 cases
as regression controls. Expand to all 47 frozen cases and add the separate
24-case broadcast rejection/review arm. The [design and wider-test plan](DETECTOR_DECISIONS.md#next-design-and-wider-test)
sets the scope, review categories and older leads worth testing afterwards.

The deployment target is scene-level detection from a small set of sampled
frames, with agreement and proposal reuse across a verified camera view.
The wider fixed-view run is a coverage check, not a single-frame architecture
decision. Test SVD search reduction for speed and preserved coverage; defer
graph-search implementation until the search rules are settled.

Do not postpone that coverage check while adding every older idea. Separate
clean fits, rare tolerable fallbacks and unacceptable fits; inspect the far
end explicitly. GX G1 `143:158` is a skewed fallback. G1 `16:44` loses much of
the far backcourt strip and is tolerable only as an exceptional last resort.
Frequent errors of that size would block deployment.

Use `(4,3)` as the least restrictive tied development setting; `(5,3)` chooses
identical geometry. The existing player-support gate rescues three wall
selections but can also reject a better court. Test rejection and fallback,
not just selected-winner quality.

Each floor counts visible projected court markings, ordered lengthwise then
cross-court, before the 256-candidate limit. The earlier nine-view post-hoc
result does not establish that admission works on the expanded pool.
The [W5 evidence account](evidence/holistic_admission/README.md) preserves
that distinction and the remaining review requirements.

## Comparisons

- [G0/G1 assessment](evidence/g0_g1/README.md): crossed scoring complete on
  all 27 cases for S0 and 26 for S1; proposal access and ranking kept separate
- [Person-mask comparison](evidence/holistic_admission/box_provenance.md):
  all five corrected matcher comparisons are complete
- [Independent versus shared scoring](evidence/pixel_temporal/README.md):
  GX and Am3 complete on their verified automatic candidate unions; pooling
  improves GX paint selection, while shared median line scoring still fails

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

The W5 packet belongs to `evidence/holistic_admission/`. The local-only
`worklog/remote_records_20260921/README.md` still indexes the complete preserved
G0/G1 data. Keep those inputs and candidate populations.

Temporary W5 preparation, monitoring records and the campaign worktree are
sealed in `.recovery/w5-preparation-20260922.tar.gz`. The archive preserves
their original repository-relative paths, including the previous pickup.
The [index](INDEX.md#recovery-not-another-reading-path) describes recovery.
Remote files have not been cleaned by this local filing pass.
