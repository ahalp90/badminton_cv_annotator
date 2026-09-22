# G0/G1: preserved populations, unfinished assessment

The expanded 27-case G0/G1 comparison remains unassessed. It is a separate
research task beside W5. Running W5 on these inputs does not establish which
generator supplies better courts or which scoring input ranks them better.

G0 is the baseline automatic proposal population. G1 keeps the selected
directions fixed and filters the line observations for paint evidence.
That changes both candidate generation and the observations later used for
scoring. Both use cached DeepLSD fragments; G1 is not a new line detector.

## What is available

[The 27-case ledger](cases.csv.gz) records each source path, checksum, cohort
and available stage. It leaves unavailable stages blank and marks the expanded
comparison unassessed. The original data remains in the protected
[remote packet](../../worklog/remote_records_20260921/README.md).

| Saved evidence | Cases | Meaning |
| --- | ---: | --- |
| G0 all-camera records | 27 | Full saved records, including per-pair shortlists |
| Direct G0 generation records located locally | 6 | Two frozen baseline records and four older broadcast records |
| G0 camera-first records located in this check | 0 | Missing from the inspected baseline locations; not a failed experiment |
| G1 generation records | 27 | Original nine, 17 new r3 cases, and legacy scene 0029 |
| G1 camera-first and all-camera records | 26 each | Later rescoring stages; keep them separate |
| G1 input/estimator pairs | 26 | Scene 0029 has no preserved pair |

The r3 validation receipt confirms nine original input cases plus 17 new
cases. `shuttleset_03_scene_0029` is the additional generation-only legacy
record. Its missing inputs and rescores are a known exception, not a failed
r3 case. Preserve its complete result even though it cannot currently be
rerun from the preserved packet.

This is a known reproduction gap, not a W5 blocker: all 256 saved G1
candidates remain available for evaluation. The missing files are
`line_identity/inputs/paint_observations/cases/shuttleset_03_scene_0029.json.gz`
and the corresponding `estimators/shuttleset_03_scene_0029.json.gz`.
Neither appeared in the pre-cleanup local inventory. Do not repeat a broad
local search during session pickup. The user suspects copies may remain on
Carmack; that possibility has not been checked. If exact regeneration is
needed for the G0/G1 assessment, make a bounded check there first. Do not
treat the files as globally lost or silently substitute regenerated inputs
for the original experiment.

The six direct G0 records are not the only useful baseline evidence. The L2
replay reconstructs generation retention from saved all-camera per-pair
shortlists when a direct record is absent. That route is tested in L2's four
cases; it must be checked before extending it to the expanded corpus. A
reconstructed stage must remain distinguishable from an original stage file.

## What the earlier comparisons established

The earlier line-identity experiment compared nine development views. L2
then separated candidate access from scoring on four: GX0, Amateur-2 frames
150 and 28019, and Amateur-3 frame 0. It held the scoring functions fixed and
crossed the two candidate populations with the two observation sets.

With the original observations held fixed, changing G0 to G1 changed both
line and paint winners in all four cases. With G0 held fixed, changing the
observations changed the line winner in three cases and no paint winner.
On the fixed G0+G1 union, the observation change altered all four line
winners and the Amateur-2 frame-28019 paint winner.

These are winner changes, not four demonstrated improvements. For example,
G0 had no eligible Amateur-2 frame-28019 winner in that replay, while G1
supplied one. GX0's one-pair proxy improved under paint filtering while its
nearest full-pool court worsened. Candidate origin and evaluation stage are
therefore essential to interpreting a gain.

Retained evidence:

- [L2 comparison cells](../../next_steps_20260916/L2_scoring/comparison.csv)
  and [exact witnesses](../../next_steps_20260916/L2_scoring/witnesses.json)
- [L2 replay source](../../next_steps_20260916/L2_scoring/run_l2_scoring.py):
  `load_generation_population` records direct versus reconstructed inputs
- [Nine-view matcher records](../../line_identity/runs/line_identity_20260915_222437/matcher/):
  other saved arms remain here, with the person-mask qualifications below.
  The complete [paint-observation population](../../worklog/remote_records_20260921/preserved_data/line_identity/runs/line_identity_20260915_222437/matcher/paint_observations/)
  is in the remote packet; its temporary nine-view links have been retired
- [Direction and cap findings](../direction_search/README.md) for the C2 and
  six-pair L1 probes; neither is a full-corpus G0/G1 assessment

## Questions for the expanded assessment

1. Does G1 add useful courts, lose useful G0 courts, or merely change ordering?
2. On one fixed candidate union, which changes come from scoring observations
   rather than access to different candidates?
3. Which effects arise at generation, camera-first filtering and all-camera
   rescoring? Report available stages separately.
4. Do the 17 new cases preserve the original gains and reveal regressions?
5. How should the generation-only scene be reported without imputing missing
   inputs, rescores or visual judgements?

Keep the complete saved candidate records, input pairs, producing code and
receipts as working evidence. Do not replace them with winners or infer
visual approval from corner error. Structural validation establishes that
records arrived coherently; it does not assess scientific quality.

## Separate unfinished work

The five-case corrected `person_observations` comparison is still owed.
The old person-filtered arms used invalid spatial masks on GX5 and four
ShuttleSet views. Same-image cases and paint-only arms remain useful; see
[box provenance](../holistic_admission/box_provenance.md).

The [temporal comparison](../pixel_temporal/README.md) is another distinct
question. Its completed rank-sum calculation neither assesses this corpus
nor answers independent versus shared scoring on the same candidate union.

Historical source reports are preserved by the tidy backup:
`line_identity/{results,evidence}.md` and
`next_steps_20260916/L2_scoring/result.md`, relative to `scratch/court_det_fix/`.
They are provenance, not additional starting points.
