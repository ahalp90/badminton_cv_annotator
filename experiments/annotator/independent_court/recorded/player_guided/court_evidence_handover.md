# Court evidence investigation: session handover

The single raster-preserving experiment is complete. Its combined evidence-access
change passes the usable GX frame-5 court and rejects both known wrong courts.
It also rejects one established good Amateur-3 fit. Keep the result as diagnostic
evidence; it does not establish a replacement scorer.

The committed and pushed experiment checkpoint is
`58b2f47a30c5459bd33fbf80b9228832eb0c3da9` on `fix/court-det`. Production and the original detector remain unchanged.

## Start here

Read the mandatory repository instructions and
[raster court matching](raster_court_matching.md). That report contains the one
result table, fixed comparison, decisive matches and limitations. Its
[replay archive](raster_court_matching.zip) supplies all portable inputs/results,
producer scripts, saved checks and replay instructions. No broad history reread
is needed to recover this checkpoint.

The experiment holds 63 courts across 22 frames fixed. The first four arms
separate original-family/projected-direction fragment access from
original-family/all-fragment merged pools. The fifth removes fragment filtering
at the same all-fragment pool. All five keep original raster distances, corners,
thresholds, residual matching and the 32-line merge cap. No merged-line angle
filter is added. The all-fragment pool can lose lines through merging and its cap.

## Verified findings and retained judgements

- Combined access passes GX references at frames 0, 5 and 5111, plus the saved
  frame-5 complete-search court. Neither access change alone passes those courts.
  The saved frame-5 court needs both restored fragment support and the broader
  merged pool to reach three distinct lengthwise lines.
- The user considers that specific frame-5 geometry usable. Left and bottom
  follow the outside of the paint; the mild right-edge inset is acceptable.
  Full right-side paint width is a preference, not a usability requirement.
  See the right-hand court in the existing
  [full-court comparison](gx_trace_overlays/gxBQ_window_00_frame_5__cap.jpg).
- Amateur-3 frame 0, stripe_legacy_0000_original, passes the original raster rule
  but fails directional fragment access under either merged pool. Far short-service
  support falls from 17/24 to 11/24, crosscourt distinct lines from three to two,
  and crosscourt mean from 0.6875 to 0.5486. Fragments 0, 62, 84 and 108 supply the
  six lost samples; their angles differ from that marking by about 9–13 degrees.
- The wrong broadcast junction court in scene 0017 fails directional access with
  lengthwise mean 0.5486, just below 0.55. Unrestricted access passes it. The wrong
  GX court fails either access change. Its Lines rejection loses two matched
  groups at the shared 32-line cap, not through better geometric discrimination.
- Seven older reference courts retain their passes. Five of six saved good stripe
  fits pass combined access. All 24 placements on eight non-court frames reject
  in every arm. These placements do not measure a search false-acceptance rate.
- All 63 baseline controls match the original scorer and previous saved raster
  support/count/score fields exactly. Corners and settings stay fixed. Six
  decisive intervals have exact raw-fragment/union-raster coverage agreement.
- Frame-0 legacy refit remains a useful but visibly imperfect start: the user
  reported top-left undershoot and bottom-right overshoot. The physical refit
  was marginally worse. Neither geometry received the frame-5 usability judgement.

## One recommended next action

In a future session, inspect the rejected Amateur-3 short-service fragments
against the image and fixed projected court. Use case am3_window_00_frame_0,
court stripe_legacy_0000_original, interval far_short_service (index 8), sample
indices 1, 7, 8, 9, 22, 23 and raw fragment IDs 0, 62, 84, 108. The archived
inspect_matches.py and decisive_matches.json.gz provide the exact coordinates.

The open question is whether these fragments are real support with an angle
mismatch or incidental support credited by the original score. Resolve this
before changing tolerances or comparing searches. A passing old score alone
does not establish that its individual matches are correct. Likewise, missing
far-centre support does not establish occlusion; visibility remains unresolved.

This session stops after the single experiment and handover. Seed selection,
refitting, paint-width corrections, threshold tuning, graph search and production
integration were excluded. No automatic continuation into those tasks is queued.

## Validation and on-demand history

38 relevant tests and scoped lint/types passed, exit 0. Input-byte equality,
source provenance, saved arithmetic and archive checks passed. The focused
Fable 5.1 review reproduced two courts and checked all 63 recorded results.
No implementation defect was established. Its shared-cap finding was verified
by inspecting pre-cap groups; no additional score arm was run.
The report passed factual-fidelity review and scored 96.9/100 overall: 89.6/100
mechanically and 4/4 on all four cold-reader dimensions. No material reading
problem remains. No experiment or review process remains active.

Use the [previous finite-distance experiment](fixed_court_matching.md) only for
the earlier distance-method regression. The [GX proposal trace](gx_proposal_trace.md)
records search coverage and previous refits. Its complete-search examples are
label-guided coverage diagnostics, not automatic selection results. The frozen
inputs and original annotations remain unchanged throughout these comparisons.
