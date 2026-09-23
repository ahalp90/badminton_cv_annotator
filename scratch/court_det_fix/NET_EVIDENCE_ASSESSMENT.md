# Net-evidence pre-evaluation — 23 September 2026

## Current result

**Positive net support helps Am1 once useful proposals exist.** The completed
[recovery trial](colour_consistency/AM1_RECOVERY.md) adds three automatic
vanishing-point seeds, then prefers the first eligible court with strongly
supported tape halves and at least one post. It selects the original rank-2
court: maximum visible-landmark disagreement falls from 45.0 to 7.2 working
pixels. GX5 and Letterboxed45 retain their saved selections. The user accepts
the Am1 improvement despite a remaining inset on the right yellow stripe.
The later [71-case saved-pool scan](net_recovery/WORKLOG.md) changes 13 choices
and visibly loses good courts on Letterboxed78 and frame 52563. Unrestricted
net preference is therefore unsuitable for core promotion. Net evidence remains
useful on Am1; a safe way to combine it with paint evidence is unresolved.
The replay cache is repaired. Projection checks reveal both false fragment
matches and a focal-search-bound limitation. Consistently measured baseline
and seeded pools are required before further generation comparisons. No
production rule is adopted.

The earlier no-go concerned the old candidate pool. Net preference still cannot
recover Am1 from that pool. Naive candidate-free post assembly also remains
unpromising. These limits do not invalidate the new candidate-conditioned
comparison against observed fragments. Missing evidence remains neutral.

See [pickup](pickup.md) for next steps and the linked report for exact settings,
controls, retained data and limitations. Historical findings below retain their
original scope.

## Earlier feasibility probe on saved pools

All generic Hough fragments were compared with the four projected net pieces:
left/right tape and left/right posts. Support is the fraction of visible sample
locations within four working pixels of an aligned fragment (eight-degree
angle tolerance). These are exploratory fixed settings, not validated detector
thresholds. Reference projections are retrospective information-value checks;
references do not select an automatic output.

| View and court | Left/right tape support | Left/right post support |
| --- | --- | --- |
| Am1 selected wrong court | 0.083 / 0.250 | 0.125 / 0.000 |
| Am1 reference, retrospective | 0.958 / 0.917 | 1.000 / 1.000 |
| GX5 selected good court | 0.833 / 0.833 | 0.833 / 0.000 |
| GX5 reference, retrospective | 0.875 / 0.833 | 0.875 / 0.000 |

The stronger GX support differs from the received packet's weak net result
because this probe uses all generic fragments. The packet used only retained
court fragments. The missing right post remains a reason to keep absent
evidence neutral; it does not make GX's visible tape and left post unusable.

A simple candidate-free assembly produced 32 Am1 and 12 GX post/tape triples.
It missed Am1's real pair because a detected right-post fragment extends above
the tape. Requiring lower-band support also removed the real Am1 net; GX's
survivor was a background structure. These results reject those simple rules,
not every inexpensive net method. Cutting a post at its intersection with tape
was not tested. No automatic net-assembly detector was implemented.

The old Am1 pool contains no close correct court. Its 471 candidates passing
this probe’s camera-error filter include none with strong support for both tape and both posts.
That result alone cannot assess a new proposal pool. Predicting net geometry
from a candidate and checking independent image fragments is a legitimate
consistency test; it is not inherently circular. Low support still cannot
serve as a rejection by itself.

The Opus 5-5 high report and five diagnostic scripts are under
`local_scratch/external_delegate/20260923-am1-net-feasibility/`. Its source and
saved numerical outputs were checked, including coordinate conversion, selected
case identity, coverage fractions and counts. The scripts and scoped Ruff exited
0. One Am1 crop was inspected to interpret the overshooting post fragment;
that interpretation remains tentative. No net trial was adopted.

## Historical packet: evidence and judgement

The [frontier packet](webui_net_evidence_frontier_handover/README.md) accurately
distinguishes projected geometry from an independent observation of a net.
Current W5 retains the camera-error diagnostic; it discards the projected net
segments before ranking. The older scorer samples generic line support along
the projection. A wall line or floor stripe can therefore reward it.

The historical replay still establishes a real Yellow improvement: its selected
worst-corner error changes from 221.19 to 26.01 pixels at 1280 × 720. The
Letterboxed change from 10.06 to 17.38 comes from breaking an exact floor-score
tie. That explains the mechanism but does not erase the regression. Fine
alignment remains part of the user's standard. These values were checked in
`replay.zip:joint_short/results.json.gz` under
`experiments/annotator/independent_court/recorded/player_guided/`.

The old formula also lowers a candidate's score when no net component is
visible: zero net support turns the blend into 0.75 times the floor score.
That behaviour conflicts with the intended treatment of missing evidence.
Restoring this scorer would require a new evaluation, not just wiring it in.

The modern W5 coverage diagnostics report no close Am1-54 or Yellow14 proposal:
the best maximum visible-landmark errors are 97.7 and 304.4 working pixels,
respectively. Those statements concern the inspected W5 pools. A new rejection
cue could still usefully cause abstention, but ranking cannot recover geometry
absent from those pools. See the
[coverage assessment](archive/20260922/evaluation_results_20260922.md#what-the-remaining-failures-need).

## What to retain from the independent review

Opus 5-5 high completed one read-only review without time limits. Its useful
findings were the missing-support penalty, the Letterboxed tie, and the need to
separate candidate coverage from ranking. Source inspection and saved replay
records support those findings.

Several recommendations need qualification:

- Do not run its optional absence-based rejection test. Visible projected
  segments with weak line support can reflect occlusion or missed extraction.
  That is not a positive observation contradicting the court.
- An oracle test does not require annotating all 71 views. A few directly
  marked nets could test whether accurate net observations distinguish named
  alternatives. It would establish information value, not a deployable net
  detector. Projecting those annotations from reference courts would make
  the test circular.
- The reported net projections from reference courts are model calculations.
  They do not verify the actual posts or tape. No agent inspected images;
  visual judgement remains with the user.
- Keep any older full-search coverage lead separate from this net decision.
  It does not justify restarting the historical experiment catalogue.

The raw review and launch record are local-only under
`local_scratch/external_delegate/20260923-net-frontier-audit-unlimited/`.
The result reports only `claude-opus-5-5` in model usage. The earlier timed
launch was stopped after the user requested no time limits; its replacement
completed with exit 0.

## Colour trial preparation

The saved SVD runs retain capped parent pools and successful refit children for
Am2, GX0, GX5 and Am1. They preserve native-pixel geometry and selected keys,
but omit the full final ranking and ranker scores. Rejection of a saved choice
can therefore be tested directly. Reselection would first require recovering
the original ordering; a reference-best candidate must never be inserted as a
replacement.

The automatic floor trial is now closed without a useful rejection. Existing person boxes
are not automatically reliable court-floor labels. GX5 also has frame-6 boxes
for a frame-5 image. The current colour diagnostic uses those raw boxes, while
`verifier.mask_boxes_working()` correctly checks the frame relationship. This
limits the existing GX5 occlusion masking and must be addressed before reusing
it in a decision trial. No effect size from that mismatch has been measured.

Sol owns gallery building. Its brief must require literal reuse of
`svd_search/gallery_template.html`, clearly labelled before/after decisions,
explicit abstention, unchanged good controls, identical candidate access in
each comparison, and visible locations of the evidence used. Preserve the
working zoom and overlay controls. The user performs visual inspection.

## Checks and scope

The reviewer checked all 15 packet manifest entries and replayed the supplied
net/mesh diagnostic with matching stdout (exit 0). The main session checked
the material score calculations, current W5 call path and GX5 box provenance
with read-only commands (exit 0). The independent pool inventory also made no
edits. No production code changed and no runtime test suite was required.
Local Markdown link targets and `git diff --check` passed with exit 0.

The review briefly wrote replay output to `/dev/shm` despite its scoped write
boundary, then removed it. No review target was changed.
