# Three independent court-geometry assessments

Use the three prompts below in separate conversations. They can run alongside Claude's
fixed experiments. Bring the assessments and Claude's results to the next audit;
do not redirect the experiment matrix from an unverified suggestion.

The committed evidence is on `fix/court-det` in repository
`ahalp90/badminton_cv_annotator`, under:
`experiments/annotator/independent_court/recorded/player_guided/projective_patterns/`.
Prompts 1–2 use evidence commit `7299ff3`; prompt 3 adds `d0c9a12`.
The `evaluation/README.md` explains the compact supplement. If using attachments
instead of repository access, provide the files listed inside the relevant prompt.
Galleries need the existing sibling `images/` directory to display their backgrounds.
No new images or videos are required. Prompt 3 adds cached temporal evidence and
the older committed shared-court replay. The actual fragment-merge memberships are pending
Claude's verified replay; the prompts explicitly accommodate that missing evidence.

Copy each fenced block as a complete prompt. Nothing elsewhere in the worklog is needed.

## Prompt 1 — Does our direction objective preserve accurate court geometry?

```text
Assess a badminton court detector's geometry and the logic of its next experiments.
This is a read-only mathematical and empirical critique. Do not edit the repository,
run a new real-frame search, or redesign the pending experiments during execution.
Take the time needed for careful reasoning. A concrete counterexample or a well-supported
finding that no change is needed is more useful than a list of speculative methods.

Repository: ahalp90/badminton_cv_annotator, branch fix/court-det, evidence commit 7299ff3.
Evidence root: experiments/annotator/independent_court/recorded/player_guided/projective_patterns/
Read these files first, relative to that root:
- evaluation/README.md
- evaluation/method_excerpts.md (direction, SVD and error-metric sections)
- followup_status.md (next experiments and completed SVD result)
Use evaluation/direction_records.json.gz for named numerical checks. It includes the
three difficult cases' full saved banks by construction, original selected support masks,
and complete saved SVD pair-fit records. Use gx0_control_measurements.json.gz and
svd_fixed_measurements.json.gz when checking their published summaries.

First state which files and revision you could actually inspect. If compressed records
cannot be opened, state that limitation and continue the source/mathematical assessment.
Do not imply that summaries or an abstract counterexample reproduce a real-frame result.
The evidence guide defines coordinate transforms, array indexing and control provenance.
Actual fragment-to-merged-line memberships and fragment midpoints are not yet published.
The old compatibility lists are not those assignments. Do not infer them from line proximity.

Established position: precise observed directions discarded by automatic selection can
produce visually ideal GX0 courts through the unchanged matcher. Fixed-support SVD yields
partial diagnostic improvement, with little change on GX5. Direction recovery and court
ranking remain unresolved. Small corner errors are not a definition of usability.

Two linked questions:
1. Does our direction-quality objective measure the precision that matters downstream?
Analyse the original foot-anchored angle, proposed projected-fragment-midpoint angle,
capped residual representative score, and unit-2D-normal algebraic SVD objective.
Where can any of them improve while the projected court becomes more sheared or displaced?
Consider homogeneous scale/sign, finite and infinite vanishing points, a vanishing point
near an anchor, short observed spans and perspective conditioning. Distinguish an arbitrary
coordinate effect from genuine ambiguity in the observations. Can a useful label-free
quality measure exist before court scale/position is known, and what assumptions would it
need? Do not assume that a homogeneous angular distance equals court-space accuracy.

2. Does allocating coverage first, then choosing precision within suppression buckets,
preserve the alternatives we need? Could bucket formation already discard the solution?
Could choosing each direction independently miss an important relation between the pair?
Analyse the actual greedy algorithm and the exact B/M/R/MR matrix in the evidence guide.
The representative minimises capped squared angles on a fixed leader mask. Leaders alone
update coverage; candidates outside allocated buckets cannot be rescued by that rule.
Which outcomes would support the proposed explanation, contradict it, or leave competing
explanations unresolved? Include improvements, regressions and no-change outcomes; distinguish
diagnostic control fitting from actual generation and ranking. Do not treat control-selected
SVD groups as known correct membership or a performance ceiling.

Required return:
- Lead with the strongest conclusion and its practical implication for the later audit.
- Give at most three consequential findings, each with a derivation or explicit construction,
  a source symbol/data key, assumptions, contrary evidence and something that would falsify it.
- For any synthetic example, provide enough numbers or code to reproduce it; clearly label
  whether it was executed. Use existing saved fits only for claims about actual cases.
- Give a compact interpretation table for the fixed experiment matrix. Explain what it can
  and cannot distinguish without requesting an unbounded parameter search.
- Identify at most one additional discriminating check, only if a material ambiguity survives
  the specified experiments. State the minimum data needed and whether Claude already returns it.
- Separate mathematical deductions, source observations, saved empirical findings and hypotheses.

All nine views are development data from five videos. No new annotations are planned now.
No universal acceptance threshold, general kNN accuracy gain or end-to-end SVD speedup is
established. Preserve those limits. Do not supply a broad literature review or generic detector
replacement. If external literature is essential to one claim, cite the primary source and
explain exactly which assumption it supplies. Your assessment will be independently audited.
```

## Prompt 2 — Can the available evidence separate a court from a convincing false match?

```text
Assess the identifiability and evaluation of a badminton court detector's ranking evidence.
This is a read-only critique using existing examples, not an instruction to tune a score,
edit code, launch experiments or declare production readiness. Take the time needed to
reason carefully. Start from the concrete counterexamples rather than a generic cue wishlist.

Repository: ahalp90/badminton_cv_annotator, branch fix/court-det, evidence commit 7299ff3.
Evidence root: experiments/annotator/independent_court/recorded/player_guided/projective_patterns/
Read these files first, relative to that root:
- evaluation/README.md (ranking and interpretation sections)
- automatic_axes_results.md
- automatic_axes_visual_judgements.md
- evaluation/method_excerpts.md (paint, visibility and winner-selection sections)
Use evaluation/ranking_records.json.gz and measurements.json.gz for exact candidate evidence.
Use gx0_control_measurements.json.gz for the ideal GX0 controls and their approval.
Inspect automatic_axes_visual_check.html and gx0_control_visual_check.html with their existing
images/ backgrounds when accessible. The written judgements remain evidence if rendering is
unavailable; do not claim to have visually inspected an image you could not open.

Begin by listing which evidence and revision you actually accessed. The guide explains
coordinates, IDs and missing data. Candidate identity includes case and population, not
candidate_id alone. The exported records are selected winners, not a complete negative
population. No new direction-experiment result is needed for this assessment.

Confront all of these facts:
- Amateur-2 frame28019 automatic_all_camera candidate 184:4123 is a grossly wrong paint
  winner despite passing profiles for all eleven court markings.
- ShuttleSet03 scene19 automatic_all_camera candidate 165:6702 is also grossly wrong.
  It gets paint score 1.0 from five available markings; six are unavailable.
- Both false winners fail the original floor gate. But approved GX0 candidate89 and the
  two essentially ideal observed-bank GX0 control winners also fail that gate.
- Paint ranking repairs some Amateur-2 fits but worsens other cases. Neither choosing
  line score everywhere nor choosing paint score everywhere is supported.
- The current visibility calculation requires a clipped span of at least 12 working
  pixels inside the image. Shorter intersecting intervals are also unavailable. This
  does not establish whether a marking is occluded, distinguishable or truly absent.

Main question: what can the existing observations establish about court identity when
some expected paint is unavailable and background structures can imitate the visible pattern?
Separate three states where evidence permits: unavailable observation, observed contradiction,
and observed support. Explain where those states cannot be distinguished with the present data.
Analyse how a candidate might benefit by projecting inconvenient markings outside the image.
Also address the eleven-profile false winner: a visible-count penalty alone cannot explain it.
Do not assume independent markings, calibrated scores or independent frames.

Consider whether finite extent, cross-marking consistency, player position, scene support
or another already available observation could resolve a named ambiguity. For every proposed
cue, say what it distinguishes, the physical assumption it needs, and a plausible valid court
it would wrongly penalise. The floor gate's observed outcomes are available; its full
implementation and all rejected candidates are not in this supplement. Do not invent its cause.

Required return:
- Lead with whether the present evidence supports a discriminating rule or exposes missing
  information. A carefully demonstrated ambiguity is a useful answer.
- Give at most three consequential findings tied to exact candidates, fields or source symbols.
  Separate deductions, recorded facts and untested hypotheses; include counterexamples.
- Specify at most one smallest fixed-pool comparison that could distinguish the strongest
  surviving explanations. Name its candidate population, required features, positive contrary
  examples, recorded outcomes and failure conditions. Do not tune and evaluate on the same
  examples and call the result generalisation; these remain development diagnostics.
- Explain which conclusions could be reached without more annotations, and which require
  visual judgement or evidence not currently available. Do not invent a pixel acceptance cutoff.
- Keep candidate availability, ranking, acceptance and end-to-end detection claims separate.

Use the existing varied amateur and ShuttleSet samples. There are no designated holdouts;
two additional very hard videos are reserved for a final readiness check. Apparent line bowing
has no verified cause. Avoid a broad literature review or a new detector architecture. Cite a
primary source only if it is necessary for a specific external claim. Your return will be
reviewed alongside independent direction experiments; it does not alter their fixed protocol.
```

## Prompt 3 — Is calibration per stable camera view the faster practical route?
*Return markdown, as well as any generated assets suitable for handover.*
```text
Assess a practical change of target: a fully automatic badminton court detector that
calibrates once per stable camera view and reuses that court. No human calibration,
manual candidate choice or CourtKeyNet neural prediction is allowed in the target system.
This is a read-only evidence assessment, not an instruction to alter the current experiment
matrix, rerun a detector, or claim the existing system already meets that target.

Repository: ahalp90/badminton_cv_annotator, branch fix/court-det, evidence commit d0c9a12.
Evidence root: experiments/annotator/independent_court/recorded/player_guided/projective_patterns/
Start with evaluation/temporal_assessment.md, evaluation/README.md and automatic_axes_results.md.
Inspect evaluation/temporal_view.html and its linked existing images when available.
Use evaluation/temporal_records.json.gz for the 13 cached GX/Amateur-2/Amateur-3 frames,
line observations, selected directions, historical candidates and frame/time metadata.
Use evaluation/ranking_records.json.gz for the modern winners available on five of those
frames. Do not treat missing modern results as failures or mix different pipelines' scores.

There is also an older committed shared-court experiment. From the evidence root, its
paths are ../README.md (Replayed short-clip comparison), ../summary.json.gz and ../replay.zip.
Inspect the archive's joint_short.py and saved joint_short/results.json.gz before drawing
conclusions from its headline results. Other archive entries are listed in the temporal guide.
Do not execute the archived code; inspect saved records. It uses one shared court across
three frames per clip in all its scoring arms, so it does not isolate the benefit of temporal
aggregation against independent frame selection. Net scoring and refinement change other factors.

First list the revision and evidence actually inspected. If compressed records, the archive
or images are inaccessible, distinguish the remaining conceptual assessment from empirical
findings. Actual camera stability has NOT been verified for the 13-frame supplement.
They are grouped by video, not certified camera view. Annotations and historical overlays
must not serve as proof of stability. Amateur-2/3 timestamps in seconds are unavailable;
GX seconds are nominal frame index divided by recorded frame rate. Some images are annotated
renderings. Sampled endpoints cannot establish uninterrupted stability between them.

Main question: does this evidence make automatic calibration per stable camera view a
credible nearer-term route, and what is the smallest missing test needed to decide?

Work through these distinctions:
1. Which failures could pooling observations repair because court paint is intermittently
   occluded or missed? Which persist because the directions, line identities or background
   structures are consistently misleading? Cite particular frames/records and include the
   already-good Amateur-3 case as a possible regression, not only the difficult GX cases.
2. Compare generating from pooled line evidence, scoring fixed court candidates across frames,
   and reusing a chosen court. Which is the smallest defensible first comparison given the
   modern failure diagnosis and the older shared-court results? Do not equate selecting the
   highest score anywhere in a video with repeated support for one geometry.
3. What minimal automatic evidence could decide that frames share a camera view, and detect
   when that assumption stops holding? Confront cuts, pan/zoom, near-identical adjacent frames,
   moving players and persistent scene structure. Identify circularity if the estimated court
   is used both to establish stability and to validate the shared court. Separate within-sample
   agreement from any claim about unseen intervals.
4. What would establish a useful CourtKeyNet-free result without silently introducing human
   candidate selection, reference-based view grouping, or a newly tuned acceptance threshold?
   References can evaluate the development experiment; they cannot select its output.

Return a concise recommendation, then at most three consequential findings with exact
case IDs/data keys or source symbols, contrary evidence and falsifiable implications.
Distinguish observed facts, mathematical deductions and hypotheses. If justified, specify
one bounded experiment: cases/frame sampling, automatic grouping assumptions, matched baseline,
what changes, what stays fixed, separate generation/ranking/acceptance outcomes, and stopping
conditions. State which required inputs already exist and the minimum missing evidence.
If current data cannot rank the options, say so and identify the one observation that would
change that judgement. Do not invent results, demand new annotations by default, or propose
an unbounded sweep. Existing samples are development data, with final hard holdouts reserved.
The independent direction experiments continue unchanged; your proposal is for the later audit.
```

#Is calibration per stable camera view the faster practical route?

*Return markdown, as well as any generated assets suitable for handover.*

Assess a practical change of target: a fully automatic badminton court detector that
calibrates once per stable camera view and reuses that court. No human calibration,
manual candidate choice or CourtKeyNet neural prediction is allowed in the target system.
We do not claim the existing system already meets that target.

Repository: ahalp90/badminton_cv_annotator, branch fix/court-det, evidence commit d0c9a12.
Evidence root: experiments/annotator/independent_court/recorded/player_guided/projective_patterns/
Start with evaluation/temporal_assessment.md, evaluation/README.md and automatic_axes_results.md.
Inspect evaluation/temporal_view.html and its linked existing images when available.
Use evaluation/temporal_records.json.gz for the 13 cached GX/Amateur-2/Amateur-3 frames,
line observations, selected directions, historical candidates and frame/time metadata.
Use evaluation/ranking_records.json.gz for the modern winners available on five of those
frames. Do not treat missing modern results as failures or mix different pipelines' scores.

There is also an older committed shared-court experiment. From the evidence root, its
paths are ../README.md (Replayed short-clip comparison), ../summary.json.gz and ../replay.zip.
Inspect the archive's joint_short.py and saved joint_short/results.json.gz before drawing
conclusions from its headline results. Other archive entries are listed in the temporal guide.
Prefer inspecting  saved records to executing repo code. It uses one shared court across
three frames per clip in all its scoring arms, so it does not isolate the benefit of temporal
aggregation against independent frame selection. Net scoring and refinement change other factors.

First list the revision and evidence actually inspected. If compressed records, the archive
or images are inaccessible, distinguish the remaining conceptual assessment from empirical
findings. Actual camera stability has NOT been verified for the 13-frame supplement.
They are grouped by video, not certified camera view. Annotations and historical overlays
must not serve as proof of stability. Amateur-2/3 timestamps in seconds are unavailable;
GX seconds are nominal frame index divided by recorded frame rate. Some images are annotated
renderings. Sampled endpoints cannot establish uninterrupted stability between them.

Main question: does this evidence make automatic calibration per stable camera view a
credible nearer-term route, and what is the smallest missing test needed to decide?

Work through these distinctions:
1. Which failures could pooling observations repair because court paint is intermittently
   occluded or missed? Which persist because the directions, line identities or background
   structures are consistently misleading? Cite particular frames/records and include the
   already-good Amateur-3 case as a possible regression, not only the difficult GX cases.
2. Compare generating from pooled line evidence, scoring fixed court candidates across frames,
   and reusing a chosen court. Which is the smallest defensible first comparison given the
   modern failure diagnosis and the older shared-court results? Do not equate selecting the
   highest score anywhere in a video with repeated support for one geometry.
3. What minimal automatic evidence could decide that frames share a camera view, and detect
   when that assumption stops holding? Confront cuts, pan/zoom, near-identical adjacent frames,
   moving players and persistent scene structure. Identify circularity if the estimated court
   is used both to establish stability and to validate the shared court. Separate within-sample
   agreement from any claim about unseen intervals.
4. What would establish a useful CourtKeyNet-free result without silently introducing human
   candidate selection, reference-based view grouping, or a newly tuned acceptance threshold?
   References can evaluate the development experiment; they cannot select its output.

Return a concise recommendation, then at most three consequential findings with exact
case IDs/data keys or source symbols, contrary evidence and falsifiable implications.
Distinguish observed facts, mathematical deductions and hypotheses. If justified, specify
one bounded experiment: cases/frame sampling, automatic grouping assumptions, matched baseline,
what changes, what stays fixed, separate generation/ranking/acceptance outcomes, and stopping
conditions. State which required inputs already exist and the minimum missing evidence.
If current data cannot rank the options, say so and identify the one observation that would
change that judgement. Do not require new annotations or propose
an unbounded sweep. Existing samples are development data, with final hard holdouts reserved.
The independent direction experiments continue unchanged; your proposal is for the later audit.
