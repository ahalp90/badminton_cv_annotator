# Evidence by question

Use the published report for a conclusion, then the named private record when a
measurement or implementation detail is disputed. These are completed experiments.
Historical queues and review suggestions are not current tasks.

| Question | Finding and evidence |
| --- | --- |
| Does SVD of the existing groups restore direction precision? | It gives partial diagnostic gains at 1.1–1.9 ms per case, with little GX5 improvement. No matcher run or smaller-budget claim. [Results](checks/independent/player_guided/20260914/automatic_axes/svd_fixed/results.md), [run record](checks/independent/player_guided/20260914/automatic_axes/svd_fixed/runs.md) |
| Can the matcher build a usable court? | Supplied directions produce usable geometry in eight inspected views; scene16 remains unjudged. [Report](../../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/axis_matching_results.md), [exact visual rulings](../../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/axis_matching_visual_judgements.md), [private run record](checks/independent/player_guided/20260914/axis_matching/runs.md) |
| Does GX0 also lose precise observed directions? | Yes: direction1183 was pruned as redundant and direction122 hit the cap. One unchanged, label-guided matcher control produces winners 3.12/4.05 display pixels from approved candidate89. The user judged both essentially ideal; the line winner has only a mild bottom-right inset to the white-line midpoint. [Results and gallery](checks/independent/player_guided/20260914/automatic_axes/gx0_control/results.md), [run record](checks/independent/player_guided/20260914/automatic_axes/gx0_control/runs.md) |
| Where does automatic GX5 matching fail? | Removing global retention still leaves the nearest camera-eligible court 89.94 working pixels from the inspected control. Precise discarded directions produce a 10.58 display-pixel winner in a label-guided control. [Report](../../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/automatic_axes_results.md), [all three arms and controls](checks/independent/player_guided/20260914/automatic_axes/collected/), [run record](checks/independent/player_guided/20260914/automatic_axes/runs.md) |
| Is paint ranking the answer? | It resolves major Amateur-2 errors but regresses GX0. In the larger automatic pool, a false Amateur-2 court passes all 11 profiles; a false scene19 court scores perfectly from five visible markings. Both fail the original floor gate. [Automatic measurements](../../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/measurements.json.gz) |
| Did the old rectangle search ever contain the approved GX5 court? | Yes. Coverage-ranked direction pruning recovers its exact seed; floor support still rejects it. Four detector acceptances are not four visually usable courts. [Pruning report](../../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/vp_pruning_results.md), [private record](checks/independent/player_guided/20260914/vp_pruning/runs.md) |
| Did better far-service scores fix the geometry? | Three ShuttleSet refits are good, while scene17 remains flawed. GX and Amateur-2 choose wrong structures or identities. [Marking report](../../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/marking_diagnosis_results.md), [visual rulings](../../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/marking_diagnosis_visual_judgements.md) |
| Why stop rectangle-sampling sweeps? | Seventeen covered-length runs improve numerical coverage but accept no GX court. The user rejects the closest GX5 example: 16.30 px corner error and 4.84 px boundary RMS do not establish usability. [Seed report and replay](../../../experiments/annotator/independent_court/recorded/player_guided/evidence_ranked_seeds.md) |
| Did local refinement help? | Close-start GX5 refits are marginally better visually and retain floor rejection. Earlier rejected-start and post-weight trials do not justify another broad sweep. [GX trace and close-start control](../../../experiments/annotator/independent_court/recorded/player_guided/gx_proposal_trace.md) |
| Are Amateur-3's missing far-service fragments a tolerance problem? | Raw 0/62/84 are not court lines, probably sunlight. Raw108 is the slightly inset right boundary. None supports the far short-service marking. [Scientific handover](../../../experiments/annotator/independent_court/recorded/player_guided/court_evidence_handover.md) |
| What about earlier evidence access and detector comparisons? | [Historical scientific handover](../../../experiments/annotator/independent_court/recorded/player_guided/court_evidence_handover.md) and [independent-detector report](../../../experiments/annotator/independent_court/README.md) locate those results; the [archive map](archive/ARCHIVE_MAP.md) locates exact earlier logs |

Working-pixel distances above use 960 × 540; display errors use 1280 × 720.
Keep their comparators distinct. No new success cutoff is inferred.

## Automatic visual judgements

[Exact nine-view feedback](../../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/automatic_axes_visual_judgements.md)
finds usable line winners in five views and paint winners in four. Six views have
a usable winner after choosing between rankings by inspection. GX0, GX5 and
Amateur-2 frame28019 remain poor, including their displayed nearest-control
candidates. GX0 is an explicit shear regression despite low corner errors.
Its separate discarded-direction control now has visual approval for both winners. Apparent curvature
and suspected post/person matches remain unverified explanations.

## Verification and review provenance

The automatic stage passed 22 synthetic tests, scoped lint/types, nine original
selection replays and exact final-score replays for eighteen rescoring cases.
The [automatic run record](checks/independent/player_guided/20260914/automatic_axes/runs.md)
records tolerances, corrections and exit receipts. Its serial-wrapper exit 1 was
an intentional scheduling handoff, not a missing case. Both Opus reviews completed;
the material findings and dispositions are recorded there.

All 27 published automatic comparisons and both bank controls match source
measurements. Publication checks covered links, PNG metadata, geometry and browser
controls. The reports contain no private access details. Original chronological
worklogs and completed review folders are indexed in [archive/ARCHIVE_MAP.md](archive/ARCHIVE_MAP.md).
