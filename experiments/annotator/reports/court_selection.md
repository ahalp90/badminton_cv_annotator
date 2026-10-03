# Court selection failure modes

The line-based court search often found a usable court and several plausible wrong ones. This report records the September work on why the wrong candidate could win.

Two ideas were investigated at checkpoint `b90518c` on `fix/court-det`:

- small errors in the estimated line directions might distort an otherwise good fit;
- the ranking score might favour background markings or partial courts over the court in use.

This is historical work. Later results and the current detector are summarised in [earlier court-detector approaches](../../court_detector/comparisons/earlier_approaches.md) and the [court-detector experiment map](../../court_detector/README.md).

GX, Amateur-2 and Amateur-3 are video-group names used in the saved records.
The search compares court candidates using detected-line support and a paint
score measuring how well their projected markings match bright stripes.

## Direction experiment

The saved [direction records](../independent_court/recorded/player_guided/projective_patterns/evaluation/direction_records.json.gz) cover three difficult cases: GX frame 0, GX frame 5 and Amateur-2 frame 28019. They preserve the merged lines, direction candidates, selected support masks and the fixed-support SVD diagnostics used at the time. SVD (singular value
decomposition) fits a direction to the selected lines by least squares.

The proposed change was small. The existing test measured each line's direction at the point on the merged line closest to the image centre. The alternative measured it at the midpoint of the longest contributing fragment. Everything after that angle measurement stayed fixed.

| Arm | Angle measured at | Candidate retained from each suppression bucket |
| --- | --- | --- |
| B | Existing closest-point location | Existing greedy leader |
| M | Contributing-fragment midpoint | Existing greedy leader |
| R | Existing closest-point location | Lowest-residual candidate in the leader's bucket |
| MR | Contributing-fragment midpoint | Lowest-residual candidate in the leader's bucket |

The existing greedy search kept at most 16 leading direction candidates. Each
leader defined a group, or bucket, of candidates using nearly the same lines:
the intersection of their supporting-line sets divided by their union (IoU)
had to exceed 0.8. R and MR chose the lowest `mean(min(angle, 1.5)**2)` within that fixed bucket, then used that candidate's support mask for the SVD fit. M and R were intended to continue through the normal matcher; MR and the SVD variants were diagnostic arms.

The original fragment-to-merged-line membership and fragment midpoints were not saved. The committed records can replay the old angle test, bank selection and fixed-support SVD calculations, but they cannot reconstruct the midpoint-anchor arm exactly from the saved data alone. The exact geometry conventions and implementation excerpts remain in [method excerpts](../independent_court/recorded/player_guided/projective_patterns/evaluation/method_excerpts.md).

At checkpoint `b90518c`, this comparison had been specified but not completed.
Later midpoint-direction trials recovered useful courts in some views but broke
GX frame 0 and Amateur-3. The [later summary](../../court_detector/comparisons/earlier_approaches.md)
records why the change was not adopted.

## Ranking failures

The saved [ranking records](../independent_court/recorded/player_guided/projective_patterns/evaluation/ranking_records.json.gz) contain the automatic line and paint winners from nine all-camera pools, plus two visually approved GX0 controls. The [visual judgements](../independent_court/recorded/player_guided/projective_patterns/automatic_axes_visual_judgements.md) and [gallery](../independent_court/recorded/player_guided/projective_patterns/automatic_axes_visual_check.html) record the corresponding inspections.

Two wrong paint winners show the problem clearly:

| Case | Saved candidate ID | Result |
| --- | --- | --- |
| Amateur-2 frame 28019 | `184:4123` | Wrong court even though all 11 marking profiles passed |
| ShuttleSet03 scene 19 | `165:6702` | Wrong court with paint score 1.0 from the five visible markings; six markings were unavailable |

Both candidates fail the old floor gate, a pass/fail check on support for the
projected floor markings. The same check also rejects approved GX frame 0
candidate 89 and two near-ideal reference fits supplied for comparison. Counting visible markings has the same problem: good and bad candidates overlap.

The paint score also ignores markings that are unavailable. A projected marking is treated as visible when at least 12 working pixels remain inside the image; that test says nothing about whether the paint is unobscured or whether a background structure looks similar. Line and paint scores are therefore ranking evidence, not probabilities that a court is correct.

The practical result was that candidate generation and candidate selection had become separate problems. The search could produce a good fit, while the available score still preferred a convincing wrong one. No simple floor-gate or visibility-count rule separated the saved good and bad examples.

## Stable-camera reuse

A separate [temporal assessment](court_temporal.md) looked at whether several frames from the same camera view could share one calibration. It used 13 cached frames and an older three-frame replay. That work did not verify camera stability between samples, so it remained a different question from the ranking failures above.

## Scope

These records come from development examples across five videos rather than a held-out evaluation. The visual judgements provide the saved good/bad labels for the inspected candidates. Display errors are measured at 1280 × 720; direction-fit diagnostics use 960 × 540.

Maximum corner distance can also miss internal marking errors. A fit produced from manually supplied control directions shows that the geometry can be matched under those directions; it does not show that the automatic search will recover them.

Later detector work added player evidence, multi-frame composition and court sharing rather than relying on a single ranking scalar. That development is documented under [`experiments/court_detector/`](../../court_detector/README.md).

## Record conventions

The details below matter when replaying the saved diagnostics; they are not needed to understand the result.

- `direction_lines` are homogeneous lines in working-pixel coordinates.
- Row-vector lines transform to normalised coordinates with `lines @ normalised_to_working`.
- Normalised homogeneous points transform back with `points @ normalised_to_working.T`.
- The normalisation centres the image and scales both axes by the image diagonal.
- Direction-bank IDs are assigned before degenerate candidates are removed. Support arrays follow the surviving `candidate_ids`, so a raw candidate ID is not an array index.
- Selected support masks index merged lines, not raw fragments.
- Court coordinates are metres: x runs left-to-right from 0 to 6.10; y runs far-baseline-to-near-baseline from 0 to 13.40. Corner order is far-left, far-right, near-right, near-left.
- Court projection uses `H @ [x, y, 1]`, followed by division by the third coordinate.

The SVD diagnostics use the control named by each record's `control_source`;
GX frame 0 uses approved generated candidate 89. `control_selected_svd` uses that supplied control and therefore measures fitting capacity, not automatic recovery. `control_fit` minimises coordinate residuals with a bounded local solver; its reported maximum corner error is a reporting metric rather than the optimisation target.
