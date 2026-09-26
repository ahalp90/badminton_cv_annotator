# Centre-to-edge correction explains part of the court inset

## Final policy — 23 September 2026

**Carry automatic stripe polarity forward as the preferred integration
candidate. Keep the current fitter unchanged until that integration.** The
user reviewed the six-case decision gallery and found automatic polarity
practically identical to the earlier bright-paint rule, with both improving
on the original fit. This is a qualitative ruling on the displayed comparison,
not a claim that every displayed court is correct.

The historical rule assumes brighter paint. A fixed-geometry brightness
inversion reverses both its edge swaps and its centre choice. Automatic polarity
handles that synthetic inversion and preserves original labels when polarity
is unresolved. It is therefore preferable to requiring bright paint or manual
polarity selection. Robustness on real dark-taped courts remains unestablished.
The hue-only colour trial does not resolve that uncertainty.

Nine of twelve cases have label sets different from the earlier saved arms.
All twelve optimise to convergence; eleven pass both the camera and historical
full-court gates. Yellow14 fails both. Am1 remains the wrong net-based court:
its furthest corner moves about 7,245 working pixels from the earlier correction
while still passing both gates. This exposes the acceptance/assignment failure;
it does not by itself disqualify a local fitting correction. Neither rule
resolves the wrong object being identified as a baseline.

Of 382 retained fragments, 109 have unresolved polarity and 111 change labels
from the original. The observed points, weights, intervals and fragment/sample
IDs remain fixed. Automatic polarity describes a local stripe; it does not
establish that the stripe is court paint. The earlier user preference covered the historical comparison. The latest
review covers the six displayed automatic comparisons; the other six automatic
fits have not received a user visual ruling.

The final hue-only trial supplies no additional rejection: it keeps one saved
choice and is inconclusive on eleven. It cannot remove the concern about dark
or neutral paint, and it does not reject Am1's false baseline. The floor trial
also supplies no useful rejection. No manual paint-colour or polarity selection
is permitted in the auto-annotator.

The user confirms that amateur annotations mix paint centres and outside edges.
The convention is unknown per case. Small signed reference drift is therefore
inconclusive. Preserve the initial visual preferences below without converting
them into a per-case accuracy count.

[The six-case gallery](../../colour_consistency/decision_gallery/index.html)
compares original labels, the earlier bright-paint rule and automatic polarity.
It includes the initial four cases, Am1 and an unchanged Am4 control. The user
retains visual judgement. [Saved automatic results](../../colour_consistency/edge_auto_trial.json.gz)
and the [trial worklog](../../colour_consistency/PLAN.md) preserve the checks.
The selected direction is automatic polarity for the next integration, with
uncertain evidence retaining original labels. No production fitting change is
made by these experiments. Preserve the old rule as a comparison, not a
bright-paint requirement in the auto-annotator.

### Complexity and measured cost

The automatic rule adds two small decision functions and reuses the observed
stripe sampler. It samples 24 positions along each retained fragment, tries
five perpendicular shifts, and compares each shift with two flanks. The older
rule samples 14 positions at two side distances. Both use the same optimiser;
neither adds a court-proposal search.

On the 12 saved cases, a local single-thread timing probe measured a median
64 ms for the bright-rule evidence stage. Adding automatic sampling and native
image reading/conversion gives a conservative median total of about 154 ms
(range 71–220 ms). Extra sampling alone costs a median 43 ms; native reading
and conversion cost 48 ms. These are medians of three timed repeats after one
warm-up per case. Context preparation, proposal generation, fitting and scoring
are excluded. This standalone path does not reuse the existing greyscale cache;
it is an implementation-cost estimate, not production latency.

The extra cost is modest for sparse scene sampling. The reason to prefer the
automatic version is its explicit handling of polarity, not an established
real-world robustness gain. Local dark profiles can occur on bright-painted
courts too: SS03-19 fragment 178 was inferred dark and retained its original
label. That does not establish a dark physical paint stripe.

## Historical bright-paint findings

The correction explains a supported label error and was preferred in the
initial four-case gallery. On SS03-34, it moves the upper-left corner about half
a working pixel left. A substantial offset remains, especially upwards. The
evidence does not support changing the projected paint width or adding a fixed
corner offset. The SVD retention experiment proposed afterwards is now complete.

The wider aim is a scene-level court detector that works without CourtKeyNet. This comparison asks whether fragments labelled as stripe centres should instead represent paint edges when their two sides differ strongly in brightness. The comparator corrects contradicted inner and outer edge labels. The candidate also resolves strongly polarised centre fragments to the matching edge. Both use the existing threshold of 10 grey levels sampled at ±1 working pixel. The parent court, points, marking identities, finite intervals and weights are frozen; reference corners do not choose labels. All cases are development comparisons.

## Initial four cases

The first sample contains ShuttleSet video 03 scenes 34, 29 and 19 (SS03 in the table), plus the previously approved amateur GX frame 5. Movements below are relative to the polarity-only comparator, in working pixels. Negative x is left; negative y is up.

| Case | Centre fragments changed | Upper-left movement (x, y) | Paint score, comparator → candidate |
| --- | ---: | ---: | ---: |
| SS03-34 | 2 | −0.506, −0.061 | 0.838148 → 0.845341 |
| SS03-29 | 3 | −0.667, +0.012 | 0.907194 → 0.892942 |
| SS03-19 | 4 | −0.285, −0.584 | 0.806984 → 0.792377 |
| GX frame 5 | 8 | +0.829, +0.013 | 0.428446 → 0.442430 |

The comparators reproduce exactly. Every arm converges and passes the existing camera and full-court gates. Paint score measures image support; it is not a visual acceptance ruling. The user prefers the candidate overall in the [four-case gallery](gallery/index.html), despite the score falls on SS03-29 and SS03-19. On SS03-19, the user reports that the comparator marginally undershoots the back line while the candidate hugs its inner paint edge. That supports a relative improvement. It does not establish alignment with the outside boundary represented by the default outline.

## What moves SS03-34

Fragment 111 was labelled as the left singles stripe centre. Its strong brightness contrast selects the inner edge. It carries 7.75% of the fitting weight. Relabelling fragment 111 alone moves the upper-left corner **(−0.497, −0.070) px**. The leftward direction matches the prediction, though the full fit moves less than the simplified two-line proxy’s −1.154 px.

Fragment 179 carries 0.40% of the weight. Its isolated change moves the corner **(−0.009, +0.009) px**: a negligible downward movement. With both centre fragments corrected, the upper-left corner moves **(−0.506, −0.061) px** from the polarity-only fit. Including the earlier edge-polarity correction, the total movement from the original refit is **(−1.584, −0.589) px**.

The user’s preferred diagnostic corner was (−3, −2) px from the original refit. The corrected upper-left corner remains **1.416 px right and 1.411 px down** from that position. The preferred position is a post-fit diagnostic, not a fitting target.

To explain the residual before resolving the centre labels, start from the **polarity-only fit**. Move its upper-left corner to the preferred position while holding its other three corners fixed. The weighted squared distance to the assigned finite stripes rises from 0.528815 to 0.732596 for leftward movement, 0.770678 for upward movement and 0.909233 for both. These objectives use polarity-only labels throughout.

| Movement | Opposing fragment and assigned marking | Added objective (working px²) |
| --- | --- | ---: |
| Left | 111, left singles centre label | +0.195319 |
| Up | 88, far long-service edge | +0.062942 |
| Up | 291, centre marking edge | +0.061704 |
| Up | 24, far baseline edge | +0.049125 |
| Up | 209, near short-service edge | +0.043493 |

The saved preferred diagnostic is a different comparison: it restores the other three corners to their original-fit positions. Its objective under polarity-only labels is 1.495117, with fragment 236 contributing strongly. Objective values under different label assignments also describe different objectives, so they should not be read as one continuous score.

The brightness profiles suggest why some corrected constraints still oppose the visually preferred position. For horizontal fragments 88, 24 and 209, the strongest falling transitions in the sampled median profiles lie about 1.125, 0.875 and 1.125 px from the raw fragment coordinates. Their projected 40 mm stripe widths are about 0.511, 0.488 and 0.905 px. An edge label still asks the fitter to match the raw coordinate, which can differ from the observed brightness transition.

![Six fragment brightness profiles against projected paint edges](figures/residual_profiles.png)

The purple curves show median image intensity perpendicular to each raw fragment; the black dashed line marks its zero coordinate. Blue edges are median projected model positions along each fragment, whose projected width can vary along its length. These profiles describe image evidence, not verified physical paint boundaries. Blur, raster sampling and effective width cannot be separated uniquely here. No numerical change to the 40 mm width is justified.

## Amateur extension and next step

The same rule was applied to the earliest saved frame from each of eight amateur sources, chosen before outcomes were known: Am1 54, Am2 150, Am3 0, Am4 0, Centre 36, GX 0, Letterboxed 45 and Yellow 14. GX 0 differs from the initial gallery’s GX 5. The [amateur gallery](gallery/amateur/index.html) retains every case.

The user reports one insignificant regression without naming the case. Separately, the user judges Am1 very bad and worse than its also-bad saved court. These rulings do not establish a per-case win count. In Am1, the user identifies the net’s bottom white band as the false far baseline. That is a marking-assignment failure: correcting brightness-side labels can fit the wrong object more closely. The original-label and polarity-only refits have invalid projections; the candidate converges to an oversized court that passes the camera gate. Twelve of its 14 retained fragments have strong polarity. Yellow’s corrected fits fail the camera gate, while Am4 has no strong centre fragments and therefore no change.

Independent Opus 5.5 High work reproduces the fragment 111, fragment 179 and combined fits. Its proxy-weight qualification is already acknowledged. Different 0.3 and 0.1 px rejection thresholds do not affect fragment 111, which exceeds both. The listed scripts, counterfactual, case runs and checks exit 0. The first amateur attempt stopped on invalid Am1; the completed diagnostic retains failed arms explicitly. Seven successful original-label refits match saved attempts within 1.4e−9 native px.

The historical investigation closed with this experimental candidate. The subsequent singular-value decomposition (SVD) retention check is complete. Preserve the full-source comparator; do not mix a direction-pruning change with the fitting correction. The remaining SS03-34 offset is a weighted compromise among constraints whose raw coordinates do not consistently match observed paint transitions. Its blur/width decomposition remains unresolved. Am1’s net-versus-court confusion needs separate selection or rejection work. Supporting results are in the [four-case diagnostics](diagnostics.json.gz), [fragment 111 diagnostic](fragment111.json.gz), [amateur results](amateur_results.json.gz) and [execution record](WORKLOG.md).
